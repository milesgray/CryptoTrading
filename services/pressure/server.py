import os
import torch
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
import datetime as dt
# pyrefly: ignore [missing-import]
from fastapi import HTTPException, BackgroundTasks
from pydantic import BaseModel

# pyrefly: ignore [missing-import]
from cryptotrading.analysis.book import OrderBookFeaturizer, OrderBookSnapshot
# pyrefly: ignore [missing-import]
from cryptotrading.service import ServiceServer

# pyrefly: ignore [missing-import]
from model import get_model
# pyrefly: ignore [missing-import]
from train import TrainingConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pressure_service")

model = None
featurizer = OrderBookFeaturizer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = TrainingConfig()

training_status = {
    "is_training": False,
    "current_step": "idle",
    "progress_percent": 0.0,
    "epoch": 0,
    "total_epochs": 0,
    "train_loss": 0.0,
    "val_loss": 0.0,
    "message": "Not training"
}

class SnapshotInput(BaseModel):
    token: str
    timestamp: float
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    mid_price: float


service = ServiceServer(title="Pressure Service", description="Order book pressure model and features")
app = service.app

@service.on_startup
async def startup_event():
    global model, featurizer
    logger.info("Initializing featurizer...")
    featurizer = OrderBookFeaturizer()
    
    # Load model
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "/app/checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    
    # Try downloading from Artifact Service if not present locally
    try:
        # pyrefly: ignore [missing-import]
        from cryptotrading.client.artifact.service import download_artifact
        if not os.path.exists(checkpoint_path):
            download_artifact("pressure", "best_model.pt", checkpoint_path)
    except Exception as art_err:
        logger.warning(f"Could not check Artifact Service for pressure model: {art_err}")
    
    logger.info(f"Loading model on {device}...")
    model = get_model(config)
    model = model.to(device)
    
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}. Model will use untrained weights.")
        
    model.eval()

@app.post("/features")
async def get_features(snapshot: SnapshotInput):
    """Calculate and return the orderbook features without running the model."""
    try:
        obs = OrderBookSnapshot(
            timestamp=snapshot.timestamp,
            bids=snapshot.bids,
            asks=snapshot.asks,
            mid_price=snapshot.mid_price
        )
        # Assuming token is default/unknown for isolated requests
        features_dict = featurizer.extract_features(obs, token=snapshot.token, validate=True)
        flat_features = featurizer.flatten_features(features_dict)
        return {"features": flat_features.tolist(), "feature_dict": {k: v.tolist() for k,v in features_dict.items()}}
    except Exception as e:
        logger.error(f"Error in get_features: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
async def predict_pressure(snapshot: SnapshotInput):
    """Calculate features and return the model's pressure prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
        
    try:
        obs = OrderBookSnapshot(
            timestamp=snapshot.timestamp,
            bids=snapshot.bids,
            asks=snapshot.asks,
            mid_price=snapshot.mid_price
        )
        features_dict = featurizer.extract_features(obs, token=snapshot.token, validate=True)
        flat_features = featurizer.flatten_features(features_dict)
        
        features_tensor = torch.FloatTensor(flat_features).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(features_tensor)
            
        result = {}
        for k, v in output.items():
            result[k] = v.item() if v.numel() == 1 else v.cpu().numpy().tolist()
            
        return result
    except Exception as e:
        logger.error(f"Error in predict_pressure: {e}")
        raise HTTPException(status_code=400, detail=str(e))

class TrainRequest(BaseModel):
    token: str = "BTC"
    hours_back: float = 24.0
    epochs: Optional[int] = None
    batch_size: Optional[int] = None

async def run_training_task(req: TrainRequest):
    global training_status
    training_status.update({
        "is_training": True,
        "current_step": "initializing",
        "progress_percent": 0.0,
        "epoch": 0,
        "total_epochs": req.epochs or TrainingConfig().num_epochs,
        "train_loss": 0.0,
        "val_loss": 0.0,
        "message": f"Starting background training task for {req.token}..."
    })
    logger.info(f"Starting background training task for {req.token}...")
    try:
        # pyrefly: ignore [missing-import]
        from data_loader import OrderBookDataLoader
        # pyrefly: ignore [missing-import]
        from oracle import PressureOracle
        # pyrefly: ignore [missing-import]
        from train import PressureTrainer, prepare_temporal_dataloaders
        import numpy as np
        
        loader = OrderBookDataLoader()
        await loader.initialize()
        
        end_time = dt.datetime.now(dt.timezone.utc)
        start_time = end_time - dt.timedelta(hours=req.hours_back)
        
        training_status.update({"current_step": "loading_data", "progress_percent": 10.0, "message": "Loading orderbook data..."})
        logger.info("Loading orderbook data...")
        snapshots, quality_metrics = await loader.load_orderbook_data(
            token=req.token,
            start_time=start_time,
            end_time=end_time,
            validate_data=True,
            fill_gaps=True
        )
        
        if not snapshots or len(snapshots) < 100:
            logger.error("Not enough snapshots for training.")
            training_status.update({
                "is_training": False,
                "current_step": "error",
                "message": "Not enough snapshots for training."
            })
            return
            
        logger.info("Fetching price history for oracle...")
        price_adapter = loader.price_adapter
        candles = await price_adapter.get_candlestick_data(
            token=req.token, start_time=start_time, end_time=end_time, granularity=60
        )
        prices = [c['close'] for c in candles] if candles else [s.mid_price for s in snapshots]
        
        training_status.update({"current_step": "extracting_features", "progress_percent": 40.0, "message": "Extracting features..."})
        logger.info("Extracting features...")
        all_features = []
        metadata_list = []
        for s in snapshots:
            feats = featurizer.extract_features(s, req.token, validate=False)
            all_features.append(featurizer.flatten_features(feats))
            metadata_list.append({"timestamp": s.timestamp})
            
        feature_array = np.array(all_features, dtype=np.float32)
        
        training_status.update({"current_step": "generating_labels", "progress_percent": 60.0, "message": "Generating labels with PressureOracle..."})
        logger.info("Generating labels with PressureOracle...")
        oracle = PressureOracle()
        labels = []
        for i, s in enumerate(snapshots):
            future_prices = prices[i:] if i < len(prices) else [s.mid_price]
            price_hist = prices[:i+1] if i < len(prices) else prices
            lbl = oracle.compute_pressure_labels(s, future_prices, price_hist, current_idx=len(price_hist)-1)
            labels.append([lbl.buy_pressure, lbl.sell_pressure, lbl.total_pressure])
            
        labels_array = np.array(labels, dtype=np.float32)
        
        train_cfg = TrainingConfig()
        if req.epochs: train_cfg.num_epochs = req.epochs
        if req.batch_size: train_cfg.batch_size = req.batch_size
        
        dataset_dict = {
            "features": feature_array,
            "labels": labels_array,
            "metadata": metadata_list
        }
        
        logger.info("Preparing temporal dataloaders...")
        train_loader, val_loader, test_loader = prepare_temporal_dataloaders(
            dataset_dict, train_cfg, featurizer=featurizer
        )
        
        training_status.update({"current_step": "training", "progress_percent": 80.0, "message": "Initializing trainer and training..."})
        logger.info("Initializing trainer and training...")
        trainer = PressureTrainer(train_cfg, device=device.type)
        
        def progress_cb(info):
            pct = 80.0 + (info["epoch"] / info["total_epochs"]) * 15.0
            training_status.update({
                "epoch": info["epoch"],
                "total_epochs": info["total_epochs"],
                "train_loss": info["train_loss"],
                "val_loss": info["val_loss"],
                "progress_percent": pct,
                "message": f"Training epoch {info['epoch']}/{info['total_epochs']}"
            })

        trainer.train(train_loader, val_loader, progress_callback=progress_cb)
        
        # Reload model
        global model
        checkpoint_path = os.path.join(train_cfg.checkpoint_dir, "best_model.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            
            training_status.update({
                "is_training": False,
                "current_step": "done",
                "progress_percent": 100.0,
                "message": "Training complete. New model loaded into service."
            })
            logger.info("Training complete. New model loaded into service.")
            
    except Exception as e:
        training_status.update({"is_training": False, "current_step": "error", "message": f"Training failed: {e}"})
        logger.error(f"Training failed: {e}")

@app.get("/train/status")
async def get_train_status():
    return training_status

@app.post("/train")
async def train_model_endpoint(request: TrainRequest, background_tasks: BackgroundTasks):
    if training_status["is_training"]:
        raise HTTPException(status_code=409, detail="Training task is already in progress.")
    background_tasks.add_task(run_training_task, request)
    return {"message": "Training task started in the background", "config": request.dict()}

@app.get("/{token}")
@app.get("/pressure/{token}")
async def get_token_pressure(token: str):
    """Return order book pressure metrics for a given token symbol calculated in real time."""
    try:
        # pyrefly: ignore [missing-import]
        from data_loader import OrderBookDataLoader
        loader = OrderBookDataLoader()
        await loader.initialize()
        
        end_time = dt.datetime.now(dt.timezone.utc)
        start_time = end_time - dt.timedelta(minutes=15)
        
        snapshots, _ = await loader.load_orderbook_data(
            token=token.upper(),
            start_time=start_time,
            end_time=end_time,
            validate_data=False,
            fill_gaps=False
        )
        
        if snapshots:
            obs = snapshots[-1]
        else:
            raise HTTPException(
                status_code=404,
                detail=f"No recent orderbook data available for token '{token.upper()}'."
            )
            
        features_dict = featurizer.extract_features(obs, token=token.upper(), validate=False)
        flat_features = featurizer.flatten_features(features_dict)
        
        buy_pressure = 0.50
        sell_pressure = 0.50
        total_pressure = 0.00
        recommendation = "STANDBY"
        confidence = 0.50

        if model is not None:
            features_tensor = torch.FloatTensor(flat_features).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(features_tensor)
            if isinstance(output, dict):
                buy_pressure = float(output.get("buy_pressure", 0.50))
                sell_pressure = float(output.get("sell_pressure", 0.50))
                total_pressure = float(output.get("total_pressure", 0.00))
            elif isinstance(output, torch.Tensor):
                preds = output.squeeze(0).tolist()
                if len(preds) >= 3:
                    buy_pressure, sell_pressure, total_pressure = preds[:3]

        if total_pressure > 0.15:
            recommendation = "BUY"
            confidence = min(0.50 + abs(total_pressure), 0.95)
        elif total_pressure < -0.15:
            recommendation = "SELL"
            confidence = min(0.50 + abs(total_pressure), 0.95)
        else:
            recommendation = "STANDBY"
            confidence = 0.50

        ofi = float(features_dict.get("ofi", np.array([0.0]))[0]) if "ofi" in features_dict else 0.0
        cvd = float(features_dict.get("depth_imbalance", np.array([0.0]))[0]) if "depth_imbalance" in features_dict else 0.0
        bap = float(features_dict.get("bid_ask_spread", np.array([0.0]))[0]) if "bid_ask_spread" in features_dict else 0.0
        volatility = float(features_dict.get("volatility", np.array([0.001]))[0]) if "volatility" in features_dict else 0.001
        
        market_regime = "trending_up" if total_pressure > 0.2 else ("trending_down" if total_pressure < -0.2 else "sideways")

        return {
            "ofi": round(ofi, 4),
            "cvd": round(cvd, 4),
            "bap": round(bap, 4),
            "buy_pressure": round(buy_pressure, 4),
            "sell_pressure": round(sell_pressure, 4),
            "total_pressure": round(total_pressure, 4),
            "market_regime": market_regime,
            "volatility": round(volatility, 6),
            "recommendation": recommendation,
            "confidence": round(confidence, 4)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating real-time token pressure for {token}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate token pressure for {token}: {str(e)}"
        )


if __name__ == "__main__":
    # pyrefly: ignore [missing-import]
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
