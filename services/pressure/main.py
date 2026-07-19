import os
import torch
import logging
from typing import List, Dict, Any, Tuple, Optional
import datetime as dt
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from pressure_features import OrderBookFeaturizer, OrderBookSnapshot
from model import get_model
from train import TrainingConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pressure_service")

app = FastAPI(title="Pressure Service", description="Order book pressure model and features")

model = None
featurizer = None
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

@app.on_event("startup")
async def startup_event():
    global model, featurizer
    logger.info("Initializing featurizer...")
    featurizer = OrderBookFeaturizer()
    
    # Load model
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "/app/checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    
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
        from data_loader import OrderBookDataLoader
        from oracle import PressureOracle
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
    background_tasks.add_task(run_training_task, request)
    return {"message": "Training task started in the background", "config": request.dict()}
