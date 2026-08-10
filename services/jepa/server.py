import os
import torch
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import KoopmanJEPAModel
from trading_integration import JEPAStateAugmentation, RegimeAwareLeverageController

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("jepa_service")

from cryptotrading.service import ServiceServer

service = ServiceServer(title="JEPA Service", description="Koopman-JEPA Market Regime & Dynamic Leverage Controller")
app = service.app

model = None
state_augmentation = None
leverage_controller = None

class RegimeRequest(BaseModel):
    prices: List[float]
    timestamps: Optional[List[float]] = None
    token: str = "BTC"

@service.on_startup
async def startup_event():
    global model, state_augmentation, leverage_controller
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "./checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "jepa_best.pth")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Initializing Koopman-JEPA model on {device}...")
    
    model = KoopmanJEPAModel(input_dim=1, hidden_dim=64, latent_dim=32, num_regimes=8)
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            logger.info(f"Loaded JEPA checkpoint from {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Could not load JEPA checkpoint: {e}")
    else:
        logger.warning("No JEPA checkpoint found. Running with initial weights.")
        
    model.eval()
    state_augmentation = JEPAStateAugmentation(jepa_model=model)
    leverage_controller = RegimeAwareLeverageController(jepa_augmentation=state_augmentation)

@app.post("/regime")
async def predict_regime(req: RegimeRequest):
    if not req.prices or len(req.prices) < 5:
        raise HTTPException(status_code=400, detail="Insufficient price data points (min 5 required)")
        
    prices = np.array(req.prices, dtype=np.float32)
    timestamps = np.array(req.timestamps, dtype=np.float32) if req.timestamps else np.arange(len(prices), dtype=np.float32)
    
    try:
        import asyncio
        regime_id, regime_probs = await asyncio.to_thread(state_augmentation.encode_price_history, prices, timestamps)
        optimal_leverage = await asyncio.to_thread(leverage_controller.compute_optimal_leverage, prices, timestamps)
        
        return {
            "token": req.token,
            "regime": int(regime_id),
            "regime_probabilities": regime_probs.tolist() if isinstance(regime_probs, np.ndarray) else regime_probs,
            "recommended_leverage": float(optimal_leverage)
        }
    except Exception as e:
        logger.error(f"Error computing regime: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
