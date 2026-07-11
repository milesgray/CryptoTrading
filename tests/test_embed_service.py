import os
import sys
import numpy as np
import pytest
import torch
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root and services/embed to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "services" / "embed"))

from services.embed.models.encoder import PriceWindowEncoder, TradeSetup, TradeDirection
from services.embed.models.trainer import EncoderTrainer
from services.embed.server import app, state

def test_embed_service_full_trace(tmp_path):
    """
    Perform a full trace of the embed service:
    1. Create mock trade setups
    2. Train an encoder using EncoderTrainer
    3. Save the trained checkpoint
    4. Start FastAPI server TestClient, load checkpoint, and call /embed
    """
    # 1. Create mock trade setups (upward and downward trends)
    setups = []
    window_size = 100
    
    # 10 upward trending setups
    for i in range(10):
        # We need normalized price returns (length window_size - 1)
        price_window = np.random.normal(0.01, 0.005, window_size - 1).astype(np.float32)
        setups.append(TradeSetup(
            price_window=price_window,
            direction=TradeDirection.LONG,
            profit_pct=0.03,
            leverage=5.0,
            hold_duration=10,
            entry_idx=50
        ))
        
    # 10 downward trending setups
    for i in range(10):
        price_window = np.random.normal(-0.01, 0.005, window_size - 1).astype(np.float32)
        setups.append(TradeSetup(
            price_window=price_window,
            direction=TradeDirection.SHORT,
            profit_pct=0.04,
            leverage=5.0,
            hold_duration=10,
            entry_idx=50
        ))

    # 2. Train the encoder
    trainer = EncoderTrainer(
        window_size=window_size,
        embedding_dim=128,
        hidden_dim=64,
        batch_size=4,
        learning_rate=1e-3,
        device="cpu"
    )
    
    # EncoderTrainer.save() creates the path as a directory
    checkpoint_dir = tmp_path / "models_dir"
    
    logger_info = trainer.train(
        train_setups=setups,
        val_setups=setups,
        num_epochs=2,
        save_path=checkpoint_dir,
        patience=2
    )
    
    weights_file = checkpoint_dir / "encoder.pt"
    assert weights_file.exists()
    
    # 3. Configure the server app state with the trained checkpoint
    state.window_size = window_size
    state.embedding_dim = 128
    state.device = "cpu"
    state.encoder = PriceWindowEncoder(
        window_size=window_size - 1,
        embedding_dim=128,
        hidden_dim=64
    ).to(state.device)
    
    state.encoder.load_state_dict(torch.load(weights_file, map_location="cpu"))
    state.encoder.eval()
    
    # 4. Query /embed via TestClient
    client = TestClient(app)
    
    # Payload contains raw prices (length window_size)
    raw_prices = np.linspace(100.0, 110.0, window_size).tolist()
    
    response = client.post("/embed", json={"prices": raw_prices})
    assert response.status_code == 200
    
    data = response.json()
    assert "embedding" in data
    assert "normalized_prices" in data
    assert len(data["embedding"]) == 128
    assert len(data["normalized_prices"]) == window_size - 1


def test_embed_service_batch():
    """
    Test the batch embedding endpoint /embed/batch.
    """
    window_size = 100
    state.window_size = window_size
    state.embedding_dim = 128
    state.device = "cpu"
    state.encoder = PriceWindowEncoder(
        window_size=window_size - 1,
        embedding_dim=128,
        hidden_dim=64
    ).to(state.device)
    state.encoder.eval()

    client = TestClient(app)

    # Payloads containing lists of raw prices
    raw_prices_1 = np.linspace(100.0, 110.0, window_size).tolist()
    raw_prices_2 = np.linspace(110.0, 100.0, window_size).tolist()

    response = client.post(
        "/embed/batch",
        json={"prices_list": [raw_prices_1, raw_prices_2]}
    )
    assert response.status_code == 200

    data = response.json()
    assert "embeddings" in data
    assert len(data["embeddings"]) == 2
    assert len(data["embeddings"][0]) == 128
    assert len(data["embeddings"][1]) == 128
