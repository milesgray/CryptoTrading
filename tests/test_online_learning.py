import os
import sys
import pytest
import numpy as np
import httpx
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Helper to clean sys.path and sys.modules to prevent cross-service import shadowing
def clean_imports(service_name):
    # Remove any paths containing services to keep path pristine
    sys.path = [p for p in sys.path if "services" not in p]
    
    # Add target service path and root
    sys.path.insert(0, str(PROJECT_ROOT / "services" / service_name))
    sys.path.insert(0, str(PROJECT_ROOT))
    
    # Remove colliding modules from cache to force reload in new context
    for mod in ["models", "database", "data", "config", "websocket"]:
        if mod in sys.modules:
            del sys.modules[mod]

def test_embed_service_add_setup(tmp_path):
    """Test that the embed service correctly generates an embedding for a price window and inserts it into the VectorStore."""
    clean_imports("embed")
    from services.embed.server import app as embed_app, state as embed_state
    from services.embed.models.encoder import PriceWindowEncoder
    from database.numpy_store import NumpyVectorStore

    embed_state.window_size = 100
    embed_state.embedding_dim = 128
    embed_state.device = "cpu"
    embed_state.encoder = PriceWindowEncoder(
        window_size=99,
        embedding_dim=128,
        hidden_dim=64
    ).to(embed_state.device)
    embed_state.encoder.eval()

    store_path = tmp_path / "test_store"
    embed_state.store = NumpyVectorStore(store_path=str(store_path), embedding_dim=128)

    client = TestClient(embed_app)
    
    raw_prices = [100.0 + i * 0.05 for i in range(100)]
    payload = {
        "symbol": "BTC",
        "timeframe": "1m",
        "prices": raw_prices,
        "direction": 1,
        "profit_pct": 5.0,
        "leverage": 1.0,
        "hold_duration": 10,
        "entry_timestamp": 123456789.0,
        "entry_price": 100.0,
        "exit_price": 105.0
    }

    response = client.post("/setup/add", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "id" in data
    assert data["id"] == 0

def test_retrieval_rebuild():
    """Test that the retrieval service correctly invalidates and clears the cached forecasters index for a specific symbol."""
    clean_imports("retrieval")
    from services.retrieval.main import app as retrieval_app, forecasters_cache

    # Seed forecasters cache with mock data
    forecasters_cache[("BTC", 60, 60)] = "mock_forecaster_btc"
    forecasters_cache[("ETH", 60, 60)] = "mock_forecaster_eth"

    client = TestClient(retrieval_app)

    response = client.post("/rebuild?symbol=BTC")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["cleared_configs"] == 1

    # Verify BTC was removed but ETH remains
    assert ("BTC", 60, 60) not in forecasters_cache
    assert ("ETH", 60, 60) in forecasters_cache

def test_serve_add_setup_proxy():
    """Test that the serve router proxies the setup archiving request to the embed service, computes returns/directions, and triggers retrieval cache rebuilds."""
    clean_imports("serve")
    from services.serve.routers.retrieval import router as serve_retrieval_router
    from fastapi import FastAPI

    serve_app = FastAPI()
    serve_app.include_router(serve_retrieval_router)

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_request = httpx.Request("POST", "http://localhost")
        mock_embed_resp = httpx.Response(200, json={"success": True, "id": 42}, request=mock_request)
        mock_rebuild_resp = httpx.Response(200, json={"success": True, "cleared_configs": 1}, request=mock_request)
        mock_post.side_effect = [mock_embed_resp, mock_rebuild_resp]

        client = TestClient(serve_app)

        payload = {
            "symbol": "BTC",
            "timeframe": "1m",
            "prices": [100.0, 101.0, 102.0],
            "actual_future_prices": [103.0, 104.0, 105.0],
            "leverage": 1.0
        }

        response = client.post("/retrieval/setup/add", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["id"] == 42
        assert data["profit_pct"] > 0
        assert data["direction"] == 1

        assert mock_post.call_count == 2
        first_call_args = mock_post.call_args_list[0]
        assert "setup/add" in first_call_args[0][0]
        second_call_args = mock_post.call_args_list[1]
        assert "rebuild" in second_call_args[0][0]
