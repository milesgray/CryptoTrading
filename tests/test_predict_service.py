import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
from datetime import datetime, timezone

try:
    import services.predict.service as predict_service
    PATCH_TARGET = "services.predict.service.get_price_adapter"
except ModuleNotFoundError:
    import service as predict_service
    PATCH_TARGET = "service.get_price_adapter"

# Create a mock database adapter
mock_adapter = MagicMock()
mock_adapter.initialize = AsyncMock()

class MockCandle:
    def __init__(self, timestamp, close):
        self.timestamp = timestamp
        self.close = close

# Generate 150 dummy candles
mock_candles = [
    MockCandle(datetime.now(timezone.utc), float(100.0 + i + np.sin(i)))
    for i in range(150)
]
mock_adapter.get_candlestick_data = AsyncMock(return_value=mock_candles)

@pytest.fixture(autouse=True)
def patch_price_adapter():
    """Patches get_price_adapter globally during testing to prevent actual database calls."""
    with patch(PATCH_TARGET, return_value=mock_adapter):
        yield mock_adapter

def test_predict_endpoint_success():
    """Verifies that the /predict endpoint successfully generates predictions and trading signals."""
    app = predict_service.app
    
    with TestClient(app) as client:
        response = client.get("/predict?symbol=BTC")
        assert response.status_code == 200
        data = response.json()
        
        assert data["symbol"] == "BTC"
        assert data["prediction"] in ["UP", "DOWN"]
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["recommended_action"] in ["BUY", "SELL", "HOLD"]
        assert "timestamp" in data
        assert "fallback" not in data

def test_predict_endpoint_insufficient_data():
    """Verifies that the /predict endpoint handles insufficient price points gracefully via fallback."""
    app = predict_service.app
    
    # Simulate return of too few candlestick records (less than 20)
    original_return = mock_adapter.get_candlestick_data.return_value
    mock_adapter.get_candlestick_data.return_value = mock_candles[:10]
    
    try:
        with TestClient(app) as client:
            response = client.get("/predict?symbol=ETH")
            assert response.status_code == 200
            data = response.json()
            
            assert data["symbol"] == "ETH"
            assert data["prediction"] in ["UP", "DOWN"]
            assert data["recommended_action"] in ["BUY", "SELL", "HOLD"]
            assert "timestamp" in data
    finally:
        mock_adapter.get_candlestick_data.return_value = original_return
