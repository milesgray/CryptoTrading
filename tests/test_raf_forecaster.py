import sys
import numpy as np
import pytest
import torch
from pathlib import Path

# Add project root, src, and services/retrieval to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "services" / "retrieval"))

from services.retrieval.encoder import RetrievalServiceEncoder
from services.retrieval.forecaster import ChronosRAFForecaster

class MockChronosModel:
    def __init__(self):
        self.device = "cpu"

class MockChronosPipeline:
    def __init__(self, mock_pred_value=0.5):
        self.model = MockChronosModel()
        self.mock_pred_value = mock_pred_value

    def predict(
        self,
        context,
        prediction_length,
        limit_prediction_length=False
    ):
        # Context is a list of 1D tensors
        assert isinstance(context, list)
        assert len(context) == 1
        assert isinstance(context[0], torch.Tensor)
        
        # Return mock samples of shape (1, num_samples, prediction_length)
        num_samples = 20
        # Return predictions filled with self.mock_pred_value (on normalized scale)
        return torch.full((1, num_samples, prediction_length), fill_value=self.mock_pred_value, dtype=torch.float32)

def test_raf_forecaster_normalization_and_offset():
    """Verify that ChronosRAFForecaster correctly performs separate normalization and offset alignment."""
    # Initialize encoder service with dimension matching local dimension (n_fft=32: local_dim = 32 + 17 + 3 + 4 = 56)
    # We set dim=56 and bypass HTTP calls to embed service (which defaults to local fallback when dim != embed_dim + local_dim)
    encoder_service = RetrievalServiceEncoder(window_size=60, n_fft=32, dim=56)
    
    # Add a historical upward segment
    # Context window values (length 60): linear 100 to 110
    prices_1 = np.linspace(100.0, 110.0, 60)
    future_prices_1 = np.linspace(110.0, 120.0, 60)
    order_book_1 = {"bids": [[110.0, 10.0]], "asks": [[111.0, 10.0]]}
    
    encoder_service.add_segment(prices_1, order_book_1, {
        "id": 1,
        "historical_prices": prices_1.tolist(),
        "prices": future_prices_1.tolist(),
        "order_book": order_book_1
    })
    
    encoder_service.build_index(n_trees=2)
    
    # Initialize forecaster with mock chronos pipeline (predicts constant 0.5 in normalized space)
    mock_pipeline = MockChronosPipeline(mock_pred_value=0.5)
    forecaster = ChronosRAFForecaster(encoder_service, mock_pipeline)
    
    # Query prices: 200.0 to 220.0 (std_orig = 5.7735, mean_orig = 210.0, last element = 220.0)
    query_prices = np.linspace(200.0, 220.0, 60)
    query_order_book = {"bids": [[220.0, 5.0]], "asks": [[221.0, 5.0]]}
    
    # Run forecast (k=1)
    result = forecaster.forecast(query_prices, query_order_book, k=1)
    
    assert "retrieved" in result
    assert "prediction" in result
    assert "consensus_path" in result
    
    # Expected stats calculations
    mean_orig = np.mean(query_prices)
    std_orig = np.std(query_prices)
    
    # Denormalized prediction should be: mock_value (0.5) * std_orig + mean_orig
    expected_pred = 0.5 * std_orig + mean_orig
    assert abs(result["prediction"] - expected_pred) < 1e-4
    
    # Consensus path should have the correct length (forecast horizon = 60)
    assert len(result["consensus_path"]) == 60
    assert abs(result["consensus_path"][0] - expected_pred) < 1e-4
    
    # Verify retrieved segment info
    assert len(result["retrieved"]) == 1
    assert result["retrieved"][0]["id"] == 1
