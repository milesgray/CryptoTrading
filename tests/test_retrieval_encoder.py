import numpy as np
import pytest
from cryptotrading.analysis.retrieval import RetrievalEncoder

def test_retrieval_encoder_default_window_sizes():
    encoder = RetrievalEncoder(forecast_size=15)
    assert encoder.forecast_size == 15
    assert encoder.historic_window_size == 60  # Default 4x forecast_size

    encoder_custom = RetrievalEncoder(forecast_size=10, historic_window_size=50)
    assert encoder_custom.forecast_size == 10
    assert encoder_custom.historic_window_size == 50

def test_extract_orderbook_features():
    encoder = RetrievalEncoder(forecast_size=15)
    order_book = {
        "bids": [[100.0, 2.5], [99.5, 1.0]],
        "asks": [[100.5, 1.5], [101.0, 3.0]]
    }
    ob_features = encoder.extract_orderbook_features(order_book, current_price=100.25)
    assert isinstance(ob_features, np.ndarray)
    assert len(ob_features) == 50  # OrderBookFeaturizer feature dimension (5*8 + 1 + 9)

def test_extract_price_level_features():
    encoder = RetrievalEncoder(forecast_size=15)
    prices = np.linspace(100.0, 110.0, 60)
    level_features = encoder.extract_price_level_features(prices)
    assert isinstance(level_features, np.ndarray)
    assert len(level_features) == 5

def test_encode_vector_dimensionality():
    n_fft = 32
    encoder = RetrievalEncoder(forecast_size=15, n_fft=n_fft)
    prices = np.linspace(100.0, 105.0, 60)
    order_book = {
        "bids": [[105.0, 1.0]],
        "asks": [[105.5, 1.0]]
    }
    encoded = encoder.encode(prices, order_book)
    # Spectral: n_fft + (n_fft//2 + 1) = 32 + 17 = 49
    # Orderbook: 50
    # Price levels: 5
    # Timeseries: 4
    # Total = 49 + 50 + 5 + 4 = 108
    expected_dim = (n_fft + (n_fft // 2 + 1)) + 50 + 5 + 4
    assert len(encoded) == expected_dim
