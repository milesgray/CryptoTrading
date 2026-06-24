import sys
import numpy as np
import pytest
from pathlib import Path

# Add project root, src, and services/retrieval to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "services" / "retrieval"))

from services.retrieval.encoder import RetrievalServiceEncoder
from services.retrieval.forecaster import RetrievalForecaster

def test_retrieval_forecaster_math():
    """Verify that the forecasting math (correlation, alignment, and weighted averaging) is correct."""
    # 1. Initialize the encoder (window_size=60, dim=56)
    encoder_service = RetrievalServiceEncoder(window_size=60, n_fft=32, dim=56)
    
    # 2. Add two distinct segments
    # Segment 1: Upward trending past, upward trending future
    prices_1 = np.linspace(100.0, 110.0, 60)
    future_prices_1 = np.linspace(110.0, 120.0, 60)
    order_book_1 = {"bids": [[110.0, 10.0]], "asks": [[111.0, 10.0]]}
    
    encoder_service.add_segment(prices_1, order_book_1, {
        "id": 1,
        "historical_prices": prices_1.tolist(),
        "prices": future_prices_1.tolist(),
        "order_book": order_book_1
    })
    
    # Segment 2: Downward trending past, downward trending future
    prices_2 = np.linspace(100.0, 90.0, 60)
    future_prices_2 = np.linspace(90.0, 80.0, 60)
    order_book_2 = {"bids": [[90.0, 10.0]], "asks": [[91.0, 10.0]]}
    
    encoder_service.add_segment(prices_2, order_book_2, {
        "id": 2,
        "historical_prices": prices_2.tolist(),
        "prices": future_prices_2.tolist(),
        "order_book": order_book_2
    })
    
    # Build the Vector Index
    encoder_service.build_index(n_trees=2)
    
    # 3. Instantiate the forecaster
    forecaster = RetrievalForecaster(encoder_service)
    
    # 4. Query with an upward-sloping window (should be closer to Segment 1)
    query_prices = np.linspace(100.0, 108.0, 60)
    query_order_book = {"bids": [[108.0, 5.0]], "asks": [[109.0, 5.0]]}
    
    result = forecaster.forecast(query_prices, query_order_book, k=2)
    
    # 5. Assertions on output structure and metrics
    assert "retrieved" in result
    assert "prediction" in result
    assert "consensus_path" in result
    assert "expected_return" in result
    assert "bull_ratio" in result
    assert "volatility" in result
    assert "direction" in result
    
    retrieved = result["retrieved"]
    assert len(retrieved) == 2
    
    # Verify that Segment 1 has a higher similarity score than Segment 2
    seg_1_res = next(s for s in retrieved if s["id"] == 1)
    seg_2_res = next(s for s in retrieved if s["id"] == 2)
    
    assert seg_1_res["similarity"] > seg_2_res["similarity"]
    assert 0.0 <= seg_1_res["similarity"] <= 1.0
    assert 0.0 <= seg_2_res["similarity"] <= 1.0
    
    # Verify that direction classification of segment paths is correct
    assert seg_1_res["direction"] == "BULLISH"
    assert seg_2_res["direction"] == "BEARISH"
    
    # Verify that expected return is positive since Segment 1 is more similar
    assert result["expected_return"] > 0.0
    assert result["direction"] == "BULLISH"
    
    # Verify that the consensus path matches the query's end price starting point
    assert len(result["consensus_path"]) == 60
    assert abs(result["prediction"] - result["consensus_path"][-1]) < 1e-5

def test_retrieval_forecaster_fallback():
    """Verify that the forecaster degrades gracefully with an empty index."""
    encoder_service = RetrievalServiceEncoder(window_size=60, n_fft=32, dim=56)
    # Notice we do NOT add segments or build the index here
    
    forecaster = RetrievalForecaster(encoder_service)
    
    query_prices = np.linspace(100.0, 105.0, 60)
    query_order_book = {"bids": [[105.0, 5.0]], "asks": [[106.0, 5.0]]}
    
    # The forecaster should catch the index build error and return the fallback path
    result = forecaster.forecast(query_prices, query_order_book, k=2)
    assert result["retrieved"] == []
    assert len(result["consensus_path"]) == 60
    assert result["expected_return"] > 0
    assert result["direction"] == "BULLISH"
