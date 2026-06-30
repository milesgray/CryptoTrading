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
from services.retrieval.forecaster import SpecReTFForecaster

def test_specretf_forecaster_stft():
    """Verify that the STFT method correctly partitions and transforms signals."""
    encoder_service = RetrievalServiceEncoder(window_size=60, n_fft=32, dim=56)
    forecaster = SpecReTFForecaster(encoder_service, frame_size=16, hop_size=4)
    
    # 60 points should partition into: (60 - 16) // 4 + 1 = 12 frames
    prices = np.sin(np.linspace(0, 10 * np.pi, 60))
    stft_coefs = forecaster._stft(prices)
    
    assert stft_coefs.shape == (12, 9)  # 12 frames, 16 // 2 + 1 = 9 frequency bins
    assert np.iscomplexobj(stft_coefs)

def test_specretf_similarity_metrics():
    """Verify JSD and phase coherence calculations and bounds."""
    encoder_service = RetrievalServiceEncoder(window_size=60, n_fft=32, dim=56)
    forecaster = SpecReTFForecaster(encoder_service, frame_size=16, hop_size=4)
    
    # Distributions for JSD
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    p3 = np.array([1.0, 0.0, 0.0])
    
    # Identical distributions should have JSD = 0
    assert abs(forecaster._amplitude_jsd(p1, p3)) < 1e-6
    # Orthogonal distributions should have JSD close to 1
    assert abs(forecaster._amplitude_jsd(p1, p2) - 1.0) < 1e-5
    
    # Phase coherence
    phi1 = np.array([0.0, 0.0, 0.0])
    phi2 = np.array([np.pi, np.pi, np.pi])
    phi3 = np.array([0.0, 0.0, 0.0])
    
    # In-phase coherence should be 1
    assert abs(forecaster._phase_coherence(phi1, phi3) - 1.0) < 1e-6
    # Anti-phase coherence should be -1
    assert abs(forecaster._phase_coherence(phi1, phi2) - (-1.0)) < 1e-6

def test_specretf_forecaster_end_to_end_heuristic():
    """Verify end-to-end forecasting using heuristic mode."""
    encoder_service = RetrievalServiceEncoder(window_size=60, n_fft=32, dim=56)
    
    # Add an upward segment
    prices_1 = np.linspace(100.0, 110.0, 60)
    future_prices_1 = np.linspace(110.0, 120.0, 60)
    order_book_1 = {"bids": [[110.0, 10.0]], "asks": [[111.0, 10.0]]}
    
    encoder_service.add_segment(prices_1, order_book_1, {
        "id": 1,
        "historical_prices": prices_1.tolist(),
        "prices": future_prices_1.tolist(),
        "order_book": order_book_1
    })
    
    # Add a downward segment
    prices_2 = np.linspace(100.0, 90.0, 60)
    future_prices_2 = np.linspace(90.0, 80.0, 60)
    order_book_2 = {"bids": [[90.0, 10.0]], "asks": [[91.0, 10.0]]}
    
    encoder_service.add_segment(prices_2, order_book_2, {
        "id": 2,
        "historical_prices": prices_2.tolist(),
        "prices": future_prices_2.tolist(),
        "order_book": order_book_2
    })
    
    encoder_service.build_index(n_trees=2)
    
    # Forecaster
    forecaster = SpecReTFForecaster(encoder_service, frame_size=16, hop_size=4)
    
    # Query with upward sloping window
    query_prices = np.linspace(100.0, 108.0, 60)
    query_order_book = {"bids": [[108.0, 5.0]], "asks": [[109.0, 5.0]]}
    
    result = forecaster.forecast(query_prices, query_order_book, k=2)
    
    print("\nDEBUG RESULT:", result)
    
    assert "retrieved" in result
    assert "prediction" in result
    assert "consensus_path" in result
    assert len(result["retrieved"]) == 2
    
    # Top retrieved should be segment 1 because of similar upward trend
    assert result["retrieved"][0]["id"] == 1
    assert result["retrieved"][0]["similarity"] > result["retrieved"][1]["similarity"]
    assert result["direction"] == "BULLISH"

def test_specretf_forecaster_weighted_mode():
    """Verify forecasting when linear projection weights are supplied."""
    encoder_service = RetrievalServiceEncoder(window_size=60, n_fft=32, dim=56)
    
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
    
    # Define simple weight matrices
    # For a 60-horizon prediction
    H = 60
    w_ret = np.eye(H) * 1.1  # scale retrieval forecast by 1.1
    w_dir = np.zeros((60, H)) # ignore direct query prices in direct forecast pathway
    w_fin = np.zeros((2 * H, H))
    w_fin[:H, :] = np.eye(H) * 0.9  # final forecast is 0.9 * y_hat_retrieval
    
    forecaster = SpecReTFForecaster(
        encoder_service,
        frame_size=16,
        hop_size=4,
        w_retrieval=w_ret,
        w_direct=w_dir,
        w_final=w_fin
    )
    
    query_prices = np.linspace(100.0, 108.0, 60)
    query_order_book = {"bids": [[108.0, 5.0]], "asks": [[109.0, 5.0]]}
    
    result = forecaster.forecast(query_prices, query_order_book, k=1)
    
    # Consensus path should be 1.1 * 0.9 = 0.99 * y_retrieval
    # Expected return should match this path
    assert len(result["consensus_path"]) == H
