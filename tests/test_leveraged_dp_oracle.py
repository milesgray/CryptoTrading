import sys
import numpy as np
import pytest
from pathlib import Path

# Add project root and src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cryptotrading.trade.oracle.leveraged_dp_oracle import LeveragedDPOracle
from cryptotrading.trade.oracle.util import OracleAction, OracleTradeSegment

def test_oracle_invalid_prices():
    """Verify that empty, short, or invalid shaped price arrays raise errors."""
    oracle = LeveragedDPOracle()
    
    with pytest.raises(ValueError, match="prices must have at least 2 points"):
        oracle.compute_oracle_actions(np.array([100.0]))
        
    with pytest.raises(ValueError, match="prices must be 1-D"):
        oracle.compute_oracle_actions(np.ones((2, 2)))

def test_oracle_flat_prices():
    """Flat prices should result in no trades because transaction fees would eat profits."""
    oracle = LeveragedDPOracle(transaction_cost=0.001)
    prices = np.full(50, 100.0)
    
    actions, leverages = oracle.compute_oracle_actions(prices)
    segments = oracle.extract_trade_segments(prices)
    stats = oracle.get_statistics(prices)
    
    assert len(segments) == 0
    assert np.all(actions == OracleAction.HOLD)
    assert np.all(leverages == 1.0)
    assert stats["num_trades"] == 0
    assert stats["total_return"] == 0.0

def test_oracle_bullish_trend():
    """Test oracle behavior on a steady upward trend without compounding."""
    oracle = LeveragedDPOracle(max_leverage=1.0, transaction_cost=0.01, risk_buffer=0.1)
    # Price rises from 100 to 110
    prices = np.linspace(100.0, 110.0, 11)
    
    segments = oracle.extract_trade_segments(prices)
    assert len(segments) == 1
    seg = segments[0]
    assert seg.direction == 1  # Long
    assert seg.start_idx == 0
    assert seg.end_idx == 10
    assert seg.entry_price == 100.0
    assert seg.exit_price == 110.0
    
    actions, leverages = oracle.compute_oracle_actions(prices, segments)
    assert actions[0] == OracleAction.LONG
    assert actions[10] == OracleAction.CLOSE
    assert np.all(actions[1:10] == OracleAction.HOLD)
    assert leverages[0] == 1.0

def test_oracle_bearish_trend():
    """Test oracle behavior on a steady downward trend without compounding."""
    oracle = LeveragedDPOracle(max_leverage=1.0, transaction_cost=0.01, risk_buffer=0.2)
    # Price drops from 100 to 90
    prices = np.linspace(100.0, 90.0, 11)
    
    segments = oracle.extract_trade_segments(prices)
    assert len(segments) == 1
    seg = segments[0]
    assert seg.direction == -1  # Short
    assert seg.start_idx == 0
    assert seg.end_idx == 10
    assert seg.entry_price == 100.0
    assert seg.exit_price == 90.0
    
    actions, leverages = oracle.compute_oracle_actions(prices, segments)
    assert actions[0] == OracleAction.SHORT
    assert actions[10] == OracleAction.CLOSE
    assert np.all(actions[1:10] == OracleAction.HOLD)
    assert leverages[0] == 1.0

def test_oracle_compounding_behavior():
    """Verify that DP compounds returns when leverage is high and fees are low."""
    oracle = LeveragedDPOracle(max_leverage=10.0, transaction_cost=0.0001, risk_buffer=0.1)
    prices = np.linspace(100.0, 110.0, 11)
    
    segments = oracle.extract_trade_segments(prices)
    # Compounding should result in multiple segments
    assert len(segments) > 1

def test_oracle_opportunity_map():
    """Test density opportunity map generation."""
    oracle = LeveragedDPOracle(max_holding_period=5, max_leverage=2.0)
    prices = np.array([100.0, 105.0, 95.0, 110.0, 105.0])
    
    opp = oracle.compute_opportunity_map(prices)
    assert "long_roe" in opp
    assert "short_roe" in opp
    assert len(opp["long_roe"]) == 5
    
    # At index 0 (100.0), looking ahead max 5 steps:
    # Max price is 110.0 at index 3.
    # Long from 0 -> 3: price change is (110 - 100)/100 = 10%
    # Drawdown along path: min price is 95.0 at index 2. Max adverse long = (100-95)/100 = 5%.
    # With default risk buffer 0.20, lev_long = (1-0.2)/0.05 = 16.0, capped at max_leverage=2.0.
    # Expected roe = 0.10 * 2.0 - fee_drag... (should be positive)
    assert opp["long_roe"][0] > 0.0
    assert opp["long_exit_offset"][0] == 3  # index 3 is best exit (110.0)

def test_significant_opportunities_nms():
    """Test extracting significant opportunities and NMS deduplication."""
    oracle = LeveragedDPOracle(max_holding_period=5, max_leverage=5.0)
    prices = np.array([100.0, 105.0, 102.0, 110.0, 90.0, 85.0])
    
    # All opportunities above min_roe without overlap constraint
    all_opps = oracle.extract_significant_opportunities(prices, min_roe=0.01, max_overlap_fraction=1.0)
    assert len(all_opps) > 0
    
    # With NMS overlap constraint < 1.0, should suppress overlapping trades of lower quality
    nms_opps = oracle.extract_significant_opportunities(prices, min_roe=0.01, max_overlap_fraction=0.5)
    assert len(nms_opps) <= len(all_opps)

def test_caching_behavior():
    """Verify that calling multiple oracle functions on the same array leverages cache."""
    oracle = LeveragedDPOracle()
    prices = np.array([100.0, 102.0, 101.0, 105.0], dtype=np.float64)
    
    # Run once
    oracle.extract_trade_segments(prices)
    first_result = oracle._last_result
    
    # Run again with same identity
    oracle.extract_trade_segments(prices)
    assert oracle._last_result is first_result
    
    # Run with a new copy (should miss cache because of different ID, but values are identical)
    prices_copy = prices.copy()
    oracle.extract_trade_segments(prices_copy)
    assert oracle._last_result is not first_result
    # The actual values should be identical
    for a, b in zip(oracle._last_result, first_result):
        assert np.array_equal(a, b)

def test_invalid_and_nan_prices():
    """Verify behavior when prices contain zero, negative or NaN values."""
    oracle = LeveragedDPOracle()
    # NaN in prices
    prices_nan = np.array([100.0, np.nan, 105.0, 103.0, 110.0], dtype=np.float64)
    
    segments = oracle.extract_trade_segments(prices_nan)
    # It should successfully skip the NaN step and produce reasonable output
    assert len(segments) > 0
    assert all(not np.isnan(s.entry_price) and not np.isnan(s.exit_price) for s in segments)
