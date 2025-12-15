"""
Robust Pressure Oracle with Adaptive Regimes and Log-Space Math.
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum

import logging

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"

@dataclass
class PressureLabel:
    buy_pressure: float
    sell_pressure: float
    total_pressure: float
    realized_move: float
    move_magnitude: float
    move_direction: int
    market_regime: str
    volatility: float
    
    def validate(self):
        # Pressure can be slightly >1 or <0 due to smoothing/noise, clamp it
        self.buy_pressure = np.clip(self.buy_pressure, 0.0, 1.0)
        self.sell_pressure = np.clip(self.sell_pressure, 0.0, 1.0)
        self.total_pressure = np.clip(self.total_pressure, -1.0, 1.0)

class PressureOracle:
    def __init__(self,
                 lookahead_candles: int = 5,
                 min_significant_move: float = 0.0001,
                 regime_window: int = 100):
        self.lookahead_candles = lookahead_candles
        self.min_significant_move = min_significant_move
        self.regime_window = regime_window
        
    def detect_market_regime(self, 
                            price_history: np.ndarray,
                            current_idx: int) -> MarketRegime:
        """
        Adaptive regime detection using relative metrics.
        """
        if current_idx < self.regime_window:
            return MarketRegime.SIDEWAYS
        
        # Use LOG prices for correct financial math
        recent_prices = price_history[current_idx - self.regime_window:current_idx]
        log_prices = np.log(recent_prices)
        
        # 1. Trend (Linear regression on Log Prices)
        x = np.arange(len(log_prices))
        # Slope represents exponential growth rate
        slope, _ = np.polyfit(x, log_prices, 1)
        
        # 2. Volatility (Std of Log Returns)
        log_returns = np.diff(log_prices)
        current_vol = np.std(log_returns)
        
        # 3. ADAPTIVE Thresholds using history (Z-score concept)
        # In a real system, we'd track long-running mean/std of volatility.
        # Here we use the local window statistics.
        
        # If slope is significant (> 1.5 std deviations of returns)
        trend_strength = slope / (current_vol + 1e-9)
        
        if current_vol > np.percentile(np.abs(log_returns), 90) * 2:
            return MarketRegime.HIGH_VOLATILITY
        elif current_vol < np.percentile(np.abs(log_returns), 20):
            return MarketRegime.LOW_VOLATILITY
        elif trend_strength > 0.1: # Threshold needs tuning, but is now scale-invariant
            return MarketRegime.BULL
        elif trend_strength < -0.1:
            return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS

    def compute_pressure_labels(self,
                               orderbook_snapshot,
                               future_prices: List[float],
                               price_history: List[float],
                               current_idx: int) -> PressureLabel:
        
        # Convert to numpy for faster math
        if not isinstance(price_history, np.ndarray):
            price_history = np.array(price_history)

        current_price = orderbook_snapshot.mid_price
        
        regime = self.detect_market_regime(price_history, current_idx)
        
        # Volatility context for normalization
        recent_slice = price_history[max(0, current_idx-50):current_idx]
        if len(recent_slice) > 1:
            local_vol = np.std(np.diff(np.log(recent_slice)))
        else:
            local_vol = 0.001

        # Lookahead Logic
        target_idx = min(len(future_prices) - 1, self.lookahead_candles)
        future_price = future_prices[target_idx]
        
        # Log Return
        raw_move = np.log(future_price / current_price)
        move_magnitude = abs(raw_move)
        move_dir = int(np.sign(raw_move))
        
        # Normalize move by local volatility (Z-score of the move)
        # This makes the label asset-agnostic
        normalized_strength = move_magnitude / (local_vol * np.sqrt(self.lookahead_candles) + 1e-9)
        
        # Pressure Function: Sigmoid mapping of Z-score
        # Z=1 -> 0.73, Z=2 -> 0.88, Z=3 -> 0.95
        pressure_intensity = 1.0 / (1.0 + np.exp(-(normalized_strength - 1.0) * 2.0))
        
        buy_pressure = 0.1
        sell_pressure = 0.1
        
        if raw_move > 0:
            buy_pressure = pressure_intensity
            # In high vol, sell pressure persists as resistance
            if regime == MarketRegime.HIGH_VOLATILITY:
                sell_pressure = 0.3 + (pressure_intensity * 0.2)
        else:
            sell_pressure = pressure_intensity
            if regime == MarketRegime.HIGH_VOLATILITY:
                buy_pressure = 0.3 + (pressure_intensity * 0.2)

        return PressureLabel(
            buy_pressure=float(buy_pressure),
            sell_pressure=float(sell_pressure),
            total_pressure=float(buy_pressure - sell_pressure),
            realized_move=float(raw_move),
            move_magnitude=float(move_magnitude),
            move_direction=move_dir,
            market_regime=regime.value,
            volatility=float(local_vol)
        )


class TemporalDatasetSplitter:
    """
    Proper temporal train/val/test split to avoid data leakage.
    
    FIXES: Issue #6 - Temporal data leakage
    
    Critical: In time-series, you CANNOT randomly split!
    Must respect temporal ordering: train < val < test
    """
    
    @staticmethod
    def split_temporal(data_length: int,
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15,
                      gap: int = 0) -> Tuple[range, range, range]:
        """
        Split data indices temporally with optional gap.
        
        Args:
            data_length: Total number of samples
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            gap: Number of samples to skip between splits (avoid lookahead contamination)
            
        Returns:
            (train_indices, val_indices, test_indices)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1"
        
        # Calculate split points
        train_end = int(data_length * train_ratio)
        val_start = train_end + gap
        val_end = val_start + int(data_length * val_ratio)
        test_start = val_end + gap
        
        # Create ranges
        train_indices = range(0, train_end)
        val_indices = range(val_start, val_end)
        test_indices = range(test_start, data_length)
        
        logger.info(f"Temporal split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
        logger.info(f"Train: [0, {train_end}), Val: [{val_start}, {val_end}), Test: [{test_start}, {data_length})")
        
        return train_indices, val_indices, test_indices
    
    @staticmethod
    def walk_forward_split(data_length: int,
                          train_size: int,
                          val_size: int,
                          step: int = None) -> List[Tuple[range, range]]:
        """
        Walk-forward validation splits.
        
        Creates multiple (train, val) pairs by sliding a window through time.
        This is even more robust for time-series.
        
        Args:
            data_length: Total samples
            train_size: Size of training window
            val_size: Size of validation window
            step: Step size for sliding (if None, uses val_size)
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        if step is None:
            step = val_size
        
        splits = []
        start = 0
        
        while start + train_size + val_size <= data_length:
            train_indices = range(start, start + train_size)
            val_indices = range(start + train_size, start + train_size + val_size)
            splits.append((train_indices, val_indices))
            start += step
        
        logger.info(f"Created {len(splits)} walk-forward splits")
        return splits


# Example usage showing proper temporal split
if __name__ == "__main__":
    """
    Example: Generate labels with proper temporal split
    """
    
    # Simulate price data
    num_samples = 10000
    prices = [100.0]
    for _ in range(num_samples):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.01)))
    
    oracle = PressureOracle(lookahead_candles=5)
    
    # WRONG: Random split (data leakage!)
    # indices = np.random.permutation(len(prices))
    # train = indices[:7000]  # BAD! Future mixes with past
    
    # CORRECT: Temporal split
    splitter = TemporalDatasetSplitter()
    train_idx, val_idx, test_idx = splitter.split_temporal(
        len(prices) - 10,  # Account for lookahead
        gap=10  # Skip 10 samples between splits
    )
    
    print(f"Train range: {train_idx.start} to {train_idx.stop}")
    print(f"Val range: {val_idx.start} to {val_idx.stop}")
    print(f"Test range: {test_idx.start} to {test_idx.stop}")
    
    # Generate labels (only showing a few)
    from pressure_features_v2 import OrderBookSnapshot
    
    for i in list(train_idx)[:5]:
        # Create dummy orderbook
        snapshot = OrderBookSnapshot(
            timestamp=float(i),
            bids=[(prices[i] * (1 - j * 0.0001), 100.0) for j in range(20)],
            asks=[(prices[i] * (1 + j * 0.0001), 100.0) for j in range(20)],
            mid_price=prices[i]
        )
        
        # Get future prices
        future = prices[i+1:i+11]
        
        # Generate label
        label = oracle.compute_pressure_labels(
            snapshot, future, prices, i
        )
        
        print(f"\nSample {i}:")
        print(f"  Price: ${prices[i]:.2f}")
        print(f"  Buy pressure: {label.buy_pressure:.3f}")
        print(f"  Sell pressure: {label.sell_pressure:.3f}")
        print(f"  Total pressure: {label.total_pressure:.3f}")
        print(f"  Regime: {label.market_regime}")
        print(f"  Realized move: {label.realized_move:.4f}")
