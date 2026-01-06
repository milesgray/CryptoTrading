"""
Stratified Temporal Sampling for Micro-Regime Discovery

Use case: High-leverage scalping at 15-minute granularity
Goal: Discover micro-regimes (volatility spikes, dead zones, breakouts)
       that exist across ALL macro market conditions

Key insight: A "volatility spike" looks similar whether it happens during
a bull market, bear market, or sideways period. We want to learn these
micro-patterns independent of macro trend.

Solution: Sample windows from ALL time periods proportionally in each split.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from typing import Dict, Tuple
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlobalPriceNormalizer:
    """
    Normalizes price data globally across the entire dataset.
    
    This normalizer computes mean and std from the entire training dataset
    and applies the same transformation to all data splits.
    Used to prevent OOD collapse by ensuring consistent scaling across all splits.
    
    The same normalization parameters (mean, std) are used for all splits
    to ensure that the model sees consistent scaling regardless of which
    time period it's sampling from. This prevents the model from collapsing
    to zero vectors when encountering out-of-distribution data during training
    or inference by maintaining consistent feature scales across all data splits.
    This is essential for learning micro-regimes that should be invariant to
    macro market conditions.
    """
    def __init__(self):
        self.price_mean = None
        self.price_std = None
        self.fitted = False
    
    def fit(self, all_train_prices: np.ndarray):
        self.price_mean = float(np.mean(all_train_prices))
        self.price_std = float(np.std(all_train_prices))
        self.fitted = True
        logger.info(f"Fitted normalizer: mean={self.price_mean:.2f}, std={self.price_std:.2f}")
    
    def transform(self, prices: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
        eps = 1e-8
        return (prices - self.price_mean) / (self.price_std + eps)



class CryptoPriceDataset(Dataset):   
    """
    Dataset for cryptocurrency price sequences with global normalization.
    
    This dataset provides sliding windows of price data for training the JEPA model.
    All splits (train/val/test) use the same global normalization parameters
    computed from the entire training set to ensure consistent scaling.
    """
    def __init__(
        self,
        prices: np.ndarray,
        timestamps: np.ndarray,
        normalizer: GlobalPriceNormalizer,
        window: int = 768,       
        stride: int = 50,
        mode: str = 'train',
        augment: bool = False
    ):
        """
        Args:
            prices: Raw price data
            timestamps: Corresponding timestamps
            normalizer: FITTED normalizer (must be fitted before creating dataset)
            context_window: Context length
            target_window: Target length
            prediction_horizon: Steps ahead to predict
            stride: Step size between samples
            mode: 'train', 'val', or 'test' (for logging only)
        """
        self.prices = prices.astype(np.float32)
        self.timestamps = timestamps.astype(np.float32)
        self.normalizer = normalizer
        self.context_window = window
        self.target_window = window
        self.prediction_horizon = window // 4
        self.mode = mode
        self.augment = augment
        
        # Verify normalizer is fitted
        if not normalizer.fitted:
            raise ValueError("Normalizer must be fitted before creating dataset!")
        
        # Calculate valid indices
        total_needed = self.context_window + self.prediction_horizon + self.target_window
        self.valid_indices = np.arange(0, len(prices) - total_needed, stride)
        
        logger.info(f"Created {mode} dataset: {len(self.valid_indices)} samples")
        
        # Pre-compute features (faster training)
        logger.info(f"Pre-computing {mode} features...")
        self._precompute_features()
    
    def _precompute_features(self):
        """Pre-compute features WITHOUT augmentation"""
        self.features_cache = []
        
        for idx in tqdm(range(len(self.valid_indices)), desc=f"Pre-computing {self.mode}"):
            start = self.valid_indices[idx]
            
            ctx_start = start
            ctx_end = start + self.context_window
            
            tgt_start = ctx_end + self.prediction_horizon
            tgt_end = tgt_start + self.target_window
            
            context_prices = self.prices[ctx_start:ctx_end]
            target_prices = self.prices[tgt_start:tgt_end]
            
            # Store RAW prices (augmentation applied in __getitem__)
            self.features_cache.append({
                'context_prices': context_prices,
                'target_prices': target_prices
            })
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cached = self.features_cache[idx]
        
        # Apply augmentation ONLY during training
        augment = self.augment if self.mode == 'train' else False
        
        # Create features with optional augmentation
        x_context = create_multimodal_features(
            cached['context_prices'],
            self.normalizer,
            self.context_window,
            augment=augment
        )
        
        x_target = create_multimodal_features(
            cached['target_prices'],
            self.normalizer,
            self.target_window,
            augment=augment
        )
        
        return {
            'x_context': x_context,
            'x_target': x_target
        }



class StratifiedTemporalDataset(Dataset):
    """
    Dataset with stratified temporal sampling.
    
    Ensures train/val/test all contain samples from ALL time periods,
    preventing the model from learning macro trends instead of micro regimes.
    """
    
    def __init__(
        self,
        prices: np.ndarray,
        timestamps: np.ndarray,
        normalizer: GlobalPriceNormalizer,
        sample_indices: np.ndarray,
        context_window: int = 768,
        target_window: int = 768,
        prediction_horizon: int = 256,
        mode: str = 'train',
        augment: bool = False
    ):
        self.prices = prices.astype(np.float32)
        self.timestamps = timestamps.astype(np.float32)
        self.normalizer = normalizer
        self.context_window = context_window
        self.target_window = target_window
        self.prediction_horizon = prediction_horizon
        self.mode = mode
        self.augment = augment
        
        if not normalizer.fitted:
            raise ValueError("Normalizer must be fitted!")
        
        self.sample_indices = sample_indices
        
        logger.info(
            f"Created {mode} dataset: {len(self.sample_indices)} samples "
            f"(stratified temporal, augment={augment})"
        )
        
        self._precompute_features()
    
    def _precompute_features(self):
        """Pre-compute features"""
        self.features_cache = []
        
        for idx in tqdm(self.sample_indices, desc=f"Pre-computing {self.mode}"):
            ctx_start = idx
            ctx_end = idx + self.context_window
            
            tgt_start = ctx_end + self.prediction_horizon
            tgt_end = tgt_start + self.target_window
            
            context_prices = self.prices[ctx_start:ctx_end]
            target_prices = self.prices[tgt_start:tgt_end]
            
            self.features_cache.append({
                'context_prices': context_prices,
                'target_prices': target_prices,
                'timestamp': self.timestamps[idx]
            })
    
    def __len__(self) -> int:
        return len(self.sample_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cached = self.features_cache[idx]
        
        
        # Apply augmentation ONLY during training
        augment = self.augment if self.mode == 'train' else False
        
        
        x_context = create_multimodal_features(
            cached['context_prices'],
            self.normalizer,
            self.context_window,
            augment=augment
        )
        
        x_target = create_multimodal_features(
            cached['target_prices'],
            self.normalizer,
            self.target_window,
            augment=augment
        )
        
        return {
            'x_context': x_context,
            'x_target': x_target,
            'timestamp': cached['timestamp']
        }


def create_stratified_temporal_split(
    timestamps: np.ndarray,
    valid_indices: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    n_strata: int = 20,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create stratified temporal split.
    
    Divides time into N equal strata, then samples train/val/test
    proportionally from each stratum.
    
    Args:
        timestamps: Array of timestamps for all data points
        valid_indices: Valid sample start indices
        train_ratio: Fraction for training (default: 0.7)
        val_ratio: Fraction for validation (default: 0.2)
        n_strata: Number of time strata (default: 20)
        random_seed: Random seed for reproducibility
        
    Returns:
        (train_indices, val_indices, test_indices)
    """
    np.random.seed(random_seed)
    
    # Get timestamps for valid indices
    sample_timestamps = timestamps[valid_indices]
    
    # Define time strata (equal-sized time bins)
    min_time = sample_timestamps.min()
    max_time = sample_timestamps.max()
    time_range = max_time - min_time
    
    stratum_edges = np.linspace(min_time, max_time, n_strata + 1)
    
    logger.info("\nStratified temporal splitting:")
    logger.info(f"  Time range: {min_time:.0f} to {max_time:.0f}")
    logger.info(f"  Number of strata: {n_strata}")
    logger.info(f"  Stratum size: ~{time_range / n_strata:.0f} time units")
    
    # Assign each sample to a stratum
    stratum_assignments = np.digitize(sample_timestamps, stratum_edges) - 1
    stratum_assignments = np.clip(stratum_assignments, 0, n_strata - 1)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Sample from each stratum
    for stratum_idx in range(n_strata):
        stratum_mask = stratum_assignments == stratum_idx
        stratum_samples = valid_indices[stratum_mask]
        
        if len(stratum_samples) == 0:
            continue
        
        # Shuffle samples within stratum
        shuffled = np.random.permutation(stratum_samples)
        
        # Split within stratum
        n_samples = len(shuffled)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        train_indices.extend(shuffled[:train_end])
        val_indices.extend(shuffled[train_end:val_end])
        test_indices.extend(shuffled[val_end:])
        
        logger.info(
            f"  Stratum {stratum_idx+1:2d}: {len(stratum_samples):6d} samples -> "
            f"train={len(shuffled[:train_end])}, "
            f"val={len(shuffled[train_end:val_end])}, "
            f"test={len(shuffled[val_end:])}"
        )
    
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    logger.info("\nTotal samples:")
    logger.info(f"  Train: {len(train_indices)}")
    logger.info(f"  Val: {len(val_indices)}")
    logger.info(f"  Test: {len(test_indices)}")
    
    # Verify temporal distribution
    logger.info("\nTemporal coverage verification:")
    for name, indices in [('Train', train_indices), ('Val', val_indices), ('Test', test_indices)]:
        sample_times = timestamps[indices]
        logger.info(
            f"  {name}: covers {sample_times.min():.0f} to {sample_times.max():.0f} "
            f"(span: {sample_times.max() - sample_times.min():.0f})"
        )
    
    return train_indices, val_indices, test_indices

def pull_datasets_stratified(
    price_client,
    token: str,
    context_window: int = 768,
    target_window: int = 768,
    prediction_horizon: int = 256,
    stride: int = 50,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    n_strata: int = 20,
    augment_train: bool = True,
    random_seed: int = 42
):
    logger.info(f"Loading historical prices for {token}...")
    
    historical_prices = price_client.load_historical_prices(
        token,
        days=30,
        page_size=5000
    )
    
    if historical_prices is None or len(historical_prices) == 0:
        raise ValueError(f"No historical data for {token}")

    return create_datasets_stratified(
        historical_prices[:, 0],
        historical_prices[:, 1],
        context_window,
        target_window,
        prediction_horizon,
        stride,
        train_ratio,
        val_ratio,
        n_strata,
        augment_train,
        random_seed
    )

def create_datasets_stratified(
    prices, 
    timestamps,
    context_window: int = 768,
    target_window: int = 768,
    prediction_horizon: int = 256,
    stride: int = 50,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    n_strata: int = 20,
    augment_train: bool = False,
    random_seed: int = 42
) -> Tuple:
    """
    Create datasets with stratified temporal sampling.
    
    This ensures train/val/test all contain samples from ALL time periods,
    preventing the model from learning macro trends.
    
    For 15-minute regime discovery:
    - context_window = 768 points (~1.2 hours, ~5 fifteen-min periods)
    - prediction_horizon = 156 points (~15 minutes ahead)
    - n_strata = 20-50 (more strata = finer temporal mixing)
    """    
    logger.info(f"Loaded {len(prices)} price points")
    logger.info(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    
    # Estimate data granularity
    if len(timestamps) > 1:
        time_diffs = np.diff(timestamps[:1000])  # Check first 1000 points
        median_interval = np.median(time_diffs)
        points_per_hour = 3600 / median_interval if median_interval > 0 else 0
        points_per_15min = points_per_hour / 4
        
        logger.info("\nData granularity:")
        logger.info(f"  Median interval: {median_interval:.2f} seconds")
        logger.info(f"  Points per hour: ~{points_per_hour:.0f}")
        logger.info(f"  Points per 15min: ~{points_per_15min:.0f}")
        logger.info(f"  Context window ({context_window} points) = ~{context_window/points_per_15min:.1f} fifteen-min periods")
    
    # Calculate valid indices
    total_needed = context_window + prediction_horizon + target_window
    valid_indices = np.arange(0, len(prices) - total_needed, stride)
    
    logger.info(f"\nTotal valid samples: {len(valid_indices)}")
    
    # Stratified temporal split
    train_indices, val_indices, test_indices = create_stratified_temporal_split(
        timestamps, valid_indices, train_ratio, val_ratio, n_strata, random_seed
    )
    
    # Fit normalizer on training price range
    logger.info("\nFitting normalizer on training price range...")
    train_min_idx = train_indices.min()
    train_max_idx = train_indices.max() + total_needed
    train_price_range = prices[train_min_idx:train_max_idx]
    
    normalizer = GlobalPriceNormalizer()
    normalizer.fit(train_price_range)
    
    logger.info(f"Normalizer: mean={normalizer.price_mean:.2f}, std={normalizer.price_std:.2f}")
    
    # Create datasets
    train_dataset = StratifiedTemporalDataset(
        prices, timestamps, normalizer, train_indices,
        context_window, target_window, prediction_horizon,
        mode='train', augment=augment_train
    )
    
    val_dataset = StratifiedTemporalDataset(
        prices, timestamps, normalizer, val_indices,
        context_window, target_window, prediction_horizon,
        mode='val', augment=False
    )
    
    test_dataset = StratifiedTemporalDataset(
        prices, timestamps, normalizer, test_indices,
        context_window, target_window, prediction_horizon,
        mode='test', augment=False
    )
    
    return train_dataset, val_dataset, test_dataset, normalizer



def augment_price_window(
    prices: np.ndarray,
    augment: bool = True
) -> np.ndarray:
    """
    Data augmentation to make training more robust.
    
    Augmentations:
    1. Random scaling (simulate different price levels)
    2. Random noise (simulate measurement error)
    3. Random shift (simulate different baselines)
    
    This makes the model less likely to overfit to specific
    training distribution and collapse on OOD inputs.
    """
    if not augment:
        return prices
    
    # Random scaling (±5%)
    scale = np.random.uniform(0.95, 1.05)
    
    # Random noise (0.1% of price std)
    noise_std = np.std(prices) * 0.001
    noise = np.random.randn(len(prices)) * noise_std
    
    # Random baseline shift (±1% of mean)
    shift = np.random.randn() * np.mean(prices) * 0.01
    
    augmented = prices * scale + noise + shift
    
    return augmented


def create_multimodal_features(
    prices: np.ndarray,
    normalizer: GlobalPriceNormalizer,
    window_size: int = 768,
    augment: bool = False
) -> torch.Tensor:
    """
    Create multimodal features from price data with optional augmentation.
    
    This function prepares price data for multimodal modeling by:
    1. Applying data augmentation during training
    2. Normalizing using global statistics
    3. Computing returns
    4. Creating a comprehensive feature tensor
    
    Args:
        prices: Array of price values
        normalizer: Global price normalizer instance
        window_size: Target window size for features
        augment: Whether to apply data augmentation (True for training, False for validation/test)
        
    Returns:
        torch.Tensor: Multimodal feature tensor
    """
    eps = 1e-8
    
    prices = augment_price_window(prices, augment=augment)
    
    # Ensure window size
    if len(prices) > window_size:
        prices = prices[-window_size:]
    elif len(prices) < window_size:
        pad_length = window_size - len(prices)
        prices = np.concatenate([np.full(pad_length, prices[0]), prices])
    
    # Global normalization
    prices_norm = normalizer.transform(prices)
    
    # Returns
    returns = np.zeros_like(prices)
    returns[1:] = (prices[1:] - prices[:-1]) / (prices[:-1] + eps)
    
    # Log returns
    log_returns = np.zeros_like(prices)
    log_returns[1:] = np.log(prices[1:] + eps) - np.log(prices[:-1] + eps)
    
    # Stack channels
    features = np.stack([
        prices_norm,
        returns,
        log_returns
    ], axis=0)
    
    return torch.from_numpy(features).float()


if __name__ == "__main__":
    """
    Demo: Stratified vs Pure Temporal Splitting
    """
    print("="*80)
    print("Stratified Temporal Sampling for 15-Minute Regime Discovery")
    print("="*80)
    
    # Simulate 7.5M points over 500 days
    n_points = 7_500_000
    n_days = 500
    points_per_day = n_points / n_days
    points_per_15min = points_per_day / (24 * 4)
    
    print("\nData characteristics:")
    print(f"  Total points: {n_points:,}")
    print(f"  Time span: {n_days} days")
    print(f"  Points per day: {points_per_day:,.0f}")
    print(f"  Points per 15min: {points_per_15min:.0f}")
    
    # Simulate timestamps
    timestamps = np.linspace(0, n_days * 86400, n_points)
    
    # Simulate price with macro trend + micro regimes
    # Macro: bull -> peak -> correction
    macro_trend = np.concatenate([
        np.linspace(60000, 120000, int(n_points * 0.7)),  # Bull (70%)
        np.linspace(120000, 110000, int(n_points * 0.2)),  # Peak (20%)
        np.linspace(110000, 85000, int(n_points * 0.1))    # Correction (10%)
    ])
    
    # Micro: random volatility spikes throughout
    micro_volatility = np.random.choice(
        [100, 500, 1000, 2000],  # Different volatility regimes
        size=n_points,
        p=[0.4, 0.3, 0.2, 0.1]  # 40% low vol, 30% medium, 20% high, 10% extreme
    )
    
    prices = macro_trend + np.random.randn(n_points) * micro_volatility
    
    print("\n1. PURE TEMPORAL SPLIT (Wrong for micro-regime discovery):")
    print("-" * 60)
    
    n_samples = 100000  # Simulate valid samples
    valid_indices = np.arange(0, n_samples)
    
    # Pure temporal
    train_temp = valid_indices[:70000]
    val_temp = valid_indices[70000:90000]
    test_temp = valid_indices[90000:]
    
    # Check price distribution
    train_prices_temp = prices[train_temp]
    val_prices_temp = prices[val_temp]
    test_prices_temp = prices[test_temp]
    
    print(f"Train: mean=${train_prices_temp.mean():,.0f}, std=${train_prices_temp.std():,.0f}")
    print(f"Val:   mean=${val_prices_temp.mean():,.0f}, std=${val_prices_temp.std():,.0f}")
    print(f"Test:  mean=${test_prices_temp.mean():,.0f}, std=${test_prices_temp.std():,.0f}")
    print("⚠ Different means! Model learns: train=bull, val=peak, test=correction")
    
    print("\n2. STRATIFIED TEMPORAL SPLIT (Correct for micro-regime discovery):")
    print("-" * 60)
    
    # Stratified (20 strata)
    n_strata = 20
    stratum_size = n_samples // n_strata
    
    train_strat = []
    val_strat = []
    test_strat = []
    
    for i in range(n_strata):
        stratum_start = i * stratum_size
        stratum_end = (i + 1) * stratum_size
        stratum_indices = valid_indices[stratum_start:stratum_end]
        
        shuffled = np.random.permutation(stratum_indices)
        
        train_strat.extend(shuffled[:int(len(shuffled) * 0.7)])
        val_strat.extend(shuffled[int(len(shuffled) * 0.7):int(len(shuffled) * 0.9)])
        test_strat.extend(shuffled[int(len(shuffled) * 0.9):])
    
    train_strat = np.array(train_strat)
    val_strat = np.array(val_strat)
    test_strat = np.array(test_strat)
    
    # Check price distribution
    train_prices_strat = prices[train_strat]
    val_prices_strat = prices[val_strat]
    test_prices_strat = prices[test_strat]
    
    print(f"Train: mean=${train_prices_strat.mean():,.0f}, std=${train_prices_strat.std():,.0f}")
    print(f"Val:   mean=${val_prices_strat.mean():,.0f}, std=${val_prices_strat.std():,.0f}")
    print(f"Test:  mean=${test_prices_strat.mean():,.0f}, std=${test_prices_strat.std():,.0f}")
    print("✓ Similar means! Model must learn micro-regimes (volatility, momentum)")
    
    # Check temporal coverage
    print("\nTemporal coverage:")
    print(f"  Train: t={timestamps[train_strat.min()]:.0f} to t={timestamps[train_strat.max()]:.0f}")
    print(f"  Val:   t={timestamps[val_strat.min()]:.0f} to t={timestamps[val_strat.max()]:.0f}")
    print(f"  Test:  t={timestamps[test_strat.min()]:.0f} to t={timestamps[test_strat.max()]:.0f}")
    print("✓ All splits cover full time range!")
    
    print("\n" + "="*80)
    print("Summary:")
    print("  Pure temporal: Learns macro trends (bull/bear/sideways)")
    print("  Stratified: Learns micro regimes (volatility/momentum/chop)")
    print("  For 15-minute scalping: USE STRATIFIED!")
    print("="*80)