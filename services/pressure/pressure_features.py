"""
High-Performance Vectorized Feature Extraction with State Management.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import deque
from dataclasses import dataclass
import logging
from collections import defaultdict
import warnings

logger = logging.getLogger(__name__)

@dataclass
class OrderBookSnapshot:
    """Single order book snapshot with validation"""
    timestamp: float
    bids: List[Tuple[float, float]]  # [(price, volume), ...]
    asks: List[Tuple[float, float]]
    mid_price: float
    
    def __post_init__(self):
        """Validate order book data"""
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate order book for data quality issues.
        
        Checks:
        - Non-empty sides
        - No crossed book (best bid < best ask)
        - No negative prices/volumes
        - Monotonic price levels
        """
        if not self.bids or not self.asks:
            raise ValueError("Order book has empty side")
        
        # Check for crossed book
        best_bid = self.bids[0][0]
        best_ask = self.asks[0][0]
        
        if best_bid >= best_ask:
            raise ValueError(f"Crossed book: bid={best_bid} >= ask={best_ask}")
        
        # Check for negative values
        for price, vol in self.bids + self.asks:
            if price <= 0 or vol <= 0:
                raise ValueError(f"Invalid price/volume: {price}/{vol}")
        
        # Check bid prices are descending
        bid_prices = [p for p, _ in self.bids]
        if bid_prices != sorted(bid_prices, reverse=True):
            warnings.warn("Bid prices not monotonically descending")
        
        # Check ask prices are ascending
        ask_prices = [p for p, _ in self.asks]
        if ask_prices != sorted(ask_prices):
            warnings.warn("Ask prices not monotonically ascending")
        
        # Verify mid price
        expected_mid = (best_bid + best_ask) / 2
        if abs(self.mid_price - expected_mid) > 0.01 * expected_mid:
            warnings.warn(f"Mid price mismatch: {self.mid_price} vs {expected_mid}")
            self.mid_price = expected_mid
        
        return True


class AdaptiveBucketCalculator:
    """
    Calculate adaptive bucket ranges based on asset characteristics.
    
    FIXES: Issue #10 - No more fixed bucket ranges
    """
    
    def __init__(self, asset_class: str = 'major_crypto'):
        """
        Args:
            asset_class: One of:
                - 'major_crypto': BTC, ETH (tight spreads, deep books)
                - 'altcoin': Mid-cap tokens (wider spreads)
                - 'micro_cap': Low liquidity tokens (very wide spreads)
                - 'forex': FX pairs (ultra-tight spreads)
                - 'equity': Stocks (variable)
        """
        self.asset_class = asset_class
        self.bucket_ranges = self._get_default_ranges()
        
    def _get_default_ranges(self) -> List[Tuple[float, float]]:
        """Get default bucket ranges for asset class"""
        if self.asset_class == 'major_crypto':
            # BTC/ETH on major exchanges: 1bp typical spread
            return [
                (0.00000, 0.00010),  # Ultra-tight: 0-1bp
                (0.00010, 0.00025),  # Tight: 1-2.5bp
                (0.00025, 0.00050),  # Near: 2.5-5bp
                (0.00050, 0.00100),  # Close: 5-10bp
                (0.00100, 0.00250),  # Medium: 10-25bp
                (0.00250, 0.00500),  # Far: 25-50bp
                (0.00500, 0.01000),  # Deep: 50-100bp
                (0.01000, 0.02000),  # Very deep: 100-200bp
            ]
        elif self.asset_class == 'altcoin':
            # Mid-cap: 5-10bp typical spread
            return [
                (0.00000, 0.00050),  # Tight: 0-5bp
                (0.00050, 0.00100),  # Near: 5-10bp
                (0.00100, 0.00250),  # Close: 10-25bp
                (0.00250, 0.00500),  # Medium: 25-50bp
                (0.00500, 0.01000),  # Far: 50-100bp
                (0.01000, 0.02500),  # Deep: 100-250bp
                (0.02500, 0.05000),  # Very deep: 250-500bp
                (0.05000, 0.10000),  # Extreme: 500bp-1000bp
            ]
        elif self.asset_class == 'micro_cap':
            # Low liquidity: 50bp+ spreads
            return [
                (0.00000, 0.00250),  # 0-25bp
                (0.00250, 0.00500),  # 25-50bp
                (0.00500, 0.01000),  # 50-100bp
                (0.01000, 0.02500),  # 100-250bp
                (0.02500, 0.05000),  # 250-500bp
                (0.05000, 0.10000),  # 500bp-1000bp
                (0.10000, 0.20000),  # 1000-2000bp
                (0.20000, 0.50000),  # 2000-5000bp
            ]
        elif self.asset_class == 'forex':
            # FX: sub-1bp spreads
            return [
                (0.000000, 0.000010),  # 0-0.1bp
                (0.000010, 0.000050),  # 0.1-0.5bp
                (0.000050, 0.000100),  # 0.5-1bp
                (0.000100, 0.000250),  # 1-2.5bp
                (0.000250, 0.000500),  # 2.5-5bp
                (0.000500, 0.001000),  # 5-10bp
                (0.001000, 0.002500),  # 10-25bp
                (0.002500, 0.005000),  # 25-50bp
            ]
        else:
            # Default to major crypto
            return self._get_default_ranges()
    
    def calibrate_from_data(self, snapshots: List[OrderBookSnapshot], num_buckets: int = 8):
        """
        Automatically calibrate bucket ranges from historical data.
        
        Uses percentiles of actual order book depth distribution.
        """
        all_distances = []
        
        for snapshot in snapshots[:1000]:  # Sample first 1000
            mid = snapshot.mid_price
            
            # Collect bid distances
            for price, _ in snapshot.bids[:50]:
                distance_pct = (mid - price) / mid
                all_distances.append(distance_pct)
            
            # Collect ask distances  
            for price, _ in snapshot.asks[:50]:
                distance_pct = (price - mid) / mid
                all_distances.append(distance_pct)
        
        if not all_distances:
            logger.warning("No distances collected, using defaults")
            return
        
        # Calculate percentiles for bucket boundaries
        all_distances = np.array(all_distances)
        percentiles = np.linspace(0, 100, num_buckets + 1)
        boundaries = np.percentile(all_distances, percentiles)
        
        # Create bucket ranges
        self.bucket_ranges = [
            (boundaries[i], boundaries[i + 1])
            for i in range(len(boundaries) - 1)
        ]
        
        logger.info(f"Calibrated {num_buckets} buckets from {len(snapshots)} snapshots")
        logger.info(f"Bucket ranges: {self.bucket_ranges}")

class OrderBookFeaturizer:
    """
    Convert order book snapshots into normalized, scale-invariant features.
    
    FIXES:
    - Issue #1: Complete implementation
    - Issue #4: Proper feature standardization
    - Issue #9: Data validation
    - Issue #10: Adaptive buckets
    """
    
    def __init__(self,
                 bucket_calculator: Optional[AdaptiveBucketCalculator] = None,
                 volume_window: int = 100,
                 normalize_features: bool = True,
                 outlier_clip: float = 5.0):
        """
        Args:
            bucket_calculator: Adaptive bucket calculator (if None, uses major_crypto defaults)
            volume_window: Window for normalizing volumes
            normalize_features: Whether to z-score normalize features
            outlier_clip: Clip features to [-outlier_clip, outlier_clip] std devs
        """
        self.bucket_calculator = bucket_calculator or AdaptiveBucketCalculator('major_crypto')
        self.bucket_ranges = self.bucket_calculator.bucket_ranges
        self.volume_window = volume_window
        self.normalize_features = normalize_features
        self.outlier_clip = outlier_clip
        
        # Volume history for normalization
        self.volume_history = defaultdict(lambda: deque(maxlen=volume_window))
        
        # Feature statistics for standardization
        self.feature_mean = None
        self.feature_std = None
        self.is_fitted = False
        
    def _bucket_orders(self, 
                       orders: List[Tuple[float, float]], 
                       mid_price: float,
                       side: str) -> np.ndarray:
        """
        Aggregate orders into distance-based buckets.
        
        Args:
            orders: List of (price, volume)
            mid_price: Current mid price
            side: 'bid' or 'ask'
            
        Returns:
            Array of aggregated volumes per bucket
        """
        buckets = np.zeros(len(self.bucket_ranges))
        
        for price, volume in orders:
            # Calculate distance from mid as percentage
            if side == 'bid':
                distance_pct = (mid_price - price) / mid_price
            else:  # ask
                distance_pct = (price - mid_price) / mid_price
            
            # Find appropriate bucket
            for i, (min_pct, max_pct) in enumerate(self.bucket_ranges):
                if min_pct <= distance_pct < max_pct:
                    buckets[i] += volume
                    break
        
        return buckets
    
    def _normalize_volumes(self, volumes: np.ndarray, token: str) -> np.ndarray:
        """
        Normalize volumes by recent average to handle different scales.
        
        FIXES: Issue #4 - Now properly normalizes
        """
        total_volume = np.sum(volumes)
        self.volume_history[token].append(total_volume)
        
        if len(self.volume_history[token]) < 10:
            # Not enough history, use raw volumes
            return volumes / max(total_volume, 1.0)
        
        # Compute robust average (median to handle outliers)
        recent_volumes = list(self.volume_history[token])
        median_volume = np.median(recent_volumes)
        
        # Normalize by median
        return volumes / max(median_volume, 1e-8)
    
    def _extract_microstructure(self, snapshot: OrderBookSnapshot) -> np.ndarray:
        """
        Extract additional microstructure features.
        
        Features:
        - Bid/ask volume ratio at best prices
        - Depth imbalance (total bid vs ask within thresholds)
        - Book slope (volume distribution steepness)
        - Concentration (volume in top levels vs total)
        - Spread percentage
        - Depth asymmetry
        """
        features = []
        
        mid = snapshot.mid_price
        
        # 1. Best level imbalance
        if len(snapshot.bids) > 0 and len(snapshot.asks) > 0:
            best_bid_vol = snapshot.bids[0][1]
            best_ask_vol = snapshot.asks[0][1]
            best_imbalance = (best_bid_vol - best_ask_vol) / (best_bid_vol + best_ask_vol + 1e-8)
            features.append(best_imbalance)
        else:
            features.append(0.0)
        
        # 2. Near-spread depth imbalance (within 10bp)
        threshold_pct = 0.001
        near_bid_vol = sum(vol for price, vol in snapshot.bids 
                          if (mid - price) / mid <= threshold_pct)
        near_ask_vol = sum(vol for price, vol in snapshot.asks 
                          if (price - mid) / mid <= threshold_pct)
        near_imbalance = (near_bid_vol - near_ask_vol) / (near_bid_vol + near_ask_vol + 1e-8)
        features.append(near_imbalance)
        
        # 3. Total depth imbalance
        total_bid_vol = sum(vol for _, vol in snapshot.bids)
        total_ask_vol = sum(vol for _, vol in snapshot.asks)
        total_imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-8)
        features.append(total_imbalance)
        
        # 4. Volume concentration (top 5 levels / total)
        top5_bid = sum(vol for _, vol in snapshot.bids[:5])
        top5_ask = sum(vol for _, vol in snapshot.asks[:5])
        bid_concentration = top5_bid / (total_bid_vol + 1e-8)
        ask_concentration = top5_ask / (total_ask_vol + 1e-8)
        features.append(bid_concentration)
        features.append(ask_concentration)
        
        # 5. Spread percentage
        if len(snapshot.bids) > 0 and len(snapshot.asks) > 0:
            spread_pct = (snapshot.asks[0][0] - snapshot.bids[0][0]) / mid
            features.append(spread_pct)
        else:
            features.append(0.0)
        
        # 6. Depth asymmetry (how balanced is the book)
        depth_asymmetry = abs(total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-8)
        features.append(depth_asymmetry)
        
        # 7. Book slope (how quickly volume decays with distance)
        # Measure as average volume ratio between levels
        bid_slopes = []
        for i in range(min(4, len(snapshot.bids) - 1)):
            if snapshot.bids[i][1] > 0:
                slope = snapshot.bids[i + 1][1] / (snapshot.bids[i][1] + 1e-8)
                bid_slopes.append(slope)
        
        ask_slopes = []
        for i in range(min(4, len(snapshot.asks) - 1)):
            if snapshot.asks[i][1] > 0:
                slope = snapshot.asks[i + 1][1] / (snapshot.asks[i][1] + 1e-8)
                ask_slopes.append(slope)
        
        avg_bid_slope = np.mean(bid_slopes) if bid_slopes else 1.0
        avg_ask_slope = np.mean(ask_slopes) if ask_slopes else 1.0
        features.append(avg_bid_slope)
        features.append(avg_ask_slope)
        
        return np.array(features, dtype=np.float32)
    
    def extract_features(self, 
                        snapshot: OrderBookSnapshot,
                        token: str,
                        validate: bool = True) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive order book features.
        
        FIXES: Issue #9 - Added validation
        
        Returns feature dictionary with:
        - bid_buckets: Aggregated bid volumes per distance bucket
        - ask_buckets: Aggregated ask volumes per distance bucket
        - imbalance_buckets: (bid - ask) / (bid + ask) per bucket
        - bid_depth: Cumulative bid volume per bucket
        - ask_depth: Cumulative ask volume per bucket
        - spread: Normalized spread
        - micro_structure: Additional microstructure features
        """
        # Validate if requested
        if validate:
            try:
                snapshot.validate()
            except ValueError as e:
                logger.error(f"Invalid order book for {token}: {e}")
                # Return zero features rather than crashing
                return self._get_zero_features()
        
        try:
            # Extract and bucket orders
            bid_buckets = self._bucket_orders(snapshot.bids, snapshot.mid_price, 'bid')
            ask_buckets = self._bucket_orders(snapshot.asks, snapshot.mid_price, 'ask')
            
            # Normalize volumes
            bid_buckets_norm = self._normalize_volumes(bid_buckets, f"{token}_bid")
            ask_buckets_norm = self._normalize_volumes(ask_buckets, f"{token}_ask")
            
            # Calculate imbalances
            total = bid_buckets_norm + ask_buckets_norm + 1e-8
            imbalance_buckets = (bid_buckets_norm - ask_buckets_norm) / total
            
            # Cumulative depth (market depth at each level)
            bid_depth = np.cumsum(bid_buckets_norm)
            ask_depth = np.cumsum(ask_buckets_norm)
            
            # Spread features
            if len(snapshot.bids) > 0 and len(snapshot.asks) > 0:
                best_bid = snapshot.bids[0][0]
                best_ask = snapshot.asks[0][0]
                spread_pct = (best_ask - best_bid) / snapshot.mid_price
            else:
                spread_pct = 0.0
            
            # Microstructure features
            micro_features = self._extract_microstructure(snapshot)
            
            return {
                'bid_buckets': bid_buckets_norm.astype(np.float32),
                'ask_buckets': ask_buckets_norm.astype(np.float32),
                'imbalance_buckets': imbalance_buckets.astype(np.float32),
                'bid_depth': bid_depth.astype(np.float32),
                'ask_depth': ask_depth.astype(np.float32),
                'spread': np.array([spread_pct], dtype=np.float32),
                'micro_structure': micro_features,
            }
        
        except Exception as e:
            logger.error(f"Error extracting features for {token}: {e}")
            return self._get_zero_features()
    
    def _get_zero_features(self) -> Dict[str, np.ndarray]:
        """Return zero-filled features on error"""
        n_buckets = len(self.bucket_ranges)
        return {
            'bid_buckets': np.zeros(n_buckets, dtype=np.float32),
            'ask_buckets': np.zeros(n_buckets, dtype=np.float32),
            'imbalance_buckets': np.zeros(n_buckets, dtype=np.float32),
            'bid_depth': np.zeros(n_buckets, dtype=np.float32),
            'ask_depth': np.zeros(n_buckets, dtype=np.float32),
            'spread': np.zeros(1, dtype=np.float32),
            'micro_structure': np.zeros(9, dtype=np.float32),  # Updated size
        }
    
    def flatten_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten feature dictionary into single vector for neural net"""
        feature_vector = np.concatenate([
            features['bid_buckets'],
            features['ask_buckets'],
            features['imbalance_buckets'],
            features['bid_depth'],
            features['ask_depth'],
            features['spread'],
            features['micro_structure'],
        ])
        
        return feature_vector.astype(np.float32)
    
    def fit_normalizer(self, feature_vectors: np.ndarray):
        """
        Fit feature standardization from training data.
        
        FIXES: Issue #4 - Proper z-score normalization
        
        Args:
            feature_vectors: (N, feature_dim) array of training features
        """
        if not self.normalize_features:
            return
        
        self.feature_mean = np.mean(feature_vectors, axis=0)
        self.feature_std = np.std(feature_vectors, axis=0)
        
        # Avoid division by zero
        self.feature_std = np.where(self.feature_std < 1e-8, 1.0, self.feature_std)
        
        self.is_fitted = True
        
        logger.info(f"Fitted normalizer on {len(feature_vectors)} samples")
        logger.info(f"Feature mean: {self.feature_mean[:5]}...")
        logger.info(f"Feature std: {self.feature_std[:5]}...")
    
    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization to features.
        
        Args:
            features: (N, feature_dim) or (feature_dim,) array
            
        Returns:
            Normalized features
        """
        if not self.normalize_features or not self.is_fitted:
            return features
        
        normalized = (features - self.feature_mean) / self.feature_std
        
        # Clip outliers
        if self.outlier_clip is not None:
            normalized = np.clip(normalized, -self.outlier_clip, self.outlier_clip)
        
        return normalized.astype(np.float32)
    
    def get_feature_dim(self) -> int:
        """Get total feature dimensionality"""
        n_buckets = len(self.bucket_ranges)
        return (
            n_buckets +  # bid_buckets
            n_buckets +  # ask_buckets
            n_buckets +  # imbalance_buckets
            n_buckets +  # bid_depth
            n_buckets +  # ask_depth
            1 +          # spread
            9            # micro_structure (updated)
        )