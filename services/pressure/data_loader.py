"""
Robust Data Loading Pipeline for OrderBookFeaturizer

This module provides comprehensive data loading from MongoDB with validation,
gap detection, and quality metrics for order book data.
"""

import datetime as dt
import logging
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import numpy as np

from cryptotrading.data.book import OrderBookMongoAdapter
from cryptotrading.data.price import PriceMongoAdapter
from cryptotrading.data.models import OrderBookSnapshot
from .pressure_features import OrderBookFeaturizer

logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Metrics for data quality assessment"""
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    gap_count: int = 0
    gap_duration_total: float = 0.0
    avg_gap_duration: float = 0.0
    max_gap_duration: float = 0.0
    data_completeness: float = 0.0
    timestamp_anomalies: int = 0
    crossed_books: int = 0
    negative_prices: int = 0
    zero_volumes: int = 0
    duplicate_timestamps: int = 0
    missing_fields: List[str] = field(default_factory=list)
    quality_score: float = 0.0  # 0-100 scale


@dataclass
class GapInfo:
    """Information about data gaps"""
    start_time: dt.datetime
    end_time: dt.datetime
    duration_seconds: float
    expected_interval: float
    gap_type: str  # 'missing', 'large_gap', 'anomaly'


class OrderBookDataLoader:
    """
    Comprehensive data loader for OrderBookFeaturizer with validation and gap handling.
    
    Features:
    - Robust data validation and quality metrics
    - Gap detection and reporting
    - Data cleaning and preprocessing
    - Flexible loading strategies
    - Error handling and recovery
    """
    
    def __init__(
        self,
        orderbook_adapter: Optional[OrderBookMongoAdapter] = None,
        price_adapter: Optional[PriceMongoAdapter] = None,
        featurizer: Optional[OrderBookFeaturizer] = None,
        default_token: str = "BTC",
        expected_interval_seconds: float = 1.0,
        max_gap_threshold: float = 300.0,  # 5 minutes
        quality_threshold: float = 70.0,  # Minimum quality score
    ):
        """
        Initialize the data loader.
        
        Args:
            orderbook_adapter: MongoDB adapter for order book data
            price_adapter: MongoDB adapter for price data
            featurizer: OrderBookFeaturizer instance
            default_token: Default token to use
            expected_interval_seconds: Expected time between data points
            max_gap_threshold: Maximum gap size before flagging as issue
            quality_threshold: Minimum quality score for acceptable data
        """
        self.orderbook_adapter = orderbook_adapter or OrderBookMongoAdapter()
        self.price_adapter = price_adapter or PriceMongoAdapter()
        self.featurizer = featurizer or OrderBookFeaturizer()
        self.default_token = default_token
        self.expected_interval_seconds = expected_interval_seconds
        self.max_gap_threshold = max_gap_threshold
        self.quality_threshold = quality_threshold
        
        # Data quality tracking
        self.quality_metrics = DataQualityMetrics()
        self.detected_gaps: List[GapInfo] = []
        
    async def initialize(self):
        """Initialize database connections"""
        await self.orderbook_adapter.initialize()
        await self.price_adapter.initialize()
        logger.info("Data loader initialized successfully")
    
    async def load_orderbook_data(
        self,
        token: str,
        start_time: dt.datetime,
        end_time: dt.datetime,
        validate_data: bool = True,
        fill_gaps: bool = False,
        max_records: Optional[int] = None,
    ) -> Tuple[List[OrderBookSnapshot], DataQualityMetrics]:
        """
        Load order book data with comprehensive validation.
        
        Args:
            token: Token symbol (e.g., "BTC")
            start_time: Start time for data range
            end_time: End time for data range
            validate_data: Whether to perform data validation
            fill_gaps: Whether to attempt gap filling
            max_records: Maximum number of records to load
            
        Returns:
            Tuple of (validated snapshots, quality metrics)
        """
        logger.info(f"Loading order book data for {token} from {start_time} to {end_time}")
        
        # Reset metrics
        self.quality_metrics = DataQualityMetrics()
        self.detected_gaps = []
        
        try:
            snapshots = await self.orderbook_adapter.get_orderbook_data(token, start_time, end_time)
            self.quality_metrics.total_records = len(snapshots)
            
            if validate_data:
                # Validate data and detect issues
                validated_snapshots = await self._validate_data(snapshots)
                
                # Detect gaps
                await self._detect_gaps(validated_snapshots)
                
                # Calculate quality metrics
                self._calculate_quality_metrics()
                
                # Apply gap filling if requested
                if fill_gaps and self.detected_gaps:
                    validated_snapshots = await self._fill_gaps(validated_snapshots)
                
                # Check if quality meets threshold
                if self.quality_metrics.quality_score < self.quality_threshold:
                    logger.warning(
                        f"Data quality score {self.quality_metrics.quality_score:.1f} "
                        f"below threshold {self.quality_threshold}"
                    )
                
                return validated_snapshots, self.quality_metrics
            else:
                self.quality_metrics.valid_records = len(snapshots)
                self.quality_metrics.quality_score = 100.0
                return snapshots, self.quality_metrics
                
        except Exception as e:
            logger.error(f"Error loading order book data: {e}")
            raise
        
    
    async def _validate_data(self, snapshots: List[OrderBookSnapshot]) -> List[OrderBookSnapshot]:
        """Validate order book snapshots and filter out invalid ones"""
        valid_snapshots = []
        
        for snapshot in snapshots:
            try:
                # Basic validation
                if not snapshot.bids or not snapshot.asks:
                    self.quality_metrics.missing_fields.append("empty_bids_asks")
                    self.quality_metrics.invalid_records += 1
                    continue
                
                # Check for crossed book
                if snapshot.bids and snapshot.asks:
                    best_bid = snapshot.bids[0][0]
                    best_ask = snapshot.asks[0][0]
                    
                    if best_bid >= best_ask:
                        self.quality_metrics.crossed_books += 1
                        self.quality_metrics.invalid_records += 1
                        continue
                
                # Check for negative prices
                for price, volume in snapshot.bids + snapshot.asks:
                    if price <= 0:
                        self.quality_metrics.negative_prices += 1
                        self.quality_metrics.invalid_records += 1
                        break
                    if volume <= 0:
                        self.quality_metrics.zero_volumes += 1
                
                # Check timestamp anomalies
                if snapshot.timestamp <= 0:
                    self.quality_metrics.timestamp_anomalies += 1
                    self.quality_metrics.invalid_records += 1
                    continue
                
                valid_snapshots.append(snapshot)
                
            except Exception as e:
                logger.warning(f"Validation error for snapshot: {e}")
                self.quality_metrics.invalid_records += 1
        
        self.quality_metrics.valid_records = len(valid_snapshots)
        logger.info(f"Validation complete: {len(valid_snapshots)}/{len(snapshots)} snapshots valid")
        
        return valid_snapshots
    
    async def _detect_gaps(self, snapshots: List[OrderBookSnapshot]):
        """Detect gaps in the data timeline"""
        if len(snapshots) < 2:
            return
        
        gaps = []
        total_gap_duration = 0.0
        
        for i in range(1, len(snapshots)):
            prev_time = snapshots[i-1].timestamp
            curr_time = snapshots[i].timestamp
            time_diff = curr_time - prev_time
            
            # Check for gap larger than expected interval
            if time_diff > self.expected_interval_seconds * 1.5:  # 50% tolerance
                gap_info = GapInfo(
                    start_time=dt.datetime.fromtimestamp(prev_time, tz=dt.timezone.utc),
                    end_time=dt.datetime.fromtimestamp(curr_time, tz=dt.timezone.utc),
                    duration_seconds=time_diff,
                    expected_interval=self.expected_interval_seconds,
                    gap_type="large_gap" if time_diff > self.max_gap_threshold else "missing"
                )
                gaps.append(gap_info)
                total_gap_duration += time_diff
        
        self.detected_gaps = gaps
        self.quality_metrics.gap_count = len(gaps)
        self.quality_metrics.gap_duration_total = total_gap_duration
        self.quality_metrics.avg_gap_duration = total_gap_duration / len(gaps) if gaps else 0.0
        self.quality_metrics.max_gap_duration = max(gap.duration_seconds for gap in gaps) if gaps else 0.0
        
        logger.info(f"Detected {len(gaps)} gaps, total duration: {total_gap_duration:.1f}s")
    
    def _calculate_quality_metrics(self):
        """Calculate overall data quality metrics"""
        total = self.quality_metrics.total_records
        if total == 0:
            self.quality_metrics.quality_score = 0.0
            return
        
        # Calculate data completeness (accounting for gaps)
        expected_records = int((self.detected_gaps[-1].end_time.timestamp() - 
                               self.detected_gaps[0].start_time.timestamp()) / 
                              self.expected_interval_seconds) if self.detected_gaps else total
        
        self.quality_metrics.data_completeness = (self.quality_metrics.valid_records / max(expected_records, 1)) * 100
        
        # Calculate quality score (0-100)
        validity_score = (self.quality_metrics.valid_records / total) * 100
        completeness_score = self.quality_metrics.data_completeness
        gap_penalty = min((self.quality_metrics.gap_count / total) * 100, 50)  # Max 50 point penalty
        error_penalty = min((self.quality_metrics.invalid_records / total) * 100, 30)  # Max 30 point penalty
        
        self.quality_metrics.quality_score = max(0, validity_score + completeness_score - gap_penalty - error_penalty)
        
        logger.info(f"Data quality score: {self.quality_metrics.quality_score:.1f}/100")
    
    async def _fill_gaps(self, snapshots: List[OrderBookSnapshot]) -> List[OrderBookSnapshot]:
        """Attempt to fill gaps in the data using interpolation"""
        if not self.detected_gaps:
            return snapshots
        
        logger.info(f"Attempting to fill {len(self.detected_gaps)} gaps")
        filled_snapshots = snapshots.copy()
        
        for gap in self.detected_gaps:
            if gap.gap_type != "large_gap" and gap.duration_seconds < 60:  # Only fill small gaps
                # Find snapshots before and after gap
                prev_snapshot = None
                next_snapshot = None
                
                for snapshot in filled_snapshots:
                    if snapshot.timestamp < gap.start_time.timestamp():
                        prev_snapshot = snapshot
                    elif snapshot.timestamp > gap.end_time.timestamp():
                        next_snapshot = snapshot
                        break
                
                if prev_snapshot and next_snapshot:
                    # Simple linear interpolation for missing snapshots
                    num_missing = int(gap.duration_seconds / self.expected_interval_seconds)
                    
                    for i in range(1, num_missing + 1):
                        # Interpolate timestamp
                        interp_time = prev_snapshot.timestamp + (i * self.expected_interval_seconds)
                        
                        # Interpolate order book (simple approach)
                        interp_snapshot = self._interpolate_snapshots(prev_snapshot, next_snapshot, interp_time)
                        filled_snapshots.append(interp_snapshot)
        
        # Sort by timestamp
        filled_snapshots.sort(key=lambda x: x.timestamp)
        logger.info(f"Gap filling complete, now have {len(filled_snapshots)} snapshots")
        
        return filled_snapshots
    
    def _interpolate_snapshots(self, prev_snapshot: OrderBookSnapshot, next_snapshot: OrderBookSnapshot, target_time: float) -> OrderBookSnapshot:
        """Simple linear interpolation between two snapshots"""
        # Calculate interpolation weights
        total_time = next_snapshot.timestamp - prev_snapshot.timestamp
        weight = (target_time - prev_snapshot.timestamp) / total_time if total_time > 0 else 0.5
        
        # Interpolate mid price
        interp_mid_price = prev_snapshot.mid_price + weight * (next_snapshot.mid_price - prev_snapshot.mid_price)
        
        # For order book, use a simple approach: blend the best levels
        interp_bids = []
        interp_asks = []
        
        # Take top few levels from each snapshot and interpolate
        num_levels = min(5, len(prev_snapshot.bids), len(next_snapshot.bids))
        for i in range(num_levels):
            prev_price, prev_volume = prev_snapshot.bids[i]
            next_price, next_volume = next_snapshot.bids[i]
            
            interp_price = prev_price + weight * (next_price - prev_price)
            interp_volume = prev_volume + weight * (next_volume - prev_volume)
            interp_bids.append((interp_price, interp_volume))
        
        num_levels = min(5, len(prev_snapshot.asks), len(next_snapshot.asks))
        for i in range(num_levels):
            prev_price, prev_volume = prev_snapshot.asks[i]
            next_price, next_volume = next_snapshot.asks[i]
            
            interp_price = prev_price + weight * (next_price - prev_price)
            interp_volume = prev_volume + weight * (next_volume - prev_volume)
            interp_asks.append((interp_price, interp_volume))
        
        return OrderBookSnapshot(
            timestamp=target_time,
            bids=interp_bids,
            asks=interp_asks,
            mid_price=interp_mid_price
        )
    
    async def load_and_featurize(
        self,
        token: str,
        start_time: dt.datetime,
        end_time: dt.datetime,
        validate_data: bool = True,
        fill_gaps: bool = False,
        max_records: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load data and extract features in one operation.
        
        Returns:
            Tuple of (feature_array, metadata_dict)
        """
        # Load validated data
        snapshots, quality_metrics = await self.load_orderbook_data(
            token, start_time, end_time, validate_data, fill_gaps, max_records
        )
        
        if not snapshots:
            logger.warning("No valid snapshots to featurize")
            return np.array([]), {"quality_metrics": quality_metrics, "features": {}}
        
        # Extract features
        all_features = []
        metadata = {
            "token": token,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_snapshots": len(snapshots),
            "quality_metrics": quality_metrics.__dict__,
            "features": {}
        }
        
        for i, snapshot in enumerate(snapshots):
            try:
                features = self.featurizer.extract_features(snapshot, token, validate=False)
                flat_features = self.featurizer.flatten_features(features)
                all_features.append(flat_features)
                
                # Store first snapshot's feature details as example
                if i == 0:
                    metadata["features"] = {
                        "feature_names": list(features.keys()),
                        "feature_sizes": {k: v.shape if hasattr(v, 'shape') else len(v) for k, v in features.items()},
                        "total_feature_dim": len(flat_features)
                    }
                
            except Exception as e:
                logger.warning(f"Error extracting features for snapshot {i}: {e}")
                # Add zero features to maintain sequence
                zero_features = np.zeros(self.featurizer.get_feature_dim(), dtype=np.float32)
                all_features.append(zero_features)
        
        feature_array = np.array(all_features, dtype=np.float32)
        
        logger.info(f"Extracted features: {feature_array.shape}, quality score: {quality_metrics.quality_score:.1f}")
        
        return feature_array, metadata
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Get detailed quality report"""
        report = {
            "quality_metrics": self.quality_metrics.__dict__,
            "gaps": [
                {
                    "start_time": gap.start_time.isoformat(),
                    "end_time": gap.end_time.isoformat(),
                    "duration_seconds": gap.duration_seconds,
                    "gap_type": gap.gap_type
                }
                for gap in self.detected_gaps
            ],
            "summary": {
                "total_gaps": len(self.detected_gaps),
                "total_gap_time": self.quality_metrics.gap_duration_total,
                "avg_gap_duration": self.quality_metrics.avg_gap_duration,
                "max_gap_duration": self.quality_metrics.max_gap_duration,
                "data_quality_acceptable": self.quality_metrics.quality_score >= self.quality_threshold
            }
        }
        
        return report


# Utility function for easy usage
async def load_orderbook_features(
    token: str,
    start_time: dt.datetime,
    end_time: dt.datetime,
    featurizer: Optional[OrderBookFeaturizer] = None,
    validate_data: bool = True,
    fill_gaps: bool = False,
    max_records: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function to load order book data and extract features.
    
    Args:
        token: Token symbol
        start_time: Start time
        end_time: End time
        featurizer: OrderBookFeaturizer instance
        validate_data: Whether to validate data
        fill_gaps: Whether to fill gaps
        max_records: Maximum records to load
        
    Returns:
        Tuple of (features_array, metadata)
    """
    loader = OrderBookDataLoader(featurizer=featurizer)
    await loader.initialize()
    
    return await loader.load_and_featurize(
        token=token,
        start_time=start_time,
        end_time=end_time,
        validate_data=validate_data,
        fill_gaps=fill_gaps,
        max_records=max_records
    )
