"""
Integration Tests for OrderBookDataLoader

These tests verify that the data loading pipeline works correctly
with various data scenarios and edge cases.
"""

import datetime as dt
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from data_loader import OrderBookDataLoader, DataQualityMetrics, GapInfo
from pressure_features import OrderBookFeaturizer, OrderBookSnapshot


class TestOrderBookDataLoader:
    """Test suite for OrderBookDataLoader"""
    
    @pytest.fixture
    def mock_featurizer(self):
        """Mock featurizer for testing"""
        featurizer = MagicMock(spec=OrderBookFeaturizer)
        featurizer.get_feature_dim.return_value = 50
        featurizer.extract_features.return_value = {
            'bid_buckets': np.ones(8),
            'ask_buckets': np.ones(8),
            'imbalance_buckets': np.zeros(8),
            'bid_depth': np.ones(8),
            'ask_depth': np.ones(8),
            'spread': np.array([0.001]),
            'micro_structure': np.ones(9)
        }
        featurizer.flatten_features.return_value = np.ones(50)
        return featurizer
    
    @pytest.fixture
    def mock_adapters(self):
        """Mock MongoDB adapters"""
        orderbook_adapter = MagicMock()
        price_adapter = MagicMock()
        
        # Mock initialization
        orderbook_adapter.initialize = AsyncMock()
        price_adapter.initialize = AsyncMock()
        
        # Mock collection
        orderbook_adapter.composite_order_book_collection = AsyncMock()
        price_adapter.price_collection = AsyncMock()
        
        return orderbook_adapter, price_adapter
    
    @pytest.fixture
    def sample_orderbook_data(self):
        """Sample order book data for testing"""
        return [
            {
                "timestamp": dt.datetime(2024, 1, 1, 10, 0, 0, tzinfo=dt.timezone.utc),
                "metadata": {"token": "BTC"},
                "book": {
                    "bids": [(100.0, 1.0), (99.9, 2.0), (99.8, 3.0)],
                    "asks": [(100.1, 1.0), (100.2, 2.0), (100.3, 3.0)],
                    "mid_price": 100.05
                }
            },
            {
                "timestamp": dt.datetime(2024, 1, 1, 10, 0, 1, tzinfo=dt.timezone.utc),
                "metadata": {"token": "BTC"},
                "book": {
                    "bids": [(100.1, 1.0), (100.0, 2.0), (99.9, 3.0)],
                    "asks": [(100.2, 1.0), (100.3, 2.0), (100.4, 3.0)],
                    "mid_price": 100.15
                }
            },
            {
                "timestamp": dt.datetime(2024, 1, 1, 10, 0, 2, tzinfo=dt.timezone.utc),
                "metadata": {"token": "BTC"},
                "book": {
                    "bids": [(100.2, 1.0), (100.1, 2.0), (100.0, 3.0)],
                    "asks": [(100.3, 1.0), (100.4, 2.0), (100.5, 3.0)],
                    "mid_price": 100.25
                }
            }
        ]
    
    def test_initialization(self, mock_featurizer, mock_adapters):
        """Test loader initialization"""
        orderbook_adapter, price_adapter = mock_adapters
        
        loader = OrderBookDataLoader(
            orderbook_adapter=orderbook_adapter,
            price_adapter=price_adapter,
            featurizer=mock_featurizer,
            expected_interval_seconds=2.0,
            quality_threshold=75.0
        )
        
        assert loader.orderbook_adapter == orderbook_adapter
        assert loader.price_adapter == price_adapter
        assert loader.featurizer == mock_featurizer
        assert loader.expected_interval_seconds == 2.0
        assert loader.quality_threshold == 75.0
    
    @pytest.mark.asyncio
    async def test_initialize(self, mock_featurizer, mock_adapters):
        """Test async initialization"""
        orderbook_adapter, price_adapter = mock_adapters
        loader = OrderBookDataLoader(
            orderbook_adapter=orderbook_adapter,
            price_adapter=price_adapter,
            featurizer=mock_featurizer
        )
        
        await loader.initialize()
        
        orderbook_adapter.initialize.assert_called_once()
        price_adapter.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_orderbook_data_success(self, mock_featurizer, mock_adapters, sample_orderbook_data):
        """Test successful data loading"""
        orderbook_adapter, price_adapter = mock_adapters
        
        # Mock aggregate response
        mock_cursor = AsyncMock()
        mock_cursor.__aiter__ = AsyncMock(return_value=iter(sample_orderbook_data))
        orderbook_adapter.composite_order_book_collection.aggregate.return_value = mock_cursor
        
        loader = OrderBookDataLoader(
            orderbook_adapter=orderbook_adapter,
            price_adapter=price_adapter,
            featurizer=mock_featurizer
        )
        await loader.initialize()
        
        start_time = dt.datetime(2024, 1, 1, 9, 0, 0, tzinfo=dt.timezone.utc)
        end_time = dt.datetime(2024, 1, 1, 11, 0, 0, tzinfo=dt.timezone.utc)
        
        snapshots, quality_metrics = await loader.load_orderbook_data(
            token="BTC",
            start_time=start_time,
            end_time=end_time,
            validate_data=True
        )
        
        assert len(snapshots) == 3
        assert quality_metrics.total_records == 3
        assert quality_metrics.valid_records == 3
        assert quality_metrics.quality_score > 90.0
        
        # Verify snapshot data
        for i, snapshot in enumerate(snapshots):
            assert isinstance(snapshot, OrderBookSnapshot)
            assert len(snapshot.bids) == 3
            assert len(snapshot.asks) == 3
            assert snapshot.mid_price > 0
    
    @pytest.mark.asyncio
    async def test_load_orderbook_data_with_gaps(self, mock_featurizer, mock_adapters):
        """Test data loading with gaps"""
        orderbook_adapter, price_adapter = mock_adapters
        
        # Create data with gaps
        gapped_data = [
            {
                "timestamp": dt.datetime(2024, 1, 1, 10, 0, 0, tzinfo=dt.timezone.utc),
                "metadata": {"token": "BTC"},
                "book": {
                    "bids": [(100.0, 1.0)],
                    "asks": [(100.1, 1.0)],
                    "mid_price": 100.05
                }
            },
            {
                "timestamp": dt.datetime(2024, 1, 1, 10, 0, 10, tzinfo=dt.timezone.utc),  # 10 second gap
                "metadata": {"token": "BTC"},
                "book": {
                    "bids": [(100.1, 1.0)],
                    "asks": [(100.2, 1.0)],
                    "mid_price": 100.15
                }
            }
        ]
        
        mock_cursor = AsyncMock()
        mock_cursor.__aiter__ = AsyncMock(return_value=iter(gapped_data))
        orderbook_adapter.composite_order_book_collection.aggregate.return_value = mock_cursor
        
        loader = OrderBookDataLoader(
            orderbook_adapter=orderbook_adapter,
            price_adapter=price_adapter,
            featurizer=mock_featurizer,
            expected_interval_seconds=1.0
        )
        await loader.initialize()
        
        start_time = dt.datetime(2024, 1, 1, 9, 0, 0, tzinfo=dt.timezone.utc)
        end_time = dt.datetime(2024, 1, 1, 11, 0, 0, tzinfo=dt.timezone.utc)
        
        snapshots, quality_metrics = await loader.load_orderbook_data(
            token="BTC",
            start_time=start_time,
            end_time=end_time,
            validate_data=True
        )
        
        assert len(snapshots) == 2
        assert quality_metrics.gap_count == 1
        assert quality_metrics.gap_duration_total == 10.0
        assert quality_metrics.max_gap_duration == 10.0
    
    @pytest.mark.asyncio
    async def test_load_orderbook_data_invalid_data(self, mock_featurizer, mock_adapters):
        """Test handling of invalid data"""
        orderbook_adapter, price_adapter = mock_adapters
        
        # Create data with invalid entries
        invalid_data = [
            {
                "timestamp": dt.datetime(2024, 1, 1, 10, 0, 0, tzinfo=dt.timezone.utc),
                "metadata": {"token": "BTC"},
                "book": {
                    "bids": [(100.0, 1.0)],
                    "asks": [(100.1, 1.0)],
                    "mid_price": 100.05
                }
            },
            {
                "timestamp": dt.datetime(2024, 1, 1, 10, 0, 1, tzinfo=dt.timezone.utc),
                "metadata": {"token": "BTC"},
                "book": {
                    "bids": [(100.2, 1.0)],  # Bid higher than ask (crossed)
                    "asks": [(100.1, 1.0)],
                    "mid_price": 100.15
                }
            },
            {
                "timestamp": dt.datetime(2024, 1, 1, 10, 0, 2, tzinfo=dt.timezone.utc),
                "metadata": {"token": "BTC"},
                "book": {
                    "bids": [(-10.0, 1.0)],  # Negative price
                    "asks": [(100.1, 1.0)],
                    "mid_price": 100.25
                }
            }
        ]
        
        mock_cursor = AsyncMock()
        mock_cursor.__aiter__ = AsyncMock(return_value=iter(invalid_data))
        orderbook_adapter.composite_order_book_collection.aggregate.return_value = mock_cursor
        
        loader = OrderBookDataLoader(
            orderbook_adapter=orderbook_adapter,
            price_adapter=price_adapter,
            featurizer=mock_featurizer
        )
        await loader.initialize()
        
        start_time = dt.datetime(2024, 1, 1, 9, 0, 0, tzinfo=dt.timezone.utc)
        end_time = dt.datetime(2024, 1, 1, 11, 0, 0, tzinfo=dt.timezone.utc)
        
        snapshots, quality_metrics = await loader.load_orderbook_data(
            token="BTC",
            start_time=start_time,
            end_time=end_time,
            validate_data=True
        )
        
        # Should only have 1 valid record
        assert len(snapshots) == 1
        assert quality_metrics.total_records == 3
        assert quality_metrics.valid_records == 1
        assert quality_metrics.invalid_records == 2
        assert quality_metrics.crossed_books == 1
        assert quality_metrics.negative_prices == 1
    
    @pytest.mark.asyncio
    async def test_load_and_featurize(self, mock_featurizer, mock_adapters, sample_orderbook_data):
        """Test the combined load and featurize method"""
        orderbook_adapter, price_adapter = mock_adapters
        
        mock_cursor = AsyncMock()
        mock_cursor.__aiter__ = AsyncMock(return_value=iter(sample_orderbook_data))
        orderbook_adapter.composite_order_book_collection.aggregate.return_value = mock_cursor
        
        loader = OrderBookDataLoader(
            orderbook_adapter=orderbook_adapter,
            price_adapter=price_adapter,
            featurizer=mock_featurizer
        )
        await loader.initialize()
        
        start_time = dt.datetime(2024, 1, 1, 9, 0, 0, tzinfo=dt.timezone.utc)
        end_time = dt.datetime(2024, 1, 1, 11, 0, 0, tzinfo=dt.timezone.utc)
        
        features, metadata = await loader.load_and_featurize(
            token="BTC",
            start_time=start_time,
            end_time=end_time,
            validate_data=True
        )
        
        assert features.shape == (3, 50)  # 3 snapshots, 50 features each
        assert 'quality_metrics' in metadata
        assert 'features' in metadata
        assert metadata['total_snapshots'] == 3
    
    def test_convert_to_snapshots(self, mock_featurizer, mock_adapters):
        """Test conversion of raw data to snapshots"""
        loader = OrderBookDataLoader(featurizer=mock_featurizer)
        
        raw_data = [
            {
                "timestamp": dt.datetime(2024, 1, 1, 10, 0, 0, tzinfo=dt.timezone.utc),
                "book": {
                    "bids": [(100.0, 1.0), (99.9, 2.0)],
                    "asks": [(100.1, 1.0), (100.2, 2.0)],
                    "mid_price": 100.05
                }
            }
        ]
        
        snapshots = loader._convert_to_snapshots(raw_data, "BTC")
        
        assert len(snapshots) == 1
        snapshot = snapshots[0]
        assert isinstance(snapshot, OrderBookSnapshot)
        assert len(snapshot.bids) == 2
        assert len(snapshot.asks) == 2
        assert snapshot.mid_price == 100.05
    
    def test_interpolate_snapshots(self, mock_featurizer, mock_adapters):
        """Test snapshot interpolation"""
        loader = OrderBookDataLoader(featurizer=mock_featurizer)
        
        prev_snapshot = OrderBookSnapshot(
            timestamp=1000.0,
            bids=[(100.0, 1.0), (99.9, 2.0)],
            asks=[(100.1, 1.0), (100.2, 2.0)],
            mid_price=100.05
        )
        
        next_snapshot = OrderBookSnapshot(
            timestamp=1002.0,
            bids=[(100.2, 1.0), (100.1, 2.0)],
            asks=[(100.3, 1.0), (100.4, 2.0)],
            mid_price=100.25
        )
        
        # Interpolate at midpoint
        interp_snapshot = loader._interpolate_snapshots(prev_snapshot, next_snapshot, 1001.0)
        
        assert interp_snapshot.timestamp == 1001.0
        assert interp_snapshot.mid_price == 100.15  # Midpoint
        assert len(interp_snapshot.bids) == 2
        assert len(interp_snapshot.asks) == 2
    
    def test_quality_report(self, mock_featurizer, mock_adapters):
        """Test quality report generation"""
        loader = OrderBookDataLoader(featurizer=mock_featurizer)
        
        # Set up some test metrics
        loader.quality_metrics = DataQualityMetrics(
            total_records=100,
            valid_records=95,
            invalid_records=5,
            gap_count=3,
            gap_duration_total=30.0,
            quality_score=85.0
        )
        
        loader.detected_gaps = [
            GapInfo(
                start_time=dt.datetime(2024, 1, 1, 10, 0, 0, tzinfo=dt.timezone.utc),
                end_time=dt.datetime(2024, 1, 1, 10, 0, 10, tzinfo=dt.timezone.utc),
                duration_seconds=10.0,
                expected_interval=1.0,
                gap_type="missing"
            )
        ]
        
        report = loader.get_quality_report()
        
        assert 'quality_metrics' in report
        assert 'gaps' in report
        assert 'summary' in report
        assert len(report['gaps']) == 1
        assert report['summary']['total_gaps'] == 1
        assert report['summary']['data_quality_acceptable']


# Test utility functions
def test_data_quality_metrics():
    """Test DataQualityMetrics dataclass"""
    metrics = DataQualityMetrics(
        total_records=100,
        valid_records=90,
        invalid_records=10,
        quality_score=90.0
    )
    
    assert metrics.total_records == 100
    assert metrics.valid_records == 90
    assert metrics.invalid_records == 10
    assert metrics.quality_score == 90.0


def test_gap_info():
    """Test GapInfo dataclass"""
    gap = GapInfo(
        start_time=dt.datetime(2024, 1, 1, 10, 0, 0, tzinfo=dt.timezone.utc),
        end_time=dt.datetime(2024, 1, 1, 10, 0, 10, tzinfo=dt.timezone.utc),
        duration_seconds=10.0,
        expected_interval=1.0,
        gap_type="missing"
    )
    
    assert gap.duration_seconds == 10.0
    assert gap.expected_interval == 1.0
    assert gap.gap_type == "missing"


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
