"""
Example Usage of OrderBookDataLoader with OrderBookFeaturizer

This example demonstrates how to load order book data from MongoDB,
validate it for quality issues, handle gaps, and extract features.
"""

import asyncio
import datetime as dt
import logging
import numpy as np

from data_loader import OrderBookDataLoader, load_orderbook_features
from pressure_features import OrderBookFeaturizer, AdaptiveBucketCalculator
from oracle import PressureOracle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def basic_usage_example():
    """Basic example of loading and featurizing order book data"""
    
    print("=" * 60)
    print("BASIC USAGE EXAMPLE")
    print("=" * 60)
    
    # Initialize components
    featurizer = OrderBookFeaturizer(
        bucket_calculator=AdaptiveBucketCalculator('major_crypto'),
        normalize_features=True
    )
    
    loader = OrderBookDataLoader(featurizer=featurizer)
    await loader.initialize()
    
    # Define time range (last 24 hours)
    end_time = dt.datetime.now(dt.timezone.utc)
    start_time = end_time - dt.timedelta(hours=24)
    
    # Load and featurize data
    try:
        features, metadata = await loader.load_and_featurize(
            token="BTC",
            start_time=start_time,
            end_time=end_time,
            validate_data=True,
            fill_gaps=True,
            max_records=1000
        )
        
        print(f"Loaded {features.shape[0]} snapshots")
        print(f"Feature dimension: {features.shape[1]}")
        print(f"Data quality score: {metadata['quality_metrics']['quality_score']:.1f}/100")
        print(f"Total records: {metadata['quality_metrics']['total_records']}")
        print(f"Valid records: {metadata['quality_metrics']['valid_records']}")
        print(f"Gaps detected: {metadata['quality_metrics']['gap_count']}")
        
        # Display feature information
        if 'features' in metadata:
            print("\nFeature breakdown:")
            for name, size in metadata['features']['feature_sizes'].items():
                print(f"  {name}: {size}")
        
        return features, metadata
        
    except Exception as e:
        logger.error(f"Error in basic usage: {e}")
        return None, None


async def advanced_validation_example():
    """Advanced example with detailed validation and gap analysis"""
    
    print("\n" + "=" * 60)
    print("ADVANCED VALIDATION EXAMPLE")
    print("=" * 60)
    
    # Initialize with strict validation
    bucket_calc = AdaptiveBucketCalculator('major_crypto')
    featurizer = OrderBookFeaturizer(
        bucket_calculator=bucket_calc,
        normalize_features=True,
        outlier_clip=3.0  # More strict outlier clipping
    )
    
    loader = OrderBookDataLoader(
        featurizer=featurizer,
        expected_interval_seconds=1.0,
        max_gap_threshold=60.0,  # 1 minute max gap
        quality_threshold=80.0   # Higher quality requirement
    )
    await loader.initialize()
    
    # Load data with comprehensive validation
    end_time = dt.datetime.now(dt.timezone.utc)
    start_time = end_time - dt.timedelta(hours=12)  # Last 12 hours
    
    try:
        snapshots, quality_metrics = await loader.load_orderbook_data(
            token="BTC",
            start_time=start_time,
            end_time=end_time,
            validate_data=True,
            fill_gaps=False  # Don't fill gaps for analysis
        )
        
        # Get detailed quality report
        quality_report = loader.get_quality_report()
        
        print(f"Quality Analysis:")
        print(f"  Total records: {quality_metrics.total_records}")
        print(f"  Valid records: {quality_metrics.valid_records}")
        print(f"  Invalid records: {quality_metrics.invalid_records}")
        print(f"  Quality score: {quality_metrics.quality_score:.1f}/100")
        print(f"  Data completeness: {quality_metrics.data_completeness:.1f}%")
        print(f"  Crossed books: {quality_metrics.crossed_books}")
        print(f"  Negative prices: {quality_metrics.negative_prices}")
        print(f"  Zero volumes: {quality_metrics.zero_volumes}")
        
        print(f"\nGap Analysis:")
        print(f"  Total gaps: {quality_metrics.gap_count}")
        print(f"  Total gap duration: {quality_metrics.gap_duration_total:.1f}s")
        print(f"  Average gap duration: {quality_metrics.avg_gap_duration:.1f}s")
        print(f"  Maximum gap duration: {quality_metrics.max_gap_duration:.1f}s")
        
        # Show individual gaps
        if quality_report['gaps']:
            print(f"\nIndividual Gaps (showing first 5):")
            for i, gap in enumerate(quality_report['gaps'][:5]):
                print(f"  Gap {i+1}: {gap['duration_seconds']:.1f}s ({gap['gap_type']})")
                print(f"    From: {gap['start_time']}")
                print(f"    To: {gap['end_time']}")
        
        # Extract features from validated data
        if snapshots:
            print(f"\nExtracting features from {len(snapshots)} validated snapshots...")
            
            # Fit normalizer on the data
            all_features = []
            for snapshot in snapshots:
                features = featurizer.extract_features(snapshot, "BTC", validate=False)
                flat_features = featurizer.flatten_features(features)
                all_features.append(flat_features)
            
            feature_array = np.array(all_features)
            
            # Fit normalizer
            featurizer.fit_normalizer(feature_array)
            
            # Apply normalization
            normalized_features = featurizer.normalize(feature_array)
            
            print(f"Feature extraction complete:")
            print(f"  Shape: {normalized_features.shape}")
            print(f"  Mean (first 5): {normalized_features.mean(axis=0)[:5]}")
            print(f"  Std (first 5): {normalized_features.std(axis=0)[:5]}")
            
            return normalized_features, quality_report
        
        return None, quality_report
        
    except Exception as e:
        logger.error(f"Error in advanced validation: {e}")
        return None, None


async def gap_filling_example():
    """Example demonstrating gap filling strategies"""
    
    print("\n" + "=" * 60)
    print("GAP FILLING EXAMPLE")
    print("=" * 60)
    
    loader = OrderBookDataLoader()
    await loader.initialize()
    
    # Load data with gap filling enabled
    end_time = dt.datetime.now(dt.timezone.utc)
    start_time = end_time - dt.timedelta(hours=6)
    
    print("Loading data without gap filling...")
    snapshots_no_fill, metrics_no_fill = await loader.load_orderbook_data(
        token="BTC",
        start_time=start_time,
        end_time=end_time,
        validate_data=True,
        fill_gaps=False
    )
    
    print(f"Without gap filling: {len(snapshots_no_fill)} snapshots")
    print(f"Gaps detected: {metrics_no_fill.gap_count}")
    
    print("\nLoading data with gap filling...")
    snapshots_with_fill, metrics_with_fill = await loader.load_orderbook_data(
        token="BTC",
        start_time=start_time,
        end_time=end_time,
        validate_data=True,
        fill_gaps=True
    )
    
    print(f"With gap filling: {len(snapshots_with_fill)} snapshots")
    print(f"Gaps filled: {len(snapshots_with_fill) - len(snapshots_no_fill)}")
    
    # Compare time coverage
    if snapshots_no_fill and snapshots_with_fill:
        time_span_no_fill = snapshots_no_fill[-1].timestamp - snapshots_no_fill[0].timestamp
        time_span_with_fill = snapshots_with_fill[-1].timestamp - snapshots_with_fill[0].timestamp
        
        print(f"\nTime coverage comparison:")
        print(f"  Without filling: {time_span_no_fill:.1f}s")
        print(f"  With filling: {time_span_with_fill:.1f}s")
        print(f"  Improvement: {((time_span_with_fill / time_span_no_fill - 1) * 100):.1f}%")


async def multiple_token_example():
    """Example loading data for multiple tokens"""
    
    print("\n" + "=" * 60)
    print("MULTIPLE TOKEN EXAMPLE")
    print("=" * 60)
    
    tokens = ["BTC", "ETH"]
    end_time = dt.datetime.now(dt.timezone.utc)
    start_time = end_time - dt.timedelta(hours=4)
    
    results = {}
    
    for token in tokens:
        print(f"\nProcessing {token}...")
        
        try:
            # Use convenience function
            features, metadata = await load_orderbook_features(
                token=token,
                start_time=start_time,
                end_time=end_time,
                validate_data=True,
                fill_gaps=True,
                max_records=500
            )
            
            results[token] = {
                'features': features,
                'metadata': metadata
            }
            
            print(f"  {token}: {features.shape[0]} snapshots, "
                  f"quality: {metadata['quality_metrics']['quality_score']:.1f}/100")
            
        except Exception as e:
            logger.error(f"Error processing {token}: {e}")
            results[token] = None
    
    # Compare tokens
    print(f"\nToken Comparison:")
    for token, result in results.items():
        if result is not None:
            features = result['features']
            metadata = result['metadata']
            
            print(f"  {token}:")
            print(f"    Snapshots: {features.shape[0]}")
            print(f"    Features: {features.shape[1]}")
            print(f"    Quality: {metadata['quality_metrics']['quality_score']:.1f}/100")
            print(f"    Gaps: {metadata['quality_metrics']['gap_count']}")


async def integration_with_oracle_example():
    """Example integrating with PressureOracle for label generation"""
    
    print("\n" + "=" * 60)
    print("INTEGRATION WITH PRESSURE ORACLE")
    print("=" * 60)
    
    # Initialize components
    featurizer = OrderBookFeaturizer()
    oracle = PressureOracle(lookahead_candles=5)
    loader = OrderBookDataLoader(featurizer=featurizer)
    
    await loader.initialize()
    
    # Load recent data
    end_time = dt.datetime.now(dt.timezone.utc)
    start_time = end_time - dt.timedelta(hours=2)
    
    try:
        # Load order book data
        snapshots, quality_metrics = await loader.load_orderbook_data(
            token="BTC",
            start_time=start_time,
            end_time=end_time,
            validate_data=True,
            fill_gaps=True
        )
        
        if len(snapshots) < 10:
            print("Not enough data for oracle integration")
            return
        
        print(f"Loaded {len(snapshots)} snapshots")
        
        # Generate price history for oracle
        price_history = [snapshot.mid_price for snapshot in snapshots]
        
        # Generate pressure labels
        labels = []
        for i, snapshot in enumerate(snapshots[:-5]):  # Leave room for lookahead
            future_prices = price_history[i+1:i+6]  # Next 5 prices
            current_history = price_history[:i+1]
            
            try:
                label = oracle.compute_pressure_labels(
                    snapshot,
                    future_prices,
                    current_history,
                    i
                )
                labels.append(label)
                
                if i % 100 == 0:  # Print progress
                    print(f"  Processed {i}/{len(snapshots)-5} snapshots")
                    
            except Exception as e:
                logger.warning(f"Error generating label for snapshot {i}: {e}")
                labels.append(None)
        
        # Filter valid labels
        valid_labels = [label for label in labels if label is not None]
        
        print(f"\nLabel Generation Results:")
        print(f"  Total snapshots: {len(snapshots)}")
        print(f"  Valid labels: {len(valid_labels)}")
        print(f"  Success rate: {len(valid_labels)/len(snapshots)*100:.1f}%")
        
        if valid_labels:
            # Analyze pressure distribution
            buy_pressures = [label.buy_pressure for label in valid_labels]
            sell_pressures = [label.sell_pressure for label in valid_labels]
            total_pressures = [label.total_pressure for label in valid_labels]
            
            print(f"\nPressure Statistics:")
            print(f"  Buy pressure - Mean: {np.mean(buy_pressures):.3f}, Std: {np.std(buy_pressures):.3f}")
            print(f"  Sell pressure - Mean: {np.mean(sell_pressures):.3f}, Std: {np.std(sell_pressures):.3f}")
            print(f"  Total pressure - Mean: {np.mean(total_pressures):.3f}, Std: {np.std(total_pressures):.3f}")
            
            # Regime distribution
            regimes = [label.market_regime for label in valid_labels]
            regime_counts = {}
            for regime in regimes:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            print(f"\nMarket Regime Distribution:")
            for regime, count in regime_counts.items():
                print(f"  {regime}: {count} ({count/len(valid_labels)*100:.1f}%)")
        
    except Exception as e:
        logger.error(f"Error in oracle integration: {e}")


async def main():
    """Run all examples"""
    print("OrderBook Data Loading Examples")
    print("=" * 60)
    
    try:
        # Basic usage
        await basic_usage_example()
        
        # Advanced validation
        await advanced_validation_example()
        
        # Gap filling
        await gap_filling_example()
        
        # Multiple tokens
        await multiple_token_example()
        
        # Oracle integration
        await integration_with_oracle_example()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")


if __name__ == "__main__":
    asyncio.run(main())
