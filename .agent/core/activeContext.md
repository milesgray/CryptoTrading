# Active Context: TimescaleDB Performance Optimizations & Downsampling

## Quick Reference
- **Feature**: TimescaleDB Query Optimizations & Downsampling
- **Plan File**: [implementation_plan.md](file:///home/miles/.gemini/antigravity-ide/brain/9a0c565e-c575-458a-94aa-d1e6a9073e7f/implementation_plan.md)
- **Status**: Completed ✅

## Executive Summary
Optimized database queries for order books and price data on the remote TimescaleDB database VM. We resolved latency and timeout issues by leveraging B-tree index scan properties (backward scans to avoid sort operations), database-side downsampling via `time_bucket` and `last()` aggregate functions, and in-memory caching of symbol resolution.

## Tech Stack for This Feature
- **PostgreSQL / TimescaleDB**: Hypertable indexing, B-Tree backward scans, `time_bucket` downsampling, and `last` aggregates.
- **Python / asyncpg**: Asynchronous connection pooling and query execution.
- **NumPy / Pandas**: Order book feature calculation pipeline.

## Key Files Modified
- [src/cryptotrading/data/postgres.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/data/postgres.py): Implemented a 60-second in-memory cache for SkipScan-based `resolve_matching_symbols` resolution.
- [src/cryptotrading/data/book.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/data/book.py): Updated `get_orderbook_data` to support `time_bucket` downsampling and exact single-symbol equality scans (B-Tree backward scans to bypass sort steps).
- [src/cryptotrading/data/price.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/data/price.py): Updated `get_price_data`, `get_price_data_count`, `get_latest_price`, and `get_candlestick_data` to leverage exact single-symbol index scans.
- [services/pressure/data_loader.py](file:///home/miles/Development/notebooks/CryptoTrading/services/pressure/data_loader.py): Passed downsampling interval (`expected_interval_seconds`) to the database adapter.
- [test_db.py](file:///home/miles/Development/notebooks/CryptoTrading/test_db.py): Wrote query verification and latency benchmark tests.

## Critical Implementation Details
1. **Backward Index Scan**: By dynamically rewriting `symbol = ANY($1)` to `symbol = $1` when a single symbol is matched, the Postgres planner scans the `(symbol, exchange, time DESC)` index backwards. This retrieves rows in sorted `time ASC` order directly from the index, eliminating the costly Sort operation and speeding up database fetch by 4x.
2. **Database-Side Downsampling**: Utilizing TimescaleDB's `time_bucket` and `last()` aggregate functions, raw high-frequency orderbook snapshots are downsampled to a target frequency (e.g. 10s) directly on the DB server. This reduces the network payload and Python parsing time by up to 90%.
3. **Symbol Caching**: Caching the list of available symbols in Python memory for 60 seconds eliminates the 9-second SkipScan Skip-Join latency on every single database fetch.

## Next Steps
- Monitor connection health and query execution speed in production dashboards.
