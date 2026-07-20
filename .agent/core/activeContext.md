# Active Context: TimescaleDB Candlestick and Order Book Optimizations

## Quick Reference
- **Feature**: TimescaleDB Candlestick and Order Book Query Optimizations
- **Status**: Completed ✅

## Executive Summary
Optimized candlestick and order book query paths to run entirely inside the database using TimescaleDB continuous aggregates and advanced JSONB query slicing. Built six dedicated continuous aggregate views on the `price_data` hypertable (`price_candle_1s`, `price_candle_15s`, `price_candle_30s`, `price_candle_1m`, `price_candle_5m`, `price_candle_1d`) along with matching scheduled refresh policies. Rewrote the candlestick data adapter to pull from these views or fall back to database-side `time_bucket` aggregations. Implemented SQL-side order book JSONB array slicing using PostgreSQL 15's native `jsonb_path_query_array` to dramatically reduce network load and python memory overhead, and added a lightweight `get_orderbook_summary` endpoint for statistical metrics.

## Tech Stack for This Feature
- **TimescaleDB / PostgreSQL**: Hypertables, continuous aggregates, refresh policies, SkipScan, and `jsonb_path_query_array` operators.
- **Python (asyncpg / pydantic)**: Asynchronous database connection pooling and type-safe data schemas.

## Key Files Modified
- [src/cryptotrading/data/postgres.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/data/postgres.py): Defined six continuous aggregates and their refresh policies on the `price_data` hypertable.
- [src/cryptotrading/data/price.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/data/price.py): Updated `get_candlestick_data` to map granularities to continuous aggregate views or fall back to database-side SQL `time_bucket` aggregates.
- [src/cryptotrading/data/book.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/data/book.py): Optimized `get_orderbook_data` to slice the nested bids and asks JSONB arrays inside SQL using `jsonb_path_query_array`, and added `get_orderbook_summary` for high-speed stats access.
- [src/cryptotrading/data/models.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/data/models.py): Added `PriceLevel` schema models.

## Verification Details
- Verified database schema initialization, SkipScan routing, and continuous aggregate queries via `python3 test_db.py`.
