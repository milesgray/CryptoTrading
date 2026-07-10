# Active Context: Optimize Candlestick & Price Queries to Prevent Timeouts

## Quick Reference
- **Feature**: Optimize Candlestick & Price Queries
- **Status**: Completed & Verified ✅

## Executive Summary
Resolved candlestick chart loading timeouts (`QueryCanceledError`) by recovering host disk space (freeing ~13.2 GB of space to boot up TimescaleDB) and rewriting database query interfaces to bypass slow JSONB metadata filtering and prefix-LIKE matches. Introduced a SkipScan-based `resolve_matching_symbols` helper to resolve tokens to symbol lists and query with parallel index scans (`symbol = ANY($1)`).

## Architecture Overview
Queries on the `price_data` table (9.5+ million rows) now run in under 200 ms. The SkipScan helper queries unique symbols in under 1 ms using TimescaleDB's metadata chunk indexes. All service endpoints (`serve`, `retrieval`, `analysis`) now execute highly efficient parallel B-tree index scans on `idx_price_data_symbol_time`.

## Tech Stack for This Feature
- **PostgreSQL + TimescaleDB**: Query optimizations and hypertable indexing
- **asyncpg**: Async database driver and connection pool management
- **Docker Compose**: Service container orchestration and system resource cleaning

## Key Files Created/Modified
- [postgres.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/data/postgres.py): Added `resolve_matching_symbols` SkipScan query helper.
- [price.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/data/price.py): Optimized `get_price_data`, `get_price_data_count`, `get_latest_price`, and `get_candlestick_data`.
- [book.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/data/book.py): Optimized `get_orderbook_data`, `get_latest_transformed_order_book_point`, and `get_transformed_order_book`.
- [price.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/analysis/price.py): Optimized raw exchange price analysis retrieval.
- [retrieval.py](file:///home/miles/Development/notebooks/CryptoTrading/services/serve/routers/retrieval.py): Optimized JEPA market regime calculation queries.

## Verification & Validation
- **Automated Tests**: Executed database query tests and pytest suite:
  - `PYTHONPATH=src uv run python test_db.py` -> **Passed**
  - `uv run pytest tests/` -> **26 Passed**
- **Manual Verification**: Curled the `candlestick` endpoint (`http://localhost:8362/candlestick/BTC`) and verified immediate returns (under ~200 ms) without timeouts.
