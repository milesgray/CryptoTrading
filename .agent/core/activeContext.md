# Active Context: Candlestick Query Timeout Resolution via Chunked Retrieval

## Quick Reference
- **Feature**: Candlestick Database Query Chunking & Timeout Resolution
- **Status**: Completed & Verified ✅

## Executive Summary
Resolved the statement timeout error on the `/candlestick` endpoint when querying historical price ticks from PostgreSQL/TimescaleDB. The adapter now queries the hypertable in 1-day temporal chunks, and incrementally groups the results into candlestick objects. This prevents single database statements from exceeding the 30-second connection timeout and reduces the peak RAM load in Python.

## Architecture Overview
1. **Query Performance Issue**: Even with index tuning, requesting candlestick data over long ranges (e.g. 7 days) fetched massive numbers of raw tick records in a single database query, causing connection statement timeouts.
2. **Temporal Chunking**: Implemented 1-day temporal sliding window queries in `PricePostgresAdapter.get_candlestick_data`. The query upper-bound utilizes exclusive range filters (`time < current_end`) for non-final chunks to prevent boundary duplicate processing.
3. **Incremental Memory Aggregation**: Rows are aggregated into the candlestick map chunk-by-chunk, allowing raw rows from previous chunks to be garbage collected immediately, lowering peak RAM usage.

## Tech Stack
- **PostgreSQL / TimescaleDB**: Hypertable storage
- **asyncpg**: Async connection pooling and execution

## Key Files Modified
- [price.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/data/price.py): Updated `get_candlestick_data` to loop through dates in 1-day chunks and aggregate.

## Verification & Validation
- **Unit Tests**: Ran `./src/.venv/bin/pytest tests/` successfully (all 19 tests passed).
- **Manual Verification**: Performed a `/candlestick` request using `curl`. It completed successfully in `0.22 seconds` and logged chunked retrieval execution accurately.
