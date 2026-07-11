# Active Context: Batch Embedding Optimization & Candlestick Aggregation Fix

## Quick Reference
- **Feature**: Batch Embedding Optimization
- **Status**: Completed & Verified ✅

## Executive Summary
Optimized the Retrieval Service index bootstrapping process to prevent sequential HTTP request loops and resolved an underlying PostgreSQL candlestick aggregation boundary bug that caused all raw price ticks to be returned as separate candles. Re-built and verified the system containers on the remote host, completing startup within 2.5 minutes (down from infinite loops/timeout).

## Architecture Overview
1. **Batch Embed Endpoint**: Exposed `POST /embed/batch` in the Embed service to support vectorizing multiple price windows simultaneously via PyTorch.
2. **Bulk Indexing Pipeline**: Modified `build_index_for_combination` in the Retrieval service to accumulate window segments and execute indexing in batches of 1000.
3. **Time Boundary Aggregation**: Corrected `calc_second` and `calc_minute` functions in the database adapter to correctly group seconds/minutes on clean boundaries when granularity is $\ge 60$ seconds, preventing raw tick counts (375k+) from leaking into candlestick buckets.

## Key Files Modified
- [server.py](file:///home/miles/Development/notebooks/CryptoTrading/services/embed/server.py): Implemented the batch embedding endpoint.
- [encoder.py](file:///home/miles/Development/notebooks/CryptoTrading/services/retrieval/encoder.py): Added batch encoding/adding pipeline logic.
- [main.py](file:///home/miles/Development/notebooks/CryptoTrading/services/retrieval/main.py): Refactored index builder to batch requests.
- [price.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/data/price.py): Fixed aggregation time-boundary bug.
- [data.py](file:///home/miles/Development/notebooks/CryptoTrading/services/serve/data.py): Fixed boundary bug in serve data.

## Verification & Validation
- **Unit Tests**: All 49 tests passed.
- **Docker Deployment**: Rebuilt and restarted services on remote server `cloud@50.117.53.113`.
- **Uvicorn Startup**: The retrieval service initialized default indexes for BTC and ETH in less than 2.5 minutes and successfully started listening.
- **API Functional Check**: Queried `/candlestick/BTC` on serve and verified clean, correct aggregated candles returned instantly.
