# Active Context: Candlestick Volume Aggregation Fix

## Quick Reference
- **Feature**: Candlestick Volume Aggregation Fix
- **Status**: Completed & Verified ✅

## Executive Summary
Resolved a discrepancy between the volume calculation for historical candlestick data and live updates. The historical data endpoint previously returned the snapshot volume of only the final tick in each candle's time window, whereas the live WebSocket updates continuously sum the snapshot volumes of all ticks. The historical aggregation logic was modified to accumulate the snapshot volumes of all ticks falling into the candle's bucket, ensuring smooth and consistent volume levels.

## Key Files Modified
- [price.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/data/price.py): Updated `PriceMongoAdapter.get_candlestick_data` and `PricePostgresAdapter.get_candlestick_data` to sum tick order book volumes.
- [data.py](file:///home/miles/Development/notebooks/CryptoTrading/services/serve/data.py): Updated the MongoDB-specific `get_candlestick_data` serving fallback to also accumulate tick volumes.

## Verification & Validation
- **Unit Tests**: Ran 22 tests in `tests/test_exchange.py`, `tests/test_specretf.py`, `tests/test_orchestrator.py`, and `tests/test_order_book_analytics.py`; all passed successfully.
- **Docker Deployment**: Rebuilt and restarted the `serve` and `retrieval` services to load the updated code.
- **API Functional Check**: Queried `/candlestick/ETH` on the serve port `8362` and verified that historic candles now return a correctly aggregated volume (e.g. `30340.87`) representing the sum of all ticks, rather than just the final tick's volume (e.g. `883.93`).
