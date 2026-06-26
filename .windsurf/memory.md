# Project Memory

## Last Completed Task
Fixed bid/ask validation logic bugs in CCXT feed ingestion, documented composite pricing index formulas mathematically, and created a multi-exchange order book analytics module (`metrics.py`) with full test coverage (`test_order_book_analytics.py`).

## Architecture Notes
- Added `cryptotrading/rollbit/prices/metrics.py` to isolate cross-exchange meta-statistics from core orchestrators.
- Corrected logic in `validate_feeds` in `book.py` and eliminated large dead-code DataFrame conversion overhead.

## Environment / Config
- No new environment variables introduced.

## Dependencies Added
- None.

## Known Blockers / Next Steps
- CCXT REST polling runs at 500ms intervals which can trigger rate limits; transitioning to WebSockets is recommended for production.
- Arbitrage metric does not yet deduct trading fees or slippage.
- Depth HHI metrics should be narrowed to near-mid threshold (e.g. 1%) in the future to avoid spoofing skews.
