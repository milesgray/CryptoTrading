# Plan: Retrieval Service Enhancement & Embed Service Integration

## Goal
Enhance the retrieval service to:
1. Bootstraps 7 days of 1-minute historical candlestick data via CCXT on startup if not already done, inserting into Postgres `price_data` under `exchange = 'index'` with `ON CONFLICT DO NOTHING`.
2. Connect to the embed service (`http://embed:8301/embed`) to fetch 128D embeddings for price segments rather than using local handcrafted features (56D).
3. Remove all mock price/consensuses fallbacks on startup or when forecast has empty matching results. Raise clean HTTP/Value errors.
4. Update the React frontend dashboard to display clean error/warning panels if `/forecast` returns an error status.

## Proposed Components
1. **CCXT Bootstrap (startup_event)**:
   - Queries Postgres to check if any rows exist for `symbol = 'BTC/USDT'` (or config symbols) in the past 7 days.
   - If not present, calls CCXT (`ccxt.binance()` or fallback) to download 1-minute candles.
   - Batch inserts rows into `price_data` using `ON CONFLICT (time, symbol, exchange) DO NOTHING`.
2. **Embed Service Connection**:
   - Updates `encoder.py` to make POST requests to `/embed` with the price window arrays.
   - Updates Annoy/vector index dimension to 128.
3. **Mock Data Removal**:
   - Removes random segment initialization on DB read failures.
   - Removes fallback query logic on query segment generation.
4. **React Frontend Error Alerts**:
   - Catches forecast load errors and renders an error state in the dashboard.
