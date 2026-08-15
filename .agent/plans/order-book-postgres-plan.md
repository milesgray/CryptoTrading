# Implementation Plan: Save Full Order Book Data to Postgres

## Overview
Save granular full order book data (bids and asks per price level) into the `order_book_data` hypertable in Postgres, update `OrderBookRepository` with bulk insertion and snapshot querying capabilities, and update `OrderBookPostgresAdapter` to store full order book depth for both exchange order books and composite order books.

## Proposed Changes

### 1. `src/cryptotrading/data/postgres.py`
- Add compound index `idx_order_book_data_symbol_exchange_time` on `(symbol, exchange, time DESC)`.
- Enhance `OrderBookRepository`:
  - `store_order_book(symbol, exchange, bids, asks, time, metadata, conn)`: Bulk inserts order book levels into `order_book_data` using `executemany` with deduplicated `(is_bid, price)` keys and `ON CONFLICT` handling.
  - Refine `get_order_book_snapshot(symbol, exchange, time, depth, conn)`: Query with `time = (SELECT MAX(time) FROM order_book_data WHERE symbol = $1 AND exchange = $2 AND time <= $3)`.
  - Add `get_order_books(symbol, exchange, start_time, end_time, depth, conn)`: Time range queries.

### 2. `src/cryptotrading/data/book.py`
- Update `OrderBookPostgresAdapter`:
  - In `store_exchange_order_book`: Persist bids and asks to `order_book_data` table for each exchange book.
  - In `store_composite_order_book_data`: Persist bids and asks to `order_book_data` table under `exchange='composite'`.

### 3. `tests/test_order_book_postgres.py`
- Add comprehensive unit tests covering:
  - Repository bulk insert and snapshot query.
  - Adapter store exchange and composite order books.
  - Price level deduplication and conflict safety.

## Verification
- `PYTHONPATH=. pytest tests/test_order_book_postgres.py`
- `PYTHONPATH=. pytest tests/test_order_book_analytics.py tests/test_exchange.py`
