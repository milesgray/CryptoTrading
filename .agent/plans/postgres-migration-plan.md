# Implementation Plan: Postgres with TimescaleDB Migration

Migrate the primary backend of the cryptocurrency trading application to PostgreSQL with the TimescaleDB extension. MongoDB will remain supported as an alternative backend, but Postgres will be the new default going forward.

## User Review Required

> [!IMPORTANT]
> The default database backend will switch from MongoDB to PostgreSQL/TimescaleDB. Deployments will need a running PostgreSQL instance with TimescaleDB and pgvector extensions enabled.
> 
> A new environment variable `DB_BACKEND` will be introduced to select the backend:
> - `DB_BACKEND=postgres` (default)
> - `DB_BACKEND=mongodb` (for backward compatibility)

## Open Questions

None at this time. The architecture allows support for both backends dynamically based on configuration.

## Proposed Changes

### Configuration Layer

#### [MODIFY] [config.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/config.py)
- Introduce `DB_BACKEND` environment variable (default: `"postgres"`).
- Clean up configuration to make MongoDB parameters optional or secondary.

### Database Adapters

#### [MODIFY] [postgres.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/data/postgres.py)
- Enhance existing PostgreSQL repositories to align API signatures with `PriceMongoAdapter`, `OrderBookMongoAdapter`, and `TwitterMongoAdapter`.
- Provide methods like `store_price_data`, `get_candlestick_data`, `store_exchange_order_book`, `store_composite_order_book_data`, `store_transformed_order_book_data`, and `get_orderbook_data`.
- Implement `init_db()` and `close_db()`.

#### [NEW] [factory.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/data/factory.py)
- Create a factory interface to retrieve the active database adapters (Price, OrderBook, Twitter/Sentiment) depending on the `DB_BACKEND` environment variable.
- This isolates the rest of the application from backend changes.

### Services and Scripts

#### [MODIFY] [service.py](file:///home/miles/Development/notebooks/CryptoTrading/services/price/service.py)
- Update to use the unified database adapter factory instead of importing `PriceMongoAdapter` and `OrderBookMongoAdapter` directly.

#### [MODIFY] [app.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/rollbit/prices/serve/app.py)
- Update startup events and background polling loops to support PostgreSQL.
- Specifically replace MongoDB Change Streams/polling with Postgres async queries when `DB_BACKEND` is set to `postgres`.

#### [MODIFY] [data.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/rollbit/prices/serve/data.py)
- Make sure serve-specific data access helpers use the unified factory/adapters to work transparently with either database backend.

#### [MODIFY] [analyzer.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/sentiment/analyzer.py)
- Refactor the Twitter sentiment analyzer to save tweets using the factory database adapter, allowing Postgres support.

#### [MODIFY] [docker-compose.yml](file:///home/miles/Development/notebooks/CryptoTrading/docker-compose.yml)
- Update compose file to run a TimescaleDB PostgreSQL container as the primary database, alongside the application services.

## Verification Plan

### Automated Tests
- Implement unit and integration tests comparing query outputs and insertions between Mongo and Postgres adapters.
- Run tests via `poetry run pytest`.

### Manual Verification
- Start the application using `docker-compose up` with Postgres/TimescaleDB enabled.
- Verify streaming prices are logged to the PostgreSQL container and displayed in the frontend dashboard.
