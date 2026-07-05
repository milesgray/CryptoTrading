# Plan: Polish Embed Service - Pull from Postgres

## Overview
Polish the contrastive trade setup embedding service (`services/embed/server.py`) to transition storage and similarity queries from Numpy files to a PostgreSQL database powered by `pgvector`.

## Proposed Changes

### Database Layer
- Modify [services/embed/database/pgvector_store.py](file:///home/miles/Development/notebooks/CryptoTrading/services/embed/database/pgvector_store.py) to support native DSN parameters during initialization, so that the raw PostgreSQL connection string can be supplied directly.

### Server Layer
- Modify [services/embed/server.py](file:///home/miles/Development/notebooks/CryptoTrading/services/embed/server.py):
  - On startup: check environment variables. If `DB_BACKEND` is set to `postgres`, initialize `TradeEmbeddingDB` with the connection string from `POSTGRES_URI`. Otherwise, initialize the fallback `NumpyVectorStore`.
  - In `/search`, `/setup/{id}`, `/stats`, and `/health` endpoints: await async calls to `TradeEmbeddingDB` if initialized, else run sync calls to the numpy store.
  - In WebSocket endpoint: search historical setups using async `TradeEmbeddingDB` queries.
  - On shutdown: close connection pool or save numpy vector store.

## Verification Plan

### Automated Tests
- Run simple connection/query script or verification commands using the environment parameters.
- Verify container logs of `crypto-trading-timescaledb` and check if pgdata tables and indices are loaded.

### Manual Verification
- Spin up the docker containers using `docker-compose.yml`.
- Verify embedding REST endpoints and websocket behavior using mock price vectors.
