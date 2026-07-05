# Plan: Move pgvector Store to main library

## Overview
Move the pgvector store code to the main `cryptotrading` library and connect it to the shared PostgreSQL service connection pool.

## Proposed Changes

### Centralized Schema
- Modify [src/cryptotrading/data/postgres.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/data/postgres.py):
  - In `init_schema(conn)`, add creation of the `trade_setups` table (with vector dimension 128) and all of its indexes (including HNSW index).

### Core Data Module
- Create [NEW] [src/cryptotrading/data/pgvector_store.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/data/pgvector_store.py):
  - Implement `TradeEmbeddingDB` using the shared pool or connecting custom connection.
  - Implement `StoredTradeSetup`, `SimilarSetup`, and `TradeEmbeddingDBSync`.

### Forwarder Module
- Modify [services/embed/database/pgvector_store.py](file:///home/miles/Development/notebooks/CryptoTrading/services/embed/database/pgvector_store.py):
  - Forward all class imports to `cryptotrading.data.pgvector_store` to avoid breaking existing usages in `services/embed/server.py`.

## Verification Plan

### Automated Tests
- Check compilation.
- Rebuild/restart containers using `docker compose up -d --build embed`.
- Check logs and confirm successful initialization and connection.

### Manual Verification
- Check health endpoint: `curl http://localhost:8301/health`
