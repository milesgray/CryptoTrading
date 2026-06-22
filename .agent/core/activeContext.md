# Active Context: Postgres with TimescaleDB Migration

## Quick Reference
- **Feature**: Postgres with TimescaleDB Migration
- **Branch**: `feature/postgres-timescaledb-migration`
- **Plan File**: `.agent/plans/postgres-migration-plan.md`
- **Status**: Planning Complete - Pending Approval

## Executive Summary
Migrate the primary backend of the cryptocurrency trading application to PostgreSQL with the TimescaleDB extension. MongoDB will remain supported as an alternative backend, but Postgres will be the new default going forward.

## Architecture Overview
A database backend selector (`DB_BACKEND`) in the config layer chooses the active database adapters (Mongo or Postgres/TimescaleDB) at runtime via a new database adapter factory.

## Tech Stack for This Feature
- **PostgreSQL**: Primary SQL Database.
- **TimescaleDB**: Time-series extension for hypertable structures and compression policies.
- **asyncpg**: Async Python client for Postgres.

## Key Files to Create/Modify
- `src/cryptotrading/config.py`: Add `DB_BACKEND` support.
- `src/cryptotrading/data/factory.py`: [NEW] Unified database adapter factory.
- `src/cryptotrading/data/postgres.py`: Add full-featured adapters matching MongoDB APIs.
- `services/price/service.py`: Adapt to database factory.
- `src/cryptotrading/rollbit/prices/serve/app.py` & `data.py`: Update to use Postgres database adapters.
- `src/cryptotrading/sentiment/analyzer.py`: Refactor tweet persistence to support Postgres.

## Acceptance Criteria
- [ ] Complete database factory interface implemented.
- [ ] PostgreSQL adapters implement all price/orderbook/sentiment storage and query APIs.
- [ ] Serve application runs successfully on top of PostgreSQL/TimescaleDB.
- [ ] Integration tests pass for both database backends.

## Next Prompt
Read `.agent/plans/postgres-migration-plan.md` for detailed implementation plan.
Then proceed with Phase 1 of the implementation plan after user approval.
