# Active Context: Poetry to uv Migration & Real Exchange Data Recording

## Quick Reference
- **Feature**: Switch Python Packaging and Dependencies to uv, and record/retrieve actual exchange prices
- **Branch**: `feature/uv-migration-and-retrieval`
- **Plan File**: `/home/miles/.gemini/antigravity-ide/brain/93870454-a7cb-43d9-a756-25c3ea989e4e/walkthrough.md`
- **Status**: Completed & Verified ✅

## Executive Summary
Migrate the entire repository from Poetry to `uv` for python packaging, orchestrate services under Docker Compose, and transition the pattern retrieval forecasting service from using simulated mock data to indexing and querying actual live prices recorded from exchanges.

## Key Accomplishments
- **PEP 621 Standard**: Converted all `pyproject.toml` files from `[tool.poetry]` configurations to standard PEP 621 `[project]` definitions with Hatchling build backends.
- **Lockfile Upgrades**: Removed legacy `poetry.lock` files and generated standard `uv.lock` files.
- **Docker builds optimized**: Configured all service Dockerfiles to install `uv` via fast binaries and use `uv sync --frozen --no-install-project` to speed up and cache container builds.
- **Sustained Retrieval Daemon**: Added a Uvicorn execution block in `services/retrieval/main.py` so the container runs persistently on port 8000.
- **Real Price Recording & Ingestion**: Configured `MIN_VALID_FEEDS=2` in `docker-compose.yml` to allow the `record` service to compute and record actual index prices even when some exchanges are rate-limited.
- **Real-time Logging Observability**: Added `PYTHONUNBUFFERED=1` to the python containers and added `logging.basicConfig(level=logging.INFO)` in the `record` service to enable real-time logs.
- **Successful Database Retrieval**: Bypassed simulated/mock price data fallbacks on startup. The retrieval service successfully connected to TimescaleDB, fetched 44,381 real historical price candles, indexed 44,322 sliding window segments, built the vector index, and successfully served live `/forecast` similarity queries.
- **AnyChart split() Fix**: Resolved AnyChart `.split()` runtime crash by converting volume labels format from tokenized strings to custom JavaScript formatting callback functions.
- **Standardized Forecast Scaling**: Standardized retrieved futures data using Z-score mapping and rescaled it dynamically by the moving average of the last $N$ prices to align forecasting projection lines smoothly with the historical close price.

## Next Objectives
- Integrate `pgvector` HNSW queries into the REST API for Live Pattern Matching.
- Implement TimescaleDB historical compression policies.
- Run database performance and WebSocket latency load tests.
