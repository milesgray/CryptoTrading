# Active Context: Poetry to uv Migration & Service Orchestration

## Quick Reference
- **Feature**: Switch Python Packaging and Dependencies to uv, and containerize the retrieval service
- **Branch**: `feature/uv-migration-and-retrieval`
- **Plan File**: `/home/miles/.gemini/antigravity-ide/brain/fef6a491-85b5-41dc-a91c-ef6e76e1d2fc/implementation_plan.md`
- **Status**: Completed & Verified ✅

## Executive Summary
Migrate the entire repository from Poetry to `uv` for python packaging, orchestrate services under Docker Compose, and fix frontend chart visualization crashes by overlaying rescaled/standardized retrieved forecast segments onto the AnyChart candlestick component.

## Key Accomplishments
- **PEP 621 Standard**: Converted all `pyproject.toml` files from `[tool.poetry]` configurations to standard PEP 621 `[project]` definitions with Hatchling build backends.
- **Lockfile Upgrades**: Removed legacy `poetry.lock` files and generated standard `uv.lock` files.
- **Docker builds optimized**: Configured all service Dockerfiles to install `uv` via fast binaries and use `uv sync --frozen --no-install-project` to speed up and cache container builds.
- **Added Compilation Tools**: Installed `build-essential` and `g++` in Python-slim containers to compile dependencies like `annoy` successfully.
- **Shell Scripts**: Updated `dev.sh`, `record.sh`, and `serve.sh` to run programs using `uv run` instead of `poetry run`.
- **Service Instantiation Fix**: Resolved a `TypeError` in `record` service by correctly passing the name argument (`'price_system_service'`) to the `StatusManager` constructor.
- **Async Event Loop Fix**: Resolved `RuntimeError: Task got Future attached to a different loop` by removing module import-time database auto-initialization in `postgres.py`.
- **Retrieval Routing & Serve Container Fix**: Fixed a `404` error on `/retrieval/forecast` by updating `Dockerfile.serve` to copy the new `services/serve` code, modifying the APIRouter prefix to `/retrieval`, and containerizing the `retrieval` microservice using [Dockerfile.retrieval](file:///home/miles/Development/notebooks/CryptoTrading/Dockerfile.retrieval) to run as part of the `docker-compose` network.
- **Dependency Upgrades**: Installed `scipy` using `uv` to support FFT calculations in pattern retrieval.
- **AnyChart split() Fix**: Resolved AnyChart `.split()` runtime crash by converting volume labels format from tokenized strings to custom JavaScript formatting callback functions.
- **Standardized Forecast Scaling**: Standardized retrieved futures data using Z-score mapping and rescaled it dynamically by the moving average of the last $N$ prices to align forecasting projection lines smoothly with the historical close price.
- **Component Cleanup**: Removed the redundant standalone `RetrievalVisualizer` component since overlay forecasting lines are fully integrated into the candlestick chart.

## Next Objectives
- Integrate `pgvector` HNSW queries into the REST API for Live Pattern Matching.
- Implement TimescaleDB historical compression policies.
- Run database performance and WebSocket latency load tests.
