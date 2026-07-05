# Active Context: Resolve Retrieval-Embed Service Connection & Auto-populate DB

## Quick Reference
- **Feature**: Resolve Retrieval-Embed Service Connection & Auto-populate DB
- **Branch**: `fix/retrieval-embed-connection`
- **Status**: Completed & Verified ✅

## Executive Summary
Resolved connection failures between the `retrieval` (and `serve`) services and the `embed` service by configuring the correct `EMBED_SERVICE_URL` in the docker compose files. Added an asynchronous background task `auto_populate_db` in `services/embed/server.py` to auto-populate the pgvector database at startup by pulling, DP-labeling, and embedding historical price data when the store is empty.

## Architecture Overview
The retrieval and serve services resolve the embed service container host correctly. The embed service implements a self-bootstrapping background task at startup that ensures the vector database is fully populated with trade setup embeddings if no setups are present.

## Tech Stack for This Feature
- **FastAPI**: Endpoint handler & lifespan events
- **Docker Compose**: Container orchestration and network resolution
- **PostgreSQL + pgvector**: Vector and metadata store
- **asyncpg**: Async database connector

## Key Files Created/Modified
- [docker-compose.yml](file:///home/miles/Development/notebooks/CryptoTrading/docker-compose.yml): Configured `EMBED_SERVICE_URL` for `retrieval` and `serve`.
- [docker-compose-full.yml](file:///home/miles/Development/notebooks/CryptoTrading/docker-compose-full.yml): Configured `EMBED_SERVICE_URL` for `retrieval` and `serve`.
- [services/embed/server.py](file:///home/miles/Development/notebooks/CryptoTrading/services/embed/server.py): Added the `auto_populate_db` background task at startup.

## Verification & Validation
- Verified using `docker compose up -d --build` to compile Docker images and spin up all services.
- Checked container logs to ensure `retrieval` is connecting to `embed:8301` without connection errors.
- Verified uvicorn logs in `embed` showing the database auto-population task successfully pulling price data and starting embedding extraction.
- Executed unit tests:
  - `uv run pytest tests/...` -> **38 Passed**
