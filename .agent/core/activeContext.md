# Active Context: FastAPI Serve Service Router Split

## Quick Reference
- **Feature**: FastAPI Serve Service Router Split & Modularization
- **Branch**: `refactor/split-serve-routers`
- **Status**: Completed ✅

## Executive Summary
Refactored the monolithic API gateway server in `services/serve/app.py` into separate, modular `APIRouter` submodules under `services/serve/routers/`. Verified WebSocket connection sharing, proxy forwarding to the pattern retrieval service, subprocess orchestration endpoints, and validated all integrations via local unit tests and manual curl endpoints verification.

## Key Accomplishments
- **Modularized FastAPI Endpoints**: Extracted endpoints from `app.py` into logical groups:
  - `routers/market.py`: All price REST/WebSocket streams, order books, and candlestick endpoints.
  - `routers/retrieval.py`: Proxying forecasting queries to the retrieval service.
  - `routers/services.py`: Subprocess control REST routes and services status updates WebSocket.
- **Unified Connection Management**: Instantiated a global shared `websocket_manager = ConnectionManager()` in `services/serve/websocket.py` to prevent duplicate pools between the background change watchers and active WebSocket endpoints.
- **Decoupled Architecture**: Bound application dependencies (database connection handles and ingestion adapters) directly to the incoming request/websocket context via `request.app` / `websocket.app` properties.
- **Validation**: All 13 local unit tests passed successfully. The `serve` Docker container was rebuilt, started, and manually verified via health check and retrieval forecast proxies.

## Next Objectives
- Clean up unused docker images.
- Investigate persistent state storage for service configurations.
