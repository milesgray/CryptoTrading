# Project Memory

## Last Completed Task
Created standardized service client package in `cryptotrading.client` for all microservices (`artifact`, `embed`, `jepa`, `predict`, `pressure`, `price`, `retrieval`, `sentiment`, `serve`, `trade`, `train`) and standardized FastAPI server definitions (`server.py`) across all microservice directories.

## Architecture Notes
- All service clients are cleanly exported from `cryptotrading.client.*` (e.g. `EmbedServiceClient`, `RetrievalServiceClient`).
- Every microservice standardizes on a `server.py` file exposing FastAPI `app` and invoking `uvicorn.run(app, host="0.0.0.0", port=port)` reading from `os.getenv("PORT", <default_port>)`.
- Inter-service calls and Docker entrypoints updated to target `server.py`.

## Environment / Config
- `PORT`: Standard environment variable used across all service `server.py` entrypoints to configure FastAPI / Uvicorn server ports.
- Service URLs (`EMBED_SERVICE_URL`, `RETRIEVAL_SERVICE_URL`, `PREDICT_SERVICE_URL`, etc.) are supported as defaults in client wrappers.

## Dependencies Added
- None.

## Known Blockers / Next Steps
- None.

