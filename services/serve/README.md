# API Serving Gateway

This service serves as the core REST and WebSocket gateway for the CryptoTrading dashboard. It connects to the shared database backend (TimescaleDB/PostgreSQL or MongoDB) to expose market data and coordinates other microservices.

## Architecture

The service has been split into modular FastAPI routers:

* **Entrypoint (`app.py`)**: Runs global middlewares, connects database adapters, polls DB changes, and includes all routers.
* **Market Router (`routers/market.py`)**: REST & WebSockets for price feeds, order books, and candlestick charts.
* **Retrieval Router (`routers/retrieval.py`)**: Proxies prediction queries to the pattern retrieval service.
* **Services Router (`routers/services.py`)**: Manages starting, stopping, and viewing logs for local background services.

## Sub-Modules
- `models.py`: Pydantic data schemas.
- `websocket.py`: Shared WebSocket manager singleton.
- `data.py`: Order book aggregation utilities.
