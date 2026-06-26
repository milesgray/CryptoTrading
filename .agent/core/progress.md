# Progress: CryptoTrading

## Project Timeline

### Phase 1: Foundation & Data Streaming (Completed ✅)
- [x] uv environment configuration and dependencies (`ccxt`, `motor`, `pymongo`, `fastapi`, `torch`).
- [x] Rollbit Price System composite index price formulation (500ms intervals, multi-exchange parsing, outlier removal, order size caps, exponential weightings).
- [x] Ingestion and recording service (`services/price`) running with graceful shutdown signals.
- [x] FastAPI REST/WebSocket serving application (`services/serve`) supporting:
  - WebSocket connections (`/ws/price/{token}`, `/ws/order_book/{token}`).
  - REST query handlers (`/candlestick/{token}`, `/historic/price/{token}`, `/health`).
- [x] Docker setup (`Dockerfile.serve`, `Dockerfile.record`, `/frontend/Dockerfile`, `docker-compose.yml`).

### Phase 2: Representation Learning & AI Models (Completed ✅)
- [x] **Trade Setup Pattern Matcher** (`services/embed`): CNN-based SupCon encoder mapping log return price windows to 128-dimensional vectors.
- [x] **Koopman-JEPA Regime Classifier** (`services/jepa`): Self-supervised Joint-Embedding Predictive Architecture utilizing Koopman invariants for unsupervised market regime identification.
- [x] **Twitter Sentiment Analyzer** (`services/sentiment`): Containerized stream ingestion service scoring tickers using VADER.
- [x] **TimesNet/Autoformer Trainer** (`services/train`): Configurable forecasting and movement classification trainer in PyTorch.

### Phase 3: Frontend Dashboard & Visualization (Completed ✅)
- [x] Vite React frontend layout (`frontend/src/App.jsx`).
- [x] Lightweight Charts implementation for Candlestick visualization.
- [x] Real-time order book panel listening to WebSocket streams.

### Phase 4: Database Integration & Production Simulation (Completed ✅)
- [x] Migrate primary database backend from MongoDB to PostgreSQL with TimescaleDB (Postgres as new default, Mongo as option).
- [x] Connect the FastAPI server directly to the PostgreSQL + pgvector setups library for live pattern matching queries.
- [x] Implement TimescaleDB migrations for high-frequency pricing historical databases.
- [x] Build dedicated ECharts retrieval-augmented forecasting panel and next-candle direction predictor.
- [x] Run load tests evaluating API WebSocket server latency across multiple token configurations.

### Phase 5: Deep Learning Training Pipeline & RAFT Integration (Completed ✅)
- [x] Modify timeseries dataset and dataloaders to return absolute sample indices.
- [x] Resolve Python 3 relative/absolute import pathways and case-sensitivity issues in the deep learning submodules.
- [x] Extend `ForecastExp` and `MovementExp` training runners to support RAFT pre-computation phases and index-based forward signatures.
- [x] Fix training loop evaluation bugs and implement dynamic evaluation modes in `forecast.py` and `movement.py`.
- [x] Write and run training pipeline integration tests verifying deep learning training pipelines end-to-end.

### Phase 6: Order Book Analytics & Multi-Exchange Metrics (Completed ✅)
- [x] Fix bid/ask filter logic bug in `validate_feeds` in `book.py`.
- [x] Add detailed mathematical docstrings to the Rollbit index price calculation in `formula.py`.
- [x] Implement multi-exchange order book meta-statistics (price dispersion, HHI concentration, global arbitrage spreads).
- [x] Write and verify comprehensive unit tests in `tests/test_order_book_analytics.py`.

### Phase 7: FastAPI Codebase Refactoring & Modularity (Completed ✅)
- [x] Modularize `services/serve/app.py` by extracting endpoints to specific APIRouters under `services/serve/routers/`.
- [x] Extract market feed REST/WebSocket streams to `market.py`.
- [x] Extract retrieval forecasting proxy to `retrieval.py`.
- [x] Extract background service control and status WebSocket endpoints to `services.py`.
- [x] Resolve circular imports by shifting dependencies to request/websocket contexts.
- [x] Instantiate a unified global `websocket_manager` singleton in `websocket.py`.

## Sprint Progress

### Current Goal: FastAPI Serve Service Router Split
- [x] Extract endpoints from monolithic serving app to router submodules.
- [x] Instantiate unified WebSocket ConnectionManager singleton.
- [x] Update memory bank, active context, and README documentation.
- [x] Verify local tests and run live Docker compose queries.
