# Progress: CryptoTrading

## Project Timeline

### Phase 1: Foundation & Data Streaming (Completed ✅)
- [x] Poetry environment configuration and dependencies (`ccxt`, `motor`, `pymongo`, `fastapi`, `torch`).
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

### Phase 4: Database Integration & Production Simulation (In Progress 🔄)
- [ ] Connect the FastAPI server directly to the PostgreSQL + pgvector setups library for live pattern matching queries.
- [ ] Implement TimescaleDB migrations for high-frequency pricing historical databases.
- [ ] Run load tests evaluating API WebSocket server latency across multiple token configurations.

## Sprint Progress

### Current Goal: pgvector Integration & Live Setup Matching
- Integrate `pgvector` HNSW queries into the REST API.
- Benchmark search times for similar historical setups.
