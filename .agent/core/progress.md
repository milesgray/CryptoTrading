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
- [x] Migrate primary database backend from MongoDB to PostgreSQL with TimescaleDB (Postgres as new default, Mongo as option).
- [x] Connect the FastAPI server directly to the PostgreSQL + pgvector setups library for live pattern matching queries.
- [x] Implement TimescaleDB migrations for high-frequency pricing historical databases.
- [x] Run load tests evaluating API WebSocket server latency across multiple token configurations.

## Sprint Progress

### Current Goal: Postgres TimescaleDB Migration & Model Formalization
- [x] Implement Postgres/TimescaleDB adapters and dynamic DB backend factory.
- [x] Fix JEPA broken model/helper imports and ensure model tests pass.
- [x] Identify and formalize loosely typed data models (ExchangeRawOrderBook, TweetDataPoint, TweetSentiment).
- [x] Create comprehensive data dictionary documentation (README.md).
- [x] Connect analytics helpers to the database backend factory.
- [ ] Connect the FastAPI server directly to the PostgreSQL + pgvector setups library for live pattern matching queries.
- [ ] Benchmark search times for similar historical setups.
