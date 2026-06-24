# CryptoTrading Dashboard & Forecasting Engine

A real-time cryptocurrency trading dashboard and predictive engine utilizing timeseries database indexing and retrieval-augmented forecasting.

## Core Architecture
The system is composed of five microservices coordinated via Docker Compose:
1. **TimescaleDB**: Time-series optimized PostgreSQL database with the `pgvector` extension for historical pattern search.
2. **Serve (FastAPI)**: High-performance ASGI gateway managing REST API endpoints and real-time WebSockets.
3. **Retrieval**: Core ML forecasting service that encodes, indexes, and queries historical price sequences using vector search.
4. **Record**: Ingestion daemon that polls Spot/Derivative exchanges (via CCXT) and records high-frequency price feeds and order book depth.
5. **Frontend**: Vite-based React application rendering canvas-based candlestick charts, order book heatmaps, and pattern-matching projections.

---

## Getting Started

### 1. Configuration (`.env`)
Copy `.env.example` to `.env` and customize your settings. Key configurations include:

```env
# Exposed Host Ports
TIMESCALEDB_PORT=5432
SERVE_PORT=8362
RETRIEVAL_PORT=8000
RECORD_PORT=8300
FRONTEND_PORT=8080

# Database Configuration
POSTGRES_DB=crypto_trading
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_URI=postgresql://postgres:postgres@timescaledb:5432/crypto_trading

# Price Ingestion Settings
MIN_VALID_FEEDS=2
```

### 2. Run with Docker Compose
Start all microservices in the background:
```bash
docker compose up -d
```

To rebuild the services (e.g. after code changes):
```bash
docker compose up -d --build
```

Access the dashboard in your browser at `http://localhost:8080/`.

---

## Features
* **Live Candlestick Chart**: Real-time canvas rendering of price movements with smooth time-bucket tick aggregation.
* **Order Book Panel**: Live bidding/asking order depth and spread visualization.
* **Pattern Matching & Retrieval Forecast**: A dedicated panel that queries the vector database for the top-$k$ historical cycles matching the current price momentum and order book shape.
* **Next Candle Color Predictor**: High-impact indicator that averages the active forecasted paths to predict the direction of the immediate next candle (GREEN/RED) with a consensus confidence bar.
* **Forecasting Settings & Toggles**: Interactive sliders/dropdowns in the UI to configure $k$, segment length, retrieval frequency, and order book weighting on the fly.
