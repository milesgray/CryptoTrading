# Technology Context: CryptoTrading

## Core Technology Stack

### Programming Languages
- **Python ^3.10**: Primary language for data pipelines, microservices, and deep learning encoders.
- **JavaScript (ES6+)**: Used in the Vite React frontend dashboard.

### Backend Infrastructure
- **FastAPI (0.115.11)**: Async REST & WebSocket server.
- **Uvicorn**: ASGI server for running FastAPI instances.
- **Poetry**: Python packaging and dependency management.

### Databases & Clients
- **MongoDB**: Used for storing raw price data, composite order books, and Twitter sentiment documents.
  - **pymongo (4.11.2)**: Sync MongoDB driver.
  - **motor (3.7.0)**: Async MongoDB driver for FastAPI integration.
- **PostgreSQL**: Used for pattern match storage and timeseries tracking.
  - **pgvector**: Postgres extension for vector embeddings storage and index searching.
  - **TimescaleDB**: Extension optimized for fast time-series queries.
- **CCXT (4.5.5)**: Cryptocurrency Exchange Trading Library used to fetch streaming and REST order book data from Spot/Derivative exchanges.

### Data Science & Machine Learning
- **PyTorch**: Used for building and training TimesNet, Autoformer, SupCon encoders, and Koopman-JEPA architectures.
- **Pandas (2.2.3) & Numpy (2.2.3)**: Data manipulation, return normalization, and order book aggregation calculations.
- **Matplotlib (3.10.1)**: Data visualization and distribution plots.

### Frontend Dashboard
- **React (Vite)**: Standalone UI running on port 8080 / 5173.
- **TradingView Lightweight Charts**: For rendering canvas-based interactive price candles.
- **Tailwind CSS & PostCSS**: Modern styling framework.

## Configuration Parameters

All system configurations are read from `.env` and initialized in `src/cryptotrading/config.py`.

### Environment Variables
| Variable | Default Value | Purpose |
|----------|---------------|---------|
| `MONGO_URI` | `mongodb://192.168.0.15:27017` | MongoDB connection string (to be deprecated) |
| `MONGO_DB_NAME` | `crypto_prices` | Target MongoDB database |
| `POSTGRES_URI` | `postgresql://postgres:postgres@localhost:5432/crypto_trading` | PostgreSQL connection string |
| `POSTGRES_USE_TIMESCALE` | `true` | Enables TimescaleDB hypertable features |
| `POSTGRES_USE_PGVECTOR` | `true` | Enables vector similarity checks |
| `SYMBOLS` | `BTC/USDT,ETH/USDT` | Comma-separated assets to stream and record |
| `STALE_THRESHOLD_SEC` | `30` | Max age in seconds for a price feed to be valid |
| `PRICE_DEVIATION_THRESHOLD` | `0.1` | Max deviation (10%) from median midpoint allowed |
| `MIN_VALID_FEEDS` | `6` | Minimum active feeds required to calculate index |
| `MAX_ORDER_SIZE` | `1,000,000` | Max dollar size capped per order level in composite book |

## Development & Execution Scripts

Shell scripts are located in `src/` to automate microservice startup:

- **Start Serving Server** (`src/dev.sh`):
  ```bash
  poetry run fastapi dev ../services/serve/app.py --host 0.0.0.0 --port 8362
  ```
- **Start Price Logger** (`src/record.sh`):
  ```bash
  nohup poetry run python ../services/price/service.py > record_output.log 2>&1 &
  ```
- **Alternative Serve Command** (`src/serve.sh`):
  ```bash
  poetry run fastapi ../services/serve/app.py
  ```

## Docker Infrastructure

The deployment configurations are declared across three specialized Dockerfiles:

### 1. Dockerfile.serve (REST & WebSockets)
- **Base Image**: `python:3.12-slim`
- **Port Exposed**: `8000`
- **Command**: `poetry run fastapi dev app.py` (running from serve context)

### 2. Dockerfile.record (Ingestion Engine)
- **Base Image**: `python:3.12-slim`
- **Port Exposed**: `8300`
- **Command**: `poetry run python cryptotrading/rollbit/prices/record.py`

### 3. Dockerfile (Frontend)
- Located in `/frontend/Dockerfile` (node-based production image or dev exposure on port 8080).
- Configured with Vite to proxy API requests to backend servers.

### 4. docker-compose.yml
Coordinates the containers locally:
- **serve**: built from context `./package` using `Dockerfile.serve` mapping `8000:8000`.
- **record**: built from context `./package` using `Dockerfile.record` mapping `8300:8300`.
- **frontend**: built from context `./frontend` using `Dockerfile` mapping `8080:8080`.
