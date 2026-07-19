# Pressure Service

## Overview
The `pressure` service monitors **market pressure** (liquidity, volume, order flow) and generates alerts for the trading bot.

## Features
- Tracks **real-time order book dynamics** (bid/ask pressure, slippage).
- Identifies **liquidity gaps** and **volume spikes**.
- Outputs **alerts** for the `trade` service to adjust positions.

## Structure
```
pressure/
├── main.py              # FastAPI server and training console endpoints
├── data_loader.py       # Fetches and preprocesses market data
├── model.py             # Pressure detection logic
├── oracle.py            # Alert generation
├── pressure_features.py # Feature engineering
├── train.py             # Model training
└── util.py              # Shared utilities
```

## Setup
1. Install dependencies:
   ```bash
   cd /home/miles/Development/notebooks/CryptoTrading
   uv sync
   ```

2. Configure environment variables:
   ```bash
   cp src/.env.example services/pressure/.env
   ```

3. Run the service:
   ```bash
   cd services/pressure
   uv run uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## API Endpoints
- **`POST /features`**: Takes an order book snapshot and returns extracted & normalized feature arrays.
- **`POST /predict`**: Takes an order book snapshot, runs featurization, and outputs buy/sell/total pressure predictions from the model.
- **`POST /train`**: Triggers a background model training job on historical snapshots.
- **`GET /train/status`**: Yields real-time training progress metrics (epochs, losses, progress percent).

## Development
- Extend `src/cryptotrading/analysis` for new pressure metrics.
- Backtest alerts using historical data from `src/cryptotrading/data`.