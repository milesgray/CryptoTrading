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
   uv run python oracle.py  # Alert generation
   ```

## Usage
- Input: Real-time order book data (from `price` service).
- Output: JSON alerts with pressure metrics (e.g., `{"pressure": "high", "confidence": 0.95}`).

## Development
- Extend `src/cryptotrading/analysis` for new pressure metrics.
- Backtest alerts using historical data from `src/cryptotrading/data`.