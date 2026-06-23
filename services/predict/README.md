# Predict Service

## Overview
The `predict` service generates trading signals using predictive models (e.g., Monte Carlo simulations, alpha generation) for the Polymarket trading bot.

## Features
- Runs **quantitative models** to forecast market movements.
- Outputs **actionable signals** (buy/sell/hold) for the `trade` service.
- Integrates with the `src/cryptotrading/predict` library for shared model logic.

## Setup
1. Install dependencies:
   ```bash
   cd /home/miles/Development/notebooks/CryptoTrading
   uv sync
   ```

2. Configure environment variables:
   ```bash
   cp src/.env.example services/predict/.env
   ```

3. Run the service:
   ```bash
   cd services/predict
   uv run python service.py
   ```

## Usage
- Signals are exposed via an internal API (consumed by `trade` and `serve`).
- Input: Market data (prices, volume, order book).
- Output: JSON with predicted probabilities and recommended actions.

## Development
- Extend `src/cryptotrading/predict` for new models.
- Backtest strategies using historical data from `src/cryptotrading/data`.