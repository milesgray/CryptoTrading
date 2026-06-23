# Price Service

## Overview
The `price` service fetches and normalizes **real-time price data** from multiple exchanges (e.g., Polymarket, Rollbit).

## Features
- **Unified API** for price data (candlesticks, order book, trades).
- **Normalization** (converts exchange-specific formats to a standard schema).
- **WebSocket support** for real-time updates.

## Setup
1. Install dependencies:
   ```bash
   cd /home/miles/Development/notebooks/CryptoTrading
   uv sync
   ```

2. Configure environment variables:
   ```bash
   cp src/.env.example services/price/.env
   ```

3. Run the service:
   ```bash
   cd services/price
   uv run python service.py
   ```

## Usage
- Input: Exchange API keys (configured in `.env`).
- Output: JSON with normalized price data (e.g., `{"symbol": "BTC", "price": 65000, "timestamp": "..."}`).

## Development
- Add new exchanges by extending `src/cryptotrading/data`.
- Test with `src/cryptotrading/rollbit` for Rollbit integration.