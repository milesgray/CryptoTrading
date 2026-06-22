# Services

This folder contains self-contained services that power the Polymarket trading bot and data pipeline. Each service is a standalone process that leverages the `src/cryptotrading` library for shared functionality.

## Overview
- Each subfolder represents a **microservice** with a specific responsibility (e.g., `predict`, `trade`, `sentiment`).
- Services communicate via REST APIs (exposed by `serve`) or shared data stores.
- Built with **FastAPI** (Python) and designed for horizontal scalability.

## Services
| Service      | Description                                                                                     |
|--------------|-------------------------------------------------------------------------------------------------|
| `embed`      | Generates embeddings for market data (e.g., order book, price movements).                       |
| `jepa`       | Implements a Joint Embedding Predictive Architecture (JEPA) for market prediction.              |
| `predict`    | Runs predictive models (e.g., Monte Carlo simulations, alpha generation) for trading signals.   |
| `pressure`   | Monitors market pressure (liquidity, volume, order flow) and generates alerts.                  |
| `price`      | Fetches and normalizes real-time price data from multiple exchanges.                            |
| `sentiment`  | Analyzes market sentiment from social media, news, and trading activity.                        |
| `serve`      | Aggregates data from all services and exposes a unified REST API for the `frontend`.            |
| `trade`      | Executes trades on Polymarket based on signals from `predict` and `sentiment`.                  |
| `train`      | Trains and fine-tunes predictive models using historical data.                                  |

## Setup
1. Install dependencies for all services:
   ```bash
   cd /home/miles/Development/notebooks/CryptoTrading
   poetry install
   ```

2. Configure environment variables:
   - Copy `.env.example` from `src/` to each service folder and update as needed.
   - Example:
     ```bash
     cp src/.env.example services/predict/.env
     ```

3. Run a service:
   ```bash
   cd services/<service_name>
   poetry run python main.py
   ```

## Development
- Each service has its own `main.py` entrypoint and `requirements.txt` (or `pyproject.toml`).
- Shared code (e.g., `src/cryptotrading`) is imported as a Python package.
- Use `services/serve` to test API endpoints during development.

## Deployment
- Services are designed to run in **Docker containers** or as **systemd services**.
- See `src/dev.sh` for a development orchestration script.