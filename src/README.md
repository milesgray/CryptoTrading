# Source Library (`src`)

This folder contains the core `cryptotrading` library, which provides shared functionality for all `services` and the `frontend`.

## Overview
- A **Python library** (`src/cryptotrading`) that encapsulates:
  - Data fetching and normalization
  - Trading strategies and signal generation
  - Market analysis (technical, sentiment, and predictive)
  - Utilities for logging, configuration, and API clients
- Used by all `services` and indirectly by the `frontend` via the `serve` API.

## Structure
```
src/
├── cryptotrading/
│   ├── analysis/     # Technical and quantitative analysis tools
│   ├── data/         # Data fetching, normalization, and storage
│   ├── predict/      # Predictive models and alpha generation
│   ├── rollbit/      # Rollbit exchange integration
│   ├── sentiment/    # Sentiment analysis tools
│   ├── trade/        # Trading execution and order management
│   ├── util/         # Shared utilities (logging, config, etc.)
│   ├── config.py     # Centralized configuration
│   └── __init__.py  # Package initialization
├── dist/            # Build output (Python wheel)
├── poetry.lock      # Dependency lockfile
├── pyproject.toml   # Project metadata and dependencies
└── .env.example     # Environment variable template
```

## Setup
1. Install dependencies:
   ```bash
   cd /home/miles/Development/notebooks/CryptoTrading
   poetry install
   ```

2. Configure environment variables:
   - Copy `.env.example` to `.env` and update as needed:
     ```bash
     cp src/.env.example src/.env
     ```

3. Build the library:
   ```bash
   poetry build
   ```

## Development
- The library is installed as a **editable package** during development:
  ```bash
  poetry install -e .
  ```
- Use `src/dev.sh` to run tests and linting.

## Key Features
- **Data Fetching**: Unified interface for multiple exchanges (e.g., Rollbit, Polymarket).
- **Trading Strategies**: Backtested strategies for alpha generation.
- **Predictive Models**: Monte Carlo simulations, JEPA, and sentiment analysis.
- **Utilities**: Logging, configuration, and API clients for external services.