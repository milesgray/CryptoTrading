# Train Service

## Overview
The `train` service trains and fine-tunes **predictive models** using historical market data.

## Features
- **Automated training pipelines** for models in `src/cryptotrading/predict`.
- **Hyperparameter optimization** (e.g., grid search, Bayesian optimization).
- **Backtesting** to validate model performance.

## Structure
```
train/
├── data/          # Training datasets
└── main.py        # Training entrypoint
```

## Setup
1. Install dependencies:
   ```bash
   cd /home/miles/Development/notebooks/CryptoTrading
   poetry install
   ```

2. Configure environment variables:
   ```bash
   cp src/.env.example services/train/.env
   ```

3. Run the service:
   ```bash
   cd services/train
   poetry run python main.py
   ```

## Usage
- Input: Historical data (from `src/cryptotrading/data`).
- Output: Trained model weights (saved to `src/cryptotrading/predict/models/`).

## Development
- Extend `src/cryptotrading/predict` for new model architectures.
- Use `src/cryptotrading/analysis` for feature engineering.