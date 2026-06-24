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
   uv sync
   ```

2. Configure environment variables:
   ```bash
   cp src/.env.example services/train/.env
   ```

3. Run the service:
   ```bash
   cd services/train
   uv run python main.py
   ```

## Usage
- Input: Historical data (from `src/cryptotrading/data`).
- Output: Trained model weights (saved to `src/cryptotrading/predict/models/`).

## Development
- Extend `src/cryptotrading/predict` for new model architectures.
- Use `src/cryptotrading/analysis` for feature engineering.

## Training Examples

The training entrypoint (`main.py`) supports all models in the `predict` submodules.

### 1. Training a Standard Model (e.g. Linear, Autoformer)
To train a standard deep learning model on historical price data:
```bash
uv run python services/train/main.py \
  --task_name forecast \
  --is_training 1 \
  --model_id test_linear \
  --model Linear \
  --data custom \
  --data_path services/train/data/ETTh1.csv \
  --train_epochs 5 \
  --batch_size 32
```

### 2. Training the Retrieval-Augmented Forecasting Transformer (RAFT)
To train the RAFT model, which performs historical segment similarity search and requires pre-computation and batch index-aware training:
```bash
uv run python services/train/main.py \
  --task_name forecast \
  --is_training 1 \
  --model_id test_raft \
  --model RAFT \
  --data custom \
  --data_path services/train/data/ETTh1.csv \
  --train_epochs 5 \
  --batch_size 32 \
  --n_period 2 \
  --topm 5
```
This performs a full database retrieval sweep across the training, validation, and testing splits during initialization, and trains the shape-retrieval forecasting layers.