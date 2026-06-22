# System Patterns: CryptoTrading

## Architecture Overview

CryptoTrading is structured as a collection of decoupled, specialized services. Below is a block diagram illustrating the data flows and service roles:

```text
       ┌──────────────────────────────┐
       │   Exchange API Websockets    │
       └──────────────┬───────────────┘
                      │
                      ▼
 ┌──────────────────────────────────────────┐      ┌─────────────────────────┐
 │   Price Ingestion Service (Port 8300)    │◄────►│      Twitter API        │
 │ • Filters stale & crossed book feeds     │      └────────────┬────────────┘
 │ • Builds Composite Order Book ($1M cap)  │                   │
 │ • Calculates Rollbit Index every 500ms   │                   ▼
 └────────────────────┬─────────────────────┘      ┌─────────────────────────┐
                      │                            │    Sentiment Service    │
                      ▼                            │ • Ingests Twitter stream│
 ┌──────────────────────────────────────────┐      │ • Computes VADER scores │
 │            MongoDB Database              │      └────────────┬────────────┘
 │ • price_data, tweet_data                 │                   │
 │ • transformed_order_book_data            │                   │
 └────────────────────┬─────────────────────┘                   │
                      │                                         │
                      ├─────────────────────────────────────────┘
                      ▼
 ┌──────────────────────────────────────────┐      ┌─────────────────────────┐
 │       API Serving Server (Port 8362)     │      │   Contrastive Encoder   │
 │ • FastAPI REST & WebSocket Endpoints     │◄────►│     (services/embed)    │
 │ • Serves Candlestick & Order Book feeds  │      │ • CNN (SupCon Loss)     │
 └────────────────────┬─────────────────────┘      │ • Postgres + pgvector   │
                      │                            └─────────────────────────┘
                      ▼
 ┌──────────────────────────────────────────┐      ┌─────────────────────────┐
 │           Vite React Dashboard           │      │    Koopman-JEPA Model   │
 │ • TradingView Lightweight Charts         │      │      (services/jepa)    │
 │ • Displays prices & pattern matches      │      │ • Regime indicators     │
 └──────────────────────────────────────────┘      │ • Dynamic leverage      │
                                                   └─────────────────────────┘
```

## Service Components

### 1. Ingestion & Price Recording (`services/price`)
- **EntryPoint**: `services/price/service.py` (run via `src/record.sh` which executes `poetry run python ../services/price/service.py`).
- **Core Engine**: `cryptotrading.rollbit.prices.price.PriceSystem` initialized with exchange symbols.
- **Formulation Rules**:
  - Drops feeds with no updates for >30 seconds.
  - Drops crossed-book feeds or top-of-book prices deviating >10% from the median.
  - Requires at least 6 valid feeds to update index.
  - Caps individual order sizes at $1M.
  - Computes weighted average marginal mid-prices based on exponential weights $w_i = L \cdot \exp(-L \cdot v_i)$.

### 2. API Serving (`services/serve`)
- **EntryPoint**: `services/serve/app.py` (run via `src/dev.sh` or `src/serve.sh`).
- **REST APIs**:
  - `/historic/price/{token}`: Paginated price queries.
  - `/candlestick/{token}`: OHLCV candlestick data generation with granularity and optional book data.
  - `/latest_price/{token}`: Latest price and order book summary.
  - `/transformed_order_book/{token}`: Ingested order book structures.
  - `/health`: DB connection check.
- **WebSockets**:
  - `/ws/price/{token}`: Real-time price and summary broadcasts.
  - `/ws/order_book/{token}`: Real-time order book structure updates.
  - Handled by a connection pool manager (`ConnectionManager`) supporting ping/pong heartbeat keep-alives.

### 3. Pattern Matching (`services/embed`)
- **Theory**: Contrastive learning (Supervised Contrastive / SupCon Loss) maps log return price windows ($x_t - x_{t-1}$) into 128-dimensional normalized embedding vectors.
- **Database**: Stores embeddings in PostgreSQL using `pgvector` with HNSW indices to achieve low-latency setup lookup.
- **Classification Categories**: 10 outcome classes (5 magnitudes $\times$ 2 directions: LONG_LARGE, LONG_SMALL, FLAT, SHORT_SMALL, SHORT_LARGE).

### 4. Regime Indicator (`services/jepa`)
- **Theory**: Joint-Embedding Predictive Architecture (Ruiz-Morales et al., AAAI 2025) with Koopman operator characteristics.
- **Loss Formulation**:
  - **JEPA Loss**: Predictor target estimation error.
  - **Regime Consistency Loss**: Minimizes KL divergence of regime predictions within temporal windows (exploitation of pathwise invariance).
  - **Price Direction Preservation**: Consine-similarity constraints matching returns.
  - **Volatility Alignment**: Predicts and matches volatility persistence.
  - **Spectral Regularization**: Drives linear predictor matrix $M$ eigenvalues toward 1 (identity on invariant regime subspaces).
- **Controller**: Adjusts trading position leverage dynamically according to detected regimes.

### 5. Sentiment Analyzer (`services/sentiment`)
- **Theory**: Twitter API listener computing real-time sentiment polarity metrics using VADER and feeding predictions. Saves data directly to MongoDB.

### 6. Forecasting Trainer (`services/train`)
- **Theory**: Orchestrates training of deep learning architectures (TimesNet, Autoformer, standard Transformer) for long/short term forecasting and movement classification.

## Design Patterns

### Rolling Window Normalization
Raw prices are normalized to returns and standardized prior to embedding:
```python
log_prices = np.log(prices)
returns = np.diff(log_prices)
normalized = (returns - mean) / std
```

### Exponential Weights in Order Book
Aggregates bids/asks and computes marginal prices, weighting smaller volumes closer to top-of-book higher:
```python
L = 1.0 / V  # V is min of bids/asks total depth
w = L * np.exp(-L * cumulative_sizes)
```

### Predictor Spectral Alignment in JEPA
Restricts JEPA linear predictor matrix to have near-identity projection on invariant manifolds:
```python
loss_spectral = torch.mean((torch.abs(torch.linalg.eigvals(M)) - 1.0) ** 2)
```

## Data Patterns

- **MongoDB Collections**:
  - `price_data`: Ingested raw price entries.
  - `transformed_order_book_data`: Extracted top of book, spreads, and midpoints.
  - `tweet_data`: Sentiment and tweet metadata.
- **Postgres / pgvector Schema**:
  - Vector dimensionality: 128 (default).
  - HNSW Index: Built for cosine distance metrics.

## Testing Patterns

- **Database Isolation**: Pytest test suite setups must hook up mock instances or isolated development DB collections.
- **Component Mocking**: Price calculations are validated using simulated price streams before execution on real WebSocket APIs.
