# Retrieval Service

The **Retrieval Service** is a high-performance quantitative pattern-matching and timeseries forecasting microservice. It implements shape-similarity search (via Pearson Correlation) and frequency-domain similarity (via Short-Time Fourier Transform/SpecReTF) to project future price continuations based on historical matching setup windows.

## Core Features

1. **CCXT Historical Bootstrapping**: Automatically pulls 7 days of 1-minute candlestick data from centralized exchanges (Binance, Coinbase, Kraken) on startup if sufficient reference price data is missing in PostgreSQL.
2. **Combined Representations**: Concatenates the 128D deep learning representation from the Embed Service with the 56D local handcrafted spectral and order book imbalance features, producing a highly descriptive 184D combined embedding vector (with fallback to local 56D for offline tests).
3. **No-Mock Strictness**: Removed all artificial random path generation. The service raises clean errors if reference data is missing or connection to Postgres is interrupted, propagated to the frontend dashboard.
4. **Spectral Retrospective Forecaster (SpecReTF)**:
   - Partitions returns using STFT (Hann-windowed).
   - Computes composite frame similarity using Jensen-Shannon Divergence on normalized amplitude distributions, combined with phase coherence.
   - Outputs a returns-aligned, scale-invariant consensus path projected directly from the query's terminal candlestick.

## Directory Structure

```text
services/retrieval/
├── README.md         # Service documentation
├── encoder.py        # Vector encoding wrapper (calls embed service)
├── forecaster.py     # Shape and SpecReTF forecast models
└── main.py           # FastAPI server and CCXT bootstrap lifecycle
```

## API Specification

### `GET /forecast`

Retrieves similar historical segments and calculates consensus predictions.

- **Parameters**:
  - `symbol` (str, default: `"BTC"`): Trading symbol/token to match.
  - `k` (int, default: `5`): Number of nearest neighbors to retrieve.
- **Response**:
  - `retrieved` (List[Dict]): Details of matched historical segments.
  - `prediction` (float): Projected target price at the forecast horizon.
  - `consensus_path` (List[float]): Aligned consensus price series projection.
  - `expected_return` (float): Return percentage of consensus prediction.
  - `bull_ratio` (float): Percentage of bullish matches in the retrieved set.
  - `volatility` (float): Standard deviation of retrieved segment returns.
  - `direction` (str): `"BULLISH"` or `"BEARISH"` based on expected return.
