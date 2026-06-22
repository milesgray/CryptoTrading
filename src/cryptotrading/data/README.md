# Data Layer Schema & Models Documentation

This document describes the schemas and model mappings used by both MongoDB and PostgreSQL/TimescaleDB databases in the Crypto Trading project.

## Core Models

### 1. `ExchangeRawOrderBook`
Represents the raw, un-aggregated order book returned directly by individual exchange feeds (e.g., Binance, Bybit).
- **Python Model**: `ExchangeRawOrderBook` (in [models.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/data/models.py))
- **Fields**:
  - `exchange` (str): Name of the exchange (e.g., `'binance'`).
  - `bids` (list[tuple[float, float]]): Sorted list of bid price levels `[(price, size)]`.
  - `asks` (list[tuple[float, float]]): Sorted list of ask price levels `[(price, size)]`.
  - `timestamp` (float, optional): Millisecond timestamp of data capture.

---

### 2. `OrderBookSnapshot`
A sanitized, validated order book snapshot containing sorted bids (descending) and asks (ascending).
- **Python Model**: `OrderBookSnapshot` (in [models.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/data/models.py))
- **Fields**:
  - `timestamp` (float): Unix epoch time.
  - `bids` (list[tuple[float, float]]): descending sorted bids.
  - `asks` (list[tuple[float, float]]): ascending sorted asks.
  - `mid_price` (float): Midpoint price calculated as `(best_bid + best_ask) / 2`.

---

### 3. `CandlestickData`
Aggregated timeseries price intervals used for charting and technical analysis indicators.
- **Python Model**: `CandlestickData` (in [models.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/data/models.py))
- **Fields**:
  - `timestamp` (datetime): Time bucket start.
  - `open` (float): First index price in interval.
  - `high` (float): Maximum index price in interval.
  - `low` (float): Minimum index price in interval.
  - `close` (float): Last index price in interval.
  - `volume` (float, optional): Summed volume calculated from order book depth updates.
  - `exchange_count` (float, optional): Average number of active feeds in interval.

---

### 4. `TweetDataPoint`
Highly structured model representing scraped tweets, parsed token references, engagement details, and VADER/TextBlob sentiment metrics.
- **Python Model**: `TweetDataPoint` (in [models.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/data/models.py))
- **Fields**:
  - `tweet_id` (str): Unique tweet identifier.
  - `user_id` (int/str): Author ID.
  - `username` (str): Twitter handle.
  - `text` (str): Raw text content.
  - `timestamp` (datetime): Timestamp of posting.
  - `token_symbol` (str): Upper-case token symbol matching target.
  - `sentiment` (TweetSentiment): Nested sentiment score object containing `compound`, `polarity`, `subjectivity`, and `confidence`.
  - `price_direction_signals` (dict): Directional heuristics (bullish, bearish, uncertainty, volume).
  - `follower_count` (int): Author follower count (used for sentiment weighting).

---

## TimescaleDB Configurations & Optimization

PostgreSQL stores these models under relation schemas optimized as TimescaleDB **hypertables** partitioned along the `time` dimension.

### Hypertables and Indexes
1. **`price_data`**:
   - Stores `PriceDataPoint` index, `ExchangeRawOrderBook` data, and `TransformedOrderBookData` by setting different `exchange` identifiers (`'index'`, `'exchange_raw_*'`, `'composite'`).
   - Partitioned by: `time` (TIMESTAMPTZ).
   - Core index: `idx_price_data_symbol_time ON price_data(symbol, time DESC)`.
   - Metadata storage: The JSONB column `metadata` stores unstructured key-value fields. A GIN index `idx_price_data_metadata` is built to make querying JSON fields fast.
   
2. **`tweet_data`**:
   - Stores sentiment details and engagement metadata.
   - Core index: `idx_tweet_data_created_at ON tweet_data(created_at DESC)`.
   - GIN index on `metadata` allows fast lookup on nested token symbol values.

### Compression Policies
To ensure high performance with very large datasets, compression policies are automatically attached:
```sql
SELECT add_compression_policy('price_data', INTERVAL '7 days');
```
This compresses chunks older than 7 days, providing up to 90% storage savings and faster analytic query scans.
