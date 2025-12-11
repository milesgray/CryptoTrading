# Trade Setup Pattern Matcher

A contrastive learning system for finding similar historical trade setups in real-time.

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                     OFFLINE PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│  1. Historical Prices → DP Oracle → Optimal Trade Labels     │
│  2. Extract Price Windows Before Each Entry                  │
│  3. Train Contrastive Encoder (SupCon Loss)                 │
│  4. Generate Embeddings → Store in PostgreSQL + pgvector    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     ONLINE PIPELINE                          │
├─────────────────────────────────────────────────────────────┤
│  1. Live Price WebSocket → Rolling Window Buffer            │
│  2. Embed Current Window                                     │
│  3. Query pgvector for Similar Setups                       │
│  4. Aggregate: Direction Consensus + Avg Profit             │
│  5. Display on TradingView Chart                            │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. DP Oracle (`models/dp_oracle.py`)

Dynamic Programming solver for globally optimal trading path. Uses Viterbi algorithm to find the best sequence of Long/Short/Flat states.

### 2. Contrastive Encoder (`models/encoder.py`)

CNN-based encoder trained with Supervised Contrastive Loss. Learns embeddings where:

- Profitable longs cluster together
- Profitable shorts cluster together
- Different outcome magnitudes are distinguished

### 3. PostgreSQL + pgvector (`database/pgvector_store.py`)

Efficient vector similarity search using HNSW index.

### 4. FastAPI Backend (`api/server.py`)

REST API + WebSocket for real-time matching.

### 5. React Frontend (`frontend/index.html`)

TradingView Lightweight Charts visualization with overlay of matched patterns.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup PostgreSQL with pgvector

```bash
# Install pgvector extension
# Ubuntu/Debian:
sudo apt install postgresql-15-pgvector

# Or via Docker:
docker run -d --name pgvector \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  pgvector/pgvector:pg15

# Create database
psql -U postgres -c "CREATE DATABASE trade_embeddings;"
```

### 3. Run Pipeline with Demo Data

```bash
cd trade_embeddings
python pipeline.py --demo --demo-length 50000 --epochs 50
```

### 4. Start API Server

```bash
cd trade_embeddings
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Open Frontend

Open `frontend/index.html` in a browser.

## Usage

### Running Full Pipeline

```python
import asyncio
from pipeline import TradePipeline

async def main():
    pipeline = TradePipeline(
        window_size=100,
        embedding_dim=128,
        max_leverage=20.0
    )
    
    # Load your price data
    prices = ...  # numpy array
    timestamps = ...  # numpy array
    
    result = await pipeline.run_full_pipeline(
        prices=prices,
        timestamps=timestamps,
        symbol='BTCUSDT',
        timeframe='1s',
        train_epochs=100
    )
    
    print(result)
    await pipeline.close()

asyncio.run(main())
```

### Using with PriceServerClient

```python
from price_client import PriceServerClient
from pipeline import run_pipeline_from_price_client

client = PriceServerClient(
    tokens=['BTC'],
    rest_api_url='http://your-price-server/api',
    websocket_url='ws://your-price-server/ws'
)

result = await run_pipeline_from_price_client(
    price_client=client,
    token='BTC',
    days=7
)
```

### Real-time Matching via WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/live/BTCUSDT');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'match') {
        console.log('Similar setup found!');
        console.log('Direction:', data.direction_signal);
        console.log('Confidence:', data.confidence);
        console.log('Avg Profit:', data.avg_profit);
    }
};

// Send price updates
ws.send(JSON.stringify({
    type: 'price',
    price: 50000.0,
    timestamp: Date.now() / 1000
}));
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/embed` | POST | Embed a price window |
| `/search` | POST | Find similar setups |
| `/setup/{id}` | GET | Get setup by ID |
| `/stats` | GET | Database statistics |
| `/health` | GET | Health check |
| `/ws/live/{symbol}` | WS | Live streaming |

## Configuration

Create `config.json` in the api directory:

```json
{
    "model_path": "models/trained/encoder.pt",
    "window_size": 100,
    "embedding_dim": 128,
    "db_host": "localhost",
    "db_port": 5432,
    "db_name": "trade_embeddings",
    "db_user": "postgres",
    "db_password": "postgres"
}
```

## Embedding Details

### Input Normalization

Raw prices are converted to log returns and standardized:

```python
log_prices = np.log(prices)
returns = np.diff(log_prices)
normalized = (returns - mean) / std
```

### Encoder Architecture

- 1D CNN with residual connections
- Multi-scale pooling (global + adaptive)
- MLP projection head
- L2 normalized output (128-dim by default)

### Contrastive Learning

Supervised Contrastive (SupCon) loss groups embeddings by outcome label:

- 10 classes: 5 outcomes × 2 directions
- Outcomes: LOSS_LARGE, LOSS_SMALL, FLAT, PROFIT_SMALL, PROFIT_LARGE

## License

MIT
