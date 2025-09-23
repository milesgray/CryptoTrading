# CryptoTrading

Code that is supposed to help make money predicting make believe internet coinage price behavior

## PostgreSQL Adapter with TimescaleDB and pgvector

This project includes a robust PostgreSQL adapter with TimescaleDB and pgvector support for efficient time-series data storage and vector similarity search.

### Features

- **TimescaleDB Integration**: Optimized for time-series data with automatic hypertable creation
- **pgvector Support**: Store and query vector embeddings for similarity search
- **Connection Pooling**: Efficient database connection management
- **Async/Await Support**: Built with `asyncpg` for high-performance async database access
- **Type Annotations**: Full Python type hints for better IDE support
- **Automatic Schema Management**: Tables and indexes are created automatically
- **Transaction Management**: Context managers for easy transaction handling

### Prerequisites

- Python 3.8+
- PostgreSQL 14+
- TimescaleDB extension
- pgvector extension

### Installation

1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file based on `.env.example` and configure your PostgreSQL connection:
   ```bash
   cp .env.example .env
   ```

3. Initialize the database (this will create tables and extensions):
   ```python
   from cryptotrading.data.postgres import init_db
   import asyncio
   
   async def setup():
       await init_db()
   
   asyncio.run(setup())
   ```

### Usage

#### Basic Database Operations

```python
from cryptotrading.data.postgres import (
    get_connection, 
    transaction, 
    price_repo,
    order_book_repo,
    document_embedding_repo
)
import asyncio
from datetime import datetime, timedelta

# Insert price data
async def insert_price_data():
    price_data = {
        'time': datetime.utcnow(),
        'symbol': 'BTC/USDT',
        'exchange': 'binance',
        'open': 42000.0,
        'high': 42500.0,
        'low': 41900.0,
        'close': 42350.0,
        'volume': 150.5,
        'metadata': {'source': 'websocket'}
    }
    
    # Using repository pattern
    await price_repo.insert(price_data)
    
    # Or using raw SQL with connection
    async with get_connection() as conn:
        await conn.execute('''
        INSERT INTO price_data 
        (time, symbol, exchange, open, high, low, close, volume, metadata)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ''', *price_data.values())

# Query OHLCV data
async def get_ohlcv():
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=7)
    
    ohlcv = await price_repo.get_ohlcv(
        symbol='BTC/USDT',
        timeframe='1h',
        exchange='binance',
        start_time=start_time,
        end_time=end_time,
        limit=1000
    )
    return ohlcv

# Vector similarity search
async def find_similar_embeddings():
    if document_embedding_repo is None:
        print("pgvector is not enabled")
        return []
    
    # Example embedding (1536-dimensional vector like OpenAI's embeddings)
    query_embedding = [0.1] * 1536
    
    similar_docs = await document_embedding_repo.find_similar(
        embedding=query_embedding,
        limit=5,
        min_similarity=0.7,
        filters={
            'metadata->>\'category\'': 'crypto_news'
        }
    )
    return similar_docs

# Run the examples
async def main():
    await insert_price_data()
    ohlcv = await get_ohlcv()
    similar = await find_similar_embeddings()
    
    print(f"Fetched {len(ohlcv)} OHLCV records")
    print(f"Found {len(similar)} similar documents")

if __name__ == "__main__":
    asyncio.run(main())
```

#### Database Schema

The following tables are created automatically:

1. **price_data**: Time-series cryptocurrency price data
2. **order_book_data**: Order book snapshots
3. **trade_data**: Historical trade data
4. **tweet_data**: Social media data with sentiment analysis
5. **document_embeddings**: Vector embeddings for similarity search (if pgvector is enabled)

### Configuration

Configure your database connection in `.env`:

```ini
# PostgreSQL Configuration
POSTGRES_URI="postgresql://username:password@localhost:5432/crypto_trading"
POSTGRES_POOL_MIN=5
POSTGRES_POOL_MAX=20
POSTGRES_SSL_MODE=prefer
POSTGRES_USE_TIMESCALE=true
POSTGRES_USE_PGVECTOR=true

# MongoDB Configuration (legacy, to be removed)
MONGO_URI="mongodb://localhost:27017"
MONGO_DB_NAME="crypto_prices"
```

### Performance Optimization

- **Connection Pooling**: Configured with `POSTGRES_POOL_MIN` and `POSTGRES_POOL_MAX`
- **TimescaleDB Compression**: Data older than 7 days is automatically compressed
- **Retention Policies**: Data is automatically dropped after 1 year (configurable)
- **Indexing**: Optimized indexes for common query patterns

### Migrations

For database schema changes, use Alembic:

```bash
# Install Alembic
pip install alembic

# Initialize migrations (first time only)
alembic init migrations

# Create a new migration
alembic revision --autogenerate -m "Add new columns"

# Apply migrations
alembic upgrade head
```

### Testing

Run the test suite:

```bash
pytest tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
