import os
import logging
from typing import Optional, Dict, Any
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# MongoDB Configuration (to be deprecated)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "crypto_prices")
PRICE_COLLECTION_NAME = os.getenv("MONGO_PRICE_COLLECTION_NAME", "price_data")
COMPOSITE_ORDER_BOOK_COLLECTION_NAME = os.getenv("MONGO_COMPOSITE_ORDER_BOOK_COLLECTION_NAME", "composite_order_book_data")
EXCHANGE_ORDER_BOOK_COLLECTION_NAME = os.getenv("MONGO_EXCHANGE_ORDER_BOOK_COLLECTION_NAME", "exchange_order_book_data")
TRANSFORMED_ORDER_BOOK_COLLECTION_NAME = os.getenv("MONGO_TRANSFORMED_ORDER_BOOK_COLLECTION_NAME", "transformed_order_book_data")
TWEET_COLLECTION_NAME = os.getenv("MONGO_TWEET_COLLECTION_NAME", "tweet_data")

# PostgreSQL Configuration
POSTGRES_URI = os.getenv("POSTGRES_URI", "postgresql://postgres:postgres@localhost:5432/crypto_trading")
POSTGRES_POOL_MIN = int(os.getenv("POSTGRES_POOL_MIN", "5"))
POSTGRES_POOL_MAX = int(os.getenv("POSTGRES_POOL_MAX", "20"))
POSTGRES_SSL_MODE = os.getenv("POSTGRES_SSL_MODE", "prefer")
POSTGRES_USE_TIMESCALE = os.getenv("POSTGRES_USE_TIMESCALE", "true").lower() == "true"
POSTGRES_USE_PGVECTOR = os.getenv("POSTGRES_USE_PGVECTOR", "true").lower() == "true"

# Parse connection string for additional configuration
def parse_postgres_uri(uri: str) -> Dict[str, Any]:
    """Parse PostgreSQL connection URI into components with additional parameters."""
    parsed = urlparse(uri)
    query = parse_qs(parsed.query)
    
    # Convert query parameters to single values instead of lists
    query_params = {k: v[0] if len(v) == 1 else v for k, v in query.items()}
    
    # Build connection parameters
    conn_params = {
        'host': parsed.hostname,
        'port': parsed.port or 5432,
        'user': parsed.username,
        'password': parsed.password,
        'database': parsed.path.lstrip('/').split('?')[0],
        'sslmode': POSTGRES_SSL_MODE,
        'application_name': 'crypto_trading_app',
        'connect_timeout': '10',
        'keepalives': '1',
        'keepalives_idle': '30',
        'keepalives_interval': '10',
        'keepalives_count': '5',
        **query_params  # Allow override of any parameters via query string
    }
    
    # Remove None values
    return {k: v for k, v in conn_params.items() if v is not None}

# Get PostgreSQL connection parameters
POSTGRES_PARAMS = parse_postgres_uri(POSTGRES_URI)


# Configuration for exchanges and symbols
SPOT_EXCHANGES = ["binanceus", "coinbase", "kraken", "huobi", "okx"]

DERIVATIVE_EXCHANGES = [
    "binanceus-coin-m", "binanceus-usdt-m", 
    "huobi-coin-m", "huobi-usdt-m", 
    "okx-coin-m", "okx-usdt-m"
]
SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT")
if "," in SYMBOLS:
    SYMBOLS = SYMBOLS.split(",")
elif not isinstance(SYMBOLS, list):
    SYMBOLS = [SYMBOLS]
logger.info(f"SYMBOLS loaded: {SYMBOLS}")

# System parameters

STALE_THRESHOLD_SEC = int(os.getenv("STALE_THRESHOLD_SEC", 30))
PRICE_DEVIATION_THRESHOLD = float(os.getenv("PRICE_DEVIATION_THRESHOLD", 0.1))
MIN_VALID_FEEDS = int(os.getenv("MIN_VALID_FEEDS", 6))
MAX_ORDER_SIZE = int(os.getenv("MAX_ORDER_SIZE", 1_000_000))