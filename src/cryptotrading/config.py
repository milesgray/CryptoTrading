import os
import logging

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "crypto_prices")
PRICE_COLLECTION_NAME = os.getenv("MONGO_PRICE_COLLECTION_NAME", "price_data")
COMPOSITE_ORDER_BOOK_COLLECTION_NAME = os.getenv("MONGO_COMPOSITE_ORDER_BOOK_COLLECTION_NAME", "composite_order_book_data")
EXCHANGE_ORDER_BOOK_COLLECTION_NAME = os.getenv("MONGO_EXCHANGE_ORDER_BOOK_COLLECTION_NAME", "exchange_order_book_data")
TRANSFORMED_ORDER_BOOK_COLLECTION_NAME = os.getenv("MONGO_TRANSFORMED_ORDER_BOOK_COLLECTION_NAME", "transformed_order_book_data")
TWEET_COLLECTION_NAME = os.getenv("MONGO_TWEET_COLLECTION_NAME", "tweet_data")


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