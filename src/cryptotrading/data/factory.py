import logging
from cryptotrading.config import DB_BACKEND

logger = logging.getLogger(__name__)

def get_price_adapter(backend=None):
    """Retrieve the price adapter matching the active DB_BACKEND configuration."""
    if backend is None:
        backend = DB_BACKEND
    if backend == 'mongodb':
        from cryptotrading.data.price import PriceMongoAdapter
        logger.info("Initializing MongoDB Price Adapter")
        return PriceMongoAdapter()
    else:
        from cryptotrading.data.price import PricePostgresAdapter
        logger.info("Initializing PostgreSQL Price Adapter")
        return PricePostgresAdapter()

def get_order_book_adapter(backend=None):
    """Retrieve the order book adapter matching the active DB_BACKEND configuration."""
    if backend is None:
        backend = DB_BACKEND
    if backend == 'mongodb':
        from cryptotrading.data.book import OrderBookMongoAdapter
        logger.info("Initializing MongoDB Order Book Adapter")
        return OrderBookMongoAdapter()
    else:
        from cryptotrading.data.book import OrderBookPostgresAdapter
        logger.info("Initializing PostgreSQL Order Book Adapter")
        return OrderBookPostgresAdapter()

def get_twitter_adapter(backend=None):
    """Retrieve the twitter sentiment adapter matching the active DB_BACKEND configuration."""
    if backend is None:
        backend = DB_BACKEND
    if backend == 'mongodb':
        from cryptotrading.data.twitter import TwitterMongoAdapter
        logger.info("Initializing MongoDB Twitter Adapter")
        return TwitterMongoAdapter()
    else:
        from cryptotrading.data.twitter import TwitterPostgresAdapter
        logger.info("Initializing PostgreSQL Twitter Adapter")
        return TwitterPostgresAdapter()
