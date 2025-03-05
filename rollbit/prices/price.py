import os
import asyncio
import time
import logging
import numpy as np
from typing import Optional, Any
import datetime
import motor.motor_asyncio
from pymongo import ASCENDING
import ccxt.async_support as ccxt

import formula

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('price_system')

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "crypto_prices")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "price_data")

# Configuration for exchanges and symbols
SPOT_EXCHANGES = ["binanceus", "coinbase", "kraken", "huobi", "okx"]
DERIVATIVE_EXCHANGES = [
    "binanceus-coin-m", "binanceus-usdt-m", 
    "huobi-coin-m", "huobi-usdt-m", 
    "okx-coin-m", "okx-usdt-m"
]
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

# System parameters
REFRESH_INTERVAL_MS = 500  # 500 milliseconds
STALE_THRESHOLD_SEC = 30  # 30 seconds
PRICE_DEVIATION_THRESHOLD = 0.1  # 10%
MIN_VALID_FEEDS = 6
MAX_ORDER_SIZE = 1_000_000  # $1 million cap per order

class PriceSystem:
    def __init__(self):
        self.exchanges = {}
        self.db_client = None
        self.db = None
        self.collection = None
        self.running = False
        self.last_index_prices = {}

    async def initialize(self):
        """Initialize exchange connections and database"""
        logger.info("Initializing price system...")
        
        # Initialize MongoDB connection
        self.db_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        self.db = self.db_client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]
        
        # Create time series collection if it doesn't exist
        collections = await self.db.list_collection_names()
        if COLLECTION_NAME not in collections:
            await self.db.create_collection(
                COLLECTION_NAME,
                timeseries={
                    'timeField': 'timestamp',
                    'metaField': 'metadata',
                    'granularity': 'seconds'
                }
            )
            # Create indexes for faster queries
            await self.collection.create_index([("timestamp", ASCENDING)])
            await self.collection.create_index([("metadata.symbol", ASCENDING)])
            
        # Initialize exchange connections
        for exchange_id in SPOT_EXCHANGES + DERIVATIVE_EXCHANGES:
            try:
                # Handle special case for derivative exchanges
                base_exchange_id = exchange_id.split('-')[0] if '-' in exchange_id else exchange_id
                exchange_class = getattr(ccxt, base_exchange_id)
                
                # Create exchange instance with custom options if needed
                exchange_options = {
                    'enableRateLimit': True,
                    'timeout': 30000,
                }
                
                if '-' in exchange_id:
                    # Handle different margined futures for derivatives
                    market_type = exchange_id.split('-')[1]
                    if market_type == 'coin-m':
                        exchange_options['options'] = {'defaultType': 'delivery'}
                    elif market_type == 'usdt-m':
                        exchange_options['options'] = {'defaultType': 'future'}
                
                self.exchanges[exchange_id] = exchange_class(exchange_options)
                logger.info(f"Initialized exchange: {exchange_id}")
            except Exception as e:
                logger.error(f"Failed to initialize exchange {exchange_id}: {str(e)}")
        
        logger.info("Price system initialization complete")

    async def shutdown(self):
        """Close connections and perform cleanup"""
        logger.info("Shutting down price system...")
        self.running = False
        
        # Close exchange connections
        for exchange_id, exchange in self.exchanges.items():
            try:
                await exchange.close()
                logger.info(f"Closed connection to {exchange_id}")
            except Exception as e:
                logger.error(f"Error closing connection to {exchange_id}: {str(e)}")
        
        # Close MongoDB connection
        if self.db_client:
            self.db_client.close()
            logger.info("Closed MongoDB connection")
        
        logger.info("Price system shutdown complete")

    async def fetch_order_book(self, exchange_id: str, symbol: str, verbose: bool = False) -> Optional[dict]:
        """Fetch order book data from an exchange"""
        try:
            if verbose: logger.info(f"Fetching order book for {exchange_id}")
            exchange = self.exchanges[exchange_id]
            order_book = await exchange.fetch_order_book(symbol, limit=20)  # Fetch reasonable depth
            order_book['timestamp'] = time.time() * 1000  # Add timestamp in milliseconds
            order_book['exchange'] = exchange_id
            if verbose: logger.info(f"Got order book for {exchange_id}")
            return order_book
        except Exception as e:
            logger.warning(f"Failed to fetch order book from {exchange_id} for {symbol}: {str(e)}")
            return None

    def validate_feeds(self, order_books: list[dict]) -> list[dict]:
        """Filter out stale or anomalous order book data"""
        if not order_books:
            return []
        
        current_time = time.time() * 1000  # Current time in milliseconds
        valid_books = []
        
        # Calculate median mid price
        mid_prices = []
        for book in order_books:
            if book['bids'] and book['asks']:
                best_bid = book['bids'][0][0]
                best_ask = book['asks'][0][0]
                mid_price = (best_bid + best_ask) / 2
                mid_prices.append(mid_price)
        
        if not mid_prices:
            return []
        
        median_mid_price = np.median(mid_prices)
        
        # Filter order books
        for book in order_books:
            # Check if book has bids and asks
            if not book['bids'] or not book['asks']:
                continue
                
            # Check if data is stale
            if current_time - book['timestamp'] > STALE_THRESHOLD_SEC * 1000:
                logger.debug(f"Filtered out stale data from {book['exchange']}")
                continue
            
            # Check for crossed books
            best_bid = book['bids'][0][0]
            best_ask = book['asks'][0][0]
            if best_bid >= best_ask:
                logger.debug(f"Filtered out crossed book from {book['exchange']}")
                continue
            
            # Check for price deviation
            mid_price = (best_bid + best_ask) / 2
            deviation = abs(mid_price - median_mid_price) / median_mid_price
            if deviation > PRICE_DEVIATION_THRESHOLD:
                logger.debug(f"Filtered out anomalous price from {book['exchange']}: deviation={deviation:.2f}")
                continue
            
            valid_books.append(book)
        
        return valid_books

    def create_composite_book(self, valid_books: list[dict]) -> tuple[list, list]:
        """Combine order books into a single composite book"""
        
        composite_bids = []
        composite_asks = []
        
        try:
            # Collect all bids and asks
            for book in valid_books:
                try:
                    for info in book['bids']:
                        if len(info) == 2:
                            bid_price, bid_size = info
                        elif len(info) == 3:
                            bid_price, bid_size, _ = info
                        else:
                            continue
                        # Cap order size to $1 million
                        capped_size = min(bid_size * bid_price, MAX_ORDER_SIZE) / bid_price
                        composite_bids.append((float(bid_price), float(capped_size)))
                    
                    for info in book['asks']:
                        if len(info) == 2:
                            ask_price, ask_size = info
                        elif len(info) == 3:
                            ask_price, ask_size, _ = info
                        else:
                            continue
                        # Cap order size to $1 million
                        capped_size = min(ask_size * ask_price, MAX_ORDER_SIZE) / ask_price
                        composite_asks.append((float(ask_price), float(capped_size)))
                except Exception as e:
                    logger.error(f"Failed to process order book from {book['exchange']}: {str(e)}")
                    logger.error(f"Book data: {book}")
            
            # Sort bids in descending order by price
            composite_bids.sort(key=lambda x: x[0], reverse=True)
            # Sort asks in ascending order by price
            composite_asks.sort(key=lambda x: x[0])
        except Exception as e:
            logger.error(f"Failed to create composite book:\n{str(e)}")
            logger.error(f"Valid books: {valid_books}")
        
        return composite_bids, composite_asks

    def calculate_index_price(self, composite_bids: list, composite_asks: list) -> Optional[float]:
        """Calculate index price using the weighted average of marginal mid-prices"""
        if not composite_bids or not composite_asks:
            return None
            
        # Create cumulative size arrays for buy and sell
        cumulative_bids = []
        cumulative_size_bid = 0
        for price, size in composite_bids:
            cumulative_size_bid += size
            cumulative_bids.append((price, cumulative_size_bid))
        
        cumulative_asks = []
        cumulative_size_ask = 0
        for price, size in composite_asks:
            cumulative_size_ask += size
            cumulative_asks.append((price, cumulative_size_ask))
        
        # Define the maximum size for mid-price calculation
        max_size = min(cumulative_bids[-1][1], cumulative_asks[-1][1])
        if max_size <= 0:
            return None
            
        # Define the scaling factor L
        L = 1 / max_size
        
        # Define the sizes at which to calculate mid-prices
        # Use the union of cumulative buy and sell sizes
        all_sizes = sorted(set([size for _, size in cumulative_bids] + [size for _, size in cumulative_asks]))
        
        # Calculate marginal mid-prices and weights
        weighted_sum = 0
        weight_sum = 0
        
        for size in all_sizes:
            if size <= 0 or size > max_size:
                continue
                
            # Calculate marginal buy price at this size
            buy_price = next((price for price, cum_size in cumulative_bids if cum_size >= size), composite_bids[-1][0])
            
            # Calculate marginal sell price at this size
            sell_price = next((price for price, cum_size in cumulative_asks if cum_size >= size), composite_asks[-1][0])
            
            # Calculate mid-price
            mid_price = (buy_price + sell_price) / 2
            
            # Calculate weight using exponential distribution
            weight = L * np.exp(-L * size)
            
            weighted_sum += mid_price * weight
            weight_sum += weight
        
        if weight_sum <= 0:
            return None
            
        # Calculate final index price
        index_price = weighted_sum / weight_sum
        return index_price

    async def store_price_data(self, symbol: str, index_price: float, raw_data: list[dict], verbose: bool = False):
        """Store calculated index price and raw data in MongoDB time series collection"""
        timestamp = datetime.datetime.now(datetime.UTC)
        if verbose: logger.info(f"Storing price data! {symbol}: {index_price}")
        # Store calculated index price
        index_doc = {
            "timestamp": timestamp,
            "metadata": {
                "symbol": symbol,
                "type": "index_price"
            },
            "price": index_price,
            "exchanges_count": len(raw_data)
        }
        
        # Store raw exchange data (optional - can be disabled to save space)
        raw_docs = []
        for book in raw_data:
            exchange = book.get('exchange', 'unknown')
            
            # Calculate mid price for this exchange
            if book.get('bids') and book.get('asks'):
                best_bid = book['bids'][0][0]
                best_ask = book['asks'][0][0]
                mid_price = (best_bid + best_ask) / 2
                
                raw_doc = {
                    "timestamp": timestamp,
                    "metadata": {
                        "symbol": symbol,
                        "exchange": exchange,
                        "type": "exchange_data"
                    },
                    "price": mid_price,
                    "bid": best_bid,
                    "ask": best_ask
                }
                raw_docs.append(raw_doc)
        
        # Insert documents to MongoDB
        try:
            await self.collection.insert_one(index_doc)
            if raw_docs:
                await self.collection.insert_many(raw_docs)
            logger.debug(f"Stored price data for {symbol}: {index_price}")
        except Exception as e:
            logger.error(f"Failed to store price data: {str(e)}")

    async def process_symbol(self, symbol: str, verbose: bool = False):
        """Process a single trading symbol"""
        # Fetch order books from all exchanges
        order_books = []
        fetch_tasks = []
        
        for exchange_id in self.exchanges.keys():
            task = self.fetch_order_book(exchange_id, symbol)
            fetch_tasks.append(task)
        
        # Gather results
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, dict) and not isinstance(result, Exception):
                order_books.append(result)
        
        # Validate feeds
        valid_books = self.validate_feeds(order_books)
        if verbose: logger.info(f"Got valid books for {symbol}")
        
        # Check if we have enough valid feeds
        if len(valid_books) < MIN_VALID_FEEDS:
            logger.warning(f"Not enough valid price feeds for {symbol}: {len(valid_books)}/{MIN_VALID_FEEDS}")
            return
        
        index_price = formula.calculate_index_price(valid_books)

        if verbose: logger.info(f"Got index price for {symbol}: {index_price}")
        
        if index_price is not None:
            # Store the calculated price
            await self.store_price_data(symbol, index_price, valid_books, verbose=verbose)
            
            # Update last index price
            self.last_index_prices[symbol] = index_price
            
            logger.info(f"Index price for {symbol}: {index_price:.2f} (from {len(valid_books)} feeds)")
        else:
            logger.warning(f"Failed to calculate index price for {symbol}")

    async def run(self, symbols: list[str] = SYMBOLS):
        """Main loop for the price system"""
        self.running = True
        logger.info("Start running")
        
        while self.running:
            start_time = time.time()
            
            # Process each symbol in parallel
            tasks = [self.process_symbol(symbol) for symbol in symbols]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Calculate sleep time to maintain refresh interval
            elapsed = (time.time() - start_time) * 1000  # in milliseconds
            sleep_time = max(0, REFRESH_INTERVAL_MS - elapsed) / 1000  # in seconds
            
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
