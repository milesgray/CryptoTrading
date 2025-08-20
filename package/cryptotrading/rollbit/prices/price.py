import os
import asyncio
import time
import logging
import numpy as np
from typing import Optional, Any
import datetime
from pymongo import ASCENDING
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()

import cryptotrading.rollbit.prices.formula as formula
from cryptotrading.analysis.book import find_whale_positions
from cryptotrading.data.mongo import get_db, \
    PRICE_COLLECTION_NAME, \
    COMPOSITE_ORDER_BOOK_COLLECTION_NAME, \
    EXCHANGE_ORDER_BOOK_COLLECTION_NAME

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('price_system')

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
REFRESH_INTERVAL_MS = int(os.getenv("REFRESH_INTERVAL_MS", 500))
STALE_THRESHOLD_SEC = int(os.getenv("STALE_THRESHOLD_SEC", 30))
PRICE_DEVIATION_THRESHOLD = float(os.getenv("PRICE_DEVIATION_THRESHOLD", 0.1))
MIN_VALID_FEEDS = int(os.getenv("MIN_VALID_FEEDS", 6))
MAX_ORDER_SIZE = int(os.getenv("MAX_ORDER_SIZE", 1_000_000))

class PriceSystem:
    def __init__(self):
        self.exchanges = {}
        self.db_client = None
        self.db = None
        self.collection = None
        self.running = False
        self.last_index_prices = {}
        self.last_price_times = {}

    async def initialize(self):
        """Initialize exchange connections and database"""
        logger.info("Initializing price system...")
        
        # Initialize MongoDB connection        
        self.db = get_db()
        self.collection = self.db[PRICE_COLLECTION_NAME]
        self.composite_order_book_collection = self.db[COMPOSITE_ORDER_BOOK_COLLECTION_NAME]
        self.exchange_order_book_collection = self.db[EXCHANGE_ORDER_BOOK_COLLECTION_NAME]
        self.transformed_order_book_collection = self.db[TRANSFORMED_ORDER_BOOK_COLLECTION_NAME]
        
        # Create time series collection if it doesn't exist
        collections = await self.db.list_collection_names()
        if PRICE_COLLECTION_NAME not in collections:
            try:
                await self.db.create_collection(
                    PRICE_COLLECTION_NAME,
                    timeseries={
                        'timeField': 'timestamp',
                        'metaField': 'price',
                        'granularity': 'seconds'
                    }
                )
            except:
                await self.db.create_collection(PRICE_COLLECTION_NAME)            
            # Create indexes for faster queries
            await self.collection.create_index([("timestamp", ASCENDING)])
            await self.collection.create_index([("metadata.symbol", ASCENDING)])
        if COMPOSITE_ORDER_BOOK_COLLECTION_NAME not in collections:
            try:
                await self.db.create_collection(
                    COMPOSITE_ORDER_BOOK_COLLECTION_NAME,
                    timeseries={
                        'timeField': 'timestamp',
                        'metaField': 'metadata',
                        'granularity': 'seconds'
                    }
                )
            except:
                await self.db.create_collection(COMPOSITE_ORDER_BOOK_COLLECTION_NAME)
            # Create indexes for faster queries
            await self.composite_order_book_collection.create_index([("timestamp", ASCENDING)])
            await self.composite_order_book_collection.create_index([("metadata.symbol", ASCENDING)])            
        if EXCHANGE_ORDER_BOOK_COLLECTION_NAME not in collections:
            try:
                await self.db.create_collection(
                    EXCHANGE_ORDER_BOOK_COLLECTION_NAME,
                    timeseries={
                        'timeField': 'timestamp',
                        'metaField': 'metadata',
                        'granularity': 'seconds'
                    }
                )
            except:
                await self.db.create_collection(EXCHANGE_ORDER_BOOK_COLLECTION_NAME)
            # Create indexes for faster queries
            await self.exchange_order_book_collection.create_index([("timestamp", ASCENDING)])
            await self.exchange_order_book_collection.create_index([("metadata.symbol", ASCENDING)])
            await self.exchange_order_book_collection.create_index([("metadata.exchange", ASCENDING)])
        if TRANSFORMED_ORDER_BOOK_COLLECTION_NAME not in collections:
            try:
                await self.db.create_collection(
                    TRANSFORMED_ORDER_BOOK_COLLECTION_NAME,
                    timeseries={
                        'timeField': 'timestamp',
                        'metaField': 'metadata',
                        'granularity': 'seconds'
                    }
                )
            except:
                await self.db.create_collection(TRANSFORMED_ORDER_BOOK_COLLECTION_NAME)
            # Create indexes for faster queries
            await self.transformed_order_book_collection.create_index([("timestamp", ASCENDING)])
            await self.transformed_order_book_collection.create_index([("metadata.symbol", ASCENDING)])            
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

    async def fetch_order_book(
        self, 
        exchange_id: str, 
        symbol: str, 
        verbose: bool = False
    ) -> Optional[dict]:
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

    def order_book_to_df(
        self, 
        bids, 
        asks, 
        side_column='side',
        price_column='price',
        size_column='size'
    ) -> pd.DataFrame:
        # Convert to DataFrames if lists are provided
        if isinstance(bids, list | np.ndarray):
            if isinstance(bids[0], list | tuple):
                bids = pd.DataFrame({
                    price_column: [b[0] for b in bids], 
                    size_column: [b[1] for b in bids],
                    side_column: ["b" for b in bids]})
            else:
                bids = pd.DataFrame(bids)            
        if isinstance(asks, list | np.ndarray):
            if isinstance(asks[0], list | tuple):
                asks = pd.DataFrame({
                    price_column: [a[0] for a in asks], 
                    size_column: [a[1] for a in asks], 
                    side_column: ["a" for a in asks]})
            else:
                asks = pd.DataFrame(asks)
        
        return pd.concat([bids, asks])

    def validate_feeds(
        self, 
        order_books: list[dict]
    ) -> list[dict]:
        """Filter out stale or anomalous order book data"""
        if not order_books:
            return []
        
        current_time = time.time() * 1000  # Current time in milliseconds
        valid_books = []
        book_dfs = {}
        
        # Calculate median mid price
        mid_prices = []
        spreads = []
        for book in order_books:
            if book['bids'] and book['asks']:
                best_bid = book['bids'][0][0]
                best_ask = book['asks'][0][0]
                mid_price = (best_bid + best_ask) / 2                
                mid_prices.append(mid_price)

                book_df = self.order_book_to_df(book['bids'], book['asks'])
                ask_filter = book_df['side'] == 'a'
                bid_filter = book_df['side'] == 'b'
                size_filter = book_df['size'] > 0
                highest_bid = book_df[ask_filter & size_filter]['price'].max()
                lowest_ask = book_df[bid_filter & size_filter]['price'].min()
                                
                spreads.append(abs(lowest_ask - highest_bid))
                book_dfs[book['exchange']] = book_df
        
        if not mid_prices:
            return []
        
        median_mid_price = np.median(mid_prices)
        median_spread = np.median(spreads)
        
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
    
    def find_outliers(
        self, 
        df, 
        column, 
        limit=None, 
        direction='both'
    ):
        """
        Find outliers in a DataFrame column with the option to limit the number returned.
        
        Parameters:
        df (pandas.DataFrame): DataFrame containing the data
        column (str): Column name to check for outliers
        limit (int, optional): Maximum number of outliers to return. If None, returns all outliers.
        direction (str, optional): Which outliers to find - 'both', 'upper', or 'lower'
        
        Returns:
        pandas.DataFrame: DataFrame containing the outliers
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Find outliers based on direction parameter
        if direction == 'both':
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].copy()
        elif direction == 'upper':
            outliers = df[df[column] > upper_bound].copy()
        elif direction == 'lower':
            outliers = df[df[column] < lower_bound].copy()
        else:
            raise ValueError("direction must be 'both', 'upper', or 'lower'")
        
        # Sort outliers by distance from the median to get the most extreme ones first
        median = df[column].median()
        outliers['distance'] = abs(outliers[column] - median)
        outliers = outliers.sort_values('distance', ascending=False)
        
        # Remove the distance column before returning
        outliers = outliers.drop('distance', axis=1)
        
        # Limit the number of outliers if specified
        if limit is not None and limit > 0:
            return outliers.head(limit)
        
        return outliers

    def condense_order_book(
        self, 
        df,
        num_buckets=10, 
        size_column='size', 
        price_column='price',
        side_column='side'
    ) -> dict:
        """
        Condense order book data into buckets distributed by size.
        
        Parameters:
        -----------
        df: pandas.DataFrame
            DataFrame containing the order book data
        num_buckets : int
            Number of buckets to create
        size_column : str
            Column name for the size values
        price_column : str
            Column name for the price values
            
        Returns:
        --------
        dict containing:
            'bid_buckets': List of (price_range, avg_price, total_size) for each bid bucket
            'ask_buckets': List of (price_range, avg_price, total_size) for each ask bucket
            'bid_outliers': List of (price, size) for outlier bids
            'ask_outliers': List of (price, size) for outlier asks
        """
        ask_filter = df['side'] == 'a'
        bid_filter = df['side'] == 'b'
        size_filter = df['size'] > 0

        asks = df[ask_filter & size_filter]
        bids = df[bid_filter & size_filter]
        
        # Ensure required columns exist
        required_columns = [size_column, price_column]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        
        # Function to bucket non-outlier data
        def create_buckets(df, column, num_buckets):
            if df.empty:
                return []
            
            # Remove outliers
            outliers = find_whale_positions(df, column, limit=5, min_size_multiplier=10)
            non_outliers = df[~df.index.isin(outliers.index)].copy()
            
            if non_outliers.empty:
                return [], outliers[[price_column, size_column]].values.tolist()
            
            # Create buckets based on size distribution
            size_distribution = non_outliers[column].values
            
            # If we have fewer unique values than buckets, adjust num_buckets
            unique_sizes = len(np.unique(size_distribution))
            actual_num_buckets = min(num_buckets, unique_sizes)
            
            if actual_num_buckets <= 1:
                # If only one bucket, put everything in it
                avg_price = non_outliers[price_column].mean()
                total_size = non_outliers[size_column].sum()
                min_price = non_outliers[price_column].min()
                max_price = non_outliers[price_column].max()
                buckets = [(f"{min_price:.2f}-{max_price:.2f}", avg_price, total_size)]
            else:
                # Use percentile-based bucketing for more even distribution
                percentiles = np.linspace(0, 100, actual_num_buckets + 1)
                bucket_thresholds = np.percentile(size_distribution, percentiles)
                
                buckets = []
                for i in range(actual_num_buckets):
                    lower = bucket_thresholds[i]
                    upper = bucket_thresholds[i + 1]
                    
                    # Handle edge case for the last bucket to include the maximum value
                    if i == actual_num_buckets - 1:
                        bucket_items = non_outliers[(non_outliers[column] >= lower) & 
                                                (non_outliers[column] <= upper)]
                    else:
                        bucket_items = non_outliers[(non_outliers[column] >= lower) & 
                                                (non_outliers[column] < upper)]
                    
                    if not bucket_items.empty:
                        avg_price = bucket_items[price_column].mean()
                        total_size = bucket_items[size_column].sum()
                        min_price = bucket_items[price_column].min()
                        max_price = bucket_items[price_column].max()
                        buckets.append((f"{min_price:.2f}-{max_price:.2f}", avg_price, total_size))
            
            return buckets, outliers[[price_column, size_column]].values.tolist()
        
        # Process bids and asks
        bid_buckets, bid_outliers = create_buckets(bids, size_column, num_buckets)
        ask_buckets, ask_outliers = create_buckets(asks, size_column, num_buckets)
        
        return {
            'bid_buckets': bid_buckets,
            'ask_buckets': ask_buckets,
            'bid_outliers': bid_outliers,
            'ask_outliers': ask_outliers
        }

    async def store_price_data(
        self, 
        symbol: str, 
        index_price: float, 
        book: dict,            
        raw_data: list[dict], 
        verbose: bool = False
    ) -> None:
        """Store calculated index price and raw data in MongoDB time series collection"""
        timestamp = datetime.datetime.now(datetime.UTC)
        if verbose: logger.info(f"Storing price data! {symbol}: {index_price}")
        token = symbol.split("/")[0] if "/" in symbol else symbol
        # Store calculated index price
        index_doc = {
            "timestamp": timestamp,
            "metadata": {
                "token": token,
                "symbol": symbol,
                "book": book,                
                "type": "index_price",
            },
            "price": index_price,
            "exchanges_count": len(raw_data)
        }

        try:
            await self.collection.insert_one(index_doc)
            
            logger.debug(f"Stored price data for {symbol}: {index_price}")
        except Exception as e:
            logger.error(f"Failed to store price data: {str(e)}")
        
    async def store_exchange_order_book(
            self, 
            symbol: str,            
            raw_data: list[dict], 
            verbose: bool = False
    ) -> None:
        """Store calculated index price and raw data in MongoDB time series collection"""
        timestamp = datetime.datetime.now(datetime.UTC)
        if verbose: logger.info(f"Storing order book data! {symbol}: {len(raw_data)} feeds")
        token = symbol.split("/")[0] if "/" in symbol else symbol
        # Store raw exchange data (optional - can be disabled to save space)
        raw_docs = []
        for book in raw_data:
            exchange = book.get('exchange', 'unknown')
            
            # Calculate mid price for this exchange
            if book.get('bids') and book.get('asks'):
                df = self.order_book_to_df(book['bids'], book['asks'])
                
                ask_filter = df['side'] == 'a'
                bid_filter = df['side'] == 'b'
                size_filter = df['size'] > 0

                asks = df[ask_filter & size_filter]
                bids = df[bid_filter & size_filter]

                lowest_ask = asks['price'].min()
                highest_ask = asks['price'].max()
                
                lowest_bid = bids['price'].min()
                highest_bid = bids['price'].max()
                bid_range = highest_bid - lowest_bid
                ask_range = highest_ask - lowest_ask

                spread = abs(lowest_ask - highest_bid)

                mid_price = (lowest_ask + highest_bid) / 2

                book['bids'].sort(key=lambda x: x[1], reverse=True)
                book['asks'].sort(key=lambda x: x[1], reverse=True)

                highest_volume_bid = book['bids'][0]
                highest_volume_ask = book['asks'][0]

                total_bid_size = bids["size"].sum()
                total_ask_size = asks["size"].sum()
                
                raw_doc = {
                    "timestamp": timestamp,
                    "metadata": {
                        "token": token,
                        "symbol": symbol,
                        "exchange": exchange,
                        "spread": spread, 
                        "price": mid_price,
                        "lowest_bid": lowest_bid,                        
                        "highest_bid": highest_bid,
                        "total_bid_size": total_bid_size,
                        "bid_range": bid_range,
                        "highest_volume_bid_price": highest_volume_bid[0],
                        "highest_volume_bid_vol": highest_volume_bid[1],
                        "lowest_ask": lowest_ask,
                        "highest_ask": highest_ask,                        
                        "ask_range": ask_range,                        
                        "total_ask_size": total_ask_size,                        
                        "highest_volume_ask_price": highest_volume_ask[0],
                        "highest_volume_ask_vol": highest_volume_ask[1],
                        "type": "exchange_data"
                    },
                    "book": book,                    
                    
                }
                raw_docs.append(raw_doc)
        
        # Insert documents to MongoDB
        try:            
            if raw_docs:
                await self.exchange_order_book_collection.insert_many(raw_docs)
            logger.debug(f"Stored exchange order book data for {symbol}")
        except Exception as e:
            logger.error(f"Failed to store exchange price data: {str(e)}")

    async def store_composite_order_book_data(
        self, 
        symbol: str, 
        book: dict, 
        raw_data: list[dict],
        verbose: bool=False
    ) -> None:
        """Store calculated index price and raw composite order book data in MongoDB time series collection"""
        timestamp = datetime.datetime.now(datetime.UTC)
        if verbose: logger.info(f"Storing order book data! {symbol}")
        token = symbol.split("/")[0] if "/" in symbol else symbol

        df = self.order_book_to_df(book['bids'], book['asks'])
        
        ask_filter = df['side'] == 'a'
        bid_filter = df['side'] == 'b'
        size_filter = df['size'] > 0

        asks = df[ask_filter & size_filter]
        bids = df[bid_filter & size_filter]

        lowest_ask = asks['price'].min()
        highest_ask = asks['price'].max()
        
        lowest_bid = bids['price'].min()
        highest_bid = bids['price'].max()

        spread = abs(lowest_ask - highest_bid)

        midpoint = (lowest_ask + highest_bid) / 2

        book['bids'].sort(key=lambda x: x[1], reverse=True)
        book['asks'].sort(key=lambda x: x[1], reverse=True)

        largest_size_bid = book['bids'][0]
        largest_size_ask = book['asks'][0]

        total_bid_size = bids["size"].sum()
        total_ask_size = asks["size"].sum()

        book['bids'].sort(key=lambda x: x[0], reverse=True)
        book['asks'].sort(key=lambda x: x[0], reverse=False)
        
        # Store calculated index price
        index_doc = {
            "timestamp": timestamp,
            "metadata": {
                "token": token,
                "symbol": symbol,                
                "lowest_ask": lowest_ask,
                "highest_ask": highest_ask,
                "largest_size_ask": largest_size_ask,
                "total_ask_size": total_ask_size,
                "lowest_bid": lowest_bid,
                "highest_bid": highest_bid,
                "largest_size_bid": largest_size_bid,
                "total_bid_size": total_bid_size,
                "midpoint": midpoint,
                "spread": spread,
                "type": "order_book",
            },
            "book": book,
            "exchanges_count": len(raw_data)
        }

        try:
            await self.composite_order_book_collection.insert_one(index_doc)
            
            logger.debug(f"Stored price data for {symbol}")
        except Exception as e:
            logger.error(f"Failed to store price data: {str(e)}")
    
    async def store_transformed_order_book_data(
        self, 
        symbol: str, 
        book: dict, 
        verbose: bool=False
    ) -> None:
        """Store calculated index price and raw composite order book data in MongoDB time series collection"""
        timestamp = datetime.datetime.now(datetime.UTC)
        if verbose: logger.info(f"Storing order book data! {symbol}")
        token = symbol.split("/")[0] if "/" in symbol else symbol

        df = self.order_book_to_df(book['bids'], book['asks'])
        
        ask_filter = df['side'] == 'a'
        bid_filter = df['side'] == 'b'
        size_filter = df['size'] > 0

        asks = df[ask_filter & size_filter]
        bids = df[bid_filter & size_filter]

        lowest_ask = asks['price'].min()
        highest_bid = bids['price'].max()

        spread = abs(lowest_ask - highest_bid)

        midpoint = (lowest_ask + highest_bid) / 2

        # Store calculated index price
        index_doc = {
            "timestamp": timestamp,
            "metadata": {
                "token": token,
                "symbol": symbol,                
                "lowest_ask": lowest_ask,
                "highest_bid": highest_bid,
                "midpoint": midpoint,
                "spread": spread,
                "type": "order_book",
            },            
        }

        try:
            await self.transformed_order_book_collection.insert_one(index_doc)
            
            logger.debug(f"Stored transformed order book data for {symbol}")
        except Exception as e:
            logger.error(f"Failed to store transformed order book data: {str(e)}")

    async def process_symbol(
        self, 
        symbol: str, 
        verbose: bool = False
    ):
        """Process a single trading symbol"""
        try:
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
            await self.store_exchange_order_book(symbol, order_books)
                        
            # Validate feeds
            valid_books = self.validate_feeds(order_books)
            if verbose: logger.info(f"Got valid books for {symbol}")
            
            # Check if we have enough valid feeds
            if len(valid_books) < MIN_VALID_FEEDS:
                logger.warning(f"Not enough valid price feeds for {symbol}: {len(valid_books)}/{MIN_VALID_FEEDS}")
                return
            
            calc_results = formula.calculate_index_price(
                valid_books, 
                min_valid_feeds=MIN_VALID_FEEDS, 
                return_book=True,
                logger=logger
            )

            index_price = calc_results["price"]
            if verbose: logger.info(f"Got index price for {symbol}: {index_price}")

            composite_order_book = calc_results["book"]   
            composite_order_book_df = self.order_book_to_df(
                composite_order_book['bids'], 
                composite_order_book['asks'])
            condensed_book = self.condense_order_book(
                composite_order_book_df)   

            await self.store_composite_order_book_data(
                symbol, 
                composite_order_book, 
                valid_books, 
                verbose=verbose
            )             
            await self.store_transformed_order_book_data(
                symbol,
                composite_order_book,
                verbose=verbose
            ) 

            if index_price is not None:
                # Store the calculated price
                await self.store_price_data(
                    symbol, index_price, condensed_book, valid_books, 
                    verbose=verbose)
                
                # Update last index price
                self.last_index_prices[symbol] = index_price   
                if symbol in self.last_price_times:         
                    dt = time.time() - self.last_price_times[symbol]

                    logger.info(f"Index price for {symbol}: {index_price:.2f} (from {len(valid_books)} feeds, in {dt:0.2f} seconds)")
                else:
                    logger.info(f"Index price for {symbol}: {index_price:.2f} (from {len(valid_books)} feeds)")
                self.last_price_times[symbol] = time.time()
            else:
                logger.warning(f"Failed to calculate index price for {symbol}")
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {str(e)}")

    async def run(
        self, 
        symbols: list[str] = SYMBOLS
    ):
        """Main loop for the price system"""
        self.running = True
        logger.info(f"Start running: {symbols}")
        
        while self.running:
            start_time = time.time()
            try:                                
                # Process each symbol in parallel
                tasks = [self.process_symbol(symbol) for symbol in symbols]
                await asyncio.gather(*tasks, return_exceptions=True)            
            except asyncio.exceptions.CancelledError:
                logger.info("Cancelled by user")
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, shutting down...")
                self.shutdown()
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
            finally:
                # Calculate sleep time to maintain refresh interval
                elapsed = (time.time() - start_time) * 1000  # in milliseconds
                sleep_time = max(0, REFRESH_INTERVAL_MS - elapsed) / 1000  # in seconds
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
