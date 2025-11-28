import os
import time
import logging
import asyncio
from typing import Optional
import numpy as np
import pandas as pd
import ccxt.async_support as ccxt

from cryptotrading.analysis.book import find_whale_positions
from cryptotrading.util.book import order_book_to_df
from cryptotrading.data.book import OrderBookMongoAdapter

STALE_THRESHOLD_SEC = int(os.getenv("STALE_THRESHOLD_SEC", 30))
PRICE_DEVIATION_THRESHOLD = float(os.getenv("PRICE_DEVIATION_THRESHOLD", 0.1))
MIN_VALID_FEEDS = int(os.getenv("MIN_VALID_FEEDS", 6))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('order_book_manager')


class OrderBookManager:
    def __init__(self, symbol: str, exchange_ids: list[str]):
        self.data = OrderBookMongoAdapter()
        self.running = False
        self.symbol = symbol
        self.exchange_ids = exchange_ids
        self.exchanges = {}
        self.last_index_prices = {}
        self.last_price_times = {}
        self.valid_books = []
        self.composite_order_book = None

    async def initialize(self):
        await self.data.initialize()
        
        for exchange_id in self.exchange_ids:
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
                logger.info(f"[{self.symbol}] Initialized exchange: {exchange_id}")
            except Exception as e:
                logger.error(f"[{self.symbol}] Failed to initialize exchange {exchange_id}: {str(e)}")
        
    async def shutdown(self):
        self.running = False
        
        # Close exchange connections
        for exchange_id, exchange in self.exchanges.items():
            try:
                await exchange.close()
                logger.info(f"[{self.symbol}] Closed connection to {exchange_id}")
            except Exception as e:
                logger.error(f"[{self.symbol}] Error closing connection to {exchange_id}: {str(e)}")        
        
        logger.info(f"[{self.symbol}] Price system shutdown complete")
    
    async def fetch_order_book(
        self, 
        exchange_id: str, 
        verbose: bool = False
    ) -> Optional[dict]:
        """Fetch order book data from an exchange"""
        try:
            if verbose: logger.info(f"[{self.symbol}] Fetching order book for {exchange_id}")
            exchange = self.exchanges[exchange_id]
            order_book = await exchange.fetch_order_book(self.symbol, limit=20)  # Fetch reasonable depth
            order_book['timestamp'] = time.time() * 1000  # Add timestamp in milliseconds
            order_book['exchange'] = exchange_id
            if verbose: logger.info(f"[{self.symbol}] Got order book for {exchange_id}")
            yield order_book
        except Exception as e:
            logger.warning(f"[{self.symbol}] Failed to fetch order book from {exchange_id}: {str(e)}")
            yield None

    async def fetch(self, verbose: bool = False):
        order_books = []

        tasks = [self.fetch_order_book(exchange_id) for exchange_id in self.exchanges.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True) 

        for result in results:
            if isinstance(result, dict) and not isinstance(result, Exception):
                order_books.append(result)
        await self.data.store_exchange_order_book(self.symbol, order_books)
                    
        # Validate feeds
        self.valid_books = validate_feeds(order_books)
        if verbose: logger.info(f"[{self.symbol}] Got valid books for {self.symbol}")
        return self.valid_books

    async def update(self, composite_order_book, verbose: bool = False):
        self.composite_order_book = composite_order_book
        composite_order_book_df = order_book_to_df(
            composite_order_book['bids'], 
            composite_order_book['asks'])
        condensed_book = condense_order_book(
            composite_order_book_df)   

        await self.data.store_composite_order_book_data(
            self.symbol, 
            composite_order_book, 
            self.valid_books, 
            verbose=verbose
        )             
        await self.data.store_transformed_order_book_data(
            self.symbol,
            composite_order_book,
            verbose=verbose
        ) 
        return condensed_book


def validate_feeds(
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

            book_df = order_book_to_df(book['bids'], book['asks'])
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
    df: pd.DataFrame, 
    column: str, 
    limit: Optional[int] = None, 
    direction: str = 'both'
) -> pd.DataFrame:
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
    df: pd.DataFrame,
    num_buckets: int = 10, 
    size_column: str = 'size', 
    price_column: str = 'price',
    side_column: str = 'side'
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
    ask_filter = df[side_column] == 'a'
    bid_filter = df[side_column] == 'b'
    size_filter = df[size_column] > 0

    asks = df[ask_filter & size_filter]
    bids = df[bid_filter & size_filter]
    
    # Ensure required columns exist
    required_columns = [size_column, price_column]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Data must contain columns: {required_columns}")
    
    
    # Function to bucket non-outlier data
    def create_buckets(df: pd.DataFrame, column: str, num_buckets: int) -> tuple[list[tuple[str, float, float]], list[tuple[float, float]]]:
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
