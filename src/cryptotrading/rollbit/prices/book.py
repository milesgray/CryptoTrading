import time
import asyncio
from typing import Optional

import numpy as np
import ccxt.async_support as ccxt

from cryptotrading.data.factory import get_order_book_adapter
from cryptotrading.analysis.book import condense_order_book
from cryptotrading.util.book import order_book_to_df
from cryptotrading.util.status import StatusManager
from cryptotrading.config import (
    STALE_THRESHOLD_SEC,
    PRICE_DEVIATION_THRESHOLD
)

class OrderBookManager:
    def __init__(
        self,
        symbol: str,
        exchange_ids: list[str],
        status: StatusManager = StatusManager("order_book_manager")
    ):
        self.status = status
        self.data = get_order_book_adapter()
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
                self.status.info(f"[{self.symbol}] Initialized exchange: {exchange_id}")
            except Exception as e:
                self.status.error(f"[{self.symbol}] Failed to initialize exchange {exchange_id}: {str(e)}")
        
    async def shutdown(self):
        self.status.running = False
        
        # Close exchange connections
        for exchange_id, exchange in self.exchanges.items():
            try:
                await exchange.close()
                self.status.info(f"[{self.symbol}] Closed connection to {exchange_id}")
            except Exception as e:
                self.status.error(f"[{self.symbol}] Error closing connection to {exchange_id}: {str(e)}")        
        
        self.status.info(f"[{self.symbol}] Price system shutdown complete")
    
    async def fetch_order_book(
        self, 
        exchange_id: str, 
        verbose: bool = False
    ) -> Optional[dict]:
        """Fetch order book data from an exchange"""
        try:
            if verbose: 
                self.status.info(f"[{self.symbol}] Fetching order book for {exchange_id}")
            order_book = await self.exchanges[exchange_id].fetch_order_book(self.symbol, limit=20)  # Fetch reasonable depth
            order_book['timestamp'] = time.time() * 1000  # Add timestamp in milliseconds
            order_book['exchange'] = exchange_id
            if verbose: 
                self.status.info(f"[{self.symbol}] Got order book for {exchange_id}")
            return order_book
        except Exception as e:
            self.status.error(f"[{self.symbol}] Failed to fetch order book from {exchange_id}: {str(e)}")
            return None

    async def fetch(self, verbose: bool = False):
        order_books = []

        tasks = [self.fetch_order_book(exchange_id) for exchange_id in self.exchanges.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True) 

        for result in results:
            if isinstance(result, dict) and not isinstance(result, Exception):
                order_books.append(result)
        await self.data.store_exchange_order_book(self.symbol, order_books)
                    
        # Validate feeds
        self.valid_books = self.validate_feeds(order_books)
        if verbose:
            self.status.info(f"[{self.symbol}] Got valid books for {self.symbol}")
        return self.valid_books

    async def update(self, composite_order_book, verbose: bool = False):
        self.composite_order_book = composite_order_book      
        await self.data.store_composite_order_book_data(
            self.symbol, 
            composite_order_book, 
            self.valid_books, 
            verbose=verbose
        )
        # Create condensed book
        condensed_book = condense_order_book(
            composite_order_book['bids'],
            composite_order_book['asks'])   
        # Store transformed order book
        await self.data.store_transformed_order_book_data(
            self.symbol,
            composite_order_book,
            verbose=verbose
        ) 
        return condensed_book


    def validate_feeds(
        self,
        order_books: list[dict]
    ) -> list[dict]:
        """Filter out stale or anomalous order book data"""
        if not order_books:
            return []
        
        current_time = time.time() * 1000  # Current time in milliseconds
        valid_books = []
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
                self.status.debug(f"Filtered out stale data from {book['exchange']}")
                continue
            
            # Check for crossed books
            best_bid = book['bids'][0][0]
            best_ask = book['asks'][0][0]
            if best_bid >= best_ask:
                self.status.debug(f"Filtered out crossed book from {book['exchange']}")
                continue
            
            # Check for price deviation
            mid_price = (best_bid + best_ask) / 2
            deviation = abs(mid_price - median_mid_price) / median_mid_price
            if deviation > PRICE_DEVIATION_THRESHOLD:
                self.status.debug(f"Filtered out anomalous price from {book['exchange']}: deviation={deviation:.2f}")
                continue
            
            valid_books.append(book)
        
        return valid_books
