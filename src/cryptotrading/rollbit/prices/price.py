import os
import asyncio
import time
import logging
import numpy as np
from typing import Optional, Any
import datetime

import ccxt.async_support as ccxt
import pandas as pd
import numpy as np

import cryptotrading.rollbit.prices.formula as formula
from cryptotrading.rollbit.prices.book import OrderBookManager
from cryptotrading.data.price import PriceMongoAdapter
from cryptotrading.config import (
    SPOT_EXCHANGES, 
    FUTURES_EXCHANGES,
    MIN_VALID_FEEDS,
)

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('price_system')

class PriceSystem:
    def __init__(self, symbols: list[str]):
        self.data = PriceMongoAdapter()
        self.books = {symbol: OrderBookManager(symbol, SPOT_EXCHANGES + FUTURES_EXCHANGES) for symbol in symbols}
        self.running = False
        self.last_index_prices = {}
        self.last_price_times = {}

    async def initialize(self):
        """Initialize exchange connections and database"""
        logger.info("Initializing price system...")
        
        # Initialize MongoDB connection
        await self.data.initialize()
        # Initialize exchange connections
        for book in self.books.values():
            await book.initialize()
        logger.info("Price system initialization complete")
        self.running = True

    async def shutdown(self):
        """Close connections and perform cleanup"""
        logger.info("Shutting down price system...")
        self.running = False
        await asyncio.gather(*[book.shutdown() for book in self.books.values()])

        # Close MongoDB connection
        await self.data.shutdown()
        
        logger.info("Price system shutdown complete")
    
    async def process_symbol(
        self, 
        symbol: str, 
        verbose: bool = False
    ):
        """Process a single trading symbol"""
        try:
            # Fetch order books from all exchanges
            valid_books = await self.books[symbol].fetch()
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

            condensed_book = await self.books[symbol].update(calc_results["book"])

            if index_price is not None:
                # Store the calculated price
                await self.data.store_price_data(
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

    async def run(self) -> None:
        """Main logic for the price system, gets a single price point"""
        if self.running:               
            try:                                
                # Process each symbol in parallel
                tasks = [self.process_symbol(symbol) for symbol in self.books.keys()]
                await asyncio.gather(*tasks, return_exceptions=True)            
            except asyncio.exceptions.CancelledError:
                logger.info("Cancelled by user")
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, shutting down...")
                self.shutdown()
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")

