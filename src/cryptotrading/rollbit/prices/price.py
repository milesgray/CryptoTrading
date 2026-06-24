import asyncio
import time
import logging
import traceback
import datetime as dt

import cryptotrading.rollbit.prices.formula as formula
from cryptotrading.rollbit.prices.book import OrderBookManager
from cryptotrading.data.factory import get_price_adapter
from cryptotrading.analysis.levels import PriceLevels
from cryptotrading.util.status import StatusManager
from cryptotrading.config import (
    SPOT_EXCHANGES, 
    FUTURES_EXCHANGES,
    MIN_VALID_FEEDS,
    SYMBOLS
)

class PriceSystem:
    def __init__(
        self, 
        symbols: list[str] = SYMBOLS, 
        enable_levels: bool = True,
        status: StatusManager = StatusManager('price_system')
    ):
        self.status = status
        self.data = get_price_adapter()
        self.enable_levels = enable_levels        
        self.levels = {symbol: PriceLevels() for symbol in symbols} if self.enable_levels else None
        self.books = {symbol: OrderBookManager(symbol, SPOT_EXCHANGES + FUTURES_EXCHANGES, status=status) for symbol in symbols}                
        self.last_index_prices = {}
        self.last_price_times = {}

    async def initialize(self):
        """Initialize exchange connections and database"""
        self.status.info("Initializing price system...")
        
        # Initialize MongoDB connection
        await self.data.initialize()
        # Initialize exchange connections
        for book in self.books.values():
            await book.initialize()
        self.status.info("Price system initialization complete")
        self.status.running = True

    async def shutdown(self):
        """Close connections and perform cleanup"""
        self.status.info("Shutting down price system...")
        self.status.running = False
        await asyncio.gather(*[book.shutdown() for book in self.books.values()])

        # Close MongoDB connection
        await self.data.shutdown()
        
        self.status.info("Price system shutdown complete")
    
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
                self.status.warning(f"Not enough valid price feeds for {symbol}: {len(valid_books)}/{MIN_VALID_FEEDS}")
                return
            
            calc_results = formula.calculate_index_price(
                valid_books, 
                min_valid_feeds=MIN_VALID_FEEDS, 
                return_book=True,
                status=self.status
            )

            index_price = calc_results["price"]
            if verbose: 
                self.status.info(f"Got index price for {symbol}: {index_price}")                    

            condensed_book = await self.books[symbol].update(calc_results["book"])

            if index_price is not None:
                # add price and volume to level tracker
                if self.enable_levels:
                    self.levels[symbol].add_price_point(dt.datetime.now(dt.timezone.utc), index_price, volume=calc_results["size"])
                # Store the calculated price
                await self.data.store_price_data(
                        symbol, index_price, condensed_book, valid_books, 
                    verbose=verbose)
                
                # Update last index price
                self.last_index_prices[symbol] = index_price
                self.status.update_data({"last_index_prices": self.last_index_prices})   
                if symbol in self.last_price_times:         
                    delta = time.time() - self.last_price_times[symbol]

                    if self.status.verbose:
                        self.status.info(f"Index price for {symbol}: {index_price:.2f} (from {len(valid_books)} feeds, in {delta:0.2f} seconds)")
                else:
                    if self.status.verbose:
                        self.status.info(f"Index price for {symbol}: {index_price:.2f} (from {len(valid_books)} feeds)")
                self.last_price_times[symbol] = time.time()
                self.status.update_data({"last_price_times": self.last_price_times})   
            else:
                self.status.warning(f"Failed to calculate index price for {symbol}")
        except Exception as e:
            # include stack trace
            self.status.error(traceback.format_exc())
            self.status.error(f"Error processing symbol {symbol}: {str(e)}")

    async def run(self) -> None:
        """Main logic for the price system, gets a single price point"""
        if self.status.running:               
            try:                                
                # Process each symbol in parallel
                tasks = [self.process_symbol(symbol) for symbol in self.books.keys()]
                await asyncio.gather(*tasks, return_exceptions=True)            
            except asyncio.exceptions.CancelledError:
                self.status.info("Cancelled by user")
            except KeyboardInterrupt:
                self.status.info("Keyboard interrupt received, shutting down...")
                await self.shutdown()
            except Exception as e:
                self.status.error(f"Unexpected error: {str(e)}")

