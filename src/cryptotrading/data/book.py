import logging
import datetime as dt
from typing import Optional, Any

from pymongo import ASCENDING

from cryptotrading.data.mongo import get_db
from cryptotrading.data.models import OrderBookSummaryData, PriceBucket, PriceOutlier
from cryptotrading.config import (
    COMPOSITE_ORDER_BOOK_COLLECTION_NAME,
    EXCHANGE_ORDER_BOOK_COLLECTION_NAME,
    TRANSFORMED_ORDER_BOOK_COLLECTION_NAME,
)
from cryptotrading.rollbit.prices.book import order_book_to_df

logger = logging.getLogger(__name__)

class OrderBookMongoAdapter:
    def __init__(self):
        self.db = get_db()

    async def initialize(self):
        self.collections = await self.db.list_collection_names()        
        await self.init_composite_order_book_collection()
        await self.init_exchange_order_book_collection()
        await self.init_transformed_order_book_collection()
    
    async def shutdown(self):
        pass
    
    async def init_composite_order_book_collection(self):
        self.composite_order_book_collection = self.db[COMPOSITE_ORDER_BOOK_COLLECTION_NAME]
        
        # Create time series collection if it doesn't exist
        if COMPOSITE_ORDER_BOOK_COLLECTION_NAME not in self.collections:
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

    async def init_exchange_order_book_collection(self):
        self.exchange_order_book_collection = self.db[EXCHANGE_ORDER_BOOK_COLLECTION_NAME]
        
        # Create time series collection if it doesn't exist
        if EXCHANGE_ORDER_BOOK_COLLECTION_NAME not in self.collections:
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
    
    async def init_transformed_order_book_collection(self): 
        self.transformed_order_book_collection = self.db[TRANSFORMED_ORDER_BOOK_COLLECTION_NAME]
        
        # Create time series collection if it doesn't exist
        if TRANSFORMED_ORDER_BOOK_COLLECTION_NAME not in self.collections:
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

    async def store_exchange_order_book(
            self, 
            symbol: str,            
            raw_data: list[dict], 
            verbose: bool = False
    ) -> None:
        """Store calculated index price and raw data in MongoDB time series collection"""
        timestamp = dt.datetime.now(dt.UTC)
        if verbose: logger.infPriceSystem
        token = symbol.split("/")[0] if "/" in symbol else symbol
        raw_docs = []
        for book in raw_data:
            exchange = book.get('exchange', 'unknown')
            
            # Calculate mid price for this exchange
            if book.get('bids') and book.get('asks'):
                df = order_book_to_df(book['bids'], book['asks'])
                
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
        timestamp = dt.datetime.now(dt.UTC)
        if verbose: logger.info(f"Storing order book data! {symbol}")
        token = symbol.split("/")[0] if "/" in symbol else symbol

        df = order_book_to_df(book['bids'], book['asks'])
        
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
        timestamp = dt.datetime.now(dt.UTC)
        if verbose: logger.info(f"Storing order book data! {symbol}")
        token = symbol.split("/")[0] if "/" in symbol else symbol

        df = order_book_to_df(book['bids'], book['asks'])
        
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

    @staticmethod
    def process_order_book_data(book_data: dict[str, Any]) -> OrderBookSummaryData:
        """Convert raw order book data into structured OrderBookSummaryData."""
        result = {
            "bid_buckets": [],
            "ask_buckets": [],
            "bid_outliers": [],
            "ask_outliers": [],
            "volume": 0
        }
        
        if not book_data:
            return result
            
        # Process bid buckets
        if "bid_buckets" in book_data and book_data["bid_buckets"]:
            result["bid_buckets"] = [
                PriceBucket(
                    range=bucket[0],
                    avg_price=bucket[1],
                    volume=bucket[2]
                )
                for bucket in book_data["bid_buckets"]
            ]
        
        # Process ask buckets
        if "ask_buckets" in book_data and book_data["ask_buckets"]:
            result["ask_buckets"] = [
                PriceBucket(
                    range=bucket[0],
                    avg_price=bucket[1], 
                    volume=bucket[2]
                )
                for bucket in book_data["ask_buckets"]
            ]
        
        # Process bid outliers
        if "bid_outliers" in book_data and book_data["bid_outliers"]:
            result["bid_outliers"] = [
                PriceOutlier(
                    price=outlier[0],
                    volume=outlier[1]
                )
                for outlier in book_data["bid_outliers"]
            ]
        
        # Process ask outliers
        if "ask_outliers" in book_data and book_data["ask_outliers"]:
            result["ask_outliers"] = [
                PriceOutlier(
                    price=outlier[0],
                    volume=outlier[1]
                )
                for outlier in book_data["ask_outliers"]
            ]

        result["volume"] = sum([bucket.volume for bucket in result["bid_buckets"]]) \
                        + sum([bucket.volume for bucket in result["ask_buckets"]])
        
        return OrderBookSummaryData(**result)
