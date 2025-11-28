import logging
import datetime
from typing import Optional, Any

from pymongo import ASCENDING, DESCENDING

from cryptotrading.data.mongo import get_db
from cryptotrading.data.models import (
    CandlestickData,
    OrderBookSummaryData,
    PriceBucket,
    PriceDataPoint,
    PriceOutlier,
    TransformedOrderBookData,
    TransformedOrderBookDataPoint,
)
from cryptotrading.data.book import OrderBookMongoAdapter
from cryptotrading.config import (
    PRICE_COLLECTION_NAME,
)

logger = logging.getLogger(__name__)

class PriceMongoAdapter:
    def __init__(self):
        self.db = get_db()

    @property
    async def collections(self):
        return await self.db.list_collection_names() 

    async def initialize(self):
        await self.init_price_collection()
     
    async def shutdown(self):
        pass

    async def init_price_collection(self):
        self.price_collection = self.db[PRICE_COLLECTION_NAME]
        
        # Create time series collection if it doesn't exist
        if PRICE_COLLECTION_NAME not in await self.collections:
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
            await self.price_collection.create_index([("timestamp", ASCENDING)])
            await self.price_collection.create_index([("metadata.symbol", ASCENDING)])

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
            await self.price_collection.insert_one(index_doc)
            
            logger.debug(f"Stored price data for {symbol}: {index_price}")
        except Exception as e:
            logger.error(f"Failed to store price data: {str(e)}")
        
    async def get_price_data(
        self, 
        symbol: str, 
        start_time: datetime.datetime, 
        end_time: datetime.datetime, 
        limit: int = 100,
        page: int = 1,
        sort: str = "desc"
    ) -> list[dict]:
        """Get price data for a specific symbol"""
        try:
            cursor = self.price_collection.find(
                {
                    "metadata.symbol": symbol, 
                    "timestamp": {"$gte": start_time, "$lte": end_time}
                }
            )
            if sort == "desc":
                cursor.sort([("timestamp", -1)])
            elif sort == "asc":
                cursor.sort([("timestamp", 1)])
            cursor = cursor.limit(limit).skip((page - 1) * limit)
            return [doc for doc in cursor]
        except Exception as e:
            logger.error(f"Failed to get price data: {str(e)}")
            return []

    async def get_price_data_count(
        self, 
        symbol: str, 
        start_time: datetime.datetime, 
        end_time: datetime.datetime
    ) -> int:
        """Get the count of price data for a specific symbol"""
        try:
            return self.price_collection.count_documents(
                {
                    "metadata.symbol": symbol, 
                    "timestamp": {"$gte": start_time, "$lte": end_time}
                }
            )
        except Exception as e:
            logger.error(f"Failed to get price data count: {str(e)}")
            return 0

    async def get_latest_price(self, token: str) -> Optional[dict[str, Any]]:
        """Retrieves the latest index price for a given token."""
        try:
            # Find the most recent document for the given symbol
            document = await self.price_collection.find_one(
                {"metadata.token": token, "metadata.type": "index_price"},
                sort=[("timestamp", DESCENDING)]
            )
            
            if not document:
                logger.debug(f"No price data found for token: {token}")
                return None
                
            # Extract order book data if available
            order_book = None
            if "metadata" in document and "book" in document["metadata"]:
                order_book = OrderBookMongoAdapter.process_order_book_data(document["metadata"]["book"])
                
            # Build response with all available metadata
            result = {
                "price": document.get('price'),
                "volume": order_book.volume if order_book else None,
                "timestamp": document.get('timestamp'),
                "metadata": document.get('metadata', {}),
                "order_book": order_book
            }
            
            logger.debug(f"Retrieved latest price for {token}: {result.get('price')}")
            return result
            
        except Exception as e:
            logger.error(f"Database error in get_latest_price for token {token}: {e}")
            return None
    
    async def get_candlestick_data(
        self,
        token: str,
        start_time: datetime,
        end_time: datetime,
        granularity: int,
        include_book: bool = False
    ) -> list[CandlestickData]:
        """Retrieve and aggregate price data into candlestick format.

        Args:
            token:       The trading symbol (e.g., "BTC/USDT").
            start_time:  The start of the time range (inclusive).
            end_time:    The end of the time range (inclusive).
            granularity: The candlestick interval in seconds.
            include_book: Whether to include order book data

        Returns:
            A list of CandlestickData objects.  Returns an empty list if no data is found.
            Raises an exception if there's a database error.
        """

        # Base aggregation pipeline for candlestick data
        pipeline = [
            {
                "$match": {
                    "metadata.token": token,
                    "metadata.type": "index_price",
                    "timestamp": {"$gte": start_time, "$lte": end_time},
                }
            },
            {
                "$sort": {"timestamp": 1}  # Ensure data is sorted by timestamp
            }
        ]
        
        if include_book:
            # When including order book data, we need a different approach
            # We'll fetch all documents then group them in Python
            try:
                cursor = self.price_collection.aggregate(pipeline)
                raw_data = await cursor.to_list(length=None)
                
                # Process and group the data
                candle_map = {}

                def calc_second(doc_time):
                    return doc_time.second - (doc_time.second % (granularity % 60)) if granularity % 60 > 0 else doc_time.second
                def calc_minute(doc_time):
                    return doc_time.minute - (doc_time.minute % (granularity // 60 % 60)) if granularity // 60 % 60 > 0 else doc_time.minute
                def calc_hour(doc_time):
                    return doc_time.hour - (doc_time.hour % (granularity // 3600)) if granularity // 3600 > 0 else doc_time.hour
                    
                for doc in raw_data:
                    # Calculate which candle this document belongs to
                    doc_time = doc["timestamp"]
                    candle_time = doc_time.replace(
                        microsecond=0, 
                        second=calc_second(doc_time),
                        minute=calc_minute(doc_time),
                        hour=calc_hour(doc_time),
                    )
                    
                    if candle_time not in candle_map:
                        candle_map[candle_time] = {
                            "timestamp": candle_time,
                            "prices": [],
                            "exchange_counts": [],
                            "open": None,
                            "high": float("-inf"),
                            "low": float("inf"),
                            "close": None,
                            "volume": 0,
                            "exchange_count": None,
                            "order_book": {
                                "bid_buckets": {},
                                "ask_buckets": {},
                                "bid_outliers": {},
                                "ask_outliers": {},
                            }
                        }
                    
                    # Add price data
                    price = doc["price"]
                    candle_map[candle_time]["prices"].append(price)
                    candle_map[candle_time]["exchange_counts"].append(doc.get("exchanges_count", 0))
                    
                    # Update high and low
                    candle_map[candle_time]["high"] = max(candle_map[candle_time]["high"], price)
                    candle_map[candle_time]["low"] = min(candle_map[candle_time]["low"], price)
                    
                    # If this is the first price for this candle, set it as open
                    if candle_map[candle_time]["open"] is None:
                        candle_map[candle_time]["open"] = price
                    
                    # Always update close (last price will be the close)
                    candle_map[candle_time]["close"] = price
                    
                    # Get order book data if available and requested
                    if "metadata" in doc and "book" in doc["metadata"]:
                        candle_map[candle_time]["order_book"] = {**candle_map[candle_time]["order_book"], **doc["metadata"]["book"]}
                
                # Convert the map to a list of CandlestickData objects
                candlestick_data = []
                
                for candle_time, candle in sorted(candle_map.items()):
                    # Calculate average exchange count
                    if candle["exchange_counts"] and len(candle["exchange_counts"]):
                        candle["exchange_count"] = sum(candle["exchange_counts"]) / len(candle["exchange_counts"])
                    
                    # Process order book if available
                    order_book = OrderBookMongoAdapter.process_order_book_data(candle["order_book"])
                    
                    # Create the candlestick data object
                    candlestick = CandlestickData(
                        timestamp=candle["timestamp"],
                        open=candle["open"],
                        high=candle["high"],
                        low=candle["low"],
                        close=candle["close"],
                        volume=order_book.volume if order_book else 0,
                        exchange_count=candle["exchange_count"],
                        order_book=order_book
                    )
                    
                    candlestick_data.append(candlestick)
                
                return candlestick_data

            except Exception as e:
                #  include stack trace
                import traceback
                logger.error(traceback.format_exc())
                logger.error(f"Database error in get_candlestick_data with order book: {e}")
                raise Exception(f"Database error: {e}")
        
        else:
            # If not including order book, use the standard aggregation pipeline
            group_pipeline = [
                {
                    "$group": {
                        "_id": {
                            "$toDate": {
                                "$subtract": [
                                    {"$toLong": "$timestamp"},
                                    {"$mod": [{"$toLong": "$timestamp"}, granularity * 1000]},
                                ]
                            }
                        },
                        "open": {"$first": "$price"},
                        "high": {"$max": "$price"},
                        "low": {"$min": "$price"},
                        "close": {"$last": "$price"},                    
                        "exchange_count": {"$avg": "$exchanges_count"}, # Average number of exchanges
                    }
                },
                {
                    "$sort": {"_id": 1}  # Sort by timestamp (ascending)
                },
                {
                    "$project": {
                        "_id": 0,
                        "timestamp": "$_id",
                        "open": 1,
                        "high": 1,
                        "low": 1,
                        "close": 1,
                        "exchange_count": 1,
                    }
                },
            ]
            
            pipeline.extend(group_pipeline)
            
            try:
                cursor = self.price_collection.aggregate(pipeline)
                raw_data = await cursor.to_list(length=None)  # Fetch all results

                # Convert raw data to Pydantic models
                candlestick_data = [CandlestickData(**item) for item in raw_data]
                return candlestick_data

            except Exception as e:
                logger.error(f"Database error in get_candlestick_data: {e}")
                raise Exception(f"Database error: {e}")