import logging
import datetime as dt
from datetime import datetime, timezone
from typing import Any, Optional, Union
from abc import ABC, abstractmethod

from cryptotrading.data.postgres import get_connection, init_pool, _pool, resolve_matching_symbols

from pymongo import ASCENDING

from cryptotrading.data.mongo import get_db
from cryptotrading.data.models import (
    TransformedOrderBookData,
    TransformedOrderBookDataPoint,
    OrderBookSnapshot, 
    OrderBookSummaryData, 
    PriceBucket, 
    PriceOutlier,
    ExchangeRawOrderBook,
)
from cryptotrading.config import (
    COMPOSITE_ORDER_BOOK_COLLECTION_NAME,
    EXCHANGE_ORDER_BOOK_COLLECTION_NAME,
    TRANSFORMED_ORDER_BOOK_COLLECTION_NAME,
)
from cryptotrading.util.book import order_book_to_df

logger = logging.getLogger(__name__)

class OrderBookAdapter(ABC):
    @abstractmethod
    async def store_exchange_order_book(
        self, 
        symbol: str,            
        raw_data: list[Union[ExchangeRawOrderBook, dict]], 
        verbose: bool = False
    ) -> None:        
        raise NotImplementedError
    
    @abstractmethod
    async def store_composite_order_book_data(
        self, 
        symbol: str,            
        raw_data: list[Union[TransformedOrderBookData, dict]], 
        verbose: bool = False
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def store_transformed_order_book_data(
        self, 
        symbol: str, 
        book: dict, 
        verbose: bool=False
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_orderbook_data(
        self, 
        symbol: str, 
        start_time: datetime, 
        end_time: datetime, 
        verbose: bool = False,
        include_book: bool = False,
        count: int = None,
        chunk_size: int = 1000
    ) -> list[dict]:
        raise NotImplementedError


    

class OrderBookMongoAdapter(OrderBookAdapter):
    def __init__(self):
        self.db = get_db()

    @property
    async def collections(self):
        return await self.db.list_collection_names() 

    async def initialize(self):        
        await self.init_composite_order_book_collection()
        await self.init_exchange_order_book_collection()
        await self.init_transformed_order_book_collection()
    
    async def shutdown(self):
        pass
    
    async def init_composite_order_book_collection(self):
        self.composite_order_book_collection = self.db[COMPOSITE_ORDER_BOOK_COLLECTION_NAME]
        
        # Create time series collection if it doesn't exist
        if COMPOSITE_ORDER_BOOK_COLLECTION_NAME not in await self.collections:
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
        if EXCHANGE_ORDER_BOOK_COLLECTION_NAME not in await self.collections:
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
        if TRANSFORMED_ORDER_BOOK_COLLECTION_NAME not in await self.collections:
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
            raw_data: list[Union[ExchangeRawOrderBook, dict]], 
            verbose: bool = False
    ) -> None:
        """Store calculated index price and raw data in MongoDB time series collection"""
        timestamp = dt.datetime.now(dt.timezone.utc)
        if verbose: 
            logger.info(f"Storing exchange order book data for {symbol}")
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
        raw_data: list[Union[ExchangeRawOrderBook, dict]],
        verbose: bool=False
    ) -> None:
        """Store calculated index price and raw composite order book data in MongoDB time series collection"""
        timestamp = dt.datetime.now(dt.timezone.utc)
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
        timestamp = dt.datetime.now(dt.timezone.utc)
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

    async def get_orderbook_data(
        self,
        token: str,
        start_time: dt.datetime,
        end_time: dt.datetime,
    ) -> list[OrderBookSnapshot]:
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
                    "timestamp": {"$gte": start_time, "$lte": end_time},
                }
            },
            {
                "$sort": {"timestamp": 1}  # Ensure data is sorted by timestamp
            }
        ]
        cursor = self.composite_order_book_collection.aggregate(pipeline)
        return [OrderBookSnapshot.from_mongodb_doc(doc) async for doc in cursor]

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

        return OrderBookSummaryData(**result)


class OrderBookPostgresAdapter:
    def __init__(self):
        pass

    async def initialize(self):
        if _pool is None:
            await init_pool()

    async def shutdown(self):
        pass

    async def store_exchange_order_book(
        self, 
        symbol: str,            
        raw_data: list[Union[ExchangeRawOrderBook, dict]], 
        verbose: bool = False
    ) -> None:
        # Same logic as Mongo: extracts stats and stores
        timestamp = datetime.now(timezone.utc)
        token = symbol.split("/")[0] if "/" in symbol else symbol
        
        async with get_connection() as conn:
            for book in raw_data:
                exchange = book.get('exchange', 'unknown')
                
                # Check bids and asks
                if book.get('bids') and book.get('asks'):
                    df = order_book_to_df(book['bids'], book['asks'])
                    
                    ask_filter = df['side'] == 'a'
                    bid_filter = df['side'] == 'b'
                    size_filter = df['size'] > 0
 
                    asks = df[ask_filter & size_filter]
                    bids = df[bid_filter & size_filter]
 
                    lowest_ask = float(asks['price'].min())
                    highest_ask = float(asks['price'].max())
                    lowest_bid = float(bids['price'].min())
                    highest_bid = float(bids['price'].max())
                    
                    spread = abs(lowest_ask - highest_bid)
                    mid_price = (lowest_ask + highest_bid) / 2
                    
                    book['bids'].sort(key=lambda x: x[1], reverse=True)
                    book['asks'].sort(key=lambda x: x[1], reverse=True)
 
                    highest_volume_bid = book['bids'][0]
                    highest_volume_ask = book['asks'][0]
 
                    total_bid_size = float(bids["size"].sum())
                    total_ask_size = float(asks["size"].sum())
                    
                    metadata = {
                        "token": token,
                        "symbol": symbol,
                        "exchange": exchange,
                        "spread": spread, 
                        "price": mid_price,
                        "lowest_bid": lowest_bid,                        
                        "highest_bid": highest_bid,
                        "total_bid_size": total_bid_size,
                        "highest_volume_bid_price": highest_volume_bid[0],
                        "highest_volume_bid_vol": highest_volume_bid[1],
                        "lowest_ask": lowest_ask,
                        "highest_ask": highest_ask,                        
                        "total_ask_size": total_ask_size,                        
                        "highest_volume_ask_price": highest_volume_ask[0],
                        "highest_volume_ask_vol": highest_volume_ask[1],
                        "type": "exchange_data"
                    }
                    
                    # Store as a regular row in price_data
                    await conn.execute('''
                        INSERT INTO price_data (time, symbol, exchange, close, metadata)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (time, symbol, exchange) DO UPDATE
                        SET close = EXCLUDED.close, metadata = EXCLUDED.metadata;
                    ''', timestamp, symbol, f"exchange_raw_{exchange}", mid_price, metadata)

    async def store_composite_order_book_data(
        self, 
        symbol: str, 
        book: dict, 
        raw_data: list[Union[ExchangeRawOrderBook, dict]],
        verbose: bool = False
    ) -> None:
        timestamp = datetime.now(timezone.utc)
        token = symbol.split("/")[0] if "/" in symbol else symbol
 
        df = order_book_to_df(book['bids'], book['asks'])
        
        ask_filter = df['side'] == 'a'
        bid_filter = df['side'] == 'b'
        size_filter = df['size'] > 0
 
        asks = df[ask_filter & size_filter]
        bids = df[bid_filter & size_filter]
 
        lowest_ask = float(asks['price'].min())
        highest_ask = float(asks['price'].max())
        lowest_bid = float(bids['price'].min())
        highest_bid = float(bids['price'].max())
 
        spread = abs(lowest_ask - highest_bid)
        midpoint = (lowest_ask + highest_bid) / 2
 
        book['bids'].sort(key=lambda x: x[1], reverse=True)
        book['asks'].sort(key=lambda x: x[1], reverse=True)
 
        largest_size_bid = book['bids'][0]
        largest_size_ask = book['asks'][0]
 
        total_bid_size = float(bids["size"].sum())
        total_ask_size = float(asks["size"].sum())
 
        book['bids'].sort(key=lambda x: x[0], reverse=True)
        book['asks'].sort(key=lambda x: x[0], reverse=False)
        
        metadata = {
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
            "type": "order_book"
        }
        
        # Serialize lists/tuples for JSONB compatibility
        serializable_book = {
            "bids": [[float(price), float(qty)] for price, qty in book["bids"]],
            "asks": [[float(price), float(qty)] for price, qty in book["asks"]]
        }
        
        async with get_connection() as conn:
            await conn.execute('''
                INSERT INTO price_data (time, symbol, exchange, close, metadata)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (time, symbol, exchange) DO UPDATE
                SET close = EXCLUDED.close, metadata = EXCLUDED.metadata;
            ''', timestamp, symbol, 'composite', midpoint, {
                **metadata,
                "book": serializable_book,
                "exchanges_count": len(raw_data)
            })
 
    async def store_transformed_order_book_data(
        self, 
        symbol: str, 
        book: dict, 
        verbose: bool = False
    ) -> None:
        timestamp = datetime.now(timezone.utc)
        token = symbol.split("/")[0] if "/" in symbol else symbol
 
        df = order_book_to_df(book['bids'], book['asks'])
        
        ask_filter = df['side'] == 'a'
        bid_filter = df['side'] == 'b'
        size_filter = df['size'] > 0
 
        asks = df[ask_filter & size_filter]
        bids = df[bid_filter & size_filter]
 
        lowest_ask = float(asks['price'].min())
        highest_bid = float(bids['price'].max())
 
        spread = abs(lowest_ask - highest_bid)
        midpoint = (lowest_ask + highest_bid) / 2
 
        metadata = {
            "token": token,
            "symbol": symbol,                
            "lowest_ask": lowest_ask,
            "highest_bid": highest_bid,
            "midpoint": midpoint,
            "spread": spread,
            "type": "order_book"
        }
        
        async with get_connection() as conn:
            await conn.execute('''
                INSERT INTO price_data (time, symbol, exchange, close, metadata)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (time, symbol, exchange) DO UPDATE
                SET close = EXCLUDED.close, metadata = EXCLUDED.metadata;
            ''', timestamp, symbol, 'transformed', midpoint, metadata)
 
    async def get_orderbook_data(
        self,
        token: str,
        start_time: datetime,
        end_time: datetime,
        interval: Optional[Union[float, dt.timedelta, str]] = None
    ) -> list[OrderBookSnapshot]:
        matching_symbols = await resolve_matching_symbols(token)
        if not matching_symbols:
            return []
            
        bucket_width = None
        if interval is not None:
            seconds = 0.0
            if isinstance(interval, dt.timedelta):
                seconds = interval.total_seconds()
            elif isinstance(interval, (int, float)):
                seconds = float(interval)
            elif isinstance(interval, str):
                clean_str = "".join(c for c in interval if c.isdigit() or c == '.')
                try:
                    seconds = float(clean_str)
                except ValueError:
                    seconds = 0.0
            
            if seconds > 1.0:
                bucket_width = f"{seconds} seconds"

        async with get_connection() as conn:
            if bucket_width is not None:
                if len(matching_symbols) == 1:
                    query = f'''
                        SELECT 
                            time_bucket('{bucket_width}'::interval, time) as timestamp, 
                            last(close, time) as midpoint, 
                            last(metadata, time) as metadata
                        FROM price_data
                        WHERE symbol = $1 AND exchange = 'composite' AND time >= $2 AND time <= $3
                        GROUP BY timestamp, symbol
                        ORDER BY timestamp ASC;
                    '''
                    rows = await conn.fetch(query, matching_symbols[0], start_time, end_time)
                else:
                    query = f'''
                        SELECT 
                            time_bucket('{bucket_width}'::interval, time) as timestamp, 
                            last(close, time) as midpoint, 
                            last(metadata, time) as metadata
                        FROM price_data
                        WHERE symbol = ANY($1) AND exchange = 'composite' AND time >= $2 AND time <= $3
                        GROUP BY timestamp, symbol
                        ORDER BY timestamp ASC;
                    '''
                    rows = await conn.fetch(query, matching_symbols, start_time, end_time)
            else:
                if len(matching_symbols) == 1:
                    query = '''
                        SELECT time as timestamp, close as midpoint, metadata
                        FROM price_data
                        WHERE symbol = $1 AND exchange = 'composite' AND time >= $2 AND time <= $3
                        ORDER BY time ASC;
                    '''
                    rows = await conn.fetch(query, matching_symbols[0], start_time, end_time)
                else:
                    query = '''
                        SELECT time as timestamp, close as midpoint, metadata
                        FROM price_data
                        WHERE symbol = ANY($1) AND exchange = 'composite' AND time >= $2 AND time <= $3
                        ORDER BY time ASC;
                    '''
                    rows = await conn.fetch(query, matching_symbols, start_time, end_time)
            
        snapshots = []
        for r in rows:
            meta = r["metadata"] or {}
            book = meta.get("book", {})
            bids = sorted([(float(p), float(q)) for p, q in book.get("bids", [])], key=lambda x: x[0], reverse=True)
            asks = sorted([(float(p), float(q)) for p, q in book.get("asks", [])], key=lambda x: x[0])
            
            snapshots.append(OrderBookSnapshot(
                timestamp=r["timestamp"].timestamp(),
                bids=bids,
                asks=asks,
                mid_price=meta.get("midpoint", r["midpoint"])
            ))
        return snapshots
 
    @staticmethod
    def process_order_book_data(book_data: dict[str, Any]) -> OrderBookSummaryData:
        
        result = {
            "bid_buckets": [],
            "ask_buckets": [],
            "bid_outliers": [],
            "ask_outliers": [],
            "volume": 0.0
        }
        
        if not book_data:
            return OrderBookSummaryData(**result)
            
        if "bid_buckets" in book_data and book_data["bid_buckets"]:
            result["bid_buckets"] = [
                PriceBucket(range=bucket[0], avg_price=bucket[1], volume=bucket[2])
                for bucket in book_data["bid_buckets"]
            ]
        
        if "ask_buckets" in book_data and book_data["ask_buckets"]:
            result["ask_buckets"] = [
                PriceBucket(range=bucket[0], avg_price=bucket[1], volume=bucket[2])
                for bucket in book_data["ask_buckets"]
            ]
        
        if "bid_outliers" in book_data and book_data["bid_outliers"]:
            result["bid_outliers"] = [
                PriceOutlier(price=outlier[0], volume=outlier[1])
                for outlier in book_data["bid_outliers"]
            ]
        
        if "ask_outliers" in book_data and book_data["ask_outliers"]:
            result["ask_outliers"] = [
                PriceOutlier(price=outlier[0], volume=outlier[1])
                for outlier in book_data["ask_outliers"]
            ]
 
        result["volume"] = sum([bucket.volume for bucket in result["bid_buckets"]]) \
                        + sum([bucket.volume for bucket in result["ask_buckets"]])
        
        return OrderBookSummaryData(**result)
 
    async def get_transformed_order_book_since(self, last_checked: datetime) -> list[dict]:
        query = '''
            SELECT time as timestamp, close as midpoint, metadata
            FROM price_data
            WHERE exchange = 'transformed' AND time > $1
            ORDER BY time ASC;
        '''
        async with get_connection() as conn:
            rows = await conn.fetch(query, last_checked)
        return [dict(r) for r in rows]
 
    async def get_latest_transformed_order_book_point(self, token: str) -> Optional[TransformedOrderBookDataPoint]:
        matching_symbols = await resolve_matching_symbols(token)
        if not matching_symbols:
            return None
            
        query = '''
            SELECT time as timestamp, close as midpoint, metadata
            FROM price_data
            WHERE symbol = ANY($1) AND exchange = 'transformed'
            ORDER BY time DESC LIMIT 1;
        '''
        async with get_connection() as conn:
            row = await conn.fetchrow(query, matching_symbols)
            
        if not row:
            return None
            
        meta = row['metadata'] or {}
        return TransformedOrderBookDataPoint(
            timestamp=row['timestamp'].replace(tzinfo=timezone.utc) if row['timestamp'].tzinfo is None else row['timestamp'],
            lowest_ask=meta.get('lowest_ask'),
            highest_bid=meta.get('highest_bid'),
            midpoint=meta.get('midpoint', row['midpoint']),
            spread=meta.get('spread')
        )
 
    async def get_transformed_order_book(
        self,
        token: str,
        start_time: datetime,
        end_time: datetime,
        granularity: int
    ) -> Optional[Any]:
        matching_symbols = await resolve_matching_symbols(token)
        if not matching_symbols:
            return None
            
        query = '''
            SELECT time as timestamp, close as midpoint, metadata
            FROM price_data
            WHERE symbol = ANY($1) AND exchange = 'transformed' AND time >= $2 AND time <= $3
            ORDER BY time ASC;
        '''
        async with get_connection() as conn:
            rows = await conn.fetch(query, matching_symbols, start_time, end_time)
            
        if not rows:
            return None
            
        points = []
        current_bucket = None
        bucket_start = None
        
        for row in rows:
            timestamp = row["timestamp"]
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            ts_ms = int(timestamp.timestamp() * 1000)
            bucket_time = ts_ms - (ts_ms % (granularity * 1000))
            
            if current_bucket is None or bucket_time != bucket_start:
                if current_bucket is not None:
                    points.append(current_bucket)
                
                bucket_start = bucket_time
                current_bucket = {
                    "timestamp": datetime.fromtimestamp(bucket_time / 1000, tz=timezone.utc),
                    "lowest_ask": float('inf'),
                    "highest_bid": float('-inf'),
                    "points_in_bucket": 0
                }
            
            meta = row["metadata"] or {}
            ask = meta.get("lowest_ask")
            bid = meta.get("highest_bid")
            
            if ask is not None:
                current_bucket["lowest_ask"] = min(current_bucket["lowest_ask"], ask)
            if bid is not None:
                current_bucket["highest_bid"] = max(current_bucket["highest_bid"], bid)
            
            current_bucket["points_in_bucket"] += 1
            
        if current_bucket is not None and current_bucket["points_in_bucket"] > 0:
            points.append(current_bucket)
            
        result_points = []
        for point in points:
            if point["points_in_bucket"] == 0:
                continue
                
            lowest_ask = point["lowest_ask"] if point["lowest_ask"] != float('inf') else None
            highest_bid = point["highest_bid"] if point["highest_bid"] != float('-inf') else None
            
            if lowest_ask is not None and highest_bid is not None:
                midpoint = (lowest_ask + highest_bid) / 2
                spread = abs(lowest_ask - highest_bid)
            else:
                midpoint = None
                spread = None
            
            result_points.append(TransformedOrderBookDataPoint(
                timestamp=point["timestamp"],
                lowest_ask=lowest_ask,
                highest_bid=highest_bid,
                midpoint=midpoint,
                spread=spread
            ))
            
        if result_points:
            return TransformedOrderBookData(
                token=token,
                points=result_points
            )
        return None
