import datetime as dt
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from fastapi import FastAPI
from cryptotrading.rollbit.prices.serve.models import (
    OrderBookSummaryData, PriceBucket, 
    PriceOutlier, 
    CandlestickData, 
    PriceDataPoint, 
    TransformedOrderBookDataPoint, 
    TransformedOrderBookData
)


def process_order_book_data(book_data: Dict[str, Any]) -> OrderBookSummaryData:
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

async def get_candlestick_data(
    app: FastAPI,
    token: str,
    start_time: datetime,
    end_time: datetime,
    granularity: int,
    include_book: bool = False
) -> List[CandlestickData]:
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
            cursor = app.price_collection.aggregate(pipeline)
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
                order_book = process_order_book_data(candle["order_book"])
                
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
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
    
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
            cursor = app.price_collection.aggregate(pipeline)
            raw_data = await cursor.to_list(length=None)  # Fetch all results

            # Convert raw data to Pydantic models
            candlestick_data = [CandlestickData(**item) for item in raw_data]
            return candlestick_data

        except Exception as e:
            logger.error(f"Database error in get_candlestick_data: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

async def get_historic_price(
    app: FastAPI,
    token: str, 
    start_time: datetime,
    end_time: datetime,
    page: int = 1,
    page_size: int = 1000
) -> Tuple[List[PriceDataPoint], int]:
    """
    Get paginated historic price data for a token within a time range.
    
    Args:
        token: The token symbol to fetch data for
        start_time: Start of the time range (inclusive)
        end_time: End of the time range (inclusive)
        page: Page number (1-based)
        page_size: Number of items per page
        
    Returns:
        Tuple of (list of PriceDataPoint, total_count)
    """
    try:
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 10000:  # Enforce reasonable limits
            page_size = 1000
            
        skip = (page - 1) * page_size
        
        query = {
            "metadata.token": token, 
            "metadata.type": "index_price", 
            "timestamp": {
                "$gte": start_time, 
                "$lte": end_time
            }
        }
        
        # Get total count first
        total_count = await app.price_collection.count_documents(query)
        
        # Get paginated results
        cursor = app.price_collection.find(
            query,
            sort=[("timestamp", 1)],  # 1 for ascending, -1 for descending
            skip=skip,
            limit=page_size
        )
        
        raw_data = await cursor.to_list(length=page_size)
        result = [PriceDataPoint.from_mongodb_doc(doc) for doc in raw_data]
        
        return result, total_count
        
    except Exception as e:
        logger.error(f"Database error in get_historic_price: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

async def get_latest_price(app: FastAPI,token: str) -> Optional[Dict[str, Any]]:
    """Retrieves the latest index price for a given token."""
    try:
        # Find the most recent document for the given symbol
        document = await app.price_collection.find_one(
            {"metadata.token": token, "metadata.type": "index_price"},
            sort=[("timestamp", DESCENDING)]
        )
        
        if not document:
            logger.debug(f"No price data found for token: {token}")
            return None
            
        # Extract order book data if available
        order_book = None
        if "metadata" in document and "book" in document["metadata"]:
            order_book = process_order_book_data(document["metadata"]["book"])
            
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
        return None  # Return None instead of raising to allow graceful degradation

async def get_latest_transformed_order_book_point(app: FastAPI, token: str) -> Optional[TransformedOrderBookDataPoint]:
    """Retrieves the latest transformed order book point for a given token."""

    try:
        # Find the most recent document for the given symbol
        document = await app.transformed_order_book_collection.find_one(
            {"metadata.token": token},
            sort=[("timestamp", DESCENDING)]
        )
        if document:
            return TransformedOrderBookDataPoint.from_mongodb_doc(document)
        else:
            return None  # No data found for the token

    except Exception as e:
        logger.error(f"Database error in get_latest_transformed_order_book_point: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

async def get_transformed_order_book(
    app: FastAPI,
    token: str,
    start_time: datetime,
    end_time: datetime,
    granularity: int
) -> Optional[TransformedOrderBookData]:
    """Retrieves the transformed order book for a given token.
    
    Note: This implementation works with MongoDB timeseries collections by using
    $setWindowFields for time-based bucketing instead of $group.
    """
    try:
        # First, get the raw data points within the time range
        cursor = app.transformed_order_book_collection.find({
            "metadata.token": token,
            "timestamp": {"$gte": start_time, "$lte": end_time}
        }).sort("timestamp", 1)
        
        # Process data in memory for timeseries collections
        points = []
        current_bucket = None
        bucket_start = None
        
        async for doc in cursor:
            timestamp = doc.get("timestamp")
            if not timestamp:
                continue
                
            # Convert to timestamp in milliseconds for bucketing
            ts_ms = int(timestamp.timestamp() * 1000)
            bucket_time = ts_ms - (ts_ms % (granularity * 1000))
            
            if current_bucket is None or bucket_time != bucket_start:
                # If we have a complete bucket, add it to points
                if current_bucket is not None:
                    points.append(current_bucket)
                
                # Start a new bucket
                bucket_start = bucket_time
                current_bucket = {
                    "timestamp": datetime.fromtimestamp(bucket_time / 1000, tz=dt.timezone.utc),
                    "lowest_ask": float('inf'),
                    "highest_bid": float('-inf'),
                    "points_in_bucket": 0
                }
            
            # Update bucket stats
            if "metadata" in doc:
                ask = doc["metadata"].get("lowest_ask")
                bid = doc["metadata"].get("highest_bid")
                
                if ask is not None:
                    current_bucket["lowest_ask"] = min(current_bucket["lowest_ask"], ask)
                if bid is not None:
                    current_bucket["highest_bid"] = max(current_bucket["highest_bid"], bid)
                
                current_bucket["points_in_bucket"] += 1
        
        # Add the last bucket if it exists
        if current_bucket is not None and current_bucket["points_in_bucket"] > 0:
            points.append(current_bucket)
        
        # Calculate derived fields
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
            
            result_points.append({
                "timestamp": point["timestamp"],
                "lowest_ask": lowest_ask,
                "highest_bid": highest_bid,
                "midpoint": midpoint,
                "spread": spread
            })
        
        if result_points:
            return TransformedOrderBookData(
                token=token,
                points=[TransformedOrderBookDataPoint(**doc) for doc in result_points]
            )
        return None  # No data found for the token

    except Exception as e:
        logger.error(f"Database error in get_transformed_order_book: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
