import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any, Union
import datetime
import logging
import motor.motor_asyncio
from pymongo import ASCENDING, DESCENDING
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import pytz
from cryptotrading.data.mongo import get_db, PRICE_COLLECTION_NAME


# Configure logging (optional, but good practice)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fastapi_server")

app = FastAPI(title="Crypto Price API", description="Provides candlestick data for cryptocurrency prices.", version="1.0")

# --- CORS (Cross-Origin Resource Sharing) Configuration ---
# Allow requests from specific origins (replace with your frontend's URL)
origins = [
    "http://localhost:3000",  # Example: Allow requests from a local React app
    "http://localhost:8000",
    "http://localhost:8080",
    "https://your-frontend-domain.com",  # Example: Allow requests from a deployed frontend
]

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True, # Allow sending cookies (if your app uses them)
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

app.db = get_db()
app.collection = app.db[PRICE_COLLECTION_NAME]
logger.info("Successfully connected to MongoDB.")


# --- Pydantic Models ---
class PriceBucket(BaseModel):
    range: str
    avg_price: float
    volume: float

class PriceOutlier(BaseModel):
    price: float
    volume: float

class OrderBookData(BaseModel):
    bid_buckets: List[PriceBucket] = []
    ask_buckets: List[PriceBucket] = []
    bid_outliers: List[PriceOutlier] = []
    ask_outliers: List[PriceOutlier] = []

class LatestPriceData(BaseModel):
    price: float
    timestamp: datetime.datetime
    order_book: Optional[OrderBookData] = None

class CandlestickData(BaseModel):
    timestamp: datetime.datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None  # Traditional candlestick volume (if you have it)
    exchange_count: Optional[float] = None
    order_book: Optional[OrderBookData] = None

    class Config:
        json_encoders = {
            datetime.datetime: lambda dt: dt.isoformat()
        }

# --- Helper Functions ---

def process_order_book_data(book_data: Dict[str, Any]) -> OrderBookData:
    """Convert raw order book data into structured OrderBookData."""
    result = OrderBookData()
    
    if not book_data:
        return result
        
    # Process bid buckets
    if "bid_buckets" in book_data and book_data["bid_buckets"]:
        result.bid_buckets = [
            PriceBucket(
                range=bucket[0],
                avg_price=bucket[1],
                volume=bucket[2]
            )
            for bucket in book_data["bid_buckets"]
        ]
    
    # Process ask buckets
    if "ask_buckets" in book_data and book_data["ask_buckets"]:
        result.ask_buckets = [
            PriceBucket(
                range=bucket[0],
                avg_price=bucket[1], 
                volume=bucket[2]
            )
            for bucket in book_data["ask_buckets"]
        ]
    
    # Process bid outliers
    if "bid_outliers" in book_data and book_data["bid_outliers"]:
        result.bid_outliers = [
            PriceOutlier(
                price=outlier[0],
                volume=outlier[1]
            )
            for outlier in book_data["bid_outliers"]
        ]
    
    # Process ask outliers
    if "ask_outliers" in book_data and book_data["ask_outliers"]:
        result.ask_outliers = [
            PriceOutlier(
                price=outlier[0],
                volume=outlier[1]
            )
            for outlier in book_data["ask_outliers"]
        ]
    
    return result

async def get_candlestick_data(
    token: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
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
            cursor = app.collection.aggregate(pipeline)
            raw_data = await cursor.to_list(length=None)
            
            # Process and group the data
            candle_map = {}
            
            for doc in raw_data:
                # Calculate which candle this document belongs to
                doc_time = doc["timestamp"]
                candle_time = doc_time.replace(
                    microsecond=0, 
                    second=doc_time.second - (doc_time.second % (granularity % 60)),
                    minute=doc_time.minute - (doc_time.minute % (granularity // 60 % 60)),
                    hour=doc_time.hour - (doc_time.hour % (granularity // 3600)),
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
                        "exchange_count": None,
                        "order_book": None if not include_book else {
                            "first_book": None,  # First book in the candle
                            "last_book": None,   # Last book in the candle
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
                if include_book and "metadata" in doc and "book" in doc["metadata"]:
                    # If this is the first or last order book we've seen, update accordingly
                    if candle_map[candle_time]["order_book"]["first_book"] is None:
                        candle_map[candle_time]["order_book"]["first_book"] = doc["metadata"]["book"]
                    
                    # Always update last book (the most recent one will be the last)
                    candle_map[candle_time]["order_book"]["last_book"] = doc["metadata"]["book"]
            
            # Convert the map to a list of CandlestickData objects
            candlestick_data = []
            
            for candle_time, candle in sorted(candle_map.items()):
                # Calculate average exchange count
                if candle["exchange_counts"]:
                    candle["exchange_count"] = sum(candle["exchange_counts"]) / len(candle["exchange_counts"])
                
                # Process order book if available
                order_book = None
                if include_book and candle["order_book"]["last_book"]:
                    # Use the last order book data from the candle period
                    order_book = process_order_book_data(candle["order_book"]["last_book"])
                
                # Create the candlestick data object
                candlestick = CandlestickData(
                    timestamp=candle["timestamp"],
                    open=candle["open"],
                    high=candle["high"],
                    low=candle["low"],
                    close=candle["close"],
                    exchange_count=candle["exchange_count"],
                    order_book=order_book
                )
                
                candlestick_data.append(candlestick)
            
            return candlestick_data

        except Exception as e:
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
            cursor = app.collection.aggregate(pipeline)
            raw_data = await cursor.to_list(length=None)  # Fetch all results

            # Convert raw data to Pydantic models
            candlestick_data = [CandlestickData(**item) for item in raw_data]
            return candlestick_data

        except Exception as e:
            logger.error(f"Database error in get_candlestick_data: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

async def get_latest_price(token: str) -> Optional[Dict[str, Any]]:
    """Retrieves the latest index price for a given token."""

    try:
        # Find the most recent document for the given symbol
        document = await app.collection.find_one(
            {"metadata.token": token, "metadata.type": "index_price"},
            sort=[("timestamp", DESCENDING)]
        )
        if document:
            # Extract order book data if available
            order_book = None
            if "metadata" in document and "book" in document["metadata"]:
                order_book = process_order_book_data(document["metadata"]["book"])
                
            return {
                "price": document['price'],
                "timestamp": document['timestamp'],
                "order_book": order_book
            }
        else:
            return None  # No data found for the token

    except Exception as e:
        logger.error(f"Database error in get_latest_price: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


# --- API Endpoints ---

@app.get("/candlestick/{token}", response_model=List[CandlestickData])
async def read_candlestick(
    token: str,
    start: datetime.datetime = Query(..., description="Start time (ISO 8601 format)"),
    end: datetime.datetime = Query(..., description="End time (ISO 8601 format)"),
    granularity: int = Query(..., description="Candlestick interval in seconds"),
    include_book: bool = Query(False, description="Include order book data in response")
):
    """
    Retrieve candlestick data for a specific symbol within a given time range and granularity.
    Optionally includes order book data for each candlestick.
    """
    try:
        # Convert start and end to UTC
        start_utc = start.astimezone(pytz.utc)
        end_utc = end.astimezone(pytz.utc)

        if end_utc <= start_utc:
            raise HTTPException(status_code=400, detail="End time must be greater than start time.")
        if granularity <= 0:
            raise HTTPException(status_code=400, detail="Granularity must be a positive integer.")
        data = await get_candlestick_data(token, start_utc, end_utc, granularity, include_book)

        if not data:
             raise HTTPException(status_code=404, detail=f"No data found for {token} between {start_utc} and {end_utc}")
        return data
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {ve}")


@app.get("/latest_price/{token}", response_model=LatestPriceData)
async def read_latest_price(token: str):
    """
    Retrieve the latest index price for a specific token.
    Includes order book data if available.
    """
    price_data = await get_latest_price(token)
    if price_data is None:
        raise HTTPException(status_code=404, detail=f"No data found for token {token}")
    return price_data

@app.get("/health")
async def health_check():
    """
    Health check endpoint.  Returns 200 OK if the server is running and can connect to the database.
    """
    try:
        # Check database connection (simple ping)
        await app.db.command("ping")
        return {"status": "OK", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail={"status": "unhealthy", "database": "disconnected", "error": str(e)})