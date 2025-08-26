import os
import json
import asyncio
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any, Union, Set, Callable, Awaitable
import datetime
import logging
import motor.motor_asyncio
from pymongo import ASCENDING, DESCENDING
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import pytz
from cryptotrading.data.mongo import get_db, PRICE_COLLECTION_NAME, TRANSFORMED_ORDER_BOOK_COLLECTION_NAME


# Configure logging (optional, but good practice)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fastapi_server")

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {
            'price': set(),
            'order_book': set()
        }
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, channel: str):
        await websocket.accept()
        async with self.lock:
            self.active_connections[channel].add(websocket)

    def disconnect(self, websocket: WebSocket, channel: str):
        if websocket in self.active_connections[channel]:
            self.active_connections[channel].remove(websocket)

    async def broadcast(self, message: str, channel: str):
        if channel not in self.active_connections:
            return
            
        disconnected = set()
        for connection in self.active_connections[channel]:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Error sending to WebSocket: {e}")
                disconnected.add(connection)
        
        if disconnected:
            async with self.lock:
                self.active_connections[channel] -= disconnected

app = FastAPI(title="Crypto Price API", 
             description="Provides candlestick data for cryptocurrency prices with WebSocket support for real-time updates.", 
             version="1.0")

# Initialize WebSocket manager
websocket_manager = ConnectionManager()

# --- CORS (Cross-Origin Resource Sharing) Configuration ---
# Allow requests from specific origins (replace with your frontend's URL)
origins = [
    "http://localhost:3000",  # Default React port
    "http://localhost:8000",  # Default FastAPI port
    "http://localhost:8080",  # Common alternative port
    "http://localhost:5173",  # Vite dev server
    "https://your-frontend-domain.com",  # Production frontend
]

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"http(s)?://(localhost|127\.0\.0\.1)(:[0-9]+)?",  # Allow any localhost port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add WebSocket specific CORS headers
@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    origin = request.headers.get('origin')
    if origin and any(origin.startswith(allowed) for allowed in ["http://localhost", "http://127.0.0.1"]):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Headers"] = "*"
    return response

app.db = get_db()
app.price_collection = app.db[PRICE_COLLECTION_NAME]
app.transformed_order_book_collection = app.db[TRANSFORMED_ORDER_BOOK_COLLECTION_NAME]
logger.info("Successfully connected to MongoDB.")

# Background task for MongoDB change streams
async def watch_price_changes():
    last_checked = datetime.datetime.utcnow()
    
    while True:
        try:
            # First try to use change streams if available
            try:
                async with app.price_collection.watch([
                    {'$match': {'operationType': 'insert'}}
                ]) as stream:
                    logger.info("Successfully connected to price change stream")
                    async for change in stream:
                        try:
                            doc = change['fullDocument']
                            token = doc.get('metadata', {}).get('token')
                            if token and 'price' in doc:
                                # Ensure timestamp is serializable
                                timestamp = doc['timestamp']
                                if hasattr(timestamp, 'isoformat'):
                                    timestamp = timestamp.isoformat()
                                    
                                price_data = {
                                    'price': doc['price'],
                                    'timestamp': timestamp,
                                    'order_book': None
                                }
                                await websocket_manager.broadcast(
                                    json.dumps({
                                        'type': 'price_update',
                                        'token': token,
                                        'data': price_data
                                    }, default=json_serial),
                                    'price'
                                )
                        except Exception as e:
                            logger.error(f"Error processing price change: {e}")
            except Exception as e:
                if "replica sets" in str(e).lower():
                    logger.warning("Change streams not available, falling back to polling")
                    break  # Exit change stream mode and use polling
                raise  # Re-raise other errors
                
        except Exception as e:
            logger.error(f"Price change stream error: {e}")
            await asyncio.sleep(5)  # Wait before retrying
            
    # Fall back to polling if change streams aren't available
    logger.info("Starting price polling")
    while True:
        try:
            # Query for new documents since last check
            cursor = app.price_collection.find({
                'timestamp': {'$gt': last_checked}
            }).sort('timestamp', 1)
            
            async for doc in cursor:
                token = doc.get('metadata', {}).get('token')
                if token and 'price' in doc:
                    # Ensure timestamp is serializable
                    timestamp = doc['timestamp']
                    if hasattr(timestamp, 'isoformat'):
                        timestamp = timestamp.isoformat()
                        
                    price_data = {
                        'price': doc['price'],
                        'timestamp': timestamp,
                        'order_book': None
                    }
                    await websocket_manager.broadcast(
                        json.dumps({
                            'type': 'price_update',
                            'token': token,
                            'data': price_data
                        }, default=json_serial),
                        'price'
                    )
                    last_checked = doc['timestamp']
            
            await asyncio.sleep(1)  # Poll every second
            
        except Exception as e:
            logger.error(f"Price polling error: {e}")
            await asyncio.sleep(5)  # Wait before retrying

async def watch_order_book_changes():
    last_checked = datetime.datetime.utcnow()
    
    while True:
        try:
            # First try to use change streams if available
            try:
                async with app.transformed_order_book_collection.watch([
                    {'$match': {'operationType': 'insert'}}
                ]) as stream:
                    logger.info("Successfully connected to order book change stream")
                    async for change in stream:
                        try:
                            doc = change['fullDocument']
                            token = doc.get('metadata', {}).get('token')
                            if token:
                                try:
                                    latest = TransformedOrderBookDataPoint.from_mongodb_doc(doc)
                                    await websocket_manager.broadcast(
                                        json.dumps({
                                            'type': 'order_book_update',
                                            'token': token,
                                            'data': latest.dict()
                                        }, default=json_serial),
                                        'order_book'
                                    )
                                except Exception as e:
                                    logger.error(f"Error creating order book point: {e}")
                        except Exception as e:
                            logger.error(f"Error processing order book change: {e}")
            except Exception as e:
                if "replica sets" in str(e).lower():
                    logger.warning("Change streams not available, falling back to polling")
                    break  # Exit change stream mode and use polling
                raise  # Re-raise other errors
                
        except Exception as e:
            logger.error(f"Order book change stream error: {e}")
            await asyncio.sleep(5)  # Wait before retrying
    
    # Fall back to polling if change streams aren't available
    logger.info("Starting order book polling")
    while True:
        try:
            # Query for new documents since last check
            cursor = app.transformed_order_book_collection.find({
                'timestamp': {'$gt': last_checked}
            }).sort('timestamp', 1)
            
            async for doc in cursor:
                token = doc.get('metadata', {}).get('token')
                if token:
                    try:
                        latest = TransformedOrderBookDataPoint.from_mongodb_doc(doc)
                        await websocket_manager.broadcast(
                            json.dumps({
                                'type': 'order_book_update',
                                'token': token,
                                'data': latest.dict()
                            }, default=json_serial),
                            'order_book'
                        )
                        last_checked = doc['timestamp']
                    except Exception as e:
                        logger.error(f"Error creating order book point: {e}")
            
            await asyncio.sleep(1)  # Poll every second
            
        except Exception as e:
            logger.error(f"Order book polling error: {e}")
            await asyncio.sleep(5)  # Wait before retrying

@app.on_event("startup")
async def startup_event():
    # Start change stream watchers
    asyncio.create_task(watch_price_changes())
    asyncio.create_task(watch_order_book_changes())
    logger.info("Started MongoDB change stream watchers")


# --- Pydantic Models ---


class TransformedOrderBookDataPoint(BaseModel):
    timestamp: datetime.datetime
    lowest_ask: float
    highest_bid: float
    midpoint: float
    spread: float
    
    class Config:
        json_encoders = {
            datetime.datetime: lambda dt: dt.isoformat()
        }
    
    @classmethod
    def from_mongodb_doc(cls, doc: Dict[str, Any]) -> 'TransformedOrderBookDataPoint':
        """Create a TransformedOrderBookDataPoint from a MongoDB document."""
        if 'metadata' in doc and 'lowest_ask' in doc['metadata']:
            # Handle documents where fields are nested under metadata
            return cls(
                timestamp=doc['timestamp'],
                lowest_ask=doc['metadata']['lowest_ask'],
                highest_bid=doc['metadata']['highest_bid'],
                midpoint=doc['metadata'].get('midpoint', (doc['metadata']['lowest_ask'] + doc['metadata']['highest_bid']) / 2),
                spread=doc['metadata'].get('spread', abs(doc['metadata']['lowest_ask'] - doc['metadata']['highest_bid']))
            )
        else:
            # Handle documents where fields are at the top level
            return cls(
                timestamp=doc['timestamp'],
                lowest_ask=doc['lowest_ask'],
                highest_bid=doc['highest_bid'],
                midpoint=doc.get('midpoint', (doc['lowest_ask'] + doc['highest_bid']) / 2),
                spread=doc.get('spread', abs(doc['lowest_ask'] - doc['highest_bid']))
            )

class TransformedOrderBookData(BaseModel):
    token: str
    points: list[TransformedOrderBookDataPoint]
    
class PriceBucket(BaseModel):
    range: str
    avg_price: float
    volume: float

class PriceOutlier(BaseModel):
    price: float
    volume: float

class OrderBookSummaryData(BaseModel):
    bid_buckets: List[PriceBucket] = []
    ask_buckets: List[PriceBucket] = []
    bid_outliers: List[PriceOutlier] = []
    ask_outliers: List[PriceOutlier] = []

class LatestPriceData(BaseModel):
    price: float
    timestamp: datetime.datetime
    order_book: Optional[OrderBookSummaryData] = None

class CandlestickData(BaseModel):
    timestamp: datetime.datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None  # Traditional candlestick volume (if you have it)
    exchange_count: Optional[float] = None
    order_book: Optional[OrderBookSummaryData] = None

    class Config:
        json_encoders = {
            datetime.datetime: lambda dt: dt.isoformat()
        }

# --- Helper Functions ---

def process_order_book_data(book_data: Dict[str, Any]) -> OrderBookSummaryData:
    """Convert raw order book data into structured OrderBookSummaryData."""
    result = OrderBookSummaryData()
    
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
            cursor = app.price_collection.aggregate(pipeline)
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
            cursor = app.price_collection.aggregate(pipeline)
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
        document = await app.price_collection.find_one(
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

async def get_latest_transformed_order_book_point(token: str) -> Optional[TransformedOrderBookDataPoint]:
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
    token: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    granularity: int
) -> Optional[TransformedOrderBookData]:
    """Retrieves the transformed order book for a given token."""
    pipeline = [
        {
            "$match": {
                "metadata.token": token,                
                "timestamp": {"$gte": start_time, "$lte": end_time},
            }
        },
        {
            "$sort": {"timestamp": 1}  # Ensure data is sorted by timestamp
        },
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
                "lowest_ask": {"$min": "$metadata.lowest_ask"},
                "highest_bid": {"$max": "$metadata.highest_bid"},
                "midpoint": {"$avg": [{"$min": "$metadata.lowest_ask"}, {"$max": "$metadata.highest_bid"}]},
                "spread": {"$abs": {"$subtract": [{"$min": "$metadata.lowest_ask"}, {"$max": "$metadata.highest_bid"}]}},
            }
        },
        {
            "$sort": {"_id": 1}  # Sort by timestamp (ascending)
        },
        {
            "$project": {
                "_id": 0,
                "timestamp": "$_id",
                "lowest_ask": 1,
                "highest_bid": 1,
                "midpoint": 1,
                "spread": 1,
            }
        },
    ]
    try:
        cursor = await app.transformed_order_book_collection.aggregate(pipeline)
        if cursor:
            return TransformedOrderBookData(
                token=token,
                points=[TransformedOrderBookDataPoint(**doc) for doc in cursor]
            )
        else:
            return None  # No data found for the token

    except Exception as e:
        logger.error(f"Database error in get_transformed_order_book: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

# --- WebSocket Endpoints ---

@app.websocket("/ws/price/{token}")
async def websocket_price(websocket: WebSocket, token: str):
    await websocket_manager.connect(websocket, 'price')
    
    async def send_message(message: dict):
        try:
            await websocket.send_text(json.dumps(message, default=json_serial))
            return True
        except (WebSocketDisconnect, RuntimeError) as e:
            logger.debug(f"Failed to send message (connection closed): {e}")
            return False
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    try:
        # Send current price immediately on connection
        price_data = await get_latest_price(token)
        if price_data and not await send_message({
            'type': 'price_update',
            'token': token,
            'data': price_data.dict() if hasattr(price_data, 'dict') else price_data
        }):
            return  # Stop if we couldn't send the initial message
        
        # Keep connection alive with heartbeats
        while True:
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            if not await send_message({'type': 'ping'}):
                break  # Stop if we can't send a heartbeat
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for token {token}")
    except Exception as e:
        logger.error(f"WebSocket error for token {token}: {e}")
    finally:
        websocket_manager.disconnect(websocket, 'price')

@app.websocket("/ws/order_book/{token}")
async def websocket_order_book(websocket: WebSocket, token: str):
    await websocket_manager.connect(websocket, 'order_book')
    
    async def send_message(message: dict):
        try:
            await websocket.send_text(json.dumps(message, default=json_serial))
            return True
        except (WebSocketDisconnect, RuntimeError) as e:
            logger.debug(f"Failed to send order book message (connection closed): {e}")
            return False
        except Exception as e:
            logger.error(f"Error sending order book message: {e}")
            return False
    
    try:
        # Send current order book immediately on connection
        order_book = await get_latest_transformed_order_book_point(token)
        if order_book and not await send_message({
            'type': 'order_book_update',
            'token': token,
            'data': order_book.dict()
        }):
            return  # Stop if we couldn't send the initial message
        
        # Keep connection alive with heartbeats
        while True:
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            if not await send_message({'type': 'ping'}):
                break  # Stop if we can't send a heartbeat
                
    except WebSocketDisconnect:
        logger.info(f"Order book WebSocket disconnected for token {token}")
    except Exception as e:
        logger.error(f"Order book WebSocket error for token {token}: {e}")
    finally:
        websocket_manager.disconnect(websocket, 'order_book')

# --- REST API Endpoints ---

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

@app.get("/transformed_order_book/{token}", response_model=TransformedOrderBookData)
async def read_transformed_order_book(token: str):
    """
    Retrieve the transformed order book for a specific token.
    """
    transformed_order_book = await get_transformed_order_book(token)
    if transformed_order_book is None:
        raise HTTPException(status_code=404, detail=f"No data found for token {token}")
    return transformed_order_book

@app.get("/latest_transformed_order_book_point/{token}", response_model=TransformedOrderBookDataPoint)
async def read_latest_transformed_order_book_point(token: str):
    """
    Retrieve the latest transformed order book point for a specific token.
    """
    transformed_order_book_point = await get_latest_transformed_order_book_point(token)
    if transformed_order_book_point is None:
        raise HTTPException(status_code=404, detail=f"No data found for token {token}")
    return transformed_order_book_point

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

# Add this helper function to handle JSON serialization
def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if hasattr(obj, 'dict'):
        return obj.dict()
    raise TypeError(f"Type {type(obj)} not serializable")