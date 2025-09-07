
import json
import asyncio
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pymongo import DESCENDING
from typing import List, Optional, Dict, Any, Set, Tuple
import datetime
import datetime as dt
import logging
from pydantic import BaseModel
import pytz
from cryptotrading.data.mongo import get_db
from cryptotrading.config import PRICE_COLLECTION_NAME, TRANSFORMED_ORDER_BOOK_COLLECTION_NAME


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
app.use_stream_watch = False
logger.info("Successfully connected to MongoDB.")


# Background task for MongoDB change streams
async def watch_price_changes():
    """Background task that polls for price changes and broadcasts them to WebSocket clients."""
    logger.info("Starting price polling")
    last_checked = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=10)  # Look back 10 seconds initially
    
    while True:
        try:
            # Get list of tokens we're tracking (if any)
            tokens = set()
            for ws_set in websocket_manager.active_connections['price']:                
                # Extract token from WebSocket query parameters
                try:
                    token = ws_set.scope.get('path_params', {}).get('token', '')
                    if token:
                        tokens.add(token)
                except Exception as e:
                    logger.warning(f"Error parsing query string: {e}")
                    continue
            
            if not tokens:
                await asyncio.sleep(1)
                continue
                
            # Get latest price for each token
            current_time = datetime.datetime.now(datetime.timezone.utc)
            for token in tokens:
                # Query for new documents since last check
                cursor = app.price_collection.find({
                    'metadata.token': token,
                    'timestamp': {'$gt': last_checked}
                }).sort('timestamp', 1)
            
            if not tokens:
                logger.info("No tokens to track, waiting...")
                await asyncio.sleep(1)
                continue
                
            # Get latest price for each token
            current_time = datetime.datetime.now(datetime.timezone.utc)
            for token in tokens:
                # Query for new documents since last check
                cursor = app.price_collection.find({
                    'metadata.token': token,
                    'timestamp': {'$gt': last_checked}
                }).sort('timestamp', 1)
            
                async for doc in cursor:
                    token = doc.get('metadata', {}).get('token')
                    if token:
                        try:
                            latest = PriceDataPoint.from_mongodb_doc(doc)
                            if latest.timestamp.astimezone(dt.timezone.utc) > last_checked.astimezone(dt.timezone.utc):
                                await websocket_manager.broadcast(
                                    json.dumps({
                                        'type': 'price_update',
                                        'token': token,
                                        'data': latest.dict()
                                    }, default=json_serial),
                                    'price'
                                )                                
                        except Exception as e:
                            logger.error(f"Error creating order book point: {e}")
                
            last_checked = current_time
            await asyncio.sleep(1)  # Poll every second
            
        except Exception as e:
            logger.error(f"Error in price polling: {e}")
            await asyncio.sleep(5)  # Wait longer on error

async def watch_order_book_changes():
    logger.info("Starting order book polling")
    last_checked = datetime.datetime.now(datetime.timezone.utc)
    
    while True:
        try:
            if app.use_stream_watch:
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
                        app.use_stream_watch = False
            else:
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
    
class PaginatedResponse(BaseModel):
    data: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool

class PriceDataPoint(BaseModel):
    timestamp: datetime.datetime
    price: float
    token: str

    @classmethod
    def from_mongodb_doc(cls, doc: Dict[str, Any]) -> 'PriceDataPoint':
        """Create a PriceDataPoint from a MongoDB document."""
        return cls(
            timestamp=doc['timestamp'],
            price=doc['price'],
            token=doc['metadata']['token']
        )
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

async def get_historic_price(
    token: str, 
    start_time: datetime.datetime,
    end_time: datetime.datetime,
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

async def get_latest_price(token: str) -> Optional[Dict[str, Any]]:
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
            "timestamp": document.get('timestamp'),
            "metadata": document.get('metadata', {}),
            "order_book": order_book
        }
        
        logger.debug(f"Retrieved latest price for {token}: {result.get('price')}")
        return result
        
    except Exception as e:
        logger.error(f"Database error in get_latest_price for token {token}: {e}")
        return None  # Return None instead of raising to allow graceful degradation

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
                    "timestamp": datetime.datetime.fromtimestamp(bucket_time / 1000, tz=datetime.timezone.utc),
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
@app.get("/historic/price/{token}", response_model=PaginatedResponse)
async def read_historic_price(
    token: str,
    start: datetime.datetime = Query(..., description="Start time (ISO 8601 format)"),
    end: datetime.datetime = Query(..., description="End time (ISO 8601 format)"),
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(1000, ge=1, le=10000, description="Number of items per page (max 10000)")
):
    """
    Get paginated historic price data for a token within a time range.
    
    This endpoint returns price data in pages to improve performance with large datasets.
    """
    try:
        # Convert start and end to UTC
        start_utc = start.astimezone(pytz.utc)
        end_utc = end.astimezone(pytz.utc)

        if end_utc <= start_utc:
            raise HTTPException(status_code=400, detail="End time must be after start time")

        # Fetch paginated historic price data
        price_data, total_count = await get_historic_price(
            token=token,
            start_time=start_utc,
            end_time=end_utc,
            page=page,
            page_size=page_size
        )
        
        # Calculate pagination metadata
        total_pages = (total_count + page_size - 1) // page_size if page_size > 0 else 1
        has_next = page < total_pages
        has_previous = page > 1
        
        return {
            "data": price_data,
            "total": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "has_next": has_next,
            "has_previous": has_previous
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching historic price data for {token}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching historic price data: {e}")

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