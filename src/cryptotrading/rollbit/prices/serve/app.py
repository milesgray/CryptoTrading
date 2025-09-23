
import json
import asyncio
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pymongo import DESCENDING
from typing import List
from datetime import datetime
import datetime as dt
import logging
import pytz
from cryptotrading.data.mongo import get_db
from cryptotrading.config import PRICE_COLLECTION_NAME, TRANSFORMED_ORDER_BOOK_COLLECTION_NAME
from .models import (
    PriceDataPoint,
    TransformedOrderBookDataPoint,
    CandlestickData,
    PaginatedResponse,
    LatestPriceData,
    TransformedOrderBookData
)
from .data import (
    process_order_book_data,
    get_latest_transformed_order_book_point,
    get_latest_price,
    get_historic_price,
    get_candlestick_data    
)
from .websocket import ConnectionManager

# Configure logging (optional, but good practice)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fastapi_server")

# Initialize WebSocket manager
websocket_manager = ConnectionManager()


app = FastAPI(title="Crypto Price API", 
             description="Provides candlestick data for cryptocurrency prices with WebSocket support for real-time updates.", 
             version="1.0")

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
    last_checked = datetime.now(dt.timezone.utc) - dt.timedelta(seconds=10)  # Look back 10 seconds initially
    
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
            current_time = datetime.now(dt.timezone.utc)
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
            current_time = datetime.now(dt.timezone.utc)
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
                                order_book = process_order_book_data(latest.book)
                                
                                result = {
                                    "price": latest.price,
                                    "volume": order_book.volume if order_book else None,
                                    "timestamp": latest.timestamp,
                                    "order_book": order_book
                                }
                                await websocket_manager.broadcast(
                                    json.dumps({
                                        'type': 'price_update',
                                        'token': token,
                                        'data': result
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
    last_checked = datetime.now(dt.timezone.utc)
    
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
        price_data = await get_latest_price(app, token)
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
        order_book = await get_latest_transformed_order_book_point(app, token)
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
    start: datetime = Query(..., description="Start time (ISO 8601 format)"),
    end: datetime = Query(..., description="End time (ISO 8601 format)"),
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
            app,
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
    start: datetime = Query(..., description="Start time (ISO 8601 format)"),
    end: datetime = Query(..., description="End time (ISO 8601 format)"),
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
        data = await get_candlestick_data(app, token, start_utc, end_utc, granularity, include_book)

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
    price_data = await get_latest_price(app, token)
    if price_data is None:
        raise HTTPException(status_code=404, detail=f"No data found for token {token}")
    return price_data

@app.get("/transformed_order_book/{token}", response_model=TransformedOrderBookData)
async def read_transformed_order_book(token: str):
    """
    Retrieve the transformed order book for a specific token.
    """
    transformed_order_book = await get_transformed_order_book(app, token)
    if transformed_order_book is None:
        raise HTTPException(status_code=404, detail=f"No data found for token {token}")
    return transformed_order_book

@app.get("/latest_transformed_order_book_point/{token}", response_model=TransformedOrderBookDataPoint)
async def read_latest_transformed_order_book_point(token: str):
    """
    Retrieve the latest transformed order book point for a specific token.
    """
    transformed_order_book_point = await get_latest_transformed_order_book_point(app, token)
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
    if isinstance(obj, (datetime, dt.date)):
        return obj.isoformat()
    if hasattr(obj, 'dict'):
        return obj.dict()
    raise TypeError(f"Type {type(obj)} not serializable")