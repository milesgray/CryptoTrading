import json
import logging
import datetime as dt
from datetime import datetime
import pytz
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from cryptotrading.data.mongo import get_db
from cryptotrading.data.factory import get_price_adapter, get_order_book_adapter
from cryptotrading.config import PRICE_COLLECTION_NAME, TRANSFORMED_ORDER_BOOK_COLLECTION_NAME, DB_BACKEND
try:
    from .models import (
        PriceDataPoint,
        TransformedOrderBookDataPoint
    )
    from .data import process_order_book_data
    from .websocket import websocket_manager

    from .routers.market import router as market_router
    from .routers.retrieval import router as retrieval_router
    from .routers.services import router as services_router, ws_router as services_ws_router
    from .routers.broker import router as broker_router
except ImportError:
    from models import (
        PriceDataPoint,
        TransformedOrderBookDataPoint
    )
    from data import process_order_book_data
    from websocket import websocket_manager

    from routers.market import router as market_router
    from routers.retrieval import router as retrieval_router
    from routers.services import router as services_router, ws_router as services_ws_router
    from routers.broker import router as broker_router

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fastapi_server")

app = FastAPI(title="Crypto Price API", 
             description="Provides candlestick data for cryptocurrency prices with WebSocket support for real-time updates.", 
             version="1.0")

# --- CORS (Cross-Origin Resource Sharing) Configuration ---
origins = [
    "http://localhost:3000",  # Default React port
    "http://localhost:8000",  # Default FastAPI port
    "http://localhost:8080",  # Common alternative port
    "http://localhost:5173",  # Vite dev server
    "https://your-frontend-domain.com",  # Production frontend
]

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

app.price_adapter = get_price_adapter()
app.order_book_adapter = get_order_book_adapter()

if DB_BACKEND == 'mongodb':
    app.db = get_db()
    app.price_collection = app.db[PRICE_COLLECTION_NAME]
    app.transformed_order_book_collection = app.db[TRANSFORMED_ORDER_BOOK_COLLECTION_NAME]
    app.use_stream_watch = False
    logger.info("Successfully connected to MongoDB.")
else:
    app.db = None
    app.price_collection = None
    app.transformed_order_book_collection = None
    app.use_stream_watch = False
    logger.info("Successfully connected to PostgreSQL/TimescaleDB.")


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, dt.date)):
        return obj.isoformat()
    if hasattr(obj, 'dict'):
        return obj.dict()
    raise TypeError(f"Type {type(obj)} not serializable")


# Background task for database change streams
async def watch_price_changes():
    """Background task that polls for price changes and broadcasts them to WebSocket clients."""
    logger.info("Starting price polling")
    last_checked = datetime.now(dt.timezone.utc) - dt.timedelta(seconds=10)  # Look back 10 seconds initially
    
    while True:
        try:
            # Get list of tokens we're tracking (if any)
            tokens = set()
            for ws_set in websocket_manager.active_connections['price']:                
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
                
            current_time = datetime.now(dt.timezone.utc)
            for token in tokens:
                if DB_BACKEND == 'postgres':
                    raw_docs = await app.price_adapter.get_price_data(
                        token, 
                        start_time=last_checked, 
                        end_time=current_time, 
                        limit=100, 
                        sort="asc"
                    )
                    for doc in raw_docs:
                        token_meta = doc.get('metadata', {}).get('token')
                        if token_meta:
                            try:
                                order_book = process_order_book_data(doc['metadata'].get('book', {}))
                                result = {
                                    "price": doc["price"],
                                    "volume": order_book.volume if order_book else None,
                                    "timestamp": doc["timestamp"],
                                    "order_book": order_book
                                }
                                await websocket_manager.broadcast(
                                    json.dumps({
                                        'type': 'price_update',
                                        'token': token_meta,
                                        'data': result
                                    }, default=json_serial),
                                    'price'
                                )
                            except Exception as e:
                                logger.error(f"Error creating order book point: {e}")
                else:
                    cursor = app.price_collection.find({
                        'metadata.token': token,
                        'timestamp': {'$gt': last_checked}
                    }).sort('timestamp', 1)
                    
                    async for doc in cursor:
                        token_meta = doc.get('metadata', {}).get('token')
                        if token_meta:
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
                                            'token': token_meta,
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
            if DB_BACKEND == 'postgres':
                raw_docs = await app.order_book_adapter.get_transformed_order_book_since(last_checked)
                for doc in raw_docs:
                    token = doc.get('metadata', {}).get('token')
                    if token:
                        try:
                            meta = doc.get('metadata', {})
                            latest = {
                                "timestamp": doc["timestamp"],
                                "lowest_ask": meta.get("lowest_ask"),
                                "highest_bid": meta.get("highest_bid"),
                                "midpoint": meta.get("midpoint", doc["midpoint"]),
                                "spread": meta.get("spread")
                            }
                            await websocket_manager.broadcast(
                                json.dumps({
                                    'type': 'order_book_update',
                                    'token': token,
                                    'data': latest
                                }, default=json_serial),
                                'order_book'
                            )
                            last_checked = doc['timestamp']
                        except Exception as e:
                            logger.error(f"Error creating order book point: {e}")
            else:
                if app.use_stream_watch:
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
    logger.info("Started database change stream watchers")


# --- REST API Endpoints ---
@app.get("/health")
async def health_check():
    """
    Health check endpoint. Returns 200 OK if the server is running and can connect to the database.
    """
    try:
        # Check database connection based on backend
        if DB_BACKEND == 'mongodb':
            await app.db.command("ping")
        else:
            from cryptotrading.data.postgres import get_connection
            async with get_connection() as conn:
                await conn.execute("SELECT 1")
        return {"status": "OK", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail={"status": "unhealthy", "database": "disconnected", "error": str(e)})


# --- Include Routers ---
app.include_router(market_router)
app.include_router(retrieval_router)
app.include_router(services_router)
app.include_router(services_ws_router)
app.include_router(broker_router)