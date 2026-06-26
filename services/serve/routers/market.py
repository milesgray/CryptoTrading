import json
import logging
import asyncio
import datetime as dt
from datetime import datetime
from typing import List
import pytz

from fastapi import APIRouter, HTTPException, Query, WebSocket, Request
from starlette.websockets import WebSocketDisconnect
from websockets.exceptions import ConnectionClosed

from ..models import (
    PaginatedResponse,
    LatestPriceData,
    TransformedOrderBookData,
    TransformedOrderBookDataPoint,
    CandlestickData
)
from ..data import (
    get_latest_price,
    get_latest_transformed_order_book_point,
    get_transformed_order_book,
    get_historic_price,
    get_candlestick_data
)
from ..websocket import websocket_manager

logger = logging.getLogger("fastapi_server")

router = APIRouter()

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, dt.date)):
        return obj.isoformat()
    if hasattr(obj, 'dict'):
        return obj.dict()
    raise TypeError(f"Type {type(obj)} not serializable")


@router.websocket("/ws/price/{token}")
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
        price_data = await get_latest_price(websocket.app, token)
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
        await websocket_manager.disconnect(websocket, 'price')


@router.websocket("/ws/order_book/{token}")
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
        order_book = await get_latest_transformed_order_book_point(websocket.app, token)
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
        await websocket_manager.disconnect(websocket, 'order_book')


# --- REST API Endpoints ---
@router.get("/historic/price/{token}", response_model=PaginatedResponse)
async def read_historic_price(
    request: Request,
    token: str,
    start: datetime = Query(..., description="Start time (ISO 8601 format)"),
    end: datetime = Query(..., description="End time (ISO 8601 format)"),
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(1000, ge=1, le=10000, description="Number of items per page (max 10000)")
):
    """
    Get paginated historic price data for a token within a time range.
    """
    try:
        # Convert start and end to UTC
        start_utc = start.astimezone(pytz.utc)
        end_utc = end.astimezone(pytz.utc)

        if end_utc <= start_utc:
            raise HTTPException(status_code=400, detail="End time must be after start time")

        # Fetch paginated historic price data
        price_data, total_count = await get_historic_price(
            request.app,
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


@router.get("/candlestick/{token}", response_model=List[CandlestickData])
async def read_candlestick(
    request: Request,
    token: str,
    start: datetime = Query(..., description="Start time (ISO 8601 format)"),
    end: datetime = Query(..., description="End time (ISO 8601 format)"),
    granularity: int = Query(..., description="Candlestick interval in seconds"),
    include_book: bool = Query(False, description="Include order book data in response")
):
    """
    Retrieve candlestick data for a specific symbol within a given time range and granularity.
    """
    try:
        # Convert start and end to UTC
        start_utc = start.astimezone(pytz.utc)
        end_utc = end.astimezone(pytz.utc)

        if end_utc <= start_utc:
            raise HTTPException(status_code=400, detail="End time must be greater than start time.")
        if granularity <= 0:
            raise HTTPException(status_code=400, detail="Granularity must be a positive integer.")
        data = await get_candlestick_data(request.app, token, start_utc, end_utc, granularity, include_book)

        if not data:
             raise HTTPException(status_code=404, detail=f"No data found for {token} between {start_utc} and {end_utc}")
        return data
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {ve}")


@router.get("/latest_price/{token}", response_model=LatestPriceData)
async def read_latest_price(request: Request, token: str):
    """
    Retrieve the latest index price for a specific token.
    """
    price_data = await get_latest_price(request.app, token)
    if price_data is None:
        raise HTTPException(status_code=404, detail=f"No data found for token {token}")
    return price_data


@router.get("/transformed_order_book/{token}", response_model=TransformedOrderBookData)
async def read_transformed_order_book(request: Request, token: str):
    """
    Retrieve the transformed order book for a specific token.
    """
    transformed_order_book = await get_transformed_order_book(request.app, token)
    if transformed_order_book is None:
        raise HTTPException(status_code=404, detail=f"No data found for token {token}")
    return transformed_order_book


@router.get("/latest_transformed_order_book_point/{token}", response_model=TransformedOrderBookDataPoint)
async def read_latest_transformed_order_book_point(request: Request, token: str):
    """
    Retrieve the latest transformed order book point for a specific token.
    """
    transformed_order_book_point = await get_latest_transformed_order_book_point(request.app, token)
    if transformed_order_book_point is None:
        raise HTTPException(status_code=404, detail=f"No data found for token {token}")
    return transformed_order_book_point
