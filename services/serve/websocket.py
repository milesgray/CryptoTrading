
import asyncio
from fastapi import WebSocket
from typing import Dict, Set
import logging
from starlette.websockets import WebSocketDisconnect
from websockets.exceptions import ConnectionClosed

try:
    from uvicorn.protocols.utils import ClientDisconnected
except ImportError:
    ClientDisconnected = None

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

    async def disconnect(self, websocket: WebSocket, channel: str):
        async with self.lock:
            if websocket in self.active_connections[channel]:
                self.active_connections[channel].remove(websocket)

    async def broadcast(self, message: str, channel: str):
        if channel not in self.active_connections:
            return
            
        disconnected = set()
        # Use list to avoid RuntimeError if active_connections is modified during iteration
        for connection in list(self.active_connections[channel]):
            try:
                await connection.send_text(message)
            except (WebSocketDisconnect, ConnectionClosed) as e:
                logger.debug(f"WebSocket client disconnected on {channel} channel: {e}")
                disconnected.add(connection)
            except Exception as e:
                if ClientDisconnected and isinstance(e, ClientDisconnected):
                    logger.debug(f"WebSocket client disconnected on {channel} channel: {e}")
                else:
                    import traceback
                    logger.warning(f"Error sending to WebSocket on {channel} channel: {e}")
                    logger.warning(traceback.format_exc())
                disconnected.add(connection)
        
        if disconnected:
            async with self.lock:
                self.active_connections[channel] -= disconnected


websocket_manager = ConnectionManager()

