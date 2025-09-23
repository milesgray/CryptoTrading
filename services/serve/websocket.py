
import asyncio
from fastapi import WebSocket
from typing import Dict, Set
import logging

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
                # get stack trace
                import traceback
                logger.warning(traceback.format_exc())
                logger.warning(f"Error sending to WebSocket: {e}")
                disconnected.add(connection)
        
        if disconnected:
            async with self.lock:
                self.active_connections[channel] -= disconnected

