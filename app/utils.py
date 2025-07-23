from fastapi import WebSocket
from typing import List
import logging

logger = logging.getLogger(__name__)

# --- WebSocket Manager ---
# This is a simple class to manage active WebSocket connections.
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_json(self, data: dict, websocket: WebSocket):
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.warning(f"Could not send message to a websocket: {e}")
            self.disconnect(websocket)

    async def broadcast_json(self, data: dict):
        """Broadcasts a JSON message to all active connections."""
        for connection in self.active_connections[:]:
            await self.send_json(data, connection)