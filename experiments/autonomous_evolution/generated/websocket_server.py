#!/usr/bin/env python3
"""
WEBSOCKET SERVER - Iteration 2.4
=================================
Real-time bidirectional communication with rooms, broadcast, presence
"""

import asyncio
from typing import Dict, Set, Callable, Any
from dataclasses import dataclass
import json
import uuid

@dataclass
class WebSocketClient:
    client_id: str
    websocket: Any  # websocket connection
    rooms: Set[str]
    metadata: Dict[str, Any]

class WebSocketServer:
    """WebSocket server with rooms and broadcasting"""
    
    def __init__(self):
        self.clients: Dict[str, WebSocketClient] = {}
        self.rooms: Dict[str, Set[str]] = {}  # room -> client_ids
        self.handlers: Dict[str, Callable] = {}
        
    def register_handler(self, event: str, handler: Callable):
        """Register message handler"""
        self.handlers[event] = handler
        
    async def connect(self, websocket: Any, metadata: Dict = None) -> str:
        """Handle new connection"""
        client_id = str(uuid.uuid4())
        client = WebSocketClient(
            client_id=client_id,
            websocket=websocket,
            rooms=set(),
            metadata=metadata or {}
        )
        self.clients[client_id] = client
        return client_id
        
    async def disconnect(self, client_id: str):
        """Handle disconnection"""
        client = self.clients.get(client_id)
        if client:
            # Leave all rooms
            for room in list(client.rooms):
                await self.leave_room(client_id, room)
            del self.clients[client_id]
            
    async def join_room(self, client_id: str, room: str):
        """Join a room"""
        client = self.clients.get(client_id)
        if client:
            client.rooms.add(room)
            if room not in self.rooms:
                self.rooms[room] = set()
            self.rooms[room].add(client_id)
            
    async def leave_room(self, client_id: str, room: str):
        """Leave a room"""
        client = self.clients.get(client_id)
        if client and room in client.rooms:
            client.rooms.remove(room)
            if room in self.rooms:
                self.rooms[room].discard(client_id)
                if not self.rooms[room]:
                    del self.rooms[room]
                    
    async def send_to_client(self, client_id: str, message: Dict):
        """Send message to specific client"""
        client = self.clients.get(client_id)
        if client:
            await self._send(client.websocket, message)
            
    async def broadcast_to_room(self, room: str, message: Dict, exclude: str = None):
        """Broadcast to all clients in room"""
        client_ids = self.rooms.get(room, set())
        for client_id in client_ids:
            if client_id != exclude:
                await self.send_to_client(client_id, message)
                
    async def broadcast_to_all(self, message: Dict, exclude: str = None):
        """Broadcast to all connected clients"""
        for client_id in self.clients:
            if client_id != exclude:
                await self.send_to_client(client_id, message)
                
    async def handle_message(self, client_id: str, message: Dict):
        """Handle incoming message"""
        event = message.get('event')
        data = message.get('data', {})
        
        handler = self.handlers.get(event)
        if handler:
            await handler(client_id, data)
            
    async def _send(self, websocket: Any, message: Dict):
        """Send JSON message"""
        # In real implementation would use actual websocket send
        await asyncio.sleep(0)  # Simulated async send

# 120 lines - Focused WebSocket server
