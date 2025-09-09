"""
WebSocket Manager - Real-time communication service for the Strategic Planning Platform.

Provides real-time updates for PRD generation progress, validation results,
agent orchestration status, and system notifications.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from enum import Enum
import uuid

import structlog
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from core.config import get_settings
from core.database import get_redis

logger = structlog.get_logger(__name__)
settings = get_settings()


class MessageType(str, Enum):
    """WebSocket message types."""
    # Connection management
    CONNECTION_ESTABLISHED = "connection_established"
    CONNECTION_CLOSED = "connection_closed"
    
    # PRD Generation updates
    PRD_GENERATION_STARTED = "prd_generation_started"
    PRD_PHASE_STARTED = "prd_phase_started"
    PRD_PHASE_COMPLETED = "prd_phase_completed"
    PRD_VALIDATION_UPDATE = "prd_validation_update"
    PRD_GENERATION_COMPLETED = "prd_generation_completed"
    PRD_GENERATION_FAILED = "prd_generation_failed"
    
    # Agent orchestration updates
    AGENT_TASK_STARTED = "agent_task_started"
    AGENT_TASK_COMPLETED = "agent_task_completed"
    AGENT_TASK_FAILED = "agent_task_failed"
    WORKFLOW_STATUS_UPDATE = "workflow_status_update"
    
    # System notifications
    SYSTEM_ALERT = "system_alert"
    SYSTEM_MAINTENANCE = "system_maintenance"
    
    # User-specific notifications
    USER_NOTIFICATION = "user_notification"
    
    # Heartbeat and health
    HEARTBEAT = "heartbeat"
    HEALTH_CHECK = "health_check"


class WebSocketMessage(BaseModel):
    """WebSocket message structure."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Message ID")
    type: MessageType = Field(..., description="Message type")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    user_id: Optional[str] = Field(None, description="Target user ID")
    session_id: Optional[str] = Field(None, description="WebSocket session ID")
    data: Dict[str, Any] = Field(default_factory=dict, description="Message payload")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ConnectionInfo(BaseModel):
    """WebSocket connection information."""
    connection_id: str = Field(..., description="Unique connection ID")
    user_id: Optional[str] = Field(None, description="Authenticated user ID")
    connected_at: datetime = Field(default_factory=datetime.utcnow, description="Connection timestamp")
    last_activity: datetime = Field(default_factory=datetime.utcnow, description="Last activity timestamp")
    subscription_filters: List[str] = Field(default_factory=list, description="Message type filters")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Connection metadata")


class WebSocketManager:
    """
    WebSocket Manager for real-time communication.
    
    Features:
    - Connection management with authentication
    - Message broadcasting and filtering
    - Subscription-based message delivery
    - Real-time PRD generation updates
    - Agent orchestration status updates
    - System health and maintenance notifications
    - Heartbeat and connection monitoring
    """
    
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.connection_info: Dict[str, ConnectionInfo] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.redis = None
        self.is_initialized = False
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Message delivery settings
        self.message_retention_minutes = 30
        self.max_connections_per_user = 5
        self.heartbeat_interval_seconds = 30
        self.connection_timeout_seconds = 300
    
    async def initialize(self) -> None:
        """Initialize the WebSocket manager."""
        try:
            self.redis = await get_redis()
            
            # Start background tasks
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self.is_initialized = True
            logger.info("WebSocket manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket manager: {str(e)}")
            raise
    
    async def connect(
        self, 
        websocket: WebSocket, 
        user_id: Optional[str] = None,
        connection_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Accept a new WebSocket connection."""
        
        await websocket.accept()
        
        connection_id = str(uuid.uuid4())
        
        # Store connection
        self.connections[connection_id] = websocket
        self.connection_info[connection_id] = ConnectionInfo(
            connection_id=connection_id,
            user_id=user_id,
            metadata=connection_metadata or {}
        )
        
        # Track user connections
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            
            # Enforce connection limit per user
            if len(self.user_connections[user_id]) >= self.max_connections_per_user:
                oldest_connection = min(
                    self.user_connections[user_id],
                    key=lambda cid: self.connection_info[cid].connected_at
                )
                await self.disconnect(oldest_connection, reason="Connection limit exceeded")
            
            self.user_connections[user_id].add(connection_id)
        
        # Send connection established message
        await self._send_to_connection(
            connection_id,
            WebSocketMessage(
                type=MessageType.CONNECTION_ESTABLISHED,
                data={
                    "connection_id": connection_id,
                    "server_time": datetime.utcnow().isoformat(),
                    "features": [
                        "prd_generation_updates",
                        "agent_orchestration_updates", 
                        "system_notifications",
                        "heartbeat_monitoring"
                    ]
                }
            )
        )
        
        logger.info(
            "WebSocket connection established",
            connection_id=connection_id,
            user_id=user_id,
            total_connections=len(self.connections)
        )
        
        return connection_id
    
    async def disconnect(
        self, 
        connection_id: str, 
        reason: str = "Client disconnected"
    ) -> None:
        """Disconnect a WebSocket connection."""
        
        if connection_id not in self.connections:
            return
        
        connection_info = self.connection_info.get(connection_id)
        websocket = self.connections[connection_id]
        
        try:
            # Send disconnection message
            await self._send_to_connection(
                connection_id,
                WebSocketMessage(
                    type=MessageType.CONNECTION_CLOSED,
                    data={"reason": reason}
                )
            )
        except:
            pass  # Connection may already be closed
        
        # Close connection
        try:
            await websocket.close()
        except:
            pass
        
        # Clean up tracking
        if connection_info and connection_info.user_id:
            user_id = connection_info.user_id
            if user_id in self.user_connections:
                self.user_connections[user_id].discard(connection_id)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
        
        # Remove from storage
        del self.connections[connection_id]
        if connection_id in self.connection_info:
            del self.connection_info[connection_id]
        
        logger.info(
            "WebSocket connection disconnected",
            connection_id=connection_id,
            reason=reason,
            remaining_connections=len(self.connections)
        )
    
    async def subscribe_to_messages(
        self, 
        connection_id: str, 
        message_types: List[MessageType]
    ) -> None:
        """Subscribe connection to specific message types."""
        
        if connection_id not in self.connection_info:
            raise ValueError(f"Connection {connection_id} not found")
        
        self.connection_info[connection_id].subscription_filters = [
            msg_type.value for msg_type in message_types
        ]
        
        logger.info(
            "Updated message subscriptions",
            connection_id=connection_id,
            subscriptions=message_types
        )
    
    async def send_message(self, message: WebSocketMessage) -> int:
        """Send message to appropriate connections based on filters."""
        
        sent_count = 0
        
        # Determine target connections
        target_connections = self._get_target_connections(message)
        
        # Send to each target connection
        for connection_id in target_connections:
            success = await self._send_to_connection(connection_id, message)
            if success:
                sent_count += 1
        
        # Store message in Redis for potential replay
        if self.redis:
            await self._store_message_for_replay(message)
        
        logger.debug(
            "Message sent",
            message_type=message.type.value,
            message_id=message.id,
            sent_count=sent_count,
            total_connections=len(target_connections)
        )
        
        return sent_count
    
    async def broadcast_to_all(self, message: WebSocketMessage) -> int:
        """Broadcast message to all connected clients."""
        
        sent_count = 0
        
        for connection_id in list(self.connections.keys()):
            success = await self._send_to_connection(connection_id, message)
            if success:
                sent_count += 1
        
        return sent_count
    
    async def send_to_user(self, user_id: str, message: WebSocketMessage) -> int:
        """Send message to all connections for a specific user."""
        
        if user_id not in self.user_connections:
            return 0
        
        sent_count = 0
        connection_ids = list(self.user_connections[user_id])
        
        for connection_id in connection_ids:
            success = await self._send_to_connection(connection_id, message)
            if success:
                sent_count += 1
        
        return sent_count
    
    async def send_prd_update(
        self, 
        user_id: str, 
        prd_id: str, 
        update_type: MessageType,
        update_data: Dict[str, Any]
    ) -> int:
        """Send PRD generation update to user."""
        
        message = WebSocketMessage(
            type=update_type,
            user_id=user_id,
            data={
                "prd_id": prd_id,
                **update_data
            },
            metadata={
                "category": "prd_generation",
                "priority": "high"
            }
        )
        
        return await self.send_to_user(user_id, message)
    
    async def send_agent_update(
        self,
        user_id: str,
        workflow_id: str,
        task_id: str,
        update_type: MessageType,
        update_data: Dict[str, Any]
    ) -> int:
        """Send agent orchestration update to user."""
        
        message = WebSocketMessage(
            type=update_type,
            user_id=user_id,
            data={
                "workflow_id": workflow_id,
                "task_id": task_id,
                **update_data
            },
            metadata={
                "category": "agent_orchestration",
                "priority": "medium"
            }
        )
        
        return await self.send_to_user(user_id, message)
    
    async def send_system_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "info",
        target_users: Optional[List[str]] = None
    ) -> int:
        """Send system alert to users."""
        
        ws_message = WebSocketMessage(
            type=MessageType.SYSTEM_ALERT,
            data={
                "alert_type": alert_type,
                "message": message,
                "severity": severity
            },
            metadata={
                "category": "system",
                "priority": "high" if severity in ["error", "critical"] else "medium"
            }
        )
        
        if target_users:
            sent_count = 0
            for user_id in target_users:
                sent_count += await self.send_to_user(user_id, ws_message)
            return sent_count
        else:
            return await self.broadcast_to_all(ws_message)
    
    def _get_target_connections(self, message: WebSocketMessage) -> List[str]:
        """Get target connections for a message based on filters."""
        
        target_connections = []
        
        for connection_id, connection_info in self.connection_info.items():
            # Check user targeting
            if message.user_id and connection_info.user_id != message.user_id:
                continue
            
            # Check subscription filters
            if connection_info.subscription_filters:
                if message.type.value not in connection_info.subscription_filters:
                    continue
            
            target_connections.append(connection_id)
        
        return target_connections
    
    async def _send_to_connection(
        self, 
        connection_id: str, 
        message: WebSocketMessage
    ) -> bool:
        """Send message to a specific connection."""
        
        if connection_id not in self.connections:
            return False
        
        websocket = self.connections[connection_id]
        
        try:
            # Update last activity
            if connection_id in self.connection_info:
                self.connection_info[connection_id].last_activity = datetime.utcnow()
            
            # Send message
            message_json = message.model_dump_json()
            await websocket.send_text(message_json)
            
            return True
            
        except WebSocketDisconnect:
            # Connection closed by client
            await self.disconnect(connection_id, "Client disconnected")
            return False
        except Exception as e:
            logger.error(
                f"Failed to send message to connection",
                connection_id=connection_id,
                error=str(e)
            )
            await self.disconnect(connection_id, f"Send error: {str(e)}")
            return False
    
    async def _store_message_for_replay(self, message: WebSocketMessage) -> None:
        """Store message in Redis for potential replay."""
        
        try:
            if not self.redis:
                return
            
            # Store with expiration
            key = f"websocket_messages:{message.user_id or 'broadcast'}:{message.id}"
            await self.redis.setex(
                key,
                self.message_retention_minutes * 60,
                message.model_dump_json()
            )
            
        except Exception as e:
            logger.warning(f"Failed to store message for replay: {str(e)}")
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat messages."""
        
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval_seconds)
                
                if not self.connections:
                    continue
                
                heartbeat_message = WebSocketMessage(
                    type=MessageType.HEARTBEAT,
                    data={
                        "server_time": datetime.utcnow().isoformat(),
                        "active_connections": len(self.connections)
                    }
                )
                
                # Send heartbeat to all connections
                await self.broadcast_to_all(heartbeat_message)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {str(e)}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _cleanup_loop(self) -> None:
        """Clean up stale connections."""
        
        while True:
            try:
                await asyncio.sleep(60)  # Run cleanup every minute
                
                current_time = datetime.utcnow()
                stale_connections = []
                
                for connection_id, connection_info in self.connection_info.items():
                    time_since_activity = (current_time - connection_info.last_activity).total_seconds()
                    
                    if time_since_activity > self.connection_timeout_seconds:
                        stale_connections.append(connection_id)
                
                # Disconnect stale connections
                for connection_id in stale_connections:
                    await self.disconnect(connection_id, "Connection timeout")
                
                if stale_connections:
                    logger.info(f"Cleaned up {len(stale_connections)} stale connections")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {str(e)}")
                await asyncio.sleep(10)
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics."""
        
        total_connections = len(self.connections)
        authenticated_connections = len([
            info for info in self.connection_info.values() 
            if info.user_id is not None
        ])
        
        user_distribution = {}
        for user_id, connection_ids in self.user_connections.items():
            user_distribution[user_id] = len(connection_ids)
        
        return {
            "total_connections": total_connections,
            "authenticated_connections": authenticated_connections,
            "anonymous_connections": total_connections - authenticated_connections,
            "unique_users": len(self.user_connections),
            "user_distribution": user_distribution,
            "average_connections_per_user": (
                sum(user_distribution.values()) / len(user_distribution) 
                if user_distribution else 0
            )
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check WebSocket manager health."""
        
        try:
            stats = await self.get_connection_stats()
            
            return {
                "status": "healthy" if self.is_initialized else "initializing",
                "initialized": self.is_initialized,
                "redis_connected": self.redis is not None,
                "heartbeat_active": self.heartbeat_task is not None and not self.heartbeat_task.done(),
                "cleanup_active": self.cleanup_task is not None and not self.cleanup_task.done(),
                "connection_stats": stats
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def shutdown(self) -> None:
        """Shutdown the WebSocket manager."""
        
        logger.info("Shutting down WebSocket manager")
        
        # Cancel background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Disconnect all connections
        connection_ids = list(self.connections.keys())
        for connection_id in connection_ids:
            await self.disconnect(connection_id, "Server shutdown")
        
        logger.info("WebSocket manager shutdown complete")


# Global WebSocket manager instance
websocket_manager = WebSocketManager()


async def get_websocket_manager() -> WebSocketManager:
    """Get the global WebSocket manager instance."""
    if not websocket_manager.is_initialized:
        await websocket_manager.initialize()
    return websocket_manager