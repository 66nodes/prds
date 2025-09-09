"""
WebSocket endpoints for real-time communication.
"""

from typing import Optional, Dict, Any
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

from services.websocket_manager import get_websocket_manager, WebSocketManager, MessageType
from services.collaboration_manager import (
    get_collaboration_manager,
    CollaborationManager,
    CollaborationType,
    Operation,
    OperationType,
    UserPresence
)
from core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()
security = HTTPBearer()

router = APIRouter()


async def get_user_from_token(token: Optional[str] = None) -> Optional[str]:
    """Extract user ID from JWT token (simplified for WebSocket)."""
    
    # In a real implementation, this would validate the JWT token
    # and extract the user ID. For now, we'll use a placeholder.
    
    if not token:
        return None
    
    # TODO: Implement proper JWT validation
    # For now, assume token format is "user_<user_id>"
    if token.startswith("user_"):
        return token[5:]  # Extract user_id part
    
    return None


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="Authentication token"),
    client_id: Optional[str] = Query(None, description="Client identifier")
):
    """
    WebSocket endpoint for real-time communication.
    
    Query Parameters:
    - token: JWT authentication token (optional)
    - client_id: Client identifier for tracking (optional)
    
    Supported message types:
    - prd_generation_updates: Real-time PRD generation progress
    - agent_orchestration_updates: Agent task execution status  
    - system_notifications: System alerts and maintenance
    - heartbeat: Connection health monitoring
    """
    
    websocket_manager = await get_websocket_manager()
    
    # Extract user ID from token
    user_id = await get_user_from_token(token)
    
    # Prepare connection metadata
    connection_metadata = {
        "client_id": client_id,
        "user_agent": websocket.headers.get("user-agent"),
        "origin": websocket.headers.get("origin"),
        "connected_at": websocket.headers.get("sec-websocket-accept")
    }
    
    connection_id = None
    
    try:
        # Establish connection
        connection_id = await websocket_manager.connect(
            websocket=websocket,
            user_id=user_id,
            connection_metadata=connection_metadata
        )
        
        logger.info(
            "WebSocket connection established",
            connection_id=connection_id,
            user_id=user_id,
            client_id=client_id
        )
        
        # Message handling loop
        while True:
            try:
                # Wait for client messages
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                # Handle client messages
                await handle_client_message(
                    websocket_manager, 
                    connection_id, 
                    user_id,
                    message_data
                )
                
            except WebSocketDisconnect:
                logger.info(
                    "WebSocket client disconnected",
                    connection_id=connection_id,
                    user_id=user_id
                )
                break
                
            except json.JSONDecodeError:
                # Send error message for invalid JSON
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": "now"
                }))
                
            except Exception as e:
                logger.error(
                    "Error processing WebSocket message",
                    connection_id=connection_id,
                    error=str(e),
                    exc_info=True
                )
                
                await websocket.send_text(json.dumps({
                    "type": "error", 
                    "message": "Message processing error",
                    "timestamp": "now"
                }))
    
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed during setup")
    
    except Exception as e:
        logger.error(
            "WebSocket connection error",
            connection_id=connection_id,
            error=str(e),
            exc_info=True
        )
    
    finally:
        # Clean up connection
        if connection_id:
            await websocket_manager.disconnect(
                connection_id, 
                "Connection closed"
            )


async def handle_client_message(
    websocket_manager: WebSocketManager,
    connection_id: str,
    user_id: Optional[str],
    message_data: Dict[str, Any]
) -> None:
    """Handle messages received from WebSocket clients."""
    
    message_type = message_data.get("type")
    
    if message_type == "subscribe":
        # Handle subscription requests
        await handle_subscription_request(
            websocket_manager, 
            connection_id, 
            message_data
        )
    
    elif message_type == "unsubscribe":
        # Handle unsubscription requests
        await handle_unsubscription_request(
            websocket_manager,
            connection_id,
            message_data
        )
    
    elif message_type == "ping":
        # Handle ping/pong for connection testing
        await handle_ping_request(
            websocket_manager,
            connection_id,
            message_data
        )
    
    elif message_type == "get_status":
        # Handle status requests
        await handle_status_request(
            websocket_manager,
            connection_id,
            user_id,
            message_data
        )
    
    elif message_type == "collaboration":
        # Handle collaboration messages
        await handle_collaboration_message(
            websocket_manager,
            connection_id,
            user_id,
            message_data
        )
    
    else:
        logger.warning(
            "Unknown WebSocket message type",
            connection_id=connection_id,
            message_type=message_type
        )


async def handle_subscription_request(
    websocket_manager: WebSocketManager,
    connection_id: str,
    message_data: Dict[str, Any]
) -> None:
    """Handle client subscription requests."""
    
    try:
        requested_types = message_data.get("message_types", [])
        
        # Validate and convert message types
        valid_types = []
        for msg_type in requested_types:
            try:
                valid_types.append(MessageType(msg_type))
            except ValueError:
                logger.warning(
                    "Invalid message type requested",
                    connection_id=connection_id,
                    message_type=msg_type
                )
        
        if valid_types:
            await websocket_manager.subscribe_to_messages(
                connection_id, 
                valid_types
            )
            
            logger.info(
                "Client subscribed to message types",
                connection_id=connection_id,
                subscriptions=valid_types
            )
    
    except Exception as e:
        logger.error(
            "Subscription request error",
            connection_id=connection_id,
            error=str(e)
        )


async def handle_unsubscription_request(
    websocket_manager: WebSocketManager,
    connection_id: str,
    message_data: Dict[str, Any]
) -> None:
    """Handle client unsubscription requests."""
    
    try:
        # Clear all subscriptions for now (could be more granular)
        await websocket_manager.subscribe_to_messages(connection_id, [])
        
        logger.info(
            "Client unsubscribed from all message types",
            connection_id=connection_id
        )
        
    except Exception as e:
        logger.error(
            "Unsubscription request error",
            connection_id=connection_id,
            error=str(e)
        )


async def handle_ping_request(
    websocket_manager: WebSocketManager,
    connection_id: str,
    message_data: Dict[str, Any]
) -> None:
    """Handle ping requests from clients."""
    
    try:
        from services.websocket_manager import WebSocketMessage
        
        pong_message = WebSocketMessage(
            type=MessageType.HEARTBEAT,
            data={
                "type": "pong",
                "client_timestamp": message_data.get("timestamp"),
                "server_timestamp": "now"
            }
        )
        
        await websocket_manager._send_to_connection(connection_id, pong_message)
        
    except Exception as e:
        logger.error(
            "Ping request error",
            connection_id=connection_id, 
            error=str(e)
        )


async def handle_status_request(
    websocket_manager: WebSocketManager,
    connection_id: str,
    user_id: Optional[str],
    message_data: Dict[str, Any]
) -> None:
    """Handle status requests from clients."""
    
    try:
        from services.websocket_manager import WebSocketMessage
        
        # Get connection stats
        stats = await websocket_manager.get_connection_stats()
        
        status_message = WebSocketMessage(
            type=MessageType.HEALTH_CHECK,
            data={
                "connection_id": connection_id,
                "user_id": user_id,
                "server_stats": stats,
                "timestamp": "now"
            }
        )
        
        await websocket_manager._send_to_connection(connection_id, status_message)
        
    except Exception as e:
        logger.error(
            "Status request error", 
            connection_id=connection_id,
            error=str(e)
        )


@router.get("/ws/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics."""
    
    try:
        websocket_manager = await get_websocket_manager()
        stats = await websocket_manager.get_connection_stats()
        health = await websocket_manager.health_check()
        
        return {
            "status": "success",
            "connection_stats": stats,
            "health_status": health,
            "timestamp": "now"
        }
        
    except Exception as e:
        logger.error(f"Failed to get WebSocket stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve WebSocket statistics"
        )


@router.post("/ws/broadcast")
async def broadcast_message(
    message_data: Dict[str, Any],
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Broadcast message to all connected WebSocket clients.
    (Admin only endpoint)
    """
    
    try:
        # TODO: Add proper admin authorization check
        # For now, we'll allow any authenticated request
        
        from services.websocket_manager import WebSocketMessage
        
        websocket_manager = await get_websocket_manager()
        
        # Create message
        message = WebSocketMessage(
            type=MessageType.SYSTEM_ALERT,
            data=message_data,
            metadata={"source": "admin_broadcast"}
        )
        
        # Broadcast to all connections
        sent_count = await websocket_manager.broadcast_to_all(message)
        
        return {
            "status": "success",
            "message": "Message broadcasted successfully",
            "recipients": sent_count,
            "timestamp": "now"
        }
        
    except Exception as e:
        logger.error(f"Failed to broadcast message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to broadcast message"
        )


@router.post("/ws/notify/{user_id}")
async def notify_user(
    user_id: str,
    message_data: Dict[str, Any],
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Send notification to specific user's WebSocket connections.
    """
    
    try:
        from services.websocket_manager import WebSocketMessage
        
        websocket_manager = await get_websocket_manager()
        
        # Create user notification message
        message = WebSocketMessage(
            type=MessageType.USER_NOTIFICATION,
            user_id=user_id,
            data=message_data,
            metadata={"source": "api_notification"}
        )
        
        # Send to user's connections
        sent_count = await websocket_manager.send_to_user(user_id, message)
        
        return {
            "status": "success",
            "message": f"Notification sent to user {user_id}",
            "connections_notified": sent_count,
            "timestamp": "now"
        }
        
    except Exception as e:
        logger.error(f"Failed to notify user: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to send user notification"
        )


async def handle_collaboration_message(
    websocket_manager: WebSocketManager,
    connection_id: str,
    user_id: Optional[str],
    message_data: Dict[str, Any]
) -> None:
    """Handle real-time collaboration messages."""
    
    collaboration_manager = await get_collaboration_manager()
    action = message_data.get("action")
    
    try:
        if action == "join_session":
            await handle_join_session(
                websocket_manager,
                collaboration_manager,
                connection_id,
                user_id,
                message_data
            )
        
        elif action == "leave_session":
            await handle_leave_session(
                websocket_manager,
                collaboration_manager,
                connection_id,
                user_id,
                message_data
            )
        
        elif action == "edit":
            await handle_collaboration_edit(
                websocket_manager,
                collaboration_manager,
                connection_id,
                user_id,
                message_data
            )
        
        elif action == "cursor_update":
            await handle_cursor_update(
                websocket_manager,
                collaboration_manager,
                connection_id,
                user_id,
                message_data
            )
        
        elif action == "get_session_state":
            await handle_get_session_state(
                websocket_manager,
                collaboration_manager,
                connection_id,
                message_data
            )
        
        else:
            logger.warning(
                "Unknown collaboration action",
                connection_id=connection_id,
                action=action
            )
    
    except Exception as e:
        logger.error(
            "Collaboration message error",
            connection_id=connection_id,
            action=action,
            error=str(e)
        )
        
        from services.websocket_manager import WebSocketMessage
        
        error_message = WebSocketMessage(
            type=MessageType.USER_NOTIFICATION,
            data={
                "type": "collaboration_error",
                "error": str(e),
                "action": action
            }
        )
        
        await websocket_manager._send_to_connection(connection_id, error_message)


async def handle_join_session(
    websocket_manager: WebSocketManager,
    collaboration_manager: CollaborationManager,
    connection_id: str,
    user_id: Optional[str],
    message_data: Dict[str, Any]
) -> None:
    """Handle joining a collaboration session."""
    
    if not user_id:
        raise ValueError("User authentication required for collaboration")
    
    document_id = message_data.get("document_id")
    username = message_data.get("username", f"User_{user_id[:8]}")
    initial_content = message_data.get("initial_content", "")
    
    if not document_id:
        raise ValueError("Document ID is required")
    
    # Get or create session for document
    session = await collaboration_manager.get_or_create_session(
        document_id,
        initial_content
    )
    
    # Join the session
    await collaboration_manager.join_session(
        session.session_id,
        user_id,
        username
    )
    
    # Get session state
    session_state = await collaboration_manager.get_session_state(session.session_id)
    
    # Send session joined notification to user
    from services.websocket_manager import WebSocketMessage
    
    join_message = WebSocketMessage(
        type=MessageType.USER_NOTIFICATION,
        data={
            "type": CollaborationType.USER_JOIN,
            "session_id": session.session_id,
            "document_id": document_id,
            "session_state": session_state,
            "user_id": user_id,
            "username": username
        }
    )
    
    await websocket_manager._send_to_connection(connection_id, join_message)
    
    # Notify other users in the session
    for other_user_id in session.document_state.active_users:
        if other_user_id != user_id:
            presence_message = WebSocketMessage(
                type=MessageType.USER_NOTIFICATION,
                user_id=other_user_id,
                data={
                    "type": CollaborationType.PRESENCE_UPDATE,
                    "session_id": session.session_id,
                    "new_user": {
                        "user_id": user_id,
                        "username": username
                    }
                }
            )
            await websocket_manager.send_to_user(other_user_id, presence_message)
    
    logger.info(
        "User joined collaboration session",
        connection_id=connection_id,
        user_id=user_id,
        session_id=session.session_id,
        document_id=document_id
    )


async def handle_leave_session(
    websocket_manager: WebSocketManager,
    collaboration_manager: CollaborationManager,
    connection_id: str,
    user_id: Optional[str],
    message_data: Dict[str, Any]
) -> None:
    """Handle leaving a collaboration session."""
    
    if not user_id:
        return
    
    session_id = message_data.get("session_id")
    if not session_id:
        return
    
    # Get session before leaving
    session = await collaboration_manager.get_session(session_id)
    if not session:
        return
    
    # Store other users before leaving
    other_users = [uid for uid in session.document_state.active_users if uid != user_id]
    
    # Leave the session
    await collaboration_manager.leave_session(session_id, user_id)
    
    # Notify other users
    from services.websocket_manager import WebSocketMessage
    
    for other_user_id in other_users:
        leave_message = WebSocketMessage(
            type=MessageType.USER_NOTIFICATION,
            user_id=other_user_id,
            data={
                "type": CollaborationType.USER_LEAVE,
                "session_id": session_id,
                "left_user_id": user_id
            }
        )
        await websocket_manager.send_to_user(other_user_id, leave_message)
    
    logger.info(
        "User left collaboration session",
        connection_id=connection_id,
        user_id=user_id,
        session_id=session_id
    )


async def handle_collaboration_edit(
    websocket_manager: WebSocketManager,
    collaboration_manager: CollaborationManager,
    connection_id: str,
    user_id: Optional[str],
    message_data: Dict[str, Any]
) -> None:
    """Handle collaborative edit operation."""
    
    if not user_id:
        raise ValueError("User authentication required for editing")
    
    session_id = message_data.get("session_id")
    operation_data = message_data.get("operation")
    
    if not session_id or not operation_data:
        raise ValueError("Session ID and operation are required")
    
    # Create operation from data
    operation = Operation(
        type=OperationType(operation_data.get("type")),
        position=operation_data.get("position"),
        content=operation_data.get("content"),
        length=operation_data.get("length"),
        user_id=user_id,
        version=operation_data.get("version", 0),
        parent_version=operation_data.get("parent_version"),
        metadata=operation_data.get("metadata", {})
    )
    
    # Apply the edit
    success, transformed_op, document_state = await collaboration_manager.apply_edit(
        session_id,
        operation
    )
    
    if not success:
        # Send conflict notification
        from services.websocket_manager import WebSocketMessage
        
        conflict_message = WebSocketMessage(
            type=MessageType.USER_NOTIFICATION,
            data={
                "type": CollaborationType.CONFLICT_DETECTED,
                "session_id": session_id,
                "operation": operation_data,
                "reason": "Operation could not be applied"
            }
        )
        await websocket_manager._send_to_connection(connection_id, conflict_message)
        return
    
    # Get session for broadcasting
    session = await collaboration_manager.get_session(session_id)
    if not session:
        return
    
    # Broadcast transformed operation to all users in session
    from services.websocket_manager import WebSocketMessage
    
    for session_user_id in session.document_state.active_users:
        edit_message = WebSocketMessage(
            type=MessageType.USER_NOTIFICATION,
            user_id=session_user_id,
            data={
                "type": CollaborationType.DOCUMENT_EDIT,
                "session_id": session_id,
                "operation": {
                    "id": transformed_op.id,
                    "type": transformed_op.type.value,
                    "position": transformed_op.position,
                    "content": transformed_op.content,
                    "length": transformed_op.length,
                    "user_id": transformed_op.user_id,
                    "version": transformed_op.version,
                    "timestamp": transformed_op.timestamp.isoformat()
                },
                "document_state": {
                    "content": document_state.content,
                    "version": document_state.version,
                    "checksum": document_state.checksum
                }
            }
        )
        await websocket_manager.send_to_user(session_user_id, edit_message)
    
    logger.debug(
        "Collaborative edit applied",
        session_id=session_id,
        user_id=user_id,
        operation_type=operation.type
    )


async def handle_cursor_update(
    websocket_manager: WebSocketManager,
    collaboration_manager: CollaborationManager,
    connection_id: str,
    user_id: Optional[str],
    message_data: Dict[str, Any]
) -> None:
    """Handle cursor position update."""
    
    if not user_id:
        return
    
    session_id = message_data.get("session_id")
    cursor_position = message_data.get("cursor_position")
    selection_start = message_data.get("selection_start")
    selection_end = message_data.get("selection_end")
    
    if not session_id:
        return
    
    # Update user presence
    presence = await collaboration_manager.update_user_presence(
        session_id,
        user_id,
        cursor_position,
        selection_start,
        selection_end
    )
    
    if not presence:
        return
    
    # Get session for broadcasting
    session = await collaboration_manager.get_session(session_id)
    if not session:
        return
    
    # Broadcast cursor update to other users
    from services.websocket_manager import WebSocketMessage
    
    for other_user_id in session.document_state.active_users:
        if other_user_id != user_id:
            cursor_message = WebSocketMessage(
                type=MessageType.USER_NOTIFICATION,
                user_id=other_user_id,
                data={
                    "type": CollaborationType.USER_CURSOR_UPDATE,
                    "session_id": session_id,
                    "user_id": user_id,
                    "cursor_position": cursor_position,
                    "selection_start": selection_start,
                    "selection_end": selection_end,
                    "color": presence.color
                }
            )
            await websocket_manager.send_to_user(other_user_id, cursor_message)


async def handle_get_session_state(
    websocket_manager: WebSocketManager,
    collaboration_manager: CollaborationManager,
    connection_id: str,
    message_data: Dict[str, Any]
) -> None:
    """Handle request for session state."""
    
    session_id = message_data.get("session_id")
    if not session_id:
        return
    
    # Get session state
    session_state = await collaboration_manager.get_session_state(session_id)
    
    if not session_state:
        return
    
    # Send session state to requester
    from services.websocket_manager import WebSocketMessage
    
    state_message = WebSocketMessage(
        type=MessageType.USER_NOTIFICATION,
        data={
            "type": CollaborationType.DOCUMENT_SYNC,
            "session_id": session_id,
            "session_state": session_state
        }
    )
    
    await websocket_manager._send_to_connection(connection_id, state_message)