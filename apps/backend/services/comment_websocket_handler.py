"""
Comment WebSocket Handler - Real-time comment and annotation updates.

Extends the WebSocket manager with comment-specific functionality for
collaborative feedback, live updates, and real-time notifications.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from enum import Enum
import structlog

from .websocket_manager import WebSocketManager, WebSocketMessage, MessageType
from ..models.comments import Comment, CommentType, CommentStatus

logger = structlog.get_logger(__name__)


class CommentMessageType(str, Enum):
    """Comment-specific WebSocket message types."""
    # Comment lifecycle events
    COMMENT_CREATED = "comment_created"
    COMMENT_UPDATED = "comment_updated"
    COMMENT_DELETED = "comment_deleted"
    COMMENT_RESOLVED = "comment_resolved"
    COMMENT_REOPENED = "comment_reopened"
    
    # Thread and reply events
    REPLY_ADDED = "reply_added"
    THREAD_UPDATED = "thread_updated"
    
    # Interaction events
    COMMENT_REACTION_ADDED = "comment_reaction_added"
    COMMENT_REACTION_REMOVED = "comment_reaction_removed"
    COMMENT_MENTION = "comment_mention"
    COMMENT_ASSIGNED = "comment_assigned"
    
    # Collaboration events
    USER_TYPING_COMMENT = "user_typing_comment"
    USER_STOPPED_TYPING = "user_stopped_typing"
    DOCUMENT_FOCUS_CHANGED = "document_focus_changed"
    
    # Annotation events
    ANNOTATION_CREATED = "annotation_created"
    ANNOTATION_UPDATED = "annotation_updated"
    ANNOTATION_HIGHLIGHTED = "annotation_highlighted"
    
    # Notification events
    COMMENT_NOTIFICATION = "comment_notification"
    COMMENT_REMINDER = "comment_reminder"


class CommentWebSocketHandler:
    """
    WebSocket handler for comment system real-time features.
    
    Handles:
    - Real-time comment creation, updates, and deletions
    - Live typing indicators and presence
    - Threaded conversation updates
    - Mention and assignment notifications
    - Annotation and highlighting synchronization
    - Document collaboration state
    """
    
    def __init__(self, websocket_manager: WebSocketManager):
        self.websocket_manager = websocket_manager
        self.typing_users: Dict[str, Set[str]] = {}  # document_id -> {user_ids}
        self.document_subscribers: Dict[str, Set[str]] = {}  # document_id -> {connection_ids}
        self.comment_subscribers: Dict[str, Set[str]] = {}  # comment_id -> {connection_ids}
        self.user_focus: Dict[str, str] = {}  # user_id -> document_id
        
        # Settings
        self.typing_timeout_seconds = 3
        self.presence_update_interval = 5
        
        # Background tasks
        self.typing_cleanup_task: Optional[asyncio.Task] = None
        self.presence_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> None:
        """Initialize the comment WebSocket handler."""
        try:
            # Start background tasks
            self.typing_cleanup_task = asyncio.create_task(self._typing_cleanup_loop())
            self.presence_task = asyncio.create_task(self._presence_update_loop())
            
            logger.info("Comment WebSocket handler initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize comment WebSocket handler: {str(e)}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the comment WebSocket handler."""
        try:
            if self.typing_cleanup_task:
                self.typing_cleanup_task.cancel()
                await self.typing_cleanup_task
            
            if self.presence_task:
                self.presence_task.cancel()
                await self.presence_task
                
            logger.info("Comment WebSocket handler shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during comment WebSocket handler shutdown: {str(e)}")
    
    async def subscribe_to_document(self, connection_id: str, document_id: str) -> None:
        """Subscribe a connection to document comment updates."""
        if document_id not in self.document_subscribers:
            self.document_subscribers[document_id] = set()
        
        self.document_subscribers[document_id].add(connection_id)
        
        logger.debug(f"Connection {connection_id} subscribed to document {document_id}")
    
    async def unsubscribe_from_document(self, connection_id: str, document_id: str) -> None:
        """Unsubscribe a connection from document comment updates."""
        if document_id in self.document_subscribers:
            self.document_subscribers[document_id].discard(connection_id)
            
            # Clean up empty sets
            if not self.document_subscribers[document_id]:
                del self.document_subscribers[document_id]
        
        logger.debug(f"Connection {connection_id} unsubscribed from document {document_id}")
    
    async def subscribe_to_comment(self, connection_id: str, comment_id: str) -> None:
        """Subscribe a connection to specific comment updates."""
        if comment_id not in self.comment_subscribers:
            self.comment_subscribers[comment_id] = set()
        
        self.comment_subscribers[comment_id].add(connection_id)
        
        logger.debug(f"Connection {connection_id} subscribed to comment {comment_id}")
    
    async def unsubscribe_from_comment(self, connection_id: str, comment_id: str) -> None:
        """Unsubscribe a connection from specific comment updates."""
        if comment_id in self.comment_subscribers:
            self.comment_subscribers[comment_id].discard(connection_id)
            
            if not self.comment_subscribers[comment_id]:
                del self.comment_subscribers[comment_id]
        
        logger.debug(f"Connection {connection_id} unsubscribed from comment {comment_id}")
    
    async def broadcast_comment_created(self, comment: Comment, author_name: str) -> int:
        """Broadcast comment creation to document subscribers."""
        message_data = {
            "comment_id": comment.id,
            "document_id": comment.document_id,
            "parent_id": comment.parent_id,
            "thread_id": comment.thread_id,
            "content": comment.content[:200] + "..." if len(comment.content) > 200 else comment.content,
            "comment_type": comment.comment_type.value,
            "priority": comment.priority.value,
            "author_id": comment.author_id,
            "author_name": author_name,
            "position": comment.position.model_dump() if comment.position else None,
            "selection_range": comment.selection_range.model_dump() if comment.selection_range else None,
            "created_at": comment.created_at.isoformat(),
            "mentions": comment.mentions,
            "assignees": comment.assignees,
            "tags": comment.tags
        }
        
        return await self._broadcast_to_document(
            document_id=comment.document_id,
            message_type=CommentMessageType.COMMENT_CREATED,
            data=message_data
        )
    
    async def broadcast_comment_updated(self, comment: Comment, updated_fields: List[str]) -> int:
        """Broadcast comment updates to subscribers."""
        message_data = {
            "comment_id": comment.id,
            "document_id": comment.document_id,
            "updated_fields": updated_fields,
            "content": comment.content[:200] + "..." if len(comment.content) > 200 else comment.content,
            "status": comment.status.value,
            "priority": comment.priority.value,
            "updated_at": comment.updated_at.isoformat(),
            "resolved_by": comment.resolved_by,
            "resolution_note": comment.resolution_note
        }
        
        # Send to document subscribers
        sent_count = await self._broadcast_to_document(
            document_id=comment.document_id,
            message_type=CommentMessageType.COMMENT_UPDATED,
            data=message_data
        )
        
        # Also send to comment-specific subscribers
        sent_count += await self._broadcast_to_comment(
            comment_id=comment.id,
            message_type=CommentMessageType.COMMENT_UPDATED,
            data=message_data
        )
        
        return sent_count
    
    async def broadcast_comment_deleted(self, comment_id: str, document_id: str) -> int:
        """Broadcast comment deletion to subscribers."""
        message_data = {
            "comment_id": comment_id,
            "document_id": document_id,
            "deleted_at": datetime.utcnow().isoformat()
        }
        
        return await self._broadcast_to_document(
            document_id=document_id,
            message_type=CommentMessageType.COMMENT_DELETED,
            data=message_data
        )
    
    async def broadcast_reply_added(self, reply: Comment, parent_comment_id: str) -> int:
        """Broadcast new reply to thread subscribers."""
        message_data = {
            "reply_id": reply.id,
            "parent_comment_id": parent_comment_id,
            "thread_id": reply.thread_id,
            "document_id": reply.document_id,
            "content": reply.content[:200] + "..." if len(reply.content) > 200 else reply.content,
            "author_name": reply.author_name,
            "created_at": reply.created_at.isoformat(),
            "depth": reply.depth
        }
        
        # Send to document subscribers
        sent_count = await self._broadcast_to_document(
            document_id=reply.document_id,
            message_type=CommentMessageType.REPLY_ADDED,
            data=message_data
        )
        
        # Send to parent comment subscribers
        sent_count += await self._broadcast_to_comment(
            comment_id=parent_comment_id,
            message_type=CommentMessageType.REPLY_ADDED,
            data=message_data
        )
        
        return sent_count
    
    async def broadcast_reaction_added(
        self, 
        comment_id: str, 
        document_id: str, 
        reaction_type: str, 
        user_name: str,
        total_reactions: int
    ) -> int:
        """Broadcast reaction addition to subscribers."""
        message_data = {
            "comment_id": comment_id,
            "document_id": document_id,
            "reaction_type": reaction_type,
            "user_name": user_name,
            "total_reactions": total_reactions,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self._broadcast_to_comment(
            comment_id=comment_id,
            message_type=CommentMessageType.COMMENT_REACTION_ADDED,
            data=message_data
        )
    
    async def send_mention_notification(
        self,
        mentioned_user_id: str,
        comment: Comment,
        mentioned_by: str
    ) -> int:
        """Send mention notification to specific user."""
        message_data = {
            "comment_id": comment.id,
            "document_id": comment.document_id,
            "mentioned_by": mentioned_by,
            "content_preview": comment.content[:100] + "..." if len(comment.content) > 100 else comment.content,
            "comment_url": f"/documents/{comment.document_id}#comment-{comment.id}",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        message = WebSocketMessage(
            type=CommentMessageType.COMMENT_MENTION,
            user_id=mentioned_user_id,
            data=message_data,
            metadata={
                "category": "comment_notification",
                "priority": "high",
                "document_id": comment.document_id,
                "comment_id": comment.id
            }
        )
        
        return await self.websocket_manager.send_to_user(mentioned_user_id, message)
    
    async def send_assignment_notification(
        self,
        assigned_user_id: str,
        comment: Comment,
        assigned_by: str
    ) -> int:
        """Send assignment notification to specific user."""
        message_data = {
            "comment_id": comment.id,
            "document_id": comment.document_id,
            "assigned_by": assigned_by,
            "content_preview": comment.content[:100] + "..." if len(comment.content) > 100 else comment.content,
            "priority": comment.priority.value,
            "due_date": None,  # Could be extended to support due dates
            "comment_url": f"/documents/{comment.document_id}#comment-{comment.id}",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        message = WebSocketMessage(
            type=CommentMessageType.COMMENT_ASSIGNED,
            user_id=assigned_user_id,
            data=message_data,
            metadata={
                "category": "comment_notification",
                "priority": "high",
                "document_id": comment.document_id,
                "comment_id": comment.id
            }
        )
        
        return await self.websocket_manager.send_to_user(assigned_user_id, message)
    
    async def handle_user_typing(
        self, 
        user_id: str, 
        user_name: str, 
        document_id: str, 
        comment_id: Optional[str] = None
    ) -> None:
        """Handle user typing indicator."""
        # Track typing user
        if document_id not in self.typing_users:
            self.typing_users[document_id] = set()
        
        self.typing_users[document_id].add(user_id)
        
        # Broadcast typing indicator
        message_data = {
            "user_id": user_id,
            "user_name": user_name,
            "document_id": document_id,
            "comment_id": comment_id,
            "is_typing": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self._broadcast_to_document(
            document_id=document_id,
            message_type=CommentMessageType.USER_TYPING_COMMENT,
            data=message_data,
            exclude_user_id=user_id  # Don't send back to typing user
        )
    
    async def handle_user_stopped_typing(
        self, 
        user_id: str, 
        user_name: str, 
        document_id: str, 
        comment_id: Optional[str] = None
    ) -> None:
        """Handle user stopped typing."""
        # Remove from typing users
        if document_id in self.typing_users:
            self.typing_users[document_id].discard(user_id)
            
            if not self.typing_users[document_id]:
                del self.typing_users[document_id]
        
        # Broadcast stopped typing
        message_data = {
            "user_id": user_id,
            "user_name": user_name,
            "document_id": document_id,
            "comment_id": comment_id,
            "is_typing": False,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self._broadcast_to_document(
            document_id=document_id,
            message_type=CommentMessageType.USER_STOPPED_TYPING,
            data=message_data,
            exclude_user_id=user_id
        )
    
    async def handle_annotation_created(self, comment: Comment, author_name: str) -> int:
        """Handle annotation creation with text selection."""
        if not comment.selection_range:
            return 0
        
        message_data = {
            "annotation_id": comment.id,
            "document_id": comment.document_id,
            "selection_range": comment.selection_range.model_dump(),
            "content": comment.content,
            "author_name": author_name,
            "comment_type": comment.comment_type.value,
            "created_at": comment.created_at.isoformat()
        }
        
        return await self._broadcast_to_document(
            document_id=comment.document_id,
            message_type=CommentMessageType.ANNOTATION_CREATED,
            data=message_data
        )
    
    async def handle_document_focus_changed(
        self, 
        user_id: str, 
        user_name: str, 
        document_id: str,
        section_id: Optional[str] = None
    ) -> None:
        """Handle user focus change within document."""
        # Update user focus tracking
        self.user_focus[user_id] = document_id
        
        # Broadcast focus change
        message_data = {
            "user_id": user_id,
            "user_name": user_name,
            "document_id": document_id,
            "section_id": section_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self._broadcast_to_document(
            document_id=document_id,
            message_type=CommentMessageType.DOCUMENT_FOCUS_CHANGED,
            data=message_data,
            exclude_user_id=user_id
        )
    
    async def get_document_presence(self, document_id: str) -> Dict[str, Any]:
        """Get current presence information for a document."""
        # Get subscribers
        subscribers = self.document_subscribers.get(document_id, set())
        
        # Get typing users
        typing_users = list(self.typing_users.get(document_id, set()))
        
        # Get focused users
        focused_users = [
            user_id for user_id, focused_doc in self.user_focus.items()
            if focused_doc == document_id
        ]
        
        return {
            "document_id": document_id,
            "active_connections": len(subscribers),
            "typing_users": typing_users,
            "focused_users": focused_users,
            "total_active_users": len(set(typing_users + focused_users)),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _broadcast_to_document(
        self, 
        document_id: str, 
        message_type: CommentMessageType, 
        data: Dict[str, Any],
        exclude_user_id: Optional[str] = None
    ) -> int:
        """Broadcast message to all document subscribers."""
        if document_id not in self.document_subscribers:
            return 0
        
        message = WebSocketMessage(
            type=message_type,
            data=data,
            metadata={
                "category": "comment_update",
                "document_id": document_id
            }
        )
        
        sent_count = 0
        connection_ids = list(self.document_subscribers[document_id])
        
        for connection_id in connection_ids:
            # Skip excluded user's connections
            if exclude_user_id:
                connection_info = self.websocket_manager.connection_info.get(connection_id)
                if connection_info and connection_info.user_id == exclude_user_id:
                    continue
            
            success = await self.websocket_manager._send_to_connection(connection_id, message)
            if success:
                sent_count += 1
        
        return sent_count
    
    async def _broadcast_to_comment(
        self, 
        comment_id: str, 
        message_type: CommentMessageType, 
        data: Dict[str, Any]
    ) -> int:
        """Broadcast message to comment-specific subscribers."""
        if comment_id not in self.comment_subscribers:
            return 0
        
        message = WebSocketMessage(
            type=message_type,
            data=data,
            metadata={
                "category": "comment_update",
                "comment_id": comment_id
            }
        )
        
        sent_count = 0
        connection_ids = list(self.comment_subscribers[comment_id])
        
        for connection_id in connection_ids:
            success = await self.websocket_manager._send_to_connection(connection_id, message)
            if success:
                sent_count += 1
        
        return sent_count
    
    async def _typing_cleanup_loop(self) -> None:
        """Clean up stale typing indicators."""
        while True:
            try:
                await asyncio.sleep(self.typing_timeout_seconds)
                
                # This is a simplified cleanup - in production, you'd track timestamps
                # and remove users who haven't sent typing updates recently
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Typing cleanup loop error: {str(e)}")
                await asyncio.sleep(1)
    
    async def _presence_update_loop(self) -> None:
        """Send periodic presence updates for active documents."""
        while True:
            try:
                await asyncio.sleep(self.presence_update_interval)
                
                # Send presence updates for documents with activity
                for document_id in list(self.document_subscribers.keys()):
                    if self.document_subscribers[document_id]:  # Has subscribers
                        presence_data = await self.get_document_presence(document_id)
                        
                        await self._broadcast_to_document(
                            document_id=document_id,
                            message_type="presence_update",
                            data=presence_data
                        )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Presence update loop error: {str(e)}")
                await asyncio.sleep(5)
    
    def connection_disconnected(self, connection_id: str) -> None:
        """Handle connection disconnection cleanup."""
        # Clean up document subscriptions
        for document_id, subscribers in list(self.document_subscribers.items()):
            if connection_id in subscribers:
                subscribers.discard(connection_id)
                if not subscribers:
                    del self.document_subscribers[document_id]
        
        # Clean up comment subscriptions
        for comment_id, subscribers in list(self.comment_subscribers.items()):
            if connection_id in subscribers:
                subscribers.discard(connection_id)
                if not subscribers:
                    del self.comment_subscribers[comment_id]


# Global comment WebSocket handler instance
comment_websocket_handler: Optional[CommentWebSocketHandler] = None


async def get_comment_websocket_handler() -> CommentWebSocketHandler:
    """Get the global comment WebSocket handler instance."""
    global comment_websocket_handler
    
    if comment_websocket_handler is None:
        from .websocket_manager import get_websocket_manager
        
        websocket_manager = await get_websocket_manager()
        comment_websocket_handler = CommentWebSocketHandler(websocket_manager)
        await comment_websocket_handler.initialize()
    
    return comment_websocket_handler