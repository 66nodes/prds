"""
Real-time collaboration manager for multi-user document editing.

Handles operational transforms, conflict resolution, and user presence tracking.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import uuid
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel, Field

from core.config import get_settings
from core.database import get_redis

logger = structlog.get_logger(__name__)
settings = get_settings()


class OperationType(str, Enum):
    """Types of editing operations."""
    INSERT = "insert"
    DELETE = "delete"
    REPLACE = "replace"
    FORMAT = "format"
    CURSOR_MOVE = "cursor_move"


class CollaborationType(str, Enum):
    """Types of collaboration events."""
    USER_JOIN = "user_join"
    USER_LEAVE = "user_leave"
    USER_CURSOR_UPDATE = "user_cursor_update"
    USER_SELECTION_UPDATE = "user_selection_update"
    DOCUMENT_EDIT = "document_edit"
    DOCUMENT_SYNC = "document_sync"
    PRESENCE_UPDATE = "presence_update"
    CONFLICT_DETECTED = "conflict_detected"
    CONFLICT_RESOLVED = "conflict_resolved"


@dataclass
class UserPresence:
    """User presence information in a collaborative session."""
    user_id: str
    username: str
    cursor_position: int = 0
    selection_start: Optional[int] = None
    selection_end: Optional[int] = None
    color: str = field(default_factory=lambda: f"#{uuid.uuid4().hex[:6]}")
    last_activity: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class Operation(BaseModel):
    """Represents a document edit operation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: OperationType
    position: int
    content: Optional[str] = None
    length: Optional[int] = None
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: int
    parent_version: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentState(BaseModel):
    """Represents the current state of a collaborative document."""
    document_id: str
    content: str
    version: int
    last_modified: datetime = Field(default_factory=datetime.utcnow)
    operation_history: List[Operation] = Field(default_factory=list)
    active_users: List[str] = Field(default_factory=list)
    checksum: Optional[str] = None


class CollaborationSession:
    """Manages a single collaborative editing session."""
    
    def __init__(self, session_id: str, document_id: str):
        self.session_id = session_id
        self.document_id = document_id
        self.document_state = DocumentState(
            document_id=document_id,
            content="",
            version=0
        )
        self.users: Dict[str, UserPresence] = {}
        self.pending_operations: List[Operation] = []
        self.operation_queue = asyncio.Queue()
        self.lock = asyncio.Lock()
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        
    async def add_user(self, user_id: str, username: str) -> UserPresence:
        """Add a user to the collaboration session."""
        async with self.lock:
            if user_id not in self.users:
                self.users[user_id] = UserPresence(
                    user_id=user_id,
                    username=username
                )
                self.document_state.active_users.append(user_id)
            
            self.last_activity = datetime.utcnow()
            return self.users[user_id]
    
    async def remove_user(self, user_id: str) -> None:
        """Remove a user from the collaboration session."""
        async with self.lock:
            if user_id in self.users:
                del self.users[user_id]
                if user_id in self.document_state.active_users:
                    self.document_state.active_users.remove(user_id)
            
            self.last_activity = datetime.utcnow()
    
    async def update_user_presence(
        self,
        user_id: str,
        cursor_position: Optional[int] = None,
        selection_start: Optional[int] = None,
        selection_end: Optional[int] = None
    ) -> Optional[UserPresence]:
        """Update user presence information."""
        async with self.lock:
            if user_id in self.users:
                user = self.users[user_id]
                if cursor_position is not None:
                    user.cursor_position = cursor_position
                if selection_start is not None:
                    user.selection_start = selection_start
                if selection_end is not None:
                    user.selection_end = selection_end
                user.last_activity = datetime.utcnow()
                self.last_activity = datetime.utcnow()
                return user
            return None
    
    async def apply_operation(self, operation: Operation) -> Tuple[bool, Optional[Operation]]:
        """
        Apply an operation to the document using operational transformation.
        Returns (success, transformed_operation).
        """
        async with self.lock:
            # Check if operation is based on current version
            if operation.parent_version != self.document_state.version:
                # Transform operation against pending operations
                transformed_op = await self._transform_operation(operation)
                if not transformed_op:
                    return False, None
                operation = transformed_op
            
            # Apply the operation
            success = await self._apply_to_document(operation)
            if success:
                self.document_state.version += 1
                operation.version = self.document_state.version
                self.document_state.operation_history.append(operation)
                self.document_state.last_modified = datetime.utcnow()
                self.last_activity = datetime.utcnow()
                return True, operation
            
            return False, None
    
    async def _transform_operation(self, operation: Operation) -> Optional[Operation]:
        """
        Transform an operation against concurrent operations using OT.
        """
        # Get operations that happened after the operation's parent version
        concurrent_ops = [
            op for op in self.document_state.operation_history
            if op.version > (operation.parent_version or 0)
        ]
        
        transformed = operation.model_copy()
        
        for concurrent_op in concurrent_ops:
            # Transform based on operation types
            if concurrent_op.type == OperationType.INSERT:
                if transformed.position >= concurrent_op.position:
                    transformed.position += len(concurrent_op.content or "")
                    
            elif concurrent_op.type == OperationType.DELETE:
                if transformed.position > concurrent_op.position:
                    delete_length = concurrent_op.length or 0
                    if transformed.position >= concurrent_op.position + delete_length:
                        transformed.position -= delete_length
                    else:
                        # Operation falls within deleted range
                        transformed.position = concurrent_op.position
        
        return transformed
    
    async def _apply_to_document(self, operation: Operation) -> bool:
        """Apply an operation to the document content."""
        try:
            content = self.document_state.content
            
            if operation.type == OperationType.INSERT:
                if operation.content:
                    self.document_state.content = (
                        content[:operation.position] +
                        operation.content +
                        content[operation.position:]
                    )
                    
            elif operation.type == OperationType.DELETE:
                if operation.length:
                    self.document_state.content = (
                        content[:operation.position] +
                        content[operation.position + operation.length:]
                    )
                    
            elif operation.type == OperationType.REPLACE:
                if operation.content and operation.length:
                    self.document_state.content = (
                        content[:operation.position] +
                        operation.content +
                        content[operation.position + operation.length:]
                    )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply operation: {str(e)}", operation=operation)
            return False
    
    def get_active_users(self) -> List[UserPresence]:
        """Get list of active users in the session."""
        # Filter out inactive users (no activity in last 5 minutes)
        cutoff = datetime.utcnow() - timedelta(minutes=5)
        return [
            user for user in self.users.values()
            if user.last_activity > cutoff
        ]


class CollaborationManager:
    """
    Manages real-time collaboration sessions for document editing.
    
    Features:
    - Multi-user document editing with conflict resolution
    - Operational transformation for concurrent edits
    - User presence and cursor tracking
    - Session management and persistence
    - Real-time synchronization across clients
    """
    
    def __init__(self):
        self.sessions: Dict[str, CollaborationSession] = {}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> session_ids
        self.redis = None
        self.is_initialized = False
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.session_timeout_minutes = 60
        self.max_operation_history = 1000
        self.sync_interval_seconds = 5
    
    async def initialize(self) -> None:
        """Initialize the collaboration manager."""
        try:
            self.redis = await get_redis()
            
            # Start background cleanup task
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            # Load existing sessions from Redis
            await self._load_sessions()
            
            self.is_initialized = True
            logger.info("Collaboration manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize collaboration manager: {str(e)}")
            raise
    
    async def create_session(
        self,
        document_id: str,
        initial_content: str = ""
    ) -> CollaborationSession:
        """Create a new collaboration session."""
        session_id = str(uuid.uuid4())
        session = CollaborationSession(session_id, document_id)
        session.document_state.content = initial_content
        
        self.sessions[session_id] = session
        
        # Persist to Redis
        await self._persist_session(session)
        
        logger.info(
            "Created collaboration session",
            session_id=session_id,
            document_id=document_id
        )
        
        return session
    
    async def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """Get a collaboration session by ID."""
        return self.sessions.get(session_id)
    
    async def get_or_create_session(
        self,
        document_id: str,
        initial_content: str = ""
    ) -> CollaborationSession:
        """Get existing session for document or create new one."""
        # Check for existing session for this document
        for session in self.sessions.values():
            if session.document_id == document_id:
                return session
        
        # Create new session
        return await self.create_session(document_id, initial_content)
    
    async def join_session(
        self,
        session_id: str,
        user_id: str,
        username: str
    ) -> Optional[CollaborationSession]:
        """Join a user to a collaboration session."""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        # Add user to session
        await session.add_user(user_id, username)
        
        # Track user's sessions
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = set()
        self.user_sessions[user_id].add(session_id)
        
        # Persist changes
        await self._persist_session(session)
        
        logger.info(
            "User joined session",
            session_id=session_id,
            user_id=user_id,
            username=username
        )
        
        return session
    
    async def leave_session(
        self,
        session_id: str,
        user_id: str
    ) -> None:
        """Remove a user from a collaboration session."""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        await session.remove_user(user_id)
        
        # Update user sessions tracking
        if user_id in self.user_sessions:
            self.user_sessions[user_id].discard(session_id)
            if not self.user_sessions[user_id]:
                del self.user_sessions[user_id]
        
        # Persist changes
        await self._persist_session(session)
        
        logger.info(
            "User left session",
            session_id=session_id,
            user_id=user_id
        )
    
    async def apply_edit(
        self,
        session_id: str,
        operation: Operation
    ) -> Tuple[bool, Optional[Operation], Optional[DocumentState]]:
        """
        Apply an edit operation to a session.
        Returns (success, transformed_operation, updated_document_state).
        """
        session = self.sessions.get(session_id)
        if not session:
            return False, None, None
        
        # Apply operation with OT
        success, transformed_op = await session.apply_operation(operation)
        
        if success:
            # Persist updated state
            await self._persist_session(session)
            
            logger.debug(
                "Applied edit operation",
                session_id=session_id,
                operation_type=operation.type,
                user_id=operation.user_id
            )
            
            return True, transformed_op, session.document_state
        
        return False, None, None
    
    async def update_user_presence(
        self,
        session_id: str,
        user_id: str,
        cursor_position: Optional[int] = None,
        selection_start: Optional[int] = None,
        selection_end: Optional[int] = None
    ) -> Optional[UserPresence]:
        """Update user presence in a session."""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        presence = await session.update_user_presence(
            user_id,
            cursor_position,
            selection_start,
            selection_end
        )
        
        if presence:
            # Persist changes
            await self._persist_session(session)
        
        return presence
    
    async def get_session_state(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get complete state of a collaboration session."""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        return {
            "session_id": session.session_id,
            "document_id": session.document_id,
            "document_state": session.document_state.model_dump(),
            "active_users": [
                {
                    "user_id": user.user_id,
                    "username": user.username,
                    "cursor_position": user.cursor_position,
                    "selection_start": user.selection_start,
                    "selection_end": user.selection_end,
                    "color": user.color,
                    "is_active": user.is_active
                }
                for user in session.get_active_users()
            ],
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat()
        }
    
    async def get_user_sessions(self, user_id: str) -> List[str]:
        """Get all session IDs for a user."""
        return list(self.user_sessions.get(user_id, set()))
    
    async def _persist_session(self, session: CollaborationSession) -> None:
        """Persist session state to Redis."""
        if not self.redis:
            return
        
        try:
            key = f"collaboration:session:{session.session_id}"
            value = json.dumps({
                "document_state": session.document_state.model_dump_json(),
                "users": {
                    user_id: {
                        "username": user.username,
                        "cursor_position": user.cursor_position,
                        "selection_start": user.selection_start,
                        "selection_end": user.selection_end,
                        "color": user.color
                    }
                    for user_id, user in session.users.items()
                },
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat()
            })
            
            await self.redis.setex(
                key,
                self.session_timeout_minutes * 60,
                value
            )
            
        except Exception as e:
            logger.error(f"Failed to persist session: {str(e)}", session_id=session.session_id)
    
    async def _load_sessions(self) -> None:
        """Load existing sessions from Redis."""
        if not self.redis:
            return
        
        try:
            # Get all collaboration session keys
            keys = await self.redis.keys("collaboration:session:*")
            
            for key in keys:
                try:
                    data = await self.redis.get(key)
                    if data:
                        session_data = json.loads(data)
                        session_id = key.split(":")[-1]
                        
                        # Recreate session
                        # Note: This is simplified - in production, you'd fully restore the session
                        logger.info(f"Loaded session from Redis: {session_id}")
                        
                except Exception as e:
                    logger.error(f"Failed to load session from Redis: {str(e)}", key=key)
                    
        except Exception as e:
            logger.error(f"Failed to load sessions from Redis: {str(e)}")
    
    async def _cleanup_loop(self) -> None:
        """Clean up inactive sessions."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                current_time = datetime.utcnow()
                sessions_to_remove = []
                
                for session_id, session in self.sessions.items():
                    # Remove sessions with no activity for timeout period
                    if (current_time - session.last_activity).total_seconds() > self.session_timeout_minutes * 60:
                        sessions_to_remove.append(session_id)
                
                # Remove stale sessions
                for session_id in sessions_to_remove:
                    session = self.sessions[session_id]
                    
                    # Clean up user tracking
                    for user_id in list(session.users.keys()):
                        await self.leave_session(session_id, user_id)
                    
                    # Remove session
                    del self.sessions[session_id]
                    
                    # Remove from Redis
                    if self.redis:
                        await self.redis.delete(f"collaboration:session:{session_id}")
                    
                    logger.info(f"Cleaned up inactive session: {session_id}")
                
                if sessions_to_remove:
                    logger.info(f"Cleaned up {len(sessions_to_remove)} inactive sessions")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {str(e)}")
                await asyncio.sleep(10)
    
    async def shutdown(self) -> None:
        """Shutdown the collaboration manager."""
        logger.info("Shutting down collaboration manager")
        
        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Persist all sessions
        for session in self.sessions.values():
            await self._persist_session(session)
        
        logger.info("Collaboration manager shutdown complete")


# Global collaboration manager instance
collaboration_manager = CollaborationManager()


async def get_collaboration_manager() -> CollaborationManager:
    """Get the global collaboration manager instance."""
    if not collaboration_manager.is_initialized:
        await collaboration_manager.initialize()
    return collaboration_manager