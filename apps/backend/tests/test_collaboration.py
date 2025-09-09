"""
Tests for real-time collaboration features including WebSocket communication,
operational transforms, and conflict resolution.
"""

import asyncio
import json
import pytest
from datetime import datetime
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket
import websockets

from services.collaboration_manager import (
    CollaborationManager,
    CollaborationSession,
    Operation,
    OperationType,
    DocumentState,
    UserPresence,
    get_collaboration_manager
)
from services.websocket_manager import (
    WebSocketManager,
    WebSocketMessage,
    MessageType,
    get_websocket_manager
)


@pytest.fixture
async def collaboration_manager():
    """Create a test collaboration manager."""
    manager = CollaborationManager()
    await manager.initialize()
    yield manager
    await manager.shutdown()


@pytest.fixture
async def websocket_manager():
    """Create a test WebSocket manager."""
    manager = WebSocketManager()
    await manager.initialize()
    yield manager
    await manager.shutdown()


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    ws = AsyncMock(spec=WebSocket)
    ws.accept = AsyncMock()
    ws.send_text = AsyncMock()
    ws.receive_text = AsyncMock()
    ws.close = AsyncMock()
    return ws


class TestCollaborationManager:
    """Test collaboration manager functionality."""
    
    @pytest.mark.asyncio
    async def test_create_session(self, collaboration_manager):
        """Test creating a new collaboration session."""
        document_id = "test-doc-1"
        initial_content = "Initial content"
        
        session = await collaboration_manager.create_session(
            document_id=document_id,
            initial_content=initial_content
        )
        
        assert session is not None
        assert session.document_id == document_id
        assert session.document_state.content == initial_content
        assert session.document_state.version == 0
        assert len(session.users) == 0
    
    @pytest.mark.asyncio
    async def test_join_and_leave_session(self, collaboration_manager):
        """Test users joining and leaving a session."""
        # Create session
        session = await collaboration_manager.create_session("test-doc-2")
        
        # Join users
        user1 = await session.add_user("user1", "Alice")
        user2 = await session.add_user("user2", "Bob")
        
        assert len(session.users) == 2
        assert "user1" in session.users
        assert "user2" in session.users
        assert session.users["user1"].username == "Alice"
        assert session.users["user2"].username == "Bob"
        
        # Leave session
        await session.remove_user("user1")
        assert len(session.users) == 1
        assert "user1" not in session.users
        assert "user2" in session.users
    
    @pytest.mark.asyncio
    async def test_user_presence_tracking(self, collaboration_manager):
        """Test user presence and cursor tracking."""
        session = await collaboration_manager.create_session("test-doc-3")
        await session.add_user("user1", "Alice")
        
        # Update presence
        presence = await session.update_user_presence(
            user_id="user1",
            cursor_position=42,
            selection_start=10,
            selection_end=20
        )
        
        assert presence is not None
        assert presence.cursor_position == 42
        assert presence.selection_start == 10
        assert presence.selection_end == 20
        assert presence.is_active == True
    
    @pytest.mark.asyncio
    async def test_apply_operation_insert(self, collaboration_manager):
        """Test applying insert operation."""
        session = await collaboration_manager.create_session(
            "test-doc-4",
            initial_content="Hello World"
        )
        
        operation = Operation(
            type=OperationType.INSERT,
            position=6,
            content="Beautiful ",
            user_id="user1",
            version=0,
            parent_version=0
        )
        
        success, transformed_op = await session.apply_operation(operation)
        
        assert success == True
        assert session.document_state.content == "Hello Beautiful World"
        assert session.document_state.version == 1
        assert transformed_op.version == 1
    
    @pytest.mark.asyncio
    async def test_apply_operation_delete(self, collaboration_manager):
        """Test applying delete operation."""
        session = await collaboration_manager.create_session(
            "test-doc-5",
            initial_content="Hello Beautiful World"
        )
        
        operation = Operation(
            type=OperationType.DELETE,
            position=6,
            length=10,  # Delete "Beautiful "
            user_id="user1",
            version=0,
            parent_version=0
        )
        
        success, transformed_op = await session.apply_operation(operation)
        
        assert success == True
        assert session.document_state.content == "Hello World"
        assert session.document_state.version == 1
    
    @pytest.mark.asyncio
    async def test_operational_transform_concurrent_inserts(self, collaboration_manager):
        """Test OT with concurrent insert operations."""
        session = await collaboration_manager.create_session(
            "test-doc-6",
            initial_content="Hello World"
        )
        
        # User 1 inserts at position 6
        op1 = Operation(
            type=OperationType.INSERT,
            position=6,
            content="Beautiful ",
            user_id="user1",
            version=0,
            parent_version=0
        )
        
        success1, transformed_op1 = await session.apply_operation(op1)
        assert success1 == True
        assert session.document_state.content == "Hello Beautiful World"
        
        # User 2 inserts at position 6 (but based on version 0)
        op2 = Operation(
            type=OperationType.INSERT,
            position=6,
            content="Amazing ",
            user_id="user2",
            version=1,
            parent_version=0  # Based on old version
        )
        
        success2, transformed_op2 = await session.apply_operation(op2)
        assert success2 == True
        # Position should be transformed to account for op1
        assert transformed_op2.position == 17  # After "Beautiful "
        assert session.document_state.content == "Hello Beautiful Amazing World"
    
    @pytest.mark.asyncio
    async def test_operational_transform_insert_delete_conflict(self, collaboration_manager):
        """Test OT with conflicting insert and delete operations."""
        session = await collaboration_manager.create_session(
            "test-doc-7",
            initial_content="The quick brown fox"
        )
        
        # User 1 deletes "quick "
        op1 = Operation(
            type=OperationType.DELETE,
            position=4,
            length=6,
            user_id="user1",
            version=0,
            parent_version=0
        )
        
        success1, _ = await session.apply_operation(op1)
        assert success1 == True
        assert session.document_state.content == "The brown fox"
        
        # User 2 inserts "very " at position 10 (based on original)
        op2 = Operation(
            type=OperationType.INSERT,
            position=10,
            content="very ",
            user_id="user2",
            version=1,
            parent_version=0
        )
        
        success2, transformed_op2 = await session.apply_operation(op2)
        assert success2 == True
        # Position should be transformed due to deletion
        assert transformed_op2.position == 4  # Adjusted for deletion
        assert session.document_state.content == "The very brown fox"


class TestWebSocketCollaboration:
    """Test WebSocket-based collaboration features."""
    
    @pytest.mark.asyncio
    async def test_websocket_connection_lifecycle(self, websocket_manager, mock_websocket):
        """Test WebSocket connection and disconnection."""
        # Connect
        connection_id = await websocket_manager.connect(
            websocket=mock_websocket,
            user_id="test-user",
            connection_metadata={"client": "test"}
        )
        
        assert connection_id is not None
        assert connection_id in websocket_manager.connections
        assert "test-user" in websocket_manager.user_connections
        
        # Verify connection message sent
        mock_websocket.send_text.assert_called()
        
        # Disconnect
        await websocket_manager.disconnect(connection_id)
        assert connection_id not in websocket_manager.connections
    
    @pytest.mark.asyncio
    async def test_collaboration_message_broadcasting(self, websocket_manager):
        """Test broadcasting collaboration messages to session users."""
        # Create mock WebSockets for multiple users
        ws1 = AsyncMock(spec=WebSocket)
        ws2 = AsyncMock(spec=WebSocket)
        ws3 = AsyncMock(spec=WebSocket)
        
        # Connect users
        conn1 = await websocket_manager.connect(ws1, "user1")
        conn2 = await websocket_manager.connect(ws2, "user2")
        conn3 = await websocket_manager.connect(ws3, "user3")
        
        # Create collaboration message
        message = WebSocketMessage(
            type=MessageType.USER_NOTIFICATION,
            data={
                "type": "document_edit",
                "operation": {
                    "type": "insert",
                    "position": 0,
                    "content": "Test"
                }
            }
        )
        
        # Broadcast to all
        sent_count = await websocket_manager.broadcast_to_all(message)
        
        assert sent_count == 3
        ws1.send_text.assert_called()
        ws2.send_text.assert_called()
        ws3.send_text.assert_called()
    
    @pytest.mark.asyncio 
    async def test_user_presence_updates(self, websocket_manager):
        """Test user presence update broadcasting."""
        ws1 = AsyncMock(spec=WebSocket)
        ws2 = AsyncMock(spec=WebSocket)
        
        conn1 = await websocket_manager.connect(ws1, "user1")
        conn2 = await websocket_manager.connect(ws2, "user2")
        
        # Send presence update to specific user
        presence_message = WebSocketMessage(
            type=MessageType.USER_NOTIFICATION,
            user_id="user2",
            data={
                "type": "presence_update",
                "cursor_position": 42,
                "username": "user1"
            }
        )
        
        sent_count = await websocket_manager.send_to_user("user2", presence_message)
        
        assert sent_count == 1
        ws2.send_text.assert_called()
        ws1.send_text.assert_not_called()


class TestConflictResolution:
    """Test conflict detection and resolution."""
    
    @pytest.mark.asyncio
    async def test_detect_conflicting_operations(self, collaboration_manager):
        """Test detection of conflicting operations."""
        session = await collaboration_manager.create_session(
            "test-doc-conflict",
            initial_content="ABC"
        )
        
        # Two users try to modify the same position
        op1 = Operation(
            type=OperationType.INSERT,
            position=1,
            content="X",
            user_id="user1",
            version=0,
            parent_version=0
        )
        
        op2 = Operation(
            type=OperationType.INSERT,
            position=1,
            content="Y",
            user_id="user2",
            version=0,
            parent_version=0
        )
        
        # Apply first operation
        success1, _ = await session.apply_operation(op1)
        assert success1 == True
        assert session.document_state.content == "AXBC"
        
        # Apply second operation (should be transformed)
        success2, transformed_op2 = await session.apply_operation(op2)
        assert success2 == True
        # Position should be adjusted
        assert transformed_op2.position == 2
        assert session.document_state.content == "AXYBC"
    
    @pytest.mark.asyncio
    async def test_concurrent_edits_stress_test(self, collaboration_manager):
        """Stress test with many concurrent edits."""
        session = await collaboration_manager.create_session(
            "test-doc-stress",
            initial_content=""
        )
        
        operations = []
        num_users = 5
        ops_per_user = 10
        
        # Generate operations from multiple users
        for user_id in range(num_users):
            for op_num in range(ops_per_user):
                op = Operation(
                    type=OperationType.INSERT,
                    position=0,  # Always insert at beginning
                    content=f"U{user_id}O{op_num} ",
                    user_id=f"user{user_id}",
                    version=0,
                    parent_version=0
                )
                operations.append(op)
        
        # Apply all operations concurrently
        results = await asyncio.gather(*[
            session.apply_operation(op) for op in operations
        ])
        
        # All operations should succeed
        assert all(success for success, _ in results)
        
        # Document should contain all insertions
        content = session.document_state.content
        for user_id in range(num_users):
            for op_num in range(ops_per_user):
                assert f"U{user_id}O{op_num}" in content


class TestPerformance:
    """Test performance and latency requirements."""
    
    @pytest.mark.asyncio
    async def test_operation_latency(self, collaboration_manager):
        """Test that operations complete within 200ms latency requirement."""
        import time
        
        session = await collaboration_manager.create_session(
            "test-doc-perf",
            initial_content="Test content " * 100  # Larger document
        )
        
        operation = Operation(
            type=OperationType.INSERT,
            position=50,
            content="Performance test insert",
            user_id="user1",
            version=0,
            parent_version=0
        )
        
        start_time = time.time()
        success, _ = await session.apply_operation(operation)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        
        assert success == True
        assert latency_ms < 200, f"Operation latency {latency_ms}ms exceeds 200ms requirement"
    
    @pytest.mark.asyncio
    async def test_websocket_broadcast_latency(self, websocket_manager):
        """Test WebSocket message broadcast latency."""
        import time
        
        # Create multiple connections
        connections = []
        for i in range(10):
            ws = AsyncMock(spec=WebSocket)
            conn_id = await websocket_manager.connect(ws, f"user{i}")
            connections.append((conn_id, ws))
        
        message = WebSocketMessage(
            type=MessageType.USER_NOTIFICATION,
            data={"test": "data"}
        )
        
        start_time = time.time()
        sent_count = await websocket_manager.broadcast_to_all(message)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        
        assert sent_count == 10
        assert latency_ms < 200, f"Broadcast latency {latency_ms}ms exceeds 200ms requirement"


@pytest.mark.asyncio
async def test_end_to_end_collaboration_flow():
    """Test complete end-to-end collaboration flow."""
    # Initialize managers
    collab_manager = CollaborationManager()
    await collab_manager.initialize()
    
    ws_manager = WebSocketManager()
    await ws_manager.initialize()
    
    try:
        # Create document session
        session = await collab_manager.create_session(
            "e2e-doc",
            initial_content="Initial document content"
        )
        
        # Connect users via WebSocket
        ws1 = AsyncMock(spec=WebSocket)
        ws2 = AsyncMock(spec=WebSocket)
        
        conn1 = await ws_manager.connect(ws1, "user1")
        conn2 = await ws_manager.connect(ws2, "user2")
        
        # Users join collaboration session
        await collab_manager.join_session(session.session_id, "user1", "Alice")
        await collab_manager.join_session(session.session_id, "user2", "Bob")
        
        # User 1 makes an edit
        op1 = Operation(
            type=OperationType.INSERT,
            position=0,
            content="Updated: ",
            user_id="user1",
            version=0,
            parent_version=0
        )
        
        success1, transformed_op1, doc_state1 = await collab_manager.apply_edit(
            session.session_id, op1
        )
        
        assert success1 == True
        assert doc_state1.content == "Updated: Initial document content"
        
        # User 2 makes a concurrent edit
        op2 = Operation(
            type=OperationType.INSERT,
            position=24,  # Based on original position
            content=" with new information",
            user_id="user2",
            version=1,
            parent_version=0
        )
        
        success2, transformed_op2, doc_state2 = await collab_manager.apply_edit(
            session.session_id, op2
        )
        
        assert success2 == True
        assert "Updated:" in doc_state2.content
        assert "new information" in doc_state2.content
        
        # Verify both users have consistent document state
        final_state = await collab_manager.get_session_state(session.session_id)
        assert final_state is not None
        assert len(final_state["active_users"]) == 2
        
    finally:
        await collab_manager.shutdown()
        await ws_manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])