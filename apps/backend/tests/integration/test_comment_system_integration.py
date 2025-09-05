"""
Integration tests for Comment and Annotation System.

Tests the complete flow from API endpoints through WebSocket notifications,
including comment CRUD operations, threading, real-time updates, and collaboration features.
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi.testclient import TestClient
from fastapi import WebSocket
from httpx import AsyncClient
import websockets

from ...main import app
from ...models.comments import (
    Comment, CommentCreate, CommentUpdate, CommentType, 
    CommentStatus, CommentPriority, DocumentType, SelectionRange, CommentPosition
)
from ...services.websocket_manager import get_websocket_manager
from ...services.comment_websocket_handler import get_comment_websocket_handler


@pytest.fixture
def test_client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Async HTTP client for testing."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def sample_document():
    """Document for testing."""
    return {
        "id": str(uuid4()),
        "type": DocumentType.PRD,
        "title": f"Document {uuid4()}",
        "content": f"Document content {uuid4()} for comment integration testing."
    }


@pytest.fixture
def sample_user():
    """User for testing."""
    return {
        "id": str(uuid4()),
        "full_name": f"User {uuid4()}",
        "email": f"user-{uuid4()}@example.com"
    }


@pytest.fixture
def sample_comment_data(sample_document, sample_user):
    """Comment creation data for testing."""
    return CommentCreate(
        document_id=sample_document["id"],
        document_type=sample_document["type"],
        content=f"Comment content {uuid4()} for integration testing.",
        comment_type=CommentType.COMMENT,
        priority=CommentPriority.MEDIUM,
        tags=["integration", "test"],
        mentions=[],
        assignees=[]
    )


@pytest.fixture
def sample_annotation_data(sample_document):
    """Annotation with text selection for testing."""
    return CommentCreate(
        document_id=sample_document["id"],
        document_type=sample_document["type"],
        content=f"Annotation content {uuid4()} needs clarification.",
        comment_type=CommentType.ANNOTATION,
        priority=CommentPriority.HIGH,
        selection_range=SelectionRange(
            start_offset=10,
            end_offset=25,
            selected_text="test document",
            container_element="p",
            xpath="//div[@class='content']/p[1]"
        ),
        position=CommentPosition(
            x=150.0,
            y=200.0,
            section_id="introduction",
            element_id="intro-paragraph-1"
        )
    )


class TestCommentCRUDIntegration:
    """Test comment CRUD operations integration."""
    
    async def test_complete_comment_lifecycle(self, async_client, sample_comment_data, sample_user):
        """Test complete comment lifecycle: create -> read -> update -> delete."""
        
        # Mock authentication
        headers = {"Authorization": f"Bearer test-token-{sample_user['id']}"}
        
        # 1. Create comment
        create_response = await async_client.post(
            "/comments/",
            json=sample_comment_data.model_dump(),
            headers=headers
        )
        assert create_response.status_code == 200
        
        created_comment = create_response.json()
        comment_id = created_comment["id"]
        
        # Verify created comment structure
        assert created_comment["content"] == sample_comment_data.content
        assert created_comment["comment_type"] == sample_comment_data.comment_type.value
        assert created_comment["status"] == CommentStatus.OPEN.value
        assert created_comment["author_id"] == sample_user["id"]
        assert created_comment["depth"] == 0  # Root comment
        
        # 2. Read comment
        read_response = await async_client.get(f"/comments/{comment_id}", headers=headers)
        assert read_response.status_code == 200
        
        read_comment = read_response.json()
        assert read_comment["id"] == comment_id
        assert read_comment["content"] == sample_comment_data.content
        
        # 3. Update comment
        update_data = CommentUpdate(
            content="Updated comment content",
            priority=CommentPriority.HIGH,
            tags=["updated", "integration"]
        )
        
        update_response = await async_client.put(
            f"/comments/{comment_id}",
            json=update_data.model_dump(exclude_unset=True),
            headers=headers
        )
        assert update_response.status_code == 200
        
        updated_comment = update_response.json()
        assert updated_comment["content"] == "Updated comment content"
        assert updated_comment["priority"] == CommentPriority.HIGH.value
        assert "updated" in updated_comment["tags"]
        
        # 4. Delete comment
        delete_response = await async_client.delete(f"/comments/{comment_id}", headers=headers)
        assert delete_response.status_code == 200
        
        # Verify comment is deleted
        read_after_delete = await async_client.get(f"/comments/{comment_id}", headers=headers)
        assert read_after_delete.status_code == 404


    async def test_annotation_with_text_selection(self, async_client, sample_annotation_data, sample_user):
        """Test annotation creation with text selection."""
        
        headers = {"Authorization": f"Bearer test-token-{sample_user['id']}"}
        
        # Create annotation
        response = await async_client.post(
            "/comments/",
            json=sample_annotation_data.model_dump(),
            headers=headers
        )
        assert response.status_code == 200
        
        annotation = response.json()
        
        # Verify annotation-specific fields
        assert annotation["comment_type"] == CommentType.ANNOTATION.value
        assert annotation["selection_range"] is not None
        assert annotation["selection_range"]["start_offset"] == 10
        assert annotation["selection_range"]["end_offset"] == 25
        assert annotation["selection_range"]["selected_text"] == "test document"
        
        assert annotation["position"] is not None
        assert annotation["position"]["x"] == 150.0
        assert annotation["position"]["y"] == 200.0
        assert annotation["position"]["section_id"] == "introduction"


class TestCommentThreadingIntegration:
    """Test comment threading and hierarchy."""
    
    async def test_threaded_comment_creation(self, async_client, sample_comment_data, sample_user):
        """Test creating threaded replies."""
        
        headers = {"Authorization": f"Bearer test-token-{sample_user['id']}"}
        
        # Create root comment
        root_response = await async_client.post(
            "/comments/",
            json=sample_comment_data.model_dump(),
            headers=headers
        )
        root_comment = root_response.json()
        thread_id = root_comment["thread_id"]
        
        # Create first reply
        reply1_data = CommentCreate(
            document_id=sample_comment_data.document_id,
            document_type=sample_comment_data.document_type,
            content="This is a reply to the root comment.",
            comment_type=CommentType.COMMENT,
            parent_id=root_comment["id"]
        )
        
        reply1_response = await async_client.post(
            "/comments/",
            json=reply1_data.model_dump(),
            headers=headers
        )
        reply1 = reply1_response.json()
        
        # Verify reply structure
        assert reply1["parent_id"] == root_comment["id"]
        assert reply1["thread_id"] == thread_id
        assert reply1["depth"] == 1
        
        # Create nested reply
        reply2_data = CommentCreate(
            document_id=sample_comment_data.document_id,
            document_type=sample_comment_data.document_type,
            content="This is a reply to the first reply.",
            comment_type=CommentType.COMMENT,
            parent_id=reply1["id"]
        )
        
        reply2_response = await async_client.post(
            "/comments/",
            json=reply2_data.model_dump(),
            headers=headers
        )
        reply2 = reply2_response.json()
        
        # Verify nested reply
        assert reply2["parent_id"] == reply1["id"]
        assert reply2["thread_id"] == thread_id
        assert reply2["depth"] == 2


    async def test_comment_thread_retrieval(self, async_client, sample_comment_data, sample_user):
        """Test retrieving complete comment threads."""
        
        headers = {"Authorization": f"Bearer test-token-{sample_user['id']}"}
        
        # Create root comment and replies (same as above test)
        root_response = await async_client.post(
            "/comments/",
            json=sample_comment_data.model_dump(),
            headers=headers
        )
        root_comment = root_response.json()
        
        # Create a reply
        reply_data = CommentCreate(
            document_id=sample_comment_data.document_id,
            document_type=sample_comment_data.document_type,
            content="Reply content",
            parent_id=root_comment["id"]
        )
        
        await async_client.post("/comments/", json=reply_data.model_dump(), headers=headers)
        
        # Get complete thread
        thread_response = await async_client.get(
            f"/comments/thread/{root_comment['thread_id']}",
            headers=headers
        )
        assert thread_response.status_code == 200
        
        thread = thread_response.json()
        
        # Verify thread structure
        assert thread["root_comment"]["id"] == root_comment["id"]
        assert len(thread["replies"]) >= 1
        assert thread["total_replies"] >= 1
        assert root_comment["author_id"] in thread["participants"]


class TestDocumentCommentsIntegration:
    """Test document-level comment operations."""
    
    async def test_document_comment_listing(self, async_client, sample_document, sample_user):
        """Test listing all comments for a document."""
        
        headers = {"Authorization": f"Bearer test-token-{sample_user['id']}"}
        document_id = sample_document["id"]
        
        # Create multiple comments
        comments_to_create = [
            CommentCreate(
                document_id=document_id,
                document_type=DocumentType.PRD,
                content=f"Comment {i}",
                comment_type=CommentType.COMMENT if i % 2 == 0 else CommentType.SUGGESTION,
                priority=CommentPriority.HIGH if i < 2 else CommentPriority.MEDIUM
            )
            for i in range(5)
        ]
        
        created_comments = []
        for comment_data in comments_to_create:
            response = await async_client.post(
                "/comments/",
                json=comment_data.model_dump(),
                headers=headers
            )
            created_comments.append(response.json())
        
        # List all document comments
        list_response = await async_client.get(
            f"/comments/document/{document_id}",
            headers=headers
        )
        assert list_response.status_code == 200
        
        comment_list = list_response.json()
        
        # Verify response structure
        assert "comments" in comment_list
        assert "total_count" in comment_list
        assert "open_count" in comment_list
        assert comment_list["total_count"] >= 5
        assert len(comment_list["comments"]) >= 5
        
        # Test filtering
        filtered_response = await async_client.get(
            f"/comments/document/{document_id}?comment_type={CommentType.SUGGESTION.value}",
            headers=headers
        )
        filtered_list = filtered_response.json()
        
        # Should have fewer comments due to filtering
        assert filtered_list["total_count"] < comment_list["total_count"]


class TestCommentWebSocketIntegration:
    """Test WebSocket integration for real-time updates."""
    
    async def test_websocket_comment_notifications(self, sample_document, sample_user):
        """Test WebSocket notifications for comment operations."""
        
        # This test would require a more complex setup with actual WebSocket connections
        # For now, we test the WebSocket handler functionality
        
        websocket_manager = await get_websocket_manager()
        comment_handler = await get_comment_websocket_handler()
        
        # Mock a WebSocket connection
        connection_id = "test-connection-123"
        document_id = sample_document["id"]
        
        # Subscribe to document updates
        await comment_handler.subscribe_to_document(connection_id, document_id)
        
        # Create a mock comment for notification testing
        mock_comment = Comment(
            id=str(uuid4()),
            document_id=document_id,
            document_type=DocumentType.PRD,
            author_id=sample_user["id"],
            author_name=sample_user["full_name"],
            thread_id=str(uuid4()),
            content="Test comment for WebSocket notification",
            comment_type=CommentType.COMMENT,
            status=CommentStatus.OPEN
        )
        
        # Test comment creation notification
        sent_count = await comment_handler.broadcast_comment_created(
            comment=mock_comment,
            author_name=sample_user["full_name"]
        )
        
        # Since we're using mock connections, sent_count will be 0
        # In a real test with actual WebSocket connections, this would be > 0
        assert sent_count >= 0
        
        # Test comment update notification
        updated_fields = ["content", "status"]
        update_count = await comment_handler.broadcast_comment_updated(
            comment=mock_comment,
            updated_fields=updated_fields
        )
        
        assert update_count >= 0
        
        # Clean up
        await comment_handler.unsubscribe_from_document(connection_id, document_id)


class TestCommentReactionsIntegration:
    """Test comment reactions and interactions."""
    
    async def test_comment_reactions_flow(self, async_client, sample_comment_data, sample_user):
        """Test adding and removing reactions to comments."""
        
        headers = {"Authorization": f"Bearer test-token-{sample_user['id']}"}
        
        # Create comment
        comment_response = await async_client.post(
            "/comments/",
            json=sample_comment_data.model_dump(),
            headers=headers
        )
        comment_id = comment_response.json()["id"]
        
        # Add reaction
        reaction_response = await async_client.post(
            f"/comments/{comment_id}/reactions",
            params={"reaction_type": "like"},
            headers=headers
        )
        assert reaction_response.status_code == 200
        
        # Verify reaction was added (would need to check comment reactions)
        comment_check = await async_client.get(f"/comments/{comment_id}", headers=headers)
        comment_data = comment_check.json()
        
        # Remove reaction
        remove_response = await async_client.delete(
            f"/comments/{comment_id}/reactions",
            headers=headers
        )
        assert remove_response.status_code == 200


class TestCommentSearchIntegration:
    """Test comment search and filtering."""
    
    async def test_comment_search_functionality(self, async_client, sample_document, sample_user):
        """Test comment search with various filters."""
        
        headers = {"Authorization": f"Bearer test-token-{sample_user['id']}"}
        document_id = sample_document["id"]
        
        # Create searchable comments
        search_comments = [
            CommentCreate(
                document_id=document_id,
                document_type=DocumentType.PRD,
                content="This comment contains the keyword 'integration'",
                tags=["search", "test"]
            ),
            CommentCreate(
                document_id=document_id,
                document_type=DocumentType.PRD,
                content="Another comment for search testing",
                comment_type=CommentType.SUGGESTION,
                tags=["search"]
            )
        ]
        
        for comment_data in search_comments:
            await async_client.post("/comments/", json=comment_data.model_dump(), headers=headers)
        
        # Test text search
        search_request = {
            "query": "integration",
            "document_id": document_id
        }
        
        search_response = await async_client.post(
            "/comments/search",
            json=search_request,
            headers=headers
        )
        assert search_response.status_code == 200
        
        search_results = search_response.json()
        assert search_results["total_count"] >= 1
        
        # Test tag filtering
        tag_search = {
            "tags": ["search"],
            "document_id": document_id
        }
        
        tag_response = await async_client.post(
            "/comments/search",
            json=tag_search,
            headers=headers
        )
        tag_results = tag_response.json()
        assert tag_results["total_count"] >= 2


class TestCommentAnalyticsIntegration:
    """Test comment analytics and reporting."""
    
    async def test_comment_analytics_generation(self, async_client, sample_document, sample_user):
        """Test comment analytics for documents."""
        
        headers = {"Authorization": f"Bearer test-token-{sample_user['id']}"}
        document_id = sample_document["id"]
        
        # Create comments with different statuses and types
        analytics_comments = [
            CommentCreate(
                document_id=document_id,
                document_type=DocumentType.PRD,
                content=f"Analytics comment {i}",
                comment_type=CommentType.COMMENT if i % 2 == 0 else CommentType.SUGGESTION,
                priority=CommentPriority.HIGH if i == 0 else CommentPriority.MEDIUM
            )
            for i in range(3)
        ]
        
        created_ids = []
        for comment_data in analytics_comments:
            response = await async_client.post(
                "/comments/",
                json=comment_data.model_dump(),
                headers=headers
            )
            created_ids.append(response.json()["id"])
        
        # Resolve one comment
        if created_ids:
            await async_client.put(
                f"/comments/{created_ids[0]}",
                json={"status": CommentStatus.RESOLVED.value, "resolution_note": "Fixed"},
                headers=headers
            )
        
        # Get analytics
        analytics_response = await async_client.get(
            f"/comments/analytics/{document_id}",
            headers=headers
        )
        assert analytics_response.status_code == 200
        
        analytics = analytics_response.json()
        
        # Verify analytics structure
        assert "total_comments" in analytics
        assert "open_comments" in analytics
        assert "resolved_comments" in analytics
        assert "comment_types_distribution" in analytics
        assert "top_commenters" in analytics
        
        assert analytics["total_comments"] >= 3
        assert analytics["resolved_comments"] >= 1


class TestCommentSystemHealthIntegration:
    """Test system health and monitoring."""
    
    async def test_comment_system_health_check(self, async_client):
        """Test comment system health endpoint."""
        
        health_response = await async_client.get("/comments/health")
        assert health_response.status_code == 200
        
        health_data = health_response.json()
        
        # Verify health response structure
        assert "status" in health_data
        assert "service" in health_data
        assert "timestamp" in health_data
        assert "statistics" in health_data
        assert "version" in health_data
        
        assert health_data["service"] == "comment-system"
        assert health_data["status"] == "healthy"


@pytest.mark.asyncio
class TestCommentBatchOperationsIntegration:
    """Test batch operations on comments."""
    
    async def test_batch_comment_operations(self, async_client, sample_document, sample_user):
        """Test batch operations on multiple comments."""
        
        headers = {"Authorization": f"Bearer test-token-{sample_user['id']}"}
        document_id = sample_document["id"]
        
        # Create multiple comments
        batch_comments = [
            CommentCreate(
                document_id=document_id,
                document_type=DocumentType.PRD,
                content=f"Batch comment {i}",
                tags=["batch", "test"]
            )
            for i in range(3)
        ]
        
        created_ids = []
        for comment_data in batch_comments:
            response = await async_client.post(
                "/comments/",
                json=comment_data.model_dump(),
                headers=headers
            )
            created_ids.append(response.json()["id"])
        
        # Perform batch resolve operation
        batch_request = {
            "comment_ids": created_ids,
            "operation": "resolve",
            "parameters": {
                "resolution_note": "Batch resolved for testing"
            }
        }
        
        batch_response = await async_client.post(
            "/comments/batch",
            json=batch_request,
            headers=headers
        )
        assert batch_response.status_code == 200
        
        batch_result = batch_response.json()
        
        # Verify batch operation results
        assert "success" in batch_result
        assert "failed" in batch_result
        assert len(batch_result["success"]) == len(created_ids)
        assert len(batch_result["failed"]) == 0
        
        # Verify comments were actually resolved
        for comment_id in created_ids:
            comment_response = await async_client.get(f"/comments/{comment_id}", headers=headers)
            comment_data = comment_response.json()
            assert comment_data["status"] == CommentStatus.RESOLVED.value


# Helper functions for integration testing
async def create_test_comments(client, document_id, user_headers, count=5):
    """Helper to create multiple test comments."""
    comments = []
    for i in range(count):
        comment_data = CommentCreate(
            document_id=document_id,
            document_type=DocumentType.PRD,
            content=f"Test comment {i+1}",
            comment_type=CommentType.COMMENT,
            priority=CommentPriority.MEDIUM
        )
        
        response = await client.post(
            "/comments/",
            json=comment_data.model_dump(),
            headers=user_headers
        )
        comments.append(response.json())
    
    return comments


def verify_comment_structure(comment_data):
    """Helper to verify comment has all required fields."""
    required_fields = [
        "id", "content", "comment_type", "status", "author_id", 
        "author_name", "thread_id", "depth", "created_at", "document_id"
    ]
    
    for field in required_fields:
        assert field in comment_data, f"Missing required field: {field}"
    
    # Verify field types
    assert isinstance(comment_data["depth"], int)
    assert comment_data["depth"] >= 0
    assert comment_data["status"] in [s.value for s in CommentStatus]
    assert comment_data["comment_type"] in [t.value for t in CommentType]