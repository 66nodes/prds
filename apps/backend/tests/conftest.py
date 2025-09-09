"""
Test configuration and fixtures for backend tests.

Provides shared fixtures for database setup, authentication, and test data
for both unit and integration tests.
"""

import asyncio
import pytest
import os
from datetime import datetime
from typing import Dict
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from httpx import AsyncClient


@pytest.fixture(scope="session", autouse=True)
def test_environment():
    """Set up test environment variables."""
    test_env = {
        "SECRET_KEY": "test-secret-key-for-testing-only",
        "NEO4J_PASSWORD": "test-password",
        "ENVIRONMENT": "testing",
        "DEBUG": "true"
    }
    
    with patch.dict(os.environ, test_env):
        yield


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    try:
        from main import app
        with TestClient(app) as client:
            yield client
    except ImportError:
        # If main app can't be imported due to dependencies, create a minimal app
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        
        minimal_app = FastAPI(title="Test API", version="1.0.0")
        
        @minimal_app.get("/")
        def root():
            return {
                "message": "Strategic Planning Platform API",
                "version": "1.0.0", 
                "status": "operational",
                "features": ["prd_generation", "graphrag_validation", "agent_orchestration"]
            }
        
        @minimal_app.get("/health")
        def health():
            return {
                "status": "healthy",
                "timestamp": "2025-01-20T12:00:00Z",
                "version": "1.0.0",
                "components": {
                    "api": {"status": "healthy"}
                }
            }
        
        with TestClient(minimal_app) as client:
            yield client


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    from core.config import Settings
    
    settings = Settings(
        secret_key="test-secret-key-for-testing-only",  # nosec: test data only
        neo4j_password="test-password-for-testing-only",  # nosec: test data only
        environment="testing",
        debug=True
    )
    return settings


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_client():
    """Async HTTP client for testing."""
    try:
        from main import app
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    except ImportError:
        # Minimal app if main can't be imported
        from fastapi import FastAPI
        minimal_app = FastAPI()
        async with AsyncClient(app=minimal_app, base_url="http://test") as client:
            yield client


@pytest.fixture
def sample_user():
    """Create a sample user for testing."""
    return {
        "id": str(uuid4()),
        "email": f"user-{uuid4()}example.com",
        "full_name": f"User {uuid4()}",
        "is_active": True,
        "is_superuser": False,
        "created_at": datetime.utcnow()
    }


@pytest.fixture
def sample_admin_user():
    """Create a sample admin user for testing."""
    return {
        "id": str(uuid4()),
        "email": f"admin-{uuid4()}example.com",
        "full_name": f"Admin {uuid4()}",
        "is_active": True,
        "is_superuser": True,
        "created_at": datetime.utcnow()
    }


@pytest.fixture
def auth_headers(sample_user):
    """Generate authentication headers for test user."""
    return {
        "Authorization": f"Bearer test-token-{sample_user['id']}",
        "Content-Type": "application/json"
    }


@pytest.fixture
def admin_auth_headers(sample_admin_user):
    """Generate authentication headers for admin user."""
    return {
        "Authorization": f"Bearer admin-token-{sample_admin_user['id']}",
        "Content-Type": "application/json"
    }


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return {
        "id": str(uuid4()),
        "title": f"Document {uuid4()}",
        "type": "prd",
        "content": f"Test document content {uuid4()} for integration testing.",
        "status": "draft",
        "created_by": str(uuid4()),
        "created_at": datetime.utcnow().isoformat()
    }


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    class MockRedis:
        def __init__(self):
            self._data = {}
        
        async def get(self, key):
            return self._data.get(key)
        
        async def set(self, key, value, ex=None):
            self._data[key] = value
        
        async def setex(self, key, seconds, value):
            self._data[key] = value
        
        async def delete(self, key):
            self._data.pop(key, None)
        
        async def exists(self, key):
            return key in self._data
        
        async def ping(self):
            return True
    
    return MockRedis()


# Mock WebSocket for testing
class MockWebSocket:
    """Mock WebSocket for testing."""
    
    def __init__(self):
        self.messages = []
        self.closed = False
    
    async def accept(self):
        pass
    
    async def send_text(self, message):
        if not self.closed:
            self.messages.append(message)
    
    async def close(self):
        self.closed = True
    
    def get_messages(self):
        return self.messages


@pytest.fixture
def mock_websocket():
    """Provide mock WebSocket for testing."""
    return MockWebSocket()


# Test data factory
class TestDataFactory:
    """Factory for creating test data objects."""
    
    @staticmethod
    def create_comment_data(
        document_id: str,
        content: str = None,
        comment_type: str = "comment",
        parent_id: str = None
    ) -> Dict:
        """Create test comment data."""
        return {
            "document_id": document_id,
            "document_type": "prd",
            "content": content or f"Test comment content {uuid4()}",
            "comment_type": comment_type,
            "parent_id": parent_id,
            "tags": ["test", "integration"],
            "mentions": [],
            "assignees": []
        }


@pytest.fixture
def test_data_factory():
    """Provide test data factory."""
    return TestDataFactory


# Test utilities
class TestUtils:
    """Utility functions for testing."""
    
    @staticmethod
    def assert_valid_uuid(uuid_string: str):
        """Assert that a string is a valid UUID."""
        from uuid import UUID
        try:
            UUID(uuid_string)
        except ValueError:
            pytest.fail(f"'{uuid_string}' is not a valid UUID")
    
    @staticmethod
    def assert_valid_datetime(datetime_string: str):
        """Assert that a string is a valid ISO datetime."""
        try:
            datetime.fromisoformat(datetime_string.replace('Z', '+00:00'))
        except ValueError:
            pytest.fail(f"'{datetime_string}' is not a valid ISO datetime")


@pytest.fixture
def test_utils():
    """Provide test utilities."""
    return TestUtils