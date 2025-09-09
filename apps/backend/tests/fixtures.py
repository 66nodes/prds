"""
Test fixtures and utilities for the Strategic Planning Platform backend.

Provides reusable test data, database setup, and common testing utilities
for comprehensive test coverage across all components.
"""

import asyncio
import os
import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, AsyncGenerator
from unittest.mock import Mock, patch

# FastAPI and database imports
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Redis imports
import redis.asyncio as aioredis
from fakeredis import aioredis as fake_aioredis

# Application imports
from main import app
from core.database import Base, get_db
from core.config import get_settings
from services.auth_service import create_access_token, get_password_hash
from services.graphrag.graph_service import GraphService

# Test configuration
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"
REDIS_URL = "redis://localhost:6379/1"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={
            "check_same_thread": False,
        },
        poolclass=StaticPool,
        echo=False,
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest_asyncio.fixture
async def test_db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async_session = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
def test_client(test_db_session):
    """Create test client with database dependency override."""
    def override_get_db():
        return test_db_session
    
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def redis_client():
    """Create Redis test client."""
    if os.getenv("ENVIRONMENT") == "test":
        # Use fake Redis for testing
        client = fake_aioredis.FakeRedis(decode_responses=True)
    else:
        # Use real Redis if available
        client = aioredis.from_url(REDIS_URL, decode_responses=True)
    
    yield client
    
    # Cleanup
    await client.flushdb()
    await client.close()


@pytest.fixture
def mock_graph_service():
    """Mock GraphRAG service for testing."""
    mock_service = Mock(spec=GraphService)
    
    # Mock common methods
    mock_service.validate_content.return_value = {
        "is_valid": True,
        "hallucination_score": 0.01,
        "confidence": 0.95,
        "evidence": ["test evidence"]
    }
    
    mock_service.extract_entities.return_value = [
        {"name": "Test Entity", "type": "CONCEPT", "confidence": 0.9}
    ]
    
    mock_service.extract_relationships.return_value = [
        {
            "source": "Entity A",
            "target": "Entity B", 
            "relationship": "RELATES_TO",
            "confidence": 0.8
        }
    ]
    
    return mock_service


# User fixtures
@pytest.fixture
def test_user_data() -> Dict:
    """Test user data."""
    return {
        "id": "test-user-123",
        "email": "user@company.local",
        "full_name": "Test User",
        "is_active": True,
        "is_superuser": False,
        "created_at": datetime.utcnow(),
        "last_login": datetime.utcnow(),
    }


@pytest.fixture
def admin_user_data() -> Dict:
    """Admin user data."""
    return {
        "id": "admin-user-456",
        "email": "admin@strategic-planning.ai",
        "full_name": "Admin User",
        "is_active": True,
        "is_superuser": True,
        "created_at": datetime.utcnow(),
        "last_login": datetime.utcnow(),
    }


@pytest.fixture
def test_user_token(test_user_data) -> str:
    """Generate test user JWT token."""
    return create_access_token(
        subject=test_user_data["id"],
        expires_delta=timedelta(hours=1)
    )


@pytest.fixture
def admin_user_token(admin_user_data) -> str:
    """Generate admin user JWT token."""
    return create_access_token(
        subject=admin_user_data["id"],
        expires_delta=timedelta(hours=1)
    )


@pytest.fixture
def auth_headers(test_user_token) -> Dict[str, str]:
    """Generate authorization headers."""
    return {"Authorization": f"Bearer {test_user_token}"}


@pytest.fixture
def admin_auth_headers(admin_user_token) -> Dict[str, str]:
    """Generate admin authorization headers."""
    return {"Authorization": f"Bearer {admin_user_token}"}


# Project fixtures
@pytest.fixture
def test_project_data() -> Dict:
    """Test project data."""
    return {
        "id": "project-123",
        "title": "Test Project",
        "description": "A test project for automated testing",
        "status": "active",
        "owner_id": "test-user-123",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "settings": {
            "visibility": "private",
            "collaboration_enabled": True,
            "notifications_enabled": True
        }
    }


@pytest.fixture
def test_project_list() -> List[Dict]:
    """List of test projects."""
    return [
        {
            "id": f"project-{i}",
            "title": f"Test Project {i}",
            "description": f"Test project number {i}",
            "status": "active" if i % 2 == 0 else "draft",
            "owner_id": "test-user-123",
            "created_at": datetime.utcnow() - timedelta(days=i),
            "updated_at": datetime.utcnow() - timedelta(hours=i),
        }
        for i in range(1, 6)
    ]


# PRD fixtures
@pytest.fixture
def test_prd_data() -> Dict:
    """Test PRD data."""
    return {
        "id": "prd-123",
        "title": "Test PRD",
        "description": "A test PRD for automated testing",
        "content": """
# Test PRD

## Overview
This is a test PRD created for automated testing purposes.

## Requirements
- Functional requirement 1
- Functional requirement 2
- Non-functional requirement 1

## Success Metrics
- Metric 1: 95% accuracy
- Metric 2: <200ms response time
        """.strip(),
        "status": "draft",
        "project_id": "project-123",
        "author_id": "test-user-123",
        "version": "1.0.0",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "metadata": {
            "word_count": 150,
            "estimated_reading_time": 2,
            "complexity_score": 0.6,
            "hallucination_score": 0.01
        }
    }


@pytest.fixture
def test_prd_request() -> Dict:
    """Test PRD creation request."""
    return {
        "title": "New Test PRD",
        "description": "Creating a new PRD for testing",
        "requirements": [
            "User authentication system",
            "Dashboard with analytics", 
            "API rate limiting"
        ],
        "success_metrics": [
            "99.9% uptime",
            "Sub-100ms API response times",
            "Support 10,000+ concurrent users"
        ],
        "target_audience": "Software engineers and product managers",
        "timeline": "3 months",
        "budget": "$100,000"
    }


# Agent fixtures
@pytest.fixture
def test_agent_data() -> Dict:
    """Test agent data."""
    return {
        "id": "agent-123",
        "name": "Test Agent",
        "type": "prd_generator",
        "version": "1.0.0",
        "status": "active",
        "capabilities": [
            "requirement_analysis",
            "content_generation",
            "validation"
        ],
        "configuration": {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 4000
        },
        "performance_metrics": {
            "success_rate": 0.95,
            "avg_response_time": 2.5,
            "hallucination_rate": 0.02
        }
    }


# Comment system fixtures
@pytest.fixture
def test_comment_data() -> Dict:
    """Test comment data."""
    return {
        "id": "comment-123",
        "content": "This is a test comment for automated testing",
        "document_id": "prd-123",
        "document_type": "prd",
        "author_id": "test-user-123",
        "parent_id": None,
        "status": "active",
        "metadata": {
            "selection_start": 100,
            "selection_end": 150,
            "selected_text": "This is the selected text"
        },
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }


# Performance test fixtures
@pytest.fixture
def performance_test_config() -> Dict:
    """Performance testing configuration."""
    return {
        "load_test": {
            "concurrent_users": 10,
            "duration": 30,
            "ramp_up_time": 10
        },
        "stress_test": {
            "concurrent_users": 100,
            "duration": 60,
            "ramp_up_time": 30
        },
        "spike_test": {
            "concurrent_users": 500,
            "duration": 10,
            "ramp_up_time": 2
        },
        "thresholds": {
            "response_time_p95": 500,  # 500ms
            "response_time_p99": 1000,  # 1s
            "error_rate": 0.01,  # 1%
            "throughput_min": 100  # requests/second
        }
    }


# Security test fixtures
@pytest.fixture
def security_test_payloads() -> Dict[str, List[str]]:
    """Security testing payloads."""
    return {
        "sql_injection": [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "' UNION SELECT * FROM users --"
        ],
        "xss": [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>"
        ],
        "path_traversal": [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ],
        "command_injection": [
            "; ls -la",
            "| cat /etc/passwd",
            "`whoami`"
        ]
    }


# Utility functions
def create_test_file(content: str, filename: str = "test_file.txt") -> str:
    """Create a temporary test file."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix=filename, delete=False) as f:
        f.write(content)
        return f.name


def assert_response_structure(response: Dict, expected_keys: List[str]):
    """Assert that response has expected structure."""
    for key in expected_keys:
        assert key in response, f"Expected key '{key}' not found in response"


def assert_coverage_threshold(coverage_data: Dict, threshold: float = 0.9):
    """Assert that test coverage meets threshold."""
    line_coverage = coverage_data.get("line_rate", 0)
    branch_coverage = coverage_data.get("branch_rate", 0)
    
    assert line_coverage >= threshold, f"Line coverage {line_coverage} below threshold {threshold}"
    assert branch_coverage >= (threshold - 0.05), f"Branch coverage {branch_coverage} below threshold {threshold - 0.05}"


# Async utilities
async def wait_for_condition(condition_func, timeout: int = 10, interval: float = 0.1):
    """Wait for a condition to be true with timeout."""
    start_time = asyncio.get_event_loop().time()
    while True:
        if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
            return True
        
        if asyncio.get_event_loop().time() - start_time > timeout:
            raise TimeoutError(f"Condition not met within {timeout} seconds")
        
        await asyncio.sleep(interval)


# Database utilities
async def clear_test_data(session: AsyncSession, table_names: List[str]):
    """Clear test data from specified tables."""
    for table_name in table_names:
        await session.execute(f"DELETE FROM {table_name}")
    await session.commit()


async def create_test_user(session: AsyncSession, user_data: Dict) -> str:
    """Create a test user in the database."""
    # Implementation would depend on your User model
    # This is a placeholder
    return user_data["id"]


# Mock external services
@pytest.fixture
def mock_openrouter_client():
    """Mock OpenRouter client for LLM interactions."""
    with patch('services.llm.openrouter_client.OpenRouterClient') as mock:
        mock_instance = mock.return_value
        mock_instance.generate.return_value = {
            "content": "Mock generated content",
            "usage": {"prompt_tokens": 100, "completion_tokens": 200},
            "model": "gpt-4",
            "finish_reason": "stop"
        }
        yield mock_instance


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver for graph database operations."""
    with patch('neo4j.GraphDatabase.driver') as mock:
        mock_driver = mock.return_value
        mock_session = Mock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = []
        yield mock_driver


# Test data cleanup
@pytest.fixture(autouse=True)
async def cleanup_test_data(test_db_session: AsyncSession):
    """Automatically cleanup test data after each test."""
    yield
    
    # Cleanup logic - this would depend on your specific models
    # Example:
    # await clear_test_data(test_db_session, [
    #     "comments", "prds", "projects", "users"  
    # ])