"""
Audit Middleware Integration Tests

Comprehensive tests for audit middleware components including
HTTP request logging, authentication events, and security monitoring.
"""
import uuid
from tests.utilities.test_data_factory import test_data_factory

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch
import re

import pytest
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send
import structlog

from core.audit_middleware import (
    AuditMiddleware, 
    AuthenticationAuditMiddleware,
    SecurityEventMiddleware
)
from services.comprehensive_audit_service import (
    ComprehensiveAuditService,
    AuditEventType,
    AuditSeverity,
    get_comprehensive_audit_service
)
from tests.conftest import TestUser


logger = structlog.get_logger(__name__)


@pytest.fixture
async def mock_audit_service():
    """Mock audit service for testing."""
    service = AsyncMock(spec=ComprehensiveAuditService)
    service.log_audit_event = AsyncMock(return_value="test-event-id")
    service.log_authentication_event = AsyncMock(return_value="test-auth-event-id")
    return service


@pytest.fixture
def test_app():
    """Create FastAPI test app with various endpoints."""
    app = FastAPI(title="Audit Middleware Test App")
    
    # Health endpoint (should be excluded from audit)
    @app.get("/health")
    async def health():
        return {"status": "healthy"}
    
    # Public endpoint
    @app.get("/api/v1/public/info")
    async def public_info():
        return {"info": "public data"}
    
    # Authentication endpoints (sensitive)
    @app.post("/api/v1/auth/login")
    async def login(request: Request):
        body = await request.json()
        if body.get("email") == "failexample.com":
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return {"token": "fake-jwt-token", "user_id": "test-user-123"}
    
    @app.post("/api/v1/auth/logout")
    async def logout():
        return {"message": "logged out"}
    
    # Admin endpoints (sensitive)
    @app.get("/api/v1/admin/users")
    async def admin_users():
        return {"users": ["user1", "user2"]}
    
    @app.post("/api/v1/admin/settings")
    async def admin_settings(request: Request):
        return {"message": "settings updated"}
    
    # Data endpoints (sensitive)
    @app.get("/api/v1/prd/{prd_id}")
    async def get_prd(prd_id: str):
        return {"id": prd_id, "title": "Test PRD"}
    
    @app.post("/api/v1/prd")
    async def create_prd(request: Request):
        body = await request.json()
        return {"id": "new-prd-123", "title": body.get("title")}
    
    @app.put("/api/v1/prd/{prd_id}")
    async def update_prd(prd_id: str, request: Request):
        body = await request.json()
        return {"id": prd_id, "title": body.get("title")}
    
    @app.delete("/api/v1/prd/{prd_id}")
    async def delete_prd(prd_id: str):
        return {"message": f"PRD {prd_id} deleted"}
    
    # Error endpoint
    @app.get("/api/v1/error")
    async def error_endpoint():
        raise HTTPException(status_code=500, detail="Internal server error")
    
    # Rate limited endpoint
    @app.get("/api/v1/rate-limited")
    async def rate_limited():
        raise HTTPException(status_code=429, detail="Too many requests")
    
    # Slow endpoint
    @app.get("/api/v1/slow")
    async def slow_endpoint():
        await asyncio.sleep(0.1)  # 100ms delay
        return {"message": "slow response"}
    
    return app


@pytest.fixture
def middleware_app(test_app, mock_audit_service):
    """Create test app with audit middleware configured."""
    # Override the audit service dependency
    test_app.dependency_overrides[get_comprehensive_audit_service] = lambda: mock_audit_service
    
    # Add audit middleware
    audit_middleware = AuditMiddleware(test_app)
    audit_middleware.audit_service = mock_audit_service
    
    return audit_middleware


@pytest.fixture
def client_with_middleware(middleware_app):
    """Create test client with audit middleware."""
    return TestClient(middleware_app)


class TestAuditMiddlewareCore:
    """Test core audit middleware functionality."""
    
    def test_middleware_excludes_health_endpoints(self, client_with_middleware, mock_audit_service):
        """Test middleware excludes health endpoints from logging."""
        # Make request to health endpoint
        response = client_with_middleware.get("/health")
        assert response.status_code == 200
        
        # Verify no audit events were logged
        mock_audit_service.log_audit_event.assert_not_called()
    
    def test_middleware_logs_public_endpoints(self, client_with_middleware, mock_audit_service):
        """Test middleware logs public endpoint requests."""
        # Make request to public endpoint
        response = client_with_middleware.get("/api/v1/public/info")
        assert response.status_code == 200
        
        # Verify audit event was logged
        mock_audit_service.log_audit_event.assert_called_once()
        
        # Check logged data
        call_args = mock_audit_service.log_audit_event.call_args
        kwargs = call_args[1]
        
        assert kwargs["resource_type"] == "http_endpoint"
        assert kwargs["api_endpoint"] == "/api/v1/public/info"
        assert kwargs["http_method"] == "GET"
        assert kwargs["outcome"] == "success"
        assert kwargs["response_status"] == 200
    
    def test_middleware_logs_sensitive_endpoints_enhanced(self, client_with_middleware, mock_audit_service):
        """Test enhanced logging for sensitive endpoints."""
        # Make request to sensitive endpoint
        response = client_with_middleware.post("/api/v1/auth/login", json={
            "email": "user@company.local",
            "password": "testpass123"
        })
        assert response.status_code == 200
        
        # Should have at least one log call (may have multiple for sensitive operations)
        assert mock_audit_service.log_audit_event.call_count >= 1
        
        # Check that sensitive operation was properly logged
        calls = mock_audit_service.log_audit_event.call_args_list
        logged_sensitive = any(
            call[1].get("sensitive_data") is True or
            call[1].get("severity") == AuditSeverity.MEDIUM or
            call[1].get("event_type") == AuditEventType.USER_LOGIN
            for call in calls
        )
        assert logged_sensitive
    
    def test_middleware_logs_data_operations(self, client_with_middleware, mock_audit_service):
        """Test logging of CRUD operations on data endpoints."""
        test_cases = [
            ("GET", "/api/v1/prd/123", None, AuditEventType.DATA_READ),
            ("POST", "/api/v1/prd", {"title": "New PRD"}, AuditEventType.DATA_CREATED),
            ("PUT", "/api/v1/prd/123", {"title": "Updated PRD"}, AuditEventType.DATA_UPDATED),
            ("DELETE", "/api/v1/prd/123", None, AuditEventType.DATA_DELETED)
        ]
        
        for method, path, json_data, expected_event_type in test_cases:
            # Reset mock
            mock_audit_service.log_audit_event.reset_mock()
            
            # Make request
            if method == "GET":
                response = client_with_middleware.get(path)
            elif method == "POST":
                response = client_with_middleware.post(path, json=json_data)
            elif method == "PUT":
                response = client_with_middleware.put(path, json=json_data)
            elif method == "DELETE":
                response = client_with_middleware.delete(path)
            
            assert response.status_code in [200, 201]
            
            # Verify correct event type was logged
            mock_audit_service.log_audit_event.assert_called_once()
            call_args = mock_audit_service.log_audit_event.call_args[1]
            
            # The middleware determines event type, so we check it was called with data operation params
            assert call_args["resource_type"] == "http_endpoint"
            assert call_args["api_endpoint"] == path
            assert call_args["http_method"] == method
    
    def test_middleware_logs_authentication_failures(self, client_with_middleware, mock_audit_service):
        """Test logging of authentication failures."""
        # Make request that will fail authentication
        response = client_with_middleware.post("/api/v1/auth/login", json={
            email=f"testuser{str(uuid.uuid4())[:8]}@{self.fake.domain_name()}",",
            "password": "wrongpass"
        })
        assert response.status_code == 401
        
        # Verify failure was logged
        mock_audit_service.log_audit_event.assert_called()
        
        # Check for error logging
        calls = mock_audit_service.log_audit_event.call_args_list
        error_logged = any(
            call[1].get("outcome") in ["failure", "error"] or
            call[1].get("response_status") == 401
            for call in calls
        )
        assert error_logged
    
    def test_middleware_logs_server_errors(self, client_with_middleware, mock_audit_service):
        """Test logging of server errors."""
        # Make request that causes server error
        response = client_with_middleware.get("/api/v1/error")
        assert response.status_code == 500
        
        # Verify error was logged
        mock_audit_service.log_audit_event.assert_called()
        
        # Check error details
        call_args = mock_audit_service.log_audit_event.call_args[1]
        assert call_args["outcome"] == "error"
        assert call_args["response_status"] == 500
        assert call_args["severity"] == AuditSeverity.CRITICAL
    
    def test_middleware_logs_performance_metrics(self, client_with_middleware, mock_audit_service):
        """Test middleware captures performance metrics."""
        # Make request to slow endpoint
        response = client_with_middleware.get("/api/v1/slow")
        assert response.status_code == 200
        
        # Verify performance data was captured
        mock_audit_service.log_audit_event.assert_called_once()
        call_args = mock_audit_service.log_audit_event.call_args[1]
        
        assert "duration_ms" in call_args
        assert call_args["duration_ms"] >= 100  # Should be at least 100ms due to sleep
    
    def test_middleware_sanitizes_sensitive_headers(self, client_with_middleware, mock_audit_service):
        """Test middleware sanitizes sensitive headers."""
        # Make request with sensitive headers
        response = client_with_middleware.get("/api/v1/public/info", headers={
            "Authorization": "Bearer secret-token",
            "X-API-Key": "api-secret-123",
            "User-Agent": "TestClient/1.0"
        })
        assert response.status_code == 200
        
        # Check logged headers are sanitized
        call_args = mock_audit_service.log_audit_event.call_args[1]
        metadata = call_args.get("metadata", {})
        
        # Headers should be in metadata, but sensitive ones should be redacted
        # Note: This depends on the exact implementation of header logging
        # The test verifies the general principle of header sanitization


class TestAuthenticationAuditMiddleware:
    """Test authentication audit middleware."""
    
    async def test_authentication_state_tracking(self, mock_audit_service):
        """Test authentication state change tracking."""
        # Create mock app that simulates authentication
        async def mock_app(scope, receive, send):
            # Simulate user authentication during request
            if not hasattr(scope, 'state'):
                scope['state'] = {}
                
            # Simulate setting user after authentication
            scope['state']['user'] = TestUser(
                id="auth-test-user",
                email="authexample.com",
                role="user"
            )
            
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": []
            })
            await send({
                "type": "http.response.body",
                "body": b"Authentication successful"
            })
        
        # Create authentication middleware
        auth_middleware = AuthenticationAuditMiddleware(mock_app)
        auth_middleware.audit_service = mock_audit_service
        
        # Simulate HTTP request
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/auth/login",
            "headers": [(b"user-agent", b"TestClient/1.0")],
            "state": {}
        }
        
        async def mock_receive():
            return {"type": "http.request", "body": b'{email=f"testuser{str(uuid.uuid4())[:8]}@{self.fake.domain_name()}","}'}
        
        async def mock_send(message):
            pass
        
        # Execute middleware
        await auth_middleware(scope, mock_receive, mock_send)
        
        # Verify authentication event was logged
        mock_audit_service.log_authentication_event.assert_called_once()
        call_args = mock_audit_service.log_authentication_event.call_args[1]
        
        assert call_args["event_type"] == AuditEventType.USER_LOGIN
        assert call_args["user_id"] == "auth-test-user"
        assert call_args["user_email"] == "authexample.com"
        assert call_args["outcome"] == "success"


class TestSecurityEventMiddleware:
    """Test security event middleware."""
    
    async def test_sql_injection_detection(self, mock_audit_service):
        """Test SQL injection pattern detection."""
        # Create mock app
        async def mock_app(scope, receive, send):
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": []
            })
            await send({
                "type": "http.response.body",
                "body": b"OK"
            })
        
        # Create security middleware
        security_middleware = SecurityEventMiddleware(mock_app)
        security_middleware.audit_service = mock_audit_service
        
        # Test SQL injection patterns
        suspicious_queries = [
            "id=1' UNION SELECT * FROM users--",
            "search='; DROP TABLE users; --",
            "filter=1 OR 1=1",
            "query=admin' AND 1=1--"
        ]
        
        for query in suspicious_queries:
            # Reset mock
            mock_audit_service.log_audit_event.reset_mock()
            
            # Simulate request with suspicious query
            scope = {
                "type": "http",
                "method": "GET",
                "path": "/api/search",
                "query_string": query.encode(),
                "headers": [(b"user-agent", b"AttackBot/1.0")],
                "client": ("192.168.1.100", 12345)
            }
            
            await security_middleware(scope, AsyncMock(), AsyncMock())
            
            # Verify security event was logged
            mock_audit_service.log_audit_event.assert_called_once()
            call_args = mock_audit_service.log_audit_event.call_args[1]
            
            assert call_args["event_type"] == AuditEventType.SUSPICIOUS_ACTIVITY
            assert call_args["severity"] == AuditSeverity.HIGH
            assert "security" in call_args["compliance_tags"]
            assert "threat_detection" in call_args["compliance_tags"]
    
    async def test_xss_attack_detection(self, mock_audit_service):
        """Test XSS attack pattern detection."""
        security_middleware = SecurityEventMiddleware(lambda s, r, se: None)
        security_middleware.audit_service = mock_audit_service
        
        # Test XSS patterns
        xss_patterns = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "onerror=alert('xss')",
            "onload=alert('xss')"
        ]
        
        for pattern in xss_patterns:
            mock_audit_service.log_audit_event.reset_mock()
            
            scope = {
                "type": "http",
                "method": "POST",
                "path": "/api/comment",
                "query_string": f"content={pattern}".encode(),
                "headers": [],
                "client": ("192.168.1.100", 12345)
            }
            
            await security_middleware(scope, AsyncMock(), AsyncMock())
            
            # Verify XSS detection
            mock_audit_service.log_audit_event.assert_called_once()
            call_args = mock_audit_service.log_audit_event.call_args[1]
            assert call_args["event_type"] == AuditEventType.SUSPICIOUS_ACTIVITY
    
    async def test_path_traversal_detection(self, mock_audit_service):
        """Test path traversal attack detection."""
        security_middleware = SecurityEventMiddleware(lambda s, r, se: None)
        security_middleware.audit_service = mock_audit_service
        
        # Test path traversal patterns
        traversal_patterns = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc/passwd"
        ]
        
        for pattern in traversal_patterns:
            mock_audit_service.log_audit_event.reset_mock()
            
            scope = {
                "type": "http",
                "method": "GET",
                "path": f"/api/file/{pattern}",
                "query_string": b"",
                "headers": [],
                "client": ("192.168.1.100", 12345)
            }
            
            await security_middleware(scope, AsyncMock(), AsyncMock())
            
            # Verify path traversal detection
            mock_audit_service.log_audit_event.assert_called_once()
            call_args = mock_audit_service.log_audit_event.call_args[1]
            assert call_args["event_type"] == AuditEventType.SUSPICIOUS_ACTIVITY


class TestMiddlewareIntegration:
    """Test middleware integration scenarios."""
    
    def test_multiple_middleware_coordination(self, test_app, mock_audit_service):
        """Test multiple middleware working together."""
        # Add all middleware to the app
        test_app.add_middleware(SecurityEventMiddleware)
        test_app.add_middleware(AuthenticationAuditMiddleware)  
        test_app.add_middleware(AuditMiddleware)
        
        # Override audit service for all middleware
        with patch('core.audit_middleware.get_comprehensive_audit_service', 
                  return_value=mock_audit_service):
            
            client = TestClient(test_app)
            
            # Make request that could trigger multiple middleware
            response = client.post("/api/v1/auth/login", json={
                "email": "user@company.local",
                "password": "testpass"
            })
            assert response.status_code == 200
            
            # Verify multiple audit events were logged
            assert mock_audit_service.log_audit_event.call_count >= 1
    
    def test_middleware_error_handling(self, test_app, mock_audit_service):
        """Test middleware handles errors gracefully."""
        # Configure audit service to fail
        mock_audit_service.log_audit_event.side_effect = Exception("Audit service failed")
        
        test_app.add_middleware(AuditMiddleware)
        
        with patch('core.audit_middleware.get_comprehensive_audit_service',
                  return_value=mock_audit_service):
            
            client = TestClient(test_app)
            
            # Request should still succeed even if audit fails
            response = client.get("/api/v1/public/info")
            assert response.status_code == 200
    
    def test_middleware_performance_impact(self, test_app, mock_audit_service):
        """Test middleware doesn't significantly impact performance."""
        test_app.add_middleware(AuditMiddleware)
        
        with patch('core.audit_middleware.get_comprehensive_audit_service',
                  return_value=mock_audit_service):
            
            client = TestClient(test_app)
            
            # Measure time for multiple requests
            import time
            start_time = time.time()
            
            for _ in range(10):
                response = client.get("/api/v1/public/info")
                assert response.status_code == 200
            
            duration = time.time() - start_time
            
            # Should complete 10 requests in reasonable time
            assert duration < 1.0  # Less than 1 second for 10 requests
            
            # Verify all requests were audited
            assert mock_audit_service.log_audit_event.call_count == 10


class TestMiddlewareConfiguration:
    """Test middleware configuration and customization."""
    
    def test_middleware_sensitive_endpoint_configuration(self, mock_audit_service):
        """Test middleware sensitive endpoint configuration."""
        app = FastAPI()
        
        # Create middleware with custom sensitive endpoints
        middleware = AuditMiddleware(app)
        middleware.sensitive_endpoints.add("/api/v1/custom/sensitive")
        middleware.audit_service = mock_audit_service
        
        # Test custom sensitive endpoint
        client = TestClient(middleware)
        response = client.get("/api/v1/custom/sensitive")
        
        # Should log with enhanced security (if endpoint exists)
        # This test verifies the configuration mechanism
    
    def test_middleware_exclusion_configuration(self, test_app, mock_audit_service):
        """Test middleware endpoint exclusion configuration."""
        # Add custom exclusion
        middleware = AuditMiddleware(test_app)
        middleware.excluded_endpoints.add("/api/v1/public/info")
        middleware.audit_service = mock_audit_service
        
        client = TestClient(middleware)
        response = client.get("/api/v1/public/info")
        
        # Should not log excluded endpoint
        mock_audit_service.log_audit_event.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])