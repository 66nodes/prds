"""
Comprehensive Integration Tests for Audit System

Tests the complete audit logging system including service, middleware,
API endpoints, compliance reporting, and data management features.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient
import structlog

from core.config import get_settings
from services.comprehensive_audit_service import (
    ComprehensiveAuditService,
    AuditEventType,
    AuditSeverity,
    ComplianceStandard,
    get_comprehensive_audit_service
)
from core.audit_middleware import AuditMiddleware, AuthenticationAuditMiddleware, SecurityEventMiddleware
from api.endpoints.audit import router as audit_router
from core.redis import get_redis_client
from tests.conftest import TestUser


logger = structlog.get_logger(__name__)
settings = get_settings()


@pytest.fixture
async def audit_service():
    """Create audit service instance for testing."""
    service = ComprehensiveAuditService()
    await service.initialize()
    yield service
    await service.cleanup()


@pytest.fixture
def audit_app(audit_service):
    """Create FastAPI app with audit system configured."""
    app = FastAPI(title="Audit Test App")
    
    # Add audit middleware
    app.add_middleware(AuditMiddleware)
    
    # Add audit router
    app.include_router(audit_router)
    
    # Mock the service dependency
    app.dependency_overrides[get_comprehensive_audit_service] = lambda: audit_service
    
    return app


@pytest.fixture
async def audit_client(audit_app):
    """Create async test client for audit system."""
    async with AsyncClient(app=audit_app, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    redis_mock = AsyncMock()
    redis_mock.setex = AsyncMock()
    redis_mock.get = AsyncMock()
    redis_mock.keys = AsyncMock(return_value=[])
    redis_mock.delete = AsyncMock()
    redis_mock.scan_iter = AsyncMock()
    redis_mock.ping = AsyncMock(return_value=True)
    return redis_mock


@pytest.fixture
def test_user():
    """Create test user for authentication."""
    return TestUser(
        id="test-user-123",
        email="user@company.local",
        role="admin",
        is_active=True
    )


class TestAuditServiceCore:
    """Test core audit service functionality."""
    
    async def test_service_initialization(self):
        """Test audit service initialization."""
        service = ComprehensiveAuditService()
        assert not service.is_initialized
        
        await service.initialize()
        assert service.is_initialized
        assert service.redis_client is not None
        
        await service.cleanup()
    
    async def test_basic_audit_event_logging(self, audit_service, mock_redis):
        """Test basic audit event logging."""
        with patch('services.comprehensive_audit_service.get_redis_client', return_value=mock_redis):
            # Log a basic audit event
            event_id = await audit_service.log_audit_event(
                event_type=AuditEventType.USER_LOGIN,
                user_id="test-user-123",
                user_email="user@company.local",
                ip_address="192.168.1.1",
                outcome="success"
            )
            
            assert event_id is not None
            assert isinstance(event_id, str)
            
            # Verify Redis storage was called
            mock_redis.setex.assert_called_once()
            
            # Check stored data structure
            call_args = mock_redis.setex.call_args
            key, ttl, data = call_args[0]
            
            assert key.startswith("audit_event:")
            assert ttl == audit_service.retention_days * 24 * 3600
            
            stored_event = json.loads(data)
            assert stored_event["event_type"] == AuditEventType.USER_LOGIN.value
            assert stored_event["user_id"] == "test-user-123"
            assert stored_event["outcome"] == "success"
    
    async def test_authentication_event_logging(self, audit_service, mock_redis):
        """Test authentication-specific event logging."""
        with patch('services.comprehensive_audit_service.get_redis_client', return_value=mock_redis):
            # Log authentication event
            event_id = await audit_service.log_authentication_event(
                event_type=AuditEventType.USER_LOGIN_FAILED,
                user_email="user@company.local",
                ip_address="192.168.1.1",
                user_agent="Mozilla/5.0 Test Browser",
                outcome="failure",
                error_message="Invalid credentials"
            )
            
            assert event_id is not None
            
            # Verify storage
            mock_redis.setex.assert_called_once()
            call_args = mock_redis.setex.call_args
            stored_event = json.loads(call_args[0][2])
            
            assert stored_event["event_type"] == AuditEventType.USER_LOGIN_FAILED.value
            assert stored_event["user_agent"] == "Mozilla/5.0 Test Browser"
            assert stored_event["error_message"] == "Invalid credentials"
            assert stored_event["severity"] == AuditSeverity.HIGH.value
    
    async def test_data_operation_logging(self, audit_service, mock_redis):
        """Test data operation event logging."""
        with patch('services.comprehensive_audit_service.get_redis_client', return_value=mock_redis):
            # Log data creation event
            event_id = await audit_service.log_data_operation(
                event_type=AuditEventType.DATA_CREATED,
                user_id="test-user-123",
                resource_type="prd",
                resource_id="prd-456",
                data_classification="confidential",
                data_volume_bytes=1024,
                operation_details={
                    "table": "product_requirements",
                    "fields_modified": ["title", "description", "requirements"]
                }
            )
            
            assert event_id is not None
            
            # Verify compliance tags were added
            stored_event = json.loads(mock_redis.setex.call_args[0][2])
            assert "gdpr" in stored_event["compliance_tags"]
            assert "data_protection" in stored_event["compliance_tags"]
            assert stored_event["sensitive_data"] is True
    
    async def test_audit_event_search(self, audit_service, mock_redis):
        """Test audit event search functionality."""
        # Mock Redis search results
        mock_events = [
            {
                "event_id": "event-1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": AuditEventType.USER_LOGIN.value,
                "user_id": "user-123",
                "outcome": "success"
            },
            {
                "event_id": "event-2", 
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": AuditEventType.DATA_READ.value,
                "user_id": "user-123",
                "resource_type": "prd"
            }
        ]
        
        # Mock Redis keys and get calls
        mock_redis.keys.return_value = ["audit_event:event-1", "audit_event:event-2"]
        mock_redis.get.side_effect = [json.dumps(event) for event in mock_events]
        
        with patch('services.comprehensive_audit_service.get_redis_client', return_value=mock_redis):
            # Search for events
            events, total_count = await audit_service.search_audit_events(
                user_id="user-123",
                start_time=datetime.now(timezone.utc) - timedelta(hours=1),
                end_time=datetime.now(timezone.utc),
                limit=10
            )
            
            assert len(events) == 2
            assert total_count == 2
            assert all(event["user_id"] == "user-123" for event in events)
    
    async def test_compliance_report_generation(self, audit_service, mock_redis):
        """Test compliance report generation."""
        # Mock audit events for compliance report
        mock_events = [
            {
                "event_id": f"event-{i}",
                "timestamp": (datetime.now(timezone.utc) - timedelta(days=i)).isoformat(),
                "event_type": AuditEventType.USER_LOGIN.value,
                "severity": AuditSeverity.MEDIUM.value,
                "compliance_tags": ["soc2", "security"]
            }
            for i in range(10)
        ]
        
        mock_redis.keys.return_value = [f"audit_event:event-{i}" for i in range(10)]
        mock_redis.get.side_effect = [json.dumps(event) for event in mock_events]
        
        with patch('services.comprehensive_audit_service.get_redis_client', return_value=mock_redis):
            # Generate SOC 2 compliance report
            report = await audit_service.generate_compliance_report(
                standard=ComplianceStandard.SOC2_TYPE2,
                period_start=datetime.now(timezone.utc) - timedelta(days=30),
                period_end=datetime.now(timezone.utc)
            )
            
            assert report is not None
            assert report.report_type == ComplianceStandard.SOC2_TYPE2
            assert report.total_events == 10
            assert report.compliance_score >= 0.0
            assert len(report.sections) > 0
            assert report.generated_at is not None
    
    async def test_retention_policy_application(self, audit_service, mock_redis):
        """Test audit log retention policy application."""
        # Mock old events beyond retention period
        old_keys = [f"audit_event:old-{i}" for i in range(5)]
        recent_keys = [f"audit_event:recent-{i}" for i in range(3)]
        
        # Mock Redis scan to return old and recent events
        mock_redis.scan_iter.return_value = old_keys + recent_keys
        
        # Mock get calls for timestamp checking
        old_timestamp = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        recent_timestamp = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        
        def mock_get(key):
            if "old-" in key:
                return json.dumps({"timestamp": old_timestamp})
            else:
                return json.dumps({"timestamp": recent_timestamp})
        
        mock_redis.get.side_effect = mock_get
        
        with patch('services.comprehensive_audit_service.get_redis_client', return_value=mock_redis):
            # Apply retention policy
            summary = await audit_service.apply_retention_policy()
            
            assert summary is not None
            assert "events_deleted" in summary
            assert "events_retained" in summary
            
            # Verify old events were deleted
            assert mock_redis.delete.call_count > 0


class TestAuditMiddleware:
    """Test audit middleware functionality."""
    
    async def test_middleware_request_logging(self, audit_app, audit_client, audit_service):
        """Test middleware logs HTTP requests automatically."""
        with patch.object(audit_service, 'log_audit_event') as mock_log:
            # Make a test request
            response = await audit_client.get("/health")
            
            # Verify audit event was logged
            mock_log.assert_called_once()
            call_args = mock_log.call_args[1]
            
            assert call_args["resource_type"] == "http_endpoint"
            assert call_args["http_method"] == "GET"
            assert call_args["api_endpoint"] == "/health"
    
    async def test_middleware_sensitive_endpoint_logging(self, audit_app, audit_client, audit_service):
        """Test enhanced logging for sensitive endpoints."""
        with patch.object(audit_service, 'log_audit_event') as mock_log:
            # Make request to sensitive endpoint
            response = await audit_client.post("/api/v1/auth/login", json={
                "email": "user@company.local",
                "password": "testpass"
            })
            
            # Should have multiple log calls for sensitive operations
            assert mock_log.call_count >= 1
            
            # Check that request was marked as sensitive
            call_args = mock_log.call_args[1]
            assert call_args.get("sensitive_data") is True or call_args.get("severity") == AuditSeverity.MEDIUM
    
    async def test_middleware_error_logging(self, audit_app, audit_client, audit_service):
        """Test middleware logs errors and failures."""
        with patch.object(audit_service, 'log_audit_event') as mock_log:
            # Make request that will cause an error
            response = await audit_client.get("/nonexistent-endpoint")
            
            # Verify error was logged
            mock_log.assert_called()
            
            # Check for error-related logging
            calls = mock_log.call_args_list
            error_logged = any(
                call[1].get("outcome") in ["failure", "error"] or
                call[1].get("event_type") == AuditEventType.SYSTEM_ERROR
                for call in calls
            )
            assert error_logged
    
    async def test_authentication_middleware_integration(self, audit_service):
        """Test authentication audit middleware."""
        # Create mock ASGI app
        async def mock_app(scope, receive, send):
            # Simulate setting user in scope
            if not hasattr(scope, 'state'):
                scope['state'] = {}
            scope['state']['user'] = TestUser(
                id="test-user",
                email="user@company.local", 
                role="user"
            )
            
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": []
            })
            await send({
                "type": "http.response.body",
                "body": b"OK"
            })
        
        # Create auth middleware
        auth_middleware = AuthenticationAuditMiddleware(mock_app)
        auth_middleware.audit_service = audit_service
        
        with patch.object(audit_service, 'log_authentication_event') as mock_log:
            # Simulate request
            scope = {
                "type": "http",
                "method": "POST",
                "path": "/auth/login",
                "headers": [(b"user-agent", b"test-browser")]
            }
            
            await auth_middleware(scope, AsyncMock(), AsyncMock())
            
            # Verify authentication event was logged
            mock_log.assert_called_once()


class TestAuditAPIEndpoints:
    """Test audit API endpoints."""
    
    async def test_audit_search_endpoint(self, audit_client, audit_service, test_user):
        """Test audit log search API endpoint."""
        # Mock search results
        mock_events = [
            {
                "event_id": "event-1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": AuditEventType.USER_LOGIN.value,
                "user_id": "user-123",
                "severity": AuditSeverity.LOW.value,
                "outcome": "success"
            }
        ]
        
        with patch.object(audit_service, 'search_audit_events') as mock_search:
            mock_search.return_value = (mock_events, 1)
            
            # Make API request
            response = await audit_client.get(
                "/api/v1/audit/search",
                params={"user_id": "user-123", "limit": 10},
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "events" in data
            assert "total_count" in data
            assert data["total_count"] == 1
            assert len(data["events"]) == 1
    
    async def test_compliance_report_generation_endpoint(self, audit_client, audit_service, test_user):
        """Test compliance report generation API."""
        from services.comprehensive_audit_service import ComplianceReport
        
        # Mock compliance report
        mock_report = ComplianceReport(
            report_id="report-123",
            report_type=ComplianceStandard.SOC2_TYPE2,
            period_start=datetime.now(timezone.utc) - timedelta(days=30),
            period_end=datetime.now(timezone.utc),
            generated_at=datetime.now(timezone.utc),
            compliance_score=95.5,
            total_events=100,
            critical_events=2,
            security_events=25,
            violations=[],
            sections={
                "access_controls": {"score": 98.0, "status": "compliant"},
                "data_protection": {"score": 93.0, "status": "compliant"}
            },
            recommendations=["Implement additional monitoring"]
        )
        
        with patch.object(audit_service, 'generate_compliance_report') as mock_generate:
            mock_generate.return_value = mock_report
            
            # Make API request
            response = await audit_client.post(
                "/api/v1/audit/reports/compliance",
                json={
                    "standard": "soc2_type2",
                    "period_start": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                    "period_end": datetime.now(timezone.utc).isoformat(),
                    "include_evidence": True,
                    "format": "json"
                },
                headers={"Authorization": "Bearer admin-token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["report_id"] == "report-123"
            assert data["compliance_score"] == 95.5
            assert data["total_events"] == 100
            assert data["status"] == "completed"
    
    async def test_audit_statistics_endpoint(self, audit_client, audit_service, test_user):
        """Test audit statistics API endpoint."""
        # Mock statistics data
        mock_stats = [
            {
                "event_id": f"event-{i}",
                "timestamp": (datetime.now(timezone.utc) - timedelta(hours=i)).isoformat(),
                "event_type": AuditEventType.USER_LOGIN.value if i % 2 == 0 else AuditEventType.DATA_READ.value,
                "severity": AuditSeverity.CRITICAL.value if i == 0 else AuditSeverity.LOW.value,
                "user_id": f"user-{i % 3}"
            }
            for i in range(10)
        ]
        
        with patch.object(audit_service, 'search_audit_events') as mock_search:
            # Configure different return values for different time ranges
            def search_side_effect(*args, **kwargs):
                start_time = kwargs.get('start_time')
                if start_time:
                    hours_ago = (datetime.now(timezone.utc) - start_time).total_seconds() / 3600
                    if hours_ago <= 24:
                        return (mock_stats[:5], 5)  # Last 24 hours
                    elif hours_ago <= 168:  # 7 days
                        return (mock_stats[:8], 8)
                return (mock_stats, 10)  # All time
            
            mock_search.side_effect = search_side_effect
            
            # Make API request
            response = await audit_client.get(
                "/api/v1/audit/stats",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "total_events" in data
            assert "events_last_24h" in data
            assert "events_last_7d" in data
            assert "critical_events_last_24h" in data
            assert "top_event_types" in data
            assert "compliance_scores" in data
    
    async def test_audit_export_endpoint(self, audit_client, audit_service, test_user):
        """Test audit data export API endpoint."""
        with patch.object(audit_service, 'export_audit_data') as mock_export:
            mock_export.return_value = "export_20240120_123456.json"
            
            # Make API request
            response = await audit_client.post(
                "/api/v1/audit/export",
                json={
                    "format": "json",
                    "start_time": (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),
                    "end_time": datetime.now(timezone.utc).isoformat(),
                    "encrypt": False,
                    "include_sensitive": False
                },
                headers={"Authorization": "Bearer admin-token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "export_id" in data
            assert "download_url" in data
            assert data["status"] == "completed"
            assert data["format"] == "json"
    
    async def test_audit_health_check_endpoint(self, audit_client, audit_service):
        """Test audit system health check endpoint."""
        with patch.object(audit_service, 'health_check') as mock_health:
            mock_health.return_value = {
                "status": "healthy",
                "initialized": True,
                "redis_connected": True,
                "buffer_size": 0,
                "audit_logging_functional": True,
                "storage_available": True
            }
            
            # Make API request (no auth required for health check)
            response = await audit_client.get("/api/v1/audit/health")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "healthy"
            assert "components" in data
            assert "metrics" in data
            assert data["components"]["audit_service"]["status"] == "healthy"


class TestAuditSystemIntegration:
    """Test end-to-end audit system integration."""
    
    async def test_complete_audit_workflow(self, audit_app, audit_client, audit_service):
        """Test complete audit workflow from request to compliance report."""
        with patch.object(audit_service, 'log_audit_event') as mock_log, \
             patch.object(audit_service, 'search_audit_events') as mock_search, \
             patch.object(audit_service, 'generate_compliance_report') as mock_report:
            
            # Step 1: Make authenticated requests that generate audit events
            await audit_client.post("/api/v1/auth/login", json={
                "email": "user@company.local",
                "password": "testpass"
            })
            
            await audit_client.get("/api/v1/prd/123")
            await audit_client.post("/api/v1/prd", json={"title": "Test PRD"})
            
            # Verify audit events were logged
            assert mock_log.call_count >= 3
            
            # Step 2: Search for audit events
            mock_search.return_value = ([
                {
                    "event_id": "event-1",
                    "event_type": AuditEventType.USER_LOGIN.value,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                {
                    "event_id": "event-2",
                    "event_type": AuditEventType.DATA_READ.value,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ], 2)
            
            search_response = await audit_client.get(
                "/api/v1/audit/search",
                params={"event_types": ["user_login", "data_read"]},
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert search_response.status_code == 200
            search_data = search_response.json()
            assert len(search_data["events"]) == 2
            
            # Step 3: Generate compliance report
            from services.comprehensive_audit_service import ComplianceReport
            
            mock_report.return_value = ComplianceReport(
                report_id="integration-test-report",
                report_type=ComplianceStandard.SOC2_TYPE2,
                period_start=datetime.now(timezone.utc) - timedelta(days=30),
                period_end=datetime.now(timezone.utc),
                generated_at=datetime.now(timezone.utc),
                compliance_score=96.0,
                total_events=150,
                critical_events=1,
                security_events=45,
                violations=[],
                sections={
                    "security": {"score": 98.0, "status": "compliant"},
                    "availability": {"score": 94.0, "status": "compliant"}
                },
                recommendations=[]
            )
            
            report_response = await audit_client.post(
                "/api/v1/audit/reports/compliance",
                json={
                    "standard": "soc2_type2",
                    "period_start": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                    "period_end": datetime.now(timezone.utc).isoformat()
                },
                headers={"Authorization": "Bearer admin-token"}
            )
            
            assert report_response.status_code == 200
            report_data = report_response.json()
            assert report_data["compliance_score"] == 96.0
            assert report_data["total_events"] == 150
    
    async def test_audit_system_performance_under_load(self, audit_service, mock_redis):
        """Test audit system performance under high load."""
        with patch('services.comprehensive_audit_service.get_redis_client', return_value=mock_redis):
            # Simulate high-volume audit event generation
            tasks = []
            for i in range(100):
                task = audit_service.log_audit_event(
                    event_type=AuditEventType.DATA_READ,
                    user_id=f"user-{i % 10}",
                    resource_type="prd",
                    resource_id=f"prd-{i}",
                    outcome="success"
                )
                tasks.append(task)
            
            # Execute all tasks concurrently
            start_time = datetime.now()
            event_ids = await asyncio.gather(*tasks)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Verify all events were logged
            assert len(event_ids) == 100
            assert all(event_id is not None for event_id in event_ids)
            
            # Check performance - should handle 100 events in reasonable time
            assert duration < 5.0  # Less than 5 seconds for 100 events
            
            # Verify Redis was called for each event
            assert mock_redis.setex.call_count == 100
    
    async def test_audit_system_error_handling(self, audit_service):
        """Test audit system handles errors gracefully."""
        # Test with failing Redis connection
        with patch('services.comprehensive_audit_service.get_redis_client') as mock_get_redis:
            mock_get_redis.side_effect = Exception("Redis connection failed")
            
            # Service should handle Redis failures gracefully
            try:
                event_id = await audit_service.log_audit_event(
                    event_type=AuditEventType.USER_LOGIN,
                    user_id="test-user",
                    outcome="success"
                )
                # Should return None or handle gracefully, not raise exception
                assert event_id is None or isinstance(event_id, str)
            except Exception as e:
                pytest.fail(f"Audit service should handle Redis failures gracefully: {e}")
    
    async def test_audit_data_integrity(self, audit_service, mock_redis):
        """Test audit data integrity and consistency."""
        with patch('services.comprehensive_audit_service.get_redis_client', return_value=mock_redis):
            # Create audit event with comprehensive data
            original_data = {
                "event_type": AuditEventType.DATA_UPDATED,
                "user_id": "integrity-test-user",
                "user_email=f"testuser{str(uuid.uuid4())[:8]}@{self.fake.domain_name()}",",
                "resource_type": "prd",
                "resource_id": "prd-integrity-test",
                "action": "update_requirements",
                "outcome": "success",
                "metadata": {
                    "fields_modified": ["title", "description"],
                    "data_sensitivity": "confidential",
                    "business_impact": "high"
                }
            }
            
            event_id = await audit_service.log_audit_event(**original_data)
            
            # Verify data was stored correctly
            mock_redis.setex.assert_called_once()
            stored_data = json.loads(mock_redis.setex.call_args[0][2])
            
            # Check all original data is preserved
            assert stored_data["event_type"] == original_data["event_type"].value
            assert stored_data["user_id"] == original_data["user_id"]
            assert stored_data["user_email"] == original_data["user_email"]
            assert stored_data["resource_type"] == original_data["resource_type"]
            assert stored_data["metadata"] == original_data["metadata"]
            
            # Check additional fields were added
            assert "event_id" in stored_data
            assert "timestamp" in stored_data
            assert "compliance_tags" in stored_data
            assert stored_data["sensitive_data"] is True  # Should be marked as sensitive


@pytest.mark.asyncio
class TestAuditSystemCleanup:
    """Test audit system cleanup and resource management."""
    
    async def test_service_cleanup(self, audit_service):
        """Test proper service cleanup."""
        # Initialize service
        await audit_service.initialize()
        assert audit_service.is_initialized
        
        # Cleanup service
        await audit_service.cleanup()
        
        # Verify cleanup
        assert not audit_service.is_initialized
        
        # Should handle multiple cleanup calls gracefully
        await audit_service.cleanup()
    
    async def test_batch_processing_cleanup(self, audit_service, mock_redis):
        """Test batch processing system cleanup."""
        with patch('services.comprehensive_audit_service.get_redis_client', return_value=mock_redis):
            # Add events to buffer
            for i in range(10):
                await audit_service.log_audit_event(
                    event_type=AuditEventType.DATA_READ,
                    user_id=f"user-{i}",
                    outcome="success"
                )
            
            # Verify events are in buffer
            assert len(audit_service.event_buffer) > 0
            
            # Cleanup should flush buffer
            await audit_service.cleanup()
            
            # Verify buffer was flushed
            assert len(audit_service.event_buffer) == 0
            assert mock_redis.setex.call_count >= 10


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])