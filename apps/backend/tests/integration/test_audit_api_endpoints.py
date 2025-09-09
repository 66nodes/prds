"""
Audit API Endpoints Integration Tests

Comprehensive tests for all audit-related API endpoints including
search, compliance reporting, data export, and administrative functions.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient
import structlog

from api.endpoints.audit import router as audit_router
from api.dependencies.auth import get_current_user, require_admin, require_audit_access
from services.comprehensive_audit_service import (
    ComprehensiveAuditService,
    AuditEventType,
    AuditSeverity,
    ComplianceStandard,
    ComplianceReport,
    get_comprehensive_audit_service
)
from tests.conftest import TestUser


logger = structlog.get_logger(__name__)


@pytest.fixture
async def mock_audit_service():
    """Mock comprehensive audit service."""
    service = AsyncMock(spec=ComprehensiveAuditService)
    
    # Mock common methods
    service.search_audit_events = AsyncMock()
    service.generate_compliance_report = AsyncMock()
    service.export_audit_data = AsyncMock()
    service.apply_retention_policy = AsyncMock()
    service.health_check = AsyncMock()
    service.log_audit_event = AsyncMock(return_value="mock-event-id")
    
    return service


@pytest.fixture
def mock_admin_user():
    """Mock admin user for testing."""
    return TestUser(
        id="admin-user-123",
        email="admin@strategic-planning.ai",
        role="admin",
        is_active=True
    )


@pytest.fixture
def mock_audit_user():
    """Mock user with audit access."""
    return TestUser(
        id="audit-user-456",
        email="auditorexample.com",
        role="auditor",
        is_active=True
    )


@pytest.fixture
def mock_regular_user():
    """Mock regular user without special permissions."""
    return TestUser(
        id="regular-user-789",
        email="userexample.com",
        role="user",
        is_active=True
    )


@pytest.fixture
def audit_api_app(mock_audit_service):
    """Create FastAPI app with audit endpoints."""
    app = FastAPI(title="Audit API Test App")
    
    # Add audit router
    app.include_router(audit_router)
    
    # Override dependencies
    app.dependency_overrides[get_comprehensive_audit_service] = lambda: mock_audit_service
    
    return app


@pytest.fixture
async def audit_api_client(audit_api_app):
    """Create async test client for audit API."""
    async with AsyncClient(app=audit_api_app, base_url="http://test") as client:
        yield client


def create_mock_auth_dependency(user):
    """Create mock authentication dependency."""
    return lambda: user


class TestAuditSearchEndpoints:
    """Test audit search API endpoints."""
    
    async def test_search_audit_logs_success(self, audit_api_app, audit_api_client, mock_audit_service, mock_audit_user):
        """Test successful audit log search."""
        # Override auth dependency
        audit_api_app.dependency_overrides[require_audit_access] = create_mock_auth_dependency(mock_audit_user)
        
        # Mock search results
        mock_events = [
            {
                "event_id": "event-1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": AuditEventType.USER_LOGIN.value,
                "severity": AuditSeverity.MEDIUM.value,
                "user_id": "test-user-1",
                "user_email=f"testuser{str(uuid.uuid4())[:8]}@{self.fake.domain_name()}",",
                "ip_address": "192.168.1.1",
                "outcome": "success",
                "compliance_tags": ["soc2"],
                "metadata": {"browser": "Chrome"}
            },
            {
                "event_id": "event-2",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": AuditEventType.DATA_READ.value,
                "severity": AuditSeverity.LOW.value,
                "user_id": "test-user-2",
                "resource_type": "prd",
                "resource_id": "prd-123",
                "outcome": "success",
                "compliance_tags": ["gdpr"],
                "metadata": {}
            }
        ]
        
        mock_audit_service.search_audit_events.return_value = (mock_events, 2)
        
        # Make API request
        response = await audit_api_client.get(
            "/api/v1/audit/search",
            params={
                "event_types": ["user_login", "data_read"],
                "start_time": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
                "end_time": datetime.now(timezone.utc).isoformat(),
                "limit": 10,
                "offset": 0
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "events" in data
        assert "total_count" in data
        assert "page_info" in data
        assert "search_metadata" in data
        
        # Verify data
        assert data["total_count"] == 2
        assert len(data["events"]) == 2
        
        # Verify event structure
        event = data["events"][0]
        assert "event_id" in event
        assert "timestamp" in event
        assert "event_type" in event
        assert "severity" in event
        
        # Verify search was called with correct parameters
        mock_audit_service.search_audit_events.assert_called_once()
        call_args = mock_audit_service.search_audit_events.call_args
        assert call_args[1]["limit"] == 10
        assert call_args[1]["offset"] == 0
    
    async def test_search_audit_logs_with_filters(self, audit_api_app, audit_api_client, mock_audit_service, mock_audit_user):
        """Test audit log search with various filters."""
        audit_api_app.dependency_overrides[require_audit_access] = create_mock_auth_dependency(mock_audit_user)
        
        mock_audit_service.search_audit_events.return_value = ([], 0)
        
        # Test with comprehensive filters
        response = await audit_api_client.get(
            "/api/v1/audit/search",
            params={
                "user_id": "specific-user",
                "user_email": "user@company.local",
                "resource_type": "prd",
                "resource_id": "prd-123",
                "severity": "high",
                "outcome": "success",
                "compliance_tags": ["soc2", "gdpr"],
                "ip_address": "192.168.1.1",
                "limit": 50,
                "offset": 10
            }
        )
        
        assert response.status_code == 200
        
        # Verify all filters were passed to service
        call_args = mock_audit_service.search_audit_events.call_args[1]
        assert call_args["user_id"] == "specific-user"
        assert call_args["resource_type"] == "prd"
        assert call_args["limit"] == 50
        assert call_args["offset"] == 10
    
    async def test_search_audit_logs_validation_errors(self, audit_api_app, audit_api_client, mock_audit_user):
        """Test validation errors in search parameters."""
        audit_api_app.dependency_overrides[require_audit_access] = create_mock_auth_dependency(mock_audit_user)
        
        # Test invalid event type
        response = await audit_api_client.get(
            "/api/v1/audit/search",
            params={
                "event_types": ["invalid_event_type"],
                "limit": 10
            }
        )
        assert response.status_code == 422  # Validation error
        
        # Test invalid severity
        response = await audit_api_client.get(
            "/api/v1/audit/search",
            params={
                "severity": "invalid_severity",
                "limit": 10
            }
        )
        assert response.status_code == 422
        
        # Test invalid limit
        response = await audit_api_client.get(
            "/api/v1/audit/search",
            params={
                "limit": 2000  # Exceeds maximum
            }
        )
        assert response.status_code == 422
    
    async def test_get_audit_event_details(self, audit_api_app, audit_api_client, mock_audit_service, mock_audit_user):
        """Test getting specific audit event details."""
        audit_api_app.dependency_overrides[require_audit_access] = create_mock_auth_dependency(mock_audit_user)
        
        # Mock specific event
        mock_event = {
            "event_id": "specific-event-123",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": AuditEventType.DATA_UPDATED.value,
            "user_id": "user-123",
            "resource_type": "prd",
            "resource_id": "prd-456",
            "outcome": "success",
            "metadata": {"fields_updated": ["title", "description"]}
        }
        
        mock_audit_service.search_audit_events.return_value = ([mock_event], 1)
        
        # Make API request
        response = await audit_api_client.get("/api/v1/audit/events/specific-event-123")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["event_id"] == "specific-event-123"
        assert data["event_type"] == AuditEventType.DATA_UPDATED.value
        assert data["user_id"] == "user-123"
    
    async def test_get_audit_event_not_found(self, audit_api_app, audit_api_client, mock_audit_service, mock_audit_user):
        """Test getting non-existent audit event."""
        audit_api_app.dependency_overrides[require_audit_access] = create_mock_auth_dependency(mock_audit_user)
        
        # Mock empty search results
        mock_audit_service.search_audit_events.return_value = ([], 0)
        
        response = await audit_api_client.get("/api/v1/audit/events/nonexistent-event")
        
        assert response.status_code == 404
    
    async def test_get_audit_statistics(self, audit_api_app, audit_api_client, mock_audit_service, mock_audit_user):
        """Test audit statistics endpoint."""
        audit_api_app.dependency_overrides[require_audit_access] = create_mock_auth_dependency(mock_audit_user)
        
        # Mock statistics data
        def mock_search_stats(*args, **kwargs):
            start_time = kwargs.get('start_time')
            if start_time:
                hours_ago = (datetime.now(timezone.utc) - start_time).total_seconds() / 3600
                if hours_ago <= 24:
                    return ([{"event_type": "user_login", "severity": "critical"}] * 5, 5)
                elif hours_ago <= 168:
                    return ([{"event_type": "data_read", "user_id": f"user-{i%3}"}] * 8, 8)
            return ([{"event_type": "user_login"}] * 10, 10)
        
        mock_audit_service.search_audit_events.side_effect = mock_search_stats
        
        response = await audit_api_client.get("/api/v1/audit/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify statistics structure
        assert "total_events" in data
        assert "events_last_24h" in data
        assert "events_last_7d" in data
        assert "critical_events_last_24h" in data
        assert "top_event_types" in data
        assert "top_users" in data
        assert "compliance_scores" in data
        assert "security_alerts" in data
        
        # Verify data values
        assert data["total_events"] == 10
        assert data["events_last_24h"] == 5
        assert data["events_last_7d"] == 8


class TestComplianceReportingEndpoints:
    """Test compliance reporting API endpoints."""
    
    async def test_generate_compliance_report(self, audit_api_app, audit_api_client, mock_audit_service, mock_admin_user):
        """Test compliance report generation."""
        audit_api_app.dependency_overrides[require_admin] = create_mock_auth_dependency(mock_admin_user)
        
        # Mock compliance report
        mock_report = ComplianceReport(
            report_id="compliance-report-123",
            report_type=ComplianceStandard.SOC2_TYPE2,
            period_start=datetime.now(timezone.utc) - timedelta(days=30),
            period_end=datetime.now(timezone.utc),
            generated_at=datetime.now(timezone.utc),
            compliance_score=96.5,
            total_events=1500,
            critical_events=3,
            security_events=125,
            violations=[],
            sections={
                "access_controls": {
                    "score": 98.5,
                    "status": "compliant",
                    "events_reviewed": 500,
                    "violations": []
                },
                "data_protection": {
                    "score": 94.5,
                    "status": "compliant",
                    "events_reviewed": 800,
                    "violations": []
                }
            },
            recommendations=["Implement additional failed login attempt monitoring"]
        )
        
        mock_audit_service.generate_compliance_report.return_value = mock_report
        
        # Make API request
        response = await audit_api_client.post(
            "/api/v1/audit/reports/compliance",
            json={
                "standard": "soc2_type2",
                "period_start": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                "period_end": datetime.now(timezone.utc).isoformat(),
                "include_evidence": True,
                "format": "json"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert data["report_id"] == "compliance-report-123"
        assert data["report_type"] == "soc2_type2"
        assert data["compliance_score"] == 96.5
        assert data["total_events"] == 1500
        assert data["critical_events"] == 3
        assert data["security_events"] == 125
        assert data["violations_count"] == 0
        assert "download_url" in data
        assert data["status"] == "completed"
        
        # Verify service was called correctly
        mock_audit_service.generate_compliance_report.assert_called_once()
        call_args = mock_audit_service.generate_compliance_report.call_args[1]
        assert call_args["standard"] == ComplianceStandard.SOC2_TYPE2
        assert call_args["include_evidence"] is True
    
    async def test_generate_compliance_report_validation_errors(self, audit_api_app, audit_api_client, mock_admin_user):
        """Test validation errors in compliance report generation."""
        audit_api_app.dependency_overrides[require_admin] = create_mock_auth_dependency(mock_admin_user)
        
        # Test invalid compliance standard
        response = await audit_api_client.post(
            "/api/v1/audit/reports/compliance",
            json={
                "standard": "invalid_standard",
                "period_start": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                "period_end": datetime.now(timezone.utc).isoformat()
            }
        )
        assert response.status_code == 422
        
        # Test invalid date range (end before start)
        response = await audit_api_client.post(
            "/api/v1/audit/reports/compliance",
            json={
                "standard": "soc2_type2",
                "period_start": datetime.now(timezone.utc).isoformat(),
                "period_end": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
            }
        )
        assert response.status_code == 422
        
        # Test period too large
        response = await audit_api_client.post(
            "/api/v1/audit/reports/compliance",
            json={
                "standard": "soc2_type2",
                "period_start": (datetime.now(timezone.utc) - timedelta(days=400)).isoformat(),
                "period_end": datetime.now(timezone.utc).isoformat()
            }
        )
        assert response.status_code == 400
    
    async def test_get_compliance_report_details(self, audit_api_app, audit_api_client, mock_audit_service, mock_admin_user):
        """Test getting compliance report details."""
        audit_api_app.dependency_overrides[require_admin] = create_mock_auth_dependency(mock_admin_user)
        
        response = await audit_api_client.get("/api/v1/audit/reports/test-report-123")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify report structure (mocked data)
        assert "report_id" in data
        assert "compliance_score" in data
        assert "sections" in data
        assert "violations" in data
        assert "recommendations" in data


class TestAuditDataExportEndpoints:
    """Test audit data export API endpoints."""
    
    async def test_export_audit_data_json(self, audit_api_app, audit_api_client, mock_audit_service, mock_admin_user):
        """Test audit data export in JSON format."""
        audit_api_app.dependency_overrides[require_admin] = create_mock_auth_dependency(mock_admin_user)
        
        # Mock export
        mock_audit_service.export_audit_data.return_value = "audit_export_20240120_123456.json"
        
        response = await audit_api_client.post(
            "/api/v1/audit/export",
            json={
                "format": "json",
                "start_time": (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),
                "end_time": datetime.now(timezone.utc).isoformat(),
                "event_types": ["user_login", "data_read"],
                "user_id": "specific-user",
                "encrypt": False,
                "include_sensitive": False
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "export_id" in data
        assert "download_url" in data
        assert data["status"] == "completed"
        assert data["format"] == "json"
        assert data["encrypted"] is False
        
        # Verify service was called correctly
        mock_audit_service.export_audit_data.assert_called_once()
        call_args = mock_audit_service.export_audit_data.call_args[1]
        assert call_args["export_format"] == "json"
        assert call_args["user_id"] == "specific-user"
    
    async def test_export_audit_data_csv(self, audit_api_app, audit_api_client, mock_audit_service, mock_admin_user):
        """Test audit data export in CSV format."""
        audit_api_app.dependency_overrides[require_admin] = create_mock_auth_dependency(mock_admin_user)
        
        mock_audit_service.export_audit_data.return_value = "audit_export_20240120_123456.csv"
        
        response = await audit_api_client.post(
            "/api/v1/audit/export",
            json={
                "format": "csv",
                "start_time": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                "end_time": datetime.now(timezone.utc).isoformat(),
                "encrypt": True,
                "include_sensitive": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["format"] == "csv"
        assert data["encrypted"] is True
    
    async def test_export_audit_data_validation(self, audit_api_app, audit_api_client, mock_admin_user):
        """Test export validation errors."""
        audit_api_app.dependency_overrides[require_admin] = create_mock_auth_dependency(mock_admin_user)
        
        # Test period too large
        response = await audit_api_client.post(
            "/api/v1/audit/export",
            json={
                "format": "json",
                "start_time": (datetime.now(timezone.utc) - timedelta(days=800)).isoformat(),
                "end_time": datetime.now(timezone.utc).isoformat()
            }
        )
        assert response.status_code == 400


class TestAuditAdminEndpoints:
    """Test audit administration API endpoints."""
    
    async def test_apply_retention_policies(self, audit_api_app, audit_api_client, mock_audit_service, mock_admin_user):
        """Test manual retention policy application."""
        audit_api_app.dependency_overrides[require_admin] = create_mock_auth_dependency(mock_admin_user)
        
        # Mock retention summary
        mock_audit_service.apply_retention_policy.return_value = {
            "events_deleted": 150,
            "events_retained": 2000,
            "deleted_event_types": {
                "user_login": 50,
                "data_read": 100
            },
            "retention_period_days": 90
        }
        
        response = await audit_api_client.post("/api/v1/audit/retention/apply")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "completed"
        assert "retention_summary" in data
        assert "applied_at" in data
        
        retention_summary = data["retention_summary"]
        assert retention_summary["events_deleted"] == 150
        assert retention_summary["events_retained"] == 2000
    
    async def test_audit_health_check(self, audit_api_app, audit_api_client, mock_audit_service):
        """Test audit system health check."""
        # Mock health status
        mock_audit_service.health_check.return_value = {
            "status": "healthy",
            "initialized": True,
            "redis_connected": True,
            "buffer_size": 5,
            "audit_logging_functional": True,
            "storage_available": True
        }
        
        response = await audit_api_client.get("/api/v1/audit/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "components" in data
        assert "metrics" in data
        assert "timestamp" in data
        
        # Verify component health
        components = data["components"]
        assert components["audit_service"]["status"] == "healthy"
        assert components["storage"]["status"] == "healthy"
        assert components["buffer"]["status"] == "healthy"
    
    async def test_audit_health_check_unhealthy(self, audit_api_app, audit_api_client, mock_audit_service):
        """Test audit health check when system is unhealthy."""
        # Mock unhealthy status
        mock_audit_service.health_check.side_effect = Exception("Health check failed")
        
        response = await audit_api_client.get("/api/v1/audit/health")
        
        assert response.status_code == 200  # Should still return 200 with error info
        data = response.json()
        
        assert data["status"] == "unhealthy"
        assert "error" in data


class TestAuditStreamingEndpoints:
    """Test audit event streaming endpoints."""
    
    async def test_stream_audit_events(self, audit_api_app, audit_api_client, mock_audit_service, mock_audit_user):
        """Test real-time audit event streaming."""
        audit_api_app.dependency_overrides[require_audit_access] = create_mock_auth_dependency(mock_audit_user)
        
        # Mock recent events
        mock_events = [
            {
                "event_id": "stream-event-1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "user_login",
                "user_id": "streaming-user"
            }
        ]
        
        mock_audit_service.search_audit_events.return_value = (mock_events, 1)
        
        # Note: Testing SSE streaming is complex in async tests
        # This test verifies the endpoint exists and handles auth
        async with audit_api_client.stream("GET", "/api/v1/audit/events/stream") as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/plain; charset=utf-8"


class TestAuditAPIPermissions:
    """Test audit API permission handling."""
    
    async def test_search_requires_audit_access(self, audit_api_app, audit_api_client, mock_regular_user):
        """Test search endpoint requires audit access."""
        # Override with regular user (no audit access)
        audit_api_app.dependency_overrides[require_audit_access] = lambda: mock_regular_user
        
        with patch('api.dependencies.auth.require_audit_access') as mock_require:
            mock_require.side_effect = HTTPException(status_code=403, detail="Insufficient permissions")
            
            response = await audit_api_client.get("/api/v1/audit/search")
            assert response.status_code == 403
    
    async def test_compliance_reports_require_admin(self, audit_api_app, audit_api_client, mock_regular_user):
        """Test compliance report endpoints require admin access."""
        audit_api_app.dependency_overrides[require_admin] = lambda: mock_regular_user
        
        with patch('api.dependencies.auth.require_admin') as mock_require:
            mock_require.side_effect = HTTPException(status_code=403, detail="Admin access required")
            
            response = await audit_api_client.post(
                "/api/v1/audit/reports/compliance",
                json={
                    "standard": "soc2_type2",
                    "period_start": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                    "period_end": datetime.now(timezone.utc).isoformat()
                }
            )
            assert response.status_code == 403
    
    async def test_export_requires_admin(self, audit_api_app, audit_api_client, mock_regular_user):
        """Test export endpoint requires admin access."""
        audit_api_app.dependency_overrides[require_admin] = lambda: mock_regular_user
        
        with patch('api.dependencies.auth.require_admin') as mock_require:
            mock_require.side_effect = HTTPException(status_code=403, detail="Admin access required")
            
            response = await audit_api_client.post(
                "/api/v1/audit/export",
                json={"format": "json"}
            )
            assert response.status_code == 403


if __name__ == "__main__":
    pytest.main([__file__, "-v"])