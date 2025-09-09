"""
Security tests for authentication and authorization systems.

These tests verify security controls, vulnerability prevention,
and compliance with security best practices.
"""
import uuid
from tests.utilities.test_data_factory import test_data_factory

import pytest
import jwt
import hashlib
import time
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Any

from backend.services.auth_service import AuthService
from backend.services.security_service import SecurityService
from backend.core.security import (
    hash_password,
    verify_password,
    generate_token,
    verify_token
)
from backend.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    SecurityViolationError
)


class TestAuthenticationSecurity:
    """Test suite for authentication security."""

    @pytest.fixture
    def auth_service(self):
        """Create authentication service."""
        return AuthService()

    @pytest.fixture
    def security_service(self):
        """Create security service."""
        return SecurityService()

    @pytest.fixture
    def valid_user_data(self):
        """Valid user registration data."""
        return {
            "email": "security.user@company.local",
            "password": "SecurePassword123!",
            "name": "Security Test User"
        }

    @pytest.fixture
    def attack_payloads(self):
        """Common attack payloads for security testing."""
        return {
            "sql_injection": [
                "' OR '1'='1",
                "'; DROP TABLE users; --",
                "' UNION SELECT * FROM users --",
                "admin'--",
                "1' OR 1=1#"
            ],
            "xss_payloads": [
                "<script>alert('XSS')</script>",
                "javascript:alert('XSS')",
                "<img src=x onerror=alert('XSS')>",
                "<svg onload=alert('XSS')>",
                "'\"><script>alert('XSS')</script>"
            ],
            "ldap_injection": [
                "*)(&",
                "*))%00",
                "*(|(password=*))",
                "admin*)((|password=*)"
            ],
            "nosql_injection": [
                "'; return true; var dummy='",
                "' || 1==1//",
                "{'$ne': null}",
                "{'$regex': '.*'}"
            ]
        }

    async def test_password_hashing_security(self, auth_service):
        """Test password hashing security measures."""
        password = "TestPassword123!"
        
        # Test password hashing
        hashed1 = await auth_service.hash_password(password)
        hashed2 = await auth_service.hash_password(password)
        
        # Hashes should be different (salt randomization)
        assert hashed1 != hashed2
        
        # Both should verify correctly
        assert await auth_service.verify_password(password, hashed1)
        assert await auth_service.verify_password(password, hashed2)
        
        # Should use strong hashing algorithm (bcrypt/scrypt/argon2)
        assert len(hashed1) > 50  # Minimum length for secure hash
        assert '$' in hashed1  # Should contain algorithm identifier

    async def test_password_complexity_requirements(self, auth_service):
        """Test password complexity validation."""
        weak_passwords = [
            "123456",
            "password",
            "qwerty",
            "abc123",
            "Password",  # No numbers/special chars
            "password123",  # No uppercase/special chars
            "PASSWORD123!",  # No lowercase
            "Pass!1"  # Too short
        ]
        
        for weak_password in weak_passwords:
            with pytest.raises(SecurityViolationError) as exc_info:
                await auth_service.validate_password_strength(weak_password)
            
            assert "password strength" in str(exc_info.value).lower()

        # Strong password should pass
        strong_password = "SecurePassword123!"
        result = await auth_service.validate_password_strength(strong_password)
        assert result.is_strong is True

    async def test_jwt_token_security(self, auth_service):
        """Test JWT token security implementation."""
        user_id = "test_user_123"
        
        # Generate token
        token = await auth_service.generate_access_token(user_id)
        
        # Verify token structure
        assert isinstance(token, str)
        assert len(token.split('.')) == 3  # Header.Payload.Signature
        
        # Decode and verify token
        decoded = await auth_service.verify_token(token)
        assert decoded["user_id"] == user_id
        assert "exp" in decoded  # Expiration claim
        assert "iat" in decoded  # Issued at claim
        
        # Token should have reasonable expiration (not too long)
        exp_time = datetime.fromtimestamp(decoded["exp"])
        iat_time = datetime.fromtimestamp(decoded["iat"])
        token_lifetime = exp_time - iat_time
        
        assert token_lifetime <= timedelta(hours=24)  # Max 24 hours

    async def test_token_expiration_handling(self, auth_service):
        """Test handling of expired tokens."""
        user_id = "test_user_123"
        
        # Create expired token (mock)
        with patch('backend.core.security.datetime') as mock_datetime:
            # Mock current time to be in the past for token creation
            mock_datetime.utcnow.return_value = datetime.utcnow() - timedelta(hours=25)
            expired_token = await auth_service.generate_access_token(user_id)
        
        # Verify expired token is rejected
        with pytest.raises(AuthenticationError) as exc_info:
            await auth_service.verify_token(expired_token)
        
        assert "expired" in str(exc_info.value).lower()

    async def test_token_tampering_detection(self, auth_service):
        """Test detection of tampered JWT tokens."""
        user_id = "test_user_123"
        token = await auth_service.generate_access_token(user_id)
        
        # Tamper with token by changing a character
        tampered_token = token[:-5] + "XXXXX"
        
        with pytest.raises(AuthenticationError) as exc_info:
            await auth_service.verify_token(tampered_token)
        
        assert "invalid" in str(exc_info.value).lower() or "signature" in str(exc_info.value).lower()

    async def test_sql_injection_prevention(self, auth_service, attack_payloads):
        """Test SQL injection prevention in authentication."""
        for payload in attack_payloads["sql_injection"]:
            # Test login with malicious payload
            with pytest.raises((AuthenticationError, SecurityViolationError)):
                await auth_service.authenticate_user(payload, "any_password")
            
            # Test registration with malicious payload
            with pytest.raises((SecurityViolationError, ValueError)):
                await auth_service.register_user({
                    "email": payload,
                    "password": "TestPassword123!",
                    "name": "Test User"
                })

    async def test_xss_prevention_in_auth(self, auth_service, attack_payloads):
        """Test XSS prevention in authentication inputs."""
        for payload in attack_payloads["xss_payloads"]:
            # XSS payloads should be sanitized or rejected
            with pytest.raises((SecurityViolationError, ValueError)):
                await auth_service.register_user({
                    "email": "user@company.local",
                    "password": "TestPassword123!",
                    "name": payload  # XSS in name field
                })

    async def test_brute_force_protection(self, auth_service, security_service):
        """Test brute force attack prevention."""
        email = "victimexample.com"
        wrong_password = "WrongPassword123!"
        
        # Simulate multiple failed login attempts
        for i in range(6):  # Exceed typical rate limit
            try:
                await auth_service.authenticate_user(email, wrong_password)
            except AuthenticationError:
                pass  # Expected failure
        
        # Account should be temporarily locked
        with pytest.raises(SecurityViolationError) as exc_info:
            await auth_service.authenticate_user(email, "ActualPassword123!")
        
        assert "locked" in str(exc_info.value).lower() or "rate limit" in str(exc_info.value).lower()

    async def test_session_security(self, auth_service):
        """Test session security measures."""
        user_id = "test_user_123"
        
        # Create session
        session = await auth_service.create_session(user_id)
        
        # Session should have security attributes
        assert session.get("secure") is True  # Secure cookie flag
        assert session.get("http_only") is True  # HttpOnly flag
        assert session.get("same_site") == "strict"  # SameSite protection
        assert "csrf_token" in session  # CSRF protection token
        
        # Session ID should be cryptographically random
        session_id = session["session_id"]
        assert len(session_id) >= 32  # Minimum entropy
        
        # Multiple sessions should have different IDs
        session2 = await auth_service.create_session(user_id)
        assert session["session_id"] != session2["session_id"]

    async def test_csrf_protection(self, security_service):
        """Test CSRF token generation and validation."""
        session_id = "test_session_123"
        
        # Generate CSRF token
        csrf_token = await security_service.generate_csrf_token(session_id)
        
        # Token should be non-empty and random
        assert csrf_token is not None
        assert len(csrf_token) >= 32
        
        # Validate token
        is_valid = await security_service.validate_csrf_token(csrf_token, session_id)
        assert is_valid is True
        
        # Invalid token should fail
        invalid_token = "invalid_csrf_token_12345"
        is_valid_invalid = await security_service.validate_csrf_token(invalid_token, session_id)
        assert is_valid_invalid is False

    async def test_input_sanitization(self, auth_service):
        """Test input sanitization and validation."""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "../../../etc/passwd",
            "'; DROP TABLE users; --",
            "{{7*7}}",  # Template injection
            "${7*7}",   # Expression injection
            "\\x00\\x01\\x02"  # Null bytes
        ]
        
        for malicious_input in malicious_inputs:
            # Should sanitize or reject malicious input
            sanitized = await auth_service.sanitize_input(malicious_input)
            
            # Sanitized input should not contain original payload
            assert sanitized != malicious_input
            assert "<script>" not in sanitized
            assert "DROP TABLE" not in sanitized

    async def test_authorization_bypass_prevention(self, auth_service):
        """Test prevention of authorization bypass attacks."""
        regular_user_id = "regular_user_123"
        admin_user_id = "admin_user_456"
        
        # Create tokens
        regular_token = await auth_service.generate_access_token(regular_user_id)
        admin_token = await auth_service.generate_access_token(admin_user_id, role="admin")
        
        # Test role escalation prevention
        with patch.object(auth_service, 'get_user_role') as mock_get_role:
            mock_get_role.return_value = "user"  # Regular user role
            
            # Regular user should not access admin resources
            with pytest.raises(AuthorizationError):
                await auth_service.authorize_admin_action(regular_token)
            
            # Admin user should access admin resources
            mock_get_role.return_value = "admin"
            result = await auth_service.authorize_admin_action(admin_token)
            assert result is True

    async def test_timing_attack_prevention(self, auth_service):
        """Test prevention of timing attacks."""
        import time
        
        existing_email = "existingexample.com" 
        non_existing_email = "nonexistingexample.com"
        password = "TestPassword123!"
        
        # Measure response times for existing vs non-existing users
        start_time = time.time()
        try:
            await auth_service.authenticate_user(existing_email, password)
        except AuthenticationError:
            pass
        existing_time = time.time() - start_time
        
        start_time = time.time()
        try:
            await auth_service.authenticate_user(non_existing_email, password)
        except AuthenticationError:
            pass
        non_existing_time = time.time() - start_time
        
        # Response times should be similar (within 100ms difference)
        time_diff = abs(existing_time - non_existing_time)
        assert time_diff < 0.1  # Less than 100ms difference

    async def test_secure_token_storage(self, auth_service):
        """Test secure token storage practices."""
        user_id = "test_user_123"
        
        # Generate refresh token
        refresh_token = await auth_service.generate_refresh_token(user_id)
        
        # Refresh token should be hashed when stored
        stored_token_hash = await auth_service.get_stored_refresh_token(user_id)
        
        # Stored version should be hashed, not plain text
        assert stored_token_hash != refresh_token
        assert len(stored_token_hash) > 50  # Hash length
        
        # Should be able to verify token against hash
        is_valid = await auth_service.verify_refresh_token(refresh_token, user_id)
        assert is_valid is True


class TestAPISecurityHeaders:
    """Test API security headers and protections."""

    @pytest.fixture
    def security_headers_service(self):
        """Create security headers service."""
        from backend.middleware.security_headers import SecurityHeaders
        return SecurityHeaders()

    async def test_security_headers_present(self, security_headers_service):
        """Test that security headers are present."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        
        app = FastAPI()
        app.add_middleware(security_headers_service)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        client = TestClient(app)
        response = client.get("/test")
        
        # Check security headers
        headers = response.headers
        
        assert "X-Content-Type-Options" in headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        
        assert "X-Frame-Options" in headers
        assert headers["X-Frame-Options"] == "DENY"
        
        assert "X-XSS-Protection" in headers
        assert headers["X-XSS-Protection"] == "1; mode=block"
        
        assert "Strict-Transport-Security" in headers
        assert "max-age" in headers["Strict-Transport-Security"]
        
        assert "Content-Security-Policy" in headers
        assert "default-src" in headers["Content-Security-Policy"]

    async def test_cors_configuration(self, security_headers_service):
        """Test CORS configuration security."""
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.testclient import TestClient
        
        app = FastAPI()
        
        # Add CORS with security configuration
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["https://trusted-domain.com"],  # Specific origins only
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        client = TestClient(app)
        
        # Test preflight request
        response = client.options(
            "/test",
            headers={
                "Origin": "https://trusted-domain.com",
                "Access-Control-Request-Method": "GET"
            }
        )
        
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers
        
        # Test blocked origin
        response = client.options(
            "/test",
            headers={
                "Origin": "https://malicious-domain.com",
                "Access-Control-Request-Method": "GET"
            }
        )
        
        # Should not include CORS headers for untrusted origin
        cors_header = response.headers.get("Access-Control-Allow-Origin")
        assert cors_header != "https://malicious-domain.com"

    async def test_rate_limiting_middleware(self, security_headers_service):
        """Test rate limiting middleware."""
        from backend.middleware.rate_limiting import RateLimitMiddleware
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        
        app = FastAPI()
        
        # Add rate limiting middleware
        rate_limiter = RateLimitMiddleware(
            requests_per_minute=5,
            burst_size=10
        )
        app.add_middleware(rate_limiter)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        client = TestClient(app)
        
        # Make requests within limit
        for i in range(5):
            response = client.get("/test")
            assert response.status_code == 200
        
        # Exceed rate limit
        response = client.get("/test")
        assert response.status_code == 429  # Too Many Requests
        
        rate_limit_headers = response.headers
        assert "X-RateLimit-Limit" in rate_limit_headers
        assert "X-RateLimit-Remaining" in rate_limit_headers
        assert "Retry-After" in rate_limit_headers


class TestDataEncryption:
    """Test data encryption and protection."""

    @pytest.fixture
    def encryption_service(self):
        """Create encryption service."""
        from backend.services.encryption_service import EncryptionService
        return EncryptionService()

    async def test_data_at_rest_encryption(self, encryption_service):
        """Test encryption of sensitive data at rest."""
        sensitive_data = "credit_card_number_1234567890123456"
        
        # Encrypt data
        encrypted_data = await encryption_service.encrypt_sensitive_data(sensitive_data)
        
        # Encrypted data should be different from original
        assert encrypted_data != sensitive_data
        assert len(encrypted_data) > len(sensitive_data)  # Includes IV/salt
        
        # Decrypt data
        decrypted_data = await encryption_service.decrypt_sensitive_data(encrypted_data)
        assert decrypted_data == sensitive_data

    async def test_field_level_encryption(self, encryption_service):
        """Test field-level encryption for specific data types."""
        test_data = {
            email=f"testuser{str(uuid.uuid4())[:8]}@{self.fake.domain_name()}",",
            "phone": "+1-555-123-4567", 
            "ssn": "123-45-6789",
            "credit_card": "4111111111111111"
        }
        
        # Encrypt sensitive fields
        encrypted_record = await encryption_service.encrypt_record(
            test_data,
            sensitive_fields=["phone", "ssn", "credit_card"]
        )
        
        # Non-sensitive fields should remain plain text
        assert encrypted_record["email"] == test_data["email"]
        
        # Sensitive fields should be encrypted
        assert encrypted_record["phone"] != test_data["phone"]
        assert encrypted_record["ssn"] != test_data["ssn"]
        assert encrypted_record["credit_card"] != test_data["credit_card"]
        
        # Decrypt record
        decrypted_record = await encryption_service.decrypt_record(
            encrypted_record,
            sensitive_fields=["phone", "ssn", "credit_card"]
        )
        
        assert decrypted_record == test_data

    async def test_key_rotation_support(self, encryption_service):
        """Test encryption key rotation capabilities."""
        data = "sensitive_information_to_encrypt"
        
        # Encrypt with current key
        encrypted_v1 = await encryption_service.encrypt_with_key_version(data, key_version=1)
        
        # Rotate to new key
        await encryption_service.rotate_encryption_key()
        
        # Encrypt with new key
        encrypted_v2 = await encryption_service.encrypt_with_key_version(data, key_version=2)
        
        # Both encryptions should be different
        assert encrypted_v1 != encrypted_v2
        
        # Both should decrypt correctly with their respective keys
        decrypted_v1 = await encryption_service.decrypt_with_key_version(encrypted_v1, key_version=1)
        decrypted_v2 = await encryption_service.decrypt_with_key_version(encrypted_v2, key_version=2)
        
        assert decrypted_v1 == data
        assert decrypted_v2 == data

    async def test_pii_redaction(self, encryption_service):
        """Test PII detection and redaction."""
        text_with_pii = """
        Contact Joshua Lawson at john.doe@email.com or call 555-123-4567.
        His SSN is 123-45-6789 and credit card number is 4111111111111111.
        Address: 123 Main St, Anytown, NY 12345
        """
        
        redacted_text = await encryption_service.redact_pii(text_with_pii)
        
        # Should redact sensitive information
        assert "john.doe@email.com" not in redacted_text
        assert "555-123-4567" not in redacted_text  
        assert "123-45-6789" not in redacted_text
        assert "4111111111111111" not in redacted_text
        
        # Should contain redaction markers
        assert "[EMAIL_REDACTED]" in redacted_text or "***" in redacted_text
        assert "[PHONE_REDACTED]" in redacted_text or "***" in redacted_text


class TestSecurityCompliance:
    """Test security compliance and standards."""

    @pytest.fixture
    def compliance_checker(self):
        """Create compliance checker service."""
        from backend.services.compliance_checker import ComplianceChecker
        return ComplianceChecker()

    async def test_gdpr_compliance_checks(self, compliance_checker):
        """Test GDPR compliance requirements."""
        user_data = {
            email=f"testuser{str(uuid.uuid4())[:8]}@{self.fake.domain_name()}",",
            name=self.fake.name(),",
            "location": "EU",
            "consent_given": True,
            "consent_timestamp": datetime.utcnow(),
            "data_processing_purpose": "service_provision"
        }
        
        compliance_result = await compliance_checker.check_gdpr_compliance(user_data)
        
        assert compliance_result.is_compliant is True
        assert compliance_result.consent_valid is True
        assert compliance_result.lawful_basis is not None
        
        # Test non-compliant case
        non_compliant_data = {
            email=f"testuser{str(uuid.uuid4())[:8]}@{self.fake.domain_name()}",",
            name=self.fake.name(),",
            "location": "EU",
            "consent_given": False,  # No consent
        }
        
        non_compliant_result = await compliance_checker.check_gdpr_compliance(non_compliant_data)
        assert non_compliant_result.is_compliant is False

    async def test_data_retention_policies(self, compliance_checker):
        """Test data retention policy enforcement."""
        old_user_data = {
            "user_id": "user_123",
            "created_at": datetime.utcnow() - timedelta(days=400),  # Over 1 year old
            "last_active": datetime.utcnow() - timedelta(days=200),
            "data_type": "user_profile"
        }
        
        retention_check = await compliance_checker.check_retention_policy(old_user_data)
        
        # Should flag for deletion
        assert retention_check.requires_deletion is True
        assert retention_check.retention_period_exceeded is True

    async def test_audit_logging(self, compliance_checker):
        """Test security audit logging."""
        security_event = {
            "event_type": "authentication_failure",
            "user_id": "user_123",
            "ip_address": "192.168.1.100",
            "timestamp": datetime.utcnow(),
            "details": {
                "reason": "invalid_password",
                "attempts": 3
            }
        }
        
        # Log security event
        audit_entry = await compliance_checker.log_security_event(security_event)
        
        assert audit_entry.event_id is not None
        assert audit_entry.immutable_hash is not None  # Tamper-proof
        assert audit_entry.retention_period > 0
        
        # Retrieve audit log
        retrieved_log = await compliance_checker.get_audit_log(
            start_date=datetime.utcnow() - timedelta(minutes=5),
            end_date=datetime.utcnow(),
            event_types=["authentication_failure"]
        )
        
        assert len(retrieved_log) > 0
        assert retrieved_log[0]["event_type"] == "authentication_failure"

    async def test_vulnerability_scanning(self, compliance_checker):
        """Test automated vulnerability scanning."""
        # Mock dependency list
        dependencies = [
            {"name": "fastapi", "version": "0.68.0"},
            {"name": "pydantic", "version": "1.8.2"},
            {"name": "requests", "version": "2.25.1"},  # Potentially outdated
        ]
        
        scan_result = await compliance_checker.scan_vulnerabilities(dependencies)
        
        assert scan_result.scan_id is not None
        assert isinstance(scan_result.vulnerabilities_found, list)
        assert scan_result.risk_score >= 0
        
        # Should have recommendations if vulnerabilities found
        if scan_result.vulnerabilities_found:
            assert len(scan_result.recommendations) > 0


class TestSecurityIntegration:
    """Integration tests for complete security system."""

    @pytest.fixture
    async def security_system(self):
        """Create complete security system."""
        from backend.services.security_system import SecuritySystem
        system = SecuritySystem()
        await system.initialize()
        return system

    async def test_end_to_end_secure_flow(self, security_system):
        """Test complete secure authentication and authorization flow."""
        # User registration with security checks
        user_data = {
            "email": "security.user@company.local",
            "password": "SecurePassword123!",
            "name": "Security Test User"
        }
        
        registration_result = await security_system.secure_user_registration(user_data)
        
        assert registration_result.success is True
        assert registration_result.user_id is not None
        assert registration_result.security_checks_passed is True
        
        # Secure authentication
        auth_result = await security_system.secure_authentication(
            user_data["email"],
            user_data["password"]
        )
        
        assert auth_result.success is True
        assert auth_result.access_token is not None
        assert auth_result.refresh_token is not None
        assert auth_result.security_score > 0.8
        
        # Authorized resource access
        protected_resource = await security_system.access_protected_resource(
            auth_result.access_token,
            resource_id="protected_document_123",
            required_permission="read"
        )
        
        assert protected_resource.access_granted is True
        assert protected_resource.audit_logged is True

    async def test_security_incident_response(self, security_system):
        """Test security incident detection and response."""
        # Simulate suspicious activity
        suspicious_activities = [
            {
                "type": "multiple_failed_logins",
                "user_id": "user_123",
                "count": 10,
                "timeframe": timedelta(minutes=5)
            },
            {
                "type": "unusual_access_pattern",
                "user_id": "user_123", 
                "locations": ["US", "RU", "CN"],  # Impossible travel
                "timeframe": timedelta(hours=1)
            }
        ]
        
        incident_response = await security_system.analyze_security_incidents(
            suspicious_activities
        )
        
        assert incident_response.threat_level > 0.7  # High threat
        assert len(incident_response.recommended_actions) > 0
        assert incident_response.auto_response_triggered is True
        
        # Should automatically block user
        user_status = await security_system.get_user_security_status("user_123")
        assert user_status.is_blocked is True
        assert user_status.block_reason == "suspicious_activity"

    async def test_security_monitoring_alerts(self, security_system):
        """Test real-time security monitoring and alerting."""
        # Configure security rules
        security_rules = [
            {
                "name": "excessive_api_calls",
                "threshold": 1000,
                "timeframe": timedelta(minutes=10),
                "severity": "high"
            },
            {
                "name": "admin_access_off_hours",
                "condition": "admin_login_outside_business_hours",
                "severity": "critical"
            }
        ]
        
        await security_system.configure_monitoring_rules(security_rules)
        
        # Simulate triggering conditions
        alert_events = [
            {
                "rule_name": "excessive_api_calls",
                "user_id": "user_456",
                "api_calls": 1500,
                "timestamp": datetime.utcnow()
            }
        ]
        
        alerts = await security_system.process_security_events(alert_events)
        
        assert len(alerts) > 0
        assert alerts[0].severity == "high"
        assert alerts[0].rule_triggered == "excessive_api_calls"
        assert alerts[0].response_required is True