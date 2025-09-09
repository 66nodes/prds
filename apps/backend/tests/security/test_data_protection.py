"""
Security tests for data protection and privacy compliance.

Tests encryption, data masking, access logging, and compliance
with privacy regulations like GDPR.
"""
import uuid
from tests.utilities.test_data_factory import test_data_factory

import pytest
import hashlib
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Any

from backend.services.data_protection_service import DataProtectionService
from backend.services.encryption_service import EncryptionService
from backend.services.audit_service import AuditService
from backend.core.exceptions import DataProtectionError, ComplianceError


class TestDataEncryptionSecurity:
    """Test data encryption and key management."""

    @pytest.fixture
    def encryption_service(self):
        """Create encryption service."""
        return EncryptionService()

    @pytest.fixture
    def data_protection_service(self):
        """Create data protection service."""
        return DataProtectionService()

    @pytest.fixture
    def sensitive_prd_data(self):
        """Sample PRD with sensitive information."""
        return {
            "title": "Banking Application PRD",
            "description": "Mobile banking application with advanced security",
            "requirements": [
                "PCI DSS compliance for payment processing",
                "API integration with credit score provider",
                "Customer SSN: 123-45-6789 for identity verification",
                "Database connection: postgres://admin:SecretPass123@db.bank.com/prod",
                "Encryption key: AES-256-GCM with key rotation every 30 days"
            ],
            "stakeholders": [
                {
                    name=self.fake.name(),",
                    "email": "john.smith@bank.com", 
                    "phone": "+1-555-123-4567",
                    "role": "Product Manager"
                }
            ],
            "compliance": {
                "regulations": ["PCI DSS", "SOX", "GDPR"],
                "data_classification": "confidential",
                "retention_period": "7_years"
            }
        }

    async def test_field_level_encryption(self, encryption_service, sensitive_prd_data):
        """Test field-level encryption of sensitive data."""
        # Define sensitive fields for encryption
        sensitive_fields = [
            "requirements",
            "stakeholders.email", 
            "stakeholders.phone",
            "compliance.data_classification"
        ]
        
        # Encrypt sensitive fields
        encrypted_data = await encryption_service.encrypt_sensitive_fields(
            sensitive_prd_data,
            sensitive_fields
        )
        
        # Verify encryption
        assert encrypted_data["title"] == sensitive_prd_data["title"]  # Non-sensitive unchanged
        assert encrypted_data["requirements"] != sensitive_prd_data["requirements"]  # Encrypted
        
        # Encrypted data should be longer due to encryption overhead
        assert len(str(encrypted_data["requirements"])) > len(str(sensitive_prd_data["requirements"]))
        
        # Verify stakeholder email is encrypted
        assert encrypted_data["stakeholders"][0]["email"] != sensitive_prd_data["stakeholders"][0]["email"]
        assert encrypted_data["stakeholders"][0]["name"] == sensitive_prd_data["stakeholders"][0]["name"]  # Non-sensitive

    async def test_pii_detection_and_encryption(self, data_protection_service, sensitive_prd_data):
        """Test automatic PII detection and encryption."""
        pii_analysis = await data_protection_service.analyze_pii(sensitive_prd_data)
        
        # Should detect various PII types
        assert len(pii_analysis.detected_pii) > 0
        
        detected_types = [pii["type"] for pii in pii_analysis.detected_pii]
        assert "ssn" in detected_types
        assert "email" in detected_types
        assert "phone" in detected_types
        assert "database_credentials" in detected_types
        
        # Auto-encrypt detected PII
        protected_data = await data_protection_service.protect_detected_pii(
            sensitive_prd_data,
            pii_analysis
        )
        
        # SSN should be masked/encrypted
        protected_content = str(protected_data)
        assert "123-45-6789" not in protected_content
        assert "SecretPass123" not in protected_content

    async def test_encryption_key_rotation(self, encryption_service):
        """Test encryption key rotation process."""
        test_data = "Sensitive information that needs protection"
        
        # Encrypt with current key (version 1)
        encrypted_v1 = await encryption_service.encrypt_with_key_version(
            test_data,
            key_version=1
        )
        
        # Rotate encryption key
        rotation_result = await encryption_service.rotate_encryption_keys()
        assert rotation_result.success is True
        assert rotation_result.new_key_version > 1
        
        # Encrypt with new key (version 2)
        encrypted_v2 = await encryption_service.encrypt_with_key_version(
            test_data,
            key_version=2
        )
        
        # Encrypted values should be different
        assert encrypted_v1 != encrypted_v2
        
        # Both should decrypt correctly
        decrypted_v1 = await encryption_service.decrypt_with_key_version(
            encrypted_v1,
            key_version=1
        )
        decrypted_v2 = await encryption_service.decrypt_with_key_version(
            encrypted_v2, 
            key_version=2
        )
        
        assert decrypted_v1 == test_data
        assert decrypted_v2 == test_data

    async def test_encryption_performance(self, encryption_service):
        """Test encryption performance with large datasets."""
        import time
        
        # Large dataset
        large_data = {
            "large_requirements": ["Requirement " + str(i) for i in range(1000)],
            "large_description": "A" * 10000,  # 10KB description
            "metadata": {"key_" + str(i): "value_" + str(i) for i in range(100)}
        }
        
        start_time = time.time()
        encrypted_data = await encryption_service.encrypt_sensitive_fields(
            large_data,
            ["large_requirements", "large_description", "metadata"]
        )
        encryption_time = time.time() - start_time
        
        # Should complete encryption within reasonable time
        assert encryption_time < 5.0  # Less than 5 seconds
        
        start_time = time.time()
        decrypted_data = await encryption_service.decrypt_sensitive_fields(
            encrypted_data,
            ["large_requirements", "large_description", "metadata"]
        )
        decryption_time = time.time() - start_time
        
        # Should complete decryption within reasonable time
        assert decryption_time < 5.0
        
        # Verify data integrity
        assert decrypted_data["large_requirements"] == large_data["large_requirements"]

    async def test_secure_key_storage(self, encryption_service):
        """Test secure key storage and retrieval."""
        key_metadata = await encryption_service.get_key_metadata()
        
        # Keys should not be stored in plain text
        assert "plain_key" not in key_metadata
        assert "raw_key" not in key_metadata
        
        # Should have proper key management metadata
        assert "key_version" in key_metadata
        assert "created_at" in key_metadata
        assert "algorithm" in key_metadata
        assert "key_length" in key_metadata
        
        # Algorithm should be secure
        assert key_metadata["algorithm"] in ["AES-256-GCM", "ChaCha20-Poly1305", "AES-256-CBC"]
        
        # Key length should be secure
        assert key_metadata["key_length"] >= 256  # At least 256 bits


class TestDataMaskingAndAnonymization:
    """Test data masking and anonymization features."""

    @pytest.fixture
    def data_protection_service(self):
        """Create data protection service.""" 
        return DataProtectionService()

    async def test_pii_masking_strategies(self, data_protection_service):
        """Test different PII masking strategies."""
        pii_data = {
            "email": "john.doe@company.com",
            "phone": "+1-555-123-4567",
            "ssn": "123-45-6789",
            "credit_card": "4111-1111-1111-1111",
            "ip_address": "192.168.1.100",
            name=self.fake.name(),"
        }
        
        # Test partial masking
        partial_masked = await data_protection_service.apply_partial_masking(pii_data)
        
        assert "j***@company.com" in partial_masked["email"] or "***" in partial_masked["email"]
        assert "***-123-4567" in partial_masked["phone"] or "***" in partial_masked["phone"]
        assert "***-45-6789" in partial_masked["ssn"] or "***" in partial_masked["ssn"]
        
        # Test full masking
        full_masked = await data_protection_service.apply_full_masking(pii_data)
        
        for key, value in pii_data.items():
            assert full_masked[key] != value  # Should be different
            assert "***" in full_masked[key] or "[REDACTED]" in full_masked[key]

    async def test_data_anonymization(self, data_protection_service):
        """Test data anonymization for analytics."""
        user_data = {
            "users": [
                {
                    "id": "user_001",
                    name=self.fake.name(),",
                    email=f"testuser{str(uuid.uuid4())[:8]}@{self.fake.domain_name()}",",
                    "age": 28,
                    "department": "Engineering",
                    "salary": 95000,
                    "location": "New York"
                },
                {
                    "id": "user_002", 
                    "name": "Bob Smith",
                    email=f"testuser{str(uuid.uuid4())[:8]}@{self.fake.domain_name()}",",
                    "age": 35,
                    "department": "Marketing",
                    "salary": 75000,
                    "location": "California"
                }
            ]
        }
        
        # Anonymize for analytics
        anonymized_data = await data_protection_service.anonymize_for_analytics(
            user_data,
            preserve_fields=["department", "age", "location"],
            anonymize_fields=["name", "email", "salary"]
        )
        
        # Preserved fields should remain
        assert anonymized_data["users"][0]["department"] == "Engineering"
        assert anonymized_data["users"][0]["age"] == 28
        
        # Anonymized fields should be changed
        assert anonymized_data["users"][0]["name"] != "Alice Johnson"
        assert anonymized_data["users"][0]["email"] != "aliceexample.com"
        
        # Should maintain data utility (age ranges, salary bands)
        anonymized_salary = anonymized_data["users"][0]["salary"]
        assert isinstance(anonymized_salary, str)  # Salary band like "90k-100k"

    async def test_differential_privacy(self, data_protection_service):
        """Test differential privacy mechanisms."""
        sensitive_metrics = {
            "user_count": 1000,
            "average_session_duration": 15.5,
            "conversion_rate": 0.23,
            "revenue": 250000
        }
        
        # Apply differential privacy noise
        private_metrics = await data_protection_service.apply_differential_privacy(
            sensitive_metrics,
            epsilon=1.0,  # Privacy budget
            sensitivity=1.0
        )
        
        # Values should be slightly different due to noise
        assert private_metrics["user_count"] != sensitive_metrics["user_count"]
        assert private_metrics["average_session_duration"] != sensitive_metrics["average_session_duration"]
        
        # But should be within reasonable bounds
        user_count_diff = abs(private_metrics["user_count"] - sensitive_metrics["user_count"])
        assert user_count_diff < 100  # Within 10% for reasonable privacy


class TestAccessControlAndAuditing:
    """Test access control and audit logging."""

    @pytest.fixture
    def audit_service(self):
        """Create audit service."""
        return AuditService()

    @pytest.fixture
    def data_protection_service(self):
        """Create data protection service."""
        return DataProtectionService()

    async def test_data_access_logging(self, audit_service):
        """Test logging of sensitive data access."""
        access_event = {
            "user_id": "user_123",
            "action": "view_prd",
            "resource_id": "prd_456", 
            "resource_type": "confidential_document",
            "timestamp": datetime.utcnow(),
            "ip_address": "192.168.1.100",
            "user_agent": "Mozilla/5.0 Chrome/91.0",
            "data_classification": "confidential"
        }
        
        # Log access event
        audit_entry = await audit_service.log_data_access(access_event)
        
        assert audit_entry.audit_id is not None
        assert audit_entry.event_hash is not None  # Tamper-proof hash
        assert audit_entry.classification == "confidential"
        
        # Retrieve audit log
        audit_logs = await audit_service.get_data_access_logs(
            user_id="user_123",
            start_date=datetime.utcnow() - timedelta(minutes=5),
            end_date=datetime.utcnow()
        )
        
        assert len(audit_logs) > 0
        assert audit_logs[0]["action"] == "view_prd"

    async def test_unauthorized_data_access_detection(self, audit_service):
        """Test detection of unauthorized data access patterns."""
        # Simulate suspicious access patterns
        suspicious_events = [
            # Multiple failed access attempts
            {"user_id": "user_123", "action": "access_denied", "resource": "confidential_prd_1"},
            {"user_id": "user_123", "action": "access_denied", "resource": "confidential_prd_2"},
            {"user_id": "user_123", "action": "access_denied", "resource": "confidential_prd_3"},
            # Off-hours access
            {
                "user_id": "user_456", 
                "action": "view_prd", 
                "timestamp": datetime.utcnow().replace(hour=2),  # 2 AM
                "resource": "sensitive_data"
            },
            # Bulk data access
            {"user_id": "user_789", "action": "bulk_export", "record_count": 10000}
        ]
        
        for event in suspicious_events:
            await audit_service.log_security_event(event)
        
        # Analyze for suspicious patterns
        suspicious_analysis = await audit_service.analyze_suspicious_access_patterns(
            time_window=timedelta(hours=24)
        )
        
        assert suspicious_analysis.total_suspicious_events > 0
        assert len(suspicious_analysis.flagged_users) > 0
        assert any(alert["type"] == "multiple_failures" for alert in suspicious_analysis.alerts)

    async def test_data_retention_compliance(self, data_protection_service):
        """Test data retention policy compliance."""
        # Create data with different retention requirements
        data_records = [
            {
                "id": "record_1",
                "type": "user_session",
                "created_at": datetime.utcnow() - timedelta(days=400),  # 400 days old
                "retention_policy": "1_year",
                "data": {"session_id": "sess_123", "user_id": "user_456"}
            },
            {
                "id": "record_2", 
                "type": "financial_transaction",
                "created_at": datetime.utcnow() - timedelta(days=2000),  # 5+ years old
                "retention_policy": "7_years",
                "data": {"amount": 100.00, "user_id": "user_789"}
            },
            {
                "id": "record_3",
                "type": "audit_log",
                "created_at": datetime.utcnow() - timedelta(days=3000),  # 8+ years old
                "retention_policy": "7_years", 
                "data": {"action": "login", "user_id": "user_101"}
            }
        ]
        
        # Check retention compliance
        retention_analysis = await data_protection_service.check_retention_compliance(
            data_records
        )
        
        # Should identify records for deletion
        assert len(retention_analysis.expired_records) > 0
        assert "record_1" in [r["id"] for r in retention_analysis.expired_records]  # 1 year policy exceeded
        assert "record_3" in [r["id"] for r in retention_analysis.expired_records]  # 7 year policy exceeded
        
        # Should respect retention periods
        assert "record_2" not in [r["id"] for r in retention_analysis.expired_records]  # Still within 7 year limit

    async def test_right_to_be_forgotten(self, data_protection_service):
        """Test GDPR 'Right to be Forgotten' implementation."""
        user_id = "user_to_forget_123"
        
        # Simulate user data across different systems
        user_data_locations = [
            {"system": "user_profiles", "records": 3},
            {"system": "audit_logs", "records": 150},
            {"system": "session_data", "records": 45},
            {"system": "analytics", "records": 200}
        ]
        
        # Execute right to be forgotten
        deletion_result = await data_protection_service.execute_right_to_be_forgotten(
            user_id,
            user_data_locations
        )
        
        assert deletion_result.request_id is not None
        assert deletion_result.total_records_identified > 0
        assert deletion_result.deletion_scheduled is True
        
        # Should track deletion across systems
        assert len(deletion_result.system_deletions) == len(user_data_locations)
        
        for system_deletion in deletion_result.system_deletions:
            assert system_deletion["status"] in ["scheduled", "completed", "in_progress"]

    async def test_consent_management(self, data_protection_service):
        """Test user consent management."""
        user_id = "consent_test_user_456"
        
        # Record consent for different purposes
        consent_records = [
            {
                "user_id": user_id,
                "purpose": "marketing_emails",
                "consent_given": True,
                "timestamp": datetime.utcnow(),
                "consent_method": "web_form"
            },
            {
                "user_id": user_id,
                "purpose": "analytics_tracking", 
                "consent_given": False,
                "timestamp": datetime.utcnow(),
                "consent_method": "cookie_banner"
            },
            {
                "user_id": user_id,
                "purpose": "service_provision",
                "consent_given": True,
                "timestamp": datetime.utcnow(),
                "consent_method": "terms_acceptance"
            }
        ]
        
        # Store consent records
        for consent in consent_records:
            await data_protection_service.record_user_consent(consent)
        
        # Check consent status
        consent_status = await data_protection_service.get_user_consent_status(user_id)
        
        assert consent_status.marketing_emails is True
        assert consent_status.analytics_tracking is False
        assert consent_status.service_provision is True
        
        # Verify data processing restrictions
        processing_allowed = await data_protection_service.check_processing_allowed(
            user_id,
            "analytics_tracking"
        )
        assert processing_allowed is False  # User did not consent


class TestComplianceReporting:
    """Test compliance reporting and documentation."""

    @pytest.fixture
    def data_protection_service(self):
        """Create data protection service."""
        return DataProtectionService()

    async def test_gdpr_compliance_report(self, data_protection_service):
        """Test GDPR compliance reporting."""
        report_config = {
            "report_type": "gdpr_compliance",
            "time_period": {"start": datetime.utcnow() - timedelta(days=30), "end": datetime.utcnow()},
            "include_sections": [
                "data_inventory",
                "consent_management", 
                "data_subject_requests",
                "security_measures",
                "data_breaches"
            ]
        }
        
        compliance_report = await data_protection_service.generate_compliance_report(
            report_config
        )
        
        assert compliance_report.report_id is not None
        assert compliance_report.compliance_score >= 0
        assert len(compliance_report.sections) > 0
        
        # Should include required GDPR sections
        section_titles = [s["title"] for s in compliance_report.sections]
        assert "Data Inventory" in section_titles
        assert "Consent Management" in section_titles
        assert "Data Subject Rights" in section_titles

    async def test_data_breach_documentation(self, data_protection_service):
        """Test data breach documentation and reporting."""
        breach_incident = {
            "incident_id": "breach_2024_001",
            "detected_at": datetime.utcnow(),
            "breach_type": "unauthorized_access",
            "affected_records": 1500,
            "data_types": ["email", "name", "phone"],
            "severity": "high",
            "containment_actions": [
                "Disabled compromised account",
                "Reset affected user passwords", 
                "Enhanced monitoring activated"
            ],
            "notification_required": True
        }
        
        # Document breach
        breach_documentation = await data_protection_service.document_data_breach(
            breach_incident
        )
        
        assert breach_documentation.breach_id is not None
        assert breach_documentation.regulatory_notification_deadline is not None
        assert breach_documentation.user_notification_required is True
        
        # Should generate breach notification template
        notification_template = await data_protection_service.generate_breach_notification(
            breach_incident["incident_id"]
        )
        
        assert "data breach" in notification_template.content.lower()
        assert "1500" in notification_template.content  # Affected record count
        assert len(notification_template.required_recipients) > 0

    async def test_privacy_impact_assessment(self, data_protection_service):
        """Test Privacy Impact Assessment (PIA) generation."""
        new_feature = {
            "feature_name": "AI-Powered Content Analysis",
            "description": "Automated analysis of user-generated content for insights",
            "data_types": ["user_content", "behavioral_data", "preferences"],
            "processing_purposes": ["content_improvement", "personalization", "analytics"],
            "data_sharing": ["third_party_analytics", "ai_training_partners"],
            "retention_period": "2_years",
            "user_control": ["opt_out", "data_export", "deletion_request"]
        }
        
        pia_assessment = await data_protection_service.conduct_privacy_impact_assessment(
            new_feature
        )
        
        assert pia_assessment.assessment_id is not None
        assert pia_assessment.privacy_risk_score >= 0
        assert len(pia_assessment.identified_risks) >= 0
        assert len(pia_assessment.mitigation_recommendations) >= 0
        
        # High-risk features should have specific recommendations
        if pia_assessment.privacy_risk_score > 0.7:
            assert len(pia_assessment.mitigation_recommendations) > 0
            assert any("consent" in rec.lower() for rec in pia_assessment.mitigation_recommendations)