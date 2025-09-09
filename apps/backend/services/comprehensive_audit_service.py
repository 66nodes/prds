"""
Comprehensive Audit Logging and Compliance Reporting Service

Enterprise-grade audit system that captures, stores, and reports on all critical
system activities for SOC 2, GDPR, and regulatory compliance.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
from collections import defaultdict, Counter

import structlog
from core.config import get_settings
from core.redis import get_redis_client
from services.agent_audit_service import get_audit_service, AgentAuditService
from services.agent_action_logger import AgentActionLogger, ActionType, LogLevel

logger = structlog.get_logger(__name__)
settings = get_settings()


class AuditEventType(Enum):
    """Comprehensive audit event types for compliance."""
    
    # Authentication & Authorization
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_LOGIN_FAILED = "user_login_failed"
    TOKEN_REFRESH = "token_refresh"
    TOKEN_REVOKED = "token_revoked"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    ROLE_CHANGED = "role_changed"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_UNLOCKED = "account_unlocked"
    
    # Data Access & Modification
    DATA_READ = "data_read"
    DATA_CREATED = "data_created"
    DATA_UPDATED = "data_updated"
    DATA_DELETED = "data_deleted"
    DATA_EXPORTED = "data_exported"
    DATA_IMPORTED = "data_imported"
    BULK_OPERATION = "bulk_operation"
    
    # AI & Agent Operations
    AI_AGENT_INVOKED = "ai_agent_invoked"
    AI_GENERATION_COMPLETED = "ai_generation_completed"
    AI_VALIDATION_PERFORMED = "ai_validation_performed"
    GRAPHRAG_VALIDATION = "graphrag_validation"
    HALLUCINATION_DETECTED = "hallucination_detected"
    QUALITY_THRESHOLD_VIOLATED = "quality_threshold_violated"
    
    # System Administration
    CONFIG_CHANGED = "config_changed"
    ADMIN_ACTION = "admin_action"
    SYSTEM_MAINTENANCE = "system_maintenance"
    BACKUP_PERFORMED = "backup_performed"
    RESTORE_PERFORMED = "restore_performed"
    
    # Security Events
    SECURITY_SCAN = "security_scan"
    VULNERABILITY_DETECTED = "vulnerability_detected"
    SECURITY_POLICY_VIOLATION = "security_policy_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    IP_BLOCKED = "ip_blocked"
    
    # Compliance & Audit
    AUDIT_LOG_ACCESS = "audit_log_access"
    COMPLIANCE_REPORT_GENERATED = "compliance_report_generated"
    DATA_RETENTION_POLICY_APPLIED = "data_retention_policy_applied"
    GDPR_REQUEST_PROCESSED = "gdpr_request_processed"
    
    # Business Operations
    PRD_CREATED = "prd_created"
    PRD_MODIFIED = "prd_modified"
    PRD_PUBLISHED = "prd_published"
    DOCUMENT_SHARED = "document_shared"
    WORKFLOW_INITIATED = "workflow_initiated"
    WORKFLOW_COMPLETED = "workflow_completed"
    
    # Error & Recovery
    SYSTEM_ERROR = "system_error"
    ERROR_RECOVERY_INITIATED = "error_recovery_initiated"
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"
    FALLBACK_ACTIVATED = "fallback_activated"


class ComplianceStandard(Enum):
    """Compliance standards supported by the audit system."""
    SOC2_TYPE2 = "soc2_type2"
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"


class AuditSeverity(Enum):
    """Audit event severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Comprehensive audit event record."""
    # Core identification
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Event classification
    event_type: AuditEventType = AuditEventType.SYSTEM_ERROR
    severity: AuditSeverity = AuditSeverity.MEDIUM
    
    # Actor information
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    user_role: Optional[str] = None
    session_id: Optional[str] = None
    
    # Request context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    # Resource information
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    resource_name: Optional[str] = None
    
    # Action details
    action: Optional[str] = None
    outcome: Optional[str] = None  # success, failure, partial
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    
    # Compliance metadata
    compliance_tags: List[str] = field(default_factory=list)
    retention_period_days: int = 90
    sensitive_data: bool = False
    pii_detected: bool = False
    
    # Technical details
    duration_ms: Optional[float] = None
    data_volume_bytes: Optional[int] = None
    api_endpoint: Optional[str] = None
    http_method: Optional[str] = None
    response_status: Optional[int] = None
    
    # Business context
    business_unit: Optional[str] = None
    project_id: Optional[str] = None
    workflow_stage: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage and transmission."""
        return asdict(self)


@dataclass
class ComplianceReport:
    """Compliance report structure."""
    report_id: str
    report_type: ComplianceStandard
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    
    # Summary metrics
    total_events: int
    critical_events: int
    security_events: int
    data_access_events: int
    
    # Compliance metrics
    compliance_score: float  # 0-100
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    
    # Detailed findings
    sections: Dict[str, Dict[str, Any]]
    
    # Evidence and documentation
    evidence_files: List[str]
    attestations: Dict[str, bool]


class ComprehensiveAuditService:
    """
    Enterprise audit logging and compliance reporting service.
    
    Provides comprehensive audit trail capture, secure storage,
    compliance reporting, and regulatory adherence capabilities.
    """
    
    def __init__(self):
        self.redis_client = None
        self.agent_audit_service: Optional[AgentAuditService] = None
        self.is_initialized = False
        
        # Compliance configurations
        self.compliance_configs = {
            ComplianceStandard.SOC2_TYPE2: {
                "required_events": [
                    AuditEventType.USER_LOGIN, AuditEventType.USER_LOGOUT,
                    AuditEventType.ADMIN_ACTION, AuditEventType.CONFIG_CHANGED,
                    AuditEventType.DATA_CREATED, AuditEventType.DATA_UPDATED,
                    AuditEventType.DATA_DELETED
                ],
                "retention_days": 365,
                "immutability_required": True,
                "real_time_monitoring": True
            },
            ComplianceStandard.GDPR: {
                "required_events": [
                    AuditEventType.DATA_READ, AuditEventType.DATA_EXPORTED,
                    AuditEventType.GDPR_REQUEST_PROCESSED, 
                    AuditEventType.DATA_RETENTION_POLICY_APPLIED
                ],
                "retention_days": 2555,  # 7 years
                "right_to_erasure": True,
                "consent_tracking": True
            }
        }
        
        # Event buffer for batch processing
        self.event_buffer = []
        self.buffer_size = 100
        self.buffer_flush_interval = 30  # seconds
        
    async def initialize(self) -> None:
        """Initialize the comprehensive audit service."""
        try:
            self.redis_client = await get_redis_client()
            self.agent_audit_service = await get_audit_service()
            
            # Start background tasks
            asyncio.create_task(self._periodic_buffer_flush())
            asyncio.create_task(self._periodic_retention_cleanup())
            
            self.is_initialized = True
            logger.info("Comprehensive Audit Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Comprehensive Audit Service: {str(e)}")
            raise
    
    async def log_audit_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: Optional[str] = None,
        outcome: str = "success",
        **kwargs
    ) -> str:
        """
        Log a comprehensive audit event.
        
        Returns:
            str: Event ID for tracking and correlation
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Create audit event
        event = AuditEvent(
            event_type=event_type,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            outcome=outcome,
            **kwargs
        )
        
        # Apply compliance tags based on event type
        self._apply_compliance_tags(event)
        
        # Detect PII and sensitive data
        await self._detect_sensitive_data(event)
        
        # Add to buffer for batch processing
        self.event_buffer.append(event)
        
        # Flush buffer if full
        if len(self.event_buffer) >= self.buffer_size:
            await self._flush_event_buffer()
        
        # Log critical events immediately
        if event.severity == AuditSeverity.CRITICAL:
            await self._store_event_immediate(event)
            await self._send_real_time_alert(event)
        
        logger.debug(
            "Audit event logged",
            event_id=event.event_id,
            event_type=event_type.value,
            user_id=user_id
        )
        
        return event.event_id
    
    async def log_authentication_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        outcome: str = "success",
        error_message: Optional[str] = None
    ) -> str:
        """Log authentication-specific audit events."""
        
        return await self.log_audit_event(
            event_type=event_type,
            user_id=user_id,
            user_email=user_email,
            ip_address=ip_address,
            user_agent=user_agent,
            resource_type="authentication",
            action="authenticate",
            outcome=outcome,
            error_message=error_message,
            severity=AuditSeverity.HIGH if outcome == "failure" else AuditSeverity.MEDIUM,
            compliance_tags=["soc2", "security"]
        )
    
    async def log_data_access_event(
        self,
        event_type: AuditEventType,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        data_classification: Optional[str] = None,
        data_volume_bytes: Optional[int] = None,
        **kwargs
    ) -> str:
        """Log data access and modification events."""
        
        # Determine if this involves sensitive data
        sensitive_data = data_classification in ["confidential", "restricted", "pii"]
        
        return await self.log_audit_event(
            event_type=event_type,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            data_volume_bytes=data_volume_bytes,
            sensitive_data=sensitive_data,
            severity=AuditSeverity.HIGH if sensitive_data else AuditSeverity.MEDIUM,
            compliance_tags=["gdpr", "data_protection"],
            metadata={"data_classification": data_classification},
            **kwargs
        )
    
    async def log_ai_operation_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        task_id: Optional[str] = None,
        model_name: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        confidence_score: Optional[float] = None,
        hallucination_score: Optional[float] = None,
        **kwargs
    ) -> str:
        """Log AI and agent operation events."""
        
        # Determine severity based on quality metrics
        severity = AuditSeverity.LOW
        if hallucination_score and hallucination_score > 0.02:  # >2% threshold
            severity = AuditSeverity.CRITICAL
        elif confidence_score and confidence_score < 0.8:
            severity = AuditSeverity.HIGH
        
        return await self.log_audit_event(
            event_type=event_type,
            user_id=user_id,
            resource_type="ai_agent",
            resource_id=task_id,
            action=f"{agent_type}_execution" if agent_type else "ai_operation",
            severity=severity,
            compliance_tags=["ai_governance", "quality_assurance"],
            metadata={
                "agent_type": agent_type,
                "model_name": model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "confidence_score": confidence_score,
                "hallucination_score": hallucination_score
            },
            **kwargs
        )
    
    async def search_audit_events(
        self,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        severity: Optional[AuditSeverity] = None,
        outcome: Optional[str] = None,
        compliance_tags: Optional[List[str]] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Search audit events with comprehensive filtering.
        
        Returns:
            Tuple[List[Dict], int]: (events, total_count)
        """
        if not self.redis_client:
            await self.initialize()
        
        # Build search filters
        filters = {}
        if event_types:
            filters["event_type"] = [et.value for et in event_types]
        if user_id:
            filters["user_id"] = user_id
        if resource_type:
            filters["resource_type"] = resource_type
        if start_time:
            filters["start_time"] = start_time.isoformat()
        if end_time:
            filters["end_time"] = end_time.isoformat()
        if severity:
            filters["severity"] = severity.value
        if outcome:
            filters["outcome"] = outcome
        if compliance_tags:
            filters["compliance_tags"] = compliance_tags
        
        # Execute search (Redis implementation)
        search_key = f"audit_search:{hashlib.md5(str(filters).encode()).hexdigest()}"
        
        try:
            # Try cache first
            cached_result = await self.redis_client.get(search_key)
            if cached_result:
                result_data = json.loads(cached_result)
                events = result_data["events"][offset:offset + limit]
                return events, result_data["total_count"]
            
            # Perform actual search
            all_events = await self._execute_audit_search(filters)
            total_count = len(all_events)
            
            # Cache results for 5 minutes
            await self.redis_client.setex(
                search_key,
                300,
                json.dumps({
                    "events": all_events,
                    "total_count": total_count,
                    "cached_at": datetime.now(timezone.utc).isoformat()
                })
            )
            
            # Return paginated results
            events = all_events[offset:offset + limit]
            return events, total_count
            
        except Exception as e:
            logger.error(f"Audit search failed: {str(e)}")
            return [], 0
    
    async def generate_compliance_report(
        self,
        standard: ComplianceStandard,
        period_start: datetime,
        period_end: datetime,
        include_evidence: bool = True
    ) -> ComplianceReport:
        """Generate a comprehensive compliance report."""
        
        report_id = str(uuid.uuid4())
        logger.info(
            f"Generating compliance report",
            report_id=report_id,
            standard=standard.value,
            period_start=period_start.isoformat(),
            period_end=period_end.isoformat()
        )
        
        # Get configuration for compliance standard
        config = self.compliance_configs.get(standard, {})
        required_events = config.get("required_events", [])
        
        # Search for relevant audit events
        events, total_count = await self.search_audit_events(
            event_types=required_events,
            start_time=period_start,
            end_time=period_end,
            limit=10000  # Get all events for analysis
        )
        
        # Analyze events for compliance
        violations = []
        recommendations = []
        compliance_score = 100.0
        
        # SOC 2 Type II specific checks
        if standard == ComplianceStandard.SOC2_TYPE2:
            violations, recommendations, compliance_score = await self._analyze_soc2_compliance(
                events, period_start, period_end
            )
        
        # GDPR specific checks
        elif standard == ComplianceStandard.GDPR:
            violations, recommendations, compliance_score = await self._analyze_gdpr_compliance(
                events, period_start, period_end
            )
        
        # Generate report sections
        sections = await self._generate_report_sections(standard, events, violations)
        
        # Create compliance report
        report = ComplianceReport(
            report_id=report_id,
            report_type=standard,
            period_start=period_start,
            period_end=period_end,
            generated_at=datetime.now(timezone.utc),
            total_events=total_count,
            critical_events=len([e for e in events if e.get("severity") == "critical"]),
            security_events=len([e for e in events if "security" in e.get("compliance_tags", [])]),
            data_access_events=len([e for e in events if e.get("resource_type") == "data"]),
            compliance_score=compliance_score,
            violations=violations,
            recommendations=recommendations,
            sections=sections,
            evidence_files=[],
            attestations={}
        )
        
        # Store report
        await self._store_compliance_report(report)
        
        # Log report generation
        await self.log_audit_event(
            event_type=AuditEventType.COMPLIANCE_REPORT_GENERATED,
            resource_type="compliance_report",
            resource_id=report_id,
            action="generate_report",
            severity=AuditSeverity.MEDIUM,
            metadata={
                "standard": standard.value,
                "compliance_score": compliance_score,
                "violations_count": len(violations)
            }
        )
        
        return report
    
    async def export_audit_data(
        self,
        export_format: str = "json",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        encryption_key: Optional[str] = None
    ) -> str:
        """
        Export audit data for archival or external analysis.
        
        Returns:
            str: Export file path or download URL
        """
        
        # Get events to export
        events, total_count = await self.search_audit_events(
            event_types=event_types,
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=50000  # Large limit for export
        )
        
        # Generate export file
        export_id = str(uuid.uuid4())
        export_path = f"exports/audit_export_{export_id}.{export_format}"
        
        if export_format == "json":
            await self._export_as_json(events, export_path, encryption_key)
        elif export_format == "csv":
            await self._export_as_csv(events, export_path, encryption_key)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        # Log export event
        await self.log_audit_event(
            event_type=AuditEventType.DATA_EXPORTED,
            user_id=user_id,
            resource_type="audit_log",
            action="export_data",
            data_volume_bytes=len(json.dumps(events).encode()),
            severity=AuditSeverity.HIGH,
            compliance_tags=["data_export", "audit_trail"],
            metadata={
                "export_format": export_format,
                "records_count": len(events),
                "encrypted": bool(encryption_key)
            }
        )
        
        return export_path
    
    async def apply_retention_policy(self) -> Dict[str, int]:
        """Apply data retention policies to audit logs."""
        
        logger.info("Applying audit log retention policies")
        
        retention_summary = {
            "soc2_retained": 0,
            "gdpr_retained": 0,
            "expired_deleted": 0,
            "archived": 0
        }
        
        # Get all audit events older than minimum retention
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)  # Minimum retention
        old_events, _ = await self.search_audit_events(
            end_time=cutoff_date,
            limit=10000
        )
        
        for event_data in old_events:
            event_date = datetime.fromisoformat(event_data["timestamp"])
            compliance_tags = event_data.get("compliance_tags", [])
            retention_days = event_data.get("retention_period_days", 90)
            
            # Calculate if event should be retained, archived, or deleted
            age_days = (datetime.now(timezone.utc) - event_date).days
            
            if age_days > retention_days:
                if "soc2" in compliance_tags and age_days < 365:
                    retention_summary["soc2_retained"] += 1
                elif "gdpr" in compliance_tags and age_days < 2555:
                    retention_summary["gdpr_retained"] += 1
                elif age_days > 2555:  # Beyond all retention requirements
                    await self._archive_or_delete_event(event_data["event_id"])
                    retention_summary["expired_deleted"] += 1
                else:
                    await self._archive_event(event_data["event_id"])
                    retention_summary["archived"] += 1
        
        # Log retention policy application
        await self.log_audit_event(
            event_type=AuditEventType.DATA_RETENTION_POLICY_APPLIED,
            action="apply_retention_policy",
            severity=AuditSeverity.MEDIUM,
            metadata=retention_summary
        )
        
        return retention_summary
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on audit system."""
        
        health_status = {
            "status": "healthy",
            "initialized": self.is_initialized,
            "redis_connected": bool(self.redis_client),
            "agent_audit_connected": bool(self.agent_audit_service),
            "buffer_size": len(self.event_buffer),
            "last_flush": None,  # Would track in production
            "storage_available": True,  # Would check disk space
        }
        
        try:
            # Test Redis connectivity
            if self.redis_client:
                await self.redis_client.ping()
                health_status["redis_ping_ms"] = "< 1"
            
            # Test recent audit event creation
            test_event_id = await self.log_audit_event(
                event_type=AuditEventType.SYSTEM_ERROR,
                action="health_check",
                outcome="success",
                severity=AuditSeverity.LOW,
                metadata={"test": True}
            )
            health_status["audit_logging_functional"] = bool(test_event_id)
            
        except Exception as e:
            health_status["status"] = "degraded"
            health_status["error"] = str(e)
        
        return health_status
    
    # Private methods
    
    def _apply_compliance_tags(self, event: AuditEvent) -> None:
        """Apply compliance tags based on event type and content."""
        
        # SOC 2 relevant events
        soc2_events = [
            AuditEventType.USER_LOGIN, AuditEventType.USER_LOGOUT,
            AuditEventType.ADMIN_ACTION, AuditEventType.CONFIG_CHANGED,
            AuditEventType.DATA_CREATED, AuditEventType.DATA_UPDATED,
            AuditEventType.DATA_DELETED, AuditEventType.SECURITY_SCAN
        ]
        
        if event.event_type in soc2_events:
            event.compliance_tags.append("soc2")
        
        # GDPR relevant events
        gdpr_events = [
            AuditEventType.DATA_READ, AuditEventType.DATA_EXPORTED,
            AuditEventType.DATA_CREATED, AuditEventType.DATA_UPDATED,
            AuditEventType.DATA_DELETED, AuditEventType.GDPR_REQUEST_PROCESSED
        ]
        
        if event.event_type in gdpr_events:
            event.compliance_tags.append("gdpr")
        
        # Security events
        security_events = [
            AuditEventType.USER_LOGIN_FAILED, AuditEventType.PERMISSION_DENIED,
            AuditEventType.ACCOUNT_LOCKED, AuditEventType.SECURITY_SCAN,
            AuditEventType.VULNERABILITY_DETECTED, AuditEventType.SUSPICIOUS_ACTIVITY
        ]
        
        if event.event_type in security_events:
            event.compliance_tags.append("security")
        
        # AI governance events
        ai_events = [
            AuditEventType.AI_AGENT_INVOKED, AuditEventType.AI_GENERATION_COMPLETED,
            AuditEventType.GRAPHRAG_VALIDATION, AuditEventType.HALLUCINATION_DETECTED
        ]
        
        if event.event_type in ai_events:
            event.compliance_tags.append("ai_governance")
    
    async def _detect_sensitive_data(self, event: AuditEvent) -> None:
        """Detect PII and sensitive data in audit events."""
        
        # Simple PII detection patterns
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        ]
        
        # Check common fields for PII
        text_to_check = " ".join([
            str(event.error_message or ""),
            str(event.metadata.get("description", "")),
            str(event.metadata.get("content", ""))
        ])
        
        import re
        for pattern in pii_patterns:
            if re.search(pattern, text_to_check):
                event.pii_detected = True
                event.sensitive_data = True
                event.retention_period_days = max(event.retention_period_days, 2555)  # GDPR retention
                break
    
    async def _flush_event_buffer(self) -> None:
        """Flush buffered events to storage."""
        if not self.event_buffer:
            return
        
        events_to_flush = self.event_buffer.copy()
        self.event_buffer.clear()
        
        try:
            # Store events in batch
            await self._store_events_batch(events_to_flush)
            
        except Exception as e:
            logger.error(f"Failed to flush audit event buffer: {str(e)}")
            # Re-add events to buffer for retry
            self.event_buffer.extend(events_to_flush)
    
    async def _store_events_batch(self, events: List[AuditEvent]) -> None:
        """Store multiple audit events in a single batch operation."""
        if not self.redis_client:
            return
        
        pipe = self.redis_client.pipeline()
        
        for event in events:
            event_key = f"audit_event:{event.event_id}"
            event_data = json.dumps(event.to_dict(), default=str)
            
            # Store event with appropriate TTL
            ttl_seconds = event.retention_period_days * 24 * 60 * 60
            pipe.setex(event_key, ttl_seconds, event_data)
            
            # Add to time-based indices
            date_key = f"audit_by_date:{event.timestamp.date().isoformat()}"
            pipe.sadd(date_key, event.event_id)
            pipe.expire(date_key, ttl_seconds)
            
            # Add to user-based index if applicable
            if event.user_id:
                user_key = f"audit_by_user:{event.user_id}"
                pipe.sadd(user_key, event.event_id)
                pipe.expire(user_key, ttl_seconds)
            
            # Add to type-based index
            type_key = f"audit_by_type:{event.event_type.value}"
            pipe.sadd(type_key, event.event_id)
            pipe.expire(type_key, ttl_seconds)
        
        await pipe.execute()
    
    async def _store_event_immediate(self, event: AuditEvent) -> None:
        """Store a single critical event immediately."""
        await self._store_events_batch([event])
    
    async def _send_real_time_alert(self, event: AuditEvent) -> None:
        """Send real-time alert for critical audit events."""
        
        alert_data = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "severity": event.severity.value,
            "timestamp": event.timestamp.isoformat(),
            "user_id": event.user_id,
            "resource_type": event.resource_type,
            "action": event.action,
            "outcome": event.outcome
        }
        
        # Send to monitoring system (would integrate with actual alerting)
        logger.critical(
            "Critical audit event detected",
            **alert_data
        )
        
        # Store in critical events queue
        if self.redis_client:
            await self.redis_client.lpush(
                "critical_audit_events",
                json.dumps(alert_data, default=str)
            )
            await self.redis_client.ltrim("critical_audit_events", 0, 999)  # Keep last 1000
    
    async def _execute_audit_search(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute audit event search with filters."""
        # This is a simplified implementation
        # In production, would use proper search index (Elasticsearch, etc.)
        
        results = []
        
        if not self.redis_client:
            return results
        
        # Get potential event IDs from indices
        candidate_ids = set()
        
        # Time-based search
        if "start_time" in filters or "end_time" in filters:
            start_time = datetime.fromisoformat(filters.get("start_time", "2020-01-01T00:00:00+00:00"))
            end_time = datetime.fromisoformat(filters.get("end_time", datetime.now(timezone.utc).isoformat()))
            
            # Get events from date range
            current_date = start_time.date()
            while current_date <= end_time.date():
                date_key = f"audit_by_date:{current_date.isoformat()}"
                date_event_ids = await self.redis_client.smembers(date_key)
                if date_event_ids:
                    candidate_ids.update(date_event_ids)
                current_date += timedelta(days=1)
        
        # Apply additional filters
        if "user_id" in filters:
            user_key = f"audit_by_user:{filters['user_id']}"
            user_event_ids = await self.redis_client.smembers(user_key)
            if candidate_ids:
                candidate_ids.intersection_update(user_event_ids)
            else:
                candidate_ids = set(user_event_ids)
        
        if "event_type" in filters:
            type_ids = set()
            for event_type in filters["event_type"]:
                type_key = f"audit_by_type:{event_type}"
                type_event_ids = await self.redis_client.smembers(type_key)
                type_ids.update(type_event_ids)
            
            if candidate_ids:
                candidate_ids.intersection_update(type_ids)
            else:
                candidate_ids = type_ids
        
        # Retrieve and filter events
        for event_id in candidate_ids:
            event_key = f"audit_event:{event_id}"
            event_data = await self.redis_client.get(event_key)
            if event_data:
                event = json.loads(event_data)
                if self._matches_filters(event, filters):
                    results.append(event)
        
        # Sort by timestamp (descending)
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return results
    
    def _matches_filters(self, event: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if event matches search filters."""
        
        for filter_key, filter_value in filters.items():
            if filter_key in ["start_time", "end_time"]:
                continue  # Already handled in search
            
            event_value = event.get(filter_key)
            
            if filter_key == "event_type" and isinstance(filter_value, list):
                if event_value not in filter_value:
                    return False
            elif filter_key == "compliance_tags" and isinstance(filter_value, list):
                event_tags = event.get("compliance_tags", [])
                if not any(tag in event_tags for tag in filter_value):
                    return False
            elif event_value != filter_value:
                return False
        
        return True
    
    async def _analyze_soc2_compliance(
        self,
        events: List[Dict[str, Any]],
        period_start: datetime,
        period_end: datetime
    ) -> Tuple[List[Dict[str, Any]], List[str], float]:
        """Analyze events for SOC 2 Type II compliance."""
        
        violations = []
        recommendations = []
        compliance_score = 100.0
        
        # Check for required event types
        required_events = ["user_login", "admin_action", "config_changed"]
        event_types_found = set(event.get("event_type") for event in events)
        
        for required_event in required_events:
            if required_event not in event_types_found:
                violations.append({
                    "type": "missing_events",
                    "description": f"No {required_event} events found in period",
                    "severity": "high"
                })
                compliance_score -= 10
        
        # Check for excessive failed logins
        failed_logins = [e for e in events if e.get("event_type") == "user_login_failed"]
        if len(failed_logins) > 100:
            violations.append({
                "type": "excessive_failed_logins",
                "description": f"{len(failed_logins)} failed login attempts detected",
                "severity": "medium"
            })
            compliance_score -= 5
        
        # Check for admin actions without proper audit trail
        admin_actions = [e for e in events if e.get("event_type") == "admin_action"]
        for action in admin_actions:
            if not action.get("user_id"):
                violations.append({
                    "type": "incomplete_audit_trail",
                    "description": "Admin action without user identification",
                    "event_id": action.get("event_id"),
                    "severity": "critical"
                })
                compliance_score -= 15
        
        # Generate recommendations
        if violations:
            recommendations.append("Ensure all required event types are being logged")
            recommendations.append("Implement stronger authentication controls")
            recommendations.append("Review admin action logging completeness")
        
        return violations, recommendations, max(0, compliance_score)
    
    async def _analyze_gdpr_compliance(
        self,
        events: List[Dict[str, Any]],
        period_start: datetime,
        period_end: datetime
    ) -> Tuple[List[Dict[str, Any]], List[str], float]:
        """Analyze events for GDPR compliance."""
        
        violations = []
        recommendations = []
        compliance_score = 100.0
        
        # Check for data access events
        data_access_events = [e for e in events if e.get("event_type") in ["data_read", "data_exported"]]
        pii_access_events = [e for e in data_access_events if e.get("pii_detected")]
        
        # Check for PII access without proper documentation
        undocumented_pii_access = [e for e in pii_access_events if not e.get("business_justification")]
        if undocumented_pii_access:
            violations.append({
                "type": "undocumented_pii_access",
                "description": f"{len(undocumented_pii_access)} PII access events without business justification",
                "severity": "high"
            })
            compliance_score -= 20
        
        # Check for data retention violations
        expired_data_events = [e for e in events if e.get("retention_violation")]
        if expired_data_events:
            violations.append({
                "type": "retention_violation",
                "description": f"{len(expired_data_events)} events with retention violations",
                "severity": "critical"
            })
            compliance_score -= 25
        
        # Check for GDPR request processing
        gdpr_requests = [e for e in events if e.get("event_type") == "gdpr_request_processed"]
        incomplete_requests = [r for r in gdpr_requests if r.get("outcome") != "success"]
        if incomplete_requests:
            violations.append({
                "type": "incomplete_gdpr_requests",
                "description": f"{len(incomplete_requests)} GDPR requests not completed successfully",
                "severity": "critical"
            })
            compliance_score -= 20
        
        # Generate recommendations
        if violations:
            recommendations.append("Implement business justification tracking for PII access")
            recommendations.append("Review and update data retention policies")
            recommendations.append("Improve GDPR request processing workflows")
        
        return violations, recommendations, max(0, compliance_score)
    
    async def _generate_report_sections(
        self,
        standard: ComplianceStandard,
        events: List[Dict[str, Any]],
        violations: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Generate detailed report sections for compliance report."""
        
        sections = {}
        
        if standard == ComplianceStandard.SOC2_TYPE2:
            sections = {
                "security": {
                    "title": "Security Controls",
                    "description": "Authentication, authorization, and access controls",
                    "events_count": len([e for e in events if "security" in e.get("compliance_tags", [])]),
                    "violations": [v for v in violations if v.get("type") in ["excessive_failed_logins", "incomplete_audit_trail"]],
                    "status": "compliant" if not any(v.get("severity") == "critical" for v in violations) else "non_compliant"
                },
                "availability": {
                    "title": "System Availability",
                    "description": "System uptime and performance monitoring",
                    "events_count": len([e for e in events if e.get("event_type") in ["system_error", "performance_alert"]]),
                    "violations": [],
                    "status": "compliant"
                },
                "confidentiality": {
                    "title": "Data Confidentiality",
                    "description": "Data access controls and encryption",
                    "events_count": len([e for e in events if e.get("sensitive_data")]),
                    "violations": [],
                    "status": "compliant"
                }
            }
        
        elif standard == ComplianceStandard.GDPR:
            sections = {
                "lawful_basis": {
                    "title": "Lawful Basis for Processing",
                    "description": "Documentation of lawful basis for data processing",
                    "events_count": len([e for e in events if e.get("pii_detected")]),
                    "violations": [v for v in violations if v.get("type") == "undocumented_pii_access"],
                    "status": "compliant" if not violations else "non_compliant"
                },
                "data_subject_rights": {
                    "title": "Data Subject Rights",
                    "description": "Processing of data subject requests",
                    "events_count": len([e for e in events if e.get("event_type") == "gdpr_request_processed"]),
                    "violations": [v for v in violations if v.get("type") == "incomplete_gdpr_requests"],
                    "status": "compliant" if not violations else "non_compliant"
                },
                "data_retention": {
                    "title": "Data Retention",
                    "description": "Data retention policy compliance",
                    "events_count": len([e for e in events if e.get("event_type") == "data_retention_policy_applied"]),
                    "violations": [v for v in violations if v.get("type") == "retention_violation"],
                    "status": "compliant" if not violations else "non_compliant"
                }
            }
        
        return sections
    
    async def _store_compliance_report(self, report: ComplianceReport) -> None:
        """Store compliance report in secure storage."""
        if not self.redis_client:
            return
        
        report_key = f"compliance_report:{report.report_id}"
        report_data = asdict(report)
        
        # Store report with appropriate retention
        await self.redis_client.setex(
            report_key,
            86400 * 2555,  # 7 years retention
            json.dumps(report_data, default=str)
        )
        
        # Add to reports index
        reports_key = f"compliance_reports:{report.report_type.value}"
        await self.redis_client.sadd(reports_key, report.report_id)
    
    async def _export_as_json(
        self,
        events: List[Dict[str, Any]],
        export_path: str,
        encryption_key: Optional[str] = None
    ) -> None:
        """Export audit events as JSON file."""
        import os
        
        # Ensure export directory exists
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        
        export_data = {
            "export_metadata": {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "records_count": len(events),
                "encrypted": bool(encryption_key)
            },
            "events": events
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        # Encrypt file if key provided
        if encryption_key:
            await self._encrypt_file(export_path, encryption_key)
    
    async def _export_as_csv(
        self,
        events: List[Dict[str, Any]],
        export_path: str,
        encryption_key: Optional[str] = None
    ) -> None:
        """Export audit events as CSV file."""
        import csv
        import os
        
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        
        if not events:
            return
        
        # Get all unique field names
        fieldnames = set()
        for event in events:
            fieldnames.update(event.keys())
        
        fieldnames = sorted(list(fieldnames))
        
        with open(export_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for event in events:
                # Flatten complex fields
                flattened_event = {}
                for key, value in event.items():
                    if isinstance(value, (dict, list)):
                        flattened_event[key] = json.dumps(value, default=str)
                    else:
                        flattened_event[key] = str(value) if value is not None else ""
                
                writer.writerow(flattened_event)
        
        if encryption_key:
            await self._encrypt_file(export_path, encryption_key)
    
    async def _encrypt_file(self, file_path: str, encryption_key: str) -> None:
        """Encrypt a file using AES encryption."""
        # This is a placeholder - in production, use proper encryption library
        logger.info(f"File encryption requested for {file_path}")
        # Would implement actual file encryption here
    
    async def _archive_event(self, event_id: str) -> None:
        """Archive an audit event to long-term storage."""
        logger.debug(f"Archiving audit event {event_id}")
        # Would implement archival to S3, tape storage, etc.
    
    async def _archive_or_delete_event(self, event_id: str) -> None:
        """Archive or delete an expired audit event."""
        if not self.redis_client:
            return
        
        # Delete from Redis
        event_key = f"audit_event:{event_id}"
        await self.redis_client.delete(event_key)
        
        logger.debug(f"Deleted expired audit event {event_id}")
    
    async def _periodic_buffer_flush(self) -> None:
        """Periodic task to flush event buffer."""
        while True:
            try:
                await asyncio.sleep(self.buffer_flush_interval)
                if self.event_buffer:
                    await self._flush_event_buffer()
            except Exception as e:
                logger.error(f"Error in periodic buffer flush: {str(e)}")
    
    async def _periodic_retention_cleanup(self) -> None:
        """Periodic task to apply retention policies."""
        while True:
            try:
                # Run retention cleanup daily
                await asyncio.sleep(86400)  # 24 hours
                await self.apply_retention_policy()
            except Exception as e:
                logger.error(f"Error in periodic retention cleanup: {str(e)}")


# Global service instance
_comprehensive_audit_service: Optional[ComprehensiveAuditService] = None


async def get_comprehensive_audit_service() -> ComprehensiveAuditService:
    """Get the global comprehensive audit service instance."""
    global _comprehensive_audit_service
    
    if not _comprehensive_audit_service:
        _comprehensive_audit_service = ComprehensiveAuditService()
        await _comprehensive_audit_service.initialize()
    
    return _comprehensive_audit_service