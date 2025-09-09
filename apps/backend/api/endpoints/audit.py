"""
Audit Logging and Compliance Reporting API Endpoints

REST API endpoints for audit log search, compliance reporting,
and administrative audit management functions.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
import structlog

from api.dependencies.auth import get_current_user, require_admin, require_audit_access
from services.comprehensive_audit_service import (
    get_comprehensive_audit_service,
    ComprehensiveAuditService,
    AuditEventType,
    AuditSeverity,
    ComplianceStandard,
    ComplianceReport
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1/audit", tags=["Audit & Compliance"])


# Request/Response Models

class AuditSearchRequest(BaseModel):
    """Request model for audit log search."""
    event_types: Optional[List[str]] = Field(None, description="Filter by event types")
    user_id: Optional[str] = Field(None, description="Filter by user ID")
    user_email: Optional[str] = Field(None, description="Filter by user email")
    resource_type: Optional[str] = Field(None, description="Filter by resource type")
    resource_id: Optional[str] = Field(None, description="Filter by resource ID")
    start_time: Optional[datetime] = Field(None, description="Start time for date range")
    end_time: Optional[datetime] = Field(None, description="End time for date range")
    severity: Optional[str] = Field(None, description="Filter by severity level")
    outcome: Optional[str] = Field(None, description="Filter by outcome (success/failure/error)")
    compliance_tags: Optional[List[str]] = Field(None, description="Filter by compliance tags")
    search_text: Optional[str] = Field(None, description="Free text search in event details")
    ip_address: Optional[str] = Field(None, description="Filter by IP address")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Results offset for pagination")
    
    @validator("event_types", pre=True)
    def validate_event_types(cls, v):
        if v is None:
            return None
        valid_types = [e.value for e in AuditEventType]
        invalid_types = [t for t in v if t not in valid_types]
        if invalid_types:
            raise ValueError(f"Invalid event types: {invalid_types}")
        return v
    
    @validator("severity", pre=True)
    def validate_severity(cls, v):
        if v is None:
            return None
        valid_severities = [s.value for s in AuditSeverity]
        if v not in valid_severities:
            raise ValueError(f"Invalid severity: {v}. Must be one of {valid_severities}")
        return v


class AuditEventResponse(BaseModel):
    """Response model for audit event."""
    event_id: str
    timestamp: datetime
    event_type: str
    severity: str
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    user_role: Optional[str] = None
    ip_address: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: Optional[str] = None
    outcome: Optional[str] = None
    error_message: Optional[str] = None
    duration_ms: Optional[float] = None
    compliance_tags: List[str] = []
    metadata: Dict[str, Any] = {}


class AuditSearchResponse(BaseModel):
    """Response model for audit search results."""
    events: List[AuditEventResponse]
    total_count: int
    page_info: Dict[str, Any]
    search_metadata: Dict[str, Any]


class ComplianceReportRequest(BaseModel):
    """Request model for compliance report generation."""
    standard: str = Field(..., description="Compliance standard (soc2_type2, gdpr, etc.)")
    period_start: datetime = Field(..., description="Report period start date")
    period_end: datetime = Field(..., description="Report period end date")
    include_evidence: bool = Field(True, description="Include supporting evidence")
    format: str = Field("json", description="Report format (json, pdf, csv)")
    
    @validator("standard")
    def validate_standard(cls, v):
        valid_standards = [s.value for s in ComplianceStandard]
        if v not in valid_standards:
            raise ValueError(f"Invalid standard: {v}. Must be one of {valid_standards}")
        return v
    
    @validator("period_end")
    def validate_period(cls, v, values):
        if "period_start" in values and v <= values["period_start"]:
            raise ValueError("Period end must be after period start")
        return v


class ComplianceReportResponse(BaseModel):
    """Response model for compliance report."""
    report_id: str
    report_type: str
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    compliance_score: float
    total_events: int
    critical_events: int
    security_events: int
    violations_count: int
    download_url: Optional[str] = None
    status: str = "completed"


class AuditExportRequest(BaseModel):
    """Request model for audit data export."""
    format: str = Field("json", description="Export format (json, csv)")
    start_time: Optional[datetime] = Field(None, description="Start time for export")
    end_time: Optional[datetime] = Field(None, description="End time for export")
    event_types: Optional[List[str]] = Field(None, description="Filter by event types")
    user_id: Optional[str] = Field(None, description="Filter by specific user")
    encrypt: bool = Field(False, description="Encrypt export file")
    include_sensitive: bool = Field(False, description="Include sensitive data (admin only)")


class AuditStatsResponse(BaseModel):
    """Response model for audit statistics."""
    total_events: int
    events_last_24h: int
    events_last_7d: int
    critical_events_last_24h: int
    top_event_types: List[Dict[str, Any]]
    top_users: List[Dict[str, Any]]
    compliance_scores: Dict[str, float]
    security_alerts: int


# API Endpoints

@router.get("/search", response_model=AuditSearchResponse, 
            summary="Search audit logs with comprehensive filtering")
async def search_audit_logs(
    request: AuditSearchRequest = Depends(),
    current_user = Depends(require_audit_access),
    audit_service: ComprehensiveAuditService = Depends(get_comprehensive_audit_service)
) -> AuditSearchResponse:
    """
    Search audit logs with comprehensive filtering capabilities.
    
    Supports filtering by event type, user, time range, severity, and more.
    Requires audit access permissions.
    """
    
    try:
        # Convert string enums to enum objects
        event_types_enum = None
        if request.event_types:
            event_types_enum = [AuditEventType(et) for et in request.event_types]
        
        severity_enum = None
        if request.severity:
            severity_enum = AuditSeverity(request.severity)
        
        # Execute search
        events, total_count = await audit_service.search_audit_events(
            event_types=event_types_enum,
            user_id=request.user_id,
            resource_type=request.resource_type,
            start_time=request.start_time,
            end_time=request.end_time,
            severity=severity_enum,
            outcome=request.outcome,
            compliance_tags=request.compliance_tags,
            limit=request.limit,
            offset=request.offset
        )
        
        # Convert to response format
        event_responses = []
        for event_data in events:
            event_responses.append(AuditEventResponse(**event_data))
        
        # Calculate page info
        page_info = {
            "page": (request.offset // request.limit) + 1,
            "page_size": request.limit,
            "total_pages": (total_count + request.limit - 1) // request.limit,
            "has_next": request.offset + request.limit < total_count,
            "has_previous": request.offset > 0
        }
        
        # Search metadata
        search_metadata = {
            "search_duration_ms": 0,  # Would be calculated
            "filters_applied": {
                "event_types": request.event_types,
                "user_id": request.user_id,
                "severity": request.severity,
                "time_range": bool(request.start_time or request.end_time)
            }
        }
        
        # Log audit search
        await audit_service.log_audit_event(
            event_type=AuditEventType.AUDIT_LOG_ACCESS,
            user_id=current_user.id,
            user_email=current_user.email,
            resource_type="audit_log",
            action="search_logs",
            severity=AuditSeverity.MEDIUM,
            metadata={
                "results_count": len(events),
                "filters": request.dict(exclude_none=True)
            }
        )
        
        return AuditSearchResponse(
            events=event_responses,
            total_count=total_count,
            page_info=page_info,
            search_metadata=search_metadata
        )
    
    except Exception as e:
        logger.error(f"Audit search failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search audit logs"
        )


@router.get("/events/{event_id}", response_model=AuditEventResponse,
            summary="Get specific audit event details")
async def get_audit_event(
    event_id: str,
    current_user = Depends(require_audit_access),
    audit_service: ComprehensiveAuditService = Depends(get_comprehensive_audit_service)
) -> AuditEventResponse:
    """
    Retrieve detailed information about a specific audit event.
    
    Requires audit access permissions.
    """
    
    try:
        # Search for specific event
        events, _ = await audit_service.search_audit_events(limit=1)
        
        # Find event by ID (simplified - in production would have direct lookup)
        event_data = None
        for event in events:
            if event.get("event_id") == event_id:
                event_data = event
                break
        
        if not event_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Audit event not found"
            )
        
        # Log audit event access
        await audit_service.log_audit_event(
            event_type=AuditEventType.AUDIT_LOG_ACCESS,
            user_id=current_user.id,
            resource_type="audit_event",
            resource_id=event_id,
            action="view_event_details",
            severity=AuditSeverity.LOW
        )
        
        return AuditEventResponse(**event_data)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve audit event {event_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit event"
        )


@router.get("/stats", response_model=AuditStatsResponse,
            summary="Get audit system statistics and overview")
async def get_audit_statistics(
    current_user = Depends(require_audit_access),
    audit_service: ComprehensiveAuditService = Depends(get_comprehensive_audit_service)
) -> AuditStatsResponse:
    """
    Get comprehensive audit system statistics and overview.
    
    Provides summary metrics, top events, users, and compliance scores.
    """
    
    try:
        # Get recent events for statistics
        now = datetime.now(timezone.utc)
        
        # Last 24 hours
        events_24h, count_24h = await audit_service.search_audit_events(
            start_time=now - timedelta(hours=24),
            limit=5000
        )
        
        # Last 7 days
        events_7d, count_7d = await audit_service.search_audit_events(
            start_time=now - timedelta(days=7),
            limit=10000
        )
        
        # All time (limited)
        all_events, total_count = await audit_service.search_audit_events(
            limit=10000
        )
        
        # Calculate statistics
        critical_events_24h = len([e for e in events_24h if e.get("severity") == "critical"])
        
        # Top event types
        from collections import Counter
        event_type_counts = Counter(e.get("event_type") for e in events_7d)
        top_event_types = [
            {"event_type": et, "count": count}
            for et, count in event_type_counts.most_common(10)
        ]
        
        # Top users
        user_counts = Counter(e.get("user_id") for e in events_7d if e.get("user_id"))
        top_users = [
            {"user_id": user_id, "event_count": count}
            for user_id, count in user_counts.most_common(10)
        ]
        
        # Compliance scores (mock - would be calculated from actual reports)
        compliance_scores = {
            "soc2_type2": 95.5,
            "gdpr": 98.2,
            "overall": 96.8
        }
        
        # Security alerts
        security_alerts = len([
            e for e in events_24h 
            if e.get("event_type") in ["suspicious_activity", "permission_denied", "user_login_failed"]
        ])
        
        return AuditStatsResponse(
            total_events=total_count,
            events_last_24h=count_24h,
            events_last_7d=count_7d,
            critical_events_last_24h=critical_events_24h,
            top_event_types=top_event_types,
            top_users=top_users,
            compliance_scores=compliance_scores,
            security_alerts=security_alerts
        )
    
    except Exception as e:
        logger.error(f"Failed to get audit statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit statistics"
        )


@router.post("/reports/compliance", response_model=ComplianceReportResponse,
             summary="Generate compliance report")
async def generate_compliance_report(
    request: ComplianceReportRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(require_admin),
    audit_service: ComprehensiveAuditService = Depends(get_comprehensive_audit_service)
) -> ComplianceReportResponse:
    """
    Generate a comprehensive compliance report for the specified standard and period.
    
    Supports SOC 2 Type II, GDPR, and other compliance frameworks.
    Requires admin permissions.
    """
    
    try:
        # Validate period is not too large
        period_days = (request.period_end - request.period_start).days
        if period_days > 365:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Report period cannot exceed 365 days"
            )
        
        # Convert string to enum
        standard_enum = ComplianceStandard(request.standard)
        
        # Generate report
        report = await audit_service.generate_compliance_report(
            standard=standard_enum,
            period_start=request.period_start,
            period_end=request.period_end,
            include_evidence=request.include_evidence
        )
        
        # Log report generation
        await audit_service.log_audit_event(
            event_type=AuditEventType.COMPLIANCE_REPORT_GENERATED,
            user_id=current_user.id,
            user_email=current_user.email,
            resource_type="compliance_report",
            resource_id=report.report_id,
            action="generate_compliance_report",
            severity=AuditSeverity.MEDIUM,
            compliance_tags=["compliance", request.standard],
            metadata={
                "standard": request.standard,
                "period_days": period_days,
                "include_evidence": request.include_evidence
            }
        )
        
        return ComplianceReportResponse(
            report_id=report.report_id,
            report_type=report.report_type.value,
            period_start=report.period_start,
            period_end=report.period_end,
            generated_at=report.generated_at,
            compliance_score=report.compliance_score,
            total_events=report.total_events,
            critical_events=report.critical_events,
            security_events=report.security_events,
            violations_count=len(report.violations),
            download_url=f"/api/v1/audit/reports/{report.report_id}/download",
            status="completed"
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to generate compliance report: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate compliance report"
        )


@router.get("/reports/{report_id}", 
            summary="Get compliance report details")
async def get_compliance_report(
    report_id: str,
    current_user = Depends(require_admin),
    audit_service: ComprehensiveAuditService = Depends(get_comprehensive_audit_service)
) -> Dict[str, Any]:
    """
    Retrieve detailed compliance report information.
    
    Returns full report with sections, violations, and recommendations.
    Requires admin permissions.
    """
    
    try:
        # This would retrieve from storage in production
        # For now, return a mock response
        
        report_data = {
            "report_id": report_id,
            "status": "completed",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "compliance_score": 95.5,
            "sections": {
                "security": {
                    "status": "compliant",
                    "score": 98.0,
                    "events_reviewed": 1250,
                    "violations": []
                },
                "availability": {
                    "status": "compliant", 
                    "score": 99.5,
                    "events_reviewed": 850,
                    "violations": []
                }
            },
            "violations": [],
            "recommendations": [
                "Consider implementing additional monitoring for failed login attempts",
                "Review admin action logging completeness"
            ]
        }
        
        # Log report access
        await audit_service.log_audit_event(
            event_type=AuditEventType.AUDIT_LOG_ACCESS,
            user_id=current_user.id,
            resource_type="compliance_report",
            resource_id=report_id,
            action="view_report",
            severity=AuditSeverity.MEDIUM
        )
        
        return report_data
    
    except Exception as e:
        logger.error(f"Failed to retrieve compliance report {report_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve compliance report"
        )


@router.post("/export", 
             summary="Export audit data for archival or analysis")
async def export_audit_data(
    request: AuditExportRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(require_admin),
    audit_service: ComprehensiveAuditService = Depends(get_comprehensive_audit_service)
) -> Dict[str, Any]:
    """
    Export audit data in various formats for archival or external analysis.
    
    Supports JSON and CSV formats with optional encryption.
    Requires admin permissions.
    """
    
    try:
        # Validate export parameters
        if request.start_time and request.end_time:
            period_days = (request.end_time - request.start_time).days
            if period_days > 730:  # 2 years max
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Export period cannot exceed 2 years"
                )
        
        # Convert string enums
        event_types_enum = None
        if request.event_types:
            event_types_enum = [AuditEventType(et) for et in request.event_types]
        
        # Start export process
        export_path = await audit_service.export_audit_data(
            export_format=request.format,
            start_time=request.start_time,
            end_time=request.end_time,
            event_types=event_types_enum,
            user_id=request.user_id,
            encryption_key="export_key" if request.encrypt else None
        )
        
        # Log export request
        await audit_service.log_audit_event(
            event_type=AuditEventType.DATA_EXPORTED,
            user_id=current_user.id,
            user_email=current_user.email,
            resource_type="audit_export",
            action="export_audit_data",
            severity=AuditSeverity.HIGH,
            compliance_tags=["data_export", "audit_trail"],
            metadata={
                "export_format": request.format,
                "encrypted": request.encrypt,
                "include_sensitive": request.include_sensitive,
                "date_range": {
                    "start": request.start_time.isoformat() if request.start_time else None,
                    "end": request.end_time.isoformat() if request.end_time else None
                }
            }
        )
        
        return {
            "export_id": export_path.split("_")[-1].split(".")[0],
            "status": "completed",
            "download_url": f"/api/v1/audit/exports/{export_path}",
            "format": request.format,
            "encrypted": request.encrypt,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to export audit data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export audit data"
        )


@router.post("/retention/apply", 
             summary="Apply data retention policies")
async def apply_retention_policies(
    current_user = Depends(require_admin),
    audit_service: ComprehensiveAuditService = Depends(get_comprehensive_audit_service)
) -> Dict[str, Any]:
    """
    Manually trigger data retention policy application.
    
    This process runs automatically, but can be triggered manually for testing
    or immediate cleanup. Requires admin permissions.
    """
    
    try:
        # Apply retention policies
        retention_summary = await audit_service.apply_retention_policy()
        
        # Log retention policy application
        await audit_service.log_audit_event(
            event_type=AuditEventType.DATA_RETENTION_POLICY_APPLIED,
            user_id=current_user.id,
            user_email=current_user.email,
            action="manual_retention_policy_application",
            severity=AuditSeverity.MEDIUM,
            compliance_tags=["data_retention", "admin"],
            metadata=retention_summary
        )
        
        return {
            "status": "completed",
            "retention_summary": retention_summary,
            "applied_at": datetime.now(timezone.utc).isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to apply retention policies: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to apply retention policies"
        )


@router.get("/health", 
            summary="Audit system health check")
async def audit_health_check(
    audit_service: ComprehensiveAuditService = Depends(get_comprehensive_audit_service)
) -> Dict[str, Any]:
    """
    Comprehensive health check for the audit system.
    
    Returns status of audit service, storage, and key metrics.
    Available to all authenticated users.
    """
    
    try:
        health_status = await audit_service.health_check()
        
        return {
            "status": health_status.get("status", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                "audit_service": {
                    "status": "healthy" if health_status.get("initialized") else "unhealthy",
                    "initialized": health_status.get("initialized", False)
                },
                "storage": {
                    "status": "healthy" if health_status.get("redis_connected") else "degraded",
                    "redis_connected": health_status.get("redis_connected", False)
                },
                "buffer": {
                    "status": "healthy",
                    "buffer_size": health_status.get("buffer_size", 0)
                }
            },
            "metrics": {
                "audit_logging_functional": health_status.get("audit_logging_functional", False),
                "storage_available": health_status.get("storage_available", True)
            }
        }
    
    except Exception as e:
        logger.error(f"Audit health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        }


# Streaming endpoints for real-time monitoring

@router.get("/events/stream",
            summary="Stream real-time audit events")
async def stream_audit_events(
    current_user = Depends(require_audit_access),
    audit_service: ComprehensiveAuditService = Depends(get_comprehensive_audit_service)
):
    """
    Stream real-time audit events for monitoring dashboards.
    
    Returns Server-Sent Events (SSE) stream of audit events as they occur.
    Requires audit access permissions.
    """
    
    async def event_generator():
        """Generator for streaming audit events."""
        
        # Log streaming session start
        await audit_service.log_audit_event(
            event_type=AuditEventType.AUDIT_LOG_ACCESS,
            user_id=current_user.id,
            action="start_event_stream",
            severity=AuditSeverity.LOW
        )
        
        try:
            while True:
                # In production, this would connect to Redis streams or message queue
                # For now, simulate with periodic updates
                
                # Get recent events
                recent_events, _ = await audit_service.search_audit_events(
                    start_time=datetime.now(timezone.utc) - timedelta(minutes=1),
                    limit=10
                )
                
                for event in recent_events:
                    event_json = json.dumps(event, default=str)
                    yield f"data: {event_json}\n\n"
                
                # Wait before next batch
                await asyncio.sleep(5)
        
        except asyncio.CancelledError:
            # Log streaming session end
            await audit_service.log_audit_event(
                event_type=AuditEventType.AUDIT_LOG_ACCESS,
                user_id=current_user.id,
                action="end_event_stream",
                severity=AuditSeverity.LOW
            )
            return
    
    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )