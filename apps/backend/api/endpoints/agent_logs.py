"""
Agent Logs API Endpoints

REST API endpoints for accessing agent action logs, audit reports,
and quality assurance data for the multi-agent orchestration system.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum
import json
import io

import structlog
from services.agent_audit_service import (
    get_audit_service, AgentAuditService, AuditQuery, TimeFrame, AuditReportType,
    PerformanceMetrics, ErrorAnalysis, AgentUtilizationReport
)
from services.agent_action_logger import ActionType, LogLevel
from services.agent_orchestrator import AgentType
from api.dependencies.auth import get_current_user
from core.logging_config import log_security_event, log_performance_event

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/agent-logs",
    tags=["agent-logs"],
    dependencies=[Depends(get_current_user)]
)


class LogQueryRequest(BaseModel):
    """Request model for log queries."""
    # Time filtering
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    time_frame: Optional[TimeFrame] = None
    
    # Entity filtering
    agent_types: Optional[List[AgentType]] = None
    action_types: Optional[List[ActionType]] = None
    log_levels: Optional[List[LogLevel]] = None
    task_ids: Optional[List[str]] = None
    session_ids: Optional[List[str]] = None
    correlation_ids: Optional[List[str]] = None
    
    # Content filtering
    error_categories: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    min_duration_ms: Optional[float] = None
    max_duration_ms: Optional[float] = None
    min_confidence_score: Optional[float] = None
    max_confidence_score: Optional[float] = None
    
    # Quality filtering
    has_errors: Optional[bool] = None
    has_retries: Optional[bool] = None
    has_validation_results: Optional[bool] = None
    min_hallucination_score: Optional[float] = None
    max_hallucination_score: Optional[float] = None
    
    # Text search
    search_text: Optional[str] = None
    search_fields: Optional[List[str]] = None
    
    # Pagination and sorting
    limit: int = Field(default=100, le=1000)
    offset: int = Field(default=0, ge=0)
    sort_by: str = "timestamp"
    sort_order: str = Field(default="desc", regex="^(asc|desc)$")
    
    # Aggregation options
    group_by: Optional[List[str]] = None
    include_aggregations: bool = False


class ReportRequest(BaseModel):
    """Request model for generating reports."""
    report_type: AuditReportType
    time_frame: TimeFrame = TimeFrame.LAST_24_HOURS
    agent_types: Optional[List[AgentType]] = None
    include_details: bool = False
    export_format: str = Field(default="json", regex="^(json|csv|xlsx)$")


class LogsResponse(BaseModel):
    """Response model for log queries."""
    logs: List[Dict[str, Any]]
    total_count: int
    filtered_count: int
    aggregations: Optional[Dict[str, Any]] = None
    query_time_ms: float
    cache_hit: bool = False
    insights: Optional[Dict[str, Any]] = None


@router.post("/query", response_model=LogsResponse)
async def query_logs(
    request: LogQueryRequest,
    background_tasks: BackgroundTasks,
    audit_service: AgentAuditService = Depends(get_audit_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Query agent logs with advanced filtering and aggregation.
    
    Supports complex queries across all agent actions, decisions,
    and outcomes with performance optimization and caching.
    """
    try:
        # Convert request to AuditQuery
        query = AuditQuery(
            start_time=request.start_time,
            end_time=request.end_time,
            time_frame=request.time_frame,
            agent_types=request.agent_types,
            action_types=request.action_types,
            log_levels=request.log_levels,
            task_ids=request.task_ids,
            session_ids=request.session_ids,
            correlation_ids=request.correlation_ids,
            error_categories=request.error_categories,
            tags=request.tags,
            min_duration_ms=request.min_duration_ms,
            max_duration_ms=request.max_duration_ms,
            min_confidence_score=request.min_confidence_score,
            max_confidence_score=request.max_confidence_score,
            has_errors=request.has_errors,
            has_retries=request.has_retries,
            has_validation_results=request.has_validation_results,
            min_hallucination_score=request.min_hallucination_score,
            max_hallucination_score=request.max_hallucination_score,
            search_text=request.search_text,
            search_fields=request.search_fields,
            limit=request.limit,
            offset=request.offset,
            sort_by=request.sort_by,
            sort_order=request.sort_order,
            group_by=request.group_by,
            include_aggregations=request.include_aggregations
        )
        
        # Execute query
        result = await audit_service.query_logs(query)
        
        # Log audit access
        background_tasks.add_task(
            log_security_event,
            "log_query_executed",
            user_id=current_user.get("user_id"),
            details={
                "query_filters": len([f for f in [
                    request.agent_types, request.action_types, request.log_levels,
                    request.task_ids, request.session_ids, request.search_text
                ] if f]),
                "result_count": result.filtered_count,
                "time_frame": request.time_frame.value if request.time_frame else "custom",
                "includes_sensitive": request.has_errors or request.search_text
            }
        )
        
        # Log performance
        background_tasks.add_task(
            log_performance_event,
            "log_query",
            result.query_time_ms,
            user_id=current_user.get("user_id"),
            metadata={
                "cache_hit": result.cache_hit,
                "result_count": result.filtered_count,
                "aggregations_included": request.include_aggregations
            }
        )
        
        return LogsResponse(
            logs=result.logs,
            total_count=result.total_count,
            filtered_count=result.filtered_count,
            aggregations=result.aggregations,
            query_time_ms=result.query_time_ms,
            cache_hit=result.cache_hit,
            insights=result.insights
        )
        
    except Exception as e:
        logger.error(f"Failed to query logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")


@router.get("/task/{task_id}/audit-trail")
async def get_task_audit_trail(
    task_id: str,
    include_related: bool = Query(default=True),
    audit_service: AgentAuditService = Depends(get_audit_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get complete audit trail for a specific task.
    
    Includes all related tasks, decisions, and validation steps
    for comprehensive traceability.
    """
    try:
        audit_trail = await audit_service.get_task_audit_trail(
            task_id=task_id,
            include_related_tasks=include_related
        )
        
        # Log audit access
        await log_security_event(
            "task_audit_trail_accessed",
            user_id=current_user.get("user_id"),
            details={
                "task_id": task_id,
                "include_related": include_related,
                "events_count": len(audit_trail)
            }
        )
        
        return {
            "task_id": task_id,
            "audit_trail": audit_trail,
            "total_events": len(audit_trail),
            "include_related_tasks": include_related,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get task audit trail: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audit trail retrieval failed: {str(e)}")


@router.get("/session/{session_id}")
async def get_session_logs(
    session_id: str,
    include_hierarchy: bool = Query(default=True),
    audit_service: AgentAuditService = Depends(get_audit_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get all logs for a specific session.
    
    Includes session hierarchy and related subsessions
    for complete session analysis.
    """
    try:
        # Get audit service (which has agent logger)
        agent_logger = audit_service.agent_logger
        session_logs = await agent_logger.get_session_logs(
            session_id=session_id,
            include_hierarchy=include_hierarchy
        )
        
        # Log audit access
        await log_security_event(
            "session_logs_accessed",
            user_id=current_user.get("user_id"),
            details={
                "session_id": session_id,
                "include_hierarchy": include_hierarchy,
                "logs_count": len(session_logs)
            }
        )
        
        return {
            "session_id": session_id,
            "logs": session_logs,
            "total_logs": len(session_logs),
            "include_hierarchy": include_hierarchy,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get session logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Session logs retrieval failed: {str(e)}")


@router.post("/reports/generate")
async def generate_report(
    request: ReportRequest,
    background_tasks: BackgroundTasks,
    audit_service: AgentAuditService = Depends(get_audit_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Generate comprehensive audit reports.
    
    Supports multiple report types including performance analysis,
    error analysis, agent utilization, and compliance reports.
    """
    try:
        report_data = None
        
        # Generate report based on type
        if request.report_type == AuditReportType.PERFORMANCE_SUMMARY:
            report_data = await audit_service.generate_performance_report(
                time_frame=request.time_frame,
                agent_types=request.agent_types
            )
            
        elif request.report_type == AuditReportType.ERROR_ANALYSIS:
            report_data = await audit_service.generate_error_analysis(
                time_frame=request.time_frame,
                agent_types=request.agent_types
            )
            
        elif request.report_type == AuditReportType.AGENT_UTILIZATION:
            report_data = await audit_service.generate_agent_utilization_report(
                time_frame=request.time_frame
            )
            
        elif request.report_type == AuditReportType.COMPLIANCE_REPORT:
            report_data = await audit_service.generate_compliance_report(
                time_frame=request.time_frame,
                include_sensitive_operations=current_user.get("role") == "admin"
            )
            
        else:
            raise HTTPException(status_code=400, detail=f"Report type {request.report_type.value} not implemented")
        
        # Convert to dict for JSON serialization
        if hasattr(report_data, '__dict__'):
            report_dict = report_data.__dict__
        else:
            report_dict = report_data
        
        # Log report generation
        background_tasks.add_task(
            log_security_event,
            "audit_report_generated",
            user_id=current_user.get("user_id"),
            details={
                "report_type": request.report_type.value,
                "time_frame": request.time_frame.value,
                "export_format": request.export_format,
                "agent_types_count": len(request.agent_types) if request.agent_types else 0
            }
        )
        
        response_data = {
            "report_type": request.report_type.value,
            "time_frame": request.time_frame.value,
            "generated_at": datetime.utcnow().isoformat(),
            "data": report_dict,
            "metadata": {
                "agent_types_filtered": len(request.agent_types) if request.agent_types else 0,
                "include_details": request.include_details
            }
        }
        
        # Handle different export formats
        if request.export_format == "json":
            return response_data
        elif request.export_format == "csv":
            csv_content = _convert_to_csv(report_dict, request.report_type)
            return StreamingResponse(
                io.StringIO(csv_content),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={request.report_type.value}_{request.time_frame.value}.csv"}
            )
        else:
            raise HTTPException(status_code=400, detail=f"Export format {request.export_format} not supported")
        
    except Exception as e:
        logger.error(f"Failed to generate report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.get("/agents/{agent_type}/performance")
async def get_agent_performance(
    agent_type: AgentType,
    time_frame: TimeFrame = Query(default=TimeFrame.LAST_WEEK),
    audit_service: AgentAuditService = Depends(get_audit_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get detailed performance insights for a specific agent type.
    
    Provides comprehensive analysis including execution patterns,
    error rates, and optimization recommendations.
    """
    try:
        insights = await audit_service.get_agent_performance_insights(
            agent_type=agent_type,
            time_frame=time_frame
        )
        
        # Log performance access
        await log_security_event(
            "agent_performance_accessed",
            user_id=current_user.get("user_id"),
            details={
                "agent_type": agent_type.value,
                "time_frame": time_frame.value
            }
        )
        
        return {
            "agent_type": agent_type.value,
            "time_frame": time_frame.value,
            "insights": insights,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get agent performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Performance analysis failed: {str(e)}")


@router.get("/search")
async def search_logs(
    query: str = Query(..., min_length=3, max_length=500),
    fields: Optional[List[str]] = Query(default=None),
    time_frame: TimeFrame = Query(default=TimeFrame.LAST_24_HOURS),
    limit: int = Query(default=100, le=500),
    audit_service: AgentAuditService = Depends(get_audit_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Search logs by text content.
    
    Performs full-text search across specified fields
    with intelligent filtering and ranking.
    """
    try:
        results = await audit_service.search_logs_by_text(
            search_text=query,
            fields=fields,
            time_frame=time_frame,
            limit=limit
        )
        
        # Log search access
        await log_security_event(
            "log_text_search",
            user_id=current_user.get("user_id"),
            details={
                "query_length": len(query),
                "fields_count": len(fields) if fields else 0,
                "results_count": len(results),
                "time_frame": time_frame.value
            }
        )
        
        return {
            "query": query,
            "fields": fields,
            "time_frame": time_frame.value,
            "results": results,
            "total_results": len(results),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to search logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Log search failed: {str(e)}")


@router.get("/statistics")
async def get_logging_statistics(
    audit_service: AgentAuditService = Depends(get_audit_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get comprehensive logging system statistics.
    
    Provides operational metrics, performance statistics,
    and system health indicators.
    """
    try:
        # Get agent logger statistics
        agent_logger = audit_service.agent_logger
        stats = await agent_logger.get_statistics()
        
        # Add audit service statistics
        audit_stats = {
            "audit_service_initialized": audit_service.is_initialized,
            "cache_enabled": audit_service.redis_client is not None,
            "cache_ttl_seconds": audit_service.cache_ttl
        }
        
        # Log statistics access
        await log_security_event(
            "logging_statistics_accessed",
            user_id=current_user.get("user_id"),
            details={"access_level": "system_statistics"}
        )
        
        return {
            "logging_system": stats,
            "audit_service": audit_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Statistics retrieval failed: {str(e)}")


@router.get("/health")
async def health_check(
    audit_service: AgentAuditService = Depends(get_audit_service)
):
    """
    Health check endpoint for logging system.
    
    Verifies system components are operational
    and returns status information.
    """
    try:
        health_status = {
            "status": "healthy",
            "audit_service_initialized": audit_service.is_initialized,
            "agent_logger_available": audit_service.agent_logger is not None,
            "redis_available": audit_service.redis_client is not None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check agent logger health if available
        if audit_service.agent_logger:
            logger_stats = await audit_service.agent_logger.get_statistics()
            health_status["agent_logger_stats"] = {
                "logs_written": logger_stats.get("logs_written", 0),
                "logs_queued": logger_stats.get("queue_size", 0),
                "errors": logger_stats.get("errors", 0),
                "uptime_seconds": logger_stats.get("uptime_seconds", 0)
            }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


def _convert_to_csv(data: Dict[str, Any], report_type: AuditReportType) -> str:
    """Convert report data to CSV format."""
    
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    if report_type == AuditReportType.PERFORMANCE_SUMMARY:
        writer.writerow(["Metric", "Value"])
        if isinstance(data, dict):
            for key, value in data.items():
                writer.writerow([key, str(value)])
    
    elif report_type == AuditReportType.ERROR_ANALYSIS:
        writer.writerow(["Error Category", "Count"])
        if hasattr(data, 'error_categories'):
            for category, count in data.error_categories.items():
                writer.writerow([category, count])
    
    elif report_type == AuditReportType.AGENT_UTILIZATION:
        writer.writerow(["Agent Type", "Executions", "Success Rate", "Avg Duration"])
        if hasattr(data, 'agent_usage_stats'):
            for agent_type, stats in data.agent_usage_stats.items():
                writer.writerow([
                    agent_type,
                    stats.get('executions', 0),
                    f"{stats.get('success_rate', 0):.2f}%",
                    f"{stats.get('avg_duration', 0):.2f}ms"
                ])
    
    return output.getvalue()


# Add router to main application
def include_router(app):
    """Include agent logs router in FastAPI app."""
    app.include_router(router)