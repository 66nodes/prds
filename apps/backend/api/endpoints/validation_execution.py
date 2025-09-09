"""
Validation Execution API Endpoints

API endpoints for validated parallel agent execution with comprehensive
GraphRAG validation and quality metrics.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio

from services.validated_parallel_executor import (
    ValidatedParallelExecutor,
    get_validated_executor,
    execute_with_validation
)
from services.graphrag_validation_middleware import (
    ValidationCheckpoint,
    ValidationStage,
    get_validation_middleware
)
from services.graphrag.validation_pipeline import ValidationLevel
from services.agent_orchestrator import AgentType, AgentTask, WorkflowContext
from services.enhanced_parallel_executor import PriorityLevel
from core.auth import require_auth

router = APIRouter(prefix="/api/v1/validation", tags=["validation_execution"])


# Request Models
class ValidationExecutionRequest(BaseModel):
    """Request model for validated execution."""
    tasks: List[Dict[str, Any]] = Field(..., description="List of agent tasks to execute")
    workflow_context: Dict[str, Any] = Field(..., description="Workflow execution context")
    priority: PriorityLevel = Field(PriorityLevel.NORMAL, description="Execution priority level")
    validation_level: ValidationLevel = Field(ValidationLevel.STANDARD, description="Validation strictness level")
    enable_corrections: bool = Field(True, description="Enable automatic corrections")
    max_validation_retries: int = Field(3, ge=1, le=10, description="Maximum validation retry attempts")
    timeout_seconds: int = Field(300, ge=30, le=3600, description="Execution timeout in seconds")


class AgentValidationConfig(BaseModel):
    """Configuration for agent-specific validation."""
    agent_type: AgentType = Field(..., description="Agent type to configure")
    validation_level: ValidationLevel = Field(ValidationLevel.STANDARD, description="Validation level")
    confidence_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Minimum confidence threshold")
    hallucination_threshold: float = Field(0.02, ge=0.0, le=1.0, description="Maximum hallucination rate")
    max_iterations: int = Field(3, ge=1, le=10, description="Maximum validation iterations")
    enable_corrections: bool = Field(True, description="Enable automatic corrections")


class ValidationMetricsRequest(BaseModel):
    """Request model for validation metrics query."""
    start_date: Optional[datetime] = Field(None, description="Start date for metrics")
    end_date: Optional[datetime] = Field(None, description="End date for metrics")
    agent_types: Optional[List[AgentType]] = Field(None, description="Filter by agent types")
    validation_levels: Optional[List[ValidationLevel]] = Field(None, description="Filter by validation levels")


# Response Models
class ValidationExecutionResponse(BaseModel):
    """Response model for validated execution results."""
    execution_id: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    execution_time_ms: float
    validation_time_ms: float
    
    # Validation metrics
    validation_success_rate: float
    avg_confidence_score: float
    avg_hallucination_rate: float
    corrections_applied: int
    quality_improvements: int
    
    # Detailed results
    task_results: Dict[str, Any]
    validation_analytics: Dict[str, Any]


class ValidationStatusResponse(BaseModel):
    """Response model for validation system status."""
    status: str
    validation_enabled: bool
    validation_level: str
    
    # System health
    executor_health: Dict[str, Any]
    middleware_health: Dict[str, Any]
    pipeline_health: Dict[str, Any]
    
    # Performance metrics
    total_validations: int
    success_rate: float
    avg_processing_time_ms: float
    active_validations: int
    
    # Quality metrics
    avg_confidence_score: float
    hallucination_prevention_rate: float
    corrections_applied_total: int


class ValidationMetricsResponse(BaseModel):
    """Response model for validation metrics."""
    period_start: datetime
    period_end: datetime
    total_validations: int
    
    # Success metrics
    successful_validations: int
    failed_validations: int
    success_rate: float
    
    # Quality metrics
    avg_confidence_score: float
    confidence_distribution: Dict[str, int]
    avg_hallucination_rate: float
    hallucination_prevention_count: int
    
    # Performance metrics
    avg_processing_time_ms: float
    processing_time_distribution: Dict[str, int]
    corrections_applied: int
    quality_improvements: int
    
    # Agent-specific metrics
    agent_performance: Dict[str, Dict[str, Any]]
    validation_level_distribution: Dict[str, int]


@router.post("/execute", response_model=ValidationExecutionResponse)
async def execute_with_validation_endpoint(
    request: ValidationExecutionRequest,
    background_tasks: BackgroundTasks,
    auth_user = Depends(require_auth)
):
    """Execute agent tasks with comprehensive GraphRAG validation."""
    
    try:
        # Convert request to internal format
        tasks = []
        for task_data in request.tasks:
            task = AgentTask(
                task_id=task_data.get("task_id", f"task_{len(tasks)}"),
                agent_type=AgentType(task_data["agent_type"]),
                input_data=task_data.get("input_data", {}),
                dependencies=task_data.get("dependencies", []),
                metadata=task_data.get("metadata", {})
            )
            tasks.append(task)
        
        # Create workflow context
        workflow = WorkflowContext(
            workflow_id=request.workflow_context.get("workflow_id", f"workflow_{int(datetime.utcnow().timestamp())}"),
            context_data=request.workflow_context.get("context_data", {}),
            metadata=request.workflow_context.get("metadata", {})
        )
        
        # Execute with validation
        start_time = datetime.utcnow()
        result = await execute_with_validation(
            tasks=tasks,
            workflow=workflow,
            priority=request.priority,
            validation_level=request.validation_level,
            enable_corrections=request.enable_corrections
        )
        end_time = datetime.utcnow()
        
        execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Extract metrics from result
        validation_analytics = result.get("validation_analytics", {})
        task_results = result.get("results", {})
        
        # Count completed/failed tasks
        completed_tasks = sum(
            1 for task_result in task_results.values()
            if task_result.get("status") == "completed"
        )
        failed_tasks = len(task_results) - completed_tasks
        
        # Calculate validation success rate
        total_validated = validation_analytics.get("total_tasks_validated", 0)
        passed_validation = validation_analytics.get("passed_validation", 0)
        validation_success_rate = (passed_validation / total_validated * 100) if total_validated > 0 else 0
        
        return ValidationExecutionResponse(
            execution_id=workflow.workflow_id,
            total_tasks=len(tasks),
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            execution_time_ms=execution_time_ms,
            validation_time_ms=validation_analytics.get("validation_time_ms", 0),
            validation_success_rate=validation_success_rate,
            avg_confidence_score=validation_analytics.get("avg_confidence_score", 0),
            avg_hallucination_rate=validation_analytics.get("avg_hallucination_rate", 0),
            corrections_applied=validation_analytics.get("corrections_applied", 0),
            quality_improvements=validation_analytics.get("quality_improvements", 0),
            task_results=task_results,
            validation_analytics=validation_analytics
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Validation execution failed: {str(e)}"
        )


@router.get("/status", response_model=ValidationStatusResponse)
async def get_validation_status(
    auth_user = Depends(require_auth)
):
    """Get comprehensive validation system status."""
    
    try:
        # Get executor status
        executor = await get_validated_executor()
        executor_status = await executor.get_enhanced_execution_status()
        
        # Get middleware status
        middleware = await get_validation_middleware()
        middleware_health = await middleware.health_check()
        middleware_stats = await middleware.get_validation_statistics()
        
        return ValidationStatusResponse(
            status=executor_status.get("status", "unknown"),
            validation_enabled=executor_status.get("validation_enabled", False),
            validation_level=executor_status.get("validation_level", "unknown"),
            executor_health={
                "active_tasks": executor_status.get("active_tasks", {}),
                "optimal_concurrency": executor_status.get("optimal_concurrency", 0),
                "resource_usage": executor_status.get("resource_usage", {})
            },
            middleware_health=middleware_health,
            pipeline_health=middleware_stats.get("pipeline_stats", {}),
            total_validations=executor_status.get("validation_analytics", {}).get("total_validations", 0),
            success_rate=executor_status.get("validation_success_rate", 0),
            avg_processing_time_ms=executor_status.get("validation_analytics", {}).get("avg_validation_time_ms", 0),
            active_validations=middleware_health.get("middleware_metrics", {}).get("active_validations", 0),
            avg_confidence_score=executor_status.get("validation_analytics", {}).get("avg_confidence_score", 0),
            hallucination_prevention_rate=executor_status.get("validation_analytics", {}).get("hallucinations_prevented", 0),
            corrections_applied_total=executor_status.get("validation_analytics", {}).get("corrections_applied", 0)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get validation status: {str(e)}"
        )


@router.post("/configure-agent")
async def configure_agent_validation(
    config: AgentValidationConfig,
    auth_user = Depends(require_auth)
):
    """Configure validation settings for specific agent type."""
    
    try:
        middleware = await get_validation_middleware()
        
        # Create validation checkpoint
        checkpoint = ValidationCheckpoint(
            stage=ValidationStage.POST_EXECUTION,
            validation_level=config.validation_level,
            confidence_threshold=config.confidence_threshold,
            hallucination_threshold=config.hallucination_threshold,
            max_iterations=config.max_iterations,
            enable_corrections=config.enable_corrections
        )
        
        # Configure agent validation
        middleware.configure_agent_validation(config.agent_type, checkpoint)
        
        return {
            "message": f"Validation configured for {config.agent_type.value}",
            "configuration": {
                "validation_level": config.validation_level.value,
                "confidence_threshold": config.confidence_threshold,
                "hallucination_threshold": config.hallucination_threshold,
                "max_iterations": config.max_iterations,
                "enable_corrections": config.enable_corrections
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to configure agent validation: {str(e)}"
        )


@router.get("/metrics", response_model=ValidationMetricsResponse)
async def get_validation_metrics(
    start_date: Optional[datetime] = Query(None, description="Start date for metrics"),
    end_date: Optional[datetime] = Query(None, description="End date for metrics"),
    agent_types: Optional[str] = Query(None, description="Comma-separated agent types"),
    auth_user = Depends(require_auth)
):
    """Get detailed validation metrics and analytics."""
    
    try:
        # Get current metrics from middleware and executor
        middleware = await get_validation_middleware()
        middleware_stats = await middleware.get_validation_statistics()
        
        executor = await get_validated_executor()
        executor_status = await executor.get_enhanced_execution_status()
        validation_analytics = executor_status.get("validation_analytics", {})
        
        # Set default date range if not provided
        end_date = end_date or datetime.utcnow()
        start_date = start_date or datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Parse agent types filter
        agent_type_filter = []
        if agent_types:
            agent_type_filter = [t.strip() for t in agent_types.split(",")]
        
        # Calculate metrics
        total_validations = validation_analytics.get("total_validations", 0)
        successful_validations = validation_analytics.get("successful_validations", 0)
        failed_validations = validation_analytics.get("failed_validations", 0)
        success_rate = (successful_validations / total_validations * 100) if total_validations > 0 else 0
        
        # Mock distribution data (in a real implementation, this would come from a time-series database)
        confidence_distribution = {
            "0.0-0.2": 0,
            "0.2-0.4": failed_validations // 4 if failed_validations > 0 else 0,
            "0.4-0.6": failed_validations // 2 if failed_validations > 0 else 0,
            "0.6-0.8": successful_validations // 3 if successful_validations > 0 else 0,
            "0.8-1.0": successful_validations // 2 if successful_validations > 0 else 0
        }
        
        processing_time_distribution = {
            "0-1000ms": total_validations // 2,
            "1000-5000ms": total_validations // 3,
            "5000-10000ms": total_validations // 6,
            "10000ms+": total_validations // 12
        }
        
        validation_level_distribution = {
            "basic": total_validations // 4,
            "standard": total_validations // 2,
            "strict": total_validations // 4
        }
        
        # Agent performance metrics
        agent_configurations = middleware_stats.get("agent_configurations", {})
        agent_performance = {}
        for agent_type, config in agent_configurations.items():
            agent_performance[agent_type] = {
                "validation_level": config.get("validation_level", "standard"),
                "confidence_threshold": config.get("confidence_threshold", 0.8),
                "hallucination_threshold": config.get("hallucination_threshold", 0.02),
                "success_rate": success_rate,  # Would be agent-specific in real implementation
                "avg_processing_time_ms": validation_analytics.get("avg_validation_time_ms", 0)
            }
        
        return ValidationMetricsResponse(
            period_start=start_date,
            period_end=end_date,
            total_validations=total_validations,
            successful_validations=successful_validations,
            failed_validations=failed_validations,
            success_rate=success_rate,
            avg_confidence_score=validation_analytics.get("avg_confidence_score", 0),
            confidence_distribution=confidence_distribution,
            avg_hallucination_rate=0.01,  # Would be calculated from actual data
            hallucination_prevention_count=validation_analytics.get("hallucinations_prevented", 0),
            avg_processing_time_ms=validation_analytics.get("avg_validation_time_ms", 0),
            processing_time_distribution=processing_time_distribution,
            corrections_applied=validation_analytics.get("corrections_applied", 0),
            quality_improvements=int(total_validations * validation_analytics.get("quality_improvement_rate", 0)),
            agent_performance=agent_performance,
            validation_level_distribution=validation_level_distribution
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get validation metrics: {str(e)}"
        )


@router.post("/disable")
async def disable_validation(
    auth_user = Depends(require_auth)
):
    """Disable validation for performance-critical operations."""
    
    try:
        executor = await get_validated_executor()
        executor.disable_validation()
        
        return {
            "message": "Validation disabled",
            "warning": "Hallucination prevention is now unavailable"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to disable validation: {str(e)}"
        )


@router.post("/enable")
async def enable_validation(
    validation_level: ValidationLevel = ValidationLevel.STANDARD,
    auth_user = Depends(require_auth)
):
    """Enable validation with specified level."""
    
    try:
        executor = await get_validated_executor()
        executor.enable_validation_mode(validation_level)
        
        return {
            "message": f"Validation enabled with level: {validation_level.value}",
            "configuration": {
                "validation_level": validation_level.value,
                "hallucination_prevention": True,
                "correction_enabled": True
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to enable validation: {str(e)}"
        )


@router.get("/health")
async def validation_system_health():
    """Get validation system health status."""
    
    try:
        # Check all components
        executor = await get_validated_executor()
        executor_health = await executor.get_enhanced_execution_status()
        
        middleware = await get_validation_middleware()
        middleware_health = await middleware.health_check()
        
        # Determine overall health
        executor_status = executor_health.get("status", "unknown")
        middleware_status = middleware_health.get("status", "unknown")
        
        overall_health = "healthy"
        if executor_status == "degraded" or middleware_status == "degraded":
            overall_health = "degraded"
        elif executor_status == "unhealthy" or middleware_status == "unhealthy":
            overall_health = "unhealthy"
        
        return {
            "status": overall_health,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "validated_executor": {
                    "status": executor_status,
                    "validation_enabled": executor_health.get("validation_enabled", False),
                    "active_tasks": len(executor_health.get("active_tasks", {}))
                },
                "validation_middleware": {
                    "status": middleware_status,
                    "active_validations": middleware_health.get("middleware_metrics", {}).get("active_validations", 0),
                    "total_validations": middleware_health.get("middleware_metrics", {}).get("total_validations", 0)
                }
            },
            "performance": {
                "avg_validation_time_ms": executor_health.get("validation_analytics", {}).get("avg_validation_time_ms", 0),
                "success_rate": executor_health.get("validation_success_rate", 0),
                "hallucination_prevention_active": executor_health.get("validation_enabled", False)
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }