"""
Enhanced Parallel Execution API Endpoints

REST API endpoints for managing and monitoring enhanced parallel agent execution,
including real-time status, analytics, and configuration management.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
import json
import asyncio

from services.agent_orchestrator import get_orchestrator, AgentTask, WorkflowContext
from services.enhanced_parallel_executor import (
    get_enhanced_executor,
    PriorityLevel,
    ExecutionStatus,
    LoadBalancingStrategy
)
from core.auth import require_auth
from core.error_handlers import ValidationError

router = APIRouter(prefix="/api/v1/execution", tags=["enhanced-execution"])


# Request/Response Models

class TaskExecutionRequest(BaseModel):
    """Request model for task execution."""
    workflow_id: str = Field(..., description="Unique workflow identifier")
    tasks: List[Dict[str, Any]] = Field(..., description="List of agent tasks to execute")
    priority: PriorityLevel = Field(PriorityLevel.NORMAL, description="Execution priority")
    timeout: Optional[float] = Field(None, description="Execution timeout in seconds")
    use_enhanced_executor: bool = Field(True, description="Whether to use enhanced executor")
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Workflow context data")


class ExecutionResponse(BaseModel):
    """Response model for task execution."""
    workflow_id: str
    status: str
    results: Dict[str, Any]
    analytics: Dict[str, Any]
    resource_usage: Dict[str, Any]
    circuit_breaker_status: Dict[str, Any]
    execution_time_ms: int
    started_at: datetime
    completed_at: Optional[datetime] = None


class ExecutorStatusResponse(BaseModel):
    """Response model for executor status."""
    active_tasks: Dict[str, Any]
    resource_usage: Dict[str, Any]
    circuit_breakers: Dict[str, Any]
    analytics: Dict[str, Any]
    optimal_concurrency: int
    agent_pools: Dict[str, Any]
    timestamp: datetime


class ExecutorConfigRequest(BaseModel):
    """Request model for executor configuration updates."""
    max_concurrent_tasks: Optional[int] = Field(None, ge=1, le=50)
    load_balancing_strategy: Optional[LoadBalancingStrategy] = None
    enable_circuit_breakers: Optional[bool] = None
    enable_agent_pooling: Optional[bool] = None
    resource_monitoring_interval: Optional[float] = Field(None, ge=1.0, le=60.0)


# API Endpoints

@router.post("/execute", response_model=ExecutionResponse)
async def execute_tasks_enhanced(
    request: TaskExecutionRequest,
    background_tasks: BackgroundTasks,
    user = Depends(require_auth)
) -> ExecutionResponse:
    """
    Execute agent tasks using the enhanced parallel executor.
    
    Provides advanced features including:
    - Dynamic load balancing
    - Adaptive concurrency control
    - Circuit breaker patterns
    - Agent pooling
    - Real-time resource monitoring
    """
    try:
        # Get orchestrator and executor
        orchestrator = await get_orchestrator()
        
        # Create workflow context
        workflow = WorkflowContext(
            workflow_id=request.workflow_id,
            context_type="api_request",
            input_data=request.context_data,
            metadata={
                "user_id": str(user.id) if hasattr(user, 'id') else "anonymous",
                "priority": request.priority.value,
                "api_endpoint": "enhanced_execution",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Convert task dictionaries to AgentTask objects
        agent_tasks = []
        for task_data in request.tasks:
            try:
                agent_task = AgentTask(
                    task_id=task_data.get("task_id", f"task_{len(agent_tasks)}"),
                    agent_type=task_data["agent_type"],
                    input_data=task_data.get("input_data", {}),
                    dependencies=task_data.get("dependencies", []),
                    priority=task_data.get("priority", "normal"),
                    estimated_execution_time=timedelta(
                        seconds=task_data.get("estimated_duration_seconds", 30)
                    )
                )
                agent_tasks.append(agent_task)
            except Exception as e:
                raise ValidationError(f"Invalid task data: {str(e)}")
        
        if not agent_tasks:
            raise HTTPException(status_code=400, detail="No valid tasks provided")
        
        # Execute tasks
        start_time = datetime.utcnow()
        
        result = await orchestrator.execute_tasks_enhanced(
            tasks=agent_tasks,
            workflow=workflow,
            priority=request.priority,
            timeout=request.timeout,
            use_enhanced_executor=request.use_enhanced_executor
        )
        
        end_time = datetime.utcnow()
        execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Schedule background cleanup if needed
        background_tasks.add_task(_cleanup_completed_workflow, workflow.workflow_id)
        
        return ExecutionResponse(
            workflow_id=request.workflow_id,
            status="completed" if result.get("results") else "failed",
            results=result.get("results", {}),
            analytics=result.get("analytics", {}),
            resource_usage=result.get("resource_usage", {}),
            circuit_breaker_status=result.get("circuit_breaker_status", {}),
            execution_time_ms=execution_time_ms,
            started_at=start_time,
            completed_at=end_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Execution failed: {str(e)}"
        )


@router.get("/status", response_model=ExecutorStatusResponse)
async def get_executor_status(
    user = Depends(require_auth)
) -> ExecutorStatusResponse:
    """
    Get comprehensive status of the enhanced parallel executor.
    
    Returns real-time information about:
    - Active tasks and their progress
    - Resource usage and performance metrics
    - Circuit breaker states
    - Agent pool statistics
    - Execution analytics
    """
    try:
        executor = await get_enhanced_executor()
        status = await executor.get_execution_status()
        
        return ExecutorStatusResponse(
            active_tasks=status.get("active_tasks", {}),
            resource_usage=status.get("resource_usage", {}),
            circuit_breakers=status.get("circuit_breakers", {}),
            analytics=status.get("analytics", {}),
            optimal_concurrency=status.get("optimal_concurrency", 0),
            agent_pools=status.get("agent_pools", {}),
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get executor status: {str(e)}"
        )


@router.get("/health")
async def get_executor_health() -> JSONResponse:
    """
    Get health status of the enhanced parallel executor.
    
    Returns basic health information including:
    - Executor availability
    - Current load
    - System resource status
    """
    try:
        orchestrator = await get_orchestrator()
        health_status = await orchestrator.health_check()
        
        enhanced_executor_info = health_status.get("enhanced_executor", {})
        
        status_code = 200 if enhanced_executor_info.get("available", False) else 503
        
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "healthy" if status_code == 200 else "degraded",
                "enhanced_executor_available": enhanced_executor_info.get("available", False),
                "active_tasks": len(enhanced_executor_info.get("active_tasks", {})),
                "optimal_concurrency": enhanced_executor_info.get("optimal_concurrency", 0),
                "resource_usage": enhanced_executor_info.get("resource_usage", {}),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.get("/metrics")
async def get_execution_metrics(
    include_history: bool = Query(False, description="Include historical metrics"),
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    user = Depends(require_auth)
) -> JSONResponse:
    """
    Get detailed execution metrics and analytics.
    
    Provides comprehensive metrics including:
    - Performance statistics
    - Resource efficiency metrics
    - Agent-specific performance data
    - Historical trends (if requested)
    """
    try:
        executor = await get_enhanced_executor()
        status = await executor.get_execution_status()
        
        metrics = {
            "analytics": status.get("analytics", {}),
            "resource_usage": status.get("resource_usage", {}),
            "agent_pools": status.get("agent_pools", {}),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Filter by agent type if specified
        if agent_type and "agent_pools" in metrics:
            filtered_pools = {
                k: v for k, v in metrics["agent_pools"].items()
                if k.lower() == agent_type.lower()
            }
            metrics["agent_pools"] = filtered_pools
        
        # Include historical data if requested
        if include_history:
            # In a production system, this would fetch from persistent storage
            metrics["historical_data"] = {
                "note": "Historical metrics would be retrieved from persistent storage",
                "available_periods": ["1h", "24h", "7d", "30d"]
            }
        
        return JSONResponse(content=metrics)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get execution metrics: {str(e)}"
        )


@router.post("/configure")
async def configure_executor(
    config: ExecutorConfigRequest,
    user = Depends(require_auth)
) -> JSONResponse:
    """
    Update executor configuration parameters.
    
    Allows runtime configuration of:
    - Maximum concurrent tasks
    - Load balancing strategy
    - Circuit breaker settings
    - Agent pooling settings
    - Resource monitoring interval
    
    Note: Some configuration changes may require executor restart.
    """
    try:
        executor = await get_enhanced_executor()
        
        updates = {}
        
        # Update configuration parameters
        if config.max_concurrent_tasks is not None:
            executor.max_concurrent_tasks = config.max_concurrent_tasks
            executor.optimal_concurrency = config.max_concurrent_tasks
            # Update semaphore
            executor.semaphore = asyncio.Semaphore(config.max_concurrent_tasks)
            updates["max_concurrent_tasks"] = config.max_concurrent_tasks
        
        if config.load_balancing_strategy is not None:
            executor.load_balancing_strategy = config.load_balancing_strategy
            updates["load_balancing_strategy"] = config.load_balancing_strategy.value
        
        if config.enable_circuit_breakers is not None:
            executor.enable_circuit_breakers = config.enable_circuit_breakers
            updates["enable_circuit_breakers"] = config.enable_circuit_breakers
        
        if config.enable_agent_pooling is not None:
            executor.enable_agent_pooling = config.enable_agent_pooling
            updates["enable_agent_pooling"] = config.enable_agent_pooling
        
        if config.resource_monitoring_interval is not None:
            executor.resource_monitoring_interval = config.resource_monitoring_interval
            updates["resource_monitoring_interval"] = config.resource_monitoring_interval
        
        return JSONResponse(content={
            "status": "success",
            "message": "Executor configuration updated",
            "updates": updates,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to configure executor: {str(e)}"
        )


@router.post("/reset-circuit-breakers")
async def reset_circuit_breakers(
    agent_type: Optional[str] = Query(None, description="Specific agent type to reset"),
    user = Depends(require_auth)
) -> JSONResponse:
    """
    Reset circuit breakers for agents.
    
    Allows manual reset of circuit breakers for recovery from failures.
    Can target specific agent types or reset all circuit breakers.
    """
    try:
        executor = await get_enhanced_executor()
        
        reset_count = 0
        
        if agent_type:
            # Reset specific agent circuit breaker
            for agent_enum, circuit_breaker in executor.circuit_breakers.items():
                if agent_enum.value.lower() == agent_type.lower():
                    circuit_breaker.state = circuit_breaker.CircuitState.CLOSED
                    circuit_breaker.failure_count = 0
                    circuit_breaker.last_failure_time = None
                    reset_count += 1
                    break
        else:
            # Reset all circuit breakers
            for circuit_breaker in executor.circuit_breakers.values():
                circuit_breaker.state = circuit_breaker.CircuitState.CLOSED
                circuit_breaker.failure_count = 0
                circuit_breaker.last_failure_time = None
                reset_count += 1
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Reset {reset_count} circuit breaker(s)",
            "agent_type": agent_type or "all",
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset circuit breakers: {str(e)}"
        )


@router.get("/stream/status")
async def stream_executor_status(
    user = Depends(require_auth)
) -> StreamingResponse:
    """
    Stream real-time executor status updates.
    
    Provides Server-Sent Events (SSE) stream of executor status updates
    for real-time monitoring dashboards.
    """
    async def generate_status_stream():
        try:
            while True:
                executor = await get_enhanced_executor()
                status = await executor.get_execution_status()
                
                # Format as SSE
                status_json = json.dumps({
                    "timestamp": datetime.utcnow().isoformat(),
                    "active_tasks": len(status.get("active_tasks", {})),
                    "optimal_concurrency": status.get("optimal_concurrency", 0),
                    "resource_usage": status.get("resource_usage", {}),
                    "analytics": status.get("analytics", {})
                })
                
                yield f"data: {status_json}\n\n"
                
                # Update every 5 seconds
                await asyncio.sleep(5)
                
        except asyncio.CancelledError:
            # Client disconnected
            pass
        except Exception as e:
            error_json = json.dumps({
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            yield f"data: {error_json}\n\n"
    
    return StreamingResponse(
        generate_status_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"}
    )


@router.post("/benchmark")
async def run_performance_benchmark(
    num_tasks: int = Query(10, description="Number of tasks to execute", ge=1, le=100),
    task_duration_ms: int = Query(100, description="Simulated task duration in milliseconds", ge=50, le=5000),
    user = Depends(require_auth)
) -> JSONResponse:
    """
    Run a performance benchmark of the enhanced parallel executor.
    
    Executes a specified number of simulated tasks to measure:
    - Throughput (tasks per second)
    - Average response time
    - Resource utilization
    - Concurrency effectiveness
    """
    try:
        import time
        
        orchestrator = await get_orchestrator()
        
        # Create benchmark workflow
        workflow = WorkflowContext(
            workflow_id=f"benchmark_{int(time.time())}",
            context_type="benchmark",
            input_data={"benchmark": True},
            metadata={"num_tasks": num_tasks, "task_duration_ms": task_duration_ms}
        )
        
        # Create benchmark tasks
        tasks = []
        for i in range(num_tasks):
            task = AgentTask(
                task_id=f"benchmark_task_{i}",
                agent_type="DRAFT_AGENT",  # Use string for API compatibility
                input_data={"content": f"benchmark content {i}", "duration_ms": task_duration_ms},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=task_duration_ms)
            )
            tasks.append(task)
        
        # Run benchmark
        start_time = time.time()
        result = await orchestrator.execute_tasks_enhanced(
            tasks=tasks,
            workflow=workflow,
            priority=PriorityLevel.NORMAL,
            use_enhanced_executor=True
        )
        end_time = time.time()
        
        total_time = end_time - start_time
        throughput = num_tasks / total_time if total_time > 0 else 0
        
        # Calculate efficiency metrics
        theoretical_sequential_time = (num_tasks * task_duration_ms) / 1000
        parallel_efficiency = theoretical_sequential_time / total_time if total_time > 0 else 0
        
        return JSONResponse(content={
            "benchmark_results": {
                "num_tasks": num_tasks,
                "task_duration_ms": task_duration_ms,
                "total_execution_time_seconds": round(total_time, 3),
                "throughput_tasks_per_second": round(throughput, 2),
                "parallel_efficiency": round(parallel_efficiency, 2),
                "successful_tasks": len([r for r in result.get("results", {}).values() 
                                      if r.get("status") == "completed"]),
                "failed_tasks": len([r for r in result.get("results", {}).values() 
                                   if r.get("status") != "completed"])
            },
            "system_metrics": result.get("resource_usage", {}),
            "analytics": result.get("analytics", {}),
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Benchmark failed: {str(e)}"
        )


# Background Tasks

async def _cleanup_completed_workflow(workflow_id: str):
    """Clean up resources for completed workflow."""
    try:
        # In a production system, this would:
        # 1. Archive workflow results
        # 2. Clean up temporary resources
        # 3. Update persistent metrics
        # 4. Send completion notifications
        pass
    except Exception as e:
        print(f"Workflow cleanup failed for {workflow_id}: {str(e)}")


# Error Handlers

@router.exception_handler(ValidationError)
async def validation_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation Error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )