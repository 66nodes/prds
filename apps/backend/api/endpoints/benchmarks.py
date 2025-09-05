"""
Benchmarking API Endpoints

API endpoints for running performance benchmarks and load testing
on the enhanced parallel execution system.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
import os

from services.execution_benchmarker import (
    ExecutionBenchmarker,
    BenchmarkConfig,
    BenchmarkMetrics,
    get_benchmarker,
    run_standard_benchmarks,
    STANDARD_BENCHMARKS
)
from services.enhanced_parallel_executor import LoadBalancingStrategy, PriorityLevel
from services.agent_orchestrator import AgentType
from core.auth import require_auth

router = APIRouter(prefix="/api/v1/benchmarks", tags=["benchmarking"])


# Request Models
class CustomBenchmarkRequest(BaseModel):
    """Request model for custom benchmark configuration."""
    name: str = Field(..., description="Benchmark name")
    description: str = Field(..., description="Benchmark description")
    task_count: int = Field(100, ge=10, le=10000, description="Number of tasks to execute")
    concurrent_users: int = Field(10, ge=1, le=100, description="Number of concurrent users to simulate")
    test_duration_seconds: int = Field(60, ge=30, le=3600, description="Test duration in seconds")
    ramp_up_seconds: int = Field(10, ge=5, le=300, description="Ramp-up time in seconds")
    max_concurrent_tasks: int = Field(20, ge=1, le=100, description="Maximum concurrent tasks")
    load_balancing_strategy: LoadBalancingStrategy = Field(
        LoadBalancingStrategy.RESOURCE_AWARE,
        description="Load balancing strategy"
    )
    enable_circuit_breakers: bool = Field(True, description="Enable circuit breaker patterns")
    enable_agent_pooling: bool = Field(True, description="Enable agent pooling")
    agent_types: Optional[List[AgentType]] = Field(None, description="Specific agent types to test")
    priority_distribution: Optional[Dict[PriorityLevel, float]] = Field(
        None, 
        description="Priority level distribution (must sum to 1.0)"
    )


class BenchmarkResponse(BaseModel):
    """Response model for benchmark execution."""
    test_name: str
    status: str
    duration_seconds: Optional[float] = None
    throughput_tasks_per_second: Optional[float] = None
    success_rate_percent: Optional[float] = None
    avg_response_time_ms: Optional[float] = None
    p95_response_time_ms: Optional[float] = None
    error_rate_percent: Optional[float] = None
    peak_cpu_usage: Optional[float] = None
    peak_memory_usage: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class BenchmarkSuiteResponse(BaseModel):
    """Response model for benchmark suite results."""
    suite_id: str
    total_benchmarks: int
    completed_benchmarks: int
    results: List[BenchmarkResponse]
    comparative_analysis: Optional[Dict[str, Any]] = None
    generated_at: datetime


# Global variables to track running benchmarks
running_benchmarks: Dict[str, asyncio.Task] = {}
benchmark_results: Dict[str, List[BenchmarkMetrics]] = {}


@router.post("/run-standard-suite", response_model=Dict[str, str])
async def run_standard_benchmark_suite(
    background_tasks: BackgroundTasks,
    auth_user = Depends(require_auth)
):
    """Run the standard benchmark suite in the background."""
    suite_id = f"standard_suite_{int(datetime.utcnow().timestamp())}"
    
    async def run_benchmarks():
        try:
            results = await run_standard_benchmarks()
            benchmark_results[suite_id] = results
        except Exception as e:
            benchmark_results[suite_id] = [
                BenchmarkMetrics(
                    test_name="error",
                    start_time=datetime.utcnow(),
                    error_message=str(e)
                )
            ]
        finally:
            if suite_id in running_benchmarks:
                del running_benchmarks[suite_id]
    
    task = asyncio.create_task(run_benchmarks())
    running_benchmarks[suite_id] = task
    
    return {
        "suite_id": suite_id,
        "status": "started",
        "message": f"Standard benchmark suite started with ID: {suite_id}",
        "check_status_url": f"/api/v1/benchmarks/status/{suite_id}"
    }


@router.post("/run-custom", response_model=Dict[str, str])
async def run_custom_benchmark(
    config: CustomBenchmarkRequest,
    background_tasks: BackgroundTasks,
    auth_user = Depends(require_auth)
):
    """Run a custom benchmark configuration."""
    
    # Validate priority distribution
    if config.priority_distribution:
        total_weight = sum(config.priority_distribution.values())
        if abs(total_weight - 1.0) > 0.01:
            raise HTTPException(
                status_code=400,
                detail=f"Priority distribution weights must sum to 1.0, got {total_weight}"
            )
    
    benchmark_id = f"custom_{config.name}_{int(datetime.utcnow().timestamp())}"
    
    # Convert request to BenchmarkConfig
    benchmark_config = BenchmarkConfig(
        name=config.name,
        description=config.description,
        task_count=config.task_count,
        concurrent_users=config.concurrent_users,
        test_duration_seconds=config.test_duration_seconds,
        ramp_up_seconds=config.ramp_up_seconds,
        max_concurrent_tasks=config.max_concurrent_tasks,
        load_balancing_strategy=config.load_balancing_strategy,
        enable_circuit_breakers=config.enable_circuit_breakers,
        enable_agent_pooling=config.enable_agent_pooling,
        agent_types=config.agent_types or [],
        priority_distribution=config.priority_distribution or {}
    )
    
    async def run_benchmark():
        try:
            benchmarker = await get_benchmarker()
            result = await benchmarker.run_single_benchmark(benchmark_config)
            benchmark_results[benchmark_id] = [result]
        except Exception as e:
            benchmark_results[benchmark_id] = [
                BenchmarkMetrics(
                    test_name=config.name,
                    start_time=datetime.utcnow(),
                    error_message=str(e)
                )
            ]
        finally:
            if benchmark_id in running_benchmarks:
                del running_benchmarks[benchmark_id]
    
    task = asyncio.create_task(run_benchmark())
    running_benchmarks[benchmark_id] = task
    
    return {
        "benchmark_id": benchmark_id,
        "status": "started",
        "message": f"Custom benchmark '{config.name}' started with ID: {benchmark_id}",
        "check_status_url": f"/api/v1/benchmarks/status/{benchmark_id}"
    }


@router.get("/status/{benchmark_id}")
async def get_benchmark_status(
    benchmark_id: str,
    auth_user = Depends(require_auth)
):
    """Get the status of a running or completed benchmark."""
    
    # Check if benchmark is still running
    if benchmark_id in running_benchmarks:
        task = running_benchmarks[benchmark_id]
        return {
            "benchmark_id": benchmark_id,
            "status": "running",
            "is_done": task.done(),
            "message": "Benchmark is currently running"
        }
    
    # Check if benchmark is completed
    if benchmark_id in benchmark_results:
        results = benchmark_results[benchmark_id]
        
        # Check if there was an error
        if len(results) == 1 and hasattr(results[0], 'error_message') and results[0].error_message:
            return {
                "benchmark_id": benchmark_id,
                "status": "failed",
                "error": results[0].error_message,
                "message": "Benchmark failed during execution"
            }
        
        return {
            "benchmark_id": benchmark_id,
            "status": "completed",
            "is_done": True,
            "results_count": len(results),
            "message": "Benchmark completed successfully",
            "get_results_url": f"/api/v1/benchmarks/results/{benchmark_id}"
        }
    
    # Benchmark not found
    raise HTTPException(
        status_code=404,
        detail=f"Benchmark with ID '{benchmark_id}' not found"
    )


@router.get("/results/{benchmark_id}", response_model=BenchmarkSuiteResponse)
async def get_benchmark_results(
    benchmark_id: str,
    auth_user = Depends(require_auth)
):
    """Get the results of a completed benchmark."""
    
    if benchmark_id not in benchmark_results:
        raise HTTPException(
            status_code=404,
            detail=f"Results for benchmark '{benchmark_id}' not found"
        )
    
    results = benchmark_results[benchmark_id]
    
    # Convert BenchmarkMetrics to BenchmarkResponse
    response_results = []
    for result in results:
        if hasattr(result, 'error_message') and result.error_message:
            response_results.append(BenchmarkResponse(
                test_name=result.test_name,
                status="failed",
                error_message=result.error_message,
                started_at=result.start_time
            ))
        else:
            success_rate = (
                (result.successful_tasks / result.total_tasks_executed) * 100
                if result.total_tasks_executed > 0 else 0
            )
            
            response_results.append(BenchmarkResponse(
                test_name=result.test_name,
                status="completed",
                duration_seconds=result.total_duration_seconds,
                throughput_tasks_per_second=result.tasks_per_second,
                success_rate_percent=success_rate,
                avg_response_time_ms=result.avg_response_time_ms,
                p95_response_time_ms=result.p95_response_time_ms,
                error_rate_percent=result.error_rate,
                peak_cpu_usage=result.peak_cpu_usage,
                peak_memory_usage=result.peak_memory_usage,
                started_at=result.start_time,
                completed_at=result.end_time
            ))
    
    return BenchmarkSuiteResponse(
        suite_id=benchmark_id,
        total_benchmarks=len(results),
        completed_benchmarks=len([r for r in response_results if r.status == "completed"]),
        results=response_results,
        generated_at=datetime.utcnow()
    )


@router.get("/export/{benchmark_id}")
async def export_benchmark_results(
    benchmark_id: str,
    format: str = Query("json", regex="^(json|csv)$"),
    auth_user = Depends(require_auth)
):
    """Export benchmark results in JSON or CSV format."""
    
    if benchmark_id not in benchmark_results:
        raise HTTPException(
            status_code=404,
            detail=f"Results for benchmark '{benchmark_id}' not found"
        )
    
    results = benchmark_results[benchmark_id]
    
    if format == "json":
        # Export as JSON
        filename = f"benchmark_results_{benchmark_id}.json"
        
        # Create temporary file
        import json
        import tempfile
        
        export_data = {
            "benchmark_id": benchmark_id,
            "results": [
                {
                    "test_name": r.test_name,
                    "duration_seconds": r.total_duration_seconds,
                    "throughput_tasks_per_second": r.tasks_per_second,
                    "success_rate_percent": ((r.successful_tasks / r.total_tasks_executed) * 100) if r.total_tasks_executed > 0 else 0,
                    "avg_response_time_ms": r.avg_response_time_ms,
                    "p95_response_time_ms": r.p95_response_time_ms,
                    "error_rate_percent": r.error_rate,
                    "peak_cpu_usage": r.peak_cpu_usage,
                    "peak_memory_usage": r.peak_memory_usage,
                    "started_at": r.start_time.isoformat(),
                    "completed_at": r.end_time.isoformat() if r.end_time else None
                }
                for r in results
            ],
            "exported_at": datetime.utcnow().isoformat()
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(export_data, f, indent=2)
            temp_filename = f.name
        
        return FileResponse(
            path=temp_filename,
            filename=filename,
            media_type="application/json"
        )
    
    elif format == "csv":
        # Export as CSV
        filename = f"benchmark_results_{benchmark_id}.csv"
        
        import csv
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                "test_name", "duration_seconds", "throughput_tasks_per_second",
                "success_rate_percent", "avg_response_time_ms", "p95_response_time_ms",
                "error_rate_percent", "peak_cpu_usage", "peak_memory_usage",
                "started_at", "completed_at"
            ])
            
            # Write data
            for result in results:
                writer.writerow([
                    result.test_name,
                    result.total_duration_seconds,
                    result.tasks_per_second,
                    ((result.successful_tasks / result.total_tasks_executed) * 100) if result.total_tasks_executed > 0 else 0,
                    result.avg_response_time_ms,
                    result.p95_response_time_ms,
                    result.error_rate,
                    result.peak_cpu_usage,
                    result.peak_memory_usage,
                    result.start_time.isoformat(),
                    result.end_time.isoformat() if result.end_time else ""
                ])
            
            temp_filename = f.name
        
        return FileResponse(
            path=temp_filename,
            filename=filename,
            media_type="text/csv"
        )


@router.get("/configurations")
async def get_standard_configurations(
    auth_user = Depends(require_auth)
):
    """Get the list of standard benchmark configurations."""
    
    configs = []
    for config in STANDARD_BENCHMARKS:
        configs.append({
            "name": config.name,
            "description": config.description,
            "task_count": config.task_count,
            "concurrent_users": config.concurrent_users,
            "test_duration_seconds": config.test_duration_seconds,
            "max_concurrent_tasks": config.max_concurrent_tasks,
            "load_balancing_strategy": config.load_balancing_strategy.value,
            "enable_circuit_breakers": config.enable_circuit_breakers,
            "enable_agent_pooling": config.enable_agent_pooling
        })
    
    return {
        "standard_configurations": configs,
        "total_configurations": len(configs)
    }


@router.delete("/results/{benchmark_id}")
async def delete_benchmark_results(
    benchmark_id: str,
    auth_user = Depends(require_auth)
):
    """Delete benchmark results."""
    
    # Stop running benchmark if it exists
    if benchmark_id in running_benchmarks:
        task = running_benchmarks[benchmark_id]
        task.cancel()
        del running_benchmarks[benchmark_id]
    
    # Delete results if they exist
    if benchmark_id in benchmark_results:
        del benchmark_results[benchmark_id]
        return {"message": f"Results for benchmark '{benchmark_id}' deleted successfully"}
    
    raise HTTPException(
        status_code=404,
        detail=f"Benchmark '{benchmark_id}' not found"
    )


@router.get("/active")
async def get_active_benchmarks(
    auth_user = Depends(require_auth)
):
    """Get list of currently active benchmarks."""
    
    active = []
    for benchmark_id, task in running_benchmarks.items():
        active.append({
            "benchmark_id": benchmark_id,
            "status": "running",
            "is_done": task.done(),
            "started_at": datetime.utcnow().isoformat()  # Approximation
        })
    
    return {
        "active_benchmarks": active,
        "total_active": len(active)
    }