"""
Validated Parallel Executor with GraphRAG Integration

Enhanced parallel execution system that integrates comprehensive GraphRAG validation
to ensure <2% hallucination rate and high-quality agent outputs.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import structlog
from services.enhanced_parallel_executor import (
    EnhancedParallelExecutor, ExecutionStatus, PriorityLevel, ExecutionMetrics,
    LoadBalancingStrategy, CircuitBreaker, AgentPool
)
from services.agent_orchestrator import AgentTask, WorkflowContext, AgentType
from services.graphrag_validation_middleware import (
    GraphRAGValidationMiddleware, ValidationCheckpoint, ValidationStage,
    AgentValidationResult, get_validation_middleware
)
from services.graphrag.validation_pipeline import ValidationLevel

logger = structlog.get_logger(__name__)


@dataclass
class ValidatedExecutionResult:
    """Enhanced execution result with validation data."""
    task_id: str
    status: ExecutionStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Execution metrics
    execution_metrics: Optional[ExecutionMetrics] = None
    
    # Validation results
    validation_result: Optional[AgentValidationResult] = None
    
    # Quality metrics
    confidence_score: float = 0.0
    hallucination_rate: float = 0.0
    quality_improvements: Dict[str, Any] = None
    
    # Processing details
    total_processing_time_ms: float = 0.0
    validation_time_ms: float = 0.0
    corrections_applied: int = 0


class ValidatedParallelExecutor(EnhancedParallelExecutor):
    """
    Enhanced parallel executor with integrated GraphRAG validation.
    
    Extends the base parallel executor to include comprehensive validation
    checkpoints ensuring high-quality outputs with <2% hallucination rate.
    """
    
    def __init__(
        self,
        max_concurrent_tasks: int = 20,
        load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.RESOURCE_AWARE,
        enable_circuit_breakers: bool = True,
        enable_agent_pooling: bool = True,
        resource_monitoring_interval: float = 5.0,
        # Validation-specific parameters
        enable_validation: bool = True,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        enable_validation_corrections: bool = True,
        max_validation_retries: int = 3,
        validation_timeout: int = 30
    ):
        super().__init__(
            max_concurrent_tasks=max_concurrent_tasks,
            load_balancing_strategy=load_balancing_strategy,
            enable_circuit_breakers=enable_circuit_breakers,
            enable_agent_pooling=enable_agent_pooling,
            resource_monitoring_interval=resource_monitoring_interval
        )
        
        # Validation configuration
        self.enable_validation = enable_validation
        self.validation_level = validation_level
        self.enable_validation_corrections = enable_validation_corrections
        self.max_validation_retries = max_validation_retries
        self.validation_timeout = validation_timeout
        
        # Validation middleware
        self.validation_middleware: Optional[GraphRAGValidationMiddleware] = None
        
        # Enhanced analytics with validation metrics
        self.validation_analytics = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "hallucinations_prevented": 0,
            "corrections_applied": 0,
            "avg_confidence_score": 0.0,
            "avg_validation_time_ms": 0.0,
            "quality_improvement_rate": 0.0
        }
        
        # Validation result cache for performance
        self.validation_cache: Dict[str, AgentValidationResult] = {}
        self.validation_cache_max_size = 1000
    
    async def initialize(self) -> None:
        """Initialize the validated parallel executor."""
        # Initialize base executor
        await super().initialize()
        
        # Initialize validation middleware if enabled
        if self.enable_validation:
            self.validation_middleware = await get_validation_middleware()
            logger.info("Validation middleware integrated with parallel executor")
    
    async def execute_parallel_with_validation(
        self,
        tasks: List[AgentTask],
        workflow: WorkflowContext,
        priority: PriorityLevel = PriorityLevel.NORMAL,
        timeout: Optional[float] = None,
        custom_validation_checkpoint: Optional[ValidationCheckpoint] = None
    ) -> Dict[str, Any]:
        """
        Execute tasks in parallel with integrated GraphRAG validation.
        
        Args:
            tasks: List of agent tasks to execute
            workflow: Workflow context
            priority: Task priority level
            timeout: Optional execution timeout
            custom_validation_checkpoint: Custom validation configuration
            
        Returns:
            Comprehensive results including validation metrics
        """
        if not self.is_initialized:
            raise RuntimeError("Validated parallel executor not initialized")
        
        start_time = time.time()
        
        logger.info(
            f"Starting validated parallel execution of {len(tasks)} tasks",
            validation_enabled=self.enable_validation,
            validation_level=self.validation_level.value if self.enable_validation else None
        )
        
        try:
            # Execute tasks with base parallel executor
            base_results = await super().execute_parallel(
                tasks=tasks,
                workflow=workflow,
                priority=priority,
                timeout=timeout
            )
            
            # If validation is disabled, return base results
            if not self.enable_validation or not self.validation_middleware:
                return base_results
            
            # Extract execution results for validation
            task_outputs = []
            execution_metrics_map = {}
            
            for task in tasks:
                task_result = base_results["results"].get(task.task_id)
                if task_result and task_result.get("status") == ExecutionStatus.COMPLETED.value:
                    # Get execution metrics
                    execution_metrics = ExecutionMetrics(
                        task_id=task.task_id,
                        agent_type=task.agent_type,
                        start_time=datetime.utcnow() - timedelta(
                            milliseconds=task_result.get("metrics", {}).get("duration_ms", 0)
                        ),
                        end_time=datetime.utcnow()
                    )
                    execution_metrics.duration_ms = task_result.get("metrics", {}).get("duration_ms", 0)
                    execution_metrics.queue_wait_time_ms = task_result.get("metrics", {}).get("queue_wait_ms", 0)
                    execution_metrics.execution_attempts = task_result.get("metrics", {}).get("attempts", 1)
                    execution_metrics.status = ExecutionStatus.COMPLETED
                    
                    execution_metrics_map[task.task_id] = execution_metrics
                    
                    # Prepare for validation
                    task_outputs.append((
                        task,
                        task_result.get("result", {}),
                        workflow,
                        execution_metrics
                    ))
            
            # Perform batch validation
            validation_start = time.time()
            validation_results = []
            
            if task_outputs:
                validation_results = await self.validation_middleware.validate_batch_outputs(
                    task_outputs,
                    custom_validation_checkpoint
                )
            
            validation_time = (time.time() - validation_start) * 1000
            
            # Integrate validation results
            enhanced_results = await self._integrate_validation_results(
                base_results,
                validation_results,
                execution_metrics_map
            )
            
            # Update validation analytics
            self._update_validation_analytics(validation_results, validation_time)
            
            # Add validation metrics to response
            enhanced_results["validation_analytics"] = {
                "total_tasks_validated": len(validation_results),
                "validation_time_ms": validation_time,
                "passed_validation": sum(1 for r in validation_results if r.passes_validation),
                "failed_validation": sum(1 for r in validation_results if not r.passes_validation),
                "avg_confidence_score": sum(r.confidence_score for r in validation_results) / len(validation_results) if validation_results else 0,
                "avg_hallucination_rate": sum(r.hallucination_rate for r in validation_results) / len(validation_results) if validation_results else 0,
                "corrections_applied": sum(len(r.corrections_applied) for r in validation_results),
                "quality_improvements": sum(1 for r in validation_results if r.quality_improvements)
            }
            
            total_time = (time.time() - start_time) * 1000
            logger.info(
                f"Validated parallel execution completed in {total_time:.2f}ms",
                validation_time_ms=validation_time,
                tasks_validated=len(validation_results),
                passed_validation=enhanced_results["validation_analytics"]["passed_validation"]
            )
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Validated parallel execution failed: {str(e)}")
            raise
    
    async def _integrate_validation_results(
        self,
        base_results: Dict[str, Any],
        validation_results: List[AgentValidationResult],
        execution_metrics_map: Dict[str, ExecutionMetrics]
    ) -> Dict[str, Any]:
        """Integrate validation results with base execution results."""
        
        enhanced_results = base_results.copy()
        validation_map = {vr.task_id: vr for vr in validation_results}
        
        # Enhance each task result with validation data
        for task_id, task_result in enhanced_results["results"].items():
            validation_result = validation_map.get(task_id)
            execution_metrics = execution_metrics_map.get(task_id)
            
            if validation_result:
                # Create enhanced result
                enhanced_task_result = ValidatedExecutionResult(
                    task_id=task_id,
                    status=ExecutionStatus.COMPLETED if task_result.get("status") == ExecutionStatus.COMPLETED.value else ExecutionStatus.FAILED,
                    result=task_result.get("result"),
                    error=task_result.get("error"),
                    execution_metrics=execution_metrics,
                    validation_result=validation_result,
                    confidence_score=validation_result.confidence_score,
                    hallucination_rate=validation_result.hallucination_rate,
                    quality_improvements=validation_result.quality_improvements,
                    total_processing_time_ms=task_result.get("metrics", {}).get("duration_ms", 0) + validation_result.processing_time_ms,
                    validation_time_ms=validation_result.processing_time_ms,
                    corrections_applied=len(validation_result.corrections_applied)
                )
                
                # Update the result with enhanced data
                enhanced_results["results"][task_id] = {
                    **task_result,
                    "validation": {
                        "passes_validation": validation_result.passes_validation,
                        "confidence_score": validation_result.confidence_score,
                        "hallucination_rate": validation_result.hallucination_rate,
                        "validation_status": validation_result.validation_result.status.value if validation_result.validation_result else "unknown",
                        "processing_time_ms": validation_result.processing_time_ms,
                        "iterations_performed": validation_result.iterations_performed,
                        "corrections_applied": len(validation_result.corrections_applied),
                        "quality_improvements": validation_result.quality_improvements or {}
                    }
                }
                
                # If validation provided corrected content, include it
                if validation_result.validated_content and validation_result.validated_content != validation_result.original_content:
                    enhanced_results["results"][task_id]["validated_content"] = validation_result.validated_content
                    enhanced_results["results"][task_id]["original_content"] = validation_result.original_content
                
                # Update task status based on validation
                if not validation_result.passes_validation:
                    enhanced_results["results"][task_id]["status"] = ExecutionStatus.FAILED.value
                    enhanced_results["results"][task_id]["validation_error"] = "Content failed GraphRAG validation"
        
        return enhanced_results
    
    def _update_validation_analytics(
        self,
        validation_results: List[AgentValidationResult],
        total_validation_time_ms: float
    ) -> None:
        """Update validation analytics with current results."""
        if not validation_results:
            return
        
        # Update counts
        self.validation_analytics["total_validations"] += len(validation_results)
        
        passed = sum(1 for r in validation_results if r.passes_validation)
        failed = sum(1 for r in validation_results if not r.passes_validation)
        
        self.validation_analytics["successful_validations"] += passed
        self.validation_analytics["failed_validations"] += failed
        
        # Count corrections and quality improvements
        corrections_count = sum(len(r.corrections_applied) for r in validation_results)
        self.validation_analytics["corrections_applied"] += corrections_count
        
        quality_improvements = sum(1 for r in validation_results if r.quality_improvements)
        
        # Update averages
        total_validations = self.validation_analytics["total_validations"]
        
        # Average confidence score
        confidence_scores = [r.confidence_score for r in validation_results]
        if confidence_scores:
            current_avg_confidence = sum(confidence_scores) / len(confidence_scores)
            self.validation_analytics["avg_confidence_score"] = (
                (self.validation_analytics["avg_confidence_score"] * (total_validations - len(validation_results)) +
                 current_avg_confidence * len(validation_results)) / total_validations
            )
        
        # Average validation time
        avg_validation_time = total_validation_time_ms / len(validation_results)
        self.validation_analytics["avg_validation_time_ms"] = (
            (self.validation_analytics["avg_validation_time_ms"] * (total_validations - len(validation_results)) +
             avg_validation_time * len(validation_results)) / total_validations
        )
        
        # Quality improvement rate
        if len(validation_results) > 0:
            improvement_rate = quality_improvements / len(validation_results)
            self.validation_analytics["quality_improvement_rate"] = (
                (self.validation_analytics["quality_improvement_rate"] * (total_validations - len(validation_results)) +
                 improvement_rate * len(validation_results)) / total_validations
            )
        
        # Hallucination prevention count
        hallucinations_prevented = sum(
            1 for r in validation_results 
            if r.hallucination_rate > 0.02  # Above 2% threshold
        )
        self.validation_analytics["hallucinations_prevented"] += hallucinations_prevented
    
    async def get_enhanced_execution_status(self) -> Dict[str, Any]:
        """Get comprehensive execution status including validation metrics."""
        base_status = await super().get_execution_status()
        
        # Add validation-specific status
        validation_status = {
            "validation_enabled": self.enable_validation,
            "validation_level": self.validation_level.value if self.enable_validation else None,
            "validation_analytics": self.validation_analytics.copy(),
            "validation_cache_size": len(self.validation_cache)
        }
        
        # Get validation middleware status if available
        if self.validation_middleware:
            middleware_health = await self.validation_middleware.health_check()
            validation_status["middleware_health"] = middleware_health
            
            middleware_stats = await self.validation_middleware.get_validation_statistics()
            validation_status["middleware_statistics"] = middleware_stats
        
        # Combine status
        enhanced_status = {**base_status, **validation_status}
        
        # Calculate validation success rate
        total_val = self.validation_analytics["total_validations"]
        if total_val > 0:
            enhanced_status["validation_success_rate"] = (
                self.validation_analytics["successful_validations"] / total_val * 100
            )
        else:
            enhanced_status["validation_success_rate"] = 0.0
        
        return enhanced_status
    
    def configure_validation(
        self,
        validation_level: ValidationLevel,
        enable_corrections: bool = True,
        max_retries: int = 3,
        timeout: int = 30
    ) -> None:
        """Configure validation settings."""
        self.validation_level = validation_level
        self.enable_validation_corrections = enable_corrections
        self.max_validation_retries = max_retries
        self.validation_timeout = timeout
        
        logger.info(
            "Validation configuration updated",
            validation_level=validation_level.value,
            enable_corrections=enable_corrections,
            max_retries=max_retries,
            timeout=timeout
        )
    
    def disable_validation(self) -> None:
        """Disable validation for performance-critical operations."""
        self.enable_validation = False
        logger.warning("Validation disabled - hallucination prevention unavailable")
    
    def enable_validation_mode(self, validation_level: ValidationLevel = ValidationLevel.STANDARD) -> None:
        """Re-enable validation with specified level."""
        self.enable_validation = True
        self.validation_level = validation_level
        logger.info(f"Validation enabled with level: {validation_level.value}")
    
    async def shutdown(self) -> None:
        """Enhanced shutdown with validation middleware cleanup."""
        # Shutdown base executor
        await super().shutdown()
        
        # Close validation middleware
        if self.validation_middleware:
            await self.validation_middleware.close()
            logger.info("Validation middleware closed")


# Global instance
_validated_executor_instance: Optional[ValidatedParallelExecutor] = None


async def get_validated_executor() -> ValidatedParallelExecutor:
    """Get the global validated parallel executor instance."""
    global _validated_executor_instance
    
    if not _validated_executor_instance:
        _validated_executor_instance = ValidatedParallelExecutor()
        await _validated_executor_instance.initialize()
    
    return _validated_executor_instance


async def execute_with_validation(
    tasks: List[AgentTask],
    workflow: WorkflowContext,
    priority: PriorityLevel = PriorityLevel.NORMAL,
    validation_level: ValidationLevel = ValidationLevel.STANDARD,
    enable_corrections: bool = True
) -> Dict[str, Any]:
    """Convenience function for validated task execution."""
    executor = await get_validated_executor()
    
    # Configure validation
    executor.configure_validation(
        validation_level=validation_level,
        enable_corrections=enable_corrections
    )
    
    return await executor.execute_parallel_with_validation(
        tasks=tasks,
        workflow=workflow,
        priority=priority
    )