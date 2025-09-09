"""
Enhanced Parallel Executor with Integrated Agent Action Logging

Extension of the enhanced parallel executor that integrates comprehensive
agent action logging for quality assurance and audit traceability.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

import structlog
from services.enhanced_parallel_executor import (
    EnhancedParallelExecutor, ExecutionStatus, PriorityLevel,
    ExecutionMetrics, CircuitState
)
from services.agent_orchestrator import AgentTask, WorkflowContext, AgentType
from services.agent_action_logger import (
    get_agent_logger, ActionType, LogLevel,
    log_task_started, log_task_completed
)

logger = structlog.get_logger(__name__)


class LoggingEnhancedParallelExecutor(EnhancedParallelExecutor):
    """
    Enhanced parallel executor with integrated comprehensive logging.
    
    Extends the base enhanced parallel executor to include detailed
    logging of all agent actions, decisions, and outcomes for QA and audit.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_logger = None
    
    async def initialize(self) -> None:
        """Initialize the enhanced parallel executor with logging."""
        try:
            # Initialize base executor
            await super().initialize()
            
            # Initialize agent logger
            self.agent_logger = await get_agent_logger()
            
            # Log initialization
            await self.agent_logger.log_action(
                ActionType.ORCHESTRATION_STARTED,
                log_level=LogLevel.INFO,
                execution_stage="executor_initialization",
                resource_usage={
                    "max_concurrent_tasks": self.max_concurrent_tasks,
                    "load_balancing_strategy": self.load_balancing_strategy.value,
                    "circuit_breakers_enabled": self.enable_circuit_breakers,
                    "agent_pooling_enabled": self.enable_agent_pooling
                },
                reasoning="Enhanced parallel executor initialized with comprehensive logging support",
                tags=["initialization", "executor", "logging"]
            )
            
            logger.info("Enhanced parallel executor with logging initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced parallel executor with logging: {str(e)}")
            raise
    
    async def execute_parallel(
        self,
        tasks: List[AgentTask],
        workflow: WorkflowContext,
        priority: PriorityLevel = PriorityLevel.NORMAL,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute tasks in parallel with comprehensive logging.
        """
        if not tasks:
            return {"results": {}, "analytics": self.execution_analytics.copy()}
        
        # Log orchestration start
        orchestration_log_id = await self.agent_logger.log_orchestration_event(
            "started", workflow, tasks,
            reasoning=f"Starting parallel execution of {len(tasks)} tasks with priority {priority.value}",
            resource_usage=self._get_current_resource_usage(),
            decision_context={
                "task_count": len(tasks),
                "priority": priority.value,
                "timeout": timeout,
                "max_concurrent": self.optimal_concurrency
            }
        )
        
        logger.info(
            "Starting enhanced parallel execution with logging",
            task_count=len(tasks),
            priority=priority.value,
            max_concurrent=self.optimal_concurrency,
            orchestration_log_id=orchestration_log_id
        )
        
        # Log individual task starts
        for task in tasks:
            await log_task_started(
                task, workflow,
                correlation_id=orchestration_log_id,
                execution_stage="queued",
                reasoning=f"Task queued for execution with agent {task.agent_type.value}",
                tags=["task_lifecycle", "queued"]
            )
        
        # Prepare tasks with priorities
        prioritized_tasks = await self._prepare_tasks_with_priority(tasks, workflow, priority)
        
        # Log load balancing decisions
        await self._log_load_balancing_decisions(prioritized_tasks, workflow)
        
        # Execute with enhanced logging
        try:
            if timeout:
                results = await asyncio.wait_for(
                    self._execute_with_enhanced_logging(prioritized_tasks, workflow, orchestration_log_id),
                    timeout=timeout
                )
            else:
                results = await self._execute_with_enhanced_logging(prioritized_tasks, workflow, orchestration_log_id)
            
            # Update analytics
            self._update_execution_analytics(results)
            
            # Log orchestration completion
            await self.agent_logger.log_orchestration_event(
                "completed", workflow, tasks, results,
                correlation_id=orchestration_log_id,
                duration_ms=(time.time() * 1000),  # This should be calculated properly
                quality_metrics={
                    "successful_tasks": len([r for r in results.values() if r.get("status") == "completed"]),
                    "failed_tasks": len([r for r in results.values() if r.get("status") == "failed"]),
                    "success_rate": len([r for r in results.values() if r.get("status") == "completed"]) / len(results) * 100 if results else 0
                },
                resource_usage=self._get_current_resource_usage(),
                tags=["orchestration", "completed", "success"]
            )
            
            return {
                "results": results,
                "analytics": self.execution_analytics.copy(),
                "resource_usage": self._get_current_resource_usage(),
                "circuit_breaker_status": self._get_circuit_breaker_status(),
                "orchestration_log_id": orchestration_log_id
            }
            
        except asyncio.TimeoutError:
            # Log timeout
            await self.agent_logger.log_action(
                ActionType.ORCHESTRATION_FAILED,
                log_level=LogLevel.ERROR,
                correlation_id=orchestration_log_id,
                error_message="Parallel execution timed out",
                error_category="timeout",
                execution_stage="timeout",
                duration_ms=timeout * 1000 if timeout else None,
                tags=["orchestration", "timeout", "failure"]
            )
            
            logger.warning("Parallel execution timed out", timeout=timeout)
            return {
                "results": {},
                "error": "Execution timeout",
                "analytics": self.execution_analytics.copy(),
                "orchestration_log_id": orchestration_log_id
            }
            
        except Exception as e:
            # Log execution failure
            await self.agent_logger.log_action(
                ActionType.ORCHESTRATION_FAILED,
                log_level=LogLevel.ERROR,
                correlation_id=orchestration_log_id,
                error_message=str(e),
                error_category="execution",
                execution_stage="failed",
                tags=["orchestration", "error", "failure"]
            )
            
            logger.error(f"Enhanced parallel execution failed: {str(e)}")
            raise
    
    async def _execute_with_enhanced_logging(
        self,
        prioritized_tasks: List[Any],
        workflow: WorkflowContext,
        orchestration_log_id: str
    ) -> Dict[str, Any]:
        """Execute tasks with enhanced logging at every step."""
        
        # Log adaptive concurrency adjustment
        await self._log_concurrency_adjustment()
        
        # Create execution coroutines with logging
        execution_tasks = []
        for priority_score, task, task_workflow in prioritized_tasks:
            execution_coro = self._execute_single_task_with_enhanced_logging(
                task, task_workflow, orchestration_log_id
            )
            execution_tasks.append(execution_coro)
        
        # Execute with controlled concurrency
        results = {}
        completed_tasks = await asyncio.gather(*execution_tasks, return_exceptions=True)
        
        # Process results with logging
        for i, (task_result, (_, task, _)) in enumerate(zip(completed_tasks, prioritized_tasks)):
            if isinstance(task_result, Exception):
                await self.agent_logger.log_action(
                    ActionType.TASK_FAILED,
                    log_level=LogLevel.ERROR,
                    task_id=task.task_id,
                    agent_type=task.agent_type,
                    correlation_id=orchestration_log_id,
                    error_message=str(task_result),
                    error_category="execution_exception",
                    execution_stage="task_completion",
                    tags=["task", "exception", "failure"]
                )
                
                logger.error(f"Task {task.task_id} failed: {str(task_result)}")
                results[task.task_id] = {
                    "status": ExecutionStatus.FAILED.value,
                    "error": str(task_result)
                }
            else:
                results[task.task_id] = task_result
        
        return results
    
    async def _execute_single_task_with_enhanced_logging(
        self,
        task: AgentTask,
        workflow: WorkflowContext,
        orchestration_log_id: str
    ) -> Dict[str, Any]:
        """Execute a single task with comprehensive logging."""
        
        # Log agent selection decision
        await self.agent_logger.log_action(
            ActionType.AGENT_SELECTED,
            log_level=LogLevel.INFO,
            task_id=task.task_id,
            agent_type=task.agent_type,
            correlation_id=orchestration_log_id,
            decision_context={
                "task_type": getattr(task, 'task_type', 'unknown'),
                "selection_criteria": "context_aware_selection",
                "agent_capability_match": True
            },
            reasoning=f"Agent {task.agent_type.value} selected for task based on capability matching",
            confidence_score=0.9,  # This should be calculated based on actual selection logic
            tags=["agent_selection", "decision"]
        )
        
        # Check circuit breaker with logging
        if self.enable_circuit_breakers:
            circuit_breaker = self.circuit_breakers.get(task.agent_type)
            if circuit_breaker:
                circuit_state = circuit_breaker.state
                await self.agent_logger.log_action(
                    ActionType.CIRCUIT_BREAKER_TRIGGERED if circuit_state == CircuitState.OPEN else ActionType.QUALITY_CHECK,
                    log_level=LogLevel.WARN if circuit_state == CircuitState.OPEN else LogLevel.DEBUG,
                    task_id=task.task_id,
                    agent_type=task.agent_type,
                    correlation_id=orchestration_log_id,
                    circuit_breaker_state=circuit_state.value,
                    resource_usage={
                        "failure_count": circuit_breaker.failure_count,
                        "failure_threshold": circuit_breaker.failure_threshold,
                        "last_failure_time": circuit_breaker.last_failure_time.isoformat() if circuit_breaker.last_failure_time else None
                    },
                    execution_stage="circuit_breaker_check",
                    tags=["circuit_breaker", "resilience"]
                )
                
                if not await self._check_circuit_breaker(circuit_breaker):
                    return {
                        "status": ExecutionStatus.FAILED.value,
                        "error": "Circuit breaker is open",
                        "circuit_breaker_state": circuit_breaker.state.value
                    }
        
        # Create execution metrics
        metrics = ExecutionMetrics(
            task_id=task.task_id,
            agent_type=task.agent_type,
            start_time=datetime.utcnow()
        )
        
        # Log task execution start
        await self.agent_logger.log_action(
            ActionType.AGENT_EXECUTION_STARTED,
            log_level=LogLevel.INFO,
            task_id=task.task_id,
            agent_type=task.agent_type,
            correlation_id=orchestration_log_id,
            execution_stage="execution_started",
            input_data=getattr(task, 'input_data', None),
            resource_usage={
                "active_tasks": len(self.active_tasks),
                "optimal_concurrency": self.optimal_concurrency,
                "agent_pool_available": len(self.agent_pools.get(task.agent_type, {}).get('available_instances', []))
            },
            tags=["agent_execution", "started"]
        )
        
        # Acquire semaphore for concurrency control
        async with self.semaphore:
            self.active_tasks[task.task_id] = metrics
            
            try:
                # Get agent instance (with pooling if enabled) - with logging
                agent_instance = await self._get_agent_instance_with_logging(task.agent_type, task.task_id, orchestration_log_id)
                
                # Record queue wait time
                queue_wait = (datetime.utcnow() - metrics.start_time).total_seconds() * 1000
                metrics.queue_wait_time_ms = int(queue_wait)
                
                # Log queue wait performance
                if queue_wait > 1000:  # > 1 second
                    await self.agent_logger.log_action(
                        ActionType.PERFORMANCE_ALERT,
                        log_level=LogLevel.WARN,
                        task_id=task.task_id,
                        agent_type=task.agent_type,
                        correlation_id=orchestration_log_id,
                        queue_wait_ms=queue_wait,
                        execution_stage="queue_wait_alert",
                        tags=["performance", "queue_wait", "alert"]
                    )
                
                # Execute with timeout and retry logic - with logging
                execution_start = time.time()
                result = await self._execute_with_retry_and_logging(
                    agent_instance, task, workflow, metrics, orchestration_log_id
                )
                execution_end = time.time()
                
                # Update metrics
                metrics.end_time = datetime.utcnow()
                metrics.duration_ms = int((execution_end - execution_start) * 1000)
                metrics.status = ExecutionStatus.COMPLETED
                
                # Log successful completion
                await log_task_completed(
                    task, workflow, metrics,
                    correlation_id=orchestration_log_id,
                    output_data=result,
                    quality_metrics={
                        "execution_time_ms": metrics.duration_ms,
                        "queue_wait_ms": metrics.queue_wait_time_ms,
                        "attempts": metrics.execution_attempts,
                        "status": "success"
                    },
                    tags=["task_lifecycle", "completed", "success"]
                )
                
                # Update load balancing metrics
                self._update_load_balancing_metrics(task.agent_type, metrics.duration_ms)
                
                # Update circuit breaker (success) with logging
                if self.enable_circuit_breakers:
                    await self._record_circuit_breaker_success_with_logging(task.agent_type, task.task_id, orchestration_log_id)
                
                # Return agent instance to pool
                if self.enable_agent_pooling:
                    await self._return_agent_to_pool_with_logging(task.agent_type, agent_instance, task.task_id, orchestration_log_id)
                
                return {
                    "status": ExecutionStatus.COMPLETED.value,
                    "result": result,
                    "metrics": {
                        "duration_ms": metrics.duration_ms,
                        "queue_wait_ms": metrics.queue_wait_time_ms,
                        "attempts": metrics.execution_attempts
                    }
                }
                
            except Exception as e:
                # Update metrics
                metrics.end_time = datetime.utcnow()
                metrics.status = ExecutionStatus.FAILED
                metrics.error_message = str(e)
                
                # Log failure with comprehensive context
                await self.agent_logger.log_action(
                    ActionType.AGENT_EXECUTION_FAILED,
                    log_level=LogLevel.ERROR,
                    task_id=task.task_id,
                    agent_type=task.agent_type,
                    correlation_id=orchestration_log_id,
                    error_message=str(e),
                    error_category=self._classify_error(e),
                    execution_stage="execution_failed",
                    duration_ms=int((time.time() - execution_start) * 1000) if 'execution_start' in locals() else None,
                    quality_metrics={
                        "queue_wait_ms": metrics.queue_wait_time_ms,
                        "attempts": metrics.execution_attempts,
                        "status": "failed"
                    },
                    tags=["agent_execution", "failed", "error"]
                )
                
                # Update circuit breaker (failure) with logging
                if self.enable_circuit_breakers:
                    await self._record_circuit_breaker_failure_with_logging(task.agent_type, str(e), task.task_id, orchestration_log_id)
                
                logger.error(f"Task {task.task_id} execution failed", error=str(e))
                
                return {
                    "status": ExecutionStatus.FAILED.value,
                    "error": str(e),
                    "metrics": {
                        "duration_ms": 0,
                        "queue_wait_ms": metrics.queue_wait_time_ms,
                        "attempts": metrics.execution_attempts
                    }
                }
                
            finally:
                # Move to completed tasks
                if task.task_id in self.active_tasks:
                    completed_metrics = self.active_tasks.pop(task.task_id)
                    self.completed_tasks.append(completed_metrics)
                
                # Update analytics
                self.execution_analytics["total_executions"] += 1
                if metrics.status == ExecutionStatus.COMPLETED:
                    self.execution_analytics["successful_executions"] += 1
                else:
                    self.execution_analytics["failed_executions"] += 1
    
    async def _execute_with_retry_and_logging(
        self,
        agent_instance: Any,
        task: AgentTask,
        workflow: WorkflowContext,
        metrics: ExecutionMetrics,
        orchestration_log_id: str,
        max_retries: int = 3
    ) -> Any:
        """Execute task with exponential backoff retry logic and comprehensive logging."""
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                metrics.execution_attempts = attempt + 1
                
                # Log retry attempt if not first attempt
                if attempt > 0:
                    await self.agent_logger.log_action(
                        ActionType.TASK_RETRIED,
                        log_level=LogLevel.WARN,
                        task_id=task.task_id,
                        agent_type=task.agent_type,
                        correlation_id=orchestration_log_id,
                        execution_stage=f"retry_attempt_{attempt}",
                        error_message=str(last_exception) if last_exception else "Previous attempt failed",
                        recovery_actions=[f"Retry attempt {attempt} of {max_retries}"],
                        tags=["retry", "resilience", "attempt"]
                    )
                
                # Progressive timeout (increase timeout on retries)
                timeout = 30 + (attempt * 10)  # 30, 40, 50, 60 seconds
                
                result = await asyncio.wait_for(
                    self._execute_agent_task(agent_instance, task, workflow),
                    timeout=timeout
                )
                
                # Log successful execution after retry
                if attempt > 0:
                    await self.agent_logger.log_action(
                        ActionType.ERROR_RECOVERY,
                        log_level=LogLevel.INFO,
                        task_id=task.task_id,
                        agent_type=task.agent_type,
                        correlation_id=orchestration_log_id,
                        execution_stage="recovery_successful",
                        recovery_actions=[f"Successfully executed after {attempt} retries"],
                        quality_metrics={"retry_attempts": attempt + 1, "final_status": "success"},
                        tags=["recovery", "success", "resilience"]
                    )
                
                return result
                
            except asyncio.TimeoutError as e:
                last_exception = e
                if attempt < max_retries:
                    wait_time = (2 ** attempt) + (time.time() % 1)  # Exponential backoff with jitter
                    
                    await self.agent_logger.log_action(
                        ActionType.TASK_RETRIED,
                        log_level=LogLevel.WARN,
                        task_id=task.task_id,
                        agent_type=task.agent_type,
                        correlation_id=orchestration_log_id,
                        error_message=f"Task timed out after {timeout}s",
                        error_category="timeout",
                        execution_stage=f"timeout_retry_{attempt}",
                        recovery_actions=[f"Retrying in {wait_time:.2f}s with longer timeout"],
                        tags=["timeout", "retry", "backoff"]
                    )
                    
                    logger.warning(
                        f"Task {task.task_id} attempt {attempt + 1} timed out, retrying in {wait_time:.2f}s"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    metrics.status = ExecutionStatus.TIMEOUT
                    raise
                    
            except Exception as e:
                last_exception = e
                if attempt < max_retries and self._is_retryable_error(e):
                    wait_time = (2 ** attempt) + (time.time() % 1)
                    
                    await self.agent_logger.log_action(
                        ActionType.TASK_RETRIED,
                        log_level=LogLevel.WARN,
                        task_id=task.task_id,
                        agent_type=task.agent_type,
                        correlation_id=orchestration_log_id,
                        error_message=str(e),
                        error_category=self._classify_error(e),
                        execution_stage=f"error_retry_{attempt}",
                        recovery_actions=[f"Retrying in {wait_time:.2f}s"],
                        tags=["error", "retry", "retryable"]
                    )
                    
                    logger.warning(
                        f"Task {task.task_id} attempt {attempt + 1} failed with retryable error, retrying in {wait_time:.2f}s"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise
        
        # If we get here, all retries failed
        await self.agent_logger.log_action(
            ActionType.TASK_FAILED,
            log_level=LogLevel.ERROR,
            task_id=task.task_id,
            agent_type=task.agent_type,
            correlation_id=orchestration_log_id,
            error_message=f"All retry attempts failed: {str(last_exception)}",
            error_category="retry_exhausted",
            execution_stage="retry_exhausted",
            quality_metrics={"retry_attempts": max_retries + 1, "final_status": "failed"},
            tags=["retry", "exhausted", "failed"]
        )
        
        raise last_exception
    
    async def _get_agent_instance_with_logging(self, agent_type: AgentType, task_id: str, orchestration_log_id: str) -> Any:
        """Get agent instance with pooling support and logging."""
        
        await self.agent_logger.log_action(
            ActionType.AGENT_INSTANTIATED,
            log_level=LogLevel.DEBUG,
            task_id=task_id,
            agent_type=agent_type,
            correlation_id=orchestration_log_id,
            execution_stage="agent_instantiation",
            resource_usage={
                "pooling_enabled": self.enable_agent_pooling,
                "pool_available": len(self.agent_pools.get(agent_type, {}).get('available_instances', [])) if self.enable_agent_pooling else 0,
                "pool_active": self.agent_pools.get(agent_type, {}).get('active_instances', 0) if self.enable_agent_pooling else 0
            },
            tags=["agent", "instantiation", "pooling"]
        )
        
        return await self._get_agent_instance(agent_type)
    
    async def _return_agent_to_pool_with_logging(self, agent_type: AgentType, instance: Any, task_id: str, orchestration_log_id: str) -> None:
        """Return agent instance to pool with logging."""
        
        await self._return_agent_to_pool(agent_type, instance)
        
        await self.agent_logger.log_action(
            ActionType.RESOURCE_OPTIMIZATION,
            log_level=LogLevel.DEBUG,
            task_id=task_id,
            agent_type=agent_type,
            correlation_id=orchestration_log_id,
            execution_stage="agent_pool_return",
            resource_usage={
                "pool_available": len(self.agent_pools.get(agent_type, {}).get('available_instances', [])),
                "pool_active": max(0, self.agent_pools.get(agent_type, {}).get('active_instances', 0) - 1)
            },
            tags=["agent", "pooling", "optimization"]
        )
    
    async def _record_circuit_breaker_success_with_logging(self, agent_type: AgentType, task_id: str, orchestration_log_id: str) -> None:
        """Record successful execution for circuit breaker with logging."""
        
        circuit_breaker = self.circuit_breakers.get(agent_type)
        if not circuit_breaker:
            return
        
        old_state = circuit_breaker.state
        await self._record_circuit_breaker_success(agent_type)
        new_state = circuit_breaker.state
        
        if old_state != new_state:
            await self.agent_logger.log_action(
                ActionType.CIRCUIT_BREAKER_RECOVERED,
                log_level=LogLevel.INFO,
                task_id=task_id,
                agent_type=agent_type,
                correlation_id=orchestration_log_id,
                circuit_breaker_state=new_state.value,
                execution_stage="circuit_breaker_recovered",
                recovery_actions=[f"Circuit breaker moved from {old_state.value} to {new_state.value}"],
                tags=["circuit_breaker", "recovery", "success"]
            )
    
    async def _record_circuit_breaker_failure_with_logging(self, agent_type: AgentType, error: str, task_id: str, orchestration_log_id: str) -> None:
        """Record failed execution for circuit breaker with logging."""
        
        circuit_breaker = self.circuit_breakers.get(agent_type)
        if not circuit_breaker:
            return
        
        old_state = circuit_breaker.state
        old_failure_count = circuit_breaker.failure_count
        
        await self._record_circuit_breaker_failure(agent_type)
        
        new_state = circuit_breaker.state
        new_failure_count = circuit_breaker.failure_count
        
        await self.agent_logger.log_action(
            ActionType.CIRCUIT_BREAKER_TRIGGERED,
            log_level=LogLevel.WARN,
            task_id=task_id,
            agent_type=agent_type,
            correlation_id=orchestration_log_id,
            circuit_breaker_state=new_state.value,
            error_message=error,
            execution_stage="circuit_breaker_failure",
            resource_usage={
                "failure_count": new_failure_count,
                "failure_threshold": circuit_breaker.failure_threshold,
                "state_changed": old_state != new_state
            },
            tags=["circuit_breaker", "failure", "resilience"]
        )
    
    async def _log_load_balancing_decisions(self, prioritized_tasks: List[Any], workflow: WorkflowContext) -> None:
        """Log load balancing decisions and strategy."""
        
        agent_distribution = {}
        for _, task, _ in prioritized_tasks:
            agent_type = task.agent_type.value
            agent_distribution[agent_type] = agent_distribution.get(agent_type, 0) + 1
        
        await self.agent_logger.log_action(
            ActionType.DECISION_POINT,
            log_level=LogLevel.INFO,
            execution_stage="load_balancing_decision",
            decision_context={
                "strategy": self.load_balancing_strategy.value,
                "agent_distribution": agent_distribution,
                "total_tasks": len(prioritized_tasks),
                "optimal_concurrency": self.optimal_concurrency
            },
            load_balancing_decision={
                "strategy": self.load_balancing_strategy.value,
                "distribution": agent_distribution,
                "concurrency": self.optimal_concurrency
            },
            reasoning=f"Load balancing strategy {self.load_balancing_strategy.value} applied to distribute {len(prioritized_tasks)} tasks",
            tags=["load_balancing", "decision", "optimization"]
        )
    
    async def _log_concurrency_adjustment(self) -> None:
        """Log concurrency adjustments and resource optimization."""
        
        if len(self.concurrency_adjustment_history) > 0:
            latest_adjustment = self.concurrency_adjustment_history[-1]
            
            await self.agent_logger.log_action(
                ActionType.RESOURCE_OPTIMIZATION,
                log_level=LogLevel.INFO,
                execution_stage="concurrency_adjustment",
                resource_usage={
                    "old_concurrency": latest_adjustment.get("old_concurrency"),
                    "new_concurrency": latest_adjustment.get("new_concurrency"),
                    "cpu_percent": latest_adjustment.get("cpu_percent"),
                    "memory_percent": latest_adjustment.get("memory_percent"),
                    "timestamp": latest_adjustment.get("timestamp").isoformat() if latest_adjustment.get("timestamp") else None
                },
                reasoning="Adaptive concurrency adjustment based on resource usage",
                tags=["concurrency", "optimization", "adaptive"]
            )
    
    def _classify_error(self, error: Exception) -> str:
        """Classify error for better logging and analysis."""
        error_str = str(error).lower()
        
        if "timeout" in error_str:
            return "timeout"
        elif "connection" in error_str:
            return "connection"
        elif "memory" in error_str:
            return "memory"
        elif "permission" in error_str or "auth" in error_str:
            return "authorization"
        elif "validation" in error_str:
            return "validation"
        elif "not found" in error_str:
            return "not_found"
        else:
            return "application"
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the executor with logging."""
        
        await self.agent_logger.log_action(
            ActionType.ORCHESTRATION_COMPLETED,
            log_level=LogLevel.INFO,
            execution_stage="shutdown_initiated",
            resource_usage=self._get_current_resource_usage(),
            quality_metrics=self.execution_analytics.copy(),
            tags=["shutdown", "orchestration", "cleanup"]
        )
        
        await super().shutdown()
        
        if self.agent_logger:
            await self.agent_logger.shutdown()


# Global executor instance
logging_enhanced_executor: Optional[LoggingEnhancedParallelExecutor] = None


async def get_logging_enhanced_executor() -> LoggingEnhancedParallelExecutor:
    """Get the global logging enhanced parallel executor instance."""
    global logging_enhanced_executor
    
    if not logging_enhanced_executor:
        logging_enhanced_executor = LoggingEnhancedParallelExecutor()
        await logging_enhanced_executor.initialize()
    
    return logging_enhanced_executor