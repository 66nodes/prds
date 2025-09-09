"""
Enhanced Parallel Agent Execution Engine

Advanced asyncio-based parallel execution system with dynamic load balancing,
adaptive concurrency control, circuit breaker patterns, and comprehensive monitoring.
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
from contextlib import asynccontextmanager

import structlog
from services.agent_orchestrator import AgentType, AgentTask, WorkflowContext
from services.agent_registry import get_agent_registry, AgentRegistry
from services.context_aware_agent_selector import get_context_aware_selector
from services.error_handling_service import ErrorHandlingService, ErrorContext, ErrorSeverity, ErrorCategory

logger = structlog.get_logger(__name__)


class ExecutionStatus(str, Enum):
    """Execution status for tasks and agents."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class PriorityLevel(str, Enum):
    """Priority levels for task execution."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit open, failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class ExecutionMetrics:
    """Metrics for execution tracking."""
    task_id: str
    agent_type: AgentType
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[int] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    error_message: Optional[str] = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    queue_wait_time_ms: int = 0
    execution_attempts: int = 0
    circuit_breaker_triggered: bool = False


@dataclass
class ResourceUsage:
    """Current system resource usage."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    active_tasks: int = 0
    queued_tasks: int = 0
    avg_response_time_ms: float = 0.0
    error_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CircuitBreaker:
    """Circuit breaker for agent resilience."""
    agent_type: AgentType
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    failure_threshold: int = 5
    recovery_timeout: int = 30  # seconds
    last_failure_time: Optional[datetime] = None
    half_open_max_calls: int = 3
    half_open_calls: int = 0


@dataclass
class AgentPool:
    """Pool of agent instances for reuse."""
    agent_type: AgentType
    pool_size: int = 3
    active_instances: int = 0
    available_instances: deque = field(default_factory=deque)
    creation_count: int = 0
    last_used: datetime = field(default_factory=datetime.utcnow)


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESPONSE_TIME_BASED = "response_time_based"
    RESOURCE_AWARE = "resource_aware"


class EnhancedParallelExecutor:
    """
    Enhanced parallel execution engine with advanced features:
    - Dynamic load balancing based on agent performance
    - Adaptive concurrency control with resource monitoring
    - Circuit breaker patterns for resilient error handling
    - Agent pooling for improved resource utilization
    - Real-time execution analytics and monitoring
    """

    def __init__(
        self,
        max_concurrent_tasks: int = 10,
        load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.RESOURCE_AWARE,
        enable_circuit_breakers: bool = True,
        enable_agent_pooling: bool = True,
        resource_monitoring_interval: float = 5.0
    ):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.load_balancing_strategy = load_balancing_strategy
        self.enable_circuit_breakers = enable_circuit_breakers
        self.enable_agent_pooling = enable_agent_pooling
        self.resource_monitoring_interval = resource_monitoring_interval

        # Core components
        self.agent_registry: Optional[AgentRegistry] = None
        self.agent_selector = None
        
        # Error handling service
        self.error_handler: Optional[ErrorHandlingService] = None
        
        # Execution management
        self.active_tasks: Dict[str, ExecutionMetrics] = {}
        self.completed_tasks: deque = deque(maxlen=1000)  # Keep last 1000 completed tasks
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.semaphore: Optional[asyncio.Semaphore] = None
        
        # Load balancing
        self.agent_loads: Dict[AgentType, int] = defaultdict(int)
        self.agent_response_times: Dict[AgentType, deque] = defaultdict(lambda: deque(maxlen=100))
        self.round_robin_counters: Dict[AgentType, int] = defaultdict(int)
        
        # Circuit breakers
        self.circuit_breakers: Dict[AgentType, CircuitBreaker] = {}
        
        # Agent pooling
        self.agent_pools: Dict[AgentType, AgentPool] = {}
        
        # Resource monitoring
        self.resource_usage_history: deque = deque(maxlen=100)
        self.resource_monitor_task: Optional[asyncio.Task] = None
        
        # Adaptive concurrency
        self.optimal_concurrency = max_concurrent_tasks
        self.concurrency_adjustment_history: deque = deque(maxlen=50)
        
        # Performance analytics
        self.execution_analytics: Dict[str, Any] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "avg_execution_time_ms": 0.0,
            "peak_concurrency": 0,
            "resource_efficiency": 0.0
        }

    async def initialize(self) -> None:
        """Initialize the enhanced parallel executor."""
        try:
            self.agent_registry = await get_agent_registry()
            self.agent_selector = await get_context_aware_selector()
            self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
            
            # Initialize error handling service
            self.error_handler = ErrorHandlingService()
            await self.error_handler.initialize()
            
            # Initialize circuit breakers for all agent types
            if self.enable_circuit_breakers:
                for agent_type in AgentType:
                    self.circuit_breakers[agent_type] = CircuitBreaker(agent_type=agent_type)
            
            # Initialize agent pools
            if self.enable_agent_pooling:
                await self._initialize_agent_pools()
            
            # Start resource monitoring
            self.resource_monitor_task = asyncio.create_task(self._monitor_resources())
            
            logger.info("Enhanced parallel executor initialized successfully")
            
        except Exception as e:
            error_context = ErrorContext(
                source="enhanced_parallel_executor",
                operation="initialize",
                agent_type="system",
                task_id="initialization",
                metadata={"error": str(e)}
            )
            
            if self.error_handler:
                await self.error_handler.handle_error(e, error_context)
            
            logger.error(f"Failed to initialize enhanced parallel executor: {str(e)}")
            raise

    async def _initialize_agent_pools(self) -> None:
        """Initialize agent pools for commonly used agents."""
        common_agents = [
            AgentType.DRAFT_AGENT,
            AgentType.JUDGE_AGENT,
            AgentType.BUSINESS_ANALYST,
            AgentType.PROJECT_ARCHITECT
        ]
        
        for agent_type in common_agents:
            self.agent_pools[agent_type] = AgentPool(
                agent_type=agent_type,
                pool_size=3
            )

    async def execute_parallel(
        self,
        tasks: List[AgentTask],
        workflow: WorkflowContext,
        priority: PriorityLevel = PriorityLevel.NORMAL,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute tasks in parallel with enhanced features.
        
        Args:
            tasks: List of agent tasks to execute
            workflow: Workflow context
            priority: Execution priority
            timeout: Optional timeout in seconds
            
        Returns:
            Dict containing execution results and analytics
        """
        if not tasks:
            return {"results": {}, "analytics": self.execution_analytics.copy()}
        
        logger.info(
            "Starting enhanced parallel execution",
            task_count=len(tasks),
            priority=priority.value,
            max_concurrent=self.optimal_concurrency
        )
        
        # Prepare tasks with priorities
        prioritized_tasks = await self._prepare_tasks_with_priority(tasks, workflow, priority)
        
        # Execute with dynamic load balancing
        try:
            if timeout:
                results = await asyncio.wait_for(
                    self._execute_with_load_balancing(prioritized_tasks, workflow),
                    timeout=timeout
                )
            else:
                results = await self._execute_with_load_balancing(prioritized_tasks, workflow)
                
            # Update analytics
            self._update_execution_analytics(results)
            
            return {
                "results": results,
                "analytics": self.execution_analytics.copy(),
                "resource_usage": self._get_current_resource_usage(),
                "circuit_breaker_status": self._get_circuit_breaker_status()
            }
            
        except asyncio.TimeoutError as e:
            logger.warning("Parallel execution timed out", timeout=timeout)
            
            # Handle timeout with error handling service
            if self.error_handler:
                error_context = ErrorContext(
                    source="enhanced_parallel_executor",
                    operation="execute_parallel",
                    agent_type="system",
                    task_id="parallel_execution",
                    metadata={
                        "timeout": timeout,
                        "task_count": len(tasks),
                        "completed_tasks": len([t for t in self.completed_tasks if t.status == ExecutionStatus.COMPLETED])
                    }
                )
                await self.error_handler.handle_error(e, error_context)
            
            return {
                "results": {},
                "error": "Execution timeout",
                "analytics": self.execution_analytics.copy()
            }
        except Exception as e:
            logger.error(f"Enhanced parallel execution failed: {str(e)}")
            
            # Handle general execution failure
            if self.error_handler:
                error_context = ErrorContext(
                    source="enhanced_parallel_executor",
                    operation="execute_parallel",
                    agent_type="system",
                    task_id="parallel_execution",
                    metadata={"task_count": len(tasks), "error": str(e)}
                )
                await self.error_handler.handle_error(e, error_context)
            
            raise

    async def _prepare_tasks_with_priority(
        self,
        tasks: List[AgentTask],
        workflow: WorkflowContext,
        priority: PriorityLevel
    ) -> List[Tuple[int, AgentTask, WorkflowContext]]:
        """Prepare tasks with priority scoring for queue ordering."""
        prioritized = []
        priority_values = {
            PriorityLevel.CRITICAL: 1,
            PriorityLevel.HIGH: 2,
            PriorityLevel.NORMAL: 3,
            PriorityLevel.LOW: 4
        }
        
        base_priority = priority_values[priority]
        
        for i, task in enumerate(tasks):
            # Adjust priority based on task characteristics
            task_priority = base_priority
            
            # Higher priority for tasks with dependencies
            if hasattr(task, 'dependencies') and task.dependencies:
                task_priority -= 1
            
            # Lower priority for resource-intensive tasks during high load
            current_load = len(self.active_tasks)
            if current_load > self.optimal_concurrency * 0.8:
                if hasattr(task, 'estimated_resource_cost') and task.estimated_resource_cost == 'high':
                    task_priority += 1
            
            prioritized.append((task_priority * 100 + i, task, workflow))
        
        return prioritized

    async def _execute_with_load_balancing(
        self,
        prioritized_tasks: List[Tuple[int, AgentTask, WorkflowContext]],
        workflow: WorkflowContext
    ) -> Dict[str, Any]:
        """Execute tasks with dynamic load balancing."""
        
        # Adaptive concurrency adjustment
        await self._adjust_optimal_concurrency()
        self.semaphore = asyncio.Semaphore(self.optimal_concurrency)
        
        # Create execution coroutines
        execution_tasks = []
        for priority_score, task, task_workflow in prioritized_tasks:
            execution_coro = self._execute_single_task_with_resilience(task, task_workflow)
            execution_tasks.append(execution_coro)
        
        # Execute with controlled concurrency
        results = {}
        completed_tasks = await asyncio.gather(*execution_tasks, return_exceptions=True)
        
        # Process results
        for i, (task_result, (_, task, _)) in enumerate(zip(completed_tasks, prioritized_tasks)):
            if isinstance(task_result, Exception):
                logger.error(f"Task {task.task_id} failed: {str(task_result)}")
                results[task.task_id] = {
                    "status": ExecutionStatus.FAILED.value,
                    "error": str(task_result)
                }
            else:
                results[task.task_id] = task_result
        
        return results

    async def _execute_single_task_with_resilience(
        self,
        task: AgentTask,
        workflow: WorkflowContext
    ) -> Dict[str, Any]:
        """Execute a single task with resilience patterns."""
        
        # Check circuit breaker
        if self.enable_circuit_breakers:
            circuit_breaker = self.circuit_breakers.get(task.agent_type)
            if circuit_breaker and not await self._check_circuit_breaker(circuit_breaker):
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
        
        # Acquire semaphore for concurrency control
        async with self.semaphore:
            self.active_tasks[task.task_id] = metrics
            
            try:
                # Get agent instance (with pooling if enabled)
                agent_instance = await self._get_agent_instance(task.agent_type)
                
                # Record queue wait time
                queue_wait = (datetime.utcnow() - metrics.start_time).total_seconds() * 1000
                metrics.queue_wait_time_ms = int(queue_wait)
                
                # Execute with timeout and retry logic
                execution_start = time.time()
                result = await self._execute_with_retry(
                    agent_instance, task, workflow, metrics
                )
                execution_end = time.time()
                
                # Update metrics
                metrics.end_time = datetime.utcnow()
                metrics.duration_ms = int((execution_end - execution_start) * 1000)
                metrics.status = ExecutionStatus.COMPLETED
                
                # Update load balancing metrics
                self._update_load_balancing_metrics(task.agent_type, metrics.duration_ms)
                
                # Update circuit breaker (success)
                if self.enable_circuit_breakers:
                    await self._record_circuit_breaker_success(task.agent_type)
                
                # Return agent instance to pool
                if self.enable_agent_pooling:
                    await self._return_agent_to_pool(task.agent_type, agent_instance)
                
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
                
                # Update circuit breaker (failure)
                if self.enable_circuit_breakers:
                    await self._record_circuit_breaker_failure(task.agent_type)
                
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

    async def _execute_with_retry(
        self,
        agent_instance: Any,
        task: AgentTask,
        workflow: WorkflowContext,
        metrics: ExecutionMetrics,
        max_retries: int = 3
    ) -> Any:
        """Execute task with comprehensive error handling and recovery."""
        
        error_context = ErrorContext(
            source="enhanced_parallel_executor",
            operation="_execute_with_retry",
            agent_type=task.agent_type.value,
            task_id=task.task_id,
            metadata={
                "attempt_limit": max_retries,
                "workflow_id": getattr(workflow, 'id', 'unknown')
            }
        )
        
        for attempt in range(max_retries + 1):
            try:
                metrics.execution_attempts = attempt + 1
                error_context.metadata["current_attempt"] = attempt + 1
                
                # Progressive timeout (increase timeout on retries)
                timeout = 30 + (attempt * 10)  # 30, 40, 50, 60 seconds
                
                result = await asyncio.wait_for(
                    self._execute_agent_task(agent_instance, task, workflow),
                    timeout=timeout
                )
                
                # Record successful execution if we had previous failures
                if attempt > 0 and self.error_handler:
                    await self.error_handler.record_recovery(
                        task.agent_type.value,
                        "retry_success",
                        {"attempts_required": attempt + 1}
                    )
                
                return result
                
            except Exception as e:
                error_context.metadata.update({
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "attempt": attempt + 1,
                    "timeout_used": timeout
                })
                
                # Use comprehensive error handling service
                if self.error_handler:
                    try:
                        recovery_result = await self.error_handler.handle_error(e, error_context)
                        
                        # Check if we should retry based on recovery strategy
                        if recovery_result.should_retry and attempt < max_retries:
                            wait_time = recovery_result.retry_delay or ((2 ** attempt) + (time.time() % 1))
                            
                            logger.warning(
                                f"Task {task.task_id} attempt {attempt + 1} failed, retrying in {wait_time:.2f}s",
                                error=str(e),
                                recovery_strategy=recovery_result.strategy.value
                            )
                            
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            # No retry recommended or max retries reached
                            if isinstance(e, asyncio.TimeoutError):
                                metrics.status = ExecutionStatus.TIMEOUT
                            raise e
                    except Exception as handler_error:
                        # If error handler fails, fall back to basic retry logic
                        logger.error(f"Error handler failed: {handler_error}")
                        if attempt < max_retries and self._is_retryable_error(e):
                            wait_time = (2 ** attempt) + (time.time() % 1)
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise e
                else:
                    # Fallback to basic retry logic if no error handler
                    if attempt < max_retries and self._is_retryable_error(e):
                        wait_time = (2 ** attempt) + (time.time() % 1)
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise e
        
        # Should not reach here with current logic
        raise RuntimeError(f"Task {task.task_id} failed after all retry attempts")

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is retryable."""
        retryable_patterns = [
            "ConnectionError",
            "TimeoutError",
            "TemporaryFailure",
            "ServiceUnavailable",
            "TooManyRequests"
        ]
        
        error_str = str(error)
        return any(pattern in error_str for pattern in retryable_patterns)

    async def _execute_agent_task(
        self, 
        agent_instance: Any, 
        task: AgentTask, 
        workflow: WorkflowContext
    ) -> Any:
        """Execute the actual agent task (placeholder for integration with agent system)."""
        # This would integrate with the actual agent execution system
        # For now, simulate execution
        await asyncio.sleep(0.1)  # Simulate work
        
        return {
            "agent_type": task.agent_type.value,
            "task_id": task.task_id,
            "result": f"Executed task {task.task_id} with agent {task.agent_type.value}",
            "timestamp": datetime.utcnow().isoformat()
        }

    async def _get_agent_instance(self, agent_type: AgentType) -> Any:
        """Get agent instance with pooling support."""
        if not self.enable_agent_pooling:
            return await self._create_agent_instance(agent_type)
        
        agent_pool = self.agent_pools.get(agent_type)
        if not agent_pool:
            # Create pool for this agent type
            agent_pool = AgentPool(agent_type=agent_type)
            self.agent_pools[agent_type] = agent_pool
        
        # Try to get from pool
        if agent_pool.available_instances:
            instance = agent_pool.available_instances.popleft()
            agent_pool.active_instances += 1
            agent_pool.last_used = datetime.utcnow()
            return instance
        
        # Create new instance if pool is empty
        if agent_pool.active_instances < agent_pool.pool_size:
            instance = await self._create_agent_instance(agent_type)
            agent_pool.active_instances += 1
            agent_pool.creation_count += 1
            return instance
        
        # Wait for available instance if pool is full
        while not agent_pool.available_instances:
            await asyncio.sleep(0.1)
        
        instance = agent_pool.available_instances.popleft()
        agent_pool.active_instances += 1
        return instance

    async def _create_agent_instance(self, agent_type: AgentType) -> Any:
        """Create a new agent instance."""
        # This would integrate with the actual agent creation system
        return {
            "agent_type": agent_type,
            "instance_id": f"{agent_type.value}_{time.time()}",
            "created_at": datetime.utcnow()
        }

    async def _return_agent_to_pool(self, agent_type: AgentType, instance: Any) -> None:
        """Return agent instance to pool."""
        agent_pool = self.agent_pools.get(agent_type)
        if agent_pool:
            agent_pool.available_instances.append(instance)
            agent_pool.active_instances = max(0, agent_pool.active_instances - 1)

    def _update_load_balancing_metrics(self, agent_type: AgentType, duration_ms: int) -> None:
        """Update metrics for load balancing decisions."""
        self.agent_response_times[agent_type].append(duration_ms)
        
        # Update agent load (decrement)
        self.agent_loads[agent_type] = max(0, self.agent_loads[agent_type] - 1)

    async def _check_circuit_breaker(self, circuit_breaker: CircuitBreaker) -> bool:
        """Check if circuit breaker allows execution with enhanced error handling integration."""
        now = datetime.utcnow()
        agent_type_str = circuit_breaker.agent_type.value
        
        # Check with error handling service if available
        if self.error_handler:
            is_healthy = await self.error_handler.check_agent_health(agent_type_str)
            if not is_healthy:
                # If error handler says agent is unhealthy, respect that
                if circuit_breaker.state == CircuitState.CLOSED:
                    circuit_breaker.state = CircuitState.OPEN
                    circuit_breaker.last_failure_time = now
                    logger.warning(f"Circuit breaker for {agent_type_str} opened due to health check failure")
                return False
        
        if circuit_breaker.state == CircuitState.CLOSED:
            return True
        elif circuit_breaker.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if (circuit_breaker.last_failure_time and 
                (now - circuit_breaker.last_failure_time).total_seconds() >= circuit_breaker.recovery_timeout):
                circuit_breaker.state = CircuitState.HALF_OPEN
                circuit_breaker.half_open_calls = 0
                logger.info(f"Circuit breaker for {agent_type_str} moved to HALF_OPEN")
                
                # Notify error handler of state change
                if self.error_handler:
                    await self.error_handler.update_circuit_breaker_state(
                        agent_type_str, "half_open", {"recovery_timeout": circuit_breaker.recovery_timeout}
                    )
                return True
            return False
        elif circuit_breaker.state == CircuitState.HALF_OPEN:
            if circuit_breaker.half_open_calls < circuit_breaker.half_open_max_calls:
                circuit_breaker.half_open_calls += 1
                return True
            return False
        
        return False

    async def _record_circuit_breaker_success(self, agent_type: AgentType) -> None:
        """Record successful execution for circuit breaker with error handling integration."""
        circuit_breaker = self.circuit_breakers.get(agent_type)
        if not circuit_breaker:
            return
        
        agent_type_str = agent_type.value
        
        # Notify error handling service of success
        if self.error_handler:
            await self.error_handler.record_recovery(
                agent_type_str, 
                "circuit_breaker_success",
                {"previous_state": circuit_breaker.state.value}
            )
        
        if circuit_breaker.state == CircuitState.HALF_OPEN:
            # If we've had enough successful calls, close the circuit
            if circuit_breaker.half_open_calls >= circuit_breaker.half_open_max_calls:
                circuit_breaker.state = CircuitState.CLOSED
                circuit_breaker.failure_count = 0
                logger.info(f"Circuit breaker for {agent_type_str} moved to CLOSED")
                
                # Notify error handler of state change
                if self.error_handler:
                    await self.error_handler.update_circuit_breaker_state(
                        agent_type_str, "closed", {"successful_calls": circuit_breaker.half_open_calls}
                    )
        
        # Reset failure count on success
        circuit_breaker.failure_count = max(0, circuit_breaker.failure_count - 1)

    async def _record_circuit_breaker_failure(self, agent_type: AgentType) -> None:
        """Record failed execution for circuit breaker with error handling integration."""
        circuit_breaker = self.circuit_breakers.get(agent_type)
        if not circuit_breaker:
            return
        
        agent_type_str = agent_type.value
        circuit_breaker.failure_count += 1
        circuit_breaker.last_failure_time = datetime.utcnow()
        
        # Open circuit if threshold exceeded
        if (circuit_breaker.state == CircuitState.CLOSED and 
            circuit_breaker.failure_count >= circuit_breaker.failure_threshold):
            circuit_breaker.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker for {agent_type_str} moved to OPEN")
            
            # Notify error handler of state change
            if self.error_handler:
                await self.error_handler.update_circuit_breaker_state(
                    agent_type_str, "open", {
                        "failure_count": circuit_breaker.failure_count,
                        "threshold": circuit_breaker.failure_threshold
                    }
                )
        elif circuit_breaker.state == CircuitState.HALF_OPEN:
            # Go back to open state
            circuit_breaker.state = CircuitState.OPEN
            circuit_breaker.half_open_calls = 0
            logger.warning(f"Circuit breaker for {agent_type_str} moved back to OPEN from HALF_OPEN")
            
            # Notify error handler of state change
            if self.error_handler:
                await self.error_handler.update_circuit_breaker_state(
                    agent_type_str, "open", {"reason": "half_open_failure"}
                )

    async def _monitor_resources(self) -> None:
        """Monitor system resources and adjust concurrency."""
        while True:
            try:
                resource_usage = await self._collect_resource_metrics()
                self.resource_usage_history.append(resource_usage)
                
                # Log resource metrics periodically
                if len(self.resource_usage_history) % 12 == 0:  # Every minute at 5s intervals
                    logger.info("Resource usage update", **resource_usage.__dict__)
                
                await asyncio.sleep(self.resource_monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Resource monitoring error: {str(e)}")
                await asyncio.sleep(self.resource_monitoring_interval)

    async def _collect_resource_metrics(self) -> ResourceUsage:
        """Collect current system resource metrics."""
        # This would integrate with actual system monitoring
        # For now, simulate metrics based on current load
        
        active_count = len(self.active_tasks)
        
        # Simulate resource usage based on active tasks
        cpu_percent = min(95.0, active_count * 8.0)
        memory_percent = min(90.0, active_count * 5.0)
        
        # Calculate average response time
        all_response_times = []
        for times in self.agent_response_times.values():
            all_response_times.extend(list(times))
        
        avg_response_time = statistics.mean(all_response_times) if all_response_times else 0.0
        
        # Calculate error rate
        total_executions = self.execution_analytics["total_executions"]
        failed_executions = self.execution_analytics["failed_executions"]
        error_rate = (failed_executions / total_executions * 100) if total_executions > 0 else 0.0
        
        return ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            active_tasks=active_count,
            queued_tasks=self.task_queue.qsize(),
            avg_response_time_ms=avg_response_time,
            error_rate=error_rate
        )

    async def _adjust_optimal_concurrency(self) -> None:
        """Dynamically adjust optimal concurrency based on resource usage."""
        if len(self.resource_usage_history) < 5:
            return
        
        recent_usage = list(self.resource_usage_history)[-5:]
        avg_cpu = statistics.mean([r.cpu_percent for r in recent_usage])
        avg_memory = statistics.mean([r.memory_percent for r in recent_usage])
        avg_response_time = statistics.mean([r.avg_response_time_ms for r in recent_usage])
        
        current_concurrency = self.optimal_concurrency
        
        # Increase concurrency if resources allow
        if avg_cpu < 60 and avg_memory < 70 and avg_response_time < 1000:
            self.optimal_concurrency = min(self.max_concurrent_tasks, current_concurrency + 1)
        
        # Decrease concurrency if resources are strained
        elif avg_cpu > 80 or avg_memory > 85 or avg_response_time > 3000:
            self.optimal_concurrency = max(1, current_concurrency - 1)
        
        # Log adjustment
        if self.optimal_concurrency != current_concurrency:
            logger.info(
                "Adjusted optimal concurrency",
                old_concurrency=current_concurrency,
                new_concurrency=self.optimal_concurrency,
                cpu_percent=avg_cpu,
                memory_percent=avg_memory,
                avg_response_time_ms=avg_response_time
            )
            
            self.concurrency_adjustment_history.append({
                "timestamp": datetime.utcnow(),
                "old_concurrency": current_concurrency,
                "new_concurrency": self.optimal_concurrency,
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory
            })

    def _update_execution_analytics(self, results: Dict[str, Any]) -> None:
        """Update execution analytics with results."""
        if not results:
            return
        
        # Calculate success rate
        total_results = len(results)
        successful_results = sum(1 for r in results.values() if r.get("status") == "completed")
        
        # Update peak concurrency
        current_active = len(self.active_tasks)
        self.execution_analytics["peak_concurrency"] = max(
            self.execution_analytics["peak_concurrency"], 
            current_active
        )
        
        # Update average execution time
        execution_times = []
        for result in results.values():
            if "metrics" in result and "duration_ms" in result["metrics"]:
                execution_times.append(result["metrics"]["duration_ms"])
        
        if execution_times:
            avg_time = statistics.mean(execution_times)
            current_avg = self.execution_analytics["avg_execution_time_ms"]
            # Exponential moving average
            self.execution_analytics["avg_execution_time_ms"] = (
                0.8 * current_avg + 0.2 * avg_time
            ) if current_avg > 0 else avg_time
        
        # Update resource efficiency (tasks completed per unit of resource usage)
        if self.resource_usage_history:
            recent_usage = self.resource_usage_history[-1]
            resource_usage = (recent_usage.cpu_percent + recent_usage.memory_percent) / 2
            if resource_usage > 0:
                efficiency = successful_results / resource_usage * 100
                self.execution_analytics["resource_efficiency"] = (
                    0.9 * self.execution_analytics["resource_efficiency"] + 0.1 * efficiency
                ) if self.execution_analytics["resource_efficiency"] > 0 else efficiency

    def _get_current_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage summary."""
        if not self.resource_usage_history:
            return {}
        
        latest = self.resource_usage_history[-1]
        return {
            "cpu_percent": latest.cpu_percent,
            "memory_percent": latest.memory_percent,
            "active_tasks": latest.active_tasks,
            "queued_tasks": latest.queued_tasks,
            "avg_response_time_ms": latest.avg_response_time_ms,
            "error_rate": latest.error_rate,
            "optimal_concurrency": self.optimal_concurrency
        }

    def _get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status summary."""
        status = {}
        for agent_type, cb in self.circuit_breakers.items():
            status[agent_type.value] = {
                "state": cb.state.value,
                "failure_count": cb.failure_count,
                "last_failure": cb.last_failure_time.isoformat() if cb.last_failure_time else None
            }
        return status

    async def get_execution_status(self) -> Dict[str, Any]:
        """Get comprehensive execution status."""
        return {
            "active_tasks": {
                task_id: {
                    "agent_type": metrics.agent_type.value,
                    "start_time": metrics.start_time.isoformat(),
                    "duration_ms": int((datetime.utcnow() - metrics.start_time).total_seconds() * 1000),
                    "status": metrics.status.value
                }
                for task_id, metrics in self.active_tasks.items()
            },
            "resource_usage": self._get_current_resource_usage(),
            "circuit_breakers": self._get_circuit_breaker_status(),
            "analytics": self.execution_analytics.copy(),
            "optimal_concurrency": self.optimal_concurrency,
            "agent_pools": {
                agent_type.value: {
                    "active_instances": pool.active_instances,
                    "available_instances": len(pool.available_instances),
                    "creation_count": pool.creation_count,
                    "last_used": pool.last_used.isoformat()
                }
                for agent_type, pool in self.agent_pools.items()
            }
        }

    async def enable_graceful_degradation(self) -> bool:
        """Enable graceful degradation mode during high error rates or resource constraints."""
        if not self.error_handler:
            return False
            
        try:
            # Check system health and enable degradation if needed
            system_health = await self._assess_system_health()
            
            if system_health["error_rate"] > 20 or system_health["resource_usage"] > 85:
                # Reduce concurrency to 50% of current optimal
                degraded_concurrency = max(1, self.optimal_concurrency // 2)
                self.optimal_concurrency = degraded_concurrency
                self.semaphore = asyncio.Semaphore(degraded_concurrency)
                
                logger.warning(
                    "Graceful degradation enabled",
                    error_rate=system_health["error_rate"],
                    resource_usage=system_health["resource_usage"],
                    new_concurrency=degraded_concurrency
                )
                
                # Notify error handler
                await self.error_handler.trigger_degradation(
                    "enhanced_parallel_executor",
                    "high_error_rate_or_resource_usage",
                    {
                        "error_rate": system_health["error_rate"],
                        "resource_usage": system_health["resource_usage"],
                        "new_concurrency": degraded_concurrency
                    }
                )
                
                return True
        except Exception as e:
            logger.error(f"Failed to enable graceful degradation: {e}")
            
        return False

    async def _assess_system_health(self) -> Dict[str, float]:
        """Assess current system health metrics."""
        # Calculate error rate from recent executions
        total = self.execution_analytics["total_executions"]
        failed = self.execution_analytics["failed_executions"]
        error_rate = (failed / total * 100) if total > 0 else 0
        
        # Get current resource usage
        resource_usage = 0.0
        if self.resource_usage_history:
            latest = self.resource_usage_history[-1]
            resource_usage = (latest.cpu_percent + latest.memory_percent) / 2
        
        return {
            "error_rate": error_rate,
            "resource_usage": resource_usage,
            "active_tasks": len(self.active_tasks),
            "circuit_breakers_open": len([
                cb for cb in self.circuit_breakers.values() 
                if cb.state == CircuitState.OPEN
            ])
        }

    async def get_error_handling_status(self) -> Dict[str, Any]:
        """Get comprehensive error handling status."""
        status = {
            "error_handler_active": self.error_handler is not None,
            "circuit_breakers": self._get_circuit_breaker_status(),
            "system_health": await self._assess_system_health(),
            "degradation_active": self.optimal_concurrency < self.max_concurrent_tasks
        }
        
        if self.error_handler:
            status.update({
                "error_analytics": await self.error_handler.get_error_analytics(),
                "recovery_status": await self.error_handler.get_recovery_status()
            })
            
        return status

    async def force_recovery_attempt(self, agent_type: str) -> bool:
        """Force a recovery attempt for a specific agent type."""
        if not self.error_handler:
            return False
            
        try:
            # Reset circuit breaker if it exists
            for agent_enum_type, circuit_breaker in self.circuit_breakers.items():
                if agent_enum_type.value == agent_type and circuit_breaker.state == CircuitState.OPEN:
                    circuit_breaker.state = CircuitState.HALF_OPEN
                    circuit_breaker.half_open_calls = 0
                    circuit_breaker.failure_count = 0
                    
                    logger.info(f"Forced recovery attempt for {agent_type}")
                    
                    await self.error_handler.record_recovery(
                        agent_type, "forced_recovery", {"initiated_by": "admin"}
                    )
                    
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Failed to force recovery for {agent_type}: {e}")
            return False

    async def shutdown(self) -> None:
        """Gracefully shutdown the executor."""
        logger.info("Shutting down enhanced parallel executor")
        
        # Shutdown error handling service first
        if self.error_handler:
            try:
                await self.error_handler.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down error handler: {e}")
        
        # Cancel resource monitoring
        if self.resource_monitor_task:
            self.resource_monitor_task.cancel()
            try:
                await self.resource_monitor_task
            except asyncio.CancelledError:
                pass
        
        # Wait for active tasks to complete (with timeout)
        if self.active_tasks:
            logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete")
            
            wait_start = time.time()
            timeout = 30  # 30 second graceful shutdown timeout
            
            while self.active_tasks and (time.time() - wait_start) < timeout:
                await asyncio.sleep(1)
            
            if self.active_tasks:
                logger.warning(f"Shutdown timeout reached, {len(self.active_tasks)} tasks still active")
        
        logger.info("Enhanced parallel executor shutdown complete")


# Global executor instance
enhanced_executor: Optional[EnhancedParallelExecutor] = None


async def get_enhanced_executor() -> EnhancedParallelExecutor:
    """Get the global enhanced parallel executor instance with comprehensive error handling."""
    global enhanced_executor
    
    if not enhanced_executor:
        enhanced_executor = EnhancedParallelExecutor(
            max_concurrent_tasks=20,  # Increased for production workloads
            enable_circuit_breakers=True,
            enable_agent_pooling=True,
            resource_monitoring_interval=3.0  # More frequent monitoring
        )
        await enhanced_executor.initialize()
        
        # Enable automatic graceful degradation monitoring
        async def _monitor_degradation():
            while True:
                try:
                    await enhanced_executor.enable_graceful_degradation()
                    await asyncio.sleep(30)  # Check every 30 seconds
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Degradation monitoring error: {e}")
                    await asyncio.sleep(30)
        
        # Start degradation monitoring task
        asyncio.create_task(_monitor_degradation())
    
    return enhanced_executor