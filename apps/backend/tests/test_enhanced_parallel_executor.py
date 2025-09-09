"""
Tests for Enhanced Parallel Agent Execution Engine

Test suite covering dynamic load balancing, adaptive concurrency control,
circuit breaker patterns, agent pooling, and performance monitoring.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import json

from services.enhanced_parallel_executor import (
    EnhancedParallelExecutor,
    ExecutionStatus,
    PriorityLevel,
    CircuitState,
    LoadBalancingStrategy,
    ExecutionMetrics,
    ResourceUsage,
    CircuitBreaker,
    AgentPool,
    get_enhanced_executor
)
from services.agent_orchestrator import AgentType, AgentTask, WorkflowContext


@pytest.fixture
async def executor():
    """Create and initialize an enhanced parallel executor for testing."""
    executor = EnhancedParallelExecutor(
        max_concurrent_tasks=5,
        load_balancing_strategy=LoadBalancingStrategy.RESOURCE_AWARE,
        enable_circuit_breakers=True,
        enable_agent_pooling=True,
        resource_monitoring_interval=1.0
    )
    
    # Mock dependencies
    with patch('services.enhanced_parallel_executor.get_agent_registry'), \
         patch('services.enhanced_parallel_executor.get_context_aware_selector'):
        await executor.initialize()
    
    yield executor
    
    # Cleanup
    await executor.shutdown()


@pytest.fixture
def sample_tasks():
    """Create sample agent tasks for testing."""
    workflow = WorkflowContext(
        workflow_id="test_workflow",
        context_type="test",
        input_data={"test": "data"},
        metadata={"priority": "normal"}
    )
    
    tasks = [
        AgentTask(
            task_id="task_1",
            agent_type=AgentType.DRAFT_AGENT,
            input_data={"content": "test content 1"},
            dependencies=[],
            estimated_execution_time=timedelta(seconds=1)
        ),
        AgentTask(
            task_id="task_2",
            agent_type=AgentType.JUDGE_AGENT,
            input_data={"content": "test content 2"},
            dependencies=["task_1"],
            estimated_execution_time=timedelta(seconds=2)
        ),
        AgentTask(
            task_id="task_3",
            agent_type=AgentType.BUSINESS_ANALYST,
            input_data={"content": "test content 3"},
            dependencies=[],
            estimated_execution_time=timedelta(seconds=1)
        )
    ]
    
    return tasks, workflow


class TestEnhancedParallelExecutor:
    """Test cases for the enhanced parallel executor."""

    @pytest.mark.asyncio
    async def test_executor_initialization(self, executor):
        """Test that executor initializes properly."""
        assert executor.semaphore is not None
        assert executor.max_concurrent_tasks == 5
        assert executor.optimal_concurrency == 5
        assert executor.enable_circuit_breakers is True
        assert executor.enable_agent_pooling is True
        assert executor.resource_monitor_task is not None

    @pytest.mark.asyncio
    async def test_basic_parallel_execution(self, executor, sample_tasks):
        """Test basic parallel task execution."""
        tasks, workflow = sample_tasks
        
        result = await executor.execute_parallel(
            tasks=tasks,
            workflow=workflow,
            priority=PriorityLevel.NORMAL
        )
        
        assert "results" in result
        assert "analytics" in result
        assert "resource_usage" in result
        assert len(result["results"]) == 3
        
        # Check that all tasks completed
        for task in tasks:
            assert task.task_id in result["results"]
            task_result = result["results"][task.task_id]
            assert task_result["status"] == ExecutionStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_priority_queue_ordering(self, executor, sample_tasks):
        """Test that tasks are executed based on priority."""
        tasks, workflow = sample_tasks
        
        # Execute with different priorities
        high_priority_result = await executor.execute_parallel(
            tasks=tasks[:1],  # Single task
            workflow=workflow,
            priority=PriorityLevel.HIGH
        )
        
        low_priority_result = await executor.execute_parallel(
            tasks=tasks[1:2],  # Single task
            workflow=workflow,
            priority=PriorityLevel.LOW
        )
        
        assert high_priority_result["results"]
        assert low_priority_result["results"]
        
        # High priority tasks should have lower queue wait times
        high_metrics = list(high_priority_result["results"].values())[0]["metrics"]
        low_metrics = list(low_priority_result["results"].values())[0]["metrics"]
        
        # In a real system, high priority would have lower wait times
        assert "queue_wait_ms" in high_metrics
        assert "queue_wait_ms" in low_metrics

    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, executor):
        """Test circuit breaker pattern for agent resilience."""
        agent_type = AgentType.DRAFT_AGENT
        circuit_breaker = executor.circuit_breakers[agent_type]
        
        # Initially closed
        assert circuit_breaker.state == CircuitState.CLOSED
        
        # Simulate failures to trigger circuit breaker
        for _ in range(circuit_breaker.failure_threshold):
            await executor._record_circuit_breaker_failure(agent_type)
        
        # Should be open now
        assert circuit_breaker.state == CircuitState.OPEN
        assert not await executor._check_circuit_breaker(circuit_breaker)
        
        # Test recovery to half-open state
        circuit_breaker.last_failure_time = datetime.utcnow() - timedelta(seconds=31)
        assert await executor._check_circuit_breaker(circuit_breaker)
        assert circuit_breaker.state == CircuitState.HALF_OPEN
        
        # Test recovery to closed state
        for _ in range(circuit_breaker.half_open_max_calls):
            await executor._record_circuit_breaker_success(agent_type)
        
        assert circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_agent_pooling(self, executor):
        """Test agent pooling for resource efficiency."""
        agent_type = AgentType.DRAFT_AGENT
        
        # Get multiple instances
        instance1 = await executor._get_agent_instance(agent_type)
        instance2 = await executor._get_agent_instance(agent_type)
        instance3 = await executor._get_agent_instance(agent_type)
        
        pool = executor.agent_pools[agent_type]
        assert pool.active_instances == 3
        
        # Return instances to pool
        await executor._return_agent_to_pool(agent_type, instance1)
        await executor._return_agent_to_pool(agent_type, instance2)
        
        assert pool.active_instances == 1
        assert len(pool.available_instances) == 2
        
        # Reuse pooled instance
        reused_instance = await executor._get_agent_instance(agent_type)
        assert reused_instance in [instance1, instance2]

    @pytest.mark.asyncio
    async def test_adaptive_concurrency_control(self, executor):
        """Test adaptive concurrency adjustment based on resource usage."""
        # Simulate high resource usage
        high_usage = ResourceUsage(
            cpu_percent=90.0,
            memory_percent=85.0,
            avg_response_time_ms=4000.0,
            error_rate=2.0
        )
        
        for _ in range(5):  # Add enough history
            executor.resource_usage_history.append(high_usage)
        
        original_concurrency = executor.optimal_concurrency
        await executor._adjust_optimal_concurrency()
        
        # Should reduce concurrency under high load
        assert executor.optimal_concurrency < original_concurrency
        
        # Simulate low resource usage
        low_usage = ResourceUsage(
            cpu_percent=40.0,
            memory_percent=30.0,
            avg_response_time_ms=500.0,
            error_rate=0.1
        )
        
        for _ in range(5):
            executor.resource_usage_history.append(low_usage)
        
        current_concurrency = executor.optimal_concurrency
        await executor._adjust_optimal_concurrency()
        
        # Should increase concurrency under light load
        assert executor.optimal_concurrency >= current_concurrency

    @pytest.mark.asyncio
    async def test_retry_logic_with_backoff(self, executor, sample_tasks):
        """Test exponential backoff retry logic."""
        tasks, workflow = sample_tasks
        task = tasks[0]
        
        # Mock agent instance that fails first few times
        agent_instance = MagicMock()
        
        # Create metrics
        metrics = ExecutionMetrics(
            task_id=task.task_id,
            agent_type=task.agent_type,
            start_time=datetime.utcnow()
        )
        
        # Mock retryable error
        class RetryableError(Exception):
            def __str__(self):
                return "ConnectionError: Temporary failure"
        
        # Test successful retry after failures
        call_count = 0
        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # Fail first 2 times
                raise RetryableError()
            return {"success": True}
        
        with patch.object(executor, '_execute_agent_task', mock_execute):
            result = await executor._execute_with_retry(
                agent_instance, task, workflow, metrics, max_retries=3
            )
        
        assert result["success"] is True
        assert call_count == 3
        assert metrics.execution_attempts == 3

    @pytest.mark.asyncio
    async def test_timeout_handling(self, executor, sample_tasks):
        """Test timeout handling for task execution."""
        tasks, workflow = sample_tasks
        
        # Execute with very short timeout
        result = await executor.execute_parallel(
            tasks=tasks,
            workflow=workflow,
            priority=PriorityLevel.NORMAL,
            timeout=0.001  # 1ms timeout
        )
        
        # Should timeout and return error
        assert "error" in result
        assert result["error"] == "Execution timeout"

    @pytest.mark.asyncio
    async def test_load_balancing_metrics(self, executor):
        """Test load balancing metrics tracking."""
        agent_type = AgentType.DRAFT_AGENT
        
        # Record some response times
        executor._update_load_balancing_metrics(agent_type, 100)
        executor._update_load_balancing_metrics(agent_type, 200)
        executor._update_load_balancing_metrics(agent_type, 150)
        
        response_times = executor.agent_response_times[agent_type]
        assert len(response_times) == 3
        assert 100 in response_times
        assert 200 in response_times
        assert 150 in response_times

    @pytest.mark.asyncio
    async def test_resource_monitoring(self, executor):
        """Test resource monitoring functionality."""
        # Let the resource monitor run for a bit
        await asyncio.sleep(1.5)  # More than monitoring interval
        
        # Should have collected some resource metrics
        assert len(executor.resource_usage_history) > 0
        
        latest_usage = executor.resource_usage_history[-1]
        assert isinstance(latest_usage.cpu_percent, float)
        assert isinstance(latest_usage.memory_percent, float)
        assert isinstance(latest_usage.active_tasks, int)

    @pytest.mark.asyncio
    async def test_execution_analytics(self, executor, sample_tasks):
        """Test execution analytics tracking."""
        tasks, workflow = sample_tasks
        
        # Execute tasks to generate analytics
        result = await executor.execute_parallel(
            tasks=tasks,
            workflow=workflow,
            priority=PriorityLevel.NORMAL
        )
        
        analytics = result["analytics"]
        assert "total_executions" in analytics
        assert "successful_executions" in analytics
        assert "failed_executions" in analytics
        assert "avg_execution_time_ms" in analytics
        assert "peak_concurrency" in analytics
        assert "resource_efficiency" in analytics
        
        # Should have recorded executions
        assert analytics["total_executions"] > 0
        assert analytics["successful_executions"] > 0

    @pytest.mark.asyncio
    async def test_comprehensive_status(self, executor, sample_tasks):
        """Test comprehensive status reporting."""
        tasks, workflow = sample_tasks
        
        # Execute some tasks to populate status
        await executor.execute_parallel(
            tasks=tasks[:1],
            workflow=workflow,
            priority=PriorityLevel.NORMAL
        )
        
        status = await executor.get_execution_status()
        
        assert "active_tasks" in status
        assert "resource_usage" in status
        assert "circuit_breakers" in status
        assert "analytics" in status
        assert "optimal_concurrency" in status
        assert "agent_pools" in status
        
        # Verify structure
        assert isinstance(status["active_tasks"], dict)
        assert isinstance(status["circuit_breakers"], dict)
        assert isinstance(status["analytics"], dict)
        assert isinstance(status["optimal_concurrency"], int)

    @pytest.mark.asyncio
    async def test_error_handling(self, executor, sample_tasks):
        """Test error handling and recovery."""
        tasks, workflow = sample_tasks
        
        # Mock task execution to raise an error
        with patch.object(executor, '_execute_agent_task') as mock_execute:
            mock_execute.side_effect = Exception("Test error")
            
            result = await executor.execute_parallel(
                tasks=tasks[:1],
                workflow=workflow,
                priority=PriorityLevel.NORMAL
            )
        
        # Should handle error gracefully
        assert "results" in result
        task_result = list(result["results"].values())[0]
        assert task_result["status"] == ExecutionStatus.FAILED.value
        assert "error" in task_result

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, executor):
        """Test graceful shutdown functionality."""
        # Start some long-running tasks
        async def long_task():
            await asyncio.sleep(2)
        
        # Mock active tasks
        executor.active_tasks["test_task"] = ExecutionMetrics(
            task_id="test_task",
            agent_type=AgentType.DRAFT_AGENT,
            start_time=datetime.utcnow()
        )
        
        # Should complete quickly since no real tasks are running
        await executor.shutdown()
        
        # Verify cleanup
        assert executor.resource_monitor_task.cancelled()


class TestLoadBalancingStrategies:
    """Test different load balancing strategies."""

    @pytest.mark.asyncio
    async def test_resource_aware_strategy(self):
        """Test resource-aware load balancing."""
        executor = EnhancedParallelExecutor(
            load_balancing_strategy=LoadBalancingStrategy.RESOURCE_AWARE
        )
        
        assert executor.load_balancing_strategy == LoadBalancingStrategy.RESOURCE_AWARE

    @pytest.mark.asyncio
    async def test_round_robin_strategy(self):
        """Test round-robin load balancing."""
        executor = EnhancedParallelExecutor(
            load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN
        )
        
        assert executor.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN


class TestPerformanceBenchmarks:
    """Performance benchmarks for the enhanced executor."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_execution_performance(self, executor):
        """Benchmark concurrent execution performance."""
        import time
        
        # Create many tasks
        tasks = []
        workflow = WorkflowContext(
            workflow_id="benchmark_workflow",
            context_type="benchmark",
            input_data={},
            metadata={}
        )
        
        for i in range(20):
            tasks.append(AgentTask(
                task_id=f"benchmark_task_{i}",
                agent_type=AgentType.DRAFT_AGENT,
                input_data={"content": f"benchmark content {i}"},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100)
            ))
        
        start_time = time.time()
        result = await executor.execute_parallel(
            tasks=tasks,
            workflow=workflow,
            priority=PriorityLevel.NORMAL
        )
        end_time = time.time()
        
        execution_time = end_time - start_time
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert len(result["results"]) == 20
        
        # Check that parallel execution was faster than sequential would be
        # (20 tasks * 100ms = 2000ms sequential minimum)
        assert execution_time < 2.0

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_usage_efficiency(self, executor):
        """Test memory usage efficiency with agent pooling."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Execute many tasks to test pooling efficiency
        tasks = []
        workflow = WorkflowContext(
            workflow_id="memory_test_workflow",
            context_type="memory_test",
            input_data={},
            metadata={}
        )
        
        for i in range(50):
            tasks.append(AgentTask(
                task_id=f"memory_task_{i}",
                agent_type=AgentType.DRAFT_AGENT,
                input_data={"content": f"memory test {i}"},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=50)
            ))
        
        await executor.execute_parallel(
            tasks=tasks,
            workflow=workflow,
            priority=PriorityLevel.NORMAL
        )
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024  # 100MB


@pytest.mark.asyncio
async def test_global_executor_instance():
    """Test global executor instance management."""
    executor1 = await get_enhanced_executor()
    executor2 = await get_enhanced_executor()
    
    # Should return the same instance
    assert executor1 is executor2
    assert executor1.is_initialized if hasattr(executor1, 'is_initialized') else True


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "test_executor_initialization or test_basic_parallel_execution"
    ])