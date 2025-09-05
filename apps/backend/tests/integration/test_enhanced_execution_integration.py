"""
Integration tests for Enhanced Parallel Execution system
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from datetime import datetime

from services.enhanced_parallel_executor import (
    EnhancedParallelExecutor,
    PriorityLevel,
    LoadBalancingStrategy,
    get_enhanced_executor
)
from services.execution_benchmarker import (
    ExecutionBenchmarker,
    BenchmarkConfig,
    get_benchmarker
)
from services.agent_orchestrator import AgentType, AgentTask, WorkflowContext


@pytest.fixture
async def enhanced_executor():
    """Enhanced parallel executor fixture."""
    with patch('services.enhanced_parallel_executor.get_agent_registry'), \
         patch('services.enhanced_parallel_executor.get_context_aware_selector'):
        
        executor = EnhancedParallelExecutor(
            max_concurrent_tasks=5,
            enable_circuit_breakers=True,
            enable_agent_pooling=True
        )
        await executor.initialize()
        
        yield executor
        
        await executor.shutdown()


@pytest.fixture
def sample_tasks():
    """Sample agent tasks for testing."""
    return [
        AgentTask(
            task_id=f"test_task_{i}",
            agent_type=AgentType.DRAFT_AGENT,
            input_data={"test": f"data_{i}"},
            metadata={"test": True}
        )
        for i in range(10)
    ]


@pytest.fixture
def workflow_context():
    """Workflow context for testing."""
    return WorkflowContext(
        workflow_id="test_workflow",
        context_data={"test": True}
    )


class TestEnhancedParallelExecutor:
    """Test suite for enhanced parallel execution."""
    
    @pytest.mark.asyncio
    async def test_parallel_execution_basic(self, enhanced_executor, sample_tasks, workflow_context):
        """Test basic parallel execution functionality."""
        
        # Execute tasks
        result = await enhanced_executor.execute_parallel(
            tasks=sample_tasks[:5],
            workflow=workflow_context,
            priority=PriorityLevel.NORMAL
        )
        
        # Verify results
        assert "results" in result
        assert "analytics" in result
        assert "resource_usage" in result
        
        # Check that all tasks were executed
        results = result["results"]
        assert len(results) == 5
        
        # Verify analytics were updated
        analytics = result["analytics"]
        assert analytics["total_executions"] >= 5
        assert analytics["successful_executions"] >= 0
    
    @pytest.mark.asyncio
    async def test_concurrent_execution_with_priority(self, enhanced_executor, sample_tasks, workflow_context):
        """Test concurrent execution with different priorities."""
        
        # Split tasks into different priority levels
        high_priority_tasks = sample_tasks[:3]
        normal_priority_tasks = sample_tasks[3:6]
        low_priority_tasks = sample_tasks[6:9]
        
        # Execute tasks concurrently with different priorities
        tasks_coroutines = [
            enhanced_executor.execute_parallel(high_priority_tasks, workflow_context, PriorityLevel.HIGH),
            enhanced_executor.execute_parallel(normal_priority_tasks, workflow_context, PriorityLevel.NORMAL),
            enhanced_executor.execute_parallel(low_priority_tasks, workflow_context, PriorityLevel.LOW)
        ]
        
        results = await asyncio.gather(*tasks_coroutines)
        
        # Verify all executions completed
        assert len(results) == 3
        for result in results:
            assert "results" in result
            assert len(result["results"]) == 3
    
    @pytest.mark.asyncio
    async def test_resource_monitoring(self, enhanced_executor):
        """Test resource monitoring functionality."""
        
        # Get initial status
        status = await enhanced_executor.get_execution_status()
        
        assert "resource_usage" in status
        assert "analytics" in status
        assert "active_tasks" in status
        
        resource_usage = status["resource_usage"]
        assert "cpu_percent" in resource_usage
        assert "memory_percent" in resource_usage
        assert "optimal_concurrency" in resource_usage
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, enhanced_executor, sample_tasks, workflow_context):
        """Test circuit breaker patterns."""
        
        # Get circuit breaker status
        status = await enhanced_executor.get_execution_status()
        circuit_breakers = status.get("circuit_breakers", {})
        
        # Should have circuit breakers for agent types
        assert len(circuit_breakers) > 0
        
        # Each circuit breaker should have proper structure
        for agent_type, cb_status in circuit_breakers.items():
            assert "state" in cb_status
            assert "failure_count" in cb_status
            assert cb_status["state"] in ["closed", "open", "half_open"]
    
    @pytest.mark.asyncio
    async def test_agent_pooling(self, enhanced_executor):
        """Test agent pooling functionality."""
        
        status = await enhanced_executor.get_execution_status()
        agent_pools = status.get("agent_pools", {})
        
        # Should have some agent pools initialized
        assert len(agent_pools) > 0
        
        # Each pool should have proper structure
        for agent_type, pool_info in agent_pools.items():
            assert "active_instances" in pool_info
            assert "available_instances" in pool_info
            assert "creation_count" in pool_info
            assert pool_info["active_instances"] >= 0
            assert pool_info["available_instances"] >= 0
    
    @pytest.mark.asyncio
    async def test_adaptive_concurrency(self, enhanced_executor, sample_tasks, workflow_context):
        """Test adaptive concurrency control."""
        
        # Execute a large batch to trigger concurrency adjustments
        large_batch = sample_tasks * 5  # 50 tasks
        
        initial_status = await enhanced_executor.get_execution_status()
        initial_concurrency = initial_status["optimal_concurrency"]
        
        # Execute tasks
        result = await enhanced_executor.execute_parallel(
            tasks=large_batch,
            workflow=workflow_context,
            priority=PriorityLevel.NORMAL
        )
        
        # Check if concurrency was adjusted
        final_status = await enhanced_executor.get_execution_status()
        final_concurrency = final_status["optimal_concurrency"]
        
        # Concurrency should be tracked and potentially adjusted
        assert isinstance(initial_concurrency, int)
        assert isinstance(final_concurrency, int)
        assert final_concurrency >= 1
    
    @pytest.mark.asyncio
    async def test_execution_with_timeout(self, enhanced_executor, sample_tasks, workflow_context):
        """Test execution with timeout."""
        
        # Execute with a short timeout
        result = await enhanced_executor.execute_parallel(
            tasks=sample_tasks[:3],
            workflow=workflow_context,
            priority=PriorityLevel.NORMAL,
            timeout=10.0  # 10 second timeout
        )
        
        # Should complete successfully within timeout
        assert "results" in result
        assert len(result["results"]) == 3


class TestExecutionBenchmarker:
    """Test suite for execution benchmarker."""
    
    @pytest.mark.asyncio
    async def test_benchmarker_initialization(self):
        """Test benchmarker initialization."""
        
        benchmarker = ExecutionBenchmarker()
        await benchmarker.initialize()
        
        # Should initialize without errors
        assert benchmarker.benchmark_results == []
        assert benchmarker.monitoring_tasks == []
    
    @pytest.mark.asyncio
    async def test_benchmark_config_validation(self):
        """Test benchmark configuration validation."""
        
        # Valid configuration
        config = BenchmarkConfig(
            name="test_benchmark",
            description="Test benchmark",
            task_count=50,
            concurrent_users=5,
            test_duration_seconds=30
        )
        
        assert config.name == "test_benchmark"
        assert config.task_count == 50
        assert config.concurrent_users == 5
        assert config.test_duration_seconds == 30
        assert config.enable_circuit_breakers is True
        assert config.enable_agent_pooling is True
    
    @pytest.mark.asyncio
    async def test_benchmark_task_generation(self):
        """Test benchmark task generation."""
        
        benchmarker = ExecutionBenchmarker()
        await benchmarker.initialize()
        
        config = BenchmarkConfig(
            name="task_generation_test",
            description="Test task generation",
            task_count=20,
            agent_types=[AgentType.DRAFT_AGENT, AgentType.JUDGE_AGENT]
        )
        
        tasks = benchmarker._generate_benchmark_tasks(config)
        
        assert len(tasks) == 20
        
        # Tasks should alternate between specified agent types
        agent_types_used = {task.agent_type for task in tasks}
        assert AgentType.DRAFT_AGENT in agent_types_used
        assert AgentType.JUDGE_AGENT in agent_types_used
        
        # All tasks should have proper structure
        for task in tasks:
            assert task.task_id.startswith("benchmark_task_")
            assert task.agent_type in [AgentType.DRAFT_AGENT, AgentType.JUDGE_AGENT]
            assert "benchmark_data" in task.input_data
            assert task.metadata["benchmark"] is True
    
    @pytest.mark.asyncio
    async def test_priority_distribution(self):
        """Test priority distribution selection."""
        
        benchmarker = ExecutionBenchmarker()
        
        distribution = {
            PriorityLevel.HIGH: 0.3,
            PriorityLevel.NORMAL: 0.5,
            PriorityLevel.LOW: 0.2
        }
        
        # Test multiple selections to check distribution
        selections = []
        for _ in range(100):
            priority = benchmarker._select_priority_from_distribution(distribution)
            selections.append(priority)
        
        # Should have selected all priority levels
        priority_counts = {p: selections.count(p) for p in PriorityLevel}
        
        # Should have selected some of each priority (not exact due to randomness)
        assert priority_counts[PriorityLevel.HIGH] > 10
        assert priority_counts[PriorityLevel.NORMAL] > 30
        assert priority_counts[PriorityLevel.LOW] > 5


class TestIntegrationScenarios:
    """Integration test scenarios combining multiple components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_execution_flow(self, sample_tasks, workflow_context):
        """Test complete end-to-end execution flow."""
        
        with patch('services.enhanced_parallel_executor.get_agent_registry'), \
             patch('services.enhanced_parallel_executor.get_context_aware_selector'):
            
            # Get global executor instance
            executor = await get_enhanced_executor()
            
            # Execute tasks
            result = await executor.execute_parallel(
                tasks=sample_tasks[:5],
                workflow=workflow_context,
                priority=PriorityLevel.HIGH
            )
            
            # Verify execution completed
            assert "results" in result
            assert "analytics" in result
            
            # Check status
            status = await executor.get_execution_status()
            assert "active_tasks" in status
            assert "resource_usage" in status
    
    @pytest.mark.asyncio
    async def test_benchmarker_with_executor(self):
        """Test benchmarker integration with enhanced executor."""
        
        with patch('services.enhanced_parallel_executor.get_agent_registry'), \
             patch('services.enhanced_parallel_executor.get_context_aware_selector'):
            
            benchmarker = await get_benchmarker()
            
            # Create a simple benchmark configuration
            config = BenchmarkConfig(
                name="integration_test",
                description="Integration test benchmark",
                task_count=10,
                concurrent_users=2,
                test_duration_seconds=5,  # Short test
                max_concurrent_tasks=3
            )
            
            # Run benchmark (this tests the full integration)
            result = await benchmarker.run_single_benchmark(config)
            
            # Verify benchmark completed
            assert result.test_name == "integration_test"
            assert result.total_tasks_executed == 10
            assert result.end_time is not None
            assert result.total_duration_seconds > 0
    
    @pytest.mark.asyncio
    async def test_load_balancing_strategies(self, sample_tasks, workflow_context):
        """Test different load balancing strategies."""
        
        strategies = [
            LoadBalancingStrategy.ROUND_ROBIN,
            LoadBalancingStrategy.RESOURCE_AWARE,
            LoadBalancingStrategy.RESPONSE_TIME_BASED
        ]
        
        for strategy in strategies:
            with patch('services.enhanced_parallel_executor.get_agent_registry'), \
                 patch('services.enhanced_parallel_executor.get_context_aware_selector'):
                
                executor = EnhancedParallelExecutor(
                    max_concurrent_tasks=5,
                    load_balancing_strategy=strategy
                )
                await executor.initialize()
                
                try:
                    # Execute tasks with this strategy
                    result = await executor.execute_parallel(
                        tasks=sample_tasks[:5],
                        workflow=workflow_context,
                        priority=PriorityLevel.NORMAL
                    )
                    
                    # Should execute successfully with any strategy
                    assert "results" in result
                    assert len(result["results"]) == 5
                    
                finally:
                    await executor.shutdown()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, workflow_context):
        """Test error handling and recovery mechanisms."""
        
        # Create tasks that might cause errors
        error_prone_tasks = [
            AgentTask(
                task_id=f"error_task_{i}",
                agent_type=AgentType.DRAFT_AGENT,
                input_data={"cause_error": True if i % 3 == 0 else False},
                metadata={"test_error_handling": True}
            )
            for i in range(9)
        ]
        
        with patch('services.enhanced_parallel_executor.get_agent_registry'), \
             patch('services.enhanced_parallel_executor.get_context_aware_selector'):
            
            executor = EnhancedParallelExecutor(
                max_concurrent_tasks=3,
                enable_circuit_breakers=True
            )
            await executor.initialize()
            
            try:
                result = await executor.execute_parallel(
                    tasks=error_prone_tasks,
                    workflow=workflow_context,
                    priority=PriorityLevel.NORMAL
                )
                
                # Should handle errors gracefully
                assert "results" in result
                
                # Some tasks might fail, but executor should continue
                results = result["results"]
                assert len(results) == 9
                
                # Analytics should track successes and failures
                analytics = result["analytics"]
                assert "total_executions" in analytics
                assert "successful_executions" in analytics
                assert "failed_executions" in analytics
                
            finally:
                await executor.shutdown()


# Performance benchmark markers
@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance-focused integration tests."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_high_concurrency_performance(self):
        """Test performance under high concurrency."""
        
        with patch('services.enhanced_parallel_executor.get_agent_registry'), \
             patch('services.enhanced_parallel_executor.get_context_aware_selector'):
            
            executor = EnhancedParallelExecutor(
                max_concurrent_tasks=20,
                enable_agent_pooling=True
            )
            await executor.initialize()
            
            try:
                # Create a large number of tasks
                large_task_set = [
                    AgentTask(
                        task_id=f"perf_task_{i}",
                        agent_type=AgentType.DRAFT_AGENT,
                        input_data={"performance_test": True},
                        metadata={"batch": i // 10}
                    )
                    for i in range(100)
                ]
                
                workflow = WorkflowContext(
                    workflow_id="performance_test",
                    context_data={"performance": True}
                )
                
                start_time = asyncio.get_event_loop().time()
                
                result = await executor.execute_parallel(
                    tasks=large_task_set,
                    workflow=workflow,
                    priority=PriorityLevel.NORMAL,
                    timeout=120.0  # 2 minute timeout
                )
                
                end_time = asyncio.get_event_loop().time()
                duration = end_time - start_time
                
                # Performance assertions
                assert duration < 60.0  # Should complete within 1 minute
                assert "results" in result
                assert len(result["results"]) == 100
                
                # Calculate throughput
                throughput = 100 / duration
                assert throughput > 5  # Should handle at least 5 tasks per second
                
                print(f"Performance test completed in {duration:.2f}s (throughput: {throughput:.2f} tasks/sec)")
                
            finally:
                await executor.shutdown()
    
    @pytest.mark.asyncio
    async def test_resource_efficiency(self):
        """Test resource efficiency under load."""
        
        with patch('services.enhanced_parallel_executor.get_agent_registry'), \
             patch('services.enhanced_parallel_executor.get_context_aware_selector'):
            
            executor = EnhancedParallelExecutor(
                max_concurrent_tasks=10,
                resource_monitoring_interval=1.0
            )
            await executor.initialize()
            
            try:
                # Monitor resource usage during execution
                initial_status = await executor.get_execution_status()
                
                tasks = [
                    AgentTask(
                        task_id=f"resource_task_{i}",
                        agent_type=AgentType.DRAFT_AGENT,
                        input_data={"resource_test": True}
                    )
                    for i in range(50)
                ]
                
                workflow = WorkflowContext(
                    workflow_id="resource_test",
                    context_data={}
                )
                
                result = await executor.execute_parallel(
                    tasks=tasks,
                    workflow=workflow,
                    priority=PriorityLevel.NORMAL
                )
                
                final_status = await executor.get_execution_status()
                
                # Resource usage should be tracked
                resource_usage = result.get("resource_usage", {})
                assert "cpu_percent" in resource_usage
                assert "memory_percent" in resource_usage
                
                # Should maintain reasonable resource levels
                assert resource_usage.get("cpu_percent", 0) < 95
                assert resource_usage.get("memory_percent", 0) < 90
                
            finally:
                await executor.shutdown()