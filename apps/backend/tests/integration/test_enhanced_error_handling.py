"""
Comprehensive integration tests for enhanced error handling and recovery mechanisms.

Tests cover:
- Error detection and classification
- Circuit breaker functionality
- Retry strategies and exponential backoff
- Graceful degradation mechanisms
- Error escalation and alerting
- Recovery and health monitoring
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from services.enhanced_parallel_executor import (
    EnhancedParallelExecutor,
    ExecutionStatus,
    PriorityLevel,
    CircuitState,
    LoadBalancingStrategy
)
from services.error_handling_service import (
    ErrorHandlingService,
    ErrorContext,
    ErrorSeverity,
    ErrorCategory,
    RecoveryStrategy,
    EscalationLevel
)
from services.agent_orchestrator import AgentType, AgentTask, WorkflowContext


class MockAgentTask:
    """Mock agent task for testing."""
    
    def __init__(self, task_id: str, agent_type: AgentType):
        self.task_id = task_id
        self.agent_type = agent_type
        self.dependencies = []
        self.estimated_resource_cost = 'normal'


class MockWorkflowContext:
    """Mock workflow context for testing."""
    
    def __init__(self, workflow_id: str = "test_workflow"):
        self.id = workflow_id
        self.project_id = "test_project"
        self.user_id = "test_user"


@pytest.fixture
async def executor():
    """Create enhanced parallel executor for testing."""
    executor = EnhancedParallelExecutor(
        max_concurrent_tasks=5,
        enable_circuit_breakers=True,
        enable_agent_pooling=True,
        resource_monitoring_interval=0.1
    )
    
    # Mock dependencies
    with patch('services.enhanced_parallel_executor.get_agent_registry'), \
         patch('services.enhanced_parallel_executor.get_context_aware_selector'):
        await executor.initialize()
    
    yield executor
    
    await executor.shutdown()


@pytest.fixture
async def error_handler():
    """Create error handling service for testing."""
    handler = ErrorHandlingService()
    await handler.initialize()
    
    yield handler
    
    await handler.shutdown()


class TestErrorDetectionAndClassification:
    """Test error detection and classification functionality."""
    
    @pytest.mark.asyncio
    async def test_error_severity_classification(self, error_handler):
        """Test that errors are properly classified by severity."""
        
        # Test critical error
        critical_error = RuntimeError("Database connection lost")
        context = ErrorContext(
            source="test",
            operation="database_query",
            agent_type="data_agent",
            task_id="test_task",
            metadata={"database": "neo4j"}
        )
        
        result = await error_handler.handle_error(critical_error, context)
        
        assert result.severity == ErrorSeverity.HIGH
        assert result.category in [ErrorCategory.SYSTEM_ERROR, ErrorCategory.DATABASE_ERROR]
        assert result.should_retry is True
    
    @pytest.mark.asyncio
    async def test_error_category_detection(self, error_handler):
        """Test that errors are categorized correctly."""
        
        test_cases = [
            (ConnectionError("Failed to connect"), ErrorCategory.NETWORK_ERROR),
            (TimeoutError("Request timed out"), ErrorCategory.TIMEOUT_ERROR),
            (ValueError("Invalid input"), ErrorCategory.VALIDATION_ERROR),
            (MemoryError("Out of memory"), ErrorCategory.RESOURCE_ERROR),
        ]
        
        for error, expected_category in test_cases:
            context = ErrorContext(
                source="test",
                operation="test_op",
                agent_type="test_agent",
                task_id="test_task",
                metadata={}
            )
            
            result = await error_handler.handle_error(error, context)
            assert result.category == expected_category


class TestCircuitBreakerFunctionality:
    """Test circuit breaker patterns and state management."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opening(self, executor):
        """Test that circuit breaker opens after failure threshold."""
        
        # Simulate multiple failures for a specific agent type
        agent_type = AgentType.DRAFT_AGENT
        circuit_breaker = executor.circuit_breakers[agent_type]
        
        # Should be closed initially
        assert circuit_breaker.state == CircuitState.CLOSED
        
        # Simulate failures up to threshold
        for _ in range(circuit_breaker.failure_threshold):
            await executor._record_circuit_breaker_failure(agent_type)
        
        # Circuit breaker should now be open
        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.failure_count == circuit_breaker.failure_threshold
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self, executor):
        """Test circuit breaker recovery through half-open state."""
        
        agent_type = AgentType.BUSINESS_ANALYST
        circuit_breaker = executor.circuit_breakers[agent_type]
        
        # Force circuit breaker to open state
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.last_failure_time = datetime.utcnow() - timedelta(seconds=35)
        
        # Check should move to half-open
        can_execute = await executor._check_circuit_breaker(circuit_breaker)
        assert can_execute is True
        assert circuit_breaker.state == CircuitState.HALF_OPEN
        
        # Simulate successful calls to close circuit
        for _ in range(circuit_breaker.half_open_max_calls):
            await executor._record_circuit_breaker_success(agent_type)
        
        # Circuit should be closed now
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_execution(self, executor):
        """Test that open circuit breaker prevents task execution."""
        
        agent_type = AgentType.JUDGE_AGENT
        circuit_breaker = executor.circuit_breakers[agent_type]
        
        # Force circuit breaker open
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.last_failure_time = datetime.utcnow()
        
        # Create mock task
        task = MockAgentTask("test_task", agent_type)
        workflow = MockWorkflowContext()
        
        # Mock the actual execution to avoid dependencies
        with patch.object(executor, '_execute_agent_task', new_callable=AsyncMock):
            result = await executor._execute_single_task_with_resilience(task, workflow)
        
        # Should fail due to circuit breaker
        assert result["status"] == ExecutionStatus.FAILED.value
        assert "circuit breaker is open" in result["error"].lower()


class TestRetryStrategiesAndBackoff:
    """Test retry mechanisms and exponential backoff."""
    
    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self, executor):
        """Test that retry delays follow exponential backoff pattern."""
        
        retry_delays = []
        
        async def mock_execute_agent_task(*args, **kwargs):
            """Mock that always fails to trigger retries."""
            raise ConnectionError("Simulated failure")
        
        async def mock_sleep(delay):
            """Capture sleep delays."""
            retry_delays.append(delay)
        
        task = MockAgentTask("test_task", AgentType.DRAFT_AGENT)
        workflow = MockWorkflowContext()
        metrics = MagicMock()
        metrics.execution_attempts = 0
        
        with patch.object(executor, '_execute_agent_task', side_effect=mock_execute_agent_task), \
             patch('asyncio.sleep', side_effect=mock_sleep), \
             pytest.raises(ConnectionError):
            
            await executor._execute_with_retry(None, task, workflow, metrics, max_retries=3)
        
        # Verify exponential backoff pattern (approximately)
        assert len(retry_delays) == 3
        assert retry_delays[1] > retry_delays[0]  # Second delay > first delay
        assert retry_delays[2] > retry_delays[1]  # Third delay > second delay
    
    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self, executor):
        """Test successful execution after initial failures."""
        
        call_count = 0
        
        async def mock_execute_agent_task(*args, **kwargs):
            """Fail first two calls, succeed on third."""
            nonlocal call_count
            call_count += 1
            
            if call_count < 3:
                raise ConnectionError("Simulated failure")
            
            return {"result": "success", "attempt": call_count}
        
        task = MockAgentTask("test_task", AgentType.BUSINESS_ANALYST)
        workflow = MockWorkflowContext()
        metrics = MagicMock()
        metrics.execution_attempts = 0
        
        with patch.object(executor, '_execute_agent_task', side_effect=mock_execute_agent_task), \
             patch('asyncio.sleep', new_callable=AsyncMock):
            
            result = await executor._execute_with_retry(None, task, workflow, metrics, max_retries=5)
        
        assert result["result"] == "success"
        assert result["attempt"] == 3
        assert call_count == 3


class TestGracefulDegradation:
    """Test graceful degradation mechanisms."""
    
    @pytest.mark.asyncio
    async def test_degradation_triggers_on_high_error_rate(self, executor):
        """Test that degradation activates when error rate is high."""
        
        # Simulate high error rate
        executor.execution_analytics["total_executions"] = 100
        executor.execution_analytics["failed_executions"] = 25  # 25% failure rate
        
        initial_concurrency = executor.optimal_concurrency
        
        degradation_enabled = await executor.enable_graceful_degradation()
        
        assert degradation_enabled is True
        assert executor.optimal_concurrency < initial_concurrency
        assert executor.optimal_concurrency == max(1, initial_concurrency // 2)
    
    @pytest.mark.asyncio
    async def test_degradation_triggers_on_resource_pressure(self, executor):
        """Test that degradation activates under resource pressure."""
        
        # Mock high resource usage
        mock_resource_usage = MagicMock()
        mock_resource_usage.cpu_percent = 90.0
        mock_resource_usage.memory_percent = 88.0
        
        executor.resource_usage_history.append(mock_resource_usage)
        
        initial_concurrency = executor.optimal_concurrency
        
        degradation_enabled = await executor.enable_graceful_degradation()
        
        assert degradation_enabled is True
        assert executor.optimal_concurrency < initial_concurrency
    
    @pytest.mark.asyncio
    async def test_system_health_assessment(self, executor):
        """Test system health assessment metrics."""
        
        # Set up test data
        executor.execution_analytics = {
            "total_executions": 200,
            "successful_executions": 180,
            "failed_executions": 20
        }
        
        # Mock circuit breaker states
        executor.circuit_breakers[AgentType.DRAFT_AGENT].state = CircuitState.OPEN
        executor.circuit_breakers[AgentType.JUDGE_AGENT].state = CircuitState.OPEN
        
        health = await executor._assess_system_health()
        
        assert health["error_rate"] == 10.0  # 20/200 * 100
        assert health["circuit_breakers_open"] == 2
        assert health["active_tasks"] == len(executor.active_tasks)


class TestErrorEscalationAndAlerting:
    """Test error escalation and alerting mechanisms."""
    
    @pytest.mark.asyncio
    async def test_error_escalation_levels(self, error_handler):
        """Test that errors escalate to appropriate levels."""
        
        # Test critical error escalation
        critical_error = RuntimeError("System failure")
        context = ErrorContext(
            source="system",
            operation="critical_operation",
            agent_type="system_agent",
            task_id="critical_task",
            metadata={"severity": "critical"}
        )
        
        result = await error_handler.handle_error(critical_error, context)
        
        # Critical errors should escalate to high levels
        assert result.escalation_level in [EscalationLevel.URGENT, EscalationLevel.EMERGENCY]
        assert result.recovery_actions is not None
        assert len(result.recovery_actions) > 0
    
    @pytest.mark.asyncio
    async def test_alert_generation(self, error_handler):
        """Test that alerts are generated for appropriate errors."""
        
        # Test multiple errors to trigger alerting
        for i in range(3):
            error = ConnectionError(f"Connection failure {i}")
            context = ErrorContext(
                source="network",
                operation="api_call",
                agent_type="api_agent",
                task_id=f"task_{i}",
                metadata={"attempt": i}
            )
            
            await error_handler.handle_error(error, context)
        
        # Check that patterns are detected and alerts generated
        analytics = await error_handler.get_error_analytics()
        
        assert analytics["total_errors"] >= 3
        assert analytics["error_patterns"] is not None
        assert len(analytics["recent_errors"]) >= 3


class TestRecoveryAndHealthMonitoring:
    """Test recovery mechanisms and health monitoring."""
    
    @pytest.mark.asyncio
    async def test_agent_health_monitoring(self, error_handler):
        """Test agent health status monitoring."""
        
        agent_type = "test_agent"
        
        # Initially should be healthy
        is_healthy = await error_handler.check_agent_health(agent_type)
        assert is_healthy is True
        
        # Record multiple failures
        for _ in range(5):
            error = RuntimeError("Agent failure")
            context = ErrorContext(
                source="test",
                operation="test_op",
                agent_type=agent_type,
                task_id="test_task",
                metadata={}
            )
            
            await error_handler.handle_error(error, context)
        
        # Should now be unhealthy
        is_healthy = await error_handler.check_agent_health(agent_type)
        assert is_healthy is False
    
    @pytest.mark.asyncio
    async def test_recovery_recording(self, error_handler):
        """Test recording of successful recoveries."""
        
        agent_type = "recovery_agent"
        recovery_type = "circuit_breaker_reset"
        metadata = {"manual": True}
        
        await error_handler.record_recovery(agent_type, recovery_type, metadata)
        
        status = await error_handler.get_recovery_status()
        
        assert agent_type in status
        assert status[agent_type]["last_recovery_type"] == recovery_type
        assert status[agent_type]["metadata"] == metadata
    
    @pytest.mark.asyncio
    async def test_forced_recovery_attempt(self, executor):
        """Test manually forcing recovery attempts."""
        
        agent_type = "force_recovery_agent"
        
        # Set up a circuit breaker in open state
        for agent_enum_type in AgentType:
            if agent_enum_type.value == agent_type:
                executor.circuit_breakers[agent_enum_type].state = CircuitState.OPEN
                break
        
        # Force recovery
        recovery_successful = await executor.force_recovery_attempt(agent_type)
        
        assert recovery_successful is True
        
        # Verify circuit breaker is in half-open state
        for agent_enum_type in AgentType:
            if agent_enum_type.value == agent_type:
                assert executor.circuit_breakers[agent_enum_type].state == CircuitState.HALF_OPEN
                break


class TestIntegrationScenarios:
    """Test complete integration scenarios combining multiple error handling features."""
    
    @pytest.mark.asyncio
    async def test_complete_failure_recovery_cycle(self, executor):
        """Test complete cycle from failure to recovery."""
        
        agent_type = AgentType.PROJECT_ARCHITECT
        task = MockAgentTask("integration_test", agent_type)
        workflow = MockWorkflowContext("integration_workflow")
        
        failure_count = 0
        
        async def mock_failing_then_succeeding_task(*args, **kwargs):
            """Fail first few attempts, then succeed."""
            nonlocal failure_count
            failure_count += 1
            
            if failure_count <= 3:
                raise ConnectionError(f"Failure {failure_count}")
            
            return {"result": "success", "recovery": True}
        
        with patch.object(executor, '_execute_agent_task', side_effect=mock_failing_then_succeeding_task), \
             patch('asyncio.sleep', new_callable=AsyncMock):
            
            result = await executor._execute_single_task_with_resilience(task, workflow)
        
        # Should eventually succeed
        assert result["status"] == ExecutionStatus.COMPLETED.value
        assert result["result"]["result"] == "success"
        assert failure_count == 4  # 3 failures + 1 success
    
    @pytest.mark.asyncio
    async def test_parallel_execution_with_mixed_outcomes(self, executor):
        """Test parallel execution with some successes and failures."""
        
        # Create mixed tasks - some will succeed, some will fail
        tasks = [
            MockAgentTask(f"task_{i}", AgentType.DRAFT_AGENT if i % 2 == 0 else AgentType.JUDGE_AGENT)
            for i in range(6)
        ]
        
        workflow = MockWorkflowContext("mixed_workflow")
        
        success_count = 0
        
        async def mock_mixed_execution(agent_instance, task, workflow_ctx):
            """Some tasks succeed, others fail."""
            nonlocal success_count
            
            if task.task_id.endswith(('0', '2', '4')):  # Even tasks succeed
                success_count += 1
                return {"result": f"success_{task.task_id}", "agent": task.agent_type.value}
            else:  # Odd tasks fail
                raise RuntimeError(f"Simulated failure for {task.task_id}")
        
        with patch.object(executor, '_execute_agent_task', side_effect=mock_mixed_execution), \
             patch('asyncio.sleep', new_callable=AsyncMock):
            
            results = await executor.execute_parallel(tasks, workflow, timeout=10.0)
        
        # Verify mixed outcomes
        assert "results" in results
        assert len(results["results"]) == 6
        
        completed_tasks = [
            r for r in results["results"].values()
            if r.get("status") == ExecutionStatus.COMPLETED.value
        ]
        failed_tasks = [
            r for r in results["results"].values()
            if r.get("status") == ExecutionStatus.FAILED.value
        ]
        
        assert len(completed_tasks) == 3  # Even tasks
        assert len(failed_tasks) == 3     # Odd tasks
    
    @pytest.mark.asyncio
    async def test_error_handling_status_reporting(self, executor):
        """Test comprehensive error handling status reporting."""
        
        # Trigger some errors and state changes
        await executor._record_circuit_breaker_failure(AgentType.DRAFT_AGENT)
        await executor._record_circuit_breaker_failure(AgentType.JUDGE_AGENT)
        
        # Get comprehensive status
        status = await executor.get_error_handling_status()
        
        # Verify status structure
        assert "error_handler_active" in status
        assert "circuit_breakers" in status
        assert "system_health" in status
        assert "degradation_active" in status
        
        # Verify circuit breaker information
        assert AgentType.DRAFT_AGENT.value in status["circuit_breakers"]
        assert AgentType.JUDGE_AGENT.value in status["circuit_breakers"]
        
        # Verify system health metrics
        health = status["system_health"]
        assert "error_rate" in health
        assert "resource_usage" in health
        assert "active_tasks" in health
        assert "circuit_breakers_open" in health


# Performance and stress tests

class TestErrorHandlingPerformance:
    """Test error handling performance under load."""
    
    @pytest.mark.asyncio
    async def test_high_volume_error_processing(self, error_handler):
        """Test error handling performance with high error volumes."""
        
        import time
        
        start_time = time.time()
        
        # Process many errors quickly
        tasks = []
        for i in range(100):
            error = RuntimeError(f"Error {i}")
            context = ErrorContext(
                source="performance_test",
                operation="bulk_operation",
                agent_type=f"agent_{i % 10}",
                task_id=f"task_{i}",
                metadata={"batch": i // 10}
            )
            
            tasks.append(error_handler.handle_error(error, context))
        
        # Process all errors concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify performance
        assert len(results) == 100
        assert processing_time < 10.0  # Should complete within 10 seconds
        
        # Verify no exceptions in results
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_circuit_breaker_operations(self, executor):
        """Test circuit breaker thread safety under concurrent operations."""
        
        agent_type = AgentType.BUSINESS_ANALYST
        
        # Concurrent failure recordings
        failure_tasks = [
            executor._record_circuit_breaker_failure(agent_type)
            for _ in range(10)
        ]
        
        # Concurrent success recordings
        success_tasks = [
            executor._record_circuit_breaker_success(agent_type)
            for _ in range(5)
        ]
        
        # Execute concurrently
        await asyncio.gather(*failure_tasks, *success_tasks)
        
        # Circuit breaker should be in a consistent state
        circuit_breaker = executor.circuit_breakers[agent_type]
        assert circuit_breaker.state in [CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN]
        assert circuit_breaker.failure_count >= 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])