"""
Unit tests for the ErrorHandlingService class.

Tests cover:
- Error classification and severity detection
- Recovery strategy selection
- Circuit breaker management
- Error analytics and reporting
- Health monitoring
- Escalation logic
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from services.error_handling_service import (
    ErrorHandlingService,
    ErrorContext,
    ErrorSeverity,
    ErrorCategory,
    RecoveryStrategy,
    EscalationLevel,
    RecoveryResult
)


@pytest.fixture
async def error_handler():
    """Create error handling service for testing."""
    handler = ErrorHandlingService()
    await handler.initialize()
    
    yield handler
    
    await handler.shutdown()


class TestErrorClassification:
    """Test error classification functionality."""
    
    @pytest.mark.asyncio
    async def test_classify_network_errors(self, error_handler):
        """Test classification of network-related errors."""
        
        network_errors = [
            ConnectionError("Connection refused"),
            ConnectionResetError("Connection reset by peer"),
            TimeoutError("Connection timed out"),
        ]
        
        for error in network_errors:
            category = await error_handler._classify_error_category(error)
            assert category in [ErrorCategory.NETWORK_ERROR, ErrorCategory.TIMEOUT_ERROR]
    
    @pytest.mark.asyncio
    async def test_classify_system_errors(self, error_handler):
        """Test classification of system-related errors."""
        
        system_errors = [
            MemoryError("Out of memory"),
            OSError("System resource unavailable"),
            RuntimeError("Critical system failure"),
        ]
        
        for error in system_errors:
            category = await error_handler._classify_error_category(error)
            assert category in [ErrorCategory.SYSTEM_ERROR, ErrorCategory.RESOURCE_ERROR]
    
    @pytest.mark.asyncio
    async def test_severity_assessment(self, error_handler):
        """Test error severity assessment."""
        
        # Test critical errors
        critical_errors = [
            ("Database connection lost", "database"),
            ("System out of memory", "system"),
            ("Authentication failed", "auth"),
        ]
        
        for error_msg, operation in critical_errors:
            context = ErrorContext(
                source="test",
                operation=operation,
                agent_type="test_agent",
                task_id="test_task",
                metadata={}
            )
            
            severity = await error_handler._assess_error_severity(
                RuntimeError(error_msg), context
            )
            
            assert severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
    
    @pytest.mark.asyncio
    async def test_context_aware_classification(self, error_handler):
        """Test that error classification considers context."""
        
        # Same error but different contexts
        error = ConnectionError("Connection failed")
        
        # Database context should be more severe
        db_context = ErrorContext(
            source="database_service",
            operation="query_execution",
            agent_type="data_agent",
            task_id="critical_query",
            metadata={"table": "users"}
        )
        
        # Cache context should be less severe
        cache_context = ErrorContext(
            source="cache_service",
            operation="cache_get",
            agent_type="cache_agent",
            task_id="cache_lookup",
            metadata={"key": "temp_data"}
        )
        
        db_result = await error_handler.handle_error(error, db_context)
        cache_result = await error_handler.handle_error(error, cache_context)
        
        # Database error should have higher severity or different strategy
        assert db_result.severity.value >= cache_result.severity.value


class TestRecoveryStrategies:
    """Test recovery strategy selection and execution."""
    
    @pytest.mark.asyncio
    async def test_retry_strategy_selection(self, error_handler):
        """Test that appropriate retry strategies are selected."""
        
        # Network error should get retry strategy
        network_error = ConnectionError("Network unavailable")
        context = ErrorContext(
            source="api_client",
            operation="api_call",
            agent_type="api_agent",
            task_id="api_task",
            metadata={}
        )
        
        result = await error_handler.handle_error(network_error, context)
        
        assert result.strategy in [RecoveryStrategy.RETRY, RecoveryStrategy.RETRY_WITH_FALLBACK]
        assert result.should_retry is True
        assert result.retry_delay is not None
        assert result.retry_delay > 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_strategy(self, error_handler):
        """Test circuit breaker strategy selection."""
        
        # Multiple failures should trigger circuit breaker
        agent_type = "failing_agent"
        
        for i in range(5):  # Trigger multiple failures
            error = RuntimeError(f"Failure {i}")
            context = ErrorContext(
                source="test",
                operation="test_op",
                agent_type=agent_type,
                task_id=f"task_{i}",
                metadata={}
            )
            
            result = await error_handler.handle_error(error, context)
        
        # After multiple failures, should recommend circuit breaker
        assert result.strategy in [RecoveryStrategy.CIRCUIT_BREAKER, RecoveryStrategy.GRACEFUL_DEGRADATION]
    
    @pytest.mark.asyncio
    async def test_fallback_strategy_for_critical_errors(self, error_handler):
        """Test fallback strategies for critical errors."""
        
        critical_error = MemoryError("System out of memory")
        context = ErrorContext(
            source="system",
            operation="memory_allocation",
            agent_type="system_agent",
            task_id="critical_task",
            metadata={"memory_requested": "large"}
        )
        
        result = await error_handler.handle_error(critical_error, context)
        
        # Critical errors should get fallback or degradation strategies
        assert result.strategy in [
            RecoveryStrategy.FAILOVER,
            RecoveryStrategy.GRACEFUL_DEGRADATION,
            RecoveryStrategy.EMERGENCY_STOP
        ]
    
    @pytest.mark.asyncio
    async def test_custom_recovery_actions(self, error_handler):
        """Test that recovery results include specific actions."""
        
        error = ValueError("Invalid configuration")
        context = ErrorContext(
            source="config_service",
            operation="load_config",
            agent_type="config_agent",
            task_id="config_task",
            metadata={"config_file": "app.yaml"}
        )
        
        result = await error_handler.handle_error(error, context)
        
        assert result.recovery_actions is not None
        assert len(result.recovery_actions) > 0
        assert any("config" in action.lower() for action in result.recovery_actions)


class TestCircuitBreakerManagement:
    """Test circuit breaker management functionality."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_state_tracking(self, error_handler):
        """Test circuit breaker state tracking."""
        
        agent_type = "test_agent"
        
        # Initially no state
        state = await error_handler._get_circuit_breaker_state(agent_type)
        assert state["state"] == "closed"
        assert state["failure_count"] == 0
        
        # Update to open state
        await error_handler.update_circuit_breaker_state(
            agent_type, "open", {"failure_count": 5}
        )
        
        state = await error_handler._get_circuit_breaker_state(agent_type)
        assert state["state"] == "open"
        assert state["failure_count"] == 5
    
    @pytest.mark.asyncio
    async def test_agent_health_monitoring(self, error_handler):
        """Test agent health status tracking."""
        
        agent_type = "health_test_agent"
        
        # Should be healthy initially
        is_healthy = await error_handler.check_agent_health(agent_type)
        assert is_healthy is True
        
        # Record multiple failures
        for i in range(3):
            error = RuntimeError(f"Health failure {i}")
            context = ErrorContext(
                source="health_test",
                operation="health_check",
                agent_type=agent_type,
                task_id=f"health_task_{i}",
                metadata={}
            )
            
            await error_handler.handle_error(error, context)
        
        # Should now be unhealthy
        is_healthy = await error_handler.check_agent_health(agent_type)
        assert is_healthy is False
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_threshold_adaptation(self, error_handler):
        """Test adaptive circuit breaker thresholds."""
        
        agent_type = "adaptive_agent"
        
        # Get initial threshold
        initial_state = await error_handler._get_circuit_breaker_state(agent_type)
        initial_threshold = initial_state.get("failure_threshold", 5)
        
        # Simulate consistent failures over time
        for i in range(10):
            error = RuntimeError(f"Consistent failure {i}")
            context = ErrorContext(
                source="test",
                operation="consistent_op",
                agent_type=agent_type,
                task_id=f"task_{i}",
                metadata={}
            )
            
            await error_handler.handle_error(error, context)
        
        # Check if threshold has been adapted
        current_state = await error_handler._get_circuit_breaker_state(agent_type)
        current_threshold = current_state.get("failure_threshold", 5)
        
        # Threshold might be adjusted based on failure patterns
        assert current_threshold >= initial_threshold


class TestErrorAnalytics:
    """Test error analytics and reporting functionality."""
    
    @pytest.mark.asyncio
    async def test_error_pattern_detection(self, error_handler):
        """Test detection of error patterns."""
        
        # Create pattern of similar errors
        base_error = ConnectionError("Connection timeout")
        agent_type = "pattern_agent"
        
        for i in range(5):
            context = ErrorContext(
                source="pattern_test",
                operation="network_call",
                agent_type=agent_type,
                task_id=f"pattern_task_{i}",
                metadata={"endpoint": "api.example.com"}
            )
            
            await error_handler.handle_error(base_error, context)
        
        # Get analytics
        analytics = await error_handler.get_error_analytics()
        
        assert analytics["total_errors"] >= 5
        assert "error_patterns" in analytics
        assert agent_type in analytics.get("agents_with_errors", [])
    
    @pytest.mark.asyncio
    async def test_error_frequency_tracking(self, error_handler):
        """Test error frequency tracking over time."""
        
        agent_type = "frequency_agent"
        
        # Generate errors at different times
        for i in range(3):
            error = RuntimeError(f"Frequency error {i}")
            context = ErrorContext(
                source="frequency_test",
                operation="timed_operation",
                agent_type=agent_type,
                task_id=f"freq_task_{i}",
                metadata={"timestamp": datetime.utcnow().isoformat()}
            )
            
            await error_handler.handle_error(error, context)
            await asyncio.sleep(0.1)  # Small delay between errors
        
        analytics = await error_handler.get_error_analytics()
        
        assert "recent_errors" in analytics
        assert len(analytics["recent_errors"]) >= 3
        
        # Errors should have timestamps
        for error_record in analytics["recent_errors"]:
            assert "timestamp" in error_record
            assert "agent_type" in error_record
    
    @pytest.mark.asyncio
    async def test_error_analytics_aggregation(self, error_handler):
        """Test error analytics aggregation by category and severity."""
        
        # Create diverse errors
        errors_data = [
            (ConnectionError("Network issue"), "network_agent"),
            (ValueError("Validation failed"), "validation_agent"),
            (MemoryError("Out of memory"), "system_agent"),
            (TimeoutError("Operation timeout"), "timeout_agent"),
        ]
        
        for error, agent_type in errors_data:
            context = ErrorContext(
                source="aggregation_test",
                operation="diverse_operation",
                agent_type=agent_type,
                task_id=f"agg_task_{agent_type}",
                metadata={}
            )
            
            await error_handler.handle_error(error, context)
        
        analytics = await error_handler.get_error_analytics()
        
        assert "error_categories" in analytics
        assert "error_severities" in analytics
        assert len(analytics["error_categories"]) > 0
        assert len(analytics["error_severities"]) > 0


class TestEscalationLogic:
    """Test error escalation logic and levels."""
    
    @pytest.mark.asyncio
    async def test_escalation_level_determination(self, error_handler):
        """Test determination of escalation levels."""
        
        # Test different error scenarios
        escalation_scenarios = [
            (RuntimeError("Critical system failure"), "critical_op", EscalationLevel.URGENT),
            (ValueError("Invalid input"), "validation", EscalationLevel.LOW),
            (MemoryError("Out of memory"), "memory_op", EscalationLevel.EMERGENCY),
            (ConnectionError("Network issue"), "network_op", EscalationLevel.NORMAL),
        ]
        
        for error, operation, expected_min_level in escalation_scenarios:
            context = ErrorContext(
                source="escalation_test",
                operation=operation,
                agent_type="escalation_agent",
                task_id=f"escalation_{operation}",
                metadata={}
            )
            
            result = await error_handler.handle_error(error, context)
            
            # Escalation level should be at least the expected minimum
            assert result.escalation_level.value >= expected_min_level.value
    
    @pytest.mark.asyncio
    async def test_repeated_error_escalation(self, error_handler):
        """Test that repeated errors increase escalation level."""
        
        agent_type = "escalation_repeat_agent"
        base_error = RuntimeError("Repeated failure")
        
        escalation_levels = []
        
        for i in range(4):
            context = ErrorContext(
                source="repeat_test",
                operation="repeat_operation",
                agent_type=agent_type,
                task_id=f"repeat_task_{i}",
                metadata={"attempt": i + 1}
            )
            
            result = await error_handler.handle_error(base_error, context)
            escalation_levels.append(result.escalation_level)
        
        # Later errors should have higher or equal escalation levels
        for i in range(1, len(escalation_levels)):
            assert escalation_levels[i].value >= escalation_levels[0].value


class TestRecoveryTracking:
    """Test recovery tracking and status monitoring."""
    
    @pytest.mark.asyncio
    async def test_recovery_recording(self, error_handler):
        """Test recording of recovery events."""
        
        agent_type = "recovery_agent"
        recovery_type = "manual_restart"
        metadata = {"operator": "admin", "reason": "maintenance"}
        
        await error_handler.record_recovery(agent_type, recovery_type, metadata)
        
        status = await error_handler.get_recovery_status()
        
        assert agent_type in status
        assert status[agent_type]["last_recovery_type"] == recovery_type
        assert status[agent_type]["metadata"] == metadata
        assert "timestamp" in status[agent_type]
    
    @pytest.mark.asyncio
    async def test_recovery_status_tracking(self, error_handler):
        """Test comprehensive recovery status tracking."""
        
        # Record multiple recovery events
        recovery_events = [
            ("agent_1", "circuit_breaker_reset", {"auto": True}),
            ("agent_2", "service_restart", {"manual": True}),
            ("agent_3", "configuration_reload", {"config": "updated"}),
        ]
        
        for agent_type, recovery_type, metadata in recovery_events:
            await error_handler.record_recovery(agent_type, recovery_type, metadata)
        
        status = await error_handler.get_recovery_status()
        
        assert len(status) >= 3
        for agent_type, recovery_type, metadata in recovery_events:
            assert agent_type in status
            assert status[agent_type]["last_recovery_type"] == recovery_type
            assert status[agent_type]["metadata"] == metadata
    
    @pytest.mark.asyncio
    async def test_recovery_success_rate_tracking(self, error_handler):
        """Test tracking of recovery success rates."""
        
        agent_type = "success_rate_agent"
        
        # Record some failures
        for i in range(3):
            error = RuntimeError(f"Failure before recovery {i}")
            context = ErrorContext(
                source="success_test",
                operation="failing_op",
                agent_type=agent_type,
                task_id=f"fail_task_{i}",
                metadata={}
            )
            
            await error_handler.handle_error(error, context)
        
        # Record recovery
        await error_handler.record_recovery(
            agent_type, "successful_recovery", {"success": True}
        )
        
        status = await error_handler.get_recovery_status()
        
        assert agent_type in status
        assert status[agent_type]["last_recovery_type"] == "successful_recovery"


class TestPerformanceAndConcurrency:
    """Test performance and concurrency aspects of error handling."""
    
    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self, error_handler):
        """Test handling multiple concurrent errors."""
        
        async def generate_error(error_id: int):
            """Generate a single error."""
            error = RuntimeError(f"Concurrent error {error_id}")
            context = ErrorContext(
                source="concurrent_test",
                operation="concurrent_operation",
                agent_type=f"concurrent_agent_{error_id % 3}",
                task_id=f"concurrent_task_{error_id}",
                metadata={"id": error_id}
            )
            
            return await error_handler.handle_error(error, context)
        
        # Generate many concurrent errors
        tasks = [generate_error(i) for i in range(50)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete successfully
        assert len(results) == 50
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0
        
        # All results should be RecoveryResult instances
        recovery_results = [r for r in results if isinstance(r, RecoveryResult)]
        assert len(recovery_results) == 50
    
    @pytest.mark.asyncio
    async def test_error_handling_memory_usage(self, error_handler):
        """Test that error handling doesn't cause memory leaks."""
        
        import gc
        
        initial_objects = len(gc.get_objects())
        
        # Generate many errors to test memory usage
        for i in range(100):
            error = RuntimeError(f"Memory test error {i}")
            context = ErrorContext(
                source="memory_test",
                operation="memory_operation",
                agent_type="memory_agent",
                task_id=f"memory_task_{i}",
                metadata={"iteration": i}
            )
            
            await error_handler.handle_error(error, context)
        
        # Force garbage collection
        gc.collect()
        
        final_objects = len(gc.get_objects())
        
        # Object count shouldn't increase dramatically
        # (allowing for some growth due to test infrastructure)
        object_increase = final_objects - initial_objects
        assert object_increase < 1000  # Reasonable threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])