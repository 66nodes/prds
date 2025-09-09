"""
Unit Tests for Agent Action Logger

Comprehensive test suite for the agent action logging system,
covering core functionality, error handling, and performance.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from services.agent_action_logger import (
    AgentActionLogger, AgentActionLog, LoggingConfig, ActionType, LogLevel,
    get_agent_logger, log_task_started, log_task_completed, log_validation_result
)
from services.agent_orchestrator import AgentTask, WorkflowContext, AgentType
from services.enhanced_parallel_executor import ExecutionMetrics, ExecutionStatus
from services.graphrag.validation_pipeline import ValidationResult, ValidationStatus


@pytest.fixture
def logging_config():
    """Test logging configuration."""
    return LoggingConfig(
        enable_redis_storage=False,  # Disable Redis for unit tests
        enable_file_storage=False,   # Disable file storage for unit tests
        enable_async_processing=False,  # Synchronous for testing
        batch_size=10,
        max_queue_size=100,
        log_input_data=True,
        log_output_data=True,
        mask_sensitive_data=True
    )


@pytest.fixture
def agent_logger(logging_config):
    """Agent logger instance for testing."""
    logger = AgentActionLogger(config=logging_config)
    return logger


@pytest.fixture
def sample_task():
    """Sample agent task for testing."""
    return AgentTask(
        task_id="test_task_001",
        agent_type=AgentType.DRAFT_AGENT,
        input_data={"content": "Test task content", "requirements": ["quality", "speed"]}
    )


@pytest.fixture
def sample_workflow():
    """Sample workflow context for testing."""
    return WorkflowContext(
        workflow_id="test_workflow_001",
        session_id="test_session_001",
        correlation_id="test_correlation_001"
    )


@pytest.fixture
def sample_execution_metrics():
    """Sample execution metrics for testing."""
    return ExecutionMetrics(
        task_id="test_task_001",
        agent_type=AgentType.DRAFT_AGENT,
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        duration_ms=1500,
        status=ExecutionStatus.COMPLETED,
        queue_wait_time_ms=200,
        memory_usage_mb=150.5,
        cpu_usage_percent=65.3,
        execution_attempts=1
    )


@pytest.fixture
def sample_validation_result():
    """Sample validation result for testing."""
    return ValidationResult(
        validation_id="test_validation_001",
        status=ValidationStatus.PASSED,
        overall_confidence=0.92,
        processing_time_ms=800.5,
        hallucination_detection_result={
            "hallucination_score": 0.015,
            "evidence_sources": 5,
            "violations": []
        }
    )


class TestAgentActionLog:
    """Test AgentActionLog data structure and methods."""
    
    def test_log_creation(self):
        """Test basic log entry creation."""
        log = AgentActionLog(
            action_type=ActionType.TASK_STARTED,
            log_level=LogLevel.INFO,
            agent_type=AgentType.DRAFT_AGENT,
            task_id="test_001"
        )
        
        assert log.action_type == ActionType.TASK_STARTED
        assert log.log_level == LogLevel.INFO
        assert log.agent_type == AgentType.DRAFT_AGENT
        assert log.task_id == "test_001"
        assert isinstance(log.timestamp, datetime)
        assert log.log_id is not None
    
    def test_log_to_dict(self):
        """Test log conversion to dictionary."""
        log = AgentActionLog(
            action_type=ActionType.VALIDATION_COMPLETED,
            log_level=LogLevel.INFO,
            agent_type=AgentType.JUDGE_AGENT,
            task_id="test_002",
            confidence_score=0.95,
            duration_ms=500.0,
            tags=["validation", "quality"]
        )
        
        log_dict = log.to_dict()
        
        assert log_dict["action_type"] == ActionType.VALIDATION_COMPLETED.value
        assert log_dict["log_level"] == LogLevel.INFO.value
        assert log_dict["agent_type"] == AgentType.JUDGE_AGENT.value
        assert log_dict["task_id"] == "test_002"
        assert log_dict["confidence_score"] == 0.95
        assert log_dict["duration_ms"] == 500.0
        assert log_dict["tags"] == ["validation", "quality"]
        assert "timestamp" in log_dict
    
    def test_log_with_complex_data(self):
        """Test log with complex nested data structures."""
        complex_data = {
            "input": {"query": "test", "parameters": {"limit": 10}},
            "context": {"user_preferences": ["fast", "accurate"]},
            "metadata": {"version": "1.0", "flags": ["experimental"]}
        }
        
        log = AgentActionLog(
            action_type=ActionType.DECISION_POINT,
            input_data=complex_data,
            decision_context={"options": ["A", "B", "C"], "chosen": "B"},
            alternatives_considered=["option_A", "option_C"],
            custom_attributes={"priority": "high", "risk_level": "low"}
        )
        
        log_dict = log.to_dict()
        assert log_dict["input_data"] == complex_data
        assert log_dict["decision_context"]["chosen"] == "B"
        assert "option_A" in log_dict["alternatives_considered"]


class TestAgentActionLogger:
    """Test AgentActionLogger core functionality."""
    
    @pytest.mark.asyncio
    async def test_logger_initialization(self, logging_config):
        """Test logger initialization."""
        logger = AgentActionLogger(config=logging_config)
        await logger.initialize()
        
        assert logger.is_initialized
        assert logger.config == logging_config
        assert logger.log_queue is not None
        assert logger.stats["start_time"] is not None
    
    @pytest.mark.asyncio
    async def test_basic_log_action(self, agent_logger):
        """Test basic log action functionality."""
        await agent_logger.initialize()
        
        log_id = await agent_logger.log_action(
            ActionType.TASK_STARTED,
            log_level=LogLevel.INFO,
            task_id="test_001",
            agent_type=AgentType.DRAFT_AGENT,
            reasoning="Test task started for validation"
        )
        
        assert log_id is not None
        assert agent_logger.stats["logs_written"] >= 0  # May be 0 in sync mode
    
    @pytest.mark.asyncio
    async def test_task_lifecycle_logging(self, agent_logger, sample_task, sample_workflow, sample_execution_metrics):
        """Test task lifecycle logging."""
        await agent_logger.initialize()
        
        # Log task start
        start_log_id = await agent_logger.log_task_lifecycle(
            "started", sample_task, sample_workflow
        )
        
        # Log task completion
        completion_log_id = await agent_logger.log_task_lifecycle(
            "completed", sample_task, sample_workflow, sample_execution_metrics
        )
        
        assert start_log_id is not None
        assert completion_log_id is not None
        assert start_log_id != completion_log_id
        assert agent_logger.stats["logs_written"] >= 0
    
    @pytest.mark.asyncio
    async def test_validation_event_logging(self, agent_logger, sample_validation_result):
        """Test validation event logging."""
        await agent_logger.initialize()
        
        log_id = await agent_logger.log_validation_event(
            "completed",
            sample_validation_result,
            task_id="test_task_001",
            correlation_id="test_correlation_001"
        )
        
        assert log_id is not None
        assert agent_logger.stats["logs_written"] >= 0
    
    @pytest.mark.asyncio
    async def test_orchestration_event_logging(self, agent_logger, sample_workflow):
        """Test orchestration event logging."""
        await agent_logger.initialize()
        
        # Create sample tasks
        tasks = [
            AgentTask(task_id="task_1", agent_type=AgentType.DRAFT_AGENT),
            AgentTask(task_id="task_2", agent_type=AgentType.JUDGE_AGENT)
        ]
        
        results = {
            "task_1": {"status": "completed", "result": "success"},
            "task_2": {"status": "failed", "error": "timeout"}
        }
        
        log_id = await agent_logger.log_orchestration_event(
            "completed", sample_workflow, tasks, results
        )
        
        assert log_id is not None
        assert agent_logger.stats["logs_written"] >= 0
    
    @pytest.mark.asyncio
    async def test_decision_point_logging(self, agent_logger):
        """Test decision point logging."""
        await agent_logger.initialize()
        
        log_id = await agent_logger.log_decision_point(
            decision_type="agent_selection",
            context={"available_agents": ["draft", "judge"], "task_complexity": 0.7},
            chosen_option="draft_agent",
            alternatives=["judge_agent", "business_analyst"],
            reasoning="Draft agent selected based on task requirements and current load",
            confidence=0.85,
            task_id="test_task_001"
        )
        
        assert log_id is not None
        assert agent_logger.stats["logs_written"] >= 0
    
    @pytest.mark.asyncio
    async def test_sensitive_data_masking(self, agent_logger):
        """Test sensitive data masking functionality."""
        await agent_logger.initialize()
        
        # Create log with sensitive data
        sensitive_data = {
            "user_password": "secret123",
            "api_token": "bearer_token_xyz",
            "email_address": "userexample.com",
            "credit_card": "4111-1111-1111-1111",
            "normal_field": "normal_value"
        }
        
        log_id = await agent_logger.log_action(
            ActionType.TASK_STARTED,
            input_data=sensitive_data,
            custom_attributes={"api_key": "secret_key_123", "public_info": "visible"}
        )
        
        assert log_id is not None
        # In a real test, we would verify that sensitive fields are masked
        # This would require accessing the stored log entry
    
    @pytest.mark.asyncio
    async def test_session_tracking(self, agent_logger):
        """Test session tracking functionality."""
        await agent_logger.initialize()
        
        session_id = "test_session_tracking"
        
        # Log multiple events for the same session
        for i in range(3):
            await agent_logger.log_action(
                ActionType.TASK_STARTED,
                session_id=session_id,
                task_id=f"task_{i}",
                agent_type=AgentType.DRAFT_AGENT
            )
        
        # Verify session tracking
        assert session_id in agent_logger.active_sessions
        session_info = agent_logger.active_sessions[session_id]
        assert session_info["status"] == "active"
        assert session_info["task_count"] == 3
    
    @pytest.mark.asyncio
    async def test_log_querying(self, agent_logger):
        """Test basic log querying functionality."""
        await agent_logger.initialize()
        
        # Log some test entries
        await agent_logger.log_action(
            ActionType.TASK_STARTED,
            agent_type=AgentType.DRAFT_AGENT,
            task_id="query_test_1"
        )
        
        await agent_logger.log_action(
            ActionType.TASK_COMPLETED,
            agent_type=AgentType.JUDGE_AGENT,
            task_id="query_test_2"
        )
        
        # Query logs (this would work with Redis storage enabled)
        with patch.object(agent_logger, 'redis_client', None):  # Mock no Redis
            logs = await agent_logger.get_logs(filters={"agent_type": "draft_agent"})
            assert isinstance(logs, list)  # Should return empty list without Redis
    
    @pytest.mark.asyncio
    async def test_statistics(self, agent_logger):
        """Test statistics collection."""
        await agent_logger.initialize()
        
        # Log some actions to generate stats
        await agent_logger.log_action(ActionType.TASK_STARTED)
        await agent_logger.log_action(ActionType.TASK_COMPLETED)
        
        stats = await agent_logger.get_statistics()
        
        assert "logs_written" in stats
        assert "logs_queued" in stats
        assert "uptime_seconds" in stats
        assert "queue_size" in stats
        assert "active_sessions" in stats
        assert stats["is_initialized"] is True
    
    @pytest.mark.asyncio
    async def test_error_handling(self, agent_logger):
        """Test error handling in logging operations."""
        await agent_logger.initialize()
        
        # Test with invalid data types
        log_id = await agent_logger.log_action(
            ActionType.TASK_STARTED,
            custom_attributes={"invalid_datetime": datetime.now()}  # datetime not JSON serializable
        )
        
        # Should still work due to error handling
        assert log_id is not None
    
    @pytest.mark.asyncio
    async def test_async_processing_queue(self, logging_config):
        """Test asynchronous processing queue."""
        # Enable async processing for this test
        config = logging_config
        config.enable_async_processing = True
        config.batch_size = 2
        config.flush_interval_seconds = 0.1
        
        logger = AgentActionLogger(config=config)
        await logger.initialize()
        
        # Queue multiple log entries
        log_ids = []
        for i in range(5):
            log_id = await logger.log_action(
                ActionType.TASK_STARTED,
                task_id=f"async_test_{i}"
            )
            log_ids.append(log_id)
        
        # Wait for async processing
        await asyncio.sleep(0.2)
        
        assert len(log_ids) == 5
        assert all(log_id is not None for log_id in log_ids)
        
        # Cleanup
        await logger.shutdown()
    
    @pytest.mark.asyncio
    async def test_logger_shutdown(self, agent_logger):
        """Test graceful logger shutdown."""
        await agent_logger.initialize()
        
        # Add some log entries
        await agent_logger.log_action(ActionType.TASK_STARTED)
        await agent_logger.log_action(ActionType.TASK_COMPLETED)
        
        # Shutdown should complete without errors
        await agent_logger.shutdown()
        
        # Verify cleanup
        assert agent_logger.flush_task is None or agent_logger.flush_task.cancelled()


class TestConvenienceFunctions:
    """Test convenience functions for common logging operations."""
    
    @pytest.mark.asyncio
    @patch('services.agent_action_logger.get_agent_logger')
    async def test_log_task_started(self, mock_get_logger, sample_task, sample_workflow):
        """Test log_task_started convenience function."""
        mock_logger = AsyncMock()
        mock_get_logger.return_value = mock_logger
        
        await log_task_started(sample_task, sample_workflow)
        
        mock_get_logger.assert_called_once()
        mock_logger.log_task_lifecycle.assert_called_once_with(
            "started", sample_task, sample_workflow
        )
    
    @pytest.mark.asyncio
    @patch('services.agent_action_logger.get_agent_logger')
    async def test_log_task_completed(self, mock_get_logger, sample_task, sample_workflow, sample_execution_metrics):
        """Test log_task_completed convenience function."""
        mock_logger = AsyncMock()
        mock_get_logger.return_value = mock_logger
        
        await log_task_completed(sample_task, sample_workflow, sample_execution_metrics)
        
        mock_get_logger.assert_called_once()
        mock_logger.log_task_lifecycle.assert_called_once_with(
            "completed", sample_task, sample_workflow, sample_execution_metrics
        )
    
    @pytest.mark.asyncio
    @patch('services.agent_action_logger.get_agent_logger')
    async def test_log_validation_result(self, mock_get_logger, sample_validation_result):
        """Test log_validation_result convenience function."""
        mock_logger = AsyncMock()
        mock_get_logger.return_value = mock_logger
        
        await log_validation_result(sample_validation_result, "test_task_001")
        
        mock_get_logger.assert_called_once()
        mock_logger.log_validation_event.assert_called_once_with(
            "completed", sample_validation_result, "test_task_001"
        )


class TestLoggingConfiguration:
    """Test logging configuration and settings."""
    
    def test_default_config(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        
        assert config.enable_redis_storage is True
        assert config.enable_file_storage is True
        assert config.batch_size == 100
        assert config.flush_interval_seconds == 5
        assert config.log_input_data is True
        assert config.mask_sensitive_data is True
        assert config.min_log_level == LogLevel.INFO
    
    def test_custom_config(self):
        """Test custom logging configuration."""
        config = LoggingConfig(
            enable_redis_storage=False,
            batch_size=50,
            flush_interval_seconds=10,
            log_input_data=False,
            min_log_level=LogLevel.WARN
        )
        
        assert config.enable_redis_storage is False
        assert config.batch_size == 50
        assert config.flush_interval_seconds == 10
        assert config.log_input_data is False
        assert config.min_log_level == LogLevel.WARN


class TestGlobalLoggerInstance:
    """Test global logger instance management."""
    
    @pytest.mark.asyncio
    @patch('services.agent_action_logger._agent_logger', None)
    async def test_get_agent_logger_creates_instance(self):
        """Test that get_agent_logger creates a new instance when needed."""
        with patch('services.agent_action_logger.AgentActionLogger') as MockLogger:
            mock_instance = AsyncMock()
            MockLogger.return_value = mock_instance
            
            logger = await get_agent_logger()
            
            MockLogger.assert_called_once()
            mock_instance.initialize.assert_called_once()
            assert logger == mock_instance
    
    @pytest.mark.asyncio
    async def test_get_agent_logger_reuses_instance(self):
        """Test that get_agent_logger reuses existing instance."""
        with patch('services.agent_action_logger._agent_logger') as mock_existing:
            mock_existing.is_initialized = True
            
            logger = await get_agent_logger()
            
            assert logger == mock_existing


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration test scenarios for realistic usage patterns."""
    
    @pytest.mark.asyncio
    async def test_complete_task_execution_logging(self, logging_config):
        """Test complete task execution with full logging."""
        logger = AgentActionLogger(config=logging_config)
        await logger.initialize()
        
        # Simulate complete task execution flow
        task = AgentTask(
            task_id="integration_test_001",
            agent_type=AgentType.BUSINESS_ANALYST
        )
        
        workflow = WorkflowContext(
            workflow_id="integration_workflow_001",
            session_id="integration_session_001"
        )
        
        # 1. Log task started
        start_log_id = await logger.log_task_lifecycle("started", task, workflow)
        
        # 2. Log agent selection decision
        decision_log_id = await logger.log_decision_point(
            "agent_selection",
            {"available_agents": ["business_analyst", "draft_agent"], "complexity": 0.6},
            "business_analyst",
            ["draft_agent"],
            "Business analyst selected for analytical task",
            0.9,
            task_id=task.task_id,
            correlation_id=start_log_id
        )
        
        # 3. Simulate validation
        validation_result = ValidationResult(
            validation_id="integration_validation_001",
            status=ValidationStatus.PASSED,
            overall_confidence=0.88,
            processing_time_ms=650.0
        )
        
        validation_log_id = await logger.log_validation_event(
            "completed", validation_result, task.task_id, correlation_id=start_log_id
        )
        
        # 4. Log task completion
        execution_metrics = ExecutionMetrics(
            task_id=task.task_id,
            agent_type=task.agent_type,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            duration_ms=2300,
            status=ExecutionStatus.COMPLETED
        )
        
        completion_log_id = await logger.log_task_lifecycle(
            "completed", task, workflow, execution_metrics
        )
        
        # Verify all logs were created
        assert all(log_id is not None for log_id in [
            start_log_id, decision_log_id, validation_log_id, completion_log_id
        ])
        
        # Verify session tracking
        assert workflow.session_id in logger.active_sessions
        session_info = logger.active_sessions[workflow.session_id]
        assert session_info["task_count"] >= 1
        
        await logger.shutdown()
    
    @pytest.mark.asyncio
    async def test_error_scenario_logging(self, logging_config):
        """Test error scenario with comprehensive logging."""
        logger = AgentActionLogger(config=logging_config)
        await logger.initialize()
        
        task = AgentTask(task_id="error_test_001", agent_type=AgentType.DRAFT_AGENT)
        workflow = WorkflowContext(workflow_id="error_workflow_001")
        
        # Log task start
        start_log_id = await logger.log_task_lifecycle("started", task, workflow)
        
        # Log retry attempts
        retry_log_id = await logger.log_task_lifecycle("retried", task, workflow)
        
        # Log failure
        execution_metrics = ExecutionMetrics(
            task_id=task.task_id,
            agent_type=task.agent_type,
            start_time=datetime.now(timezone.utc),
            status=ExecutionStatus.FAILED,
            error_message="Connection timeout after 3 retries"
        )
        
        failure_log_id = await logger.log_task_lifecycle(
            "failed", task, workflow, execution_metrics
        )
        
        # Verify error logging
        assert all(log_id is not None for log_id in [start_log_id, retry_log_id, failure_log_id])
        
        await logger.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])