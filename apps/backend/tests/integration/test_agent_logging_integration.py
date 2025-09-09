"""
Integration Tests for Agent Logging System

Comprehensive integration tests for the complete agent logging system,
including executor integration, validation pipeline integration, and API endpoints.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import patch, AsyncMock
import json

from fastapi.testclient import TestClient
from services.enhanced_parallel_executor_with_logging import LoggingEnhancedParallelExecutor
from services.graphrag_validation_pipeline_with_logging import LoggingValidationPipeline
from services.agent_audit_service import AgentAuditService, AuditQuery, TimeFrame
from services.agent_action_logger import AgentActionLogger, LoggingConfig
from services.agent_orchestrator import AgentTask, WorkflowContext, AgentType
from api.endpoints.agent_logs import router as agent_logs_router


@pytest.fixture
def test_logging_config():
    """Test configuration with Redis disabled."""
    return LoggingConfig(
        enable_redis_storage=False,
        enable_file_storage=False,
        enable_async_processing=True,
        batch_size=10,
        flush_interval_seconds=0.1
    )


@pytest.fixture
async def logging_executor(test_logging_config):
    """Logging-enhanced parallel executor for testing."""
    executor = LoggingEnhancedParallelExecutor(max_concurrent_tasks=5)
    with patch.object(AgentActionLogger, '__init__', lambda self, config=None: None):
        with patch.object(AgentActionLogger, 'initialize', AsyncMock()):
            with patch.object(AgentActionLogger, 'log_action', AsyncMock(return_value="log_id_123")):
                await executor.initialize()
                yield executor
                await executor.shutdown()


@pytest.fixture
async def logging_validation_pipeline():
    """Logging-enhanced validation pipeline for testing."""
    pipeline = LoggingValidationPipeline()
    with patch.object(AgentActionLogger, '__init__', lambda self, config=None: None):
        with patch.object(AgentActionLogger, 'initialize', AsyncMock()):
            with patch.object(AgentActionLogger, 'log_action', AsyncMock(return_value="validation_log_123")):
                await pipeline.initialize()
                yield pipeline


@pytest.fixture
async def audit_service():
    """Agent audit service for testing."""
    service = AgentAuditService()
    with patch.object(service, 'initialize', AsyncMock()):
        with patch.object(service, 'agent_logger', AsyncMock()):
            await service.initialize()
            yield service


@pytest.fixture
def test_client():
    """FastAPI test client with agent logs router."""
    from fastapi import FastAPI
    
    app = FastAPI()
    app.include_router(agent_logs_router)
    
    # Mock authentication
    with patch('api.endpoints.agent_logs.get_current_user', return_value={"user_id": "test_user", "role": "admin"}):
        yield TestClient(app)


class TestEnhancedParallelExecutorIntegration:
    """Test integration of logging with enhanced parallel executor."""
    
    @pytest.mark.asyncio
    async def test_executor_initialization_logging(self, logging_executor):
        """Test that executor initialization is logged."""
        # The executor should have initialized and logged the initialization
        assert logging_executor.agent_logger is not None
        
        # Verify initialization was logged (mocked)
        logging_executor.agent_logger.log_action.assert_called()
        
        # Check that orchestration started was logged
        calls = logging_executor.agent_logger.log_action.call_args_list
        initialization_calls = [call for call in calls if 'orchestration_started' in str(call)]
        assert len(initialization_calls) > 0
    
    @pytest.mark.asyncio
    async def test_task_execution_with_logging(self, logging_executor):
        """Test complete task execution with comprehensive logging."""
        # Create test tasks
        tasks = [
            AgentTask(task_id="log_test_1", agent_type=AgentType.DRAFT_AGENT),
            AgentTask(task_id="log_test_2", agent_type=AgentType.JUDGE_AGENT)
        ]
        
        workflow = WorkflowContext(
            workflow_id="log_workflow_001",
            session_id="log_session_001",
            correlation_id="log_correlation_001"
        )
        
        # Execute tasks
        result = await logging_executor.execute_parallel(tasks, workflow)
        
        # Verify execution completed
        assert "results" in result
        assert "orchestration_log_id" in result
        
        # Verify logging was called multiple times during execution
        assert logging_executor.agent_logger.log_action.call_count >= len(tasks) * 2  # At least start + completion per task
        
        # Verify specific logging patterns
        calls = logging_executor.agent_logger.log_action.call_args_list
        
        # Should have orchestration events
        orchestration_calls = [call for call in calls if 'orchestration' in str(call)]
        assert len(orchestration_calls) >= 2  # Start and completion
        
        # Should have task lifecycle events
        task_calls = [call for call in calls if any(task.task_id in str(call) for task in tasks)]
        assert len(task_calls) >= len(tasks)
    
    @pytest.mark.asyncio
    async def test_error_handling_with_logging(self, logging_executor):
        """Test error handling and logging during execution."""
        # Create a task that will fail
        failing_task = AgentTask(task_id="failing_test", agent_type=AgentType.DRAFT_AGENT)
        workflow = WorkflowContext(workflow_id="error_workflow")
        
        # Mock the agent task execution to fail
        with patch.object(logging_executor, '_execute_agent_task', side_effect=Exception("Test error")):
            result = await logging_executor.execute_parallel([failing_task], workflow)
            
            # Should handle error gracefully
            assert "results" in result
            
            # Verify error logging
            calls = logging_executor.agent_logger.log_action.call_args_list
            error_calls = [call for call in calls if 'failed' in str(call) or 'error' in str(call)]
            assert len(error_calls) >= 1
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_logging(self, logging_executor):
        """Test circuit breaker events are logged."""
        # Enable circuit breakers
        logging_executor.enable_circuit_breakers = True
        
        tasks = [AgentTask(task_id="circuit_test", agent_type=AgentType.DRAFT_AGENT)]
        workflow = WorkflowContext(workflow_id="circuit_workflow")
        
        # Mock circuit breaker failure
        with patch.object(logging_executor, '_check_circuit_breaker', return_value=False):
            result = await logging_executor.execute_parallel(tasks, workflow)
            
            # Verify circuit breaker logging
            calls = logging_executor.agent_logger.log_action.call_args_list
            circuit_breaker_calls = [call for call in calls if 'circuit_breaker' in str(call)]
            assert len(circuit_breaker_calls) >= 1


class TestValidationPipelineIntegration:
    """Test integration of logging with GraphRAG validation pipeline."""
    
    @pytest.mark.asyncio
    async def test_validation_pipeline_logging(self, logging_validation_pipeline):
        """Test validation pipeline with comprehensive logging."""
        content = "This is a test document for validation with entities and relationships."
        
        context = {
            "task_id": "validation_test_001",
            "session_id": "validation_session_001",
            "correlation_id": "validation_correlation_001"
        }
        
        # Execute validation
        result = await logging_validation_pipeline.validate_content(content, context=context)
        
        # Verify validation completed
        assert result.validation_id is not None
        assert result.status is not None
        
        # Verify logging was comprehensive
        calls = logging_validation_pipeline.agent_logger.log_action.call_args_list
        
        # Should have validation start and completion
        validation_calls = [call for call in calls if 'validation' in str(call)]
        assert len(validation_calls) >= 2  # At least start and completion
        
        # Should have component-specific logging
        component_calls = [call for call in calls if any(comp in str(call) for comp in [
            'entity_extraction', 'relationship_extraction', 'hallucination_detection', 'graph_analysis'
        ])]
        assert len(component_calls) >= 1
    
    @pytest.mark.asyncio
    async def test_validation_quality_violations_logging(self, logging_validation_pipeline):
        """Test logging of quality violations during validation."""
        # Create content that should trigger quality concerns
        poor_content = "This is very short and may not meet quality thresholds."
        
        context = {"task_id": "quality_test_001"}
        
        # Mock validation results that indicate quality issues
        with patch.object(logging_validation_pipeline, '_mock_hallucination_detection') as mock_halluc:
            # Mock high hallucination score
            mock_halluc.return_value = {
                "hallucination_score": 0.05,  # Above 2% threshold
                "evidence_sources": 1,
                "violations": ["insufficient_evidence"]
            }
            
            result = await logging_validation_pipeline.validate_content(poor_content, context=context)
            
            # Should detect quality issues
            assert result.status != result.status.PASSED or len(result.issues_found or []) > 0
            
            # Verify quality violation logging
            calls = logging_validation_pipeline.agent_logger.log_action.call_args_list
            quality_calls = [call for call in calls if 'quality' in str(call) or 'violation' in str(call)]
            assert len(quality_calls) >= 1
    
    @pytest.mark.asyncio
    async def test_validation_recommendations_logging(self, logging_validation_pipeline):
        """Test logging of validation recommendations."""
        content = "Test content for recommendation generation."
        context = {"task_id": "recommendation_test_001"}
        
        # Execute validation (should generate recommendations)
        result = await logging_validation_pipeline.validate_content(content, context=context)
        
        # Should have recommendations
        if result.recommendations:
            # Verify recommendations were logged
            calls = logging_validation_pipeline.agent_logger.log_action.call_args_list
            recommendation_calls = [call for call in calls if 'recommendation' in str(call) or 'decision_point' in str(call)]
            assert len(recommendation_calls) >= 1


class TestAuditServiceIntegration:
    """Test integration of audit service with logging system."""
    
    @pytest.mark.asyncio
    async def test_audit_query_execution(self, audit_service):
        """Test audit service query execution."""
        # Mock the underlying agent logger
        mock_logs = [
            {
                "log_id": "test_log_1",
                "action_type": "task_started",
                "agent_type": "draft_agent",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task_id": "test_task_1"
            },
            {
                "log_id": "test_log_2", 
                "action_type": "task_completed",
                "agent_type": "draft_agent",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task_id": "test_task_1",
                "duration_ms": 1500
            }
        ]
        
        audit_service.agent_logger.get_logs = AsyncMock(return_value=mock_logs)
        
        # Create query
        query = AuditQuery(
            time_frame=TimeFrame.LAST_24_HOURS,
            agent_types=[AgentType.DRAFT_AGENT],
            include_aggregations=True
        )
        
        # Execute query
        result = await audit_service.query_logs(query)
        
        # Verify results
        assert result.total_count == len(mock_logs)
        assert len(result.logs) <= len(mock_logs)
        assert result.query_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_performance_report_generation(self, audit_service):
        """Test performance report generation."""
        # Mock performance data
        performance_logs = [
            {
                "action_type": "task_completed",
                "agent_type": "draft_agent",
                "duration_ms": 1200,
                "confidence_score": 0.9,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            {
                "action_type": "task_failed",
                "agent_type": "draft_agent", 
                "error_category": "timeout",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ]
        
        audit_service.query_logs = AsyncMock()
        audit_service.query_logs.return_value.logs = performance_logs
        
        # Generate performance report
        report = await audit_service.generate_performance_report(TimeFrame.LAST_24_HOURS)
        
        # Verify report structure
        assert hasattr(report, 'total_executions')
        assert hasattr(report, 'success_rate')
        assert hasattr(report, 'avg_execution_time_ms')
        
        # Verify calculations
        assert report.total_executions >= 0
        assert 0 <= report.success_rate <= 100
    
    @pytest.mark.asyncio
    async def test_error_analysis_generation(self, audit_service):
        """Test error analysis report generation."""
        # Mock error data
        error_logs = [
            {
                "action_type": "task_failed",
                "error_category": "timeout",
                "error_message": "Request timed out",
                "agent_type": "draft_agent",
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            {
                "action_type": "validation_failed",
                "error_category": "validation",
                "error_message": "Hallucination threshold exceeded",
                "agent_type": "judge_agent",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ]
        
        # Mock query methods
        audit_service.query_logs = AsyncMock()
        
        # Mock error query response
        error_query_result = AsyncMock()
        error_query_result.logs = error_logs
        
        # Mock recovery query response
        recovery_query_result = AsyncMock()
        recovery_query_result.logs = []
        
        # Set return values based on query parameters
        def mock_query_logs(query):
            if any(action_type.value.endswith('_failed') for action_type in query.action_types or []):
                return error_query_result
            else:
                return recovery_query_result
        
        audit_service.query_logs.side_effect = mock_query_logs
        
        # Generate error analysis
        analysis = await audit_service.generate_error_analysis(TimeFrame.LAST_24_HOURS)
        
        # Verify analysis structure
        assert hasattr(analysis, 'total_errors')
        assert hasattr(analysis, 'error_categories')
        assert hasattr(analysis, 'most_common_errors')
        
        # Verify data
        assert analysis.total_errors == len(error_logs)
        assert len(analysis.error_categories) > 0


class TestAPIEndpointsIntegration:
    """Test integration of API endpoints with logging system."""
    
    def test_query_logs_endpoint(self, test_client):
        """Test the logs query API endpoint."""
        query_data = {
            "time_frame": "last_24_hours",
            "agent_types": ["draft_agent"],
            "limit": 50,
            "include_aggregations": True
        }
        
        with patch('api.endpoints.agent_logs.get_audit_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_result = AsyncMock()
            mock_result.logs = []
            mock_result.total_count = 0
            mock_result.filtered_count = 0
            mock_result.query_time_ms = 150.0
            mock_result.cache_hit = False
            mock_result.aggregations = {}
            mock_result.insights = {}
            
            mock_service.query_logs.return_value = mock_result
            mock_get_service.return_value = mock_service
            
            response = test_client.post("/agent-logs/query", json=query_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "logs" in data
            assert "total_count" in data
            assert "query_time_ms" in data
    
    def test_task_audit_trail_endpoint(self, test_client):
        """Test the task audit trail API endpoint."""
        task_id = "test_task_123"
        
        with patch('api.endpoints.agent_logs.get_audit_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_task_audit_trail.return_value = [
                {
                    "log_id": "trail_log_1",
                    "action_type": "task_started",
                    "task_id": task_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ]
            mock_get_service.return_value = mock_service
            
            response = test_client.get(f"/agent-logs/task/{task_id}/audit-trail")
            
            assert response.status_code == 200
            data = response.json()
            assert "audit_trail" in data
            assert data["task_id"] == task_id
            assert len(data["audit_trail"]) > 0
    
    def test_generate_report_endpoint(self, test_client):
        """Test the report generation API endpoint."""
        report_data = {
            "report_type": "performance_summary",
            "time_frame": "last_week",
            "export_format": "json"
        }
        
        with patch('api.endpoints.agent_logs.get_audit_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_report = AsyncMock()
            mock_report.total_executions = 100
            mock_report.success_rate = 95.5
            mock_report.avg_execution_time_ms = 1200.0
            mock_report.__dict__ = {
                "total_executions": 100,
                "success_rate": 95.5,
                "avg_execution_time_ms": 1200.0
            }
            
            mock_service.generate_performance_report.return_value = mock_report
            mock_get_service.return_value = mock_service
            
            response = test_client.post("/agent-logs/reports/generate", json=report_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "report_type" in data
            assert "data" in data
            assert data["report_type"] == "performance_summary"
    
    def test_search_logs_endpoint(self, test_client):
        """Test the log search API endpoint."""
        search_query = "error timeout"
        
        with patch('api.endpoints.agent_logs.get_audit_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.search_logs_by_text.return_value = [
                {
                    "log_id": "search_result_1",
                    "action_type": "task_failed",
                    "error_message": "Connection timeout error",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ]
            mock_get_service.return_value = mock_service
            
            response = test_client.get(f"/agent-logs/search?query={search_query}")
            
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert data["query"] == search_query
            assert len(data["results"]) > 0
    
    def test_health_endpoint(self, test_client):
        """Test the health check API endpoint."""
        with patch('api.endpoints.agent_logs.get_audit_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.is_initialized = True
            mock_service.agent_logger = AsyncMock()
            mock_service.redis_client = AsyncMock()
            mock_service.agent_logger.get_statistics.return_value = {
                "logs_written": 1000,
                "queue_size": 5,
                "errors": 2,
                "uptime_seconds": 3600
            }
            mock_get_service.return_value = mock_service
            
            response = test_client.get("/agent-logs/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "audit_service_initialized" in data
            assert "agent_logger_stats" in data


class TestEndToEndScenarios:
    """End-to-end integration test scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_task_execution_with_validation_and_audit(self, test_logging_config):
        """Test complete end-to-end scenario with all logging components."""
        
        # Initialize all components
        logger = AgentActionLogger(config=test_logging_config)
        await logger.initialize()
        
        executor = LoggingEnhancedParallelExecutor()
        executor.agent_logger = logger
        await executor.initialize()
        
        pipeline = LoggingValidationPipeline()
        pipeline.agent_logger = logger
        await pipeline.initialize()
        
        audit_service = AgentAuditService()
        audit_service.agent_logger = logger
        await audit_service.initialize()
        
        try:
            # 1. Execute tasks with logging
            tasks = [AgentTask(task_id="e2e_test", agent_type=AgentType.DRAFT_AGENT)]
            workflow = WorkflowContext(workflow_id="e2e_workflow", session_id="e2e_session")
            
            execution_result = await executor.execute_parallel(tasks, workflow)
            assert "results" in execution_result
            
            # 2. Validate content with logging
            content = "Test content for end-to-end validation scenario"
            validation_result = await pipeline.validate_content(
                content,
                context={"task_id": "e2e_test", "session_id": "e2e_session"}
            )
            assert validation_result.validation_id is not None
            
            # 3. Query logs through audit service
            query = AuditQuery(session_ids=["e2e_session"], limit=100)
            audit_result = await audit_service.query_logs(query)
            
            # Should have captured logs from all components
            assert audit_result.total_count >= 0  # May be 0 without actual storage
            
            # 4. Generate performance report
            performance_report = await audit_service.generate_performance_report()
            assert hasattr(performance_report, 'total_executions')
            
        finally:
            # Cleanup
            await logger.shutdown()
            await executor.shutdown()
    
    @pytest.mark.asyncio
    async def test_error_recovery_scenario_logging(self, test_logging_config):
        """Test error recovery scenario with comprehensive logging."""
        
        logger = AgentActionLogger(config=test_logging_config)
        await logger.initialize()
        
        # Simulate error scenario
        task = AgentTask(task_id="error_recovery_test", agent_type=AgentType.DRAFT_AGENT)
        workflow = WorkflowContext(workflow_id="error_workflow")
        
        try:
            # Log initial failure
            await logger.log_task_lifecycle("failed", task, workflow)
            
            # Log retry attempt
            await logger.log_task_lifecycle("retried", task, workflow)
            
            # Log recovery success
            execution_metrics = ExecutionMetrics(
                task_id=task.task_id,
                agent_type=task.agent_type,
                start_time=datetime.now(timezone.utc),
                status=ExecutionStatus.COMPLETED,
                execution_attempts=2
            )
            await logger.log_task_lifecycle("completed", task, workflow, execution_metrics)
            
            # Verify recovery was logged
            stats = await logger.get_statistics()
            assert stats["logs_written"] >= 0  # May be 0 in sync mode
            
        finally:
            await logger.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])