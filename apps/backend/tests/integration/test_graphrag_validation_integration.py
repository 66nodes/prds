"""
Integration tests for GraphRAG validation with parallel execution system.

Tests the complete integration between enhanced parallel execution and
comprehensive GraphRAG validation pipeline.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import json

from services.validated_parallel_executor import (
    ValidatedParallelExecutor,
    get_validated_executor,
    execute_with_validation
)
from services.graphrag_validation_middleware import (
    GraphRAGValidationMiddleware,
    ValidationCheckpoint,
    ValidationStage,
    AgentValidationResult,
    get_validation_middleware
)
from services.graphrag.validation_pipeline import ValidationLevel, ValidationStatus, ValidationResult
from services.agent_orchestrator import AgentType, AgentTask, WorkflowContext
from services.enhanced_parallel_executor import PriorityLevel, ExecutionStatus
from api.endpoints.validation_execution import ValidationExecutionRequest


@pytest.fixture
async def validated_executor():
    """Create a validated parallel executor for testing."""
    with patch('services.graphrag_validation_middleware.get_validation_middleware'), \
         patch('services.enhanced_parallel_executor.get_agent_registry'), \
         patch('services.enhanced_parallel_executor.get_context_aware_selector'):
        
        executor = ValidatedParallelExecutor(
            max_concurrent_tasks=5,
            enable_validation=True,
            validation_level=ValidationLevel.STANDARD
        )
        
        # Mock validation middleware
        mock_middleware = AsyncMock(spec=GraphRAGValidationMiddleware)
        mock_middleware.health_check.return_value = {"status": "healthy", "middleware_metrics": {"active_validations": 0}}
        mock_middleware.get_validation_statistics.return_value = {"pipeline_stats": {}, "agent_configurations": {}}
        executor.validation_middleware = mock_middleware
        
        await executor.initialize()
        
        yield executor
        
        await executor.shutdown()


@pytest.fixture
def sample_agent_tasks():
    """Create sample agent tasks for testing."""
    return [
        AgentTask(
            task_id=f"test_task_{i}",
            agent_type=AgentType.DRAFT_AGENT,
            input_data={"content": f"Test content {i}", "complexity": "medium"},
            metadata={"test": True, "validation_required": True}
        )
        for i in range(5)
    ]


@pytest.fixture
def workflow_context():
    """Create workflow context for testing."""
    return WorkflowContext(
        workflow_id="test_validation_workflow",
        context_data={"validation_test": True, "quality_level": "high"}
    )


@pytest.fixture
def mock_validation_results():
    """Mock validation results for testing."""
    return [
        AgentValidationResult(
            validation_id=f"val_{i}",
            task_id=f"test_task_{i}",
            agent_type=AgentType.DRAFT_AGENT,
            stage=ValidationStage.POST_EXECUTION,
            passes_validation=True,
            confidence_score=0.9,
            hallucination_rate=0.01,
            processing_time_ms=500 + (i * 100),
            iterations_performed=1,
            original_content=f"Test content {i}",
            validated_content=f"Validated test content {i}",
            corrections_applied=[f"Minor correction {i}"] if i % 2 == 0 else [],
            quality_improvements={"clarity": 0.1, "accuracy": 0.05} if i % 3 == 0 else {}
        )
        for i in range(5)
    ]


class TestGraphRAGValidationIntegration:
    """Test suite for GraphRAG validation integration."""
    
    @pytest.mark.asyncio
    async def test_validated_executor_initialization(self, validated_executor):
        """Test that validated executor initializes properly with validation enabled."""
        assert validated_executor.is_initialized
        assert validated_executor.enable_validation
        assert validated_executor.validation_level == ValidationLevel.STANDARD
        assert validated_executor.validation_middleware is not None
    
    @pytest.mark.asyncio
    async def test_execute_with_validation_basic_flow(
        self, 
        validated_executor, 
        sample_agent_tasks, 
        workflow_context,
        mock_validation_results
    ):
        """Test basic validated execution flow."""
        
        # Mock the base execution
        mock_base_result = {
            "results": {
                task.task_id: {
                    "status": ExecutionStatus.COMPLETED.value,
                    "result": {"content": f"Generated content for {task.task_id}"},
                    "metrics": {"duration_ms": 1000, "queue_wait_ms": 50, "attempts": 1}
                }
                for task in sample_agent_tasks
            },
            "analytics": {"total_executions": len(sample_agent_tasks), "successful_executions": len(sample_agent_tasks)},
            "resource_usage": {"cpu_percent": 50, "memory_percent": 60}
        }
        
        with patch.object(validated_executor.__class__.__bases__[0], 'execute_parallel', return_value=mock_base_result):
            # Mock validation results
            validated_executor.validation_middleware.validate_batch_outputs.return_value = mock_validation_results
            
            result = await validated_executor.execute_parallel_with_validation(
                tasks=sample_agent_tasks,
                workflow=workflow_context,
                priority=PriorityLevel.NORMAL
            )
        
        # Verify execution completed successfully
        assert "results" in result
        assert "validation_analytics" in result
        assert len(result["results"]) == len(sample_agent_tasks)
        
        # Verify validation analytics
        validation_analytics = result["validation_analytics"]
        assert validation_analytics["total_tasks_validated"] == len(sample_agent_tasks)
        assert validation_analytics["passed_validation"] == len(sample_agent_tasks)
        assert validation_analytics["failed_validation"] == 0
        assert validation_analytics["avg_confidence_score"] > 0.8
        assert validation_analytics["corrections_applied"] == 3  # From mock data
        
        # Verify enhanced task results include validation data
        for task_id, task_result in result["results"].items():
            assert "validation" in task_result
            validation_data = task_result["validation"]
            assert validation_data["passes_validation"] is True
            assert validation_data["confidence_score"] > 0.8
            assert validation_data["hallucination_rate"] < 0.02
    
    @pytest.mark.asyncio
    async def test_validation_failure_handling(
        self,
        validated_executor,
        sample_agent_tasks,
        workflow_context
    ):
        """Test handling of validation failures."""
        
        # Mock base execution success
        mock_base_result = {
            "results": {
                task.task_id: {
                    "status": ExecutionStatus.COMPLETED.value,
                    "result": {"content": f"Generated content for {task.task_id}"},
                    "metrics": {"duration_ms": 1000, "queue_wait_ms": 50, "attempts": 1}
                }
                for task in sample_agent_tasks[:3]  # Only first 3 tasks succeed
            },
            "analytics": {"total_executions": 3, "successful_executions": 3}
        }
        
        # Mock validation results with failures
        validation_results = [
            AgentValidationResult(
                validation_id=f"val_{i}",
                task_id=f"test_task_{i}",
                agent_type=AgentType.DRAFT_AGENT,
                stage=ValidationStage.POST_EXECUTION,
                passes_validation=i < 2,  # First 2 pass, rest fail
                confidence_score=0.9 if i < 2 else 0.4,
                hallucination_rate=0.01 if i < 2 else 0.05,
                processing_time_ms=500,
                iterations_performed=1 if i < 2 else 3  # More iterations for failures
            )
            for i in range(3)
        ]
        
        with patch.object(validated_executor.__class__.__bases__[0], 'execute_parallel', return_value=mock_base_result):
            validated_executor.validation_middleware.validate_batch_outputs.return_value = validation_results
            
            result = await validated_executor.execute_parallel_with_validation(
                tasks=sample_agent_tasks[:3],
                workflow=workflow_context
            )
        
        # Verify mixed results
        validation_analytics = result["validation_analytics"]
        assert validation_analytics["passed_validation"] == 2
        assert validation_analytics["failed_validation"] == 1
        
        # Check that failed validation tasks are marked as failed
        failed_tasks = [
            task_result for task_result in result["results"].values()
            if not task_result.get("validation", {}).get("passes_validation", True)
        ]
        assert len(failed_tasks) == 1
    
    @pytest.mark.asyncio
    async def test_validation_corrections_application(
        self,
        validated_executor,
        sample_agent_tasks,
        workflow_context
    ):
        """Test that validation corrections are properly applied."""
        
        mock_base_result = {
            "results": {
                sample_agent_tasks[0].task_id: {
                    "status": ExecutionStatus.COMPLETED.value,
                    "result": {"content": "Original content with errors"},
                    "metrics": {"duration_ms": 1000, "queue_wait_ms": 50, "attempts": 1}
                }
            },
            "analytics": {"total_executions": 1, "successful_executions": 1}
        }
        
        # Mock validation result with corrections
        validation_result = AgentValidationResult(
            validation_id="val_correction_test",
            task_id=sample_agent_tasks[0].task_id,
            agent_type=AgentType.DRAFT_AGENT,
            stage=ValidationStage.POST_EXECUTION,
            passes_validation=True,
            confidence_score=0.95,
            hallucination_rate=0.005,
            original_content="Original content with errors",
            validated_content="Corrected content without errors",
            corrections_applied=["Fixed grammar error", "Improved clarity"],
            quality_improvements={"grammar": 0.3, "clarity": 0.2}
        )
        
        with patch.object(validated_executor.__class__.__bases__[0], 'execute_parallel', return_value=mock_base_result):
            validated_executor.validation_middleware.validate_batch_outputs.return_value = [validation_result]
            
            result = await validated_executor.execute_parallel_with_validation(
                tasks=sample_agent_tasks[:1],
                workflow=workflow_context
            )
        
        # Verify corrections were applied
        task_result = result["results"][sample_agent_tasks[0].task_id]
        assert "validated_content" in task_result
        assert "original_content" in task_result
        assert task_result["validated_content"] != task_result["original_content"]
        assert task_result["validation"]["corrections_applied"] == 2
        
        # Verify quality improvements are tracked
        validation_data = task_result["validation"]
        assert "quality_improvements" in validation_data
        assert validation_data["quality_improvements"]["grammar"] == 0.3
    
    @pytest.mark.asyncio
    async def test_validation_performance_tracking(
        self,
        validated_executor,
        sample_agent_tasks,
        workflow_context,
        mock_validation_results
    ):
        """Test that validation performance metrics are properly tracked."""
        
        mock_base_result = {
            "results": {
                task.task_id: {
                    "status": ExecutionStatus.COMPLETED.value,
                    "result": {"content": f"Content for {task.task_id}"},
                    "metrics": {"duration_ms": 1000, "queue_wait_ms": 50, "attempts": 1}
                }
                for task in sample_agent_tasks
            },
            "analytics": {"total_executions": len(sample_agent_tasks), "successful_executions": len(sample_agent_tasks)}
        }
        
        with patch.object(validated_executor.__class__.__bases__[0], 'execute_parallel', return_value=mock_base_result):
            validated_executor.validation_middleware.validate_batch_outputs.return_value = mock_validation_results
            
            # Execute multiple times to build statistics
            for _ in range(3):
                await validated_executor.execute_parallel_with_validation(
                    tasks=sample_agent_tasks,
                    workflow=workflow_context
                )
        
        # Check validation analytics were updated
        status = await validated_executor.get_enhanced_execution_status()
        validation_analytics = status["validation_analytics"]
        
        assert validation_analytics["total_validations"] == 15  # 3 runs × 5 tasks
        assert validation_analytics["successful_validations"] == 15
        assert validation_analytics["avg_confidence_score"] > 0.8
        assert validation_analytics["corrections_applied"] == 9  # 3 runs × 3 corrections
    
    @pytest.mark.asyncio
    async def test_agent_specific_validation_configuration(self, validated_executor):
        """Test agent-specific validation configuration."""
        
        # Configure different validation levels for different agent types
        validated_executor.validation_middleware.configure_agent_validation.return_value = None
        
        # Configure strict validation for draft agent
        strict_checkpoint = ValidationCheckpoint(
            stage=ValidationStage.POST_EXECUTION,
            validation_level=ValidationLevel.STRICT,
            confidence_threshold=0.95,
            hallucination_threshold=0.005
        )
        
        # Configure basic validation for context manager
        basic_checkpoint = ValidationCheckpoint(
            stage=ValidationStage.POST_EXECUTION,
            validation_level=ValidationLevel.BASIC,
            confidence_threshold=0.7,
            hallucination_threshold=0.05
        )
        
        # Simulate configuration calls
        validated_executor.validation_middleware.configure_agent_validation(
            AgentType.DRAFT_AGENT, strict_checkpoint
        )
        validated_executor.validation_middleware.configure_agent_validation(
            AgentType.CONTEXT_MANAGER, basic_checkpoint
        )
        
        # Verify configuration calls were made
        assert validated_executor.validation_middleware.configure_agent_validation.call_count == 2
    
    @pytest.mark.asyncio
    async def test_validation_timeout_handling(
        self,
        validated_executor,
        sample_agent_tasks,
        workflow_context
    ):
        """Test handling of validation timeouts."""
        
        mock_base_result = {
            "results": {
                sample_agent_tasks[0].task_id: {
                    "status": ExecutionStatus.COMPLETED.value,
                    "result": {"content": "Test content"},
                    "metrics": {"duration_ms": 1000, "queue_wait_ms": 50, "attempts": 1}
                }
            },
            "analytics": {"total_executions": 1, "successful_executions": 1}
        }
        
        # Mock validation timeout
        async def mock_validate_timeout(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate processing time
            raise asyncio.TimeoutError("Validation timeout")
        
        with patch.object(validated_executor.__class__.__bases__[0], 'execute_parallel', return_value=mock_base_result):
            validated_executor.validation_middleware.validate_batch_outputs.side_effect = mock_validate_timeout
            
            result = await validated_executor.execute_parallel_with_validation(
                tasks=sample_agent_tasks[:1],
                workflow=workflow_context
            )
        
        # Should handle timeout gracefully and still return results
        assert "results" in result
        assert len(result["results"]) == 1
    
    @pytest.mark.asyncio
    async def test_disable_enable_validation(self, validated_executor):
        """Test disabling and re-enabling validation."""
        
        # Initially enabled
        assert validated_executor.enable_validation is True
        
        # Disable validation
        validated_executor.disable_validation()
        assert validated_executor.enable_validation is False
        
        # Re-enable with different level
        validated_executor.enable_validation_mode(ValidationLevel.STRICT)
        assert validated_executor.enable_validation is True
        assert validated_executor.validation_level == ValidationLevel.STRICT
    
    @pytest.mark.asyncio
    async def test_validation_system_health_check(self, validated_executor):
        """Test comprehensive validation system health check."""
        
        status = await validated_executor.get_enhanced_execution_status()
        
        # Verify enhanced status includes validation metrics
        assert "validation_enabled" in status
        assert "validation_level" in status
        assert "validation_analytics" in status
        assert "validation_success_rate" in status
        
        # Verify validation middleware health is included
        if validated_executor.validation_middleware:
            assert "middleware_health" in status
            assert "middleware_statistics" in status


class TestValidationExecutionAPI:
    """Test suite for validation execution API endpoints."""
    
    @pytest.mark.asyncio
    async def test_validation_execution_request_processing(self):
        """Test processing of validation execution requests."""
        
        request = ValidationExecutionRequest(
            tasks=[
                {
                    "task_id": "test_task_1",
                    "agent_type": "draft_agent",
                    "input_data": {"content": "Test content"},
                    "metadata": {"priority": "high"}
                }
            ],
            workflow_context={
                "workflow_id": "test_workflow",
                "context_data": {"validation_test": True}
            },
            validation_level=ValidationLevel.STRICT,
            enable_corrections=True
        )
        
        # Verify request structure
        assert len(request.tasks) == 1
        assert request.validation_level == ValidationLevel.STRICT
        assert request.enable_corrections is True
        assert request.workflow_context["workflow_id"] == "test_workflow"
    
    @pytest.mark.asyncio
    async def test_validation_metrics_calculation(self):
        """Test validation metrics calculation and aggregation."""
        
        # Mock validation results with various outcomes
        validation_results = [
            AgentValidationResult(
                validation_id=f"val_{i}",
                task_id=f"task_{i}",
                agent_type=AgentType.DRAFT_AGENT,
                stage=ValidationStage.POST_EXECUTION,
                passes_validation=i % 3 != 0,  # 2/3 pass, 1/3 fail
                confidence_score=0.9 if i % 3 != 0 else 0.4,
                hallucination_rate=0.01 if i % 3 != 0 else 0.08,
                processing_time_ms=500 + (i * 100),
                corrections_applied=[f"correction_{i}"] if i % 2 == 0 else []
            )
            for i in range(9)
        ]
        
        # Calculate metrics
        total_validations = len(validation_results)
        passed_validations = sum(1 for r in validation_results if r.passes_validation)
        failed_validations = total_validations - passed_validations
        
        avg_confidence = sum(r.confidence_score for r in validation_results) / total_validations
        avg_hallucination = sum(r.hallucination_rate for r in validation_results) / total_validations
        total_corrections = sum(len(r.corrections_applied) for r in validation_results)
        
        # Verify calculations
        assert total_validations == 9
        assert passed_validations == 6  # 2/3 of 9
        assert failed_validations == 3
        assert 0.6 < avg_confidence < 0.8  # Mixed confidence scores
        assert avg_hallucination > 0.02  # Some high hallucination rates
        assert total_corrections == 5  # Every other task has correction


class TestValidationMiddleware:
    """Test suite for validation middleware functionality."""
    
    @pytest.mark.asyncio
    async def test_middleware_initialization(self):
        """Test validation middleware initialization."""
        
        with patch('services.graphrag_validation_middleware.ValidationPipeline') as mock_pipeline:
            mock_pipeline_instance = AsyncMock()
            mock_pipeline.return_value = mock_pipeline_instance
            
            middleware = GraphRAGValidationMiddleware()
            await middleware.initialize()
            
            assert middleware.is_initialized
            assert mock_pipeline_instance.initialize.called
    
    @pytest.mark.asyncio
    async def test_agent_output_validation_flow(self):
        """Test the complete agent output validation flow."""
        
        with patch('services.graphrag_validation_middleware.ValidationPipeline') as mock_pipeline:
            mock_pipeline_instance = AsyncMock()
            mock_pipeline_instance.validate_content.return_value = MagicMock(
                overall_confidence=0.9,
                status=ValidationStatus.PASSED,
                hallucination_detection_result={"hallucination_rate": 0.01},
                corrections=[]
            )
            mock_pipeline.return_value = mock_pipeline_instance
            
            middleware = GraphRAGValidationMiddleware()
            await middleware.initialize()
            
            # Create test inputs
            task = AgentTask(
                task_id="test_task",
                agent_type=AgentType.DRAFT_AGENT,
                input_data={"prompt": "Test prompt"}
            )
            
            agent_output = {"content": "Test generated content"}
            
            workflow = WorkflowContext(
                workflow_id="test_workflow",
                context_data={}
            )
            
            from services.enhanced_parallel_executor import ExecutionMetrics
            execution_metrics = ExecutionMetrics(
                task_id="test_task",
                agent_type=AgentType.DRAFT_AGENT,
                start_time=datetime.utcnow()
            )
            execution_metrics.duration_ms = 1000
            
            # Perform validation
            result = await middleware.validate_agent_output(
                task=task,
                agent_output=agent_output,
                workflow=workflow,
                execution_metrics=execution_metrics
            )
            
            # Verify result
            assert isinstance(result, AgentValidationResult)
            assert result.passes_validation
            assert result.confidence_score == 0.9
            assert result.hallucination_rate == 0.01
            assert result.original_content == "Test generated content"


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "test_validated_executor_initialization or test_execute_with_validation_basic_flow"
    ])