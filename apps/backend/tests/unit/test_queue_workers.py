"""
Unit tests for queue workers.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from services.queue.workers import (
    TaskWorker, GraphRAGWorker, AgentWorker, NotificationWorker, WorkerManager
)
from services.queue.message_queue import QueueMessage
from services.queue.task_types import TaskType, TaskPriority


class TestTaskWorker:
    """Test abstract TaskWorker class."""
    
    class ConcreteTaskWorker(TaskWorker):
        """Concrete implementation for testing."""
        
        def can_handle(self, task_type: TaskType) -> bool:
            return task_type == TaskType.NOTIFICATION_DELIVERY
        
        async def process_task(self, message: QueueMessage) -> dict:
            return {"status": "processed"}
    
    def test_worker_initialization(self):
        """Test worker initialization."""
        worker = self.ConcreteTaskWorker()
        
        assert worker.worker_id is not None
        assert worker.running is False
        assert worker.processed_count == 0
        assert worker.error_count == 0
    
    def test_worker_can_handle(self):
        """Test worker task type handling."""
        worker = self.ConcreteTaskWorker()
        
        assert worker.can_handle(TaskType.NOTIFICATION_DELIVERY) is True
        assert worker.can_handle(TaskType.GRAPHRAG_VALIDATION) is False
    
    def test_worker_stats(self):
        """Test worker statistics."""
        worker = self.ConcreteTaskWorker()
        worker.processed_count = 10
        worker.error_count = 2
        
        stats = worker.get_stats()
        
        assert stats["worker_id"] == worker.worker_id
        assert stats["processed_count"] == 10
        assert stats["error_count"] == 2
        assert stats["success_rate"] == 10 / 12  # 10 / (10 + 2)
        assert stats["running"] is False


class TestGraphRAGWorker:
    """Test GraphRAGWorker functionality."""
    
    def test_can_handle_graphrag_tasks(self):
        """Test GraphRAG worker task handling."""
        worker = GraphRAGWorker()
        
        assert worker.can_handle(TaskType.GRAPHRAG_VALIDATION) is True
        assert worker.can_handle(TaskType.CONTENT_VALIDATION) is True
        assert worker.can_handle(TaskType.NOTIFICATION_DELIVERY) is False
    
    @pytest.mark.asyncio
    async def test_process_graphrag_validation(self):
        """Test GraphRAG validation processing."""
        worker = GraphRAGWorker()
        
        message = QueueMessage(
            task_type=TaskType.GRAPHRAG_VALIDATION,
            priority=TaskPriority.HIGH,
            payload={
                "content": "Test content for validation",
                "project_id": "project-123",
                "validation_type": "comprehensive"
            }
        )
        
        result = await worker.process_task(message)
        
        assert "content" in result
        assert "project_id" in result
        assert "hallucination_score" in result
        assert "confidence" in result
        assert result["project_id"] == "project-123"
    
    @pytest.mark.asyncio
    async def test_process_content_validation(self):
        """Test content validation processing."""
        worker = GraphRAGWorker()
        
        message = QueueMessage(
            task_type=TaskType.CONTENT_VALIDATION,
            priority=TaskPriority.NORMAL,
            payload={
                "content": "Content to validate",
                "criteria": ["quality", "accuracy"]
            }
        )
        
        result = await worker.process_task(message)
        
        assert "content_length" in result
        assert "quality_score" in result
        assert "criteria_checked" in result
        assert result["criteria_checked"] == ["quality", "accuracy"]
    
    @pytest.mark.asyncio
    async def test_process_unsupported_task(self):
        """Test processing unsupported task type."""
        worker = GraphRAGWorker()
        
        message = QueueMessage(
            task_type=TaskType.NOTIFICATION_DELIVERY,
            priority=TaskPriority.LOW,
            payload={}
        )
        
        with pytest.raises(ValueError, match="Unsupported task type"):
            await worker.process_task(message)


class TestAgentWorker:
    """Test AgentWorker functionality."""
    
    def test_can_handle_agent_tasks(self):
        """Test agent worker task handling."""
        worker = AgentWorker()
        
        assert worker.can_handle(TaskType.AGENT_ORCHESTRATION) is True
        assert worker.can_handle(TaskType.PRD_GENERATION) is True
        assert worker.can_handle(TaskType.DOCUMENT_PROCESSING) is True
        assert worker.can_handle(TaskType.NOTIFICATION_DELIVERY) is False
    
    @pytest.mark.asyncio
    async def test_process_agent_orchestration(self):
        """Test agent orchestration processing."""
        worker = AgentWorker()
        
        message = QueueMessage(
            task_type=TaskType.AGENT_ORCHESTRATION,
            priority=TaskPriority.HIGH,
            payload={
                "task_description": "Complex multi-agent task",
                "agents": ["agent1", "agent2"],
                "context": {"project": "test-project"}
            }
        )
        
        result = await worker.process_task(message)
        
        assert "task_description" in result
        assert "agents_used" in result
        assert "execution_plan" in result
        assert result["task_description"] == "Complex multi-agent task"
    
    @pytest.mark.asyncio
    async def test_process_prd_generation(self):
        """Test PRD generation processing."""
        worker = AgentWorker()
        
        message = QueueMessage(
            task_type=TaskType.PRD_GENERATION,
            priority=TaskPriority.HIGH,
            payload={
                "prompt": "Create authentication section",
                "context": "Enterprise app",
                "section_type": "requirements"
            }
        )
        
        result = await worker.process_task(message)
        
        assert "generated_content" in result
        assert "quality_score" in result
        assert "section_type" in result
        assert result["section_type"] == "requirements"
    
    @pytest.mark.asyncio
    async def test_process_document_processing(self):
        """Test document processing."""
        worker = AgentWorker()
        
        message = QueueMessage(
            task_type=TaskType.DOCUMENT_PROCESSING,
            priority=TaskPriority.NORMAL,
            payload={
                "document_type": "technical_spec",
                "content": "Document content to process",
                "options": {"extract_metadata": True}
            }
        )
        
        result = await worker.process_task(message)
        
        assert "document_type" in result
        assert "processed_content" in result
        assert "metadata" in result
        assert result["document_type"] == "technical_spec"


class TestNotificationWorker:
    """Test NotificationWorker functionality."""
    
    def test_can_handle_notification_tasks(self):
        """Test notification worker task handling."""
        worker = NotificationWorker()
        
        assert worker.can_handle(TaskType.NOTIFICATION_DELIVERY) is True
        assert worker.can_handle(TaskType.GRAPHRAG_VALIDATION) is False
    
    @pytest.mark.asyncio
    async def test_process_email_notification(self):
        """Test email notification processing."""
        worker = NotificationWorker()
        
        message = QueueMessage(
            task_type=TaskType.NOTIFICATION_DELIVERY,
            priority=TaskPriority.LOW,
            payload={
                "type": "email",
                "recipient": "userexample.com",
                "subject": "Test Subject",
                "content": "Test email content"
            }
        )
        
        result = await worker.process_task(message)
        
        assert result["type"] == "email"
        assert result["recipient"] == "userexample.com"
        assert result["subject"] == "Test Subject"
        assert result["status"] == "delivered"
        assert "message_id" in result
    
    @pytest.mark.asyncio
    async def test_process_websocket_notification(self):
        """Test WebSocket notification processing."""
        worker = NotificationWorker()
        
        message = QueueMessage(
            task_type=TaskType.NOTIFICATION_DELIVERY,
            priority=TaskPriority.NORMAL,
            payload={
                "type": "websocket",
                "recipient": "user-123",
                "content": "Real-time update"
            }
        )
        
        result = await worker.process_task(message)
        
        assert result["type"] == "websocket"
        assert result["recipient"] == "user-123"
        assert result["status"] == "delivered"
        assert "message_id" in result
    
    @pytest.mark.asyncio
    async def test_process_generic_notification(self):
        """Test generic notification processing."""
        worker = NotificationWorker()
        
        message = QueueMessage(
            task_type=TaskType.NOTIFICATION_DELIVERY,
            priority=TaskPriority.LOW,
            payload={
                "type": "sms",
                "recipient": "+1234567890",
                "content": "SMS message"
            }
        )
        
        result = await worker.process_task(message)
        
        assert result["type"] == "sms"
        assert result["recipient"] == "+1234567890"
        assert result["status"] == "delivered"


class TestWorkerManager:
    """Test WorkerManager functionality."""
    
    def test_worker_manager_initialization(self):
        """Test worker manager initialization."""
        manager = WorkerManager()
        
        assert len(manager.workers) == 0
        assert manager.running is False
    
    def test_add_worker(self):
        """Test adding worker to manager."""
        manager = WorkerManager()
        worker = NotificationWorker()
        
        manager.add_worker(worker)
        
        assert len(manager.workers) == 1
        assert manager.workers[0] == worker
    
    def test_get_all_stats(self):
        """Test getting all worker statistics."""
        manager = WorkerManager()
        
        worker1 = NotificationWorker()
        worker1.processed_count = 5
        worker1.error_count = 1
        
        worker2 = GraphRAGWorker()
        worker2.processed_count = 10
        worker2.error_count = 0
        
        manager.add_worker(worker1)
        manager.add_worker(worker2)
        
        stats = manager.get_all_stats()
        
        assert len(stats) == 2
        assert stats[0]["processed_count"] == 5
        assert stats[1]["processed_count"] == 10
    
    @pytest.mark.asyncio
    async def test_start_stop_workers(self):
        """Test starting and stopping workers."""
        manager = WorkerManager()
        
        # Mock worker start/stop methods
        worker = Mock()
        worker.start = AsyncMock()
        worker.stop = AsyncMock()
        
        manager.add_worker(worker)
        
        # Test stopping (should not call anything if not running)
        await manager.stop_all()
        worker.stop.assert_not_called()
        
        # Test that running state is managed
        assert manager.running is False


class TestTaskWorkerIntegration:
    """Integration tests for task workers with queue."""
    
    @pytest.mark.asyncio
    async def test_worker_with_mock_queue(self):
        """Test worker integration with mock queue."""
        # Create mock queue
        mock_queue = AsyncMock()
        mock_queue.dequeue.return_value = None  # Empty queue
        
        # Create worker
        class TestWorker(TaskWorker):
            def can_handle(self, task_type: TaskType) -> bool:
                return True
            
            async def process_task(self, message: QueueMessage) -> dict:
                return {"processed": True}
        
        worker = TestWorker()
        worker.queue = mock_queue
        
        # Start worker for a short time
        worker.running = True
        
        # Since queue is empty, this should return quickly
        try:
            await asyncio.wait_for(worker.start(), timeout=0.1)
        except asyncio.TimeoutError:
            # Expected - worker runs indefinitely
            worker.running = False
        
        # Verify dequeue was called
        mock_queue.dequeue.assert_called()
    
    @pytest.mark.asyncio
    async def test_worker_processes_message(self):
        """Test worker processing a message."""
        # Create test message
        message = QueueMessage(
            task_type=TaskType.NOTIFICATION_DELIVERY,
            priority=TaskPriority.LOW,
            payload={"test": "data"}
        )
        
        # Mock queue that returns our message once
        mock_queue = AsyncMock()
        mock_queue.dequeue.side_effect = [message, None]  # Return message once, then None
        mock_queue.complete_task.return_value = True
        
        # Create worker
        class TestWorker(TaskWorker):
            def can_handle(self, task_type: TaskType) -> bool:
                return True
            
            async def process_task(self, message: QueueMessage) -> dict:
                return {"result": "success"}
        
        worker = TestWorker()
        worker.queue = mock_queue
        
        # Process one message cycle
        worker.running = True
        
        try:
            await asyncio.wait_for(worker.start(), timeout=0.5)
        except asyncio.TimeoutError:
            worker.running = False
        
        # Verify message was processed
        assert worker.processed_count == 1
        mock_queue.complete_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_worker_handles_processing_error(self):
        """Test worker handling processing errors."""
        # Create test message
        message = QueueMessage(
            task_type=TaskType.NOTIFICATION_DELIVERY,
            priority=TaskPriority.LOW,
            payload={"test": "data"}
        )
        
        # Mock queue
        mock_queue = AsyncMock()
        mock_queue.dequeue.side_effect = [message, None]
        mock_queue.fail_task.return_value = True
        
        # Worker that always fails
        class FailingWorker(TaskWorker):
            def can_handle(self, task_type: TaskType) -> bool:
                return True
            
            async def process_task(self, message: QueueMessage) -> dict:
                raise Exception("Processing failed")
        
        worker = FailingWorker()
        worker.queue = mock_queue
        
        # Process one message cycle
        worker.running = True
        
        try:
            await asyncio.wait_for(worker.start(), timeout=0.5)
        except asyncio.TimeoutError:
            worker.running = False
        
        # Verify error was handled
        assert worker.error_count == 1
        mock_queue.fail_task.assert_called_once()