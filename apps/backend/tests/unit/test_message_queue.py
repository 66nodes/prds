"""
Unit tests for message queue functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

from services.queue.message_queue import MessageQueue, QueueMessage, TaskResult
from services.queue.task_types import TaskType, TaskPriority, TaskStatus


class TestQueueMessage:
    """Test QueueMessage model."""
    
    def test_queue_message_creation(self):
        """Test creating queue message."""
        message = QueueMessage(
            task_type=TaskType.GRAPHRAG_VALIDATION,
            priority=TaskPriority.HIGH,
            payload={"content": "test"},
            user_id="user-123"
        )
        
        assert message.task_type == TaskType.GRAPHRAG_VALIDATION
        assert message.priority == TaskPriority.HIGH
        assert message.payload == {"content": "test"}
        assert message.user_id == "user-123"
        assert message.retry_count == 0
        assert message.id is not None
        assert message.created_at is not None
    
    def test_queue_message_defaults(self):
        """Test queue message default values."""
        message = QueueMessage(
            task_type=TaskType.NOTIFICATION_DELIVERY,
            priority=TaskPriority.LOW
        )
        
        assert message.payload == {}
        assert message.metadata == {}
        assert message.user_id is None
        assert message.max_retries == 3
        assert message.timeout_seconds == 300


class TestTaskResult:
    """Test TaskResult model."""
    
    def test_task_result_creation(self):
        """Test creating task result."""
        started_at = datetime.now(timezone.utc)
        completed_at = datetime.now(timezone.utc)
        
        result = TaskResult(
            task_id="task-123",
            status=TaskStatus.COMPLETED,
            result={"success": True},
            started_at=started_at,
            completed_at=completed_at,
            processing_time_ms=1500,
            worker_id="worker-1"
        )
        
        assert result.task_id == "task-123"
        assert result.status == TaskStatus.COMPLETED
        assert result.result == {"success": True}
        assert result.processing_time_ms == 1500
        assert result.worker_id == "worker-1"


class TestMessageQueue:
    """Test MessageQueue functionality."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        mock = AsyncMock()
        mock.zadd = AsyncMock(return_value=1)
        mock.bzpopmin = AsyncMock(return_value=None)
        mock.set = AsyncMock(return_value=True)
        mock.get = AsyncMock(return_value=None)
        mock.delete = AsyncMock(return_value=1)
        mock.zcard = AsyncMock(return_value=0)
        mock.keys = AsyncMock(return_value=[])
        mock.hgetall = AsyncMock(return_value={})
        mock.hincrby = AsyncMock(return_value=1)
        return mock
    
    @pytest.fixture
    def message_queue(self, mock_redis):
        """Create message queue with mock Redis."""
        return MessageQueue(redis_client=mock_redis)
    
    @pytest.mark.asyncio
    async def test_enqueue_task(self, message_queue, mock_redis):
        """Test enqueueing a task."""
        task_id = await message_queue.enqueue(
            task_type=TaskType.GRAPHRAG_VALIDATION,
            payload={"content": "test content"},
            priority=TaskPriority.HIGH,
            user_id="user-123"
        )
        
        assert task_id is not None
        mock_redis.zadd.assert_called_once()
        mock_redis.hincrby.assert_called()
    
    @pytest.mark.asyncio
    async def test_enqueue_with_delay(self, message_queue, mock_redis):
        """Test enqueueing with delay."""
        task_id = await message_queue.enqueue(
            task_type=TaskType.NOTIFICATION_DELIVERY,
            payload={"message": "delayed"},
            delay_seconds=300
        )
        
        assert task_id is not None
        mock_redis.zadd.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_dequeue_empty_queue(self, message_queue, mock_redis):
        """Test dequeueing from empty queue."""
        mock_redis.bzpopmin.return_value = None
        
        message = await message_queue.dequeue("worker-1", timeout=1)
        assert message is None
    
    @pytest.mark.asyncio
    async def test_dequeue_with_message(self, message_queue, mock_redis):
        """Test successful dequeue."""
        # Mock message data
        test_message = QueueMessage(
            task_type=TaskType.GRAPHRAG_VALIDATION,
            priority=TaskPriority.HIGH,
            payload={"content": "test"}
        )
        
        mock_redis.bzpopmin.return_value = (
            b"queue:priority:8",
            test_message.json().encode(),
            1000.0
        )
        
        message = await message_queue.dequeue("worker-1", timeout=5)
        
        assert message is not None
        assert message.task_type == TaskType.GRAPHRAG_VALIDATION
        assert message.priority == TaskPriority.HIGH
        mock_redis.set.assert_called_once()  # Processing marker
    
    @pytest.mark.asyncio
    async def test_complete_task_success(self, message_queue, mock_redis):
        """Test completing a task successfully."""
        # Mock processing data
        test_message = QueueMessage(
            task_type=TaskType.GRAPHRAG_VALIDATION,
            priority=TaskPriority.HIGH
        )
        
        processing_data = {
            "message": test_message.json(),
            "worker_id": "worker-1",
            "started_at": datetime.now(timezone.utc).isoformat()
        }
        
        mock_redis.get.return_value = str(processing_data).encode()
        
        result = await message_queue.complete_task(
            task_id="test-task-id",
            result={"success": True},
            worker_id="worker-1"
        )
        
        assert result is True
        mock_redis.set.assert_called()  # Result storage
        mock_redis.delete.assert_called()  # Remove processing marker
    
    @pytest.mark.asyncio
    async def test_complete_task_not_found(self, message_queue, mock_redis):
        """Test completing non-existent task."""
        mock_redis.get.return_value = None
        
        result = await message_queue.complete_task("non-existent-task")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_fail_task_with_retry(self, message_queue, mock_redis):
        """Test failing task with retry."""
        test_message = QueueMessage(
            task_type=TaskType.GRAPHRAG_VALIDATION,
            priority=TaskPriority.HIGH,
            retry_count=0,
            max_retries=3
        )
        
        processing_data = {
            "message": test_message.json(),
            "worker_id": "worker-1",
            "started_at": datetime.now(timezone.utc).isoformat()
        }
        
        mock_redis.get.return_value = str(processing_data).encode()
        
        result = await message_queue.fail_task(
            task_id="test-task-id",
            error="Processing failed",
            worker_id="worker-1",
            retry=True
        )
        
        assert result is True
        mock_redis.zadd.assert_called()  # Re-enqueue for retry
    
    @pytest.mark.asyncio
    async def test_fail_task_max_retries_exceeded(self, message_queue, mock_redis):
        """Test failing task when max retries exceeded."""
        test_message = QueueMessage(
            task_type=TaskType.GRAPHRAG_VALIDATION,
            priority=TaskPriority.HIGH,
            retry_count=3,
            max_retries=3
        )
        
        processing_data = {
            "message": test_message.json(),
            "worker_id": "worker-1",
            "started_at": datetime.now(timezone.utc).isoformat()
        }
        
        mock_redis.get.return_value = str(processing_data).encode()
        
        result = await message_queue.fail_task(
            task_id="test-task-id",
            error="Final failure",
            worker_id="worker-1",
            retry=True
        )
        
        assert result is True
        # Should create failure result, not retry
        mock_redis.set.assert_called()  # Result storage
    
    @pytest.mark.asyncio
    async def test_get_task_result_found(self, message_queue, mock_redis):
        """Test getting existing task result."""
        result_data = TaskResult(
            task_id="test-task",
            status=TaskStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            processing_time_ms=1000
        )
        
        mock_redis.get.return_value = result_data.json().encode()
        
        result = await message_queue.get_task_result("test-task")
        
        assert result is not None
        assert result.task_id == "test-task"
        assert result.status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_get_task_result_not_found(self, message_queue, mock_redis):
        """Test getting non-existent task result."""
        mock_redis.get.return_value = None
        
        result = await message_queue.get_task_result("non-existent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_queue_stats(self, message_queue, mock_redis):
        """Test getting queue statistics."""
        # Mock queue lengths
        mock_redis.zcard.return_value = 5
        mock_redis.keys.return_value = [b"processing:task1", b"processing:task2"]
        mock_redis.hgetall.return_value = {
            b"enqueued": b"100",
            b"dequeued": b"95",
            b"completed": b"90"
        }
        
        stats = await message_queue.get_queue_stats()
        
        assert "queue_priority_1" in stats
        assert "queue_priority_5" in stats
        assert "queue_priority_8" in stats
        assert "queue_priority_10" in stats
        assert stats["processing"] == 2
        assert stats["enqueued"] == 100
        assert stats["completed"] == 90
    
    @pytest.mark.asyncio
    async def test_purge_queue_specific_priority(self, message_queue, mock_redis):
        """Test purging specific priority queue."""
        mock_redis.delete.return_value = 5
        
        removed = await message_queue.purge_queue(TaskPriority.HIGH)
        
        assert removed == 5
        mock_redis.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_purge_all_queues(self, message_queue, mock_redis):
        """Test purging all queues."""
        mock_redis.delete.return_value = 2
        
        removed = await message_queue.purge_queue()
        
        # Should call delete for each priority level
        assert mock_redis.delete.call_count == len(TaskPriority)
        assert removed == 2 * len(TaskPriority)
    
    def test_calculate_score(self, message_queue):
        """Test priority score calculation."""
        message = QueueMessage(
            task_type=TaskType.GRAPHRAG_VALIDATION,
            priority=TaskPriority.HIGH
        )
        
        score = message_queue._calculate_score(message)
        
        # Higher priority should result in lower score (higher in sorted set)
        assert isinstance(score, float)
        assert score > 0
    
    def test_get_queue_key(self, message_queue):
        """Test queue key generation."""
        key = message_queue._get_queue_key(TaskPriority.HIGH)
        assert key == "queue:priority:8"
    
    def test_get_result_key(self, message_queue):
        """Test result key generation."""
        key = message_queue._get_result_key("task-123")
        assert key == "result:task-123"
    
    def test_get_processing_key(self, message_queue):
        """Test processing key generation."""
        key = message_queue._get_processing_key("task-123")
        assert key == "processing:task-123"