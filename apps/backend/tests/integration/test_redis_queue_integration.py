"""
Integration tests for Redis message queue system.
"""

import pytest
import asyncio
from unittest.mock import patch, Mock, AsyncMock
from datetime import datetime

from services.queue.message_queue import MessageQueue, QueueMessage, get_message_queue
from services.queue.task_types import TaskType, TaskPriority, TaskStatus
from services.queue.workers import GraphRAGWorker, AgentWorker, NotificationWorker
from core.redis import RedisConnectionManager, RedisCache


class TestRedisConnectionManager:
    """Test Redis connection manager functionality."""
    
    @pytest.fixture
    def mock_redis_pool(self):
        """Mock Redis connection pool."""
        with patch('redis.asyncio.ConnectionPool') as mock_pool_class:
            mock_pool = AsyncMock()
            mock_pool_class.from_url.return_value = mock_pool
            yield mock_pool
    
    @pytest.fixture
    def mock_redis_client(self):
        """Mock Redis client."""
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_client = AsyncMock()
            mock_client.ping.return_value = True
            mock_client.close.return_value = None
            mock_client.info.return_value = {
                "redis_version": "7.0.0",
                "connected_clients": 5,
                "used_memory_human": "1.2M",
                "uptime_in_seconds": 3600
            }
            mock_redis_class.return_value = mock_client
            yield mock_client
    
    @pytest.mark.asyncio
    async def test_redis_manager_initialization(self, mock_redis_pool, mock_redis_client):
        """Test Redis connection manager initialization."""
        manager = RedisConnectionManager()
        
        await manager.initialize()
        
        assert manager._redis_client is not None
        mock_redis_client.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_redis_manager_health_check(self, mock_redis_pool, mock_redis_client):
        """Test Redis health check."""
        manager = RedisConnectionManager()
        await manager.initialize()
        
        health = await manager.health_check()
        
        assert health["status"] == "healthy"
        assert "latency_ms" in health
        assert "redis_version" in health
        assert health["redis_version"] == "7.0.0"
    
    @pytest.mark.asyncio
    async def test_redis_manager_get_stats(self, mock_redis_pool, mock_redis_client):
        """Test Redis performance statistics."""
        # Mock additional stats
        mock_redis_client.info.side_effect = [
            {
                "total_connections_received": 100,
                "connected_clients": 5,
                "used_memory": 1048576,
                "used_memory_human": "1M",
                "used_memory_peak": 2097152,
                "used_memory_peak_human": "2M"
            },
            {
                "total_commands_processed": 1000,
                "instantaneous_ops_per_sec": 10,
                "keyspace_hits": 800,
                "keyspace_misses": 200
            }
        ]
        
        manager = RedisConnectionManager()
        await manager.initialize()
        
        stats = await manager.get_stats()
        
        assert "connections" in stats
        assert "memory" in stats
        assert "operations" in stats
        assert stats["connections"]["current"] == 5
        assert stats["memory"]["used_human"] == "1M"
        assert stats["operations"]["commands_processed"] == 1000
    
    @pytest.mark.asyncio
    async def test_redis_manager_close(self, mock_redis_pool, mock_redis_client):
        """Test Redis connection cleanup."""
        manager = RedisConnectionManager()
        await manager.initialize()
        
        await manager.close()
        
        mock_redis_client.close.assert_called_once()


class TestRedisCache:
    """Test Redis cache functionality."""
    
    @pytest.fixture
    def mock_connection_manager(self):
        """Mock Redis connection manager."""
        manager = Mock()
        client = AsyncMock()
        manager.client = client
        return manager, client
    
    @pytest.mark.asyncio
    async def test_cache_get_set(self, mock_connection_manager):
        """Test cache get and set operations."""
        manager, mock_client = mock_connection_manager
        cache = RedisCache(manager)
        
        # Test set
        mock_client.set.return_value = True
        result = await cache.set("test_key", "test_value", ttl=300)
        assert result is True
        mock_client.set.assert_called_with("test_key", "test_value", ex=300)
        
        # Test get
        mock_client.get.return_value = "test_value"
        value = await cache.get("test_key")
        assert value == "test_value"
        mock_client.get.assert_called_with("test_key")
    
    @pytest.mark.asyncio
    async def test_cache_delete_exists(self, mock_connection_manager):
        """Test cache delete and exists operations."""
        manager, mock_client = mock_connection_manager
        cache = RedisCache(manager)
        
        # Test delete
        mock_client.delete.return_value = 1
        result = await cache.delete("test_key")
        assert result is True
        
        # Test exists
        mock_client.exists.return_value = 1
        exists = await cache.exists("test_key")
        assert exists is True
    
    @pytest.mark.asyncio
    async def test_cache_get_many_set_many(self, mock_connection_manager):
        """Test bulk cache operations."""
        manager, mock_client = mock_connection_manager
        cache = RedisCache(manager)
        
        # Test get_many
        mock_client.mget.return_value = ["value1", "value2", None]
        result = await cache.get_many(["key1", "key2", "key3"])
        assert result == {"key1": "value1", "key2": "value2", "key3": None}
        
        # Test set_many
        mock_pipeline = AsyncMock()
        mock_pipeline.execute.return_value = [True, True]
        mock_client.pipeline.return_value = mock_pipeline
        
        result = await cache.set_many({"key1": "value1", "key2": "value2"})
        assert result is True
    
    @pytest.mark.asyncio
    async def test_cache_clear_pattern(self, mock_connection_manager):
        """Test clearing cache by pattern."""
        manager, mock_client = mock_connection_manager
        cache = RedisCache(manager)
        
        # Mock scan_iter to return some keys
        async def mock_scan_iter(match=None):
            yield b"test:key1"
            yield b"test:key2"
        
        mock_client.scan_iter = mock_scan_iter
        mock_client.delete.return_value = 2
        
        result = await cache.clear_pattern("test:*")
        assert result == 2
        mock_client.delete.assert_called_with(b"test:key1", b"test:key2")


class TestMessageQueueIntegration:
    """Integration tests for message queue with Redis."""
    
    @pytest.fixture
    def mock_redis_queue(self):
        """Mock Redis client for queue operations."""
        mock_client = AsyncMock()
        mock_client.zadd.return_value = 1
        mock_client.bzpopmin.return_value = None
        mock_client.set.return_value = True
        mock_client.get.return_value = None
        mock_client.delete.return_value = 1
        mock_client.zcard.return_value = 0
        mock_client.keys.return_value = []
        mock_client.hgetall.return_value = {}
        mock_client.hincrby.return_value = 1
        return mock_client
    
    @pytest.mark.asyncio
    async def test_queue_enqueue_dequeue_workflow(self, mock_redis_queue):
        """Test complete enqueue-dequeue workflow."""
        queue = MessageQueue(mock_redis_queue)
        
        # Enqueue task
        task_id = await queue.enqueue(
            task_type=TaskType.GRAPHRAG_VALIDATION,
            payload={"content": "test content"},
            priority=TaskPriority.HIGH,
            user_id="user-123"
        )
        
        assert task_id is not None
        mock_redis_queue.zadd.assert_called_once()
        
        # Mock dequeue returning the enqueued message
        test_message = QueueMessage(
            id=task_id,
            task_type=TaskType.GRAPHRAG_VALIDATION,
            priority=TaskPriority.HIGH,
            payload={"content": "test content"},
            user_id="user-123"
        )
        
        mock_redis_queue.bzpopmin.return_value = (
            b"queue:priority:8",
            test_message.json().encode(),
            1000.0
        )
        
        # Dequeue task
        dequeued_message = await queue.dequeue("worker-1")
        
        assert dequeued_message is not None
        assert dequeued_message.task_type == TaskType.GRAPHRAG_VALIDATION
        assert dequeued_message.user_id == "user-123"
        
        # Verify processing marker was set
        mock_redis_queue.set.assert_called()
    
    @pytest.mark.asyncio
    async def test_queue_task_completion_workflow(self, mock_redis_queue):
        """Test task completion workflow."""
        queue = MessageQueue(mock_redis_queue)
        
        # Mock processing data
        test_message = QueueMessage(
            task_type=TaskType.GRAPHRAG_VALIDATION,
            priority=TaskPriority.HIGH
        )
        
        processing_data = {
            "message": test_message.json(),
            "worker_id": "worker-1",
            "started_at": datetime.utcnow().isoformat()
        }
        
        import json
        mock_redis_queue.get.return_value = json.dumps(processing_data)
        
        # Complete task
        result = await queue.complete_task(
            task_id="test-task",
            result={"validation_score": 0.95},
            worker_id="worker-1"
        )
        
        assert result is True
        
        # Verify result was stored and processing marker removed
        assert mock_redis_queue.set.call_count >= 1  # Result storage
        mock_redis_queue.delete.assert_called()  # Processing marker removal
    
    @pytest.mark.asyncio
    async def test_queue_task_failure_with_retry(self, mock_redis_queue):
        """Test task failure and retry workflow."""
        queue = MessageQueue(mock_redis_queue)
        
        # Mock processing data for retry scenario
        test_message = QueueMessage(
            task_type=TaskType.GRAPHRAG_VALIDATION,
            priority=TaskPriority.HIGH,
            retry_count=0,
            max_retries=3
        )
        
        processing_data = {
            "message": test_message.json(),
            "worker_id": "worker-1",
            "started_at": datetime.utcnow().isoformat()
        }
        
        import json
        mock_redis_queue.get.return_value = json.dumps(processing_data)
        
        # Fail task with retry
        result = await queue.fail_task(
            task_id="test-task",
            error="Temporary failure",
            worker_id="worker-1",
            retry=True
        )
        
        assert result is True
        
        # Verify task was re-enqueued for retry
        mock_redis_queue.zadd.assert_called()  # Re-enqueue
        mock_redis_queue.delete.assert_called()  # Remove processing marker
    
    @pytest.mark.asyncio
    async def test_queue_statistics_collection(self, mock_redis_queue):
        """Test queue statistics collection."""
        queue = MessageQueue(mock_redis_queue)
        
        # Mock statistics data
        mock_redis_queue.zcard.return_value = 5
        mock_redis_queue.keys.return_value = [b"processing:task1", b"processing:task2"]
        mock_redis_queue.hgetall.return_value = {
            b"enqueued": b"100",
            b"completed": b"95",
            b"failed": b"3"
        }
        
        stats = await queue.get_queue_stats()
        
        assert stats["processing"] == 2
        assert stats["enqueued"] == 100
        assert stats["completed"] == 95
        assert stats["failed"] == 3
        
        # Check all priority queues are included
        for priority in [1, 5, 8, 10]:
            assert f"queue_priority_{priority}" in stats
    
    @pytest.mark.asyncio
    async def test_queue_purge_operations(self, mock_redis_queue):
        """Test queue purge operations."""
        queue = MessageQueue(mock_redis_queue)
        
        # Test purging specific priority
        mock_redis_queue.delete.return_value = 3
        
        removed = await queue.purge_queue(TaskPriority.HIGH)
        assert removed == 3
        
        # Test purging all queues
        mock_redis_queue.delete.return_value = 2
        
        removed = await queue.purge_queue()
        # Should call delete for each priority level (4 total)
        assert mock_redis_queue.delete.call_count >= 4


class TestWorkerQueueIntegration:
    """Integration tests for workers with message queue."""
    
    @pytest.mark.asyncio
    async def test_graphrag_worker_integration(self):
        """Test GraphRAG worker with message queue."""
        # Mock queue that provides a GraphRAG task
        mock_queue = AsyncMock()
        
        message = QueueMessage(
            task_type=TaskType.GRAPHRAG_VALIDATION,
            priority=TaskPriority.HIGH,
            payload={
                "content": "Test content",
                "project_id": "project-123"
            }
        )
        
        # Return message once, then None to stop worker
        mock_queue.dequeue.side_effect = [message, None]
        mock_queue.complete_task.return_value = True
        
        worker = GraphRAGWorker()
        worker.queue = mock_queue
        
        # Process one cycle
        worker.running = True
        
        try:
            await asyncio.wait_for(worker.start(), timeout=1.0)
        except asyncio.TimeoutError:
            worker.running = False
        
        # Verify task was processed successfully
        assert worker.processed_count == 1
        assert worker.error_count == 0
        mock_queue.complete_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_notification_worker_integration(self):
        """Test notification worker with message queue."""
        mock_queue = AsyncMock()
        
        message = QueueMessage(
            task_type=TaskType.NOTIFICATION_DELIVERY,
            priority=TaskPriority.LOW,
            payload={
                "type": "email",
                "recipient": "user@company.local",
                "content": "Test notification"
            }
        )
        
        mock_queue.dequeue.side_effect = [message, None]
        mock_queue.complete_task.return_value = True
        
        worker = NotificationWorker()
        worker.queue = mock_queue
        
        # Process one cycle
        worker.running = True
        
        try:
            await asyncio.wait_for(worker.start(), timeout=1.0)
        except asyncio.TimeoutError:
            worker.running = False
        
        # Verify notification was processed
        assert worker.processed_count == 1
        mock_queue.complete_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_worker_handles_unsupported_task(self):
        """Test worker handling unsupported task types."""
        mock_queue = AsyncMock()
        
        # GraphRAG worker receiving notification task
        message = QueueMessage(
            task_type=TaskType.NOTIFICATION_DELIVERY,
            priority=TaskPriority.LOW,
            payload={"test": "data"}
        )
        
        mock_queue.dequeue.side_effect = [message, None]
        mock_queue.fail_task.return_value = True
        
        worker = GraphRAGWorker()  # Can't handle notifications
        worker.queue = mock_queue
        
        # Process one cycle
        worker.running = True
        
        try:
            await asyncio.wait_for(worker.start(), timeout=1.0)
        except asyncio.TimeoutError:
            worker.running = False
        
        # Task should be failed and retried
        mock_queue.fail_task.assert_called_once()
        assert "cannot handle task type" in mock_queue.fail_task.call_args[0][1]
    
    @pytest.mark.asyncio
    async def test_multiple_workers_processing(self):
        """Test multiple workers processing different task types."""
        mock_queue = AsyncMock()
        
        # Create different task types
        graphrag_message = QueueMessage(
            task_type=TaskType.GRAPHRAG_VALIDATION,
            priority=TaskPriority.HIGH,
            payload={"content": "test"}
        )
        
        notification_message = QueueMessage(
            task_type=TaskType.NOTIFICATION_DELIVERY,
            priority=TaskPriority.LOW,
            payload={"type": "email", "recipient": "user@company.local"}
        )
        
        # Each worker gets one message, then None
        mock_queue.dequeue.side_effect = [
            graphrag_message, None,  # For GraphRAG worker
            notification_message, None  # For notification worker
        ]
        mock_queue.complete_task.return_value = True
        
        # Create workers
        graphrag_worker = GraphRAGWorker()
        graphrag_worker.queue = mock_queue
        
        notification_worker = NotificationWorker()
        notification_worker.queue = mock_queue
        
        # Start both workers
        graphrag_worker.running = True
        notification_worker.running = True
        
        # Run workers concurrently for short time
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    graphrag_worker.start(),
                    notification_worker.start(),
                    return_exceptions=True
                ),
                timeout=1.0
            )
        except asyncio.TimeoutError:
            graphrag_worker.running = False
            notification_worker.running = False
        
        # Both workers should have processed their tasks
        total_processed = graphrag_worker.processed_count + notification_worker.processed_count
        assert total_processed >= 1  # At least one task processed
        
        # Complete task should be called for successful processing
        assert mock_queue.complete_task.called