"""
Redis-based message queue service for async task processing.
"""

import json
import uuid
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable, AsyncGenerator
from dataclasses import dataclass, asdict

from redis.asyncio import Redis
from redis.exceptions import RedisError
from pydantic import BaseModel, Field

from core.redis import get_redis_client
from core.config import get_settings
from .task_types import TaskType, TaskPriority, TaskStatus, get_task_config

logger = logging.getLogger(__name__)
settings = get_settings()


class QueueMessage(BaseModel):
    """Message structure for queue operations."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: TaskType
    priority: TaskPriority
    payload: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    max_retries: int = Field(default=3)
    retry_count: int = Field(default=0)
    timeout_seconds: int = Field(default=300)
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class TaskResult(BaseModel):
    """Task execution result."""
    
    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: datetime
    completed_at: datetime
    processing_time_ms: int
    retry_count: int = 0
    worker_id: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class MessageQueue:
    """Redis-based message queue with priority support."""
    
    def __init__(self, redis_client: Optional[Redis] = None):
        self.redis = redis_client or get_redis_client()
        self.queue_prefix = "queue"
        self.result_prefix = "result"
        self.processing_prefix = "processing"
        self.stats_key = "queue:stats"
        self._listeners: Dict[TaskType, List[Callable]] = {}
        
    def _get_queue_key(self, priority: TaskPriority) -> str:
        """Get Redis key for priority queue."""
        return f"{self.queue_prefix}:priority:{priority.value}"
    
    def _get_result_key(self, task_id: str) -> str:
        """Get Redis key for task result."""
        return f"{self.result_prefix}:{task_id}"
    
    def _get_processing_key(self, task_id: str) -> str:
        """Get Redis key for processing task."""
        return f"{self.processing_prefix}:{task_id}"
    
    async def enqueue(
        self,
        task_type: TaskType,
        payload: Dict[str, Any],
        priority: Optional[TaskPriority] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        delay_seconds: int = 0
    ) -> str:
        """Add task to queue."""
        try:
            # Get task configuration
            task_config = get_task_config(task_type)
            
            # Create message
            message = QueueMessage(
                task_type=task_type,
                priority=priority or task_config["default_priority"],
                payload=payload,
                user_id=user_id,
                metadata=metadata or {},
                max_retries=task_config["max_retries"],
                timeout_seconds=task_config["timeout_seconds"]
            )
            
            # Handle delayed execution
            if delay_seconds > 0:
                message.scheduled_at = datetime.now(timezone.utc).replace(
                    microsecond=0
                ) + asyncio.get_event_loop().time() + delay_seconds
            
            # Serialize message
            message_data = message.json()
            
            # Add to priority queue
            queue_key = self._get_queue_key(message.priority)
            score = self._calculate_score(message)
            
            await self.redis.zadd(queue_key, {message_data: score})
            
            # Update stats
            await self._update_stats("enqueued", task_type)
            
            logger.info(
                f"Enqueued task {message.id} of type {task_type} "
                f"with priority {message.priority}"
            )
            
            return message.id
            
        except RedisError as e:
            logger.error(f"Failed to enqueue task: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error enqueueing task: {e}")
            raise
    
    async def dequeue(
        self, 
        worker_id: str,
        timeout: int = 5
    ) -> Optional[QueueMessage]:
        """Get next task from queue with priority ordering."""
        try:
            # Try each priority level from highest to lowest
            for priority in sorted(TaskPriority, key=lambda x: x.value, reverse=True):
                queue_key = self._get_queue_key(priority)
                
                # Get highest priority message
                result = await self.redis.bzpopmin(queue_key, timeout=timeout)
                
                if result:
                    _, message_data, _ = result
                    message = QueueMessage.parse_raw(message_data)
                    
                    # Check if scheduled task is ready
                    if message.scheduled_at and message.scheduled_at > datetime.now(timezone.utc):
                        # Put back in queue for later
                        score = self._calculate_score(message)
                        await self.redis.zadd(queue_key, {message_data: score})
                        continue
                    
                    # Mark as processing
                    processing_key = self._get_processing_key(message.id)
                    processing_data = {
                        "message": message_data,
                        "worker_id": worker_id,
                        "started_at": datetime.now(timezone.utc).isoformat()
                    }
                    
                    await self.redis.set(
                        processing_key,
                        json.dumps(processing_data),
                        ex=message.timeout_seconds
                    )
                    
                    # Update stats
                    await self._update_stats("dequeued", message.task_type)
                    
                    logger.info(
                        f"Dequeued task {message.id} of type {message.task_type} "
                        f"by worker {worker_id}"
                    )
                    
                    return message
            
            return None
            
        except RedisError as e:
            logger.error(f"Failed to dequeue task: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error dequeuing task: {e}")
            return None
    
    async def complete_task(
        self,
        task_id: str,
        result: Optional[Dict[str, Any]] = None,
        worker_id: Optional[str] = None
    ) -> bool:
        """Mark task as completed."""
        try:
            # Get processing info
            processing_key = self._get_processing_key(task_id)
            processing_data = await self.redis.get(processing_key)
            
            if not processing_data:
                logger.warning(f"Task {task_id} not found in processing")
                return False
            
            processing_info = json.loads(processing_data)
            message = QueueMessage.parse_raw(processing_info["message"])
            started_at = datetime.fromisoformat(processing_info["started_at"])
            completed_at = datetime.now(timezone.utc)
            
            # Create result
            task_result = TaskResult(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                started_at=started_at,
                completed_at=completed_at,
                processing_time_ms=int((completed_at - started_at).total_seconds() * 1000),
                retry_count=message.retry_count,
                worker_id=worker_id
            )
            
            # Store result
            result_key = self._get_result_key(task_id)
            await self.redis.set(
                result_key,
                task_result.json(),
                ex=settings.cache_ttl
            )
            
            # Remove from processing
            await self.redis.delete(processing_key)
            
            # Update stats
            await self._update_stats("completed", message.task_type)
            
            logger.info(f"Completed task {task_id} in {task_result.processing_time_ms}ms")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete task {task_id}: {e}")
            return False
    
    async def fail_task(
        self,
        task_id: str,
        error: str,
        worker_id: Optional[str] = None,
        retry: bool = True
    ) -> bool:
        """Mark task as failed and optionally retry."""
        try:
            # Get processing info
            processing_key = self._get_processing_key(task_id)
            processing_data = await self.redis.get(processing_key)
            
            if not processing_data:
                logger.warning(f"Task {task_id} not found in processing")
                return False
            
            processing_info = json.loads(processing_data)
            message = QueueMessage.parse_raw(processing_info["message"])
            started_at = datetime.fromisoformat(processing_info["started_at"])
            completed_at = datetime.now(timezone.utc)
            
            # Check if we should retry
            should_retry = (
                retry and 
                message.retry_count < message.max_retries
            )
            
            if should_retry:
                # Increment retry count and re-enqueue
                message.retry_count += 1
                message.metadata["last_error"] = error
                message.metadata["failed_at"] = completed_at.isoformat()
                
                queue_key = self._get_queue_key(message.priority)
                score = self._calculate_score(message)
                await self.redis.zadd(queue_key, {message.json(): score})
                
                # Update stats
                await self._update_stats("retried", message.task_type)
                
                logger.info(
                    f"Retrying task {task_id} (attempt {message.retry_count}/{message.max_retries})"
                )
                
            else:
                # Create failure result
                task_result = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error=error,
                    started_at=started_at,
                    completed_at=completed_at,
                    processing_time_ms=int((completed_at - started_at).total_seconds() * 1000),
                    retry_count=message.retry_count,
                    worker_id=worker_id
                )
                
                # Store result
                result_key = self._get_result_key(task_id)
                await self.redis.set(
                    result_key,
                    task_result.json(),
                    ex=settings.cache_ttl
                )
                
                # Update stats
                await self._update_stats("failed", message.task_type)
                
                logger.error(f"Task {task_id} failed permanently: {error}")
            
            # Remove from processing
            await self.redis.delete(processing_key)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to fail task {task_id}: {e}")
            return False
    
    async def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get task execution result."""
        try:
            result_key = self._get_result_key(task_id)
            result_data = await self.redis.get(result_key)
            
            if result_data:
                return TaskResult.parse_raw(result_data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get result for task {task_id}: {e}")
            return None
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        try:
            stats = {}
            
            # Get queue lengths by priority
            for priority in TaskPriority:
                queue_key = self._get_queue_key(priority)
                length = await self.redis.zcard(queue_key)
                stats[f"queue_priority_{priority.value}"] = length
            
            # Get processing count
            processing_keys = await self.redis.keys(f"{self.processing_prefix}:*")
            stats["processing"] = len(processing_keys)
            
            # Get operational stats
            operational_stats = await self.redis.hgetall(self.stats_key)
            for key, value in operational_stats.items():
                stats[key.decode()] = int(value)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {}
    
    async def purge_queue(self, priority: Optional[TaskPriority] = None) -> int:
        """Purge queue(s). Use with caution."""
        try:
            removed_count = 0
            
            if priority:
                queue_key = self._get_queue_key(priority)
                removed_count = await self.redis.delete(queue_key)
            else:
                # Purge all priority queues
                for p in TaskPriority:
                    queue_key = self._get_queue_key(p)
                    removed_count += await self.redis.delete(queue_key)
            
            logger.warning(f"Purged {removed_count} tasks from queue")
            return removed_count
            
        except Exception as e:
            logger.error(f"Failed to purge queue: {e}")
            return 0
    
    def _calculate_score(self, message: QueueMessage) -> float:
        """Calculate priority score for message ordering."""
        # Higher priority and older messages get lower scores (higher priority in sorted set)
        priority_score = 1000 - message.priority.value * 100
        timestamp_score = message.created_at.timestamp()
        return priority_score + timestamp_score
    
    async def _update_stats(self, operation: str, task_type: TaskType) -> None:
        """Update operational statistics."""
        try:
            await self.redis.hincrby(self.stats_key, operation, 1)
            await self.redis.hincrby(self.stats_key, f"{operation}_{task_type}", 1)
        except Exception as e:
            logger.error(f"Failed to update stats: {e}")


# Global message queue instance
_message_queue: Optional[MessageQueue] = None


def get_message_queue() -> MessageQueue:
    """Get global message queue instance."""
    global _message_queue
    
    if not _message_queue:
        _message_queue = MessageQueue()
    
    return _message_queue