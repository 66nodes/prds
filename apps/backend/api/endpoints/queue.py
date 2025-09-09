"""
API endpoints for message queue management and monitoring.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Body
from fastapi import status
from pydantic import BaseModel, Field

from services.queue import (
    MessageQueue, QueueMessage, TaskResult, TaskType, TaskPriority, 
    get_message_queue, get_worker_manager
)
from models.user import User
from api.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models
class EnqueueTaskRequest(BaseModel):
    """Request model for enqueueing tasks."""
    
    task_type: TaskType
    payload: Dict[str, Any]
    priority: Optional[TaskPriority] = None
    metadata: Optional[Dict[str, Any]] = None
    delay_seconds: int = Field(default=0, ge=0, le=3600)


class EnqueueTaskResponse(BaseModel):
    """Response model for enqueued tasks."""
    
    task_id: str
    message: str
    queued_at: datetime


class QueueStatsResponse(BaseModel):
    """Response model for queue statistics."""
    
    stats: Dict[str, Any]
    timestamp: datetime


class WorkerStatsResponse(BaseModel):
    """Response model for worker statistics."""
    
    workers: List[Dict[str, Any]]
    total_workers: int
    active_workers: int


class TaskResultResponse(BaseModel):
    """Response model for task results."""
    
    task_id: str
    result: Optional[TaskResult]
    found: bool


# Queue Management Endpoints
@router.post(
    "/tasks/enqueue",
    response_model=EnqueueTaskResponse,
    summary="Enqueue Task",
    description="Add a new task to the message queue for async processing"
)
async def enqueue_task(
    request: EnqueueTaskRequest,
    current_user: User = Depends(get_current_user)
) -> EnqueueTaskResponse:
    """Enqueue a new task for processing."""
    try:
        queue = get_message_queue()
        
        task_id = await queue.enqueue(
            task_type=request.task_type,
            payload=request.payload,
            priority=request.priority,
            user_id=current_user.id,
            metadata=request.metadata,
            delay_seconds=request.delay_seconds
        )
        
        logger.info(
            f"User {current_user.id} enqueued task {task_id} "
            f"of type {request.task_type}"
        )
        
        return EnqueueTaskResponse(
            task_id=task_id,
            message="Task enqueued successfully",
            queued_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to enqueue task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enqueue task: {str(e)}"
        )


@router.get(
    "/tasks/{task_id}/result",
    response_model=TaskResultResponse,
    summary="Get Task Result",
    description="Retrieve the result of a completed task"
)
async def get_task_result(
    task_id: str,
    current_user: User = Depends(get_current_user)
) -> TaskResultResponse:
    """Get the result of a completed task."""
    try:
        queue = get_message_queue()
        result = await queue.get_task_result(task_id)
        
        return TaskResultResponse(
            task_id=task_id,
            result=result,
            found=result is not None
        )
        
    except Exception as e:
        logger.error(f"Failed to get task result for {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task result: {str(e)}"
        )


@router.get(
    "/stats",
    response_model=QueueStatsResponse,
    summary="Get Queue Statistics",
    description="Get comprehensive queue statistics and metrics"
)
async def get_queue_stats(
    current_user: User = Depends(get_current_user)
) -> QueueStatsResponse:
    """Get queue statistics and metrics."""
    try:
        queue = get_message_queue()
        stats = await queue.get_queue_stats()
        
        return QueueStatsResponse(
            stats=stats,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get queue stats: {str(e)}"
        )


@router.get(
    "/workers/stats",
    response_model=WorkerStatsResponse,
    summary="Get Worker Statistics",
    description="Get statistics for all task workers"
)
async def get_worker_stats(
    current_user: User = Depends(get_current_user)
) -> WorkerStatsResponse:
    """Get worker statistics."""
    try:
        worker_manager = get_worker_manager()
        worker_stats = worker_manager.get_all_stats()
        
        active_count = sum(1 for w in worker_stats if w["running"])
        
        return WorkerStatsResponse(
            workers=worker_stats,
            total_workers=len(worker_stats),
            active_workers=active_count
        )
        
    except Exception as e:
        logger.error(f"Failed to get worker stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get worker stats: {str(e)}"
        )


# Queue Health and Monitoring
@router.get(
    "/health",
    summary="Queue Health Check",
    description="Check the health of the message queue system"
)
async def queue_health_check(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Check queue system health."""
    try:
        queue = get_message_queue()
        worker_manager = get_worker_manager()
        
        # Get queue stats
        queue_stats = await queue.get_queue_stats()
        worker_stats = worker_manager.get_all_stats()
        
        # Calculate health metrics
        total_queued = sum(
            queue_stats.get(f"queue_priority_{p}", 0) 
            for p in [1, 5, 8, 10]
        )
        processing_count = queue_stats.get("processing", 0)
        active_workers = sum(1 for w in worker_stats if w["running"])
        
        # Determine overall health
        health_status = "healthy"
        issues = []
        
        if total_queued > 1000:
            health_status = "degraded"
            issues.append("High queue backlog")
        
        if active_workers == 0:
            health_status = "unhealthy"
            issues.append("No active workers")
        
        if processing_count > 50:
            health_status = "degraded"
            issues.append("High processing load")
        
        return {
            "status": health_status,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "total_queued": total_queued,
                "processing": processing_count,
                "active_workers": active_workers,
                "total_workers": len(worker_stats)
            },
            "issues": issues,
            "queue_stats": queue_stats
        }
        
    except Exception as e:
        logger.error(f"Queue health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


# Convenience Endpoints for Specific Task Types
@router.post(
    "/tasks/graphrag/validate",
    response_model=EnqueueTaskResponse,
    summary="Enqueue GraphRAG Validation",
    description="Enqueue a GraphRAG validation task"
)
async def enqueue_graphrag_validation(
    content: str = Body(..., description="Content to validate"),
    project_id: str = Body(..., description="Project ID"),
    validation_type: str = Body(default="comprehensive", description="Validation type"),
    priority: TaskPriority = Body(default=TaskPriority.HIGH),
    current_user: User = Depends(get_current_user)
) -> EnqueueTaskResponse:
    """Enqueue GraphRAG validation task."""
    try:
        queue = get_message_queue()
        
        task_id = await queue.enqueue(
            task_type=TaskType.GRAPHRAG_VALIDATION,
            payload={
                "content": content,
                "project_id": project_id,
                "validation_type": validation_type
            },
            priority=priority,
            user_id=current_user.id
        )
        
        return EnqueueTaskResponse(
            task_id=task_id,
            message="GraphRAG validation task enqueued successfully",
            queued_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to enqueue GraphRAG validation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enqueue validation task: {str(e)}"
        )


@router.post(
    "/tasks/prd/generate",
    response_model=EnqueueTaskResponse,
    summary="Enqueue PRD Generation",
    description="Enqueue a PRD generation task"
)
async def enqueue_prd_generation(
    prompt: str = Body(..., description="PRD generation prompt"),
    context: str = Body(default="", description="Additional context"),
    section_type: str = Body(default="general", description="PRD section type"),
    priority: TaskPriority = Body(default=TaskPriority.HIGH),
    current_user: User = Depends(get_current_user)
) -> EnqueueTaskResponse:
    """Enqueue PRD generation task."""
    try:
        queue = get_message_queue()
        
        task_id = await queue.enqueue(
            task_type=TaskType.PRD_GENERATION,
            payload={
                "prompt": prompt,
                "context": context,
                "section_type": section_type
            },
            priority=priority,
            user_id=current_user.id
        )
        
        return EnqueueTaskResponse(
            task_id=task_id,
            message="PRD generation task enqueued successfully",
            queued_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to enqueue PRD generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enqueue PRD generation: {str(e)}"
        )


@router.post(
    "/tasks/notification/send",
    response_model=EnqueueTaskResponse,
    summary="Enqueue Notification",
    description="Enqueue a notification delivery task"
)
async def enqueue_notification(
    notification_type: str = Body(..., description="Notification type (email, websocket, etc.)"),
    recipient: str = Body(..., description="Notification recipient"),
    subject: str = Body(default="", description="Notification subject"),
    content: str = Body(..., description="Notification content"),
    priority: TaskPriority = Body(default=TaskPriority.LOW),
    current_user: User = Depends(get_current_user)
) -> EnqueueTaskResponse:
    """Enqueue notification delivery task."""
    try:
        queue = get_message_queue()
        
        task_id = await queue.enqueue(
            task_type=TaskType.NOTIFICATION_DELIVERY,
            payload={
                "type": notification_type,
                "recipient": recipient,
                "subject": subject,
                "content": content
            },
            priority=priority,
            user_id=current_user.id
        )
        
        return EnqueueTaskResponse(
            task_id=task_id,
            message="Notification task enqueued successfully",
            queued_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to enqueue notification: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enqueue notification: {str(e)}"
        )


# Admin Endpoints (potentially restricted)
@router.delete(
    "/purge",
    summary="Purge Queue",
    description="Purge queue contents (admin only)"
)
async def purge_queue(
    priority: Optional[TaskPriority] = Query(None, description="Priority queue to purge"),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Purge queue contents. Use with extreme caution."""
    try:
        # TODO: Add admin role check
        # if current_user.role != "admin":
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        queue = get_message_queue()
        removed_count = await queue.purge_queue(priority)
        
        logger.warning(
            f"User {current_user.id} purged queue "
            f"(priority: {priority}, removed: {removed_count})"
        )
        
        return {
            "message": f"Purged {removed_count} tasks from queue",
            "priority": priority.value if priority else "all",
            "removed_count": removed_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to purge queue: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to purge queue: {str(e)}"
        )