"""
Task workers for processing different types of queue messages.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from .message_queue import MessageQueue, QueueMessage, TaskResult, get_message_queue
from .task_types import TaskType, TaskStatus
from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class TaskWorker(ABC):
    """Abstract base class for task workers."""
    
    def __init__(self, worker_id: Optional[str] = None):
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.queue = get_message_queue()
        self.running = False
        self.processed_count = 0
        self.error_count = 0
        
    @abstractmethod
    async def process_task(self, message: QueueMessage) -> Dict[str, Any]:
        """Process a task message. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def can_handle(self, task_type: TaskType) -> bool:
        """Check if worker can handle this task type."""
        pass
    
    async def start(self) -> None:
        """Start the worker event loop."""
        self.running = True
        logger.info(f"Starting worker {self.worker_id}")
        
        while self.running:
            try:
                # Get next task from queue
                message = await self.queue.dequeue(
                    worker_id=self.worker_id,
                    timeout=5
                )
                
                if not message:
                    continue
                
                # Check if we can handle this task type
                if not self.can_handle(message.task_type):
                    await self.queue.fail_task(
                        message.id,
                        f"Worker {self.worker_id} cannot handle task type {message.task_type}",
                        worker_id=self.worker_id,
                        retry=True
                    )
                    continue
                
                # Process the task
                try:
                    logger.info(f"Processing task {message.id} of type {message.task_type}")
                    
                    result = await self.process_task(message)
                    
                    # Mark task as completed
                    await self.queue.complete_task(
                        message.id,
                        result=result,
                        worker_id=self.worker_id
                    )
                    
                    self.processed_count += 1
                    logger.info(f"Completed task {message.id}")
                    
                except Exception as e:
                    logger.error(f"Task {message.id} failed: {e}")
                    
                    await self.queue.fail_task(
                        message.id,
                        str(e),
                        worker_id=self.worker_id,
                        retry=True
                    )
                    
                    self.error_count += 1
                
            except asyncio.CancelledError:
                logger.info(f"Worker {self.worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                await asyncio.sleep(1)  # Brief pause on error
        
        logger.info(f"Worker {self.worker_id} stopped")
    
    async def stop(self) -> None:
        """Stop the worker."""
        self.running = False
        logger.info(f"Stopping worker {self.worker_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "worker_id": self.worker_id,
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "success_rate": (
                self.processed_count / (self.processed_count + self.error_count)
                if self.processed_count + self.error_count > 0 else 0
            ),
            "running": self.running
        }


class GraphRAGWorker(TaskWorker):
    """Worker for GraphRAG validation tasks."""
    
    def can_handle(self, task_type: TaskType) -> bool:
        """Check if this worker handles GraphRAG tasks."""
        return task_type in [
            TaskType.GRAPHRAG_VALIDATION,
            TaskType.CONTENT_VALIDATION
        ]
    
    async def process_task(self, message: QueueMessage) -> Dict[str, Any]:
        """Process GraphRAG validation task."""
        payload = message.payload
        
        if message.task_type == TaskType.GRAPHRAG_VALIDATION:
            return await self._validate_graphrag_content(payload)
        elif message.task_type == TaskType.CONTENT_VALIDATION:
            return await self._validate_content(payload)
        else:
            raise ValueError(f"Unsupported task type: {message.task_type}")
    
    async def _validate_graphrag_content(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate content using GraphRAG."""
        content = payload.get("content", "")
        project_id = payload.get("project_id", "")
        validation_type = payload.get("validation_type", "comprehensive")
        
        # Simulate GraphRAG validation
        await asyncio.sleep(2)  # Simulate processing time
        
        # Mock validation results
        validation_result = {
            "content": content,
            "project_id": project_id,
            "validation_type": validation_type,
            "hallucination_score": 0.85,
            "entity_matches": 12,
            "relationship_matches": 8,
            "confidence": 0.92,
            "issues_found": [],
            "suggestions": [
                "Consider adding more specific technical details",
                "Verify external references"
            ],
            "processed_at": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"GraphRAG validation completed for project {project_id}")
        return validation_result
    
    async def _validate_content(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """General content validation."""
        content = payload.get("content", "")
        criteria = payload.get("criteria", [])
        
        # Simulate content validation
        await asyncio.sleep(1)
        
        validation_result = {
            "content_length": len(content),
            "criteria_checked": criteria,
            "quality_score": 0.88,
            "readability_score": 0.76,
            "completeness_score": 0.93,
            "issues": [],
            "recommendations": [
                "Content meets quality standards",
                "Minor improvements possible"
            ],
            "processed_at": datetime.now(timezone.utc).isoformat()
        }
        
        return validation_result


class AgentWorker(TaskWorker):
    """Worker for agent orchestration tasks."""
    
    def can_handle(self, task_type: TaskType) -> bool:
        """Check if this worker handles agent tasks."""
        return task_type in [
            TaskType.AGENT_ORCHESTRATION,
            TaskType.PRD_GENERATION,
            TaskType.DOCUMENT_PROCESSING
        ]
    
    async def process_task(self, message: QueueMessage) -> Dict[str, Any]:
        """Process agent orchestration task."""
        payload = message.payload
        
        if message.task_type == TaskType.AGENT_ORCHESTRATION:
            return await self._orchestrate_agents(payload)
        elif message.task_type == TaskType.PRD_GENERATION:
            return await self._generate_prd(payload)
        elif message.task_type == TaskType.DOCUMENT_PROCESSING:
            return await self._process_document(payload)
        else:
            raise ValueError(f"Unsupported task type: {message.task_type}")
    
    async def _orchestrate_agents(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate multiple agents for complex tasks."""
        task_description = payload.get("task_description", "")
        agents_required = payload.get("agents", [])
        context = payload.get("context", {})
        
        # Simulate agent orchestration
        await asyncio.sleep(3)
        
        orchestration_result = {
            "task_description": task_description,
            "agents_used": agents_required,
            "context": context,
            "execution_plan": [
                "Analysis phase completed",
                "Design phase completed", 
                "Implementation phase in progress"
            ],
            "current_status": "in_progress",
            "estimated_completion": "2024-01-20T15:30:00Z",
            "intermediate_results": {
                "analysis": "Requirements analyzed successfully",
                "design": "Architecture design completed"
            },
            "processed_at": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Agent orchestration initiated for task: {task_description}")
        return orchestration_result
    
    async def _generate_prd(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate PRD content."""
        prompt = payload.get("prompt", "")
        context = payload.get("context", "")
        section_type = payload.get("section_type", "general")
        
        # Simulate PRD generation
        await asyncio.sleep(4)
        
        prd_result = {
            "prompt": prompt,
            "context": context,
            "section_type": section_type,
            "generated_content": f"# {section_type.title()} Section\n\n{prompt}\n\nGenerated PRD content based on requirements...",
            "word_count": 450,
            "quality_score": 0.91,
            "completeness": 0.87,
            "processed_at": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"PRD generation completed for section: {section_type}")
        return prd_result
    
    async def _process_document(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process document content."""
        document_type = payload.get("document_type", "")
        content = payload.get("content", "")
        processing_options = payload.get("options", {})
        
        # Simulate document processing
        await asyncio.sleep(2)
        
        processing_result = {
            "document_type": document_type,
            "original_length": len(content),
            "processed_content": f"Processed: {content[:100]}...",
            "processing_options": processing_options,
            "metadata": {
                "sections_identified": 5,
                "images_found": 2,
                "tables_found": 1,
                "links_found": 8
            },
            "quality_metrics": {
                "structure_score": 0.89,
                "content_score": 0.84
            },
            "processed_at": datetime.now(timezone.utc).isoformat()
        }
        
        return processing_result


class NotificationWorker(TaskWorker):
    """Worker for notification delivery tasks."""
    
    def can_handle(self, task_type: TaskType) -> bool:
        """Check if this worker handles notification tasks."""
        return task_type == TaskType.NOTIFICATION_DELIVERY
    
    async def process_task(self, message: QueueMessage) -> Dict[str, Any]:
        """Process notification delivery task."""
        payload = message.payload
        
        notification_type = payload.get("type", "email")
        recipient = payload.get("recipient", "")
        subject = payload.get("subject", "")
        content = payload.get("content", "")
        
        # Simulate notification delivery
        await asyncio.sleep(1)
        
        # Mock delivery based on type
        if notification_type == "email":
            delivery_result = await self._send_email(recipient, subject, content)
        elif notification_type == "websocket":
            delivery_result = await self._send_websocket(recipient, content)
        else:
            delivery_result = await self._send_generic(notification_type, recipient, content)
        
        return delivery_result
    
    async def _send_email(self, recipient: str, subject: str, content: str) -> Dict[str, Any]:
        """Mock email delivery."""
        # Simulate email sending
        await asyncio.sleep(0.5)
        
        return {
            "type": "email",
            "recipient": recipient,
            "subject": subject,
            "status": "delivered",
            "message_id": f"email-{uuid.uuid4().hex[:12]}",
            "delivery_time": datetime.now(timezone.utc).isoformat()
        }
    
    async def _send_websocket(self, recipient: str, content: str) -> Dict[str, Any]:
        """Mock WebSocket notification."""
        await asyncio.sleep(0.1)
        
        return {
            "type": "websocket",
            "recipient": recipient,
            "status": "delivered",
            "message_id": f"ws-{uuid.uuid4().hex[:12]}",
            "delivery_time": datetime.now(timezone.utc).isoformat()
        }
    
    async def _send_generic(self, notification_type: str, recipient: str, content: str) -> Dict[str, Any]:
        """Mock generic notification delivery."""
        await asyncio.sleep(0.3)
        
        return {
            "type": notification_type,
            "recipient": recipient,
            "status": "delivered",
            "message_id": f"{notification_type}-{uuid.uuid4().hex[:12]}",
            "delivery_time": datetime.now(timezone.utc).isoformat()
        }


class WorkerManager:
    """Manages multiple task workers."""
    
    def __init__(self):
        self.workers: List[TaskWorker] = []
        self.running = False
        
    def add_worker(self, worker: TaskWorker) -> None:
        """Add a worker to the manager."""
        self.workers.append(worker)
        logger.info(f"Added worker {worker.worker_id}")
    
    async def start_all(self) -> None:
        """Start all workers."""
        if self.running:
            return
        
        self.running = True
        logger.info(f"Starting {len(self.workers)} workers")
        
        # Start all workers concurrently
        tasks = [worker.start() for worker in self.workers]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop_all(self) -> None:
        """Stop all workers."""
        if not self.running:
            return
        
        self.running = False
        logger.info(f"Stopping {len(self.workers)} workers")
        
        # Stop all workers
        for worker in self.workers:
            await worker.stop()
    
    def get_all_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all workers."""
        return [worker.get_stats() for worker in self.workers]


# Global worker manager
_worker_manager: Optional[WorkerManager] = None


def get_worker_manager() -> WorkerManager:
    """Get global worker manager instance."""
    global _worker_manager
    
    if not _worker_manager:
        _worker_manager = WorkerManager()
        
        # Add default workers
        _worker_manager.add_worker(GraphRAGWorker())
        _worker_manager.add_worker(AgentWorker())
        _worker_manager.add_worker(NotificationWorker())
    
    return _worker_manager