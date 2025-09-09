"""
Task types and priority definitions for the message queue system.
"""

from enum import Enum
from typing import Dict, Any
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Supported task types for async processing."""
    
    GRAPHRAG_VALIDATION = "graphrag_validation"
    AGENT_ORCHESTRATION = "agent_orchestration"
    NOTIFICATION_DELIVERY = "notification_delivery"
    PRD_GENERATION = "prd_generation"
    CONTENT_VALIDATION = "content_validation"
    DOCUMENT_PROCESSING = "document_processing"
    SYSTEM_MAINTENANCE = "system_maintenance"


class TaskPriority(int, Enum):
    """Task priority levels (higher number = higher priority)."""
    
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


class TaskStatus(str, Enum):
    """Task processing status."""
    
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


# Task type configurations
TASK_CONFIGS = {
    TaskType.GRAPHRAG_VALIDATION: {
        "default_priority": TaskPriority.HIGH,
        "timeout_seconds": 300,
        "max_retries": 2,
        "requires_auth": True
    },
    TaskType.AGENT_ORCHESTRATION: {
        "default_priority": TaskPriority.NORMAL,
        "timeout_seconds": 600,
        "max_retries": 3,
        "requires_auth": True
    },
    TaskType.NOTIFICATION_DELIVERY: {
        "default_priority": TaskPriority.LOW,
        "timeout_seconds": 60,
        "max_retries": 5,
        "requires_auth": False
    },
    TaskType.PRD_GENERATION: {
        "default_priority": TaskPriority.HIGH,
        "timeout_seconds": 900,
        "max_retries": 2,
        "requires_auth": True
    },
    TaskType.CONTENT_VALIDATION: {
        "default_priority": TaskPriority.NORMAL,
        "timeout_seconds": 180,
        "max_retries": 2,
        "requires_auth": True
    },
    TaskType.DOCUMENT_PROCESSING: {
        "default_priority": TaskPriority.NORMAL,
        "timeout_seconds": 300,
        "max_retries": 2,
        "requires_auth": True
    },
    TaskType.SYSTEM_MAINTENANCE: {
        "default_priority": TaskPriority.LOW,
        "timeout_seconds": 1800,
        "max_retries": 1,
        "requires_auth": False
    }
}


def get_task_config(task_type: TaskType) -> Dict[str, Any]:
    """Get configuration for a task type."""
    return TASK_CONFIGS.get(task_type, {
        "default_priority": TaskPriority.NORMAL,
        "timeout_seconds": 300,
        "max_retries": 3,
        "requires_auth": True
    })