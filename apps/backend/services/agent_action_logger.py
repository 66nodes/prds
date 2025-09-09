"""
Agent Action Logger for QA and Audit Traceability

Comprehensive logging system that captures agent actions, decisions, and outcomes
for quality assurance, debugging, and audit purposes. Integrates with enhanced
parallel executor and GraphRAG validation pipeline.
"""

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from collections import defaultdict, deque

import structlog
from services.enhanced_parallel_executor import ExecutionMetrics, ExecutionStatus
from services.agent_orchestrator import AgentType, AgentTask, WorkflowContext
from services.graphrag.validation_pipeline import ValidationResult, ValidationStatus
from core.redis import get_redis_client
from core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class ActionType(Enum):
    """Types of agent actions that can be logged."""
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    TASK_RETRIED = "task_retried"
    
    AGENT_SELECTED = "agent_selected"
    AGENT_INSTANTIATED = "agent_instantiated"
    AGENT_EXECUTION_STARTED = "agent_execution_started"
    AGENT_EXECUTION_COMPLETED = "agent_execution_completed"
    AGENT_EXECUTION_FAILED = "agent_execution_failed"
    
    VALIDATION_STARTED = "validation_started"
    VALIDATION_COMPLETED = "validation_completed"
    VALIDATION_FAILED = "validation_failed"
    VALIDATION_CORRECTED = "validation_corrected"
    
    ORCHESTRATION_STARTED = "orchestration_started"
    ORCHESTRATION_COMPLETED = "orchestration_completed"
    ORCHESTRATION_FAILED = "orchestration_failed"
    
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"
    CIRCUIT_BREAKER_RECOVERED = "circuit_breaker_recovered"
    
    RESOURCE_OPTIMIZATION = "resource_optimization"
    PERFORMANCE_ALERT = "performance_alert"
    
    ERROR_RECOVERY = "error_recovery"
    FALLBACK_ACTIVATED = "fallback_activated"
    
    DECISION_POINT = "decision_point"
    CONTEXT_UPDATE = "context_update"
    
    QUALITY_CHECK = "quality_check"
    AUDIT_EVENT = "audit_event"


class LogLevel(Enum):
    """Log levels for different types of events."""
    TRACE = "trace"      # Detailed execution flow
    DEBUG = "debug"      # Development information
    INFO = "info"        # General information
    WARN = "warn"        # Warnings and non-critical issues
    ERROR = "error"      # Errors and failures
    CRITICAL = "critical"  # Critical system issues
    AUDIT = "audit"      # Audit and compliance events


@dataclass
class AgentActionLog:
    """Comprehensive log entry for agent actions."""
    # Core identification
    log_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Action classification
    action_type: ActionType = ActionType.TASK_STARTED
    log_level: LogLevel = LogLevel.INFO
    
    # Agent and task context
    agent_type: Optional[AgentType] = None
    task_id: Optional[str] = None
    workflow_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    # Execution context
    execution_stage: Optional[str] = None
    parent_task_id: Optional[str] = None
    dependency_task_ids: List[str] = field(default_factory=list)
    
    # Performance metrics
    duration_ms: Optional[float] = None
    queue_wait_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    # Decision and reasoning
    decision_context: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None
    confidence_score: Optional[float] = None
    alternatives_considered: List[str] = field(default_factory=list)
    
    # Input/Output data
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    intermediate_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Validation results
    validation_result: Optional[Dict[str, Any]] = None
    hallucination_score: Optional[float] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    
    # Error handling
    error_message: Optional[str] = None
    error_category: Optional[str] = None
    error_stack_trace: Optional[str] = None
    recovery_actions: List[str] = field(default_factory=list)
    
    # Resource usage
    resource_usage: Optional[Dict[str, Any]] = None
    circuit_breaker_state: Optional[str] = None
    load_balancing_decision: Optional[Dict[str, Any]] = None
    
    # Business context
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    feature_flags: List[str] = field(default_factory=list)
    
    # Audit and compliance
    audit_trail: List[str] = field(default_factory=list)
    compliance_tags: List[str] = field(default_factory=list)
    sensitivity_level: Optional[str] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage and serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['action_type'] = self.action_type.value
        result['log_level'] = self.log_level.value
        if self.agent_type:
            result['agent_type'] = self.agent_type.value
        return result


@dataclass
class LoggingConfig:
    """Configuration for agent action logging."""
    # Storage settings
    enable_redis_storage: bool = True
    enable_file_storage: bool = True
    enable_database_storage: bool = False
    
    # Retention settings
    redis_ttl_seconds: int = 86400 * 7  # 7 days
    file_rotation_size_mb: int = 100
    max_log_files: int = 10
    
    # Performance settings
    batch_size: int = 100
    flush_interval_seconds: int = 5
    max_queue_size: int = 10000
    enable_async_processing: bool = True
    
    # Content filtering
    log_input_data: bool = True
    log_output_data: bool = True
    log_intermediate_results: bool = False
    mask_sensitive_data: bool = True
    
    # Log levels
    min_log_level: LogLevel = LogLevel.INFO
    enable_trace_logging: bool = False
    enable_performance_logging: bool = True
    
    # Integration settings
    enable_structlog_integration: bool = True
    enable_opentelemetry: bool = False
    enable_metrics_export: bool = True
    
    # Audit settings
    enable_audit_mode: bool = True
    require_correlation_id: bool = True
    enable_compliance_logging: bool = True


class AgentActionLogger:
    """
    Centralized logger for all agent actions and decisions.
    
    Provides comprehensive logging with multiple storage backends,
    performance optimization, and audit capabilities.
    """
    
    def __init__(self, config: Optional[LoggingConfig] = None):
        self.config = config or LoggingConfig()
        self.redis_client = None
        self.log_queue: deque = deque(maxlen=self.config.max_queue_size)
        self.flush_task: Optional[asyncio.Task] = None
        self.is_initialized = False
        
        # Statistics and monitoring
        self.stats = {
            "logs_written": 0,
            "logs_queued": 0,
            "logs_dropped": 0,
            "flush_operations": 0,
            "errors": 0,
            "start_time": datetime.now(timezone.utc)
        }
        
        # Active sessions tracking
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_hierarchies: Dict[str, List[str]] = defaultdict(list)
        
    async def initialize(self) -> None:
        """Initialize the logging system."""
        try:
            # Initialize Redis connection
            if self.config.enable_redis_storage:
                self.redis_client = await get_redis_client()
            
            # Create log directories
            if self.config.enable_file_storage:
                import os
                os.makedirs("logs/agent_actions", exist_ok=True)
            
            # Start async processing
            if self.config.enable_async_processing:
                self.flush_task = asyncio.create_task(self._flush_loop())
            
            self.is_initialized = True
            logger.info("Agent Action Logger initialized successfully", config=asdict(self.config))
            
        except Exception as e:
            logger.error(f"Failed to initialize Agent Action Logger: {str(e)}")
            raise
    
    async def log_action(
        self,
        action_type: ActionType,
        log_level: LogLevel = LogLevel.INFO,
        **kwargs
    ) -> str:
        """
        Log an agent action with comprehensive context.
        
        Args:
            action_type: Type of action being logged
            log_level: Log level for this entry
            **kwargs: Additional context and data
            
        Returns:
            log_id: Unique identifier for this log entry
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Create log entry
        log_entry = AgentActionLog(
            action_type=action_type,
            log_level=log_level,
            **kwargs
        )
        
        # Apply data masking if enabled
        if self.config.mask_sensitive_data:
            self._mask_sensitive_data(log_entry)
        
        # Add to queue for async processing
        if self.config.enable_async_processing:
            if len(self.log_queue) >= self.config.max_queue_size:
                self.stats["logs_dropped"] += 1
                logger.warning("Log queue full, dropping oldest entries")
                # Remove oldest entries to make space
                for _ in range(self.config.batch_size):
                    if self.log_queue:
                        self.log_queue.popleft()
            
            self.log_queue.append(log_entry)
            self.stats["logs_queued"] += 1
        else:
            # Synchronous processing
            await self._write_log_entry(log_entry)
        
        # Update session tracking
        if log_entry.session_id:
            self._update_session_tracking(log_entry)
        
        return log_entry.log_id
    
    async def log_task_lifecycle(
        self,
        stage: str,
        task: AgentTask,
        workflow: WorkflowContext,
        execution_metrics: Optional[ExecutionMetrics] = None,
        **kwargs
    ) -> str:
        """Log task lifecycle events with comprehensive context."""
        
        action_type_map = {
            "started": ActionType.TASK_STARTED,
            "completed": ActionType.TASK_COMPLETED,
            "failed": ActionType.TASK_FAILED,
            "cancelled": ActionType.TASK_CANCELLED,
            "retried": ActionType.TASK_RETRIED
        }
        
        action_type = action_type_map.get(stage, ActionType.TASK_STARTED)
        
        # Build context from task and workflow
        context = {
            "agent_type": task.agent_type,
            "task_id": task.task_id,
            "workflow_id": getattr(workflow, 'workflow_id', None),
            "session_id": getattr(workflow, 'session_id', None),
            "correlation_id": getattr(workflow, 'correlation_id', None),
            "execution_stage": stage,
            "input_data": getattr(task, 'input_data', None),
        }
        
        # Add execution metrics if available
        if execution_metrics:
            context.update({
                "duration_ms": execution_metrics.duration_ms,
                "queue_wait_ms": execution_metrics.queue_wait_time_ms,
                "memory_usage_mb": execution_metrics.memory_usage_mb,
                "cpu_usage_percent": execution_metrics.cpu_usage_percent,
                "error_message": execution_metrics.error_message,
            })
        
        context.update(kwargs)
        
        return await self.log_action(action_type, **context)
    
    async def log_validation_event(
        self,
        stage: str,
        validation_result: ValidationResult,
        task_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Log GraphRAG validation events."""
        
        action_type_map = {
            "started": ActionType.VALIDATION_STARTED,
            "completed": ActionType.VALIDATION_COMPLETED,
            "failed": ActionType.VALIDATION_FAILED,
            "corrected": ActionType.VALIDATION_CORRECTED
        }
        
        action_type = action_type_map.get(stage, ActionType.VALIDATION_STARTED)
        
        context = {
            "task_id": task_id,
            "validation_result": {
                "validation_id": validation_result.validation_id,
                "status": validation_result.status.value,
                "confidence": validation_result.overall_confidence,
                "processing_time_ms": validation_result.processing_time_ms,
                "issues_count": len(validation_result.issues_found or []),
                "recommendations_count": len(validation_result.recommendations or [])
            },
            "confidence_score": validation_result.overall_confidence,
            "quality_metrics": {
                "status": validation_result.status.value,
                "issues_found": len(validation_result.issues_found or []),
                "processing_time": validation_result.processing_time_ms
            }
        }
        
        # Add hallucination score if available
        if hasattr(validation_result, 'hallucination_detection_result'):
            hallucination_result = validation_result.hallucination_detection_result
            if hallucination_result and 'hallucination_score' in hallucination_result:
                context["hallucination_score"] = hallucination_result['hallucination_score']
        
        context.update(kwargs)
        
        return await self.log_action(action_type, **context)
    
    async def log_orchestration_event(
        self,
        stage: str,
        workflow: WorkflowContext,
        tasks: Optional[List[AgentTask]] = None,
        results: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Log orchestration lifecycle events."""
        
        action_type_map = {
            "started": ActionType.ORCHESTRATION_STARTED,
            "completed": ActionType.ORCHESTRATION_COMPLETED,
            "failed": ActionType.ORCHESTRATION_FAILED
        }
        
        action_type = action_type_map.get(stage, ActionType.ORCHESTRATION_STARTED)
        
        context = {
            "workflow_id": getattr(workflow, 'workflow_id', None),
            "session_id": getattr(workflow, 'session_id', None),
            "correlation_id": getattr(workflow, 'correlation_id', None),
            "execution_stage": stage,
            "tasks_count": len(tasks) if tasks else 0,
            "results_count": len(results) if results else 0
        }
        
        if tasks:
            context["task_ids"] = [task.task_id for task in tasks]
            context["agent_types"] = [task.agent_type.value for task in tasks]
        
        if results:
            context["output_data"] = {
                "successful_tasks": len([r for r in results.values() if r.get("status") == "completed"]),
                "failed_tasks": len([r for r in results.values() if r.get("status") == "failed"]),
                "total_tasks": len(results)
            }
        
        context.update(kwargs)
        
        return await self.log_action(action_type, **context)
    
    async def log_decision_point(
        self,
        decision_type: str,
        context: Dict[str, Any],
        chosen_option: str,
        alternatives: List[str],
        reasoning: str,
        confidence: float,
        **kwargs
    ) -> str:
        """Log decision points for audit and analysis."""
        
        return await self.log_action(
            ActionType.DECISION_POINT,
            log_level=LogLevel.AUDIT,
            decision_context={
                "decision_type": decision_type,
                "chosen_option": chosen_option,
                "context": context
            },
            reasoning=reasoning,
            confidence_score=confidence,
            alternatives_considered=alternatives,
            **kwargs
        )
    
    async def get_logs(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "timestamp",
        sort_order: str = "desc"
    ) -> List[Dict[str, Any]]:
        """Query logs with filtering and pagination."""
        
        if not self.config.enable_redis_storage or not self.redis_client:
            return []
        
        # Build Redis query
        # This is a simplified version - in production, you'd use Redis Streams or Search
        try:
            # Get all log keys
            keys = await self.redis_client.keys("agent_logs:*")
            
            logs = []
            for key in keys:
                log_data = await self.redis_client.get(key)
                if log_data:
                    log_entry = json.loads(log_data)
                    
                    # Apply filters
                    if filters and not self._matches_filters(log_entry, filters):
                        continue
                    
                    logs.append(log_entry)
            
            # Sort logs
            reverse = sort_order == "desc"
            logs.sort(key=lambda x: x.get(sort_by, ""), reverse=reverse)
            
            # Apply pagination
            return logs[offset:offset + limit]
            
        except Exception as e:
            logger.error(f"Failed to query logs: {str(e)}")
            return []
    
    async def get_session_logs(
        self,
        session_id: str,
        include_hierarchy: bool = True
    ) -> List[Dict[str, Any]]:
        """Get all logs for a specific session."""
        
        filters = {"session_id": session_id}
        logs = await self.get_logs(filters=filters, limit=1000, sort_by="timestamp")
        
        if include_hierarchy and session_id in self.session_hierarchies:
            # Include logs from child sessions
            for child_session in self.session_hierarchies[session_id]:
                child_logs = await self.get_logs(
                    filters={"session_id": child_session},
                    limit=1000,
                    sort_by="timestamp"
                )
                logs.extend(child_logs)
        
        return logs
    
    async def get_task_audit_trail(self, task_id: str) -> List[Dict[str, Any]]:
        """Get complete audit trail for a specific task."""
        
        filters = {"task_id": task_id}
        return await self.get_logs(filters=filters, limit=1000, sort_by="timestamp")
    
    async def get_agent_performance_logs(
        self,
        agent_type: AgentType,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get performance logs for a specific agent type."""
        
        filters = {"agent_type": agent_type.value}
        if start_time:
            filters["start_time"] = start_time.isoformat()
        if end_time:
            filters["end_time"] = end_time.isoformat()
        
        return await self.get_logs(filters=filters, limit=1000)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get logging system statistics."""
        
        uptime_seconds = (datetime.now(timezone.utc) - self.stats["start_time"]).total_seconds()
        
        return {
            **self.stats,
            "uptime_seconds": uptime_seconds,
            "queue_size": len(self.log_queue),
            "active_sessions": len(self.active_sessions),
            "config": asdict(self.config),
            "is_initialized": self.is_initialized
        }
    
    def _mask_sensitive_data(self, log_entry: AgentActionLog) -> None:
        """Mask sensitive data in log entries."""
        
        sensitive_patterns = [
            "password", "token", "key", "secret", "credential",
            "email", "phone", "ssn", "credit_card"
        ]
        
        def mask_dict(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not data:
                return data
            
            masked = {}
            for k, v in data.items():
                if any(pattern in k.lower() for pattern in sensitive_patterns):
                    masked[k] = "***MASKED***"
                elif isinstance(v, dict):
                    masked[k] = mask_dict(v)
                elif isinstance(v, str) and len(v) > 20:
                    # Mask long strings that might contain sensitive data
                    masked[k] = v[:10] + "***" + v[-10:]
                else:
                    masked[k] = v
            return masked
        
        log_entry.input_data = mask_dict(log_entry.input_data)
        log_entry.output_data = mask_dict(log_entry.output_data)
        log_entry.custom_attributes = mask_dict(log_entry.custom_attributes)
    
    def _update_session_tracking(self, log_entry: AgentActionLog) -> None:
        """Update session tracking information."""
        
        session_id = log_entry.session_id
        if not session_id:
            return
        
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                "start_time": log_entry.timestamp,
                "task_count": 0,
                "agent_types": set(),
                "status": "active"
            }
        
        session_info = self.active_sessions[session_id]
        
        if log_entry.task_id:
            session_info["task_count"] += 1
        
        if log_entry.agent_type:
            session_info["agent_types"].add(log_entry.agent_type.value)
        
        # Update parent-child relationships
        if log_entry.parent_task_id and log_entry.parent_task_id != session_id:
            self.session_hierarchies[log_entry.parent_task_id].append(session_id)
    
    def _matches_filters(self, log_entry: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if log entry matches the given filters."""
        
        for key, value in filters.items():
            if key not in log_entry:
                return False
            
            log_value = log_entry[key]
            
            # Handle different filter types
            if isinstance(value, list):
                if log_value not in value:
                    return False
            elif isinstance(value, dict):
                # Range filters
                if "start" in value and log_value < value["start"]:
                    return False
                if "end" in value and log_value > value["end"]:
                    return False
            elif log_value != value:
                return False
        
        return True
    
    async def _write_log_entry(self, log_entry: AgentActionLog) -> None:
        """Write a single log entry to configured storage."""
        
        try:
            log_dict = log_entry.to_dict()
            
            # Write to Redis
            if self.config.enable_redis_storage and self.redis_client:
                await self.redis_client.setex(
                    f"agent_logs:{log_entry.log_id}",
                    self.config.redis_ttl_seconds,
                    json.dumps(log_dict)
                )
            
            # Write to structured logging
            if self.config.enable_structlog_integration:
                structlog_logger = logger.bind(**{
                    k: v for k, v in log_dict.items() 
                    if k not in ['log_id', 'timestamp']
                })
                
                log_method = getattr(structlog_logger, log_entry.log_level.value, structlog_logger.info)
                log_method(f"Agent action: {log_entry.action_type.value}")
            
            self.stats["logs_written"] += 1
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Failed to write log entry: {str(e)}", log_id=log_entry.log_id)
    
    async def _flush_loop(self) -> None:
        """Async loop to flush queued log entries."""
        
        while True:
            try:
                if not self.log_queue:
                    await asyncio.sleep(self.config.flush_interval_seconds)
                    continue
                
                # Process batch
                batch_size = min(self.config.batch_size, len(self.log_queue))
                batch = [self.log_queue.popleft() for _ in range(batch_size)]
                
                # Write batch
                tasks = [self._write_log_entry(entry) for entry in batch]
                await asyncio.gather(*tasks, return_exceptions=True)
                
                self.stats["flush_operations"] += 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.stats["errors"] += 1
                logger.error(f"Error in flush loop: {str(e)}")
                await asyncio.sleep(1)  # Brief pause before continuing
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the logging system."""
        
        logger.info("Shutting down Agent Action Logger")
        
        # Cancel flush task
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining logs
        while self.log_queue:
            log_entry = self.log_queue.popleft()
            await self._write_log_entry(log_entry)
        
        logger.info("Agent Action Logger shutdown complete")


# Global logger instance
_agent_logger: Optional[AgentActionLogger] = None


async def get_agent_logger() -> AgentActionLogger:
    """Get the global agent action logger instance."""
    global _agent_logger
    
    if not _agent_logger:
        _agent_logger = AgentActionLogger()
        await _agent_logger.initialize()
    
    return _agent_logger


# Convenience functions for common logging operations
async def log_task_started(task: AgentTask, workflow: WorkflowContext, **kwargs) -> str:
    """Log task start event."""
    logger_instance = await get_agent_logger()
    return await logger_instance.log_task_lifecycle("started", task, workflow, **kwargs)


async def log_task_completed(task: AgentTask, workflow: WorkflowContext, execution_metrics: ExecutionMetrics, **kwargs) -> str:
    """Log task completion event."""
    logger_instance = await get_agent_logger()
    return await logger_instance.log_task_lifecycle("completed", task, workflow, execution_metrics, **kwargs)


async def log_validation_result(validation_result: ValidationResult, task_id: str, **kwargs) -> str:
    """Log validation result."""
    logger_instance = await get_agent_logger()
    return await logger_instance.log_validation_event("completed", validation_result, task_id, **kwargs)


async def log_decision(
    decision_type: str,
    context: Dict[str, Any],
    chosen_option: str,
    alternatives: List[str],
    reasoning: str,
    confidence: float,
    **kwargs
) -> str:
    """Log decision point."""
    logger_instance = await get_agent_logger()
    return await logger_instance.log_decision_point(
        decision_type, context, chosen_option, alternatives, reasoning, confidence, **kwargs
    )