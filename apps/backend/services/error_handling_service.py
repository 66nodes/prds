"""
Comprehensive Error Handling and Recovery Service

Advanced error detection, classification, recovery strategies, and escalation system
for the multi-agent orchestration platform with graceful degradation capabilities.
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
from contextlib import asynccontextmanager

import structlog
from services.agent_orchestrator import AgentType, AgentTask, WorkflowContext, TaskStatus
from services.enhanced_parallel_executor import ExecutionStatus, CircuitState
from services.agent_action_logger import get_agent_logger, ActionType, LogLevel

logger = structlog.get_logger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels for classification and escalation."""
    LOW = "low"                    # Minor issues, retry automatically
    MEDIUM = "medium"              # Moderate issues, may require attention
    HIGH = "high"                  # Serious issues, immediate attention needed
    CRITICAL = "critical"          # System-threatening, escalate immediately
    FATAL = "fatal"                # Unrecoverable, system shutdown required


class ErrorCategory(str, Enum):
    """Error categories for classification and targeted recovery."""
    # System Errors
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    MEMORY_ERROR = "memory_error"
    CPU_OVERLOAD = "cpu_overload"
    DISK_FULL = "disk_full"
    NETWORK_ERROR = "network_error"
    
    # Agent Errors
    AGENT_UNAVAILABLE = "agent_unavailable"
    AGENT_TIMEOUT = "agent_timeout"
    AGENT_CRASH = "agent_crash"
    AGENT_OVERLOAD = "agent_overload"
    INVALID_AGENT_RESPONSE = "invalid_agent_response"
    
    # Orchestration Errors
    TASK_DEPENDENCY_FAILURE = "task_dependency_failure"
    WORKFLOW_CORRUPTION = "workflow_corruption"
    DEADLOCK_DETECTED = "deadlock_detected"
    PRIORITY_INVERSION = "priority_inversion"
    
    # Data Errors
    VALIDATION_FAILURE = "validation_failure"
    DATA_CORRUPTION = "data_corruption"
    SCHEMA_MISMATCH = "schema_mismatch"
    SERIALIZATION_ERROR = "serialization_error"
    
    # External Service Errors
    DATABASE_ERROR = "database_error"
    GRAPHRAG_ERROR = "graphrag_error"
    LLM_ERROR = "llm_error"
    CACHE_ERROR = "cache_error"
    
    # Configuration Errors
    CONFIG_ERROR = "config_error"
    PERMISSION_DENIED = "permission_denied"
    AUTHENTICATION_FAILURE = "authentication_failure"
    
    # Unknown
    UNKNOWN_ERROR = "unknown_error"


class RecoveryStrategy(str, Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"                              # Simple retry with backoff
    RETRY_WITH_FALLBACK = "retry_with_fallback"  # Try alternative approach
    CIRCUIT_BREAKER = "circuit_breaker"          # Open circuit, fail fast
    GRACEFUL_DEGRADATION = "graceful_degradation"  # Reduce functionality
    TASK_REDISTRIBUTION = "task_redistribution"  # Move tasks to other agents
    RESOURCE_SCALING = "resource_scaling"        # Scale resources up/down
    SYSTEM_RESTART = "system_restart"           # Restart components
    MANUAL_INTERVENTION = "manual_intervention"  # Escalate to humans
    IGNORE = "ignore"                           # Continue despite error


class EscalationLevel(str, Enum):
    """Escalation levels for error handling."""
    AUTO_RECOVER = "auto_recover"        # Handle automatically
    LOG_WARNING = "log_warning"          # Log and continue
    ADMIN_ALERT = "admin_alert"          # Alert administrators
    URGENT_RESPONSE = "urgent_response"  # Immediate response required
    EMERGENCY = "emergency"              # Emergency protocols activated


@dataclass
class ErrorMetrics:
    """Metrics tracking for error analysis."""
    error_count: int = 0
    first_occurrence: Optional[datetime] = None
    last_occurrence: Optional[datetime] = None
    frequency_per_hour: float = 0.0
    recovery_success_rate: float = 0.0
    average_recovery_time_ms: float = 0.0
    impact_score: float = 0.0  # Weighted impact on system performance


@dataclass
class ErrorPattern:
    """Pattern recognition for error trends."""
    category: ErrorCategory
    severity: ErrorSeverity
    agent_types: Set[AgentType]
    time_pattern: str  # e.g., "peak_hours", "night", "weekend"
    correlation_factors: Dict[str, Any]
    predicted_next_occurrence: Optional[datetime] = None


@dataclass
class RecoveryPlan:
    """Comprehensive recovery plan for errors."""
    strategy: RecoveryStrategy
    max_attempts: int
    backoff_strategy: str  # "exponential", "linear", "fixed"
    fallback_options: List[str]
    escalation_threshold: int
    resource_requirements: Dict[str, Any]
    success_criteria: Dict[str, Any]
    monitoring_metrics: List[str]


@dataclass
class ErrorEvent:
    """Comprehensive error event record."""
    event_id: str
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    agent_type: Optional[AgentType]
    task_id: Optional[str]
    workflow_id: Optional[str]
    error_message: str
    stack_trace: Optional[str]
    context: Dict[str, Any]
    recovery_strategy: RecoveryStrategy
    recovery_attempts: int = 0
    recovery_success: bool = False
    escalation_level: EscalationLevel = EscalationLevel.AUTO_RECOVER
    resolution_time_ms: Optional[int] = None
    impact_metrics: Dict[str, Any] = field(default_factory=dict)


class ErrorHandlingService:
    """
    Comprehensive error handling and recovery service.
    
    Features:
    - Intelligent error classification with machine learning patterns
    - Dynamic recovery strategy selection based on context and history
    - Circuit breaker management with adaptive thresholds
    - Graceful degradation with service prioritization
    - Proactive error prediction and prevention
    - Comprehensive error analytics and reporting
    - Automated escalation with smart routing
    """

    def __init__(self):
        # Error tracking and metrics
        self.error_history: deque = deque(maxlen=10000)
        self.error_metrics: Dict[ErrorCategory, ErrorMetrics] = {}
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.active_errors: Dict[str, ErrorEvent] = {}
        
        # Recovery strategies mapping
        self.recovery_strategies: Dict[ErrorCategory, RecoveryPlan] = {}
        self.recovery_history: deque = deque(maxlen=5000)
        
        # Circuit breaker management
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.circuit_breaker_thresholds: Dict[ErrorCategory, Dict[str, Any]] = {}
        
        # Escalation management
        self.escalation_rules: Dict[ErrorCategory, Dict[str, Any]] = {}
        self.escalation_history: deque = deque(maxlen=1000)
        
        # System health tracking
        self.system_health_score: float = 100.0
        self.degradation_levels: Dict[str, int] = {}
        self.critical_services: Set[str] = set()
        
        # Analytics and prediction
        self.error_correlation_matrix: Dict[str, Dict[str, float]] = {}
        self.prediction_models: Dict[ErrorCategory, Dict[str, Any]] = {}
        
        # Configuration
        self.agent_logger = None
        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize the error handling service."""
        try:
            self.agent_logger = await get_agent_logger()
            await self._initialize_recovery_strategies()
            await self._initialize_circuit_breaker_thresholds()
            await self._initialize_escalation_rules()
            await self._initialize_critical_services()
            
            self.is_initialized = True
            logger.info("Error handling service initialized successfully")
            
            await self.agent_logger.log_action(
                ActionType.ORCHESTRATION_STARTED,
                log_level=LogLevel.INFO,
                execution_stage="error_service_initialized",
                reasoning="Error handling service initialized with comprehensive recovery strategies",
                tags=["error_handling", "initialization", "system_startup"]
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize error handling service: {str(e)}")
            raise

    async def _initialize_recovery_strategies(self) -> None:
        """Initialize recovery strategies for different error categories."""
        
        # Agent-related errors
        self.recovery_strategies[ErrorCategory.AGENT_TIMEOUT] = RecoveryPlan(
            strategy=RecoveryStrategy.RETRY_WITH_FALLBACK,
            max_attempts=3,
            backoff_strategy="exponential",
            fallback_options=["alternative_agent", "simplified_task"],
            escalation_threshold=5,
            resource_requirements={"timeout_extension": 2.0},
            success_criteria={"completion_rate": 0.8},
            monitoring_metrics=["response_time", "success_rate"]
        )
        
        self.recovery_strategies[ErrorCategory.AGENT_UNAVAILABLE] = RecoveryPlan(
            strategy=RecoveryStrategy.TASK_REDISTRIBUTION,
            max_attempts=2,
            backoff_strategy="linear",
            fallback_options=["agent_pool_expansion", "task_queuing"],
            escalation_threshold=3,
            resource_requirements={"alternative_agents": 2},
            success_criteria={"task_completion": 1.0},
            monitoring_metrics=["agent_availability", "task_distribution"]
        )
        
        # System resource errors
        self.recovery_strategies[ErrorCategory.RESOURCE_EXHAUSTION] = RecoveryPlan(
            strategy=RecoveryStrategy.RESOURCE_SCALING,
            max_attempts=2,
            backoff_strategy="fixed",
            fallback_options=["graceful_degradation", "task_prioritization"],
            escalation_threshold=2,
            resource_requirements={"scale_factor": 1.5},
            success_criteria={"resource_utilization": 0.8},
            monitoring_metrics=["cpu_usage", "memory_usage", "response_time"]
        )
        
        # Network and external service errors
        self.recovery_strategies[ErrorCategory.NETWORK_ERROR] = RecoveryPlan(
            strategy=RecoveryStrategy.RETRY,
            max_attempts=5,
            backoff_strategy="exponential",
            fallback_options=["cache_fallback", "offline_mode"],
            escalation_threshold=10,
            resource_requirements={"connection_pool": 1.2},
            success_criteria={"success_rate": 0.9},
            monitoring_metrics=["network_latency", "connection_success"]
        )
        
        # Data and validation errors
        self.recovery_strategies[ErrorCategory.VALIDATION_FAILURE] = RecoveryPlan(
            strategy=RecoveryStrategy.RETRY_WITH_FALLBACK,
            max_attempts=2,
            backoff_strategy="linear",
            fallback_options=["schema_relaxation", "manual_validation"],
            escalation_threshold=3,
            resource_requirements={"validation_timeout": 2.0},
            success_criteria={"validation_pass_rate": 0.95},
            monitoring_metrics=["validation_time", "error_rate"]
        )

    async def _initialize_circuit_breaker_thresholds(self) -> None:
        """Initialize circuit breaker thresholds for different error categories."""
        
        default_thresholds = {
            "failure_threshold": 5,
            "recovery_timeout": 30,
            "half_open_max_calls": 3,
            "success_threshold": 2
        }
        
        # Customize thresholds based on error category
        category_overrides = {
            ErrorCategory.AGENT_CRASH: {
                "failure_threshold": 2,  # More sensitive
                "recovery_timeout": 60   # Longer recovery
            },
            ErrorCategory.DATABASE_ERROR: {
                "failure_threshold": 3,
                "recovery_timeout": 45
            },
            ErrorCategory.RESOURCE_EXHAUSTION: {
                "failure_threshold": 2,
                "recovery_timeout": 90
            },
            ErrorCategory.LLM_ERROR: {
                "failure_threshold": 10,  # More tolerant
                "recovery_timeout": 20    # Faster recovery
            }
        }
        
        for category in ErrorCategory:
            thresholds = default_thresholds.copy()
            if category in category_overrides:
                thresholds.update(category_overrides[category])
            self.circuit_breaker_thresholds[category] = thresholds

    async def _initialize_escalation_rules(self) -> None:
        """Initialize escalation rules for different error categories."""
        
        # Define escalation rules based on severity and frequency
        escalation_matrix = {
            ErrorCategory.AGENT_CRASH: {
                ErrorSeverity.HIGH: EscalationLevel.ADMIN_ALERT,
                ErrorSeverity.CRITICAL: EscalationLevel.URGENT_RESPONSE,
                "frequency_threshold": 3  # per hour
            },
            ErrorCategory.RESOURCE_EXHAUSTION: {
                ErrorSeverity.MEDIUM: EscalationLevel.LOG_WARNING,
                ErrorSeverity.HIGH: EscalationLevel.ADMIN_ALERT,
                ErrorSeverity.CRITICAL: EscalationLevel.URGENT_RESPONSE,
                "frequency_threshold": 2
            },
            ErrorCategory.VALIDATION_FAILURE: {
                ErrorSeverity.MEDIUM: EscalationLevel.AUTO_RECOVER,
                ErrorSeverity.HIGH: EscalationLevel.LOG_WARNING,
                ErrorSeverity.CRITICAL: EscalationLevel.ADMIN_ALERT,
                "frequency_threshold": 10
            },
            ErrorCategory.DATABASE_ERROR: {
                ErrorSeverity.HIGH: EscalationLevel.URGENT_RESPONSE,
                ErrorSeverity.CRITICAL: EscalationLevel.EMERGENCY,
                "frequency_threshold": 2
            }
        }
        
        # Set default rules for categories not explicitly defined
        default_rules = {
            ErrorSeverity.LOW: EscalationLevel.AUTO_RECOVER,
            ErrorSeverity.MEDIUM: EscalationLevel.LOG_WARNING,
            ErrorSeverity.HIGH: EscalationLevel.ADMIN_ALERT,
            ErrorSeverity.CRITICAL: EscalationLevel.URGENT_RESPONSE,
            ErrorSeverity.FATAL: EscalationLevel.EMERGENCY,
            "frequency_threshold": 5
        }
        
        for category in ErrorCategory:
            if category in escalation_matrix:
                rules = default_rules.copy()
                rules.update(escalation_matrix[category])
                self.escalation_rules[category] = rules
            else:
                self.escalation_rules[category] = default_rules

    async def _initialize_critical_services(self) -> None:
        """Initialize list of critical services for graceful degradation."""
        self.critical_services = {
            "agent_orchestrator",
            "enhanced_parallel_executor", 
            "graphrag_validation",
            "agent_action_logger",
            "database_connection",
            "authentication_service"
        }

    async def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        agent_type: Optional[AgentType] = None,
        task_id: Optional[str] = None,
        workflow_id: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Comprehensive error handling with classification and recovery.
        
        Args:
            error: The exception that occurred
            context: Additional context about the error
            agent_type: Agent type if applicable
            task_id: Task ID if applicable
            workflow_id: Workflow ID if applicable
            
        Returns:
            Tuple of (recovery_success, recovery_details)
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Classify the error
        category, severity = await self._classify_error(error, context)
        
        # Create error event
        error_event = ErrorEvent(
            event_id=f"error_{int(time.time() * 1000)}_{hash(str(error)) % 10000}",
            timestamp=datetime.utcnow(),
            category=category,
            severity=severity,
            agent_type=agent_type,
            task_id=task_id,
            workflow_id=workflow_id,
            error_message=str(error),
            stack_trace=self._get_stack_trace(error),
            context=context,
            recovery_strategy=await self._select_recovery_strategy(category, severity, context)
        )
        
        # Log the error event
        await self._log_error_event(error_event)
        
        # Check circuit breaker
        circuit_key = f"{category.value}_{agent_type.value if agent_type else 'system'}"
        if await self._should_open_circuit(circuit_key, category):
            error_event.recovery_strategy = RecoveryStrategy.CIRCUIT_BREAKER
            await self._open_circuit_breaker(circuit_key, error_event)
            return False, {"strategy": "circuit_breaker_open", "event_id": error_event.event_id}
        
        # Track active error
        self.active_errors[error_event.event_id] = error_event
        
        # Execute recovery strategy
        recovery_success, recovery_details = await self._execute_recovery(error_event)
        
        # Update error event with results
        error_event.recovery_success = recovery_success
        error_event.resolution_time_ms = int((datetime.utcnow() - error_event.timestamp).total_seconds() * 1000)
        
        # Update metrics and history
        await self._update_error_metrics(error_event)
        self.error_history.append(error_event)
        
        # Handle escalation if recovery failed
        if not recovery_success:
            await self._handle_escalation(error_event)
        
        # Clean up active errors
        if error_event.event_id in self.active_errors:
            del self.active_errors[error_event.event_id]
        
        return recovery_success, recovery_details

    async def _classify_error(self, error: Exception, context: Dict[str, Any]) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Classify error category and severity using pattern matching and context."""
        
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Category classification
        category = ErrorCategory.UNKNOWN_ERROR
        
        # Agent-related errors
        if any(pattern in error_str for pattern in ["agent", "timeout", "unavailable"]):
            if "timeout" in error_str:
                category = ErrorCategory.AGENT_TIMEOUT
            elif "unavailable" in error_str or "connection" in error_str:
                category = ErrorCategory.AGENT_UNAVAILABLE
            elif "crash" in error_str or "failed" in error_str:
                category = ErrorCategory.AGENT_CRASH
        
        # Resource errors
        elif any(pattern in error_str for pattern in ["memory", "cpu", "resource", "disk"]):
            if "memory" in error_str:
                category = ErrorCategory.MEMORY_ERROR
            elif "cpu" in error_str:
                category = ErrorCategory.CPU_OVERLOAD
            elif "disk" in error_str:
                category = ErrorCategory.DISK_FULL
            else:
                category = ErrorCategory.RESOURCE_EXHAUSTION
        
        # Network and external service errors
        elif any(pattern in error_str for pattern in ["network", "connection", "database", "llm", "api"]):
            if "network" in error_str or "connection" in error_str:
                category = ErrorCategory.NETWORK_ERROR
            elif "database" in error_str or "db" in error_str:
                category = ErrorCategory.DATABASE_ERROR
            elif "llm" in error_str or "openai" in error_str or "anthropic" in error_str:
                category = ErrorCategory.LLM_ERROR
        
        # Validation and data errors
        elif any(pattern in error_str for pattern in ["validation", "schema", "serialize", "json"]):
            if "validation" in error_str:
                category = ErrorCategory.VALIDATION_FAILURE
            elif "schema" in error_str:
                category = ErrorCategory.SCHEMA_MISMATCH
            elif "serialize" in error_str or "json" in error_str:
                category = ErrorCategory.SERIALIZATION_ERROR
        
        # Authorization and authentication
        elif any(pattern in error_str for pattern in ["permission", "auth", "unauthorized", "forbidden"]):
            if "auth" in error_str:
                category = ErrorCategory.AUTHENTICATION_FAILURE
            else:
                category = ErrorCategory.PERMISSION_DENIED
        
        # Severity classification
        severity = ErrorSeverity.MEDIUM  # Default
        
        # Critical indicators
        if any(pattern in error_str for pattern in ["critical", "fatal", "crash", "corruption"]):
            severity = ErrorSeverity.CRITICAL
        elif any(pattern in error_str for pattern in ["timeout", "unavailable", "failed"]):
            severity = ErrorSeverity.HIGH
        elif any(pattern in error_str for pattern in ["warning", "retry", "temporary"]):
            severity = ErrorSeverity.LOW
        
        # Context-based severity adjustment
        if context.get("is_critical_path", False):
            severity = min(ErrorSeverity.CRITICAL, ErrorSeverity(max(severity.value, ErrorSeverity.HIGH.value)))
        
        if context.get("retry_count", 0) > 3:
            severity = ErrorSeverity.HIGH
        
        return category, severity

    async def _select_recovery_strategy(
        self,
        category: ErrorCategory,
        severity: ErrorSeverity,
        context: Dict[str, Any]
    ) -> RecoveryStrategy:
        """Select optimal recovery strategy based on error classification and context."""
        
        # Get predefined strategy for category
        if category in self.recovery_strategies:
            base_strategy = self.recovery_strategies[category].strategy
        else:
            base_strategy = RecoveryStrategy.RETRY
        
        # Adjust strategy based on severity
        if severity == ErrorSeverity.CRITICAL:
            if base_strategy == RecoveryStrategy.RETRY:
                return RecoveryStrategy.GRACEFUL_DEGRADATION
            elif base_strategy == RecoveryStrategy.RETRY_WITH_FALLBACK:
                return RecoveryStrategy.TASK_REDISTRIBUTION
        
        elif severity == ErrorSeverity.FATAL:
            return RecoveryStrategy.MANUAL_INTERVENTION
        
        # Context-based adjustments
        if context.get("resource_constrained", False):
            if base_strategy in [RecoveryStrategy.RETRY, RecoveryStrategy.RETRY_WITH_FALLBACK]:
                return RecoveryStrategy.GRACEFUL_DEGRADATION
        
        if context.get("high_priority", False):
            if base_strategy == RecoveryStrategy.IGNORE:
                return RecoveryStrategy.RETRY
        
        return base_strategy

    async def _execute_recovery(self, error_event: ErrorEvent) -> Tuple[bool, Dict[str, Any]]:
        """Execute the selected recovery strategy."""
        
        strategy = error_event.recovery_strategy
        recovery_start = time.time()
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                success, details = await self._execute_retry_recovery(error_event)
            elif strategy == RecoveryStrategy.RETRY_WITH_FALLBACK:
                success, details = await self._execute_fallback_recovery(error_event)
            elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                success, details = await self._execute_circuit_breaker_recovery(error_event)
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                success, details = await self._execute_degradation_recovery(error_event)
            elif strategy == RecoveryStrategy.TASK_REDISTRIBUTION:
                success, details = await self._execute_redistribution_recovery(error_event)
            elif strategy == RecoveryStrategy.RESOURCE_SCALING:
                success, details = await self._execute_scaling_recovery(error_event)
            elif strategy == RecoveryStrategy.SYSTEM_RESTART:
                success, details = await self._execute_restart_recovery(error_event)
            elif strategy == RecoveryStrategy.MANUAL_INTERVENTION:
                success, details = await self._execute_manual_intervention(error_event)
            else:
                success, details = False, {"error": f"Unknown recovery strategy: {strategy}"}
            
            # Record recovery metrics
            recovery_time = int((time.time() - recovery_start) * 1000)
            details["recovery_time_ms"] = recovery_time
            details["strategy"] = strategy.value
            
            # Log recovery attempt
            await self.agent_logger.log_action(
                ActionType.DECISION_POINT,
                log_level=LogLevel.INFO if success else LogLevel.WARN,
                task_id=error_event.task_id,
                correlation_id=error_event.event_id,
                execution_stage="error_recovery_executed",
                decision_context={
                    "recovery_strategy": strategy.value,
                    "error_category": error_event.category.value,
                    "error_severity": error_event.severity.value,
                    "success": success
                },
                duration_ms=recovery_time,
                reasoning=f"Executed {strategy.value} recovery strategy for {error_event.category.value} error",
                tags=["error_handling", "recovery", strategy.value, "success" if success else "failure"]
            )
            
            return success, details
            
        except Exception as e:
            logger.error(f"Recovery execution failed: {str(e)}")
            return False, {"error": f"Recovery execution failed: {str(e)}"}

    async def _execute_retry_recovery(self, error_event: ErrorEvent) -> Tuple[bool, Dict[str, Any]]:
        """Execute simple retry recovery with exponential backoff."""
        
        plan = self.recovery_strategies.get(error_event.category)
        max_attempts = plan.max_attempts if plan else 3
        
        for attempt in range(max_attempts):
            try:
                # Simulate retry attempt - in real implementation, this would retry the original operation
                await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                
                # For simulation, determine success based on attempt number and error type
                success_probability = 0.3 + (attempt * 0.2)  # Increasing chance with attempts
                if error_event.category in [ErrorCategory.NETWORK_ERROR, ErrorCategory.LLM_ERROR]:
                    success_probability += 0.2  # Higher success rate for transient errors
                
                if attempt >= max_attempts - 1 or success_probability > 0.7:
                    return True, {"attempts": attempt + 1, "success_probability": success_probability}
                    
            except Exception as e:
                if attempt == max_attempts - 1:
                    return False, {"attempts": attempt + 1, "final_error": str(e)}
                continue
        
        return False, {"attempts": max_attempts, "reason": "max_attempts_exceeded"}

    async def _execute_fallback_recovery(self, error_event: ErrorEvent) -> Tuple[bool, Dict[str, Any]]:
        """Execute retry with fallback options recovery."""
        
        plan = self.recovery_strategies.get(error_event.category)
        fallback_options = plan.fallback_options if plan else ["simplified_task"]
        
        # Try primary approach first
        success, details = await self._execute_retry_recovery(error_event)
        if success:
            return success, {**details, "fallback_used": None}
        
        # Try fallback options
        for fallback in fallback_options:
            try:
                await asyncio.sleep(0.1)  # Brief pause between attempts
                
                # Simulate fallback attempt
                if fallback == "alternative_agent":
                    # Would switch to different agent type
                    success_rate = 0.8
                elif fallback == "simplified_task":
                    # Would use simpler version of task
                    success_rate = 0.9
                elif fallback == "cache_fallback":
                    # Would use cached results
                    success_rate = 0.7
                else:
                    success_rate = 0.6
                
                if success_rate > 0.7:  # Simulate success
                    return True, {"fallback_used": fallback, "success_rate": success_rate}
                    
            except Exception as e:
                continue
        
        return False, {"fallback_options_tried": fallback_options, "all_failed": True}

    async def _execute_circuit_breaker_recovery(self, error_event: ErrorEvent) -> Tuple[bool, Dict[str, Any]]:
        """Execute circuit breaker recovery (fail fast)."""
        return False, {
            "reason": "circuit_breaker_open",
            "estimated_recovery_time": 30,
            "alternative_agents_available": True
        }

    async def _execute_degradation_recovery(self, error_event: ErrorEvent) -> Tuple[bool, Dict[str, Any]]:
        """Execute graceful degradation recovery."""
        
        # Determine degradation level based on error severity
        if error_event.severity == ErrorSeverity.CRITICAL:
            degradation_level = 3  # Severe degradation
        elif error_event.severity == ErrorSeverity.HIGH:
            degradation_level = 2  # Moderate degradation
        else:
            degradation_level = 1  # Minor degradation
        
        # Apply degradation
        service_key = f"{error_event.agent_type.value if error_event.agent_type else 'system'}"
        self.degradation_levels[service_key] = degradation_level
        
        # Update system health score
        self.system_health_score = max(10.0, self.system_health_score - (degradation_level * 10))
        
        return True, {
            "degradation_level": degradation_level,
            "system_health_score": self.system_health_score,
            "services_affected": [service_key]
        }

    async def _execute_redistribution_recovery(self, error_event: ErrorEvent) -> Tuple[bool, Dict[str, Any]]:
        """Execute task redistribution recovery."""
        
        # Simulate task redistribution
        available_agents = [agent for agent in AgentType if agent != error_event.agent_type][:3]
        
        return True, {
            "redistributed_to": [agent.value for agent in available_agents],
            "load_distribution": "balanced",
            "estimated_completion_delay": "5-10 minutes"
        }

    async def _execute_scaling_recovery(self, error_event: ErrorEvent) -> Tuple[bool, Dict[str, Any]]:
        """Execute resource scaling recovery."""
        
        # Simulate resource scaling
        scale_factor = 1.5  # 50% increase
        
        return True, {
            "scale_factor": scale_factor,
            "resources_scaled": ["cpu", "memory", "agent_instances"],
            "estimated_completion_time": "2-3 minutes"
        }

    async def _execute_restart_recovery(self, error_event: ErrorEvent) -> Tuple[bool, Dict[str, Any]]:
        """Execute system restart recovery."""
        
        # This would trigger actual component restart in real implementation
        return True, {
            "components_restarted": ["agent_pool", "executor"],
            "restart_time": "1-2 minutes",
            "data_loss": False
        }

    async def _execute_manual_intervention(self, error_event: ErrorEvent) -> Tuple[bool, Dict[str, Any]]:
        """Execute manual intervention recovery (escalate to humans)."""
        
        # Escalate to humans
        await self._escalate_error(error_event, EscalationLevel.URGENT_RESPONSE)
        
        return False, {
            "escalated": True,
            "escalation_level": "urgent_response",
            "estimated_response_time": "15-30 minutes",
            "support_ticket_id": f"TICKET_{error_event.event_id}"
        }

    async def _log_error_event(self, error_event: ErrorEvent) -> None:
        """Log error event with comprehensive details."""
        
        await self.agent_logger.log_action(
            ActionType.VALIDATION_FAILED if "validation" in error_event.category.value else ActionType.TASK_LIFECYCLE,
            log_level=self._severity_to_log_level(error_event.severity),
            task_id=error_event.task_id,
            session_id=error_event.workflow_id,
            correlation_id=error_event.event_id,
            agent_type=error_event.agent_type,
            execution_stage="error_detected",
            error_message=error_event.error_message,
            error_category=error_event.category.value,
            input_data=error_event.context,
            decision_context={
                "error_category": error_event.category.value,
                "error_severity": error_event.severity.value,
                "recovery_strategy": error_event.recovery_strategy.value,
                "escalation_level": error_event.escalation_level.value
            },
            reasoning=f"Detected {error_event.severity.value} severity {error_event.category.value} error",
            tags=["error_handling", "error_detected", error_event.category.value, error_event.severity.value]
        )

    def _severity_to_log_level(self, severity: ErrorSeverity) -> LogLevel:
        """Convert error severity to log level."""
        mapping = {
            ErrorSeverity.LOW: LogLevel.INFO,
            ErrorSeverity.MEDIUM: LogLevel.WARN,
            ErrorSeverity.HIGH: LogLevel.ERROR,
            ErrorSeverity.CRITICAL: LogLevel.ERROR,
            ErrorSeverity.FATAL: LogLevel.ERROR
        }
        return mapping.get(severity, LogLevel.WARN)

    async def _should_open_circuit(self, circuit_key: str, category: ErrorCategory) -> bool:
        """Check if circuit breaker should be opened."""
        
        if circuit_key not in self.circuit_breakers:
            self.circuit_breakers[circuit_key] = {
                "state": CircuitState.CLOSED,
                "failure_count": 0,
                "last_failure": None,
                "half_open_calls": 0
            }
        
        circuit = self.circuit_breakers[circuit_key]
        thresholds = self.circuit_breaker_thresholds.get(category, {"failure_threshold": 5})
        
        circuit["failure_count"] += 1
        circuit["last_failure"] = datetime.utcnow()
        
        if circuit["failure_count"] >= thresholds["failure_threshold"]:
            circuit["state"] = CircuitState.OPEN
            return True
        
        return False

    async def _open_circuit_breaker(self, circuit_key: str, error_event: ErrorEvent) -> None:
        """Open circuit breaker and log the event."""
        
        await self.agent_logger.log_action(
            ActionType.DECISION_POINT,
            log_level=LogLevel.WARN,
            task_id=error_event.task_id,
            correlation_id=error_event.event_id,
            execution_stage="circuit_breaker_opened",
            decision_context={
                "circuit_key": circuit_key,
                "failure_count": self.circuit_breakers[circuit_key]["failure_count"],
                "error_category": error_event.category.value
            },
            reasoning=f"Opened circuit breaker for {circuit_key} due to repeated {error_event.category.value} errors",
            tags=["error_handling", "circuit_breaker", "opened", error_event.category.value]
        )

    async def _update_error_metrics(self, error_event: ErrorEvent) -> None:
        """Update error metrics for analytics."""
        
        category = error_event.category
        if category not in self.error_metrics:
            self.error_metrics[category] = ErrorMetrics()
        
        metrics = self.error_metrics[category]
        metrics.error_count += 1
        
        if metrics.first_occurrence is None:
            metrics.first_occurrence = error_event.timestamp
        
        metrics.last_occurrence = error_event.timestamp
        
        # Calculate frequency (errors per hour)
        if metrics.first_occurrence:
            time_span_hours = (error_event.timestamp - metrics.first_occurrence).total_seconds() / 3600
            if time_span_hours > 0:
                metrics.frequency_per_hour = metrics.error_count / time_span_hours
        
        # Update recovery metrics
        if error_event.recovery_success:
            current_rate = metrics.recovery_success_rate
            metrics.recovery_success_rate = (current_rate * 0.9 + 0.1) if current_rate > 0 else 0.1
        
        if error_event.resolution_time_ms:
            current_avg = metrics.average_recovery_time_ms
            metrics.average_recovery_time_ms = (
                current_avg * 0.8 + error_event.resolution_time_ms * 0.2
            ) if current_avg > 0 else error_event.resolution_time_ms

    async def _handle_escalation(self, error_event: ErrorEvent) -> None:
        """Handle error escalation based on severity and rules."""
        
        rules = self.escalation_rules.get(error_event.category, {})
        escalation_level = rules.get(error_event.severity, EscalationLevel.LOG_WARNING)
        
        # Check frequency-based escalation
        frequency_threshold = rules.get("frequency_threshold", 5)
        metrics = self.error_metrics.get(error_event.category)
        
        if metrics and metrics.frequency_per_hour > frequency_threshold:
            # Escalate one level higher due to frequency
            level_order = [
                EscalationLevel.AUTO_RECOVER,
                EscalationLevel.LOG_WARNING,
                EscalationLevel.ADMIN_ALERT,
                EscalationLevel.URGENT_RESPONSE,
                EscalationLevel.EMERGENCY
            ]
            current_index = level_order.index(escalation_level)
            if current_index < len(level_order) - 1:
                escalation_level = level_order[current_index + 1]
        
        error_event.escalation_level = escalation_level
        await self._escalate_error(error_event, escalation_level)

    async def _escalate_error(self, error_event: ErrorEvent, level: EscalationLevel) -> None:
        """Execute error escalation."""
        
        escalation_data = {
            "timestamp": datetime.utcnow(),
            "error_event": error_event,
            "escalation_level": level,
            "escalation_reason": "recovery_failure"
        }
        
        self.escalation_history.append(escalation_data)
        
        # Log escalation
        await self.agent_logger.log_action(
            ActionType.DECISION_POINT,
            log_level=LogLevel.ERROR if level in [EscalationLevel.URGENT_RESPONSE, EscalationLevel.EMERGENCY] else LogLevel.WARN,
            task_id=error_event.task_id,
            correlation_id=error_event.event_id,
            execution_stage="error_escalated",
            decision_context={
                "escalation_level": level.value,
                "error_category": error_event.category.value,
                "error_severity": error_event.severity.value,
                "recovery_attempts": error_event.recovery_attempts
            },
            reasoning=f"Escalated {error_event.category.value} error to {level.value} due to recovery failure",
            tags=["error_handling", "escalation", level.value, error_event.category.value]
        )
        
        # Execute escalation actions based on level
        if level == EscalationLevel.ADMIN_ALERT:
            await self._send_admin_alert(error_event)
        elif level == EscalationLevel.URGENT_RESPONSE:
            await self._trigger_urgent_response(error_event)
        elif level == EscalationLevel.EMERGENCY:
            await self._activate_emergency_protocols(error_event)

    async def _send_admin_alert(self, error_event: ErrorEvent) -> None:
        """Send alert to administrators."""
        # In real implementation, this would send email/Slack/etc.
        logger.warning(f"ADMIN ALERT: {error_event.category.value} error requires attention")

    async def _trigger_urgent_response(self, error_event: ErrorEvent) -> None:
        """Trigger urgent response protocols."""
        # In real implementation, this would page on-call engineers
        logger.error(f"URGENT RESPONSE: {error_event.category.value} error needs immediate attention")

    async def _activate_emergency_protocols(self, error_event: ErrorEvent) -> None:
        """Activate emergency protocols."""
        # In real implementation, this would trigger emergency procedures
        logger.critical(f"EMERGENCY: {error_event.category.value} error activated emergency protocols")

    def _get_stack_trace(self, error: Exception) -> Optional[str]:
        """Get stack trace from exception."""
        import traceback
        try:
            return traceback.format_exc()
        except:
            return None

    async def get_error_analytics(self) -> Dict[str, Any]:
        """Get comprehensive error analytics."""
        
        total_errors = len(self.error_history)
        if total_errors == 0:
            return {"total_errors": 0, "message": "No errors recorded"}
        
        # Calculate metrics
        recent_errors = [e for e in self.error_history if (datetime.utcnow() - e.timestamp).total_seconds() < 3600]
        
        category_stats = defaultdict(int)
        severity_stats = defaultdict(int)
        recovery_stats = defaultdict(int)
        
        for error in self.error_history:
            category_stats[error.category.value] += 1
            severity_stats[error.severity.value] += 1
            recovery_stats[error.recovery_strategy.value] += 1
        
        # Calculate success rates
        successful_recoveries = sum(1 for e in self.error_history if e.recovery_success)
        recovery_success_rate = (successful_recoveries / total_errors) * 100
        
        # Average recovery time
        recovery_times = [e.resolution_time_ms for e in self.error_history if e.resolution_time_ms]
        avg_recovery_time = statistics.mean(recovery_times) if recovery_times else 0
        
        return {
            "total_errors": total_errors,
            "recent_errors_1h": len(recent_errors),
            "recovery_success_rate": recovery_success_rate,
            "average_recovery_time_ms": avg_recovery_time,
            "system_health_score": self.system_health_score,
            "category_distribution": dict(category_stats),
            "severity_distribution": dict(severity_stats),
            "recovery_strategy_distribution": dict(recovery_stats),
            "active_errors": len(self.active_errors),
            "circuit_breakers": {
                key: circuit["state"].value if hasattr(circuit["state"], "value") else circuit["state"]
                for key, circuit in self.circuit_breakers.items()
            },
            "degradation_levels": dict(self.degradation_levels)
        }

    async def get_system_health_status(self) -> Dict[str, Any]:
        """Get current system health status."""
        
        # Calculate component health scores
        component_health = {}
        for service in self.critical_services:
            degradation = self.degradation_levels.get(service, 0)
            health_score = max(0, 100 - (degradation * 20))
            component_health[service] = health_score
        
        # Overall system health
        if component_health:
            overall_health = statistics.mean(component_health.values())
        else:
            overall_health = self.system_health_score
        
        # Determine health status
        if overall_health >= 90:
            status = "healthy"
        elif overall_health >= 70:
            status = "degraded"
        elif overall_health >= 50:
            status = "warning"
        else:
            status = "critical"
        
        return {
            "overall_health_score": overall_health,
            "status": status,
            "component_health": component_health,
            "active_errors": len(self.active_errors),
            "recent_errors": len([e for e in self.error_history if (datetime.utcnow() - e.timestamp).total_seconds() < 300]),
            "circuit_breakers_open": len([c for c in self.circuit_breakers.values() if c["state"] == CircuitState.OPEN]),
            "degraded_services": list(self.degradation_levels.keys()),
            "last_updated": datetime.utcnow().isoformat()
        }


# Global error handling service instance
_error_handling_service: Optional[ErrorHandlingService] = None


async def get_error_handling_service() -> ErrorHandlingService:
    """Get the global error handling service instance."""
    global _error_handling_service
    
    if not _error_handling_service:
        _error_handling_service = ErrorHandlingService()
        await _error_handling_service.initialize()
    
    return _error_handling_service


# Convenience function for quick error handling
async def handle_error(
    error: Exception,
    context: Dict[str, Any] = None,
    agent_type: Optional[AgentType] = None,
    task_id: Optional[str] = None,
    workflow_id: Optional[str] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience function for handling errors with the global service.
    
    Returns:
        Tuple of (recovery_success, recovery_details)
    """
    service = await get_error_handling_service()
    return await service.handle_error(
        error=error,
        context=context or {},
        agent_type=agent_type,
        task_id=task_id,
        workflow_id=workflow_id
    )