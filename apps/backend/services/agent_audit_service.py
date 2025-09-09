"""
Agent Audit Service for Comprehensive Log Analysis and Querying

Advanced querying, analysis, and audit capabilities for agent action logs.
Provides business intelligence, compliance reporting, and operational insights.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
from collections import defaultdict, Counter

import structlog
from services.agent_action_logger import (
    get_agent_logger, AgentActionLogger, ActionType, LogLevel, AgentActionLog
)
from services.agent_orchestrator import AgentType
from core.redis import get_redis_client
from core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class AuditReportType(Enum):
    """Types of audit reports that can be generated."""
    PERFORMANCE_SUMMARY = "performance_summary"
    ERROR_ANALYSIS = "error_analysis"
    AGENT_UTILIZATION = "agent_utilization"
    QUALITY_METRICS = "quality_metrics"
    COMPLIANCE_REPORT = "compliance_report"
    SECURITY_AUDIT = "security_audit"
    RESOURCE_USAGE = "resource_usage"
    OPERATIONAL_SUMMARY = "operational_summary"
    TREND_ANALYSIS = "trend_analysis"
    CUSTOM_ANALYSIS = "custom_analysis"


class TimeFrame(Enum):
    """Time frames for analysis."""
    LAST_HOUR = "last_hour"
    LAST_24_HOURS = "last_24_hours"
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    CUSTOM_RANGE = "custom_range"


@dataclass
class AuditQuery:
    """Comprehensive query for agent logs."""
    # Time filtering
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    time_frame: Optional[TimeFrame] = None
    
    # Entity filtering
    agent_types: Optional[List[AgentType]] = None
    action_types: Optional[List[ActionType]] = None
    log_levels: Optional[List[LogLevel]] = None
    task_ids: Optional[List[str]] = None
    session_ids: Optional[List[str]] = None
    correlation_ids: Optional[List[str]] = None
    
    # Content filtering
    error_categories: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    min_duration_ms: Optional[float] = None
    max_duration_ms: Optional[float] = None
    min_confidence_score: Optional[float] = None
    max_confidence_score: Optional[float] = None
    
    # Quality filtering
    has_errors: Optional[bool] = None
    has_retries: Optional[bool] = None
    has_validation_results: Optional[bool] = None
    min_hallucination_score: Optional[float] = None
    max_hallucination_score: Optional[float] = None
    
    # Text search
    search_text: Optional[str] = None
    search_fields: Optional[List[str]] = None
    
    # Pagination and sorting
    limit: int = 100
    offset: int = 0
    sort_by: str = "timestamp"
    sort_order: str = "desc"  # asc or desc
    
    # Aggregation options
    group_by: Optional[List[str]] = None
    include_aggregations: bool = False


@dataclass
class AuditResult:
    """Result of an audit query."""
    query: AuditQuery
    logs: List[Dict[str, Any]]
    total_count: int
    filtered_count: int
    
    # Aggregations
    aggregations: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    query_time_ms: float = 0
    cache_hit: bool = False
    
    # Analysis insights
    insights: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceMetrics:
    """Performance analysis results."""
    # Execution metrics
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    success_rate: float = 0.0
    
    # Timing metrics
    avg_execution_time_ms: float = 0.0
    p50_execution_time_ms: float = 0.0
    p95_execution_time_ms: float = 0.0
    p99_execution_time_ms: float = 0.0
    avg_queue_wait_ms: float = 0.0
    
    # Resource metrics
    avg_cpu_usage: float = 0.0
    avg_memory_usage: float = 0.0
    peak_concurrency: int = 0
    
    # Quality metrics
    avg_confidence_score: float = 0.0
    retry_rate: float = 0.0
    circuit_breaker_activations: int = 0
    
    # Validation metrics
    validation_success_rate: float = 0.0
    avg_hallucination_score: float = 0.0


@dataclass
class ErrorAnalysis:
    """Error analysis results."""
    total_errors: int = 0
    error_categories: Dict[str, int] = field(default_factory=dict)
    error_trends: Dict[str, List[int]] = field(default_factory=dict)
    most_common_errors: List[Tuple[str, int]] = field(default_factory=list)
    error_recovery_rate: float = 0.0
    mean_time_to_recovery_ms: float = 0.0
    
    # Agent-specific errors
    agent_error_rates: Dict[str, float] = field(default_factory=dict)
    
    # Patterns
    recurring_patterns: List[Dict[str, Any]] = field(default_factory=list)
    critical_issues: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentUtilizationReport:
    """Agent utilization analysis."""
    agent_usage_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    load_balancing_efficiency: float = 0.0
    agent_performance_ranking: List[Tuple[str, float]] = field(default_factory=list)
    underutilized_agents: List[str] = field(default_factory=list)
    overutilized_agents: List[str] = field(default_factory=list)
    
    # Pool efficiency
    pool_hit_rate: float = 0.0
    pool_creation_rate: float = 0.0
    
    # Recommendations
    optimization_recommendations: List[str] = field(default_factory=list)


class AgentAuditService:
    """
    Comprehensive audit service for agent action logs.
    
    Provides advanced querying, analysis, reporting, and compliance
    capabilities for agent operations and quality assurance.
    """
    
    def __init__(self):
        self.agent_logger: Optional[AgentActionLogger] = None
        self.redis_client = None
        self.cache_ttl = 300  # 5 minutes cache
        self.is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the audit service."""
        try:
            self.agent_logger = await get_agent_logger()
            self.redis_client = await get_redis_client()
            self.is_initialized = True
            
            logger.info("Agent Audit Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Agent Audit Service: {str(e)}")
            raise
    
    async def query_logs(self, query: AuditQuery) -> AuditResult:
        """Execute a comprehensive log query with analysis."""
        
        if not self.is_initialized:
            await self.initialize()
        
        start_time = datetime.now()
        
        # Check cache first
        cache_key = self._generate_cache_key(query)
        cached_result = await self._get_cached_result(cache_key)
        if cached_result:
            cached_result.cache_hit = True
            return cached_result
        
        # Apply time frame if specified
        if query.time_frame:
            query.start_time, query.end_time = self._resolve_time_frame(query.time_frame)
        
        # Get filtered logs
        logs = await self._execute_query(query)
        
        # Calculate aggregations if requested
        aggregations = None
        if query.include_aggregations:
            aggregations = await self._calculate_aggregations(logs, query)
        
        # Generate insights
        insights = await self._generate_insights(logs, query)
        
        # Create result
        result = AuditResult(
            query=query,
            logs=logs[query.offset:query.offset + query.limit],
            total_count=len(logs),
            filtered_count=len(logs),
            aggregations=aggregations,
            query_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            insights=insights
        )
        
        # Cache result
        await self._cache_result(cache_key, result)
        
        return result
    
    async def generate_performance_report(
        self,
        time_frame: TimeFrame = TimeFrame.LAST_24_HOURS,
        agent_types: Optional[List[AgentType]] = None
    ) -> PerformanceMetrics:
        """Generate comprehensive performance report."""
        
        query = AuditQuery(
            time_frame=time_frame,
            agent_types=agent_types,
            include_aggregations=True,
            limit=10000  # Get more data for analysis
        )
        
        result = await self.query_logs(query)
        logs = result.logs
        
        # Calculate performance metrics
        execution_times = []
        queue_wait_times = []
        confidence_scores = []
        cpu_usage = []
        memory_usage = []
        retry_counts = 0
        circuit_breaker_events = 0
        validation_results = []
        hallucination_scores = []
        
        successful = 0
        failed = 0
        max_concurrency = 0
        
        for log_entry in logs:
            # Execution status
            if log_entry.get('action_type') in ['task_completed', 'agent_execution_completed']:
                successful += 1
                if log_entry.get('duration_ms'):
                    execution_times.append(log_entry['duration_ms'])
            elif log_entry.get('action_type') in ['task_failed', 'agent_execution_failed']:
                failed += 1
            
            # Queue wait times
            if log_entry.get('queue_wait_ms'):
                queue_wait_times.append(log_entry['queue_wait_ms'])
            
            # Confidence scores
            if log_entry.get('confidence_score'):
                confidence_scores.append(log_entry['confidence_score'])
            
            # Resource usage
            resource_usage = log_entry.get('resource_usage', {})
            if isinstance(resource_usage, dict):
                if resource_usage.get('cpu_usage_percent'):
                    cpu_usage.append(resource_usage['cpu_usage_percent'])
                if resource_usage.get('memory_usage_mb'):
                    memory_usage.append(resource_usage['memory_usage_mb'])
                if resource_usage.get('active_tasks'):
                    max_concurrency = max(max_concurrency, resource_usage['active_tasks'])
            
            # Retry events
            if log_entry.get('action_type') == 'task_retried':
                retry_counts += 1
            
            # Circuit breaker events
            if log_entry.get('action_type') == 'circuit_breaker_triggered':
                circuit_breaker_events += 1
            
            # Validation results
            if log_entry.get('validation_result'):
                validation_results.append(log_entry['validation_result'])
            
            # Hallucination scores
            if log_entry.get('hallucination_score'):
                hallucination_scores.append(log_entry['hallucination_score'])
        
        # Calculate metrics
        total_executions = successful + failed
        success_rate = (successful / total_executions * 100) if total_executions > 0 else 0
        
        return PerformanceMetrics(
            total_executions=total_executions,
            successful_executions=successful,
            failed_executions=failed,
            success_rate=success_rate,
            avg_execution_time_ms=statistics.mean(execution_times) if execution_times else 0,
            p50_execution_time_ms=statistics.median(execution_times) if execution_times else 0,
            p95_execution_time_ms=self._percentile(execution_times, 95) if execution_times else 0,
            p99_execution_time_ms=self._percentile(execution_times, 99) if execution_times else 0,
            avg_queue_wait_ms=statistics.mean(queue_wait_times) if queue_wait_times else 0,
            avg_cpu_usage=statistics.mean(cpu_usage) if cpu_usage else 0,
            avg_memory_usage=statistics.mean(memory_usage) if memory_usage else 0,
            peak_concurrency=max_concurrency,
            avg_confidence_score=statistics.mean(confidence_scores) if confidence_scores else 0,
            retry_rate=(retry_counts / total_executions * 100) if total_executions > 0 else 0,
            circuit_breaker_activations=circuit_breaker_events,
            validation_success_rate=len([v for v in validation_results if v.get('status') == 'passed']) / len(validation_results) * 100 if validation_results else 0,
            avg_hallucination_score=statistics.mean(hallucination_scores) if hallucination_scores else 0
        )
    
    async def generate_error_analysis(
        self,
        time_frame: TimeFrame = TimeFrame.LAST_24_HOURS,
        agent_types: Optional[List[AgentType]] = None
    ) -> ErrorAnalysis:
        """Generate comprehensive error analysis report."""
        
        query = AuditQuery(
            time_frame=time_frame,
            agent_types=agent_types,
            action_types=[ActionType.TASK_FAILED, ActionType.AGENT_EXECUTION_FAILED, ActionType.VALIDATION_FAILED],
            log_levels=[LogLevel.ERROR, LogLevel.CRITICAL],
            limit=10000
        )
        
        result = await self.query_logs(query)
        error_logs = result.logs
        
        # Also get recovery events
        recovery_query = AuditQuery(
            time_frame=time_frame,
            agent_types=agent_types,
            action_types=[ActionType.ERROR_RECOVERY, ActionType.CIRCUIT_BREAKER_RECOVERED],
            limit=10000
        )
        recovery_result = await self.query_logs(recovery_query)
        recovery_logs = recovery_result.logs
        
        # Analyze errors
        error_categories = Counter()
        error_messages = Counter()
        agent_error_counts = defaultdict(int)
        agent_total_counts = defaultdict(int)
        
        # Track time patterns (hourly buckets for trend analysis)
        error_trends = defaultdict(lambda: [0] * 24)  # 24 hours
        
        recovery_times = []
        
        for log_entry in error_logs:
            # Error categories
            category = log_entry.get('error_category', 'unknown')
            error_categories[category] += 1
            
            # Error messages
            error_msg = log_entry.get('error_message', '')[:100]  # First 100 chars
            error_messages[error_msg] += 1
            
            # Agent-specific errors
            agent_type = log_entry.get('agent_type')
            if agent_type:
                agent_error_counts[agent_type] += 1
            
            # Time trends
            timestamp = datetime.fromisoformat(log_entry.get('timestamp', '').replace('Z', '+00:00'))
            hour = timestamp.hour
            error_trends[category][hour] += 1
        
        # Calculate recovery metrics
        for recovery_log in recovery_logs:
            duration = recovery_log.get('duration_ms', 0)
            if duration > 0:
                recovery_times.append(duration)
        
        # Get total counts for agent error rates (would need additional query in real implementation)
        # For now, using error counts as approximation
        agent_error_rates = {}
        for agent_type, error_count in agent_error_counts.items():
            # This would be calculated properly with total execution counts
            agent_error_rates[agent_type] = error_count  # Placeholder
        
        # Identify patterns and critical issues
        recurring_patterns = []
        critical_issues = []
        
        # Find recurring error patterns
        for error_msg, count in error_messages.most_common(10):
            if count > 1:  # Recurring pattern
                recurring_patterns.append({
                    "pattern": error_msg,
                    "count": count,
                    "severity": "high" if count > 10 else "medium"
                })
        
        # Find critical issues (high-frequency errors in short time)
        for category, count in error_categories.items():
            if count > 50:  # More than 50 errors in timeframe
                critical_issues.append({
                    "category": category,
                    "count": count,
                    "severity": "critical",
                    "requires_attention": True
                })
        
        return ErrorAnalysis(
            total_errors=len(error_logs),
            error_categories=dict(error_categories),
            error_trends={k: v for k, v in error_trends.items()},
            most_common_errors=error_messages.most_common(10),
            error_recovery_rate=(len(recovery_logs) / len(error_logs) * 100) if error_logs else 0,
            mean_time_to_recovery_ms=statistics.mean(recovery_times) if recovery_times else 0,
            agent_error_rates=agent_error_rates,
            recurring_patterns=recurring_patterns,
            critical_issues=critical_issues
        )
    
    async def generate_agent_utilization_report(
        self,
        time_frame: TimeFrame = TimeFrame.LAST_24_HOURS
    ) -> AgentUtilizationReport:
        """Generate agent utilization and efficiency report."""
        
        query = AuditQuery(
            time_frame=time_frame,
            include_aggregations=True,
            limit=10000
        )
        
        result = await self.query_logs(query)
        logs = result.logs
        
        # Analyze agent usage
        agent_usage = defaultdict(lambda: {
            'executions': 0,
            'successes': 0,
            'failures': 0,
            'avg_duration': 0,
            'total_duration': 0,
            'pool_hits': 0,
            'pool_creations': 0
        })
        
        performance_scores = defaultdict(list)
        
        for log_entry in logs:
            agent_type = log_entry.get('agent_type')
            if not agent_type:
                continue
            
            action_type = log_entry.get('action_type')
            
            if action_type in ['agent_execution_completed', 'task_completed']:
                agent_usage[agent_type]['executions'] += 1
                agent_usage[agent_type]['successes'] += 1
                
                duration = log_entry.get('duration_ms', 0)
                if duration > 0:
                    agent_usage[agent_type]['total_duration'] += duration
                    performance_scores[agent_type].append(duration)
            
            elif action_type in ['agent_execution_failed', 'task_failed']:
                agent_usage[agent_type]['executions'] += 1
                agent_usage[agent_type]['failures'] += 1
            
            elif action_type == 'agent_instantiated':
                resource_usage = log_entry.get('resource_usage', {})
                if isinstance(resource_usage, dict):
                    if resource_usage.get('pooling_enabled') and resource_usage.get('pool_available', 0) > 0:
                        agent_usage[agent_type]['pool_hits'] += 1
                    else:
                        agent_usage[agent_type]['pool_creations'] += 1
        
        # Calculate derived metrics
        for agent_type, stats in agent_usage.items():
            if stats['executions'] > 0:
                stats['success_rate'] = stats['successes'] / stats['executions'] * 100
                stats['avg_duration'] = stats['total_duration'] / stats['successes'] if stats['successes'] > 0 else 0
        
        # Calculate performance ranking
        performance_ranking = []
        for agent_type, durations in performance_scores.items():
            if durations:
                avg_performance = statistics.mean(durations)
                success_rate = agent_usage[agent_type]['success_rate']
                # Combined score: lower duration (better) + higher success rate (better)
                combined_score = success_rate - (avg_performance / 1000)  # Normalize duration
                performance_ranking.append((agent_type, combined_score))
        
        performance_ranking.sort(key=lambda x: x[1], reverse=True)
        
        # Identify under/over-utilized agents
        total_executions = sum(stats['executions'] for stats in agent_usage.values())
        avg_utilization = total_executions / len(agent_usage) if agent_usage else 0
        
        underutilized = []
        overutilized = []
        
        for agent_type, stats in agent_usage.items():
            utilization_ratio = stats['executions'] / avg_utilization if avg_utilization > 0 else 0
            if utilization_ratio < 0.5:  # Less than 50% of average
                underutilized.append(agent_type)
            elif utilization_ratio > 2.0:  # More than 200% of average
                overutilized.append(agent_type)
        
        # Calculate pool efficiency
        total_pool_hits = sum(stats['pool_hits'] for stats in agent_usage.values())
        total_pool_operations = sum(stats['pool_hits'] + stats['pool_creations'] for stats in agent_usage.values())
        pool_hit_rate = (total_pool_hits / total_pool_operations * 100) if total_pool_operations > 0 else 0
        
        # Generate optimization recommendations
        recommendations = []
        
        if underutilized:
            recommendations.append(f"Consider reducing instances for underutilized agents: {', '.join(underutilized)}")
        
        if overutilized:
            recommendations.append(f"Consider increasing capacity for overutilized agents: {', '.join(overutilized)}")
        
        if pool_hit_rate < 70:
            recommendations.append(f"Pool hit rate is {pool_hit_rate:.1f}% - consider increasing pool sizes")
        
        return AgentUtilizationReport(
            agent_usage_stats=dict(agent_usage),
            load_balancing_efficiency=100 - (statistics.stdev([stats['executions'] for stats in agent_usage.values()]) / avg_utilization * 100) if len(agent_usage) > 1 and avg_utilization > 0 else 100,
            agent_performance_ranking=performance_ranking,
            underutilized_agents=underutilized,
            overutilized_agents=overutilized,
            pool_hit_rate=pool_hit_rate,
            pool_creation_rate=100 - pool_hit_rate,
            optimization_recommendations=recommendations
        )
    
    async def generate_compliance_report(
        self,
        time_frame: TimeFrame = TimeFrame.LAST_WEEK,
        include_sensitive_operations: bool = True
    ) -> Dict[str, Any]:
        """Generate compliance and audit report."""
        
        query = AuditQuery(
            time_frame=time_frame,
            log_levels=[LogLevel.AUDIT] if include_sensitive_operations else None,
            limit=10000
        )
        
        result = await self.query_logs(query)
        logs = result.logs
        
        # Compliance metrics
        audit_events = [log for log in logs if log.get('log_level') == 'audit']
        security_events = [log for log in logs if 'security' in log.get('tags', [])]
        data_access_events = [log for log in logs if 'data_access' in log.get('tags', [])]
        
        # Decision audit trail
        decision_points = [log for log in logs if log.get('action_type') == 'decision_point']
        
        # Error and exception tracking
        exceptions = [log for log in logs if log.get('log_level') in ['error', 'critical']]
        
        # Quality metrics
        validation_events = [log for log in logs if 'validation' in log.get('action_type', '')]
        hallucination_violations = [log for log in logs if log.get('hallucination_score', 0) > 0.02]  # >2% threshold
        
        return {
            "compliance_summary": {
                "total_events": len(logs),
                "audit_events": len(audit_events),
                "security_events": len(security_events),
                "data_access_events": len(data_access_events),
                "decision_points": len(decision_points),
                "exceptions": len(exceptions),
                "validation_events": len(validation_events),
                "quality_violations": len(hallucination_violations)
            },
            "audit_trail_completeness": (len(decision_points) / max(1, len([log for log in logs if 'task' in log.get('action_type', '')]))) * 100,
            "data_protection_compliance": {
                "sensitive_data_access_logged": len(data_access_events) > 0,
                "access_patterns": self._analyze_access_patterns(data_access_events),
                "anomalies_detected": len([event for event in data_access_events if event.get('anomaly_score', 0) > 0.8])
            },
            "quality_assurance": {
                "validation_coverage": (len(validation_events) / max(1, len(logs))) * 100,
                "hallucination_compliance": (1 - len(hallucination_violations) / max(1, len(validation_events))) * 100,
                "average_confidence": statistics.mean([log.get('confidence_score', 0) for log in logs if log.get('confidence_score')]) if any(log.get('confidence_score') for log in logs) else 0
            },
            "operational_integrity": {
                "error_rate": (len(exceptions) / max(1, len(logs))) * 100,
                "recovery_procedures_followed": len([log for log in logs if log.get('action_type') == 'error_recovery']),
                "circuit_breaker_activations": len([log for log in logs if log.get('action_type') == 'circuit_breaker_triggered'])
            }
        }
    
    async def search_logs_by_text(
        self,
        search_text: str,
        fields: Optional[List[str]] = None,
        time_frame: TimeFrame = TimeFrame.LAST_24_HOURS,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search logs by text content in specified fields."""
        
        query = AuditQuery(
            time_frame=time_frame,
            search_text=search_text,
            search_fields=fields or ['error_message', 'reasoning', 'decision_context'],
            limit=limit
        )
        
        result = await self.query_logs(query)
        return result.logs
    
    async def get_task_audit_trail(
        self,
        task_id: str,
        include_related_tasks: bool = True
    ) -> List[Dict[str, Any]]:
        """Get complete audit trail for a specific task."""
        
        if not self.agent_logger:
            await self.initialize()
        
        # Get direct task logs
        task_logs = await self.agent_logger.get_task_audit_trail(task_id)
        
        if include_related_tasks:
            # Find related tasks through correlation IDs
            correlation_ids = set()
            for log in task_logs:
                if log.get('correlation_id'):
                    correlation_ids.add(log['correlation_id'])
            
            # Get logs for related tasks
            for correlation_id in correlation_ids:
                related_query = AuditQuery(
                    correlation_ids=[correlation_id],
                    limit=1000
                )
                related_result = await self.query_logs(related_query)
                task_logs.extend(related_result.logs)
        
        # Sort by timestamp
        task_logs.sort(key=lambda x: x.get('timestamp', ''))
        
        return task_logs
    
    async def get_agent_performance_insights(
        self,
        agent_type: AgentType,
        time_frame: TimeFrame = TimeFrame.LAST_WEEK
    ) -> Dict[str, Any]:
        """Get detailed performance insights for a specific agent."""
        
        query = AuditQuery(
            time_frame=time_frame,
            agent_types=[agent_type],
            include_aggregations=True,
            limit=5000
        )
        
        result = await self.query_logs(query)
        agent_logs = result.logs
        
        # Analyze patterns
        execution_times = [log.get('duration_ms') for log in agent_logs if log.get('duration_ms')]
        success_rate = len([log for log in agent_logs if log.get('action_type') in ['task_completed', 'agent_execution_completed']]) / max(1, len(agent_logs)) * 100
        
        error_patterns = Counter([log.get('error_category') for log in agent_logs if log.get('error_category')])
        
        return {
            "agent_type": agent_type.value,
            "performance_summary": {
                "total_executions": len(agent_logs),
                "success_rate": success_rate,
                "avg_execution_time": statistics.mean(execution_times) if execution_times else 0,
                "p95_execution_time": self._percentile(execution_times, 95) if execution_times else 0,
            },
            "error_analysis": dict(error_patterns),
            "trends": self._calculate_trends(agent_logs),
            "recommendations": self._generate_agent_recommendations(agent_logs, agent_type)
        }
    
    def _generate_cache_key(self, query: AuditQuery) -> str:
        """Generate cache key for query."""
        import hashlib
        query_str = json.dumps(query.__dict__, default=str, sort_keys=True)
        return f"audit_query:{hashlib.md5(query_str.encode()).hexdigest()}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[AuditResult]:
        """Get cached query result."""
        if not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                # Reconstruct AuditResult (simplified)
                return AuditResult(**data)
        except Exception as e:
            logger.debug(f"Cache retrieval failed: {str(e)}")
        
        return None
    
    async def _cache_result(self, cache_key: str, result: AuditResult) -> None:
        """Cache query result."""
        if not self.redis_client:
            return
        
        try:
            # Convert to cacheable format
            cache_data = {
                "logs": result.logs,
                "total_count": result.total_count,
                "filtered_count": result.filtered_count,
                "aggregations": result.aggregations,
                "query_time_ms": result.query_time_ms,
                "insights": result.insights
            }
            
            await self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(cache_data, default=str)
            )
        except Exception as e:
            logger.debug(f"Cache storage failed: {str(e)}")
    
    def _resolve_time_frame(self, time_frame: TimeFrame) -> Tuple[datetime, datetime]:
        """Resolve time frame to start/end times."""
        now = datetime.now(timezone.utc)
        
        if time_frame == TimeFrame.LAST_HOUR:
            start = now - timedelta(hours=1)
        elif time_frame == TimeFrame.LAST_24_HOURS:
            start = now - timedelta(hours=24)
        elif time_frame == TimeFrame.LAST_WEEK:
            start = now - timedelta(days=7)
        elif time_frame == TimeFrame.LAST_MONTH:
            start = now - timedelta(days=30)
        else:
            start = now - timedelta(hours=24)  # Default to 24 hours
        
        return start, now
    
    async def _execute_query(self, query: AuditQuery) -> List[Dict[str, Any]]:
        """Execute the actual query against log storage."""
        # This is a simplified implementation
        # In production, this would query Redis, Elasticsearch, or database
        
        if not self.agent_logger:
            return []
        
        # Build filters for the agent logger query
        filters = {}
        
        if query.start_time:
            filters["start_time"] = query.start_time.isoformat()
        if query.end_time:
            filters["end_time"] = query.end_time.isoformat()
        if query.agent_types:
            filters["agent_type"] = [at.value for at in query.agent_types]
        if query.action_types:
            filters["action_type"] = [at.value for at in query.action_types]
        if query.task_ids:
            filters["task_id"] = query.task_ids
        if query.session_ids:
            filters["session_id"] = query.session_ids
        
        # Get logs from agent logger
        logs = await self.agent_logger.get_logs(
            filters=filters,
            limit=query.limit + query.offset,  # Get extra for offset
            sort_by=query.sort_by,
            sort_order=query.sort_order
        )
        
        # Apply additional filtering
        filtered_logs = []
        for log in logs:
            if self._matches_query_filters(log, query):
                filtered_logs.append(log)
        
        return filtered_logs
    
    def _matches_query_filters(self, log: Dict[str, Any], query: AuditQuery) -> bool:
        """Check if log matches additional query filters."""
        
        # Duration filtering
        if query.min_duration_ms and log.get('duration_ms', 0) < query.min_duration_ms:
            return False
        if query.max_duration_ms and log.get('duration_ms', 0) > query.max_duration_ms:
            return False
        
        # Confidence score filtering
        if query.min_confidence_score and log.get('confidence_score', 0) < query.min_confidence_score:
            return False
        if query.max_confidence_score and log.get('confidence_score', 1) > query.max_confidence_score:
            return False
        
        # Error filtering
        if query.has_errors is not None:
            has_error = bool(log.get('error_message') or log.get('log_level') in ['error', 'critical'])
            if query.has_errors != has_error:
                return False
        
        # Text search
        if query.search_text:
            fields = query.search_fields or ['error_message', 'reasoning', 'decision_context']
            found = False
            for field in fields:
                field_value = str(log.get(field, ''))
                if query.search_text.lower() in field_value.lower():
                    found = True
                    break
            if not found:
                return False
        
        # Tag filtering
        if query.tags:
            log_tags = log.get('tags', [])
            if not any(tag in log_tags for tag in query.tags):
                return False
        
        return True
    
    async def _calculate_aggregations(self, logs: List[Dict[str, Any]], query: AuditQuery) -> Dict[str, Any]:
        """Calculate aggregations for the query result."""
        
        aggregations = {}
        
        if query.group_by:
            # Group by specified fields
            for field in query.group_by:
                groups = defaultdict(int)
                for log in logs:
                    value = log.get(field, 'unknown')
                    groups[str(value)] += 1
                aggregations[f"group_by_{field}"] = dict(groups)
        
        # Common aggregations
        aggregations.update({
            "action_types": dict(Counter(log.get('action_type') for log in logs)),
            "log_levels": dict(Counter(log.get('log_level') for log in logs)),
            "agent_types": dict(Counter(log.get('agent_type') for log in logs if log.get('agent_type'))),
            "error_categories": dict(Counter(log.get('error_category') for log in logs if log.get('error_category'))),
            "hourly_distribution": self._calculate_hourly_distribution(logs),
            "success_rate": len([log for log in logs if log.get('action_type') in ['task_completed', 'agent_execution_completed']]) / max(1, len(logs)) * 100
        })
        
        return aggregations
    
    def _calculate_hourly_distribution(self, logs: List[Dict[str, Any]]) -> Dict[int, int]:
        """Calculate hourly distribution of logs."""
        hourly = defaultdict(int)
        
        for log in logs:
            timestamp = log.get('timestamp')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    hourly[dt.hour] += 1
                except:
                    continue
        
        return dict(hourly)
    
    async def _generate_insights(self, logs: List[Dict[str, Any]], query: AuditQuery) -> Dict[str, Any]:
        """Generate insights from query results."""
        
        if not logs:
            return {"message": "No data available for analysis"}
        
        insights = {}
        
        # Performance insights
        execution_times = [log.get('duration_ms') for log in logs if log.get('duration_ms')]
        if execution_times:
            avg_time = statistics.mean(execution_times)
            if avg_time > 5000:
                insights["performance_concern"] = f"Average execution time is {avg_time:.0f}ms, which is above optimal threshold"
        
        # Error rate insights
        error_count = len([log for log in logs if log.get('log_level') in ['error', 'critical']])
        error_rate = error_count / len(logs) * 100
        if error_rate > 5:
            insights["error_rate_concern"] = f"Error rate is {error_rate:.1f}%, which is above acceptable threshold"
        
        # Quality insights
        hallucination_scores = [log.get('hallucination_score') for log in logs if log.get('hallucination_score')]
        if hallucination_scores:
            avg_hallucination = statistics.mean(hallucination_scores)
            if avg_hallucination > 0.02:
                insights["quality_concern"] = f"Average hallucination score is {avg_hallucination:.3f}, above 2% threshold"
        
        return insights
    
    def _analyze_access_patterns(self, access_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze data access patterns for compliance."""
        
        if not access_events:
            return {"total_accesses": 0}
        
        # Analyze timing patterns
        hours = [datetime.fromisoformat(event.get('timestamp', '').replace('Z', '+00:00')).hour 
                for event in access_events if event.get('timestamp')]
        
        # Analyze user patterns
        users = Counter(event.get('user_id') for event in access_events if event.get('user_id'))
        
        return {
            "total_accesses": len(access_events),
            "unique_users": len(users),
            "peak_hour": max(Counter(hours).items(), key=lambda x: x[1])[0] if hours else None,
            "frequent_users": dict(users.most_common(5))
        }
    
    def _calculate_trends(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trends from log data."""
        
        # Group logs by day
        daily_counts = defaultdict(int)
        for log in logs:
            timestamp = log.get('timestamp')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    day = dt.date().isoformat()
                    daily_counts[day] += 1
                except:
                    continue
        
        # Calculate trend direction
        counts = list(daily_counts.values())
        if len(counts) > 1:
            trend = "increasing" if counts[-1] > counts[0] else "decreasing" if counts[-1] < counts[0] else "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "daily_counts": dict(daily_counts),
            "trend": trend,
            "total_days": len(daily_counts)
        }
    
    def _generate_agent_recommendations(self, logs: List[Dict[str, Any]], agent_type: AgentType) -> List[str]:
        """Generate optimization recommendations for an agent."""
        
        recommendations = []
        
        # Performance recommendations
        execution_times = [log.get('duration_ms') for log in logs if log.get('duration_ms')]
        if execution_times:
            avg_time = statistics.mean(execution_times)
            if avg_time > 5000:
                recommendations.append(f"Consider optimizing {agent_type.value} - average execution time is {avg_time:.0f}ms")
        
        # Error rate recommendations
        error_count = len([log for log in logs if log.get('log_level') in ['error', 'critical']])
        if error_count > len(logs) * 0.05:  # >5% error rate
            recommendations.append(f"Investigate error patterns for {agent_type.value} - high error rate detected")
        
        # Resource recommendations
        pool_events = [log for log in logs if log.get('action_type') == 'agent_instantiated']
        if len(pool_events) > 10:  # Frequent instantiation
            recommendations.append(f"Consider increasing pool size for {agent_type.value} - frequent instantiation detected")
        
        return recommendations
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0
        
        sorted_data = sorted(data)
        index = int(percentile / 100 * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]


# Global service instance
_audit_service: Optional[AgentAuditService] = None


async def get_audit_service() -> AgentAuditService:
    """Get the global agent audit service instance."""
    global _audit_service
    
    if not _audit_service:
        _audit_service = AgentAuditService()
        await _audit_service.initialize()
    
    return _audit_service