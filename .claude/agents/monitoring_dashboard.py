#!/usr/bin/env python3
"""
Real-time Monitoring and Observability Dashboard
For AI Multi-Agent Orchestration System
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml


@dataclass
class AgentMetrics:
    """Individual agent performance metrics"""
    agent_id: str
    agent_name: str
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_execution_time: float = 0.0
    last_execution: Optional[datetime] = None
    current_status: str = "idle"
    error_rate: float = 0.0
    throughput: float = 0.0  # executions per hour
    resource_usage: Dict[str, float] = None
    
    def __post_init__(self):
        if self.resource_usage is None:
            self.resource_usage = {"cpu": 0.0, "memory": 0.0, "tokens": 0}


@dataclass
class WorkflowMetrics:
    """Overall workflow system metrics"""
    total_workflows: int = 0
    active_workflows: int = 0
    completed_workflows: int = 0
    failed_workflows: int = 0
    avg_workflow_duration: float = 0.0
    system_uptime: float = 0.0
    throughput: float = 0.0  # workflows per hour
    success_rate: float = 0.0
    
    # Quality metrics
    avg_hallucination_rate: float = 0.0
    avg_quality_score: float = 0.0
    sla_compliance: float = 0.0
    
    # Resource metrics
    total_token_usage: int = 0
    estimated_cost: float = 0.0
    peak_memory_usage: float = 0.0
    avg_cpu_usage: float = 0.0


@dataclass
class SystemAlert:
    """System alert for monitoring dashboard"""
    alert_id: str
    severity: str  # critical, high, medium, low
    component: str  # agent, workflow, system
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class OrchestrationMonitor:
    """
    Real-time monitoring system for multi-agent orchestration
    Provides metrics collection, alerting, and dashboard data
    """
    
    def __init__(self, config_path: str = ".claude/agents/monitoring_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.workflow_metrics = WorkflowMetrics()
        self.alerts: List[SystemAlert] = []
        self.start_time = datetime.now()
        
        # Monitoring configuration
        self.metrics_retention_days = 30
        self.alert_thresholds = self.config.get('thresholds', {})
        self.monitoring_interval = 30  # seconds
        
        # Setup logging
        self.setup_logging()
        
        # Initialize metrics storage
        self.metrics_store = []
        self.is_monitoring = False
    
    def _load_config(self) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            'thresholds': {
                'hallucination_rate_max': 0.02,  # 2%
                'error_rate_max': 0.1,           # 10%
                'response_time_max': 30.0,       # 30 seconds
                'memory_usage_max': 0.8,         # 80%
                'success_rate_min': 0.9          # 90%
            },
            'alerts': {
                'enabled': True,
                'email_notifications': False,
                'slack_notifications': False,
                'webhook_url': None
            },
            'metrics': {
                'collection_interval': 30,
                'retention_days': 30,
                'aggregation_window': 300  # 5 minutes
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    return {**default_config, **config}
            except Exception as e:
                logging.error(f"Failed to load monitoring config: {e}")
        
        return default_config
    
    def setup_logging(self):
        """Setup monitoring-specific logging"""
        self.logger = logging.getLogger('OrchestrationMonitor')
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        logs_dir = Path('.claude/logs')
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler for monitoring logs
        handler = logging.FileHandler(logs_dir / 'monitoring.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    async def start_monitoring(self):
        """Start continuous monitoring"""
        self.is_monitoring = True
        self.logger.info("Starting orchestration monitoring")
        
        while self.is_monitoring:
            try:
                await self._collect_metrics()
                await self._check_alerts()
                await self._cleanup_old_data()
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5)  # Brief pause on error
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        self.logger.info("Stopping orchestration monitoring")
    
    async def _collect_metrics(self):
        """Collect current system metrics"""
        current_time = datetime.now()
        
        # Update system uptime
        self.workflow_metrics.system_uptime = (current_time - self.start_time).total_seconds()
        
        # Calculate aggregate metrics
        await self._calculate_aggregate_metrics()
        
        # Store metrics snapshot
        metrics_snapshot = {
            'timestamp': current_time,
            'workflow_metrics': asdict(self.workflow_metrics),
            'agent_metrics': {k: asdict(v) for k, v in self.agent_metrics.items()},
            'system_health': await self._get_system_health()
        }
        
        self.metrics_store.append(metrics_snapshot)
        
        # Trim old metrics
        cutoff_time = current_time - timedelta(days=self.metrics_retention_days)
        self.metrics_store = [
            m for m in self.metrics_store 
            if m['timestamp'] > cutoff_time
        ]
    
    async def _calculate_aggregate_metrics(self):
        """Calculate aggregate system metrics"""
        if not self.agent_metrics:
            return
        
        # Agent-level aggregations
        total_executions = sum(m.execution_count for m in self.agent_metrics.values())
        total_failures = sum(m.failure_count for m in self.agent_metrics.values())
        
        # Workflow success rate
        if total_executions > 0:
            self.workflow_metrics.success_rate = 1.0 - (total_failures / total_executions)
        
        # Average execution times
        execution_times = [m.avg_execution_time for m in self.agent_metrics.values() if m.avg_execution_time > 0]
        if execution_times:
            self.workflow_metrics.avg_workflow_duration = sum(execution_times) / len(execution_times)
        
        # Resource usage aggregation
        total_tokens = sum(m.resource_usage.get('tokens', 0) for m in self.agent_metrics.values())
        self.workflow_metrics.total_token_usage = total_tokens
        
        # Estimated cost calculation (rough estimate)
        self.workflow_metrics.estimated_cost = total_tokens * 0.00001  # $0.01 per 1000 tokens
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        health_status = "healthy"
        health_score = 100.0
        issues = []
        
        # Check hallucination rate
        if self.workflow_metrics.avg_hallucination_rate > self.alert_thresholds.get('hallucination_rate_max', 0.02):
            health_status = "degraded"
            health_score -= 20
            issues.append("High hallucination rate detected")
        
        # Check error rate
        error_rate = 1.0 - self.workflow_metrics.success_rate
        if error_rate > self.alert_thresholds.get('error_rate_max', 0.1):
            health_status = "critical" if error_rate > 0.2 else "degraded"
            health_score -= 30
            issues.append(f"High error rate: {error_rate:.2%}")
        
        # Check response times
        if self.workflow_metrics.avg_workflow_duration > self.alert_thresholds.get('response_time_max', 30):
            health_status = "degraded"
            health_score -= 15
            issues.append("High response times detected")
        
        # Check for active alerts
        critical_alerts = [a for a in self.alerts if a.severity == "critical" and not a.resolved]
        if critical_alerts:
            health_status = "critical"
            health_score -= 25
            issues.append(f"{len(critical_alerts)} critical alerts active")
        
        return {
            'status': health_status,
            'score': max(0, health_score),
            'issues': issues,
            'last_check': datetime.now()
        }
    
    async def _check_alerts(self):
        """Check for alert conditions"""
        current_time = datetime.now()
        
        # Hallucination rate alert
        if self.workflow_metrics.avg_hallucination_rate > self.alert_thresholds.get('hallucination_rate_max', 0.02):
            await self._create_alert(
                component="graphrag",
                severity="critical",
                message=f"Hallucination rate {self.workflow_metrics.avg_hallucination_rate:.3f} exceeds threshold"
            )
        
        # Error rate alert
        error_rate = 1.0 - self.workflow_metrics.success_rate
        if error_rate > self.alert_thresholds.get('error_rate_max', 0.1):
            severity = "critical" if error_rate > 0.2 else "high"
            await self._create_alert(
                component="workflow",
                severity=severity,
                message=f"System error rate {error_rate:.2%} exceeds threshold"
            )
        
        # Performance alert
        if self.workflow_metrics.avg_workflow_duration > self.alert_thresholds.get('response_time_max', 30):
            await self._create_alert(
                component="performance",
                severity="medium",
                message=f"Average response time {self.workflow_metrics.avg_workflow_duration:.1f}s exceeds threshold"
            )
        
        # Agent-specific alerts
        for agent_id, metrics in self.agent_metrics.items():
            if metrics.error_rate > 0.15:  # 15% error rate for individual agents
                await self._create_alert(
                    component=f"agent-{agent_id}",
                    severity="high",
                    message=f"Agent {agent_id} error rate {metrics.error_rate:.2%} is high"
                )
    
    async def _create_alert(self, component: str, severity: str, message: str):
        """Create a new system alert"""
        # Check if similar alert already exists and is unresolved
        existing_alert = next((
            alert for alert in self.alerts
            if alert.component == component 
            and alert.severity == severity
            and not alert.resolved
            and alert.message == message
        ), None)
        
        if existing_alert:
            return  # Don't create duplicate alerts
        
        alert = SystemAlert(
            alert_id=f"{component}_{int(time.time())}",
            severity=severity,
            component=component,
            message=message,
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        self.logger.warning(f"Alert created: {severity.upper()} - {component}: {message}")
        
        # Send notifications if configured
        await self._send_alert_notification(alert)
    
    async def _send_alert_notification(self, alert: SystemAlert):
        """Send alert notifications"""
        if not self.config.get('alerts', {}).get('enabled', True):
            return
        
        # Log alert (always enabled)
        self.logger.warning(f"ALERT: {alert.severity.upper()} - {alert.message}")
        
        # Additional notification methods can be implemented here
        # Email, Slack, webhook notifications, etc.
    
    async def _cleanup_old_data(self):
        """Cleanup old alerts and metrics"""
        cutoff_time = datetime.now() - timedelta(days=self.metrics_retention_days)
        
        # Remove old resolved alerts
        self.alerts = [
            alert for alert in self.alerts
            if not alert.resolved or alert.timestamp > cutoff_time
        ]
    
    def update_agent_metrics(self, agent_id: str, agent_name: str, execution_data: Dict[str, Any]):
        """Update metrics for a specific agent"""
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = AgentMetrics(agent_id=agent_id, agent_name=agent_name)
        
        metrics = self.agent_metrics[agent_id]
        
        # Update execution counts
        metrics.execution_count += 1
        if execution_data.get('status') == 'completed':
            metrics.success_count += 1
        else:
            metrics.failure_count += 1
        
        # Update timing metrics
        if 'execution_time' in execution_data:
            execution_time = execution_data['execution_time']
            if metrics.avg_execution_time == 0:
                metrics.avg_execution_time = execution_time
            else:
                # Running average
                metrics.avg_execution_time = (
                    (metrics.avg_execution_time * (metrics.execution_count - 1) + execution_time)
                    / metrics.execution_count
                )
        
        # Update status and timestamps
        metrics.current_status = execution_data.get('status', 'unknown')
        metrics.last_execution = datetime.now()
        
        # Calculate error rate
        if metrics.execution_count > 0:
            metrics.error_rate = metrics.failure_count / metrics.execution_count
        
        # Update resource usage
        if 'resource_usage' in execution_data:
            metrics.resource_usage.update(execution_data['resource_usage'])
        
        self.logger.debug(f"Updated metrics for agent {agent_id}: {metrics}")
    
    def update_workflow_metrics(self, workflow_data: Dict[str, Any]):
        """Update overall workflow metrics"""
        self.workflow_metrics.total_workflows += 1
        
        if workflow_data.get('status') == 'completed':
            self.workflow_metrics.completed_workflows += 1
        elif workflow_data.get('status') == 'failed':
            self.workflow_metrics.failed_workflows += 1
        
        # Update quality metrics
        if 'hallucination_rate' in workflow_data:
            rate = workflow_data['hallucination_rate']
            if self.workflow_metrics.avg_hallucination_rate == 0:
                self.workflow_metrics.avg_hallucination_rate = rate
            else:
                # Running average
                total = self.workflow_metrics.total_workflows
                self.workflow_metrics.avg_hallucination_rate = (
                    (self.workflow_metrics.avg_hallucination_rate * (total - 1) + rate) / total
                )
        
        if 'quality_score' in workflow_data:
            score = workflow_data['quality_score']
            if self.workflow_metrics.avg_quality_score == 0:
                self.workflow_metrics.avg_quality_score = score
            else:
                # Running average
                total = self.workflow_metrics.total_workflows
                self.workflow_metrics.avg_quality_score = (
                    (self.workflow_metrics.avg_quality_score * (total - 1) + score) / total
                )
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        current_time = datetime.now()
        
        # Active alerts
        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        critical_alerts = [alert for alert in active_alerts if alert.severity == "critical"]
        
        # Recent metrics (last hour)
        recent_metrics = [
            m for m in self.metrics_store
            if m['timestamp'] > current_time - timedelta(hours=1)
        ]
        
        dashboard_data = {
            'system_overview': {
                'status': 'healthy' if not critical_alerts else 'critical',
                'uptime': self.workflow_metrics.system_uptime,
                'total_workflows': self.workflow_metrics.total_workflows,
                'success_rate': self.workflow_metrics.success_rate,
                'avg_response_time': self.workflow_metrics.avg_workflow_duration
            },
            'quality_metrics': {
                'hallucination_rate': self.workflow_metrics.avg_hallucination_rate,
                'quality_score': self.workflow_metrics.avg_quality_score,
                'threshold_compliance': {
                    'hallucination_ok': self.workflow_metrics.avg_hallucination_rate <= 0.02,
                    'quality_ok': self.workflow_metrics.avg_quality_score >= 0.85,
                    'performance_ok': self.workflow_metrics.avg_workflow_duration <= 30.0
                }
            },
            'agent_metrics': [
                {
                    'agent_id': agent_id,
                    'name': metrics.agent_name,
                    'status': metrics.current_status,
                    'success_rate': 1.0 - metrics.error_rate,
                    'avg_execution_time': metrics.avg_execution_time,
                    'executions': metrics.execution_count,
                    'last_execution': metrics.last_execution
                }
                for agent_id, metrics in self.agent_metrics.items()
            ],
            'alerts': {
                'total': len(self.alerts),
                'active': len(active_alerts),
                'critical': len(critical_alerts),
                'recent_alerts': [
                    {
                        'id': alert.alert_id,
                        'severity': alert.severity,
                        'component': alert.component,
                        'message': alert.message,
                        'timestamp': alert.timestamp,
                        'resolved': alert.resolved
                    }
                    for alert in sorted(self.alerts, key=lambda x: x.timestamp, reverse=True)[:10]
                ]
            },
            'performance': {
                'throughput': len(recent_metrics),
                'resource_usage': {
                    'total_tokens': self.workflow_metrics.total_token_usage,
                    'estimated_cost': self.workflow_metrics.estimated_cost,
                    'memory_usage': self.workflow_metrics.peak_memory_usage,
                    'cpu_usage': self.workflow_metrics.avg_cpu_usage
                }
            },
            'trends': {
                'hourly_metrics': len(recent_metrics),
                'error_trend': self._calculate_trend('error_rate'),
                'performance_trend': self._calculate_trend('response_time'),
                'quality_trend': self._calculate_trend('quality_score')
            }
        }
        
        return dashboard_data
    
    def _calculate_trend(self, metric: str) -> str:
        """Calculate trend direction for a metric"""
        if len(self.metrics_store) < 2:
            return "stable"
        
        recent_values = []
        for snapshot in self.metrics_store[-10:]:  # Last 10 snapshots
            if metric == 'error_rate':
                value = 1.0 - snapshot['workflow_metrics']['success_rate']
            elif metric == 'response_time':
                value = snapshot['workflow_metrics']['avg_workflow_duration']
            elif metric == 'quality_score':
                value = snapshot['workflow_metrics']['avg_quality_score']
            else:
                continue
            recent_values.append(value)
        
        if len(recent_values) < 2:
            return "stable"
        
        # Simple trend calculation
        first_half = sum(recent_values[:len(recent_values)//2]) / (len(recent_values)//2)
        second_half = sum(recent_values[len(recent_values)//2:]) / (len(recent_values) - len(recent_values)//2)
        
        change = (second_half - first_half) / first_half if first_half > 0 else 0
        
        if abs(change) < 0.05:  # Less than 5% change
            return "stable"
        elif change > 0:
            return "increasing" if metric == 'quality_score' else "decreasing"
        else:
            return "decreasing" if metric == 'quality_score' else "improving"
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                self.logger.info(f"Alert resolved: {alert_id}")
                break
    
    def get_metrics_export(self, start_time: datetime = None, end_time: datetime = None) -> Dict[str, Any]:
        """Export metrics for analysis or reporting"""
        if start_time is None:
            start_time = datetime.now() - timedelta(days=1)
        if end_time is None:
            end_time = datetime.now()
        
        filtered_metrics = [
            m for m in self.metrics_store
            if start_time <= m['timestamp'] <= end_time
        ]
        
        return {
            'export_period': {
                'start': start_time,
                'end': end_time,
                'duration_hours': (end_time - start_time).total_seconds() / 3600
            },
            'metrics': filtered_metrics,
            'summary': {
                'total_snapshots': len(filtered_metrics),
                'avg_success_rate': sum(m['workflow_metrics']['success_rate'] for m in filtered_metrics) / len(filtered_metrics) if filtered_metrics else 0,
                'avg_hallucination_rate': sum(m['workflow_metrics']['avg_hallucination_rate'] for m in filtered_metrics) / len(filtered_metrics) if filtered_metrics else 0
            }
        }


# Dashboard Web Interface (Simple HTTP server)
class DashboardServer:
    """Simple HTTP server for monitoring dashboard"""
    
    def __init__(self, monitor: OrchestrationMonitor, port: int = 8080):
        self.monitor = monitor
        self.port = port
    
    async def start_server(self):
        """Start dashboard web server"""
        # This would typically use a web framework like FastAPI or Flask
        # For now, just provide the data structure
        self.monitor.logger.info(f"Dashboard server would start on port {self.port}")
        
        # In a real implementation, this would serve a web dashboard
        # showing the metrics from self.monitor.get_dashboard_data()


# CLI Interface
async def main():
    """Main monitoring execution"""
    monitor = OrchestrationMonitor()
    
    # Start monitoring in background
    monitoring_task = asyncio.create_task(monitor.start_monitoring())
    
    try:
        # Simulate some agent executions for demonstration
        await asyncio.sleep(5)
        
        # Update some test metrics
        monitor.update_agent_metrics('doc_librarian', 'documentation-librarian', {
            'status': 'completed',
            'execution_time': 2.5,
            'resource_usage': {'tokens': 1500}
        })
        
        monitor.update_workflow_metrics({
            'status': 'completed',
            'hallucination_rate': 0.015,
            'quality_score': 0.92
        })
        
        # Get dashboard data
        dashboard_data = monitor.get_dashboard_data()
        print(json.dumps(dashboard_data, indent=2, default=str))
        
        await asyncio.sleep(10)
        
    finally:
        monitor.stop_monitoring()
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass


if __name__ == '__main__':
    asyncio.run(main())