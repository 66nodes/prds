"""
Performance Benchmarking Tool for Enhanced Parallel Execution

Comprehensive benchmarking and performance analysis tool for the enhanced
parallel agent execution system with load testing and optimization metrics.
"""

import asyncio
import time
import statistics
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json

import structlog
from services.enhanced_parallel_executor import (
    EnhancedParallelExecutor,
    PriorityLevel,
    ExecutionStatus,
    LoadBalancingStrategy
)
from services.agent_orchestrator import AgentType, AgentTask, WorkflowContext

logger = structlog.get_logger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests."""
    name: str
    description: str
    task_count: int = 100
    concurrent_users: int = 10
    test_duration_seconds: int = 60
    ramp_up_seconds: int = 10
    agent_types: List[AgentType] = field(default_factory=list)
    priority_distribution: Dict[PriorityLevel, float] = field(default_factory=dict)
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.RESOURCE_AWARE
    enable_circuit_breakers: bool = True
    enable_agent_pooling: bool = True
    max_concurrent_tasks: int = 20


@dataclass
class BenchmarkMetrics:
    """Metrics collected during benchmark execution."""
    test_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_seconds: float = 0.0
    
    # Throughput metrics
    total_tasks_executed: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    tasks_per_second: float = 0.0
    peak_throughput: float = 0.0
    
    # Latency metrics
    response_times: List[float] = field(default_factory=list)
    avg_response_time_ms: float = 0.0
    p50_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    min_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    
    # Resource utilization
    peak_cpu_usage: float = 0.0
    peak_memory_usage: float = 0.0
    avg_cpu_usage: float = 0.0
    avg_memory_usage: float = 0.0
    peak_concurrent_tasks: int = 0
    
    # Error analysis
    error_rate: float = 0.0
    error_distribution: Dict[str, int] = field(default_factory=dict)
    circuit_breaker_activations: Dict[str, int] = field(default_factory=dict)
    
    # Agent performance
    agent_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Load balancing effectiveness
    load_distribution_variance: float = 0.0
    load_balancing_efficiency: float = 0.0


class ExecutionBenchmarker:
    """
    Performance benchmarker for enhanced parallel execution system.
    
    Provides comprehensive load testing, stress testing, and performance
    analysis capabilities for multi-agent orchestration systems.
    """
    
    def __init__(self):
        self.executor: Optional[EnhancedParallelExecutor] = None
        self.benchmark_results: List[BenchmarkMetrics] = []
        self.monitoring_tasks: List[asyncio.Task] = []
        
    async def initialize(self) -> None:
        """Initialize the benchmarker."""
        logger.info("Initializing execution benchmarker")
        
    async def run_benchmark_suite(
        self, 
        configs: List[BenchmarkConfig]
    ) -> List[BenchmarkMetrics]:
        """Run a complete benchmark suite with multiple configurations."""
        logger.info(f"Starting benchmark suite with {len(configs)} configurations")
        
        results = []
        for config in configs:
            try:
                logger.info(f"Running benchmark: {config.name}")
                result = await self.run_single_benchmark(config)
                results.append(result)
                
                # Cool-down period between benchmarks
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Benchmark {config.name} failed: {str(e)}")
                continue
        
        # Generate comparative analysis
        await self._generate_comparative_analysis(results)
        
        return results
    
    async def run_single_benchmark(self, config: BenchmarkConfig) -> BenchmarkMetrics:
        """Run a single benchmark test."""
        logger.info(f"Starting benchmark: {config.name}", config=config.__dict__)
        
        # Initialize executor with benchmark configuration
        self.executor = EnhancedParallelExecutor(
            max_concurrent_tasks=config.max_concurrent_tasks,
            load_balancing_strategy=config.load_balancing_strategy,
            enable_circuit_breakers=config.enable_circuit_breakers,
            enable_agent_pooling=config.enable_agent_pooling,
            resource_monitoring_interval=1.0
        )
        
        await self.executor.initialize()
        
        # Initialize metrics
        metrics = BenchmarkMetrics(
            test_name=config.name,
            start_time=datetime.utcnow()
        )
        
        try:
            # Start monitoring
            monitoring_task = asyncio.create_task(
                self._monitor_execution(metrics, config.test_duration_seconds)
            )
            self.monitoring_tasks.append(monitoring_task)
            
            # Run the benchmark
            await self._execute_benchmark_workload(config, metrics)
            
            # Stop monitoring
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
            
            # Finalize metrics
            metrics.end_time = datetime.utcnow()
            metrics.total_duration_seconds = (
                metrics.end_time - metrics.start_time
            ).total_seconds()
            
            # Calculate final statistics
            await self._calculate_final_statistics(metrics)
            
            logger.info(f"Benchmark {config.name} completed", metrics=self._metrics_summary(metrics))
            
        except Exception as e:
            logger.error(f"Benchmark execution failed: {str(e)}")
            raise
        finally:
            if self.executor:
                await self.executor.shutdown()
                self.executor = None
        
        return metrics
    
    async def _execute_benchmark_workload(
        self, 
        config: BenchmarkConfig, 
        metrics: BenchmarkMetrics
    ) -> None:
        """Execute the benchmark workload."""
        
        # Create tasks for the benchmark
        tasks = self._generate_benchmark_tasks(config)
        
        # Create workflow context
        workflow = WorkflowContext(
            workflow_id=f"benchmark_{config.name}_{int(time.time())}",
            context_data={"benchmark": True, "config": config.name}
        )
        
        # Execute tasks in batches to simulate concurrent users
        batch_size = max(1, config.task_count // config.concurrent_users)
        task_batches = [
            tasks[i:i + batch_size] 
            for i in range(0, len(tasks), batch_size)
        ]
        
        # Ramp up execution (gradual increase in load)
        ramp_up_delay = config.ramp_up_seconds / len(task_batches)
        
        execution_tasks = []
        for i, batch in enumerate(task_batches):
            # Add ramp-up delay
            if i > 0:
                await asyncio.sleep(ramp_up_delay)
            
            # Execute batch
            batch_task = asyncio.create_task(
                self._execute_task_batch(batch, workflow, metrics)
            )
            execution_tasks.append(batch_task)
        
        # Wait for all batches to complete
        await asyncio.gather(*execution_tasks, return_exceptions=True)
    
    def _generate_benchmark_tasks(self, config: BenchmarkConfig) -> List[AgentTask]:
        """Generate tasks for benchmark testing."""
        tasks = []
        
        # Use specified agent types or default mix
        if config.agent_types:
            agent_types = config.agent_types
        else:
            agent_types = [
                AgentType.DRAFT_AGENT,
                AgentType.JUDGE_AGENT,
                AgentType.BUSINESS_ANALYST,
                AgentType.PROJECT_ARCHITECT,
                AgentType.CONTEXT_MANAGER
            ]
        
        # Priority distribution
        priority_dist = config.priority_distribution or {
            PriorityLevel.CRITICAL: 0.1,
            PriorityLevel.HIGH: 0.2,
            PriorityLevel.NORMAL: 0.6,
            PriorityLevel.LOW: 0.1
        }
        
        for i in range(config.task_count):
            # Select agent type (round-robin)
            agent_type = agent_types[i % len(agent_types)]
            
            # Select priority based on distribution
            priority = self._select_priority_from_distribution(priority_dist)
            
            task = AgentTask(
                task_id=f"benchmark_task_{i}",
                agent_type=agent_type,
                input_data={
                    "benchmark_data": f"Test data for task {i}",
                    "complexity": "medium",
                    "expected_duration": "short"
                },
                metadata={
                    "benchmark": True,
                    "task_index": i,
                    "priority": priority.value
                }
            )
            
            tasks.append(task)
        
        return tasks
    
    def _select_priority_from_distribution(
        self, 
        distribution: Dict[PriorityLevel, float]
    ) -> PriorityLevel:
        """Select priority level based on distribution."""
        import random
        
        rand = random.random()
        cumulative = 0.0
        
        for priority, weight in distribution.items():
            cumulative += weight
            if rand <= cumulative:
                return priority
        
        return PriorityLevel.NORMAL
    
    async def _execute_task_batch(
        self,
        tasks: List[AgentTask],
        workflow: WorkflowContext,
        metrics: BenchmarkMetrics
    ) -> None:
        """Execute a batch of tasks and collect metrics."""
        if not self.executor or not tasks:
            return
        
        start_time = time.time()
        
        try:
            # Execute tasks with enhanced executor
            result = await self.executor.execute_parallel(
                tasks=tasks,
                workflow=workflow,
                priority=PriorityLevel.NORMAL,
                timeout=60.0
            )
            
            end_time = time.time()
            batch_duration = (end_time - start_time) * 1000  # Convert to ms
            
            # Update metrics
            batch_results = result.get("results", {})
            
            for task_id, task_result in batch_results.items():
                metrics.total_tasks_executed += 1
                
                if task_result.get("status") == ExecutionStatus.COMPLETED.value:
                    metrics.successful_tasks += 1
                    
                    # Record response time
                    task_metrics = task_result.get("metrics", {})
                    response_time = task_metrics.get("duration_ms", batch_duration)
                    metrics.response_times.append(response_time)
                    
                else:
                    metrics.failed_tasks += 1
                    error = task_result.get("error", "Unknown error")
                    metrics.error_distribution[error] = metrics.error_distribution.get(error, 0) + 1
            
            # Update resource usage metrics
            resource_usage = result.get("resource_usage", {})
            if resource_usage:
                metrics.peak_cpu_usage = max(
                    metrics.peak_cpu_usage, 
                    resource_usage.get("cpu_percent", 0)
                )
                metrics.peak_memory_usage = max(
                    metrics.peak_memory_usage, 
                    resource_usage.get("memory_percent", 0)
                )
                metrics.peak_concurrent_tasks = max(
                    metrics.peak_concurrent_tasks,
                    resource_usage.get("active_tasks", 0)
                )
            
        except Exception as e:
            logger.error(f"Batch execution failed: {str(e)}")
            for task in tasks:
                metrics.failed_tasks += 1
                metrics.total_tasks_executed += 1
                metrics.error_distribution["Batch Execution Failure"] = (
                    metrics.error_distribution.get("Batch Execution Failure", 0) + 1
                )
    
    async def _monitor_execution(
        self, 
        metrics: BenchmarkMetrics, 
        duration_seconds: int
    ) -> None:
        """Monitor execution metrics during benchmark."""
        start_time = time.time()
        last_task_count = 0
        throughput_enterprises = []
        
        try:
            while (time.time() - start_time) < duration_seconds:
                await asyncio.sleep(1.0)  # Sample every second
                
                if not self.executor:
                    continue
                
                # Get current execution status
                status = await self.executor.get_execution_status()
                
                # Calculate throughput
                current_task_count = metrics.total_tasks_executed
                throughput = current_task_count - last_task_count
                throughput_enterprises.append(throughput)
                last_task_count = current_task_count
                
                # Update peak throughput
                metrics.peak_throughput = max(metrics.peak_throughput, throughput)
                
                # Update resource usage
                resource_usage = status.get("resource_usage", {})
                if resource_usage:
                    metrics.avg_cpu_usage = (
                        metrics.avg_cpu_usage * 0.9 + 
                        resource_usage.get("cpu_percent", 0) * 0.1
                    )
                    metrics.avg_memory_usage = (
                        metrics.avg_memory_usage * 0.9 + 
                        resource_usage.get("memory_percent", 0) * 0.1
                    )
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Monitoring error: {str(e)}")
    
    async def _calculate_final_statistics(self, metrics: BenchmarkMetrics) -> None:
        """Calculate final benchmark statistics."""
        if metrics.total_duration_seconds > 0:
            metrics.tasks_per_second = metrics.total_tasks_executed / metrics.total_duration_seconds
        
        if metrics.total_tasks_executed > 0:
            metrics.error_rate = (metrics.failed_tasks / metrics.total_tasks_executed) * 100
        
        # Calculate response time percentiles
        if metrics.response_times:
            sorted_times = sorted(metrics.response_times)
            metrics.avg_response_time_ms = statistics.mean(sorted_times)
            metrics.min_response_time_ms = min(sorted_times)
            metrics.max_response_time_ms = max(sorted_times)
            
            # Percentiles
            metrics.p50_response_time_ms = self._percentile(sorted_times, 50)
            metrics.p95_response_time_ms = self._percentile(sorted_times, 95)
            metrics.p99_response_time_ms = self._percentile(sorted_times, 99)
    
    def _percentile(self, sorted_data: List[float], percentile: int) -> float:
        """Calculate percentile value from sorted data."""
        if not sorted_data:
            return 0.0
        
        index = int((percentile / 100) * len(sorted_data))
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
        
        return sorted_data[index]
    
    def _metrics_summary(self, metrics: BenchmarkMetrics) -> Dict[str, Any]:
        """Generate a summary of benchmark metrics."""
        return {
            "total_tasks": metrics.total_tasks_executed,
            "success_rate": f"{((metrics.successful_tasks / metrics.total_tasks_executed) * 100):.2f}%" if metrics.total_tasks_executed > 0 else "0%",
            "throughput": f"{metrics.tasks_per_second:.2f} tasks/sec",
            "avg_response_time": f"{metrics.avg_response_time_ms:.2f}ms",
            "p95_response_time": f"{metrics.p95_response_time_ms:.2f}ms",
            "error_rate": f"{metrics.error_rate:.2f}%",
            "peak_cpu": f"{metrics.peak_cpu_usage:.1f}%",
            "peak_memory": f"{metrics.peak_memory_usage:.1f}%"
        }
    
    async def _generate_comparative_analysis(
        self, 
        results: List[BenchmarkMetrics]
    ) -> None:
        """Generate comparative analysis of benchmark results."""
        if len(results) < 2:
            return
        
        logger.info("Generating comparative benchmark analysis")
        
        # Compare key metrics across benchmarks
        comparison = {
            "throughput_comparison": {},
            "latency_comparison": {},
            "resource_efficiency": {},
            "error_rate_comparison": {},
            "recommendations": []
        }
        
        # Throughput analysis
        best_throughput = max(results, key=lambda x: x.tasks_per_second)
        comparison["throughput_comparison"] = {
            "best_configuration": best_throughput.test_name,
            "best_throughput": f"{best_throughput.tasks_per_second:.2f} tasks/sec",
            "all_results": {
                r.test_name: f"{r.tasks_per_second:.2f} tasks/sec"
                for r in results
            }
        }
        
        # Latency analysis
        best_latency = min(results, key=lambda x: x.p95_response_time_ms)
        comparison["latency_comparison"] = {
            "best_configuration": best_latency.test_name,
            "best_p95_latency": f"{best_latency.p95_response_time_ms:.2f}ms",
            "all_results": {
                r.test_name: f"{r.p95_response_time_ms:.2f}ms"
                for r in results
            }
        }
        
        # Generate recommendations
        recommendations = []
        
        if best_throughput.test_name != best_latency.test_name:
            recommendations.append(
                f"Trade-off detected: {best_throughput.test_name} has best throughput, "
                f"but {best_latency.test_name} has best latency"
            )
        
        # Resource efficiency
        for result in results:
            if result.peak_cpu_usage > 90:
                recommendations.append(
                    f"{result.test_name}: CPU usage very high ({result.peak_cpu_usage:.1f}%), "
                    "consider reducing concurrency"
                )
            
            if result.error_rate > 5:
                recommendations.append(
                    f"{result.test_name}: High error rate ({result.error_rate:.2f}%), "
                    "investigate circuit breaker settings"
                )
        
        comparison["recommendations"] = recommendations
        
        # Log comparative analysis
        logger.info("Benchmark comparative analysis", analysis=comparison)
    
    def export_results(
        self, 
        results: List[BenchmarkMetrics], 
        output_file: str = "benchmark_results.json"
    ) -> None:
        """Export benchmark results to JSON file."""
        export_data = {
            "benchmark_suite_results": [
                {
                    "test_name": r.test_name,
                    "duration_seconds": r.total_duration_seconds,
                    "throughput_tasks_per_second": r.tasks_per_second,
                    "success_rate_percent": ((r.successful_tasks / r.total_tasks_executed) * 100) if r.total_tasks_executed > 0 else 0,
                    "avg_response_time_ms": r.avg_response_time_ms,
                    "p95_response_time_ms": r.p95_response_time_ms,
                    "p99_response_time_ms": r.p99_response_time_ms,
                    "error_rate_percent": r.error_rate,
                    "peak_cpu_usage": r.peak_cpu_usage,
                    "peak_memory_usage": r.peak_memory_usage,
                    "error_distribution": r.error_distribution,
                    "started_at": r.start_time.isoformat(),
                    "completed_at": r.end_time.isoformat() if r.end_time else None
                }
                for r in results
            ],
            "generated_at": datetime.utcnow().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Benchmark results exported to {output_file}")


# Predefined benchmark configurations
STANDARD_BENCHMARKS = [
    BenchmarkConfig(
        name="baseline_performance",
        description="Baseline performance test with moderate load",
        task_count=100,
        concurrent_users=5,
        test_duration_seconds=60,
        max_concurrent_tasks=10
    ),
    BenchmarkConfig(
        name="high_throughput",
        description="High throughput test with maximum concurrency",
        task_count=500,
        concurrent_users=20,
        test_duration_seconds=120,
        max_concurrent_tasks=50
    ),
    BenchmarkConfig(
        name="stress_test",
        description="Stress test with circuit breaker scenarios",
        task_count=1000,
        concurrent_users=50,
        test_duration_seconds=300,
        max_concurrent_tasks=25,
        enable_circuit_breakers=True
    ),
    BenchmarkConfig(
        name="agent_pooling_comparison",
        description="Compare performance with and without agent pooling",
        task_count=200,
        concurrent_users=10,
        test_duration_seconds=90,
        enable_agent_pooling=True,
        max_concurrent_tasks=15
    )
]


async def run_standard_benchmarks() -> List[BenchmarkMetrics]:
    """Run the standard benchmark suite."""
    benchmarker = ExecutionBenchmarker()
    await benchmarker.initialize()
    
    results = await benchmarker.run_benchmark_suite(STANDARD_BENCHMARKS)
    benchmarker.export_results(results, "enhanced_executor_benchmarks.json")
    
    return results


# Global benchmarker instance
benchmarker_instance: Optional[ExecutionBenchmarker] = None


async def get_benchmarker() -> ExecutionBenchmarker:
    """Get the global benchmarker instance."""
    global benchmarker_instance
    
    if not benchmarker_instance:
        benchmarker_instance = ExecutionBenchmarker()
        await benchmarker_instance.initialize()
    
    return benchmarker_instance