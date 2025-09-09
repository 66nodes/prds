#!/usr/bin/env python3
"""
Performance Benchmarking Suite for Strategic Planning Platform

Automated performance testing and benchmarking across all system components
with comprehensive metrics collection and threshold enforcement.
"""

import asyncio
import json
import time
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import aiohttp
import psutil
import sys
import argparse
from dataclasses import dataclass, asdict

# Performance test configuration
@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarks."""
    base_url: str = "http://localhost:8000"
    frontend_url: str = "http://localhost:3000"
    concurrent_users: int = 10
    test_duration: int = 60  # seconds
    ramp_up_time: int = 10  # seconds
    request_timeout: int = 30  # seconds
    
    # Performance thresholds
    max_response_time_p95: float = 500.0  # milliseconds
    max_response_time_p99: float = 1000.0  # milliseconds
    max_error_rate: float = 0.01  # 1%
    min_throughput: float = 100.0  # requests/second
    max_memory_usage: float = 500.0  # MB
    max_cpu_usage: float = 80.0  # percentage

@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""
    endpoint: str
    method: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput: float  # requests/second
    error_rate: float
    memory_usage: float  # MB
    cpu_usage: float  # percentage
    timestamp: datetime
    duration: float  # seconds


class PerformanceBenchmarker:
    """Main performance benchmarking class."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def benchmark_endpoint(
        self, 
        endpoint: str, 
        method: str = "GET", 
        payload: Optional[Dict] = None,
        headers: Optional[Dict] = None
    ) -> BenchmarkResult:
        """Benchmark a specific endpoint."""
        print(f"🔍 Benchmarking {method} {endpoint}")
        
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        # System monitoring
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = process.cpu_percent()
        
        start_time = time.time()
        end_time = start_time + self.config.test_duration
        
        # Create tasks for concurrent requests
        tasks = []
        for i in range(self.config.concurrent_users):
            # Stagger task creation for ramp-up
            delay = (i / self.config.concurrent_users) * self.config.ramp_up_time
            task = asyncio.create_task(
                self._run_user_session(endpoint, method, payload, headers, delay, end_time)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        for result in results:
            if isinstance(result, Exception):
                failed_requests += 1
                print(f"❌ Task failed: {result}")
            else:
                times, successes, failures = result
                response_times.extend(times)
                successful_requests += successes
                failed_requests += failures
        
        duration = time.time() - start_time
        
        # Calculate metrics
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            p50_response_time = statistics.median(response_times)
            p95_response_time = self._percentile(response_times, 0.95)
            p99_response_time = self._percentile(response_times, 0.99)
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p50_response_time = p95_response_time = p99_response_time = 0
        
        total_requests = successful_requests + failed_requests
        throughput = total_requests / duration if duration > 0 else 0
        error_rate = failed_requests / total_requests if total_requests > 0 else 1.0
        
        # System metrics
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        end_cpu = process.cpu_percent()
        memory_usage = end_memory - start_memory
        cpu_usage = end_cpu
        
        result = BenchmarkResult(
            endpoint=endpoint,
            method=method,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            throughput=throughput,
            error_rate=error_rate,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            timestamp=datetime.utcnow(),
            duration=duration
        )
        
        self.results.append(result)
        self._print_result(result)
        return result
    
    async def _run_user_session(
        self, 
        endpoint: str, 
        method: str, 
        payload: Optional[Dict], 
        headers: Optional[Dict],
        delay: float,
        end_time: float
    ) -> Tuple[List[float], int, int]:
        """Run a single user session."""
        await asyncio.sleep(delay)
        
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        while time.time() < end_time:
            try:
                start = time.time()
                
                async with self.session.request(
                    method, 
                    f"{self.config.base_url}{endpoint}",
                    json=payload,
                    headers=headers
                ) as response:
                    await response.read()  # Ensure response body is read
                    
                    response_time = (time.time() - start) * 1000  # Convert to milliseconds
                    response_times.append(response_time)
                    
                    if 200 <= response.status < 300:
                        successful_requests += 1
                    else:
                        failed_requests += 1
                        
            except Exception as e:
                failed_requests += 1
                print(f"⚠️ Request failed: {e}")
            
            # Small delay between requests
            await asyncio.sleep(0.01)
        
        return response_times, successful_requests, failed_requests
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * percentile
        f = int(k)
        c = k - f
        
        if f == len(sorted_data) - 1:
            return sorted_data[f]
        else:
            return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
    
    def _print_result(self, result: BenchmarkResult):
        """Print benchmark result."""
        print(f"""
📊 Benchmark Results for {result.method} {result.endpoint}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 Requests: {result.total_requests} total, {result.successful_requests} success, {result.failed_requests} failed
⚡ Throughput: {result.throughput:.2f} req/s
⏱️  Response Times (ms):
   • Average: {result.avg_response_time:.2f}
   • Min: {result.min_response_time:.2f}
   • Max: {result.max_response_time:.2f}
   • P50: {result.p50_response_time:.2f}
   • P95: {result.p95_response_time:.2f}
   • P99: {result.p99_response_time:.2f}
❌ Error Rate: {result.error_rate:.2%}
💾 Memory Usage: {result.memory_usage:.2f} MB
🖥️  CPU Usage: {result.cpu_usage:.2f}%
⏲️  Duration: {result.duration:.2f}s
        """)
    
    def validate_thresholds(self) -> Tuple[bool, List[str]]:
        """Validate results against performance thresholds."""
        violations = []
        
        for result in self.results:
            endpoint_name = f"{result.method} {result.endpoint}"
            
            if result.p95_response_time > self.config.max_response_time_p95:
                violations.append(
                    f"{endpoint_name}: P95 response time {result.p95_response_time:.2f}ms "
                    f"exceeds threshold {self.config.max_response_time_p95}ms"
                )
            
            if result.p99_response_time > self.config.max_response_time_p99:
                violations.append(
                    f"{endpoint_name}: P99 response time {result.p99_response_time:.2f}ms "
                    f"exceeds threshold {self.config.max_response_time_p99}ms"
                )
            
            if result.error_rate > self.config.max_error_rate:
                violations.append(
                    f"{endpoint_name}: Error rate {result.error_rate:.2%} "
                    f"exceeds threshold {self.config.max_error_rate:.2%}"
                )
            
            if result.throughput < self.config.min_throughput:
                violations.append(
                    f"{endpoint_name}: Throughput {result.throughput:.2f} req/s "
                    f"below threshold {self.config.min_throughput} req/s"
                )
            
            if result.memory_usage > self.config.max_memory_usage:
                violations.append(
                    f"{endpoint_name}: Memory usage {result.memory_usage:.2f}MB "
                    f"exceeds threshold {self.config.max_memory_usage}MB"
                )
            
            if result.cpu_usage > self.config.max_cpu_usage:
                violations.append(
                    f"{endpoint_name}: CPU usage {result.cpu_usage:.2f}% "
                    f"exceeds threshold {self.config.max_cpu_usage}%"
                )
        
        return len(violations) == 0, violations
    
    def export_results(self, filepath: str):
        """Export results to JSON file."""
        data = {
            "config": asdict(self.config),
            "results": [asdict(result) for result in self.results],
            "summary": self._generate_summary(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"📄 Results exported to {filepath}")
    
    def _generate_summary(self) -> Dict:
        """Generate summary statistics."""
        if not self.results:
            return {}
        
        total_requests = sum(r.total_requests for r in self.results)
        successful_requests = sum(r.successful_requests for r in self.results)
        failed_requests = sum(r.failed_requests for r in self.results)
        
        avg_response_times = [r.avg_response_time for r in self.results]
        p95_response_times = [r.p95_response_time for r in self.results]
        p99_response_times = [r.p99_response_time for r in self.results]
        throughputs = [r.throughput for r in self.results]
        error_rates = [r.error_rate for r in self.results]
        
        return {
            "total_endpoints_tested": len(self.results),
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "overall_success_rate": successful_requests / total_requests if total_requests > 0 else 0,
            "avg_response_time": {
                "mean": statistics.mean(avg_response_times),
                "median": statistics.median(avg_response_times),
                "min": min(avg_response_times),
                "max": max(avg_response_times)
            },
            "p95_response_time": {
                "mean": statistics.mean(p95_response_times),
                "median": statistics.median(p95_response_times),
                "min": min(p95_response_times),
                "max": max(p95_response_times)
            },
            "p99_response_time": {
                "mean": statistics.mean(p99_response_times),
                "median": statistics.median(p99_response_times),
                "min": min(p99_response_times),
                "max": max(p99_response_times)
            },
            "throughput": {
                "mean": statistics.mean(throughputs),
                "median": statistics.median(throughputs),
                "min": min(throughputs),
                "max": max(throughputs),
                "total": sum(throughputs)
            },
            "error_rate": {
                "mean": statistics.mean(error_rates),
                "median": statistics.median(error_rates),
                "min": min(error_rates),
                "max": max(error_rates)
            }
        }


async def run_api_benchmarks(config: BenchmarkConfig):
    """Run comprehensive API performance benchmarks."""
    print("🚀 Starting API Performance Benchmarks")
    print(f"Configuration: {config.concurrent_users} users, {config.test_duration}s duration")
    
    async with PerformanceBenchmarker(config) as benchmarker:
        
        # Core API endpoints
        endpoints = [
            ("/health", "GET"),
            ("/api/v1/auth/login", "POST", {"email": "testexample.com", "password": "test"}),
            ("/api/v1/projects", "GET"),
            ("/api/v1/projects", "POST", {
                "title": "Performance Test Project",
                "description": "Created for performance testing"
            }),
            ("/api/v1/projects/test-project/prds", "GET"),
            ("/api/v1/agents", "GET"),
            ("/api/v1/dashboard/analytics", "GET"),
        ]
        
        # Run benchmarks
        for endpoint_config in endpoints:
            endpoint = endpoint_config[0]
            method = endpoint_config[1]
            payload = endpoint_config[2] if len(endpoint_config) > 2 else None
            
            await benchmarker.benchmark_endpoint(endpoint, method, payload)
        
        # Validate thresholds
        passed, violations = benchmarker.validate_thresholds()
        
        if passed:
            print("✅ All performance thresholds met!")
        else:
            print("❌ Performance threshold violations:")
            for violation in violations:
                print(f"  • {violation}")
        
        # Export results
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        results_file = f"reports/performance_benchmark_{timestamp}.json"
        benchmarker.export_results(results_file)
        
        return passed, benchmarker.results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Performance Benchmarking Suite")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--frontend-url", default="http://localhost:3000", help="Frontend URL")
    parser.add_argument("--users", type=int, default=10, help="Concurrent users")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--ramp-up", type=int, default=10, help="Ramp-up time in seconds")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--export", default="reports/performance_results.json", help="Export file path")
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        base_url=args.base_url,
        frontend_url=args.frontend_url,
        concurrent_users=args.users,
        test_duration=args.duration,
        ramp_up_time=args.ramp_up,
        request_timeout=args.timeout
    )
    
    try:
        passed, results = asyncio.run(run_api_benchmarks(config))
        
        print("\n" + "="*60)
        print("📊 Performance Benchmark Summary")
        print("="*60)
        
        if passed:
            print("🎉 All performance benchmarks passed!")
            sys.exit(0)
        else:
            print("💥 Some performance benchmarks failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()