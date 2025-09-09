"""
Enterprise Scenario Testing Execution Script

Script to execute comprehensive enterprise-scale scenario tests,
generate reports, and provide analysis of the multi-agent
orchestration system performance.
"""

import asyncio
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent / "apps" / "backend"))

from services.scenario_testing_framework import (
    ScenarioTestingFramework,
    ScenarioComplexity,
    TestCategory,
    get_scenario_framework
)
from services.enhanced_parallel_executor import get_enhanced_executor


class EnterpriseScenarioRunner:
    """
    Enterprise scenario runner for comprehensive system validation.
    """
    
    def __init__(self):
        self.framework: Optional[ScenarioTestingFramework] = None
        self.results: Dict[str, Any] = {}
        self.start_time = datetime.utcnow()
    
    async def initialize(self):
        """Initialize the scenario runner."""
        print("🚀 Initializing Enterprise Scenario Testing Framework...")
        
        try:
            self.framework = await get_scenario_framework()
            print("✅ Framework initialized successfully")
            
            # Verify executor is available
            executor = await get_enhanced_executor()
            print(f"✅ Enhanced Parallel Executor ready (max concurrency: {executor.max_concurrent_tasks})")
            
        except Exception as e:
            print(f"❌ Failed to initialize framework: {e}")
            raise
    
    async def run_scenario_suite(
        self,
        suite_name: str = "comprehensive",
        scenarios: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run a suite of scenarios."""
        
        if scenarios is None:
            scenarios = self._get_scenario_suite(suite_name)
        
        print(f"\n📋 Running {suite_name} scenario suite ({len(scenarios)} scenarios)")
        print("=" * 60)
        
        suite_results = {}
        
        for i, scenario_id in enumerate(scenarios, 1):
            scenario_def = self.framework.scenario_registry.get(scenario_id)
            if not scenario_def:
                print(f"⚠️  Scenario '{scenario_id}' not found, skipping...")
                continue
            
            print(f"\n[{i}/{len(scenarios)}] 🧪 Executing: {scenario_def.name}")
            print(f"    Category: {scenario_def.category.value}")
            print(f"    Complexity: {scenario_def.complexity.value}")
            print(f"    Tasks: {scenario_def.task_count}, Users: {scenario_def.concurrent_users}")
            
            try:
                # Execute scenario with progress indication
                start_time = datetime.utcnow()
                
                # Mock progress indication (in real implementation, you'd have progress callbacks)
                print("    Progress: Starting execution...", end="", flush=True)
                
                metrics = await self.framework.execute_scenario(scenario_id)
                
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()
                
                # Analyze results
                success = metrics.custom_metrics.get("success", False)
                success_rate = metrics.completed_tasks / max(1, metrics.total_tasks)
                
                print(f" ✅ Completed in {duration:.1f}s")
                print(f"    Results: {metrics.completed_tasks}/{metrics.total_tasks} tasks completed ({success_rate:.1%})")
                print(f"    Performance: {metrics.throughput_tasks_per_second:.1f} TPS, {metrics.avg_response_time_ms:.0f}ms avg")
                print(f"    Status: {'✅ PASS' if success else '❌ FAIL'}")
                
                suite_results[scenario_id] = metrics
                
            except Exception as e:
                print(f" ❌ Failed: {e}")
                print(f"    Status: ❌ ERROR")
                continue
        
        print(f"\n✅ Suite '{suite_name}' completed: {len(suite_results)}/{len(scenarios)} scenarios executed")
        
        return suite_results
    
    def _get_scenario_suite(self, suite_name: str) -> List[str]:
        """Get list of scenarios for a named suite."""
        
        suites = {
            "basic": [
                "basic_concurrency",
                "performance_benchmark"
            ],
            "comprehensive": [
                "basic_concurrency",
                "high_concurrency", 
                "performance_benchmark",
                "failure_recovery",
                "edge_conditions",
                "complex_prd_workflow"
            ],
            "enterprise": [
                "basic_concurrency",
                "high_concurrency",
                "enterprise_scale",
                "performance_benchmark",
                "failure_recovery",
                "edge_conditions",
                "complex_prd_workflow",
                "adaptive_agent_selection",
                "multi_tier_validation",
                "cross_domain_collaboration"
            ],
            "stress": [
                "high_concurrency",
                "enterprise_scale",
                "failure_recovery",
                "adaptive_agent_selection",
                "extreme_concurrency",
                "rapid_scaling"
            ],
            "load": [
                "extreme_concurrency", 
                "sustained_load",
                "rapid_scaling",
                "high_concurrency"
            ],
            "workflow": [
                "complex_prd_workflow",
                "multi_tier_validation",
                "cross_domain_collaboration",
                "adaptive_agent_selection"
            ],
            "all": list(self.framework.scenario_registry.keys()) if self.framework else []
        }
        
        return suites.get(suite_name, suites["basic"])
    
    async def generate_report(
        self,
        results: Dict[str, Any],
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        
        print("\n📊 Generating comprehensive test report...")
        
        report = await self.framework.generate_comprehensive_report(results)
        
        # Add execution metadata
        report["execution_metadata"] = {
            "execution_start": self.start_time.isoformat(),
            "execution_end": datetime.utcnow().isoformat(),
            "total_duration_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "framework_version": "1.0.0",
            "python_version": sys.version,
            "scenarios_attempted": len(results)
        }
        
        # Print summary to console
        self._print_report_summary(report)
        
        # Save to file if requested
        if output_file:
            try:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                
                print(f"📁 Report saved to: {output_path}")
                
            except Exception as e:
                print(f"⚠️  Failed to save report to file: {e}")
        
        return report
    
    def _print_report_summary(self, report: Dict[str, Any]):
        """Print report summary to console."""
        
        summary = report.get("summary", {})
        
        print("\n" + "=" * 60)
        print("📋 ENTERPRISE SCENARIO TEST REPORT SUMMARY")
        print("=" * 60)
        
        # Overall results
        print(f"🎯 Total Scenarios: {summary.get('total_scenarios', 0)}")
        print(f"✅ Successful Scenarios: {summary.get('successful_scenarios', 0)}")
        print(f"📈 Overall Success Rate: {summary.get('success_rate', 0):.1%}")
        print(f"📊 Total Tasks Executed: {summary.get('total_tasks_executed', 0):,}")
        print(f"✅ Tasks Completed: {summary.get('total_tasks_completed', 0):,}")
        print(f"❌ Tasks Failed: {summary.get('total_tasks_failed', 0):,}")
        print(f"🎭 Task Success Rate: {summary.get('overall_success_rate', 0):.1%}")
        
        # Performance metrics
        print(f"\n⚡ PERFORMANCE METRICS")
        print(f"🚀 Average Throughput: {summary.get('avg_throughput_tps', 0):.1f} tasks/second")
        print(f"⏱️  Average Response Time: {summary.get('avg_response_time_ms', 0):.0f}ms")
        
        # Resource analysis
        resource_analysis = report.get("resource_analysis", {})
        cpu_util = resource_analysis.get("cpu_utilization", {})
        memory_util = resource_analysis.get("memory_utilization", {})
        concurrency = resource_analysis.get("concurrency_analysis", {})
        
        print(f"\n💻 RESOURCE UTILIZATION")
        print(f"🖥️  CPU Usage: {cpu_util.get('avg', 0):.1f}% avg, {cpu_util.get('max', 0):.1f}% peak ({cpu_util.get('efficiency', 'unknown')})")
        print(f"💾 Memory Usage: {memory_util.get('avg', 0):.1f}% avg, {memory_util.get('max', 0):.1f}% peak ({memory_util.get('efficiency', 'unknown')})")
        print(f"⚡ Concurrency: {concurrency.get('avg_peak', 0):.0f} avg peak, {concurrency.get('max_peak', 0)} max peak ({concurrency.get('scalability', 'unknown')})")
        
        # Error analysis
        error_analysis = report.get("error_analysis", {})
        
        print(f"\n🚨 ERROR ANALYSIS")
        print(f"📉 Overall Error Rate: {error_analysis.get('overall_error_rate', 0):.1%}")
        print(f"📊 Max Error Rate: {error_analysis.get('max_error_rate', 0):.1%}")
        print(f"🔧 Reliability Assessment: {error_analysis.get('reliability_assessment', 'unknown').title()}")
        
        # Circuit breaker analysis
        cb_analysis = error_analysis.get("circuit_breaker_analysis", {})
        print(f"⚡ Circuit Breaker Events: {cb_analysis.get('total_activations', 0)}")
        
        # Scenario results
        scenario_results = report.get("scenario_results", {})
        if scenario_results:
            print(f"\n📋 SCENARIO RESULTS")
            for scenario_id, results in scenario_results.items():
                status = "✅ PASS" if results.get("success", False) else "❌ FAIL"
                print(f"  {scenario_id}: {status} ({results.get('completed_tasks', 0)} tasks, {results.get('throughput_tps', 0):.1f} TPS)")
        
        # Recommendations
        recommendations = report.get("recommendations", [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("=" * 60)
    
    async def run_custom_scenario(
        self,
        name: str,
        task_count: int,
        concurrent_users: int,
        duration_seconds: int,
        complexity: str = "moderate"
    ):
        """Run a custom scenario with specified parameters."""
        
        from services.scenario_testing_framework import ScenarioDefinition, AgentType
        from services.agent_orchestrator import AgentType
        
        # Create custom scenario
        custom_scenario = ScenarioDefinition(
            id=f"custom_{name}",
            name=f"Custom {name.title()} Test",
            description=f"Custom scenario: {name}",
            category=TestCategory.INTEGRATION,
            complexity=ScenarioComplexity(complexity),
            task_count=task_count,
            concurrent_users=concurrent_users,
            duration_seconds=duration_seconds,
            agent_types=list(AgentType)[:min(10, len(AgentType))],  # Use up to 10 agent types
            workflow_contexts=[{"type": "custom", "name": name}]
        )
        
        # Add to framework
        self.framework.scenario_registry[custom_scenario.id] = custom_scenario
        
        print(f"\n🧪 Running custom scenario: {name}")
        print(f"   Tasks: {task_count}, Users: {concurrent_users}, Duration: {duration_seconds}s")
        
        try:
            metrics = await self.framework.execute_scenario(custom_scenario.id)
            
            success_rate = metrics.completed_tasks / max(1, metrics.total_tasks)
            print(f"✅ Custom scenario completed:")
            print(f"   Success rate: {success_rate:.1%}")
            print(f"   Throughput: {metrics.throughput_tasks_per_second:.1f} TPS")
            print(f"   Response time: {metrics.avg_response_time_ms:.0f}ms avg")
            
            return {custom_scenario.id: metrics}
            
        finally:
            # Clean up
            if custom_scenario.id in self.framework.scenario_registry:
                del self.framework.scenario_registry[custom_scenario.id]
    
    async def shutdown(self):
        """Shutdown the scenario runner."""
        if self.framework:
            await self.framework.shutdown()
        print("🏁 Scenario testing framework shutdown complete")


async def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description="Enterprise Scenario Testing Runner")
    parser.add_argument("--suite", default="comprehensive", 
                       choices=["basic", "comprehensive", "enterprise", "stress", "load", "workflow", "all"],
                       help="Scenario suite to run")
    parser.add_argument("--scenarios", nargs="*", 
                       help="Specific scenarios to run (overrides suite)")
    parser.add_argument("--output", "-o", 
                       help="Output file for test report (JSON)")
    parser.add_argument("--custom", nargs=4, metavar=("NAME", "TASKS", "USERS", "DURATION"),
                       help="Run custom scenario: name task_count concurrent_users duration_seconds")
    parser.add_argument("--no-report", action="store_true",
                       help="Skip report generation")
    
    args = parser.parse_args()
    
    runner = EnterpriseScenarioRunner()
    
    try:
        await runner.initialize()
        
        results = {}
        
        # Run custom scenario if specified
        if args.custom:
            name, tasks, users, duration = args.custom
            custom_results = await runner.run_custom_scenario(
                name, int(tasks), int(users), int(duration)
            )
            results.update(custom_results)
        
        # Run scenario suite
        elif args.scenarios:
            suite_results = await runner.run_scenario_suite("custom", args.scenarios)
            results.update(suite_results)
        else:
            suite_results = await runner.run_scenario_suite(args.suite)
            results.update(suite_results)
        
        # Generate report unless disabled
        if not args.no_report and results:
            await runner.generate_report(results, args.output)
        
        # Determine overall success
        if results:
            successful_scenarios = sum(1 for metrics in results.values() 
                                     if metrics.custom_metrics.get("success", False))
            overall_success_rate = successful_scenarios / len(results)
            
            if overall_success_rate >= 0.8:
                print(f"\n🎉 OVERALL RESULT: SUCCESS ({overall_success_rate:.1%} scenarios passed)")
                return 0
            else:
                print(f"\n⚠️  OVERALL RESULT: PARTIAL SUCCESS ({overall_success_rate:.1%} scenarios passed)")
                return 1
        else:
            print("\n❌ OVERALL RESULT: NO SCENARIOS EXECUTED")
            return 2
    
    except KeyboardInterrupt:
        print("\n⏹️  Execution interrupted by user")
        return 130
    
    except Exception as e:
        print(f"\n💥 EXECUTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        await runner.shutdown()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)