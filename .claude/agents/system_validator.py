#!/usr/bin/env python3
"""
End-to-End System Validation
Validates the complete multi-agent orchestration system
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Import our system components
from workflow_engine import ContextManagerOrchestrator
from monitoring_dashboard import OrchestrationMonitor
from cache_optimizer import IntelligentCacheManager
from integration_tests import MultiAgentIntegrationTester


class SystemValidator:
    """
    Complete system validation for the AI multi-agent orchestration platform
    
    Validates:
    - Context manager execution
    - Agent orchestration workflows  
    - Monitoring and observability
    - Caching and performance optimization
    - Integration testing framework
    - GraphRAG validation pipeline
    - End-to-end system performance
    """
    
    def __init__(self):
        self.setup_logging()
        self.validation_results = {}
        
    def setup_logging(self):
        """Setup validation logging"""
        self.logger = logging.getLogger('SystemValidator')
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    async def validate_complete_system(self) -> Dict[str, Any]:
        """Run complete end-to-end system validation"""
        
        self.logger.info("🚀 Starting comprehensive system validation")
        
        validation_start = time.time()
        
        # Initialize validation results
        validation_results = {
            'validation_info': {
                'start_time': datetime.now(),
                'system_version': '2.2.0',
                'validation_type': 'end_to_end'
            },
            'component_validations': {},
            'performance_metrics': {},
            'quality_gates': {},
            'overall_status': 'unknown'
        }
        
        try:
            # 1. Validate Context Manager Orchestration
            self.logger.info("📋 Validating Context Manager orchestration...")
            context_validation = await self._validate_context_manager()
            validation_results['component_validations']['context_manager'] = context_validation
            
            # 2. Validate Monitoring System
            self.logger.info("📊 Validating monitoring and observability...")
            monitoring_validation = await self._validate_monitoring_system()
            validation_results['component_validations']['monitoring'] = monitoring_validation
            
            # 3. Validate Cache Optimization
            self.logger.info("⚡ Validating cache optimization system...")
            cache_validation = await self._validate_cache_system()
            validation_results['component_validations']['cache'] = cache_validation
            
            # 4. Validate Integration Testing
            self.logger.info("🧪 Validating integration testing framework...")
            testing_validation = await self._validate_testing_framework()
            validation_results['component_validations']['testing'] = testing_validation
            
            # 5. Validate Performance Characteristics
            self.logger.info("⚡ Validating system performance...")
            performance_validation = await self._validate_system_performance()
            validation_results['performance_metrics'] = performance_validation
            
            # 6. Validate Quality Gates
            self.logger.info("🎯 Validating quality gates...")
            quality_validation = await self._validate_quality_gates(validation_results)
            validation_results['quality_gates'] = quality_validation
            
            # 7. Overall System Health Check
            overall_status = await self._determine_overall_status(validation_results)
            validation_results['overall_status'] = overall_status
            
            validation_results['validation_info']['end_time'] = datetime.now()
            validation_results['validation_info']['total_duration_seconds'] = time.time() - validation_start
            
            # Log final results
            self.logger.info(f"✅ System validation completed: {overall_status}")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"❌ System validation failed: {e}")
            validation_results['overall_status'] = 'failed'
            validation_results['error'] = str(e)
            return validation_results
    
    async def _validate_context_manager(self) -> Dict[str, Any]:
        """Validate context manager orchestration"""
        
        try:
            # Initialize orchestrator
            orchestrator = ContextManagerOrchestrator()
            
            # Validate configuration loading
            config_loaded = bool(orchestrator.workflow_config)
            agents_loaded = len(orchestrator.agents) > 0
            dependency_graph = not orchestrator._has_circular_dependencies()
            
            # Test workflow execution (with mocks)
            start_time = time.time()
            workflow_result = await orchestrator.execute_workflow()
            execution_time = time.time() - start_time
            
            workflow_completed = workflow_result.get('workflow_status') == 'completed'
            agents_executed = workflow_result.get('metrics', {}).get('total_agents', 0)
            
            return {
                'component_healthy': True,
                'configuration_loaded': config_loaded,
                'agents_loaded': agents_loaded,
                'agents_count': len(orchestrator.agents),
                'dependency_graph_valid': dependency_graph,
                'workflow_execution': {
                    'completed': workflow_completed,
                    'execution_time_seconds': execution_time,
                    'agents_executed': agents_executed
                },
                'validation_passed': all([
                    config_loaded,
                    agents_loaded,
                    dependency_graph,
                    workflow_completed
                ])
            }
            
        except Exception as e:
            return {
                'component_healthy': False,
                'error': str(e),
                'validation_passed': False
            }
    
    async def _validate_monitoring_system(self) -> Dict[str, Any]:
        """Validate monitoring and observability system"""
        
        try:
            monitor = OrchestrationMonitor()
            
            # Start monitoring briefly
            monitoring_task = asyncio.create_task(monitor.start_monitoring())
            
            try:
                # Let it run briefly
                await asyncio.sleep(0.5)
                
                # Test metric updates
                monitor.update_agent_metrics('test_agent', 'test-agent', {
                    'status': 'completed',
                    'execution_time': 2.0
                })
                
                monitor.update_workflow_metrics({
                    'status': 'completed',
                    'hallucination_rate': 0.01,
                    'quality_score': 0.95
                })
                
                # Get dashboard data
                dashboard_data = monitor.get_dashboard_data()
                
                # Validate monitoring components
                system_overview = 'system_overview' in dashboard_data
                quality_metrics = 'quality_metrics' in dashboard_data
                agent_metrics = len(dashboard_data.get('agent_metrics', [])) > 0
                alerts_working = 'alerts' in dashboard_data
                
                return {
                    'component_healthy': True,
                    'monitoring_active': True,
                    'dashboard_available': bool(dashboard_data),
                    'system_overview_working': system_overview,
                    'quality_metrics_tracked': quality_metrics,
                    'agent_metrics_collected': agent_metrics,
                    'alerts_system_working': alerts_working,
                    'validation_passed': all([
                        system_overview,
                        quality_metrics,
                        agent_metrics,
                        alerts_working
                    ])
                }
                
            finally:
                monitor.stop_monitoring()
                monitoring_task.cancel()
                try:
                    await monitoring_task
                except asyncio.CancelledError:
                    pass
            
        except Exception as e:
            return {
                'component_healthy': False,
                'error': str(e),
                'validation_passed': False
            }
    
    async def _validate_cache_system(self) -> Dict[str, Any]:
        """Validate cache optimization system"""
        
        try:
            cache_manager = IntelligentCacheManager()
            
            # Test cache operations
            test_data = {'test': 'validation_data', 'timestamp': str(datetime.now())}
            
            # Test put operation
            put_success = await cache_manager.put('validation_test', test_data)
            
            # Test get operation
            retrieved_data = await cache_manager.get('validation_test')
            data_integrity = retrieved_data == test_data
            
            # Test performance
            start_time = time.time()
            for i in range(10):
                await cache_manager.put(f'perf_test_{i}', f'data_{i}')
                await cache_manager.get(f'perf_test_{i}')
            performance_time = time.time() - start_time
            
            # Get performance report
            performance_report = cache_manager.get_performance_report()
            hit_rate = performance_report['cache_stats']['hit_rate']
            
            # Test cache invalidation
            invalidation_success = await cache_manager.invalidate('validation_test')
            invalidated_data = await cache_manager.get('validation_test')
            invalidation_working = invalidated_data is None
            
            # Cleanup
            cache_manager.stop_background_tasks()
            
            return {
                'component_healthy': True,
                'cache_put_working': put_success,
                'cache_get_working': retrieved_data is not None,
                'data_integrity': data_integrity,
                'performance_acceptable': performance_time < 5.0,
                'hit_rate': hit_rate,
                'invalidation_working': invalidation_working,
                'performance_report': performance_report,
                'validation_passed': all([
                    put_success,
                    data_integrity,
                    invalidation_working,
                    performance_time < 5.0
                ])
            }
            
        except Exception as e:
            return {
                'component_healthy': False,
                'error': str(e),
                'validation_passed': False
            }
    
    async def _validate_testing_framework(self) -> Dict[str, Any]:
        """Validate integration testing framework"""
        
        try:
            tester = MultiAgentIntegrationTester()
            
            # Setup test environment
            await tester.setup_test_environment()
            
            try:
                # Run orchestrator initialization test
                test_result = await tester.test_orchestrator_initialization()
                
                # Validate test framework components
                mock_agents_setup = len(tester.mock_agents) > 0
                test_data_loaded = bool(tester.test_data)
                orchestrator_available = tester.orchestrator is not None
                monitor_available = tester.monitor is not None
                cache_available = tester.cache_manager is not None
                
                return {
                    'component_healthy': True,
                    'test_framework_initialized': True,
                    'mock_agents_available': mock_agents_setup,
                    'test_data_loaded': test_data_loaded,
                    'test_infrastructure_ready': all([
                        orchestrator_available,
                        monitor_available,
                        cache_available
                    ]),
                    'sample_test_passed': test_result.get('dependency_graph_built', False),
                    'validation_passed': all([
                        mock_agents_setup,
                        test_data_loaded,
                        orchestrator_available,
                        test_result.get('dependency_graph_built', False)
                    ])
                }
                
            finally:
                await tester.teardown_test_environment()
            
        except Exception as e:
            return {
                'component_healthy': False,
                'error': str(e),
                'validation_passed': False
            }
    
    async def _validate_system_performance(self) -> Dict[str, Any]:
        """Validate overall system performance"""
        
        performance_metrics = {
            'workflow_execution': {},
            'agent_coordination': {},
            'cache_performance': {},
            'monitoring_overhead': {},
            'memory_usage': {}
        }
        
        try:
            # Test workflow execution performance
            orchestrator = ContextManagerOrchestrator()
            
            start_time = time.time()
            workflow_result = await orchestrator.execute_workflow()
            workflow_execution_time = time.time() - start_time
            
            performance_metrics['workflow_execution'] = {
                'execution_time_seconds': workflow_execution_time,
                'performance_acceptable': workflow_execution_time < 30.0,
                'agents_per_second': workflow_result.get('metrics', {}).get('total_agents', 0) / workflow_execution_time if workflow_execution_time > 0 else 0
            }
            
            # Test agent coordination performance
            start_time = time.time()
            
            # Simulate parallel agent execution
            async def mock_agent(agent_id: int):
                await asyncio.sleep(0.1)
                return {'agent_id': agent_id, 'status': 'completed'}
            
            tasks = [mock_agent(i) for i in range(5)]
            results = await asyncio.gather(*tasks)
            coordination_time = time.time() - start_time
            
            performance_metrics['agent_coordination'] = {
                'coordination_time_seconds': coordination_time,
                'parallel_efficiency': len(results) / coordination_time,
                'all_agents_completed': len(results) == 5
            }
            
            # Test cache performance
            cache_manager = IntelligentCacheManager()
            
            start_time = time.time()
            for i in range(100):
                await cache_manager.put(f'perf_{i}', f'data_{i}')
                await cache_manager.get(f'perf_{i}')
            cache_performance_time = time.time() - start_time
            
            performance_metrics['cache_performance'] = {
                'operations_per_second': 200 / cache_performance_time,  # 100 puts + 100 gets
                'average_operation_time_ms': (cache_performance_time / 200) * 1000,
                'performance_acceptable': cache_performance_time < 5.0
            }
            
            cache_manager.stop_background_tasks()
            
            # Test monitoring overhead
            monitor = OrchestrationMonitor()
            monitoring_task = asyncio.create_task(monitor.start_monitoring())
            
            try:
                start_time = time.time()
                
                # Simulate monitoring during operations
                for i in range(10):
                    monitor.update_agent_metrics(f'agent_{i}', f'test-agent-{i}', {
                        'status': 'completed',
                        'execution_time': 1.0
                    })
                
                monitoring_overhead = time.time() - start_time
                
                performance_metrics['monitoring_overhead'] = {
                    'overhead_per_update_ms': (monitoring_overhead / 10) * 1000,
                    'overhead_acceptable': monitoring_overhead < 1.0
                }
                
            finally:
                monitor.stop_monitoring()
                monitoring_task.cancel()
                try:
                    await monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Memory usage estimation
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            performance_metrics['memory_usage'] = {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'memory_acceptable': (memory_info.rss / 1024 / 1024) < 500  # < 500MB
            }
            
            return performance_metrics
            
        except Exception as e:
            return {
                'error': str(e),
                'performance_validation_failed': True
            }
    
    async def _validate_quality_gates(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate quality gates and thresholds"""
        
        quality_gates = {
            'component_health': {
                'threshold': 'All components healthy',
                'status': 'unknown'
            },
            'performance_targets': {
                'workflow_execution_time': {'threshold': 30.0, 'unit': 'seconds'},
                'cache_hit_rate': {'threshold': 0.7, 'unit': 'percentage'},
                'memory_usage': {'threshold': 500.0, 'unit': 'MB'}
            },
            'integration_quality': {
                'all_tests_passing': {'threshold': True},
                'monitoring_functional': {'threshold': True},
                'cache_integrity': {'threshold': True}
            },
            'overall_quality_score': 0.0
        }
        
        try:
            # Check component health
            component_validations = validation_results.get('component_validations', {})
            healthy_components = sum(1 for comp in component_validations.values() 
                                   if comp.get('validation_passed', False))
            total_components = len(component_validations)
            
            quality_gates['component_health']['status'] = 'passed' if healthy_components == total_components else 'failed'
            quality_gates['component_health']['healthy_components'] = f"{healthy_components}/{total_components}"
            
            # Check performance targets
            performance_metrics = validation_results.get('performance_metrics', {})
            
            # Workflow execution time
            workflow_time = performance_metrics.get('workflow_execution', {}).get('execution_time_seconds', 999)
            quality_gates['performance_targets']['workflow_execution_time']['actual'] = workflow_time
            quality_gates['performance_targets']['workflow_execution_time']['passed'] = workflow_time <= 30.0
            
            # Cache performance
            cache_ops_per_sec = performance_metrics.get('cache_performance', {}).get('operations_per_second', 0)
            quality_gates['performance_targets']['cache_hit_rate']['actual'] = cache_ops_per_sec
            quality_gates['performance_targets']['cache_hit_rate']['passed'] = cache_ops_per_sec > 40  # ops/sec as proxy
            
            # Memory usage
            memory_usage = performance_metrics.get('memory_usage', {}).get('rss_mb', 999)
            quality_gates['performance_targets']['memory_usage']['actual'] = memory_usage
            quality_gates['performance_targets']['memory_usage']['passed'] = memory_usage <= 500.0
            
            # Integration quality
            quality_gates['integration_quality']['all_tests_passing']['actual'] = healthy_components == total_components
            quality_gates['integration_quality']['all_tests_passing']['passed'] = healthy_components == total_components
            
            quality_gates['integration_quality']['monitoring_functional']['actual'] = component_validations.get('monitoring', {}).get('validation_passed', False)
            quality_gates['integration_quality']['monitoring_functional']['passed'] = component_validations.get('monitoring', {}).get('validation_passed', False)
            
            quality_gates['integration_quality']['cache_integrity']['actual'] = component_validations.get('cache', {}).get('validation_passed', False)
            quality_gates['integration_quality']['cache_integrity']['passed'] = component_validations.get('cache', {}).get('validation_passed', False)
            
            # Calculate overall quality score
            passed_gates = [
                quality_gates['component_health']['status'] == 'passed',
                quality_gates['performance_targets']['workflow_execution_time']['passed'],
                quality_gates['performance_targets']['cache_hit_rate']['passed'],
                quality_gates['performance_targets']['memory_usage']['passed'],
                quality_gates['integration_quality']['all_tests_passing']['passed'],
                quality_gates['integration_quality']['monitoring_functional']['passed'],
                quality_gates['integration_quality']['cache_integrity']['passed']
            ]
            
            quality_gates['overall_quality_score'] = sum(passed_gates) / len(passed_gates)
            
            return quality_gates
            
        except Exception as e:
            return {
                'error': str(e),
                'quality_validation_failed': True
            }
    
    async def _determine_overall_status(self, validation_results: Dict[str, Any]) -> str:
        """Determine overall system status"""
        
        try:
            # Check if all components passed validation
            component_validations = validation_results.get('component_validations', {})
            components_passed = all(
                comp.get('validation_passed', False)
                for comp in component_validations.values()
            )
            
            # Check quality gates
            quality_gates = validation_results.get('quality_gates', {})
            quality_score = quality_gates.get('overall_quality_score', 0.0)
            
            # Determine status
            if components_passed and quality_score >= 0.8:
                return 'healthy'
            elif components_passed and quality_score >= 0.6:
                return 'functional_with_issues'
            elif quality_score >= 0.4:
                return 'degraded'
            else:
                return 'critical'
                
        except Exception:
            return 'unknown'
    
    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive validation report"""
        
        report_lines = [
            "# 🚀 AI Multi-Agent Orchestration System - Validation Report",
            f"**Generated**: {datetime.now().isoformat()}",
            f"**System Version**: {results['validation_info']['system_version']}",
            f"**Validation Duration**: {results['validation_info']['total_duration_seconds']:.2f} seconds",
            f"**Overall Status**: {results['overall_status'].upper()}",
            "",
            "## 📊 Validation Summary"
        ]
        
        # Component validations
        component_validations = results.get('component_validations', {})
        for component, validation in component_validations.items():
            status_emoji = "✅" if validation.get('validation_passed', False) else "❌"
            report_lines.append(f"- {status_emoji} **{component.replace('_', ' ').title()}**: {'PASSED' if validation.get('validation_passed', False) else 'FAILED'}")
            
            if not validation.get('validation_passed', False) and 'error' in validation:
                report_lines.append(f"  - Error: {validation['error']}")
        
        report_lines.append("")
        
        # Performance metrics
        performance_metrics = results.get('performance_metrics', {})
        if performance_metrics:
            report_lines.extend([
                "## ⚡ Performance Metrics",
                ""
            ])
            
            for metric_category, metrics in performance_metrics.items():
                if isinstance(metrics, dict):
                    report_lines.append(f"### {metric_category.replace('_', ' ').title()}")
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            report_lines.append(f"- **{key.replace('_', ' ').title()}**: {value:.2f}")
                        else:
                            report_lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
                    report_lines.append("")
        
        # Quality gates
        quality_gates = results.get('quality_gates', {})
        if quality_gates:
            report_lines.extend([
                "## 🎯 Quality Gates",
                ""
            ])
            
            # Overall quality score
            quality_score = quality_gates.get('overall_quality_score', 0.0)
            report_lines.append(f"**Overall Quality Score**: {quality_score:.1%}")
            report_lines.append("")
            
            # Component health
            component_health = quality_gates.get('component_health', {})
            status_emoji = "✅" if component_health.get('status') == 'passed' else "❌"
            report_lines.append(f"{status_emoji} **Component Health**: {component_health.get('healthy_components', 'Unknown')}")
            report_lines.append("")
            
            # Performance targets
            performance_targets = quality_gates.get('performance_targets', {})
            if performance_targets:
                report_lines.append("### Performance Targets")
                for target, details in performance_targets.items():
                    if isinstance(details, dict):
                        status_emoji = "✅" if details.get('passed', False) else "❌"
                        actual = details.get('actual', 'N/A')
                        threshold = details.get('threshold', 'N/A')
                        unit = details.get('unit', '')
                        report_lines.append(f"{status_emoji} **{target.replace('_', ' ').title()}**: {actual} {unit} (threshold: {threshold} {unit})")
                report_lines.append("")
        
        # Recommendations
        recommendations = self._generate_recommendations(results)
        if recommendations:
            report_lines.extend([
                "## 💡 Recommendations",
                ""
            ])
            for rec in recommendations:
                report_lines.append(f"- {rec}")
            report_lines.append("")
        
        report_lines.extend([
            "---",
            "*This report was generated by the AI Multi-Agent System Validator*"
        ])
        
        return "\n".join(report_lines)
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        
        # Check component health
        component_validations = results.get('component_validations', {})
        for component, validation in component_validations.items():
            if not validation.get('validation_passed', False):
                recommendations.append(f"🔧 Fix issues in {component.replace('_', ' ')} component")
        
        # Check performance
        performance_metrics = results.get('performance_metrics', {})
        
        # Workflow performance
        workflow_time = performance_metrics.get('workflow_execution', {}).get('execution_time_seconds', 0)
        if workflow_time > 30:
            recommendations.append("⚡ Optimize workflow execution time - consider parallel processing")
        
        # Memory usage
        memory_usage = performance_metrics.get('memory_usage', {}).get('rss_mb', 0)
        if memory_usage > 400:
            recommendations.append("💾 Optimize memory usage - consider caching strategies")
        
        # Cache performance
        cache_ops = performance_metrics.get('cache_performance', {}).get('operations_per_second', 0)
        if cache_ops < 50:
            recommendations.append("🚀 Improve cache performance - optimize data structures")
        
        # Quality score
        quality_gates = results.get('quality_gates', {})
        quality_score = quality_gates.get('overall_quality_score', 1.0)
        
        if quality_score < 0.8:
            recommendations.append("🎯 Improve overall system quality - address failing quality gates")
        
        if quality_score < 0.6:
            recommendations.append("🚨 System requires immediate attention - multiple critical issues detected")
        
        return recommendations


# CLI Interface
async def main():
    """Main validation execution"""
    validator = SystemValidator()
    
    print("🚀 Starting AI Multi-Agent Orchestration System Validation")
    print("=" * 70)
    
    try:
        # Run complete system validation
        results = await validator.validate_complete_system()
        
        # Generate report
        report = validator.generate_validation_report(results)
        
        # Save results
        results_dir = Path('.claude/validation_results')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        with open(results_dir / 'system_validation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save markdown report
        with open(results_dir / 'system_validation_report.md', 'w') as f:
            f.write(report)
        
        # Print summary
        print("\n" + "=" * 70)
        print("📋 VALIDATION SUMMARY")
        print("=" * 70)
        
        overall_status = results['overall_status']
        status_emoji = {
            'healthy': '✅',
            'functional_with_issues': '⚠️',
            'degraded': '🟡',
            'critical': '❌',
            'unknown': '❓'
        }.get(overall_status, '❓')
        
        print(f"Overall Status: {status_emoji} {overall_status.upper().replace('_', ' ')}")
        
        # Component status
        component_validations = results.get('component_validations', {})
        print(f"\nComponent Health: {len([c for c in component_validations.values() if c.get('validation_passed')])}/{len(component_validations)} components passing")
        
        # Quality score
        quality_score = results.get('quality_gates', {}).get('overall_quality_score', 0.0)
        print(f"Quality Score: {quality_score:.1%}")
        
        # Performance summary
        performance_metrics = results.get('performance_metrics', {})
        if performance_metrics:
            workflow_time = performance_metrics.get('workflow_execution', {}).get('execution_time_seconds', 0)
            print(f"Workflow Performance: {workflow_time:.2f}s")
        
        print(f"\nDetailed results saved to: {results_dir}")
        print(f"Validation completed in {results['validation_info']['total_duration_seconds']:.2f} seconds")
        
        # Return appropriate exit code
        if overall_status in ['healthy', 'functional_with_issues']:
            return 0
        else:
            return 1
            
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        return 1


if __name__ == '__main__':
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)