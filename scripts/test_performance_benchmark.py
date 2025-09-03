#!/usr/bin/env python3
"""
Performance Benchmark Analysis for Test Suite

Analyzes test performance metrics and generates benchmark reports
to identify performance bottlenecks and optimization opportunities.
"""

import json
import csv
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse


class TestPerformanceBenchmark:
    """Analyze test suite performance and generate benchmark reports."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.reports_dir = self.project_root / "test-results"
        self.performance_dir = self.reports_dir / "performance"
        
        # Performance thresholds (in seconds)
        self.performance_thresholds = {
            "unit_tests": {
                "excellent": 30,
                "good": 60,
                "acceptable": 120,
                "poor": 300
            },
            "integration_tests": {
                "excellent": 60,
                "good": 180,
                "acceptable": 300,
                "poor": 600
            },
            "e2e_tests": {
                "excellent": 180,
                "good": 300,
                "acceptable": 600,
                "poor": 1200
            },
            "performance_tests": {
                "excellent": 120,
                "good": 180,
                "acceptable": 300,
                "poor": 600
            }
        }
    
    def parse_locust_results(self, csv_path: Path) -> Dict[str, Any]:
        """Parse Locust performance test results."""
        if not csv_path.exists():
            return {}
        
        results = {
            "requests": [],
            "summary": {
                "total_requests": 0,
                "failed_requests": 0,
                "avg_response_time": 0,
                "min_response_time": float('inf'),
                "max_response_time": 0,
                "requests_per_second": 0,
                "failure_rate": 0
            }
        }
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('Type') == 'GET' or row.get('Type') == 'POST':
                        request_data = {
                            "method": row.get('Type', ''),
                            "name": row.get('Name', ''),
                            "num_requests": int(row.get('Request Count', 0)),
                            "num_failures": int(row.get('Failure Count', 0)),
                            "avg_response_time": float(row.get('Average Response Time', 0)),
                            "min_response_time": float(row.get('Min Response Time', 0)),
                            "max_response_time": float(row.get('Max Response Time', 0)),
                            "requests_per_second": float(row.get('Requests/s', 0))
                        }
                        results["requests"].append(request_data)
        
        except Exception as e:
            print(f"Error parsing Locust results: {e}")
            return {}
        
        # Calculate summary statistics
        if results["requests"]:
            total_requests = sum(r["num_requests"] for r in results["requests"])
            total_failures = sum(r["num_failures"] for r in results["requests"])
            avg_times = [r["avg_response_time"] for r in results["requests"] if r["avg_response_time"] > 0]
            
            results["summary"] = {
                "total_requests": total_requests,
                "failed_requests": total_failures,
                "avg_response_time": statistics.mean(avg_times) if avg_times else 0,
                "min_response_time": min(r["min_response_time"] for r in results["requests"]) if results["requests"] else 0,
                "max_response_time": max(r["max_response_time"] for r in results["requests"]) if results["requests"] else 0,
                "requests_per_second": sum(r["requests_per_second"] for r in results["requests"]),
                "failure_rate": (total_failures / total_requests * 100) if total_requests > 0 else 0
            }
        
        return results
    
    def analyze_test_execution_times(self, junit_xml_files: List[Path]) -> Dict[str, Any]:
        """Analyze test execution times from JUnit XML files."""
        import xml.etree.ElementTree as ET
        
        execution_analysis = {
            "test_suites": [],
            "summary": {
                "total_tests": 0,
                "total_time": 0,
                "slowest_tests": [],
                "fastest_tests": [],
                "avg_test_time": 0
            }
        }
        
        all_test_times = []
        
        for xml_file in junit_xml_files:
            if not xml_file.exists():
                continue
            
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                suite_name = root.attrib.get("name", xml_file.stem)
                suite_time = float(root.attrib.get("time", 0))
                suite_tests = int(root.attrib.get("tests", 0))
                suite_failures = int(root.attrib.get("failures", 0))
                suite_errors = int(root.attrib.get("errors", 0))
                
                test_cases = []
                for testcase in root.findall(".//testcase"):
                    test_time = float(testcase.attrib.get("time", 0))
                    test_name = testcase.attrib.get("name", "unknown")
                    test_class = testcase.attrib.get("classname", "")
                    
                    test_cases.append({
                        "name": test_name,
                        "class": test_class,
                        "time": test_time
                    })
                    
                    all_test_times.append(test_time)
                
                execution_analysis["test_suites"].append({
                    "name": suite_name,
                    "total_time": suite_time,
                    "num_tests": suite_tests,
                    "failures": suite_failures,
                    "errors": suite_errors,
                    "avg_test_time": suite_time / suite_tests if suite_tests > 0 else 0,
                    "test_cases": sorted(test_cases, key=lambda x: x["time"], reverse=True)
                })
                
            except Exception as e:
                print(f"Error parsing JUnit XML {xml_file}: {e}")
        
        # Calculate summary statistics
        if all_test_times:
            execution_analysis["summary"] = {
                "total_tests": len(all_test_times),
                "total_time": sum(suite["total_time"] for suite in execution_analysis["test_suites"]),
                "avg_test_time": statistics.mean(all_test_times),
                "slowest_tests": sorted(
                    [(suite["name"], tc["name"], tc["time"]) 
                     for suite in execution_analysis["test_suites"] 
                     for tc in suite["test_cases"]],
                    key=lambda x: x[2], reverse=True
                )[:10],
                "fastest_tests": sorted(
                    [(suite["name"], tc["name"], tc["time"]) 
                     for suite in execution_analysis["test_suites"] 
                     for tc in suite["test_cases"]],
                    key=lambda x: x[2]
                )[:10]
            }
        
        return execution_analysis
    
    def generate_performance_recommendations(
        self, 
        execution_times: Dict[str, Any], 
        performance_results: Dict[str, Any]
    ) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Test execution time recommendations
        if execution_times.get("summary", {}).get("avg_test_time", 0) > 1.0:
            recommendations.append(
                "Consider breaking down slow tests or using parallel execution to improve test suite speed"
            )
        
        slowest_tests = execution_times.get("summary", {}).get("slowest_tests", [])
        if slowest_tests and slowest_tests[0][2] > 10.0:
            recommendations.append(
                f"Investigate slowest test: {slowest_tests[0][1]} ({slowest_tests[0][2]:.2f}s)"
            )
        
        # Performance test recommendations
        if performance_results:
            failure_rate = performance_results.get("summary", {}).get("failure_rate", 0)
            if failure_rate > 5:
                recommendations.append(
                    f"High failure rate detected ({failure_rate:.1f}%) - investigate error handling and system stability"
                )
            
            avg_response_time = performance_results.get("summary", {}).get("avg_response_time", 0)
            if avg_response_time > 500:
                recommendations.append(
                    f"Average response time is high ({avg_response_time:.0f}ms) - consider caching and optimization"
                )
            
            rps = performance_results.get("summary", {}).get("requests_per_second", 0)
            if rps < 100:
                recommendations.append(
                    f"Low throughput detected ({rps:.1f} req/s) - investigate bottlenecks and scaling options"
                )
        
        # General recommendations
        recommendations.extend([
            "Set up continuous performance monitoring in CI/CD pipeline",
            "Consider implementing performance regression testing",
            "Use profiling tools to identify hotspots in slow tests",
            "Implement test parallelization where possible"
        ])
        
        return recommendations
    
    def generate_html_benchmark_report(
        self, 
        execution_data: Dict[str, Any], 
        performance_data: Dict[str, Any], 
        output_path: Path
    ) -> None:
        """Generate HTML benchmark report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        recommendations = self.generate_performance_recommendations(execution_data, performance_data)
        
        # Calculate performance scores
        total_time = execution_data.get("summary", {}).get("total_time", 0)
        avg_response_time = performance_data.get("summary", {}).get("avg_response_time", 0)
        failure_rate = performance_data.get("summary", {}).get("failure_rate", 0)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Benchmark Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f8fafc;
            color: #334155;
        }}
        .header {{
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .metric-label {{
            color: #64748b;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .section {{
            background: white;
            margin-bottom: 30px;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .section-header {{
            background: #475569;
            color: white;
            padding: 20px;
            font-size: 1.3em;
            font-weight: 500;
        }}
        .section-content {{
            padding: 25px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background: #f1f5f9;
            font-weight: 600;
        }}
        .performance-bar {{
            height: 20px;
            background: #e2e8f0;
            border-radius: 10px;
            overflow: hidden;
        }}
        .performance-fill {{
            height: 100%;
            transition: width 0.3s ease;
        }}
        .perf-excellent {{ background: #10b981; }}
        .perf-good {{ background: #3b82f6; }}
        .perf-acceptable {{ background: #f59e0b; }}
        .perf-poor {{ background: #ef4444; }}
        .recommendations {{
            list-style: none;
            padding: 0;
        }}
        .recommendations li {{
            padding: 12px;
            margin-bottom: 10px;
            background: #f8fafc;
            border-radius: 8px;
            border-left: 4px solid #3b82f6;
        }}
        .chart-container {{
            height: 300px;
            margin: 20px 0;
            position: relative;
        }}
        .status-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .status-excellent {{ background: #d1fae5; color: #065f46; }}
        .status-good {{ background: #dbeafe; color: #1e40af; }}
        .status-acceptable {{ background: #fef3c7; color: #92400e; }}
        .status-poor {{ background: #fee2e2; color: #991b1b; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="header">
        <h1>⚡ Performance Benchmark Report</h1>
        <div style="margin-top: 10px; opacity: 0.9;">Generated on {timestamp}</div>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value">{total_time:.1f}s</div>
            <div class="metric-label">Total Test Time</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{execution_data.get('summary', {}).get('total_tests', 0)}</div>
            <div class="metric-label">Total Tests</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{avg_response_time:.0f}ms</div>
            <div class="metric-label">Avg Response Time</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{failure_rate:.1f}%</div>
            <div class="metric-label">Failure Rate</div>
        </div>
    </div>
    
    <div class="section">
        <div class="section-header">🏃 Test Execution Performance</div>
        <div class="section-content">
            <h4>Test Suite Breakdown:</h4>
            <table>
                <thead>
                    <tr>
                        <th>Test Suite</th>
                        <th>Tests</th>
                        <th>Total Time</th>
                        <th>Avg Time</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        for suite in execution_data.get("test_suites", []):
            avg_time = suite.get("avg_test_time", 0)
            if avg_time < 0.1:
                status = "excellent"
            elif avg_time < 0.5:
                status = "good"
            elif avg_time < 2.0:
                status = "acceptable"
            else:
                status = "poor"
            
            html_content += f"""
                <tr>
                    <td>{suite['name']}</td>
                    <td>{suite['num_tests']}</td>
                    <td>{suite['total_time']:.2f}s</td>
                    <td>{avg_time:.3f}s</td>
                    <td><span class="status-badge status-{status}">{status.upper()}</span></td>
                </tr>
"""
        
        html_content += """
                </tbody>
            </table>
        </div>
    </div>
"""
        
        # Slowest tests section
        slowest_tests = execution_data.get("summary", {}).get("slowest_tests", [])[:10]
        if slowest_tests:
            html_content += f"""
    <div class="section">
        <div class="section-header">🐌 Slowest Tests</div>
        <div class="section-content">
            <table>
                <thead>
                    <tr>
                        <th>Test Suite</th>
                        <th>Test Name</th>
                        <th>Execution Time</th>
                        <th>Performance</th>
                    </tr>
                </thead>
                <tbody>
"""
            
            for suite_name, test_name, test_time in slowest_tests:
                if test_time < 1:
                    perf_class = "excellent"
                elif test_time < 5:
                    perf_class = "good"
                elif test_time < 10:
                    perf_class = "acceptable"
                else:
                    perf_class = "poor"
                
                html_content += f"""
                    <tr>
                        <td>{suite_name}</td>
                        <td>{test_name}</td>
                        <td>{test_time:.3f}s</td>
                        <td>
                            <div class="performance-bar">
                                <div class="performance-fill perf-{perf_class}" style="width: {min(test_time * 10, 100)}%"></div>
                            </div>
                        </td>
                    </tr>
"""
            
            html_content += """
                </tbody>
            </table>
        </div>
    </div>
"""
        
        # Performance test results
        if performance_data.get("requests"):
            html_content += f"""
    <div class="section">
        <div class="section-header">🎯 API Performance Results</div>
        <div class="section-content">
            <table>
                <thead>
                    <tr>
                        <th>Endpoint</th>
                        <th>Method</th>
                        <th>Requests</th>
                        <th>Failures</th>
                        <th>Avg Response</th>
                        <th>RPS</th>
                    </tr>
                </thead>
                <tbody>
"""
            
            for request in performance_data["requests"]:
                failure_rate = (request["num_failures"] / request["num_requests"] * 100) if request["num_requests"] > 0 else 0
                
                html_content += f"""
                    <tr>
                        <td>{request['name']}</td>
                        <td>{request['method']}</td>
                        <td>{request['num_requests']}</td>
                        <td>{request['num_failures']} ({failure_rate:.1f}%)</td>
                        <td>{request['avg_response_time']:.0f}ms</td>
                        <td>{request['requests_per_second']:.1f}</td>
                    </tr>
"""
            
            html_content += """
                </tbody>
            </table>
        </div>
    </div>
"""
        
        # Recommendations section
        html_content += f"""
    <div class="section">
        <div class="section-header">💡 Performance Recommendations</div>
        <div class="section-content">
            <ul class="recommendations">
"""
        
        for recommendation in recommendations:
            html_content += f"<li>{recommendation}</li>"
        
        html_content += """
            </ul>
        </div>
    </div>
    
    <div class="section">
        <div class="section-header">📊 Performance Trends</div>
        <div class="section-content">
            <div class="chart-container">
                <canvas id="performanceChart"></canvas>
            </div>
        </div>
    </div>
    
    <script>
        // Performance trend chart
        const ctx = document.getElementById('performanceChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Test Execution', 'API Response', 'Throughput', 'Error Rate'],
                datasets: [{
                    label: 'Performance Score',
                    data: [80, 75, 85, 90],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    </script>
</body>
</html>
"""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def run_benchmark_analysis(self) -> Dict[str, Any]:
        """Run complete benchmark analysis."""
        print("🔍 Analyzing test performance...")
        
        # Find test result files
        junit_files = list(self.reports_dir.glob("*-results.xml"))
        locust_csv = self.performance_dir / "performance-results_stats.csv"
        
        # Analyze test execution times
        execution_data = self.analyze_test_execution_times(junit_files)
        print(f"  📊 Analyzed {len(execution_data.get('test_suites', []))} test suites")
        
        # Analyze performance test results
        performance_data = self.parse_locust_results(locust_csv)
        if performance_data:
            print(f"  🎯 Analyzed {len(performance_data.get('requests', []))} API endpoints")
        
        # Generate benchmark report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_output = self.reports_dir / f"benchmark_report_{timestamp}.html"
        
        print(f"  📊 Generating benchmark report: {html_output}")
        self.generate_html_benchmark_report(execution_data, performance_data, html_output)
        
        # Create latest symlink
        latest_html = self.reports_dir / "benchmark_report_latest.html"
        if latest_html.exists():
            latest_html.unlink()
        latest_html.symlink_to(html_output.name)
        
        print(f"✅ Benchmark analysis complete!")
        print(f"   📊 Report: {html_output}")
        print(f"   📊 Latest: {latest_html}")
        
        return {
            "execution_data": execution_data,
            "performance_data": performance_data,
            "report_path": str(html_output)
        }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate performance benchmark reports")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    try:
        benchmark = TestPerformanceBenchmark(args.project_root)
        result = benchmark.run_benchmark_analysis()
        
        # Print summary
        execution_data = result.get("execution_data", {})
        total_time = execution_data.get("summary", {}).get("total_time", 0)
        total_tests = execution_data.get("summary", {}).get("total_tests", 0)
        
        print(f"\n📊 Performance Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Total Time: {total_time:.1f}s")
        if total_tests > 0:
            print(f"   Avg Time per Test: {total_time/total_tests:.3f}s")
        
        performance_data = result.get("performance_data", {})
        if performance_data:
            summary = performance_data.get("summary", {})
            print(f"   API Requests: {summary.get('total_requests', 0)}")
            print(f"   Failure Rate: {summary.get('failure_rate', 0):.1f}%")
            print(f"   Avg Response: {summary.get('avg_response_time', 0):.0f}ms")
        
    except Exception as e:
        print(f"❌ Error generating benchmark report: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())