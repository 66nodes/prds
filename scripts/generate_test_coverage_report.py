#!/usr/bin/env python3
"""
Test Coverage Report Generator for Strategic Planning Platform

Generates comprehensive test coverage reports with detailed metrics,
quality insights, and actionable recommendations.
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import argparse


class TestCoverageReporter:
    """Generate comprehensive test coverage reports."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.reports_dir = self.project_root / "test-results"
        self.coverage_dir = self.reports_dir / "coverage"
        
        # Quality thresholds
        self.quality_thresholds = {
            "excellent": 95,
            "good": 85,
            "acceptable": 75,
            "poor": 60
        }
    
    def parse_coverage_xml(self, xml_path: Path) -> Dict[str, Any]:
        """Parse coverage XML file and extract metrics."""
        if not xml_path.exists():
            return {}
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Extract overall coverage
            coverage = {
                "line_rate": float(root.attrib.get("line-rate", 0)) * 100,
                "branch_rate": float(root.attrib.get("branch-rate", 0)) * 100,
                "lines_covered": int(root.attrib.get("lines-covered", 0)),
                "lines_valid": int(root.attrib.get("lines-valid", 0)),
                "branches_covered": int(root.attrib.get("branches-covered", 0)),
                "branches_valid": int(root.attrib.get("branches-valid", 0)),
                "complexity": float(root.attrib.get("complexity", 0)),
                "timestamp": root.attrib.get("timestamp", ""),
                "packages": []
            }
            
            # Extract package-level coverage
            packages = root.find("packages")
            if packages is not None:
                for package in packages.findall("package"):
                    package_info = {
                        "name": package.attrib.get("name", ""),
                        "line_rate": float(package.attrib.get("line-rate", 0)) * 100,
                        "branch_rate": float(package.attrib.get("branch-rate", 0)) * 100,
                        "complexity": float(package.attrib.get("complexity", 0)),
                        "classes": []
                    }
                    
                    # Extract class-level coverage
                    classes = package.find("classes")
                    if classes is not None:
                        for cls in classes.findall("class"):
                            class_info = {
                                "name": cls.attrib.get("name", ""),
                                "filename": cls.attrib.get("filename", ""),
                                "line_rate": float(cls.attrib.get("line-rate", 0)) * 100,
                                "branch_rate": float(cls.attrib.get("branch-rate", 0)) * 100,
                                "complexity": float(cls.attrib.get("complexity", 0))
                            }
                            package_info["classes"].append(class_info)
                    
                    coverage["packages"].append(package_info)
            
            return coverage
            
        except Exception as e:
            print(f"Error parsing coverage XML {xml_path}: {e}")
            return {}
    
    def analyze_coverage_quality(self, coverage_percent: float) -> Dict[str, Any]:
        """Analyze coverage quality and provide recommendations."""
        if coverage_percent >= self.quality_thresholds["excellent"]:
            quality = "excellent"
            emoji = "🎉"
            color = "green"
            recommendations = [
                "Maintain current coverage levels",
                "Consider property-based testing for edge cases",
                "Add performance benchmarks"
            ]
        elif coverage_percent >= self.quality_thresholds["good"]:
            quality = "good"
            emoji = "✅"
            color = "green"
            recommendations = [
                "Target 95% coverage for production readiness",
                "Focus on edge cases and error paths",
                "Add integration tests for complex workflows"
            ]
        elif coverage_percent >= self.quality_thresholds["acceptable"]:
            quality = "acceptable"
            emoji = "⚠️"
            color = "yellow"
            recommendations = [
                "Increase coverage to 85% minimum",
                "Prioritize critical business logic",
                "Add tests for authentication and security"
            ]
        elif coverage_percent >= self.quality_thresholds["poor"]:
            quality = "poor"
            emoji = "❌"
            color = "red"
            recommendations = [
                "Immediate action required - coverage too low",
                "Focus on core functionality testing",
                "Implement test-driven development practices"
            ]
        else:
            quality = "critical"
            emoji = "🚨"
            color = "red"
            recommendations = [
                "CRITICAL: Coverage dangerously low",
                "Stop feature development - focus on testing",
                "Consider code review of existing tests"
            ]
        
        return {
            "quality": quality,
            "emoji": emoji,
            "color": color,
            "recommendations": recommendations
        }
    
    def identify_low_coverage_files(self, coverage_data: Dict[str, Any], threshold: float = 80) -> List[Dict]:
        """Identify files with coverage below threshold."""
        low_coverage_files = []
        
        for package in coverage_data.get("packages", []):
            for cls in package.get("classes", []):
                if cls["line_rate"] < threshold:
                    low_coverage_files.append({
                        "file": cls["filename"],
                        "class": cls["name"],
                        "coverage": cls["line_rate"],
                        "package": package["name"]
                    })
        
        # Sort by lowest coverage first
        low_coverage_files.sort(key=lambda x: x["coverage"])
        return low_coverage_files
    
    def calculate_coverage_trends(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """Calculate coverage trends over time."""
        if len(historical_data) < 2:
            return {"trend": "insufficient_data", "change": 0}
        
        # Sort by timestamp
        historical_data.sort(key=lambda x: x.get("timestamp", ""))
        
        latest = historical_data[-1]["line_rate"]
        previous = historical_data[-2]["line_rate"]
        change = latest - previous
        
        if change > 2:
            trend = "improving"
        elif change < -2:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "change": change,
            "latest": latest,
            "previous": previous,
            "history_points": len(historical_data)
        }
    
    def generate_security_coverage_analysis(self, coverage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze coverage of security-critical components."""
        security_patterns = [
            "auth", "security", "encrypt", "hash", "token", "permission",
            "validation", "sanitiz", "csrf", "xss", "sql", "injection"
        ]
        
        security_files = []
        for package in coverage_data.get("packages", []):
            for cls in package.get("classes", []):
                filename = cls["filename"].lower()
                classname = cls["name"].lower()
                
                if any(pattern in filename or pattern in classname for pattern in security_patterns):
                    security_files.append({
                        "file": cls["filename"],
                        "class": cls["name"],
                        "coverage": cls["line_rate"],
                        "security_critical": True
                    })
        
        if security_files:
            avg_security_coverage = sum(f["coverage"] for f in security_files) / len(security_files)
            min_security_coverage = min(f["coverage"] for f in security_files)
        else:
            avg_security_coverage = 0
            min_security_coverage = 0
        
        return {
            "security_files_count": len(security_files),
            "average_coverage": avg_security_coverage,
            "minimum_coverage": min_security_coverage,
            "files": security_files,
            "risk_level": "high" if min_security_coverage < 90 else "medium" if min_security_coverage < 95 else "low"
        }
    
    def generate_html_report(self, coverage_data: Dict[str, Any], output_path: Path) -> None:
        """Generate comprehensive HTML coverage report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Analyze coverage quality
        line_coverage = coverage_data.get("line_rate", 0)
        quality_analysis = self.analyze_coverage_quality(line_coverage)
        
        # Identify problem areas
        low_coverage_files = self.identify_low_coverage_files(coverage_data)
        
        # Security analysis
        security_analysis = self.generate_security_coverage_analysis(coverage_data)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Coverage Report - Strategic Planning Platform</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f7fa;
            color: #2c3e50;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .timestamp {{
            opacity: 0.8;
            margin-top: 10px;
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
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
            border-left: 4px solid #3498db;
        }}
        .metric-card.excellent {{ border-left-color: #27ae60; }}
        .metric-card.good {{ border-left-color: #2ecc71; }}
        .metric-card.acceptable {{ border-left-color: #f39c12; }}
        .metric-card.poor {{ border-left-color: #e74c3c; }}
        .metric-card.critical {{ border-left-color: #c0392b; }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .section {{
            background: white;
            margin-bottom: 30px;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .section-header {{
            background: #34495e;
            color: white;
            padding: 20px;
            font-size: 1.3em;
            font-weight: 500;
        }}
        .section-content {{
            padding: 25px;
        }}
        .quality-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.8em;
        }}
        .quality-excellent {{ background: #d5f4e6; color: #27ae60; }}
        .quality-good {{ background: #d5f4e6; color: #2ecc71; }}
        .quality-acceptable {{ background: #fef9e7; color: #f39c12; }}
        .quality-poor {{ background: #fadbd8; color: #e74c3c; }}
        .quality-critical {{ background: #f6ddcc; color: #c0392b; }}
        .recommendations {{
            list-style: none;
            padding: 0;
        }}
        .recommendations li {{
            padding: 10px;
            margin-bottom: 10px;
            background: #ecf0f1;
            border-radius: 5px;
            border-left: 3px solid #3498db;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #bdc3c7;
        }}
        th {{
            background: #ecf0f1;
            font-weight: 600;
            color: #2c3e50;
        }}
        .coverage-bar {{
            height: 20px;
            background: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
        }}
        .coverage-fill {{
            height: 100%;
            transition: width 0.3s ease;
        }}
        .coverage-high {{ background: #27ae60; }}
        .coverage-medium {{ background: #f39c12; }}
        .coverage-low {{ background: #e74c3c; }}
        .alert {{
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .alert-warning {{
            background: #fef9e7;
            border: 1px solid #f39c12;
            color: #d68910;
        }}
        .alert-danger {{
            background: #fadbd8;
            border: 1px solid #e74c3c;
            color: #c0392b;
        }}
        .progress-ring {{
            transform: rotate(-90deg);
        }}
        .progress-ring-circle {{
            stroke: #d1d5db;
            stroke-width: 4;
            fill: transparent;
        }}
        .progress-ring-circle.progress {{
            stroke: #10b981;
            stroke-width: 6;
            stroke-linecap: round;
            transition: stroke-dashoffset 0.5s ease-in-out;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{quality_analysis['emoji']} Test Coverage Report</h1>
        <div class="timestamp">Generated on {timestamp}</div>
        <div class="timestamp">Strategic Planning Platform</div>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card {quality_analysis['quality']}">
            <div class="metric-value">{line_coverage:.1f}%</div>
            <div class="metric-label">Line Coverage</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{coverage_data.get('branch_rate', 0):.1f}%</div>
            <div class="metric-label">Branch Coverage</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{coverage_data.get('lines_covered', 0)}</div>
            <div class="metric-label">Lines Covered</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{len(coverage_data.get('packages', []))}</div>
            <div class="metric-label">Packages Analyzed</div>
        </div>
    </div>
    
    <div class="section">
        <div class="section-header">🎯 Coverage Quality Analysis</div>
        <div class="section-content">
            <p>
                <span class="quality-badge quality-{quality_analysis['quality']}">
                    {quality_analysis['quality'].upper()} QUALITY
                </span>
            </p>
            <h4>Recommendations:</h4>
            <ul class="recommendations">
"""
        
        for recommendation in quality_analysis["recommendations"]:
            html_content += f"<li>{recommendation}</li>"
        
        html_content += """
            </ul>
        </div>
    </div>
"""
        
        # Security analysis section
        if security_analysis["security_files_count"] > 0:
            risk_color = "danger" if security_analysis["risk_level"] == "high" else "warning"
            html_content += f"""
    <div class="section">
        <div class="section-header">🛡️ Security Coverage Analysis</div>
        <div class="section-content">
            <div class="alert alert-{risk_color}">
                <strong>Security Risk Level: {security_analysis['risk_level'].upper()}</strong><br>
                Found {security_analysis['security_files_count']} security-critical files with average coverage of {security_analysis['average_coverage']:.1f}%
            </div>
            <p><strong>Minimum Security Coverage:</strong> {security_analysis['minimum_coverage']:.1f}%</p>
            <p><strong>Average Security Coverage:</strong> {security_analysis['average_coverage']:.1f}%</p>
            
            <h4>Security-Critical Files:</h4>
            <table>
                <thead>
                    <tr>
                        <th>File</th>
                        <th>Class</th>
                        <th>Coverage</th>
                        <th>Visual</th>
                    </tr>
                </thead>
                <tbody>
"""
            
            for file_info in security_analysis["files"]:
                coverage_class = "high" if file_info["coverage"] >= 90 else "medium" if file_info["coverage"] >= 70 else "low"
                html_content += f"""
                    <tr>
                        <td>{file_info['file']}</td>
                        <td>{file_info['class']}</td>
                        <td>{file_info['coverage']:.1f}%</td>
                        <td>
                            <div class="coverage-bar">
                                <div class="coverage-fill coverage-{coverage_class}" style="width: {file_info['coverage']}%"></div>
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
        
        # Low coverage files section
        if low_coverage_files:
            html_content += f"""
    <div class="section">
        <div class="section-header">⚠️ Files Requiring Attention</div>
        <div class="section-content">
            <p>Found {len(low_coverage_files)} files with coverage below 80%:</p>
            <table>
                <thead>
                    <tr>
                        <th>File</th>
                        <th>Package</th>
                        <th>Coverage</th>
                        <th>Priority</th>
                        <th>Visual</th>
                    </tr>
                </thead>
                <tbody>
"""
            
            for file_info in low_coverage_files[:10]:  # Show top 10 worst files
                coverage_class = "medium" if file_info["coverage"] >= 50 else "low"
                priority = "HIGH" if file_info["coverage"] < 50 else "MEDIUM"
                priority_class = "danger" if priority == "HIGH" else "warning"
                
                html_content += f"""
                    <tr>
                        <td>{os.path.basename(file_info['file'])}</td>
                        <td>{file_info['package']}</td>
                        <td>{file_info['coverage']:.1f}%</td>
                        <td><span class="alert alert-{priority_class}" style="padding: 2px 8px; display: inline-block;">{priority}</span></td>
                        <td>
                            <div class="coverage-bar">
                                <div class="coverage-fill coverage-{coverage_class}" style="width: {file_info['coverage']}%"></div>
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
        
        # Package breakdown section
        html_content += """
    <div class="section">
        <div class="section-header">📦 Package Coverage Breakdown</div>
        <div class="section-content">
            <table>
                <thead>
                    <tr>
                        <th>Package</th>
                        <th>Line Coverage</th>
                        <th>Branch Coverage</th>
                        <th>Complexity</th>
                        <th>Classes</th>
                        <th>Visual</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        for package in coverage_data.get("packages", []):
            coverage_class = "high" if package["line_rate"] >= 85 else "medium" if package["line_rate"] >= 70 else "low"
            html_content += f"""
                <tr>
                    <td>{package['name']}</td>
                    <td>{package['line_rate']:.1f}%</td>
                    <td>{package['branch_rate']:.1f}%</td>
                    <td>{package['complexity']:.1f}</td>
                    <td>{len(package['classes'])}</td>
                    <td>
                        <div class="coverage-bar">
                            <div class="coverage-fill coverage-{coverage_class}" style="width: {package['line_rate']}%"></div>
                        </div>
                    </td>
                </tr>
"""
        
        html_content += """
                </tbody>
            </table>
        </div>
    </div>
    
    <div class="section">
        <div class="section-header">📋 Summary</div>
        <div class="section-content">
            <p>This comprehensive coverage report provides insights into the quality and completeness of your test suite. 
            Focus on improving coverage for security-critical components and files with low coverage to ensure robust 
            application reliability.</p>
            
            <h4>Next Steps:</h4>
            <ul>
                <li>Review and improve coverage for files below 80%</li>
                <li>Ensure security-critical components have >95% coverage</li>
                <li>Add integration tests for complex workflows</li>
                <li>Consider property-based testing for edge cases</li>
                <li>Set up coverage monitoring in CI/CD pipeline</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
        
        # Write HTML report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def generate_json_report(self, coverage_data: Dict[str, Any], output_path: Path) -> None:
        """Generate JSON coverage report for programmatic use."""
        line_coverage = coverage_data.get("line_rate", 0)
        quality_analysis = self.analyze_coverage_quality(line_coverage)
        low_coverage_files = self.identify_low_coverage_files(coverage_data)
        security_analysis = self.generate_security_coverage_analysis(coverage_data)
        
        json_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_metrics": {
                "line_coverage": line_coverage,
                "branch_coverage": coverage_data.get("branch_rate", 0),
                "lines_covered": coverage_data.get("lines_covered", 0),
                "lines_valid": coverage_data.get("lines_valid", 0),
                "branches_covered": coverage_data.get("branches_covered", 0),
                "branches_valid": coverage_data.get("branches_valid", 0),
                "complexity": coverage_data.get("complexity", 0)
            },
            "quality_analysis": quality_analysis,
            "security_analysis": security_analysis,
            "low_coverage_files": low_coverage_files,
            "package_breakdown": coverage_data.get("packages", []),
            "recommendations": {
                "immediate_actions": quality_analysis["recommendations"],
                "quality_gates": {
                    "minimum_coverage": 75,
                    "target_coverage": 90,
                    "security_coverage_minimum": 95
                }
            }
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2)
    
    def run_coverage_analysis(self, test_type: str = "all") -> Dict[str, Any]:
        """Run coverage analysis and generate reports."""
        print(f"📊 Generating coverage analysis for: {test_type}")
        
        # Determine which coverage files to analyze
        coverage_files = []
        if test_type in ["all", "unit"]:
            unit_xml = self.coverage_dir / "backend-unit.xml"
            if unit_xml.exists():
                coverage_files.append(("unit", unit_xml))
        
        if test_type in ["all", "integration"]:
            integration_xml = self.coverage_dir / "backend-integration.xml"
            if integration_xml.exists():
                coverage_files.append(("integration", integration_xml))
        
        if not coverage_files:
            print("❌ No coverage files found!")
            return {}
        
        # Combine coverage data
        combined_coverage = {}
        for test_name, xml_path in coverage_files:
            print(f"  📄 Processing {test_name} coverage...")
            coverage_data = self.parse_coverage_xml(xml_path)
            if coverage_data:
                if not combined_coverage:
                    combined_coverage = coverage_data
                else:
                    # Merge coverage data (simplified)
                    combined_coverage["line_rate"] = max(
                        combined_coverage.get("line_rate", 0),
                        coverage_data.get("line_rate", 0)
                    )
                    combined_coverage["branch_rate"] = max(
                        combined_coverage.get("branch_rate", 0),
                        coverage_data.get("branch_rate", 0)
                    )
        
        if not combined_coverage:
            print("❌ No valid coverage data found!")
            return {}
        
        # Generate reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # HTML report
        html_output = self.reports_dir / f"coverage_report_{timestamp}.html"
        print(f"  📊 Generating HTML report: {html_output}")
        self.generate_html_report(combined_coverage, html_output)
        
        # JSON report
        json_output = self.reports_dir / f"coverage_report_{timestamp}.json"
        print(f"  📊 Generating JSON report: {json_output}")
        self.generate_json_report(combined_coverage, json_output)
        
        # Create latest symlinks
        latest_html = self.reports_dir / "coverage_report_latest.html"
        latest_json = self.reports_dir / "coverage_report_latest.json"
        
        if latest_html.exists():
            latest_html.unlink()
        if latest_json.exists():
            latest_json.unlink()
        
        latest_html.symlink_to(html_output.name)
        latest_json.symlink_to(json_output.name)
        
        print(f"✅ Coverage analysis complete!")
        print(f"   📊 HTML Report: {html_output}")
        print(f"   📊 JSON Report: {json_output}")
        print(f"   📊 Latest HTML: {latest_html}")
        print(f"   📊 Latest JSON: {latest_json}")
        
        return {
            "coverage_data": combined_coverage,
            "html_report": str(html_output),
            "json_report": str(json_output),
            "line_coverage": combined_coverage.get("line_rate", 0)
        }


def main():
    """Main function to run coverage analysis."""
    parser = argparse.ArgumentParser(description="Generate comprehensive test coverage reports")
    parser.add_argument(
        "--type",
        choices=["all", "unit", "integration"],
        default="all",
        help="Type of coverage to analyze (default: all)"
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    try:
        reporter = TestCoverageReporter(args.project_root)
        result = reporter.run_coverage_analysis(args.type)
        
        if result:
            line_coverage = result.get("line_coverage", 0)
            print(f"\n🎯 Overall Line Coverage: {line_coverage:.1f}%")
            
            if line_coverage >= 90:
                print("🎉 Excellent coverage! Keep up the good work!")
                sys.exit(0)
            elif line_coverage >= 75:
                print("✅ Good coverage, but there's room for improvement.")
                sys.exit(0)
            else:
                print("⚠️ Coverage below recommended threshold (75%)")
                sys.exit(1)
        else:
            print("❌ Failed to generate coverage reports")
            sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error generating coverage reports: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()