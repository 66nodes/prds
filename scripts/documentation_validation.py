#!/usr/bin/env python3
"""
Documentation Validation Script
Validates documentation completeness for the AI Agent Platform.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
import re


class DocumentationValidator:
    """Documentation completeness validator."""
    
    def __init__(self):
        self.results = {
            "documentation_status": "validating",
            "checks": [],
            "missing_docs": [],
            "recommendations": [],
            "score": 0,
            "max_score": 0
        }
    
    def add_check(self, category: str, name: str, status: str, message: str, score: int = 1):
        """Add documentation check result."""
        self.results["checks"].append({
            "category": category,
            "name": name,
            "status": status,
            "message": message,
            "score": score if status == "pass" else 0
        })
        self.results["max_score"] += score
        if status == "pass":
            self.results["score"] += score
        print(f"{'✅' if status == 'pass' else '❌'} {category} - {name}: {message}")
    
    def check_file_exists(self, file_path: str, min_length: int = 100) -> bool:
        """Check if file exists and has content."""
        if not os.path.exists(file_path):
            return False
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                return len(content.strip()) >= min_length
        except:
            return False
    
    def count_lines(self, file_path: str) -> int:
        """Count non-empty lines in file."""
        if not os.path.exists(file_path):
            return 0
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return len([line for line in f.readlines() if line.strip()])
        except:
            return 0
    
    def check_core_documentation(self):
        """Check core project documentation."""
        print("\n📚 Core Documentation")
        
        core_docs = [
            ("README.md", "Main project README", 500),
            ("docs/README.md", "Documentation directory README", 200),
            ("CLAUDE.md", "Claude Code configuration", 1000),
            (".github/workflows/ci-cd.yml", "CI/CD pipeline documentation", 100),
            ("docker-stack.yml", "Docker deployment configuration", 200)
        ]
        
        for file_path, description, min_length in core_docs:
            if self.check_file_exists(file_path, min_length):
                lines = self.count_lines(file_path)
                self.add_check(
                    "Core Docs",
                    description,
                    "pass",
                    f"Complete ({lines} lines)"
                )
            else:
                self.add_check(
                    "Core Docs",
                    description,
                    "fail",
                    "Missing or incomplete"
                )
                self.results["missing_docs"].append(file_path)
    
    def check_api_documentation(self):
        """Check API documentation."""
        print("\n🌐 API Documentation")
        
        # Check for OpenAPI/Swagger documentation
        api_doc_files = [
            "backend/api/openapi.json",
            "backend/api/swagger.yml",
            "docs/api/openapi.yaml",
            "backend/main.py"  # Should contain API documentation setup
        ]
        
        api_docs_found = False
        for file_path in api_doc_files:
            if self.check_file_exists(file_path, 50):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()
                    if any(keyword in content for keyword in ['openapi', 'swagger', 'fastapi', 'docs_url']):
                        api_docs_found = True
                        break
        
        if api_docs_found:
            self.add_check(
                "API Docs",
                "API Documentation",
                "pass",
                "API documentation configuration found"
            )
        else:
            self.add_check(
                "API Docs",
                "API Documentation",
                "fail",
                "No API documentation found"
            )
        
        # Check for API endpoint documentation
        endpoint_docs = []
        backend_files = Path("backend").rglob("*.py")
        for file_path in backend_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if '@app.' in content or '@router.' in content:
                        # Count documented endpoints (those with docstrings)
                        endpoint_matches = re.findall(r'@(app|router)\.\w+.*\n.*def\s+\w+.*:\n\s*""".*?"""', content, re.DOTALL)
                        endpoint_docs.extend(endpoint_matches)
            except:
                continue
        
        if len(endpoint_docs) > 0:
            self.add_check(
                "API Docs",
                "Endpoint Documentation",
                "pass",
                f"{len(endpoint_docs)} documented endpoints found"
            )
        else:
            self.add_check(
                "API Docs", 
                "Endpoint Documentation",
                "fail",
                "No documented API endpoints found"
            )
    
    def check_deployment_documentation(self):
        """Check deployment and infrastructure documentation."""
        print("\n🚀 Deployment Documentation")
        
        deployment_docs = [
            ("scripts/production_check.sh", "Production readiness script"),
            ("scripts/backup.sh", "Backup procedures"),
            ("scripts/restore.sh", "Restore procedures"),
            ("monitoring/prometheus/prometheus.yml", "Monitoring configuration"),
            ("docker-stack.yml", "Docker Swarm configuration")
        ]
        
        for file_path, description in deployment_docs:
            if self.check_file_exists(file_path, 50):
                self.add_check(
                    "Deployment",
                    description,
                    "pass",
                    "Complete"
                )
            else:
                self.add_check(
                    "Deployment",
                    description,
                    "fail",
                    "Missing"
                )
    
    def check_development_documentation(self):
        """Check development setup and contribution documentation."""
        print("\n👨‍💻 Development Documentation")
        
        # Check for development setup instructions
        setup_files = [
            "README.md",
            "docs/setup.md",
            "docs/development.md",
            "CONTRIBUTING.md",
            ".github/CONTRIBUTING.md"
        ]
        
        setup_docs_found = False
        for file_path in setup_files:
            if self.check_file_exists(file_path, 200):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()
                    if any(keyword in content for keyword in ['install', 'setup', 'development', 'getting started']):
                        setup_docs_found = True
                        break
        
        if setup_docs_found:
            self.add_check(
                "Development",
                "Setup Instructions",
                "pass",
                "Development setup documented"
            )
        else:
            self.add_check(
                "Development",
                "Setup Instructions",
                "fail",
                "No development setup instructions found"
            )
        
        # Check for testing documentation
        test_docs_found = self.check_file_exists("README_TESTING.md", 500)
        
        if test_docs_found:
            self.add_check(
                "Development",
                "Testing Documentation",
                "pass",
                "Testing procedures documented"
            )
        else:
            self.add_check(
                "Development",
                "Testing Documentation",
                "fail",
                "No testing documentation found"
            )
    
    def check_technical_documentation(self):
        """Check technical and architectural documentation."""
        print("\n🏗️ Technical Documentation")
        
        # Count technical documents in docs/
        tech_docs = []
        if os.path.exists("docs"):
            for file_path in Path("docs").rglob("*.md"):
                if file_path.stat().st_size > 1000:  # Files > 1KB
                    tech_docs.append(str(file_path))
        
        if len(tech_docs) >= 5:
            self.add_check(
                "Technical",
                "Architecture Documentation",
                "pass",
                f"{len(tech_docs)} technical documents found"
            )
        else:
            self.add_check(
                "Technical",
                "Architecture Documentation",
                "fail",
                f"Only {len(tech_docs)} technical documents found (need ≥5)"
            )
        
        # Check for specific technical documents
        important_docs = [
            ("docs/Strategic-Planning-document.md", "Strategic planning documentation"),
            ("docs/004-FRD.md", "Functional requirements"),
            ("PROJECT_EXECUTION_MANIFEST.yaml", "Project execution manifest")
        ]
        
        for file_path, description in important_docs:
            if self.check_file_exists(file_path, 500):
                self.add_check(
                    "Technical",
                    description,
                    "pass",
                    "Complete"
                )
            else:
                self.add_check(
                    "Technical",
                    description,
                    "fail",
                    "Missing or incomplete"
                )
    
    def check_configuration_documentation(self):
        """Check configuration and environment documentation."""
        print("\n⚙️ Configuration Documentation")
        
        # Check environment documentation
        env_docs = [
            (".env.production.example", "Production environment template"),
            (".env.development", "Development environment example")
        ]
        
        for file_path, description in env_docs:
            if self.check_file_exists(file_path, 100):
                self.add_check(
                    "Configuration",
                    description,
                    "pass",
                    "Complete"
                )
            else:
                self.add_check(
                    "Configuration",
                    description,
                    "fail",
                    "Missing"
                )
        
        # Check requirements documentation
        req_files = [
            "backend/requirements.txt",
            "backend/requirements-test.txt",
            "frontend/package.json"
        ]
        
        requirements_complete = True
        for file_path in req_files:
            if not self.check_file_exists(file_path, 100):
                requirements_complete = False
                break
        
        if requirements_complete:
            self.add_check(
                "Configuration",
                "Dependencies Documentation",
                "pass",
                "All requirements files present"
            )
        else:
            self.add_check(
                "Configuration",
                "Dependencies Documentation",
                "fail",
                "Missing requirements files"
            )
    
    def generate_report(self):
        """Generate documentation completeness report."""
        score_percentage = (self.results["score"] / self.results["max_score"]) * 100 if self.results["max_score"] > 0 else 0
        
        self.results["documentation_status"] = (
            "complete" if score_percentage >= 80 else
            "needs_improvement" if score_percentage >= 60 else
            "incomplete"
        )
        
        # Add recommendations based on missing documentation
        if self.results["missing_docs"]:
            self.results["recommendations"] = [
                f"Create missing documentation: {', '.join(self.results['missing_docs'])}",
                "Ensure all API endpoints have proper documentation",
                "Add more detailed setup and development instructions",
                "Include troubleshooting guides and FAQ sections"
            ]
        
        print(f"\n{'='*60}")
        print("📚 DOCUMENTATION VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"📊 Documentation Score: {self.results['score']}/{self.results['max_score']} ({score_percentage:.1f}%)")
        print(f"📖 Documentation Status: {self.results['documentation_status'].upper()}")
        print(f"✅ Complete Sections: {len([c for c in self.results['checks'] if c['status'] == 'pass'])}")
        print(f"❌ Missing Sections: {len([c for c in self.results['checks'] if c['status'] == 'fail'])}")
        print(f"📄 Total Documentation Files: {len(Path('.').rglob('*.md')) if Path('.').exists() else 0}")
        
        if self.results["missing_docs"]:
            print(f"\n📋 MISSING DOCUMENTATION:")
            for doc in self.results["missing_docs"]:
                print(f"   • {doc}")
        
        if self.results["recommendations"]:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in self.results["recommendations"]:
                print(f"   • {rec}")
        
        # Save report
        with open("documentation_validation_report.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: documentation_validation_report.json")
        
        return score_percentage >= 80
    
    def run_validation(self):
        """Run complete documentation validation."""
        print("📚 AI AGENT PLATFORM - DOCUMENTATION VALIDATION")
        print("="*60)
        
        self.check_core_documentation()
        self.check_api_documentation()
        self.check_deployment_documentation()
        self.check_development_documentation()
        self.check_technical_documentation()
        self.check_configuration_documentation()
        
        return self.generate_report()


if __name__ == "__main__":
    validator = DocumentationValidator()
    success = validator.run_validation()
    exit(0 if success else 1)