#!/usr/bin/env python3
"""
Security Validation Script
Performs security checks on the AI Agent Platform.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import re


class SecurityValidator:
    """Security validation for the platform."""
    
    def __init__(self):
        self.results = {
            "security_status": "validating",
            "checks": [],
            "vulnerabilities": [],
            "recommendations": [],
            "score": 0,
            "max_score": 0
        }
    
    def add_check(self, category: str, name: str, status: str, message: str, score: int = 1):
        """Add security check result."""
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
    
    def add_vulnerability(self, severity: str, component: str, description: str, recommendation: str = None):
        """Add vulnerability finding."""
        self.results["vulnerabilities"].append({
            "severity": severity,
            "component": component,
            "description": description,
            "recommendation": recommendation
        })
        print(f"🚨 {severity.upper()} - {component}: {description}")
    
    def check_environment_files(self):
        """Check environment file security."""
        print("\n🔒 Environment File Security")
        
        # Check that production env file doesn't exist
        if not os.path.exists(".env.production"):
            self.add_check(
                "Environment", 
                "Production Secrets",
                "pass",
                "Production environment file is not committed"
            )
        else:
            self.add_check(
                "Environment",
                "Production Secrets", 
                "fail",
                "Production environment file found in repository"
            )
            self.add_vulnerability(
                "critical",
                ".env.production",
                "Production secrets may be exposed in version control"
            )
        
        # Check environment template
        if os.path.exists(".env.production.example"):
            with open(".env.production.example", "r") as f:
                content = f.read()
                if "CHANGE_ME" in content and "sk-" not in content:
                    self.add_check(
                        "Environment",
                        "Template Security",
                        "pass", 
                        "Environment template uses placeholders"
                    )
                else:
                    self.add_check(
                        "Environment",
                        "Template Security",
                        "fail",
                        "Environment template may contain real credentials"
                    )
    
    def check_docker_security(self):
        """Check Docker security configuration."""
        print("\n🐳 Docker Security")
        
        # Check Dockerfile security
        dockerfile_path = "backend/Dockerfile"
        if os.path.exists(dockerfile_path):
            with open(dockerfile_path, "r") as f:
                content = f.read()
                
                # Check for non-root user
                if "USER " in content and "USER root" not in content:
                    self.add_check(
                        "Docker",
                        "Non-root User",
                        "pass",
                        "Application runs as non-root user"
                    )
                else:
                    self.add_check(
                        "Docker", 
                        "Non-root User",
                        "fail",
                        "Application may run as root user"
                    )
                
                # Check for health checks
                if "HEALTHCHECK" in content:
                    self.add_check(
                        "Docker",
                        "Health Checks",
                        "pass",
                        "Health checks are configured"
                    )
                else:
                    self.add_check(
                        "Docker",
                        "Health Checks", 
                        "fail",
                        "No health checks configured"
                    )
        
        # Check docker-compose security
        compose_files = ["docker-stack.yml", "docker-compose.yml"]
        for compose_file in compose_files:
            if os.path.exists(compose_file):
                with open(compose_file, "r") as f:
                    content = f.read()
                    
                    # Check for hardcoded passwords
                    if re.search(r'password:\s*["\'](?!.*CHANGE_ME)[^"\']{8,}["\']', content, re.IGNORECASE):
                        self.add_vulnerability(
                            "high",
                            compose_file,
                            "Hardcoded passwords detected in compose file"
                        )
                    else:
                        self.add_check(
                            "Docker",
                            f"{compose_file} Secrets",
                            "pass",
                            "No hardcoded passwords found"
                        )
    
    def check_dependencies(self):
        """Check dependency security."""
        print("\n📦 Dependency Security") 
        
        # Check for known vulnerable packages (simplified check)
        vulnerable_patterns = [
            (r"pillow.*[<>=]\s*[1-9]\..*", "Pillow versions may have vulnerabilities"),
            (r"requests.*[<>=]\s*[12]\..*", "Old requests versions may have vulnerabilities"),
            (r"urllib3.*[<>=]\s*[12]\..*", "Old urllib3 versions may have vulnerabilities")
        ]
        
        requirements_files = [
            "backend/requirements.txt",
            "backend/requirements-test.txt",
            "requirements.txt"
        ]
        
        vulnerability_found = False
        for req_file in requirements_files:
            if os.path.exists(req_file):
                with open(req_file, "r") as f:
                    content = f.read()
                    for pattern, description in vulnerable_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            self.add_vulnerability("medium", req_file, description)
                            vulnerability_found = True
        
        if not vulnerability_found:
            self.add_check(
                "Dependencies",
                "Known Vulnerabilities",
                "pass", 
                "No obvious vulnerable dependencies detected"
            )
    
    def check_api_security(self):
        """Check API security configuration."""
        print("\n🌐 API Security")
        
        # Check for security headers (in middleware or config)
        security_files = [
            "backend/core/middleware.py",
            "backend/core/security.py", 
            "backend/main.py"
        ]
        
        security_headers_found = False
        for file_path in security_files:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    content = f.read()
                    if any(header in content.lower() for header in [
                        "x-frame-options", "x-content-type-options", 
                        "x-xss-protection", "strict-transport-security"
                    ]):
                        security_headers_found = True
                        break
        
        if security_headers_found:
            self.add_check(
                "API",
                "Security Headers", 
                "pass",
                "Security headers configuration found"
            )
        else:
            self.add_check(
                "API",
                "Security Headers",
                "fail", 
                "Security headers not configured"
            )
        
        # Check for CORS configuration
        cors_configured = False
        for file_path in security_files:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    content = f.read()
                    if "cors" in content.lower():
                        cors_configured = True
                        break
        
        if cors_configured:
            self.add_check(
                "API", 
                "CORS Configuration",
                "pass",
                "CORS configuration found"
            )
        else:
            self.add_check(
                "API",
                "CORS Configuration",
                "fail",
                "CORS not configured"
            )
    
    def check_secrets_management(self):
        """Check secrets management."""
        print("\n🔐 Secrets Management")
        
        # Check gitignore for secrets
        gitignore_path = ".gitignore"
        if os.path.exists(gitignore_path):
            with open(gitignore_path, "r") as f:
                content = f.read()
                
                secret_patterns = [".env", "*.key", "*.pem", "secrets"]
                protected_patterns = sum(1 for pattern in secret_patterns if pattern in content)
                
                if protected_patterns >= 3:
                    self.add_check(
                        "Secrets",
                        "Gitignore Protection",
                        "pass",
                        f"Secret patterns protected in gitignore ({protected_patterns}/4)"
                    )
                else:
                    self.add_check(
                        "Secrets", 
                        "Gitignore Protection",
                        "fail",
                        f"Insufficient secret patterns in gitignore ({protected_patterns}/4)"
                    )
        
        # Scan for accidentally committed secrets (simplified)
        try:
            result = subprocess.run([
                "find", ".", "-name", "*.py", "-o", "-name", "*.js", "-o", "-name", "*.yml"
            ], capture_output=True, text=True)
            
            files_to_check = result.stdout.strip().split('\n')[:50]  # Limit to 50 files
            secrets_found = False
            
            secret_patterns = [
                r"sk-[a-zA-Z0-9]{32,}",  # OpenAI API keys
                r"AKIA[0-9A-Z]{16}",     # AWS access keys  
                r"-----BEGIN.*PRIVATE KEY-----"  # Private keys
            ]
            
            for file_path in files_to_check:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            for pattern in secret_patterns:
                                if re.search(pattern, content):
                                    # Exclude test files and examples
                                    if "test" not in file_path.lower() and "example" not in file_path.lower():
                                        self.add_vulnerability(
                                            "critical",
                                            file_path,
                                            f"Possible secret detected: {pattern}"
                                        )
                                        secrets_found = True
                    except Exception:
                        continue
            
            if not secrets_found:
                self.add_check(
                    "Secrets",
                    "Secret Scanning",
                    "pass",
                    "No obvious secrets detected in code"
                )
                
        except subprocess.CalledProcessError:
            self.add_check(
                "Secrets",
                "Secret Scanning", 
                "fail",
                "Could not perform secret scanning"
            )
    
    def generate_report(self):
        """Generate security report."""
        score_percentage = (self.results["score"] / self.results["max_score"]) * 100 if self.results["max_score"] > 0 else 0
        
        self.results["security_status"] = (
            "secure" if score_percentage >= 80 else
            "needs_attention" if score_percentage >= 60 else
            "insecure"
        )
        
        print(f"\n{'='*60}")
        print("🛡️ SECURITY VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"📊 Security Score: {self.results['score']}/{self.results['max_score']} ({score_percentage:.1f}%)")
        print(f"🏥 Security Status: {self.results['security_status'].upper()}")
        print(f"✅ Passed Checks: {len([c for c in self.results['checks'] if c['status'] == 'pass'])}")
        print(f"❌ Failed Checks: {len([c for c in self.results['checks'] if c['status'] == 'fail'])}")
        print(f"🚨 Vulnerabilities: {len(self.results['vulnerabilities'])}")
        
        if self.results["vulnerabilities"]:
            print(f"\n🚨 VULNERABILITIES DETECTED:")
            for vuln in self.results["vulnerabilities"]:
                print(f"   • {vuln['severity'].upper()}: {vuln['component']} - {vuln['description']}")
        
        # Save report
        with open("security_validation_report.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: security_validation_report.json")
        
        return score_percentage >= 80

    def run_validation(self):
        """Run complete security validation."""
        print("🛡️ AI AGENT PLATFORM - SECURITY VALIDATION")
        print("="*60)
        
        self.check_environment_files()
        self.check_docker_security() 
        self.check_dependencies()
        self.check_api_security()
        self.check_secrets_management()
        
        return self.generate_report()


if __name__ == "__main__":
    validator = SecurityValidator()
    success = validator.run_validation()
    exit(0 if success else 1)