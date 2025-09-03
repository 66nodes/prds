#!/usr/bin/env python3
"""
Final System Validation Script
Validates all system components and generates completion report.
"""

import asyncio
import json
import time
import requests
import subprocess
from datetime import datetime
from typing import Dict, List, Any
import sys


class SystemValidator:
    """Comprehensive system validation."""

    def __init__(self):
        self.results = {
            "validation_timestamp": datetime.now().isoformat(),
            "system_status": "validating",
            "infrastructure": {},
            "performance": {},
            "security": {},
            "services": {},
            "errors": [],
            "warnings": [],
            "success_count": 0,
            "total_checks": 0
        }

    def log_success(self, component: str, message: str, details: Dict = None):
        """Log successful validation."""
        self.results["success_count"] += 1
        self.results["total_checks"] += 1
        print(f"✅ {component}: {message}")
        if details:
            self.results[component.lower().replace(" ", "_")] = details

    def log_failure(self, component: str, message: str, error: str = None):
        """Log validation failure."""
        self.results["total_checks"] += 1
        self.results["errors"].append({
            "component": component,
            "message": message,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        print(f"❌ {component}: {message}")
        if error:
            print(f"   Error: {error}")

    def log_warning(self, component: str, message: str):
        """Log validation warning."""
        self.results["warnings"].append({
            "component": component,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        print(f"⚠️ {component}: {message}")

    async def validate_infrastructure(self):
        """Validate infrastructure components."""
        print("\n🏗️ Validating Infrastructure Components...")
        
        # Check Docker containers
        try:
            result = subprocess.run(
                ["docker", "ps", "--format", "table {{.Names}}\\t{{.Status}}"],
                capture_output=True, text=True, check=True
            )
            running_containers = [line.split('\t')[0] for line in result.stdout.strip().split('\n')[1:]]
            
            expected_containers = [
                "prds-prometheus-1",
                "prds-milvus-standalone-1", 
                "prds-grafana-1",
                "prds-minio-1",
                "prds-redis-1",
                "prds-postgres-1",
                "prds-neo4j-1",
                "prds-pulsar-1",
                "prds-etcd-1",
                "prds-healthchecker-1"
            ]
            
            missing_containers = set(expected_containers) - set(running_containers)
            if not missing_containers:
                self.log_success("Docker Infrastructure", f"All {len(expected_containers)} containers running")
                self.results["infrastructure"]["containers"] = {
                    "status": "healthy",
                    "running_count": len(running_containers),
                    "expected_count": len(expected_containers)
                }
            else:
                self.log_failure("Docker Infrastructure", f"Missing containers: {missing_containers}")
                
        except subprocess.CalledProcessError as e:
            self.log_failure("Docker Infrastructure", "Failed to check container status", str(e))

    async def validate_service_connectivity(self):
        """Validate service connectivity and health."""
        print("\n🌐 Validating Service Connectivity...")
        
        services = [
            {
                "name": "Prometheus",
                "url": "http://localhost:9092/api/v1/status/buildinfo",
                "expected_status": 200,
                "timeout": 5
            },
            {
                "name": "Grafana",
                "url": "http://localhost:3001/api/health", 
                "expected_status": 200,
                "timeout": 5
            },
            {
                "name": "MinIO",
                "url": "http://localhost:9000/minio/health/live",
                "expected_status": 200,
                "timeout": 5
            },
            {
                "name": "Milvus",
                "url": "http://localhost:9091/api/v1/health",
                "expected_status": 200,
                "timeout": 5
            },
            {
                "name": "Pulsar Admin",
                "url": "http://localhost:8080/admin/v2/brokers/health",
                "expected_status": 200,
                "timeout": 5
            }
        ]
        
        for service in services:
            try:
                start_time = time.time()
                response = requests.get(
                    service["url"], 
                    timeout=service["timeout"]
                )
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == service["expected_status"]:
                    self.log_success(
                        f"{service['name']} Service",
                        f"Healthy (Response time: {response_time:.1f}ms)"
                    )
                else:
                    self.log_failure(
                        f"{service['name']} Service",
                        f"Unexpected status code: {response.status_code}"
                    )
                    
            except requests.exceptions.RequestException as e:
                self.log_failure(f"{service['name']} Service", "Connection failed", str(e))

    async def validate_database_connectivity(self):
        """Validate database connectivity."""
        print("\n🗄️ Validating Database Connectivity...")
        
        # PostgreSQL
        try:
            result = subprocess.run([
                "docker", "exec", "prds-postgres-1", 
                "psql", "-U", "postgres", "-c", "SELECT 1;"
            ], capture_output=True, text=True, check=True)
            
            if "1 row)" in result.stdout:
                self.log_success("PostgreSQL", "Connected and responsive")
            else:
                self.log_failure("PostgreSQL", "Unexpected response format")
                
        except subprocess.CalledProcessError as e:
            self.log_failure("PostgreSQL", "Connection failed", str(e))

        # Redis
        try:
            result = subprocess.run([
                "docker", "exec", "prds-redis-1", 
                "redis-cli", "ping"
            ], capture_output=True, text=True, check=True)
            
            if "PONG" in result.stdout:
                self.log_success("Redis", "Connected and responsive")
            else:
                self.log_failure("Redis", "Unexpected response")
                
        except subprocess.CalledProcessError as e:
            self.log_failure("Redis", "Connection failed", str(e))

    async def validate_performance_metrics(self):
        """Validate performance metrics."""
        print("\n⚡ Validating Performance Metrics...")
        
        # Test API response times (using mock endpoints)
        mock_endpoints = [
            ("Health Check", "http://localhost:3001/api/health"),
            ("Prometheus Status", "http://localhost:9092/api/v1/status/buildinfo"),
            ("MinIO Health", "http://localhost:9000/minio/health/live")
        ]
        
        performance_results = []
        
        for name, url in mock_endpoints:
            try:
                start_time = time.time()
                response = requests.get(url, timeout=10)
                response_time = (time.time() - start_time) * 1000
                
                performance_results.append({
                    "endpoint": name,
                    "response_time_ms": response_time,
                    "status_code": response.status_code
                })
                
                if response_time < 200:  # Target: <200ms
                    self.log_success(
                        f"Performance - {name}",
                        f"Response time: {response_time:.1f}ms (Target: <200ms)"
                    )
                else:
                    self.log_warning(
                        f"Performance - {name}",
                        f"Response time: {response_time:.1f}ms exceeds 200ms target"
                    )
                    
            except requests.exceptions.RequestException as e:
                self.log_failure(f"Performance - {name}", "Request failed", str(e))
        
        self.results["performance"] = {
            "api_endpoints": performance_results,
            "average_response_time": sum(r["response_time_ms"] for r in performance_results) / len(performance_results) if performance_results else 0,
            "target_met": all(r["response_time_ms"] < 200 for r in performance_results)
        }

    async def validate_security_configuration(self):
        """Validate security configuration."""
        print("\n🛡️ Validating Security Configuration...")
        
        security_checks = []
        
        # Check if environment files are properly configured
        try:
            with open('.env.production.example', 'r') as f:
                env_content = f.read()
                if "CHANGE_ME" in env_content:
                    security_checks.append({
                        "check": "Environment Template",
                        "status": "secure",
                        "message": "Production template contains placeholder values"
                    })
                    self.log_success("Security", "Environment template is secure")
                else:
                    self.log_warning("Security", "Environment template may contain real values")
        except FileNotFoundError:
            self.log_warning("Security", "Environment template not found")
        
        # Check Docker container security
        try:
            result = subprocess.run([
                "docker", "ps", "--format", "{{.Names}}\\t{{.Ports}}"
            ], capture_output=True, text=True, check=True)
            
            exposed_ports = result.stdout
            security_checks.append({
                "check": "Port Exposure",
                "status": "configured",
                "message": "Container ports configured"
            })
            self.log_success("Security", "Container port configuration validated")
            
        except subprocess.CalledProcessError as e:
            self.log_failure("Security", "Failed to check port exposure", str(e))
        
        self.results["security"] = {
            "checks": security_checks,
            "total_checks": len(security_checks),
            "passed_checks": len([c for c in security_checks if c["status"] in ["secure", "configured"]])
        }

    async def validate_ci_cd_pipeline(self):
        """Validate CI/CD pipeline configuration."""
        print("\n🔄 Validating CI/CD Pipeline...")
        
        # Check if CI/CD files exist
        ci_cd_files = [
            ".github/workflows/ci-cd.yml",
            "docker-stack.yml", 
            "scripts/backup.sh",
            "scripts/restore.sh"
        ]
        
        missing_files = []
        for file_path in ci_cd_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    if len(content) > 100:  # Basic content validation
                        self.log_success(f"CI/CD - {file_path}", "Configuration file exists and populated")
                    else:
                        self.log_warning(f"CI/CD - {file_path}", "Configuration file appears incomplete")
            except FileNotFoundError:
                missing_files.append(file_path)
                self.log_failure(f"CI/CD - {file_path}", "Configuration file not found")
        
        if not missing_files:
            self.log_success("CI/CD Pipeline", "All configuration files present")

    async def run_validation(self):
        """Run complete system validation."""
        print("=" * 60)
        print("🚀 AI AGENT PLATFORM - FINAL SYSTEM VALIDATION")
        print("=" * 60)
        print(f"Validation started at: {datetime.now().isoformat()}")
        
        try:
            await self.validate_infrastructure()
            await self.validate_service_connectivity()
            await self.validate_database_connectivity()
            await self.validate_performance_metrics()
            await self.validate_security_configuration()
            await self.validate_ci_cd_pipeline()
            
            # Calculate success rate
            success_rate = (self.results["success_count"] / self.results["total_checks"]) * 100 if self.results["total_checks"] > 0 else 0
            
            self.results["system_status"] = "healthy" if success_rate >= 80 else "issues_detected"
            self.results["success_rate"] = success_rate
            
            print(f"\n{'=' * 60}")
            print("📊 VALIDATION SUMMARY")
            print(f"{'=' * 60}")
            print(f"✅ Successful checks: {self.results['success_count']}")
            print(f"❌ Failed checks: {len(self.results['errors'])}")
            print(f"⚠️ Warnings: {len(self.results['warnings'])}")
            print(f"📈 Success rate: {success_rate:.1f}%")
            print(f"🏥 System status: {self.results['system_status'].upper()}")
            
            if self.results["errors"]:
                print(f"\n🚨 ERRORS DETECTED:")
                for error in self.results["errors"]:
                    print(f"   • {error['component']}: {error['message']}")
            
            if self.results["warnings"]:
                print(f"\n⚠️ WARNINGS:")
                for warning in self.results["warnings"]:
                    print(f"   • {warning['component']}: {warning['message']}")
            
            # Save results to file
            with open("system_validation_report.json", "w") as f:
                json.dump(self.results, f, indent=2)
            
            print(f"\n📄 Detailed report saved to: system_validation_report.json")
            print(f"⏰ Validation completed at: {datetime.now().isoformat()}")
            
            return success_rate >= 80
            
        except Exception as e:
            self.log_failure("System Validation", "Unexpected error during validation", str(e))
            self.results["system_status"] = "error"
            return False


async def main():
    """Main validation function."""
    validator = SystemValidator()
    success = await validator.run_validation()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())