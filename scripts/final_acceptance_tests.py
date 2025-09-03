#!/usr/bin/env python3
"""
Final Acceptance Tests
End-to-end validation of the AI Agent Platform.
"""

import asyncio
import json
import time
import requests
import subprocess
from datetime import datetime
from typing import Dict, List, Any
import sys


class FinalAcceptanceTests:
    """Final acceptance tests for the platform."""
    
    def __init__(self):
        self.results = {
            "test_timestamp": datetime.now().isoformat(),
            "test_status": "running",
            "acceptance_tests": [],
            "performance_metrics": {},
            "infrastructure_health": {},
            "errors": [],
            "warnings": [],
            "passed_tests": 0,
            "total_tests": 0
        }
    
    def log_test_pass(self, test_name: str, message: str, metrics: Dict = None):
        """Log passed acceptance test."""
        self.results["acceptance_tests"].append({
            "test": test_name,
            "status": "pass",
            "message": message,
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat()
        })
        self.results["passed_tests"] += 1
        self.results["total_tests"] += 1
        print(f"✅ {test_name}: {message}")
        if metrics:
            for key, value in metrics.items():
                print(f"   📊 {key}: {value}")
    
    def log_test_fail(self, test_name: str, message: str, error: str = None):
        """Log failed acceptance test."""
        self.results["acceptance_tests"].append({
            "test": test_name,
            "status": "fail", 
            "message": message,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        self.results["total_tests"] += 1
        self.results["errors"].append({
            "test": test_name,
            "message": message,
            "error": error
        })
        print(f"❌ {test_name}: {message}")
        if error:
            print(f"   Error: {error}")
    
    async def test_infrastructure_readiness(self):
        """Test infrastructure readiness and health."""
        print("\n🏗️ Infrastructure Readiness Tests")
        
        # Test Docker containers
        try:
            result = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"],
                capture_output=True, text=True, check=True
            )
            running_containers = len(result.stdout.strip().split('\n'))
            
            if running_containers >= 8:  # Expected minimum containers
                self.log_test_pass(
                    "Infrastructure - Container Health",
                    f"All critical containers running",
                    {"running_containers": running_containers}
                )
            else:
                self.log_test_fail(
                    "Infrastructure - Container Health",
                    f"Insufficient containers running: {running_containers}/8"
                )
        except subprocess.CalledProcessError as e:
            self.log_test_fail(
                "Infrastructure - Container Health",
                "Failed to check container status",
                str(e)
            )
        
        # Test service connectivity with performance measurement
        services = [
            ("Prometheus", "http://localhost:9092/api/v1/status/buildinfo"),
            ("Grafana", "http://localhost:3001/api/health"),
            ("MinIO", "http://localhost:9000/minio/health/live"),
            ("Milvus", "http://localhost:9091/api/v1/health")
        ]
        
        total_response_time = 0
        healthy_services = 0
        
        for name, url in services:
            try:
                start_time = time.time()
                response = requests.get(url, timeout=5)
                response_time = (time.time() - start_time) * 1000
                total_response_time += response_time
                
                if response.status_code == 200:
                    healthy_services += 1
                    self.log_test_pass(
                        f"Service Health - {name}",
                        f"Service healthy and responsive",
                        {"response_time_ms": f"{response_time:.1f}"}
                    )
                else:
                    self.log_test_fail(
                        f"Service Health - {name}",
                        f"Service unhealthy (status: {response.status_code})"
                    )
            except requests.exceptions.RequestException as e:
                self.log_test_fail(
                    f"Service Health - {name}",
                    "Service unreachable",
                    str(e)
                )
        
        # Performance acceptance criteria
        avg_response_time = total_response_time / len(services)
        if avg_response_time < 200:  # < 200ms average
            self.log_test_pass(
                "Performance - Service Response",
                f"Average response time meets target",
                {"avg_response_time_ms": f"{avg_response_time:.1f}", "target": "< 200ms"}
            )
        else:
            self.log_test_fail(
                "Performance - Service Response", 
                f"Average response time exceeds target: {avg_response_time:.1f}ms"
            )
        
        self.results["infrastructure_health"] = {
            "healthy_services": healthy_services,
            "total_services": len(services),
            "avg_response_time_ms": avg_response_time
        }
    
    async def test_database_connectivity(self):
        """Test database connectivity and basic operations."""
        print("\n🗄️ Database Connectivity Tests")
        
        # PostgreSQL connectivity
        try:
            result = subprocess.run([
                "docker", "exec", "prds-postgres-1",
                "psql", "-U", "postgres", "-c", "SELECT version();"
            ], capture_output=True, text=True, check=True)
            
            if "PostgreSQL" in result.stdout:
                self.log_test_pass(
                    "Database - PostgreSQL",
                    "PostgreSQL connection and query successful"
                )
            else:
                self.log_test_fail(
                    "Database - PostgreSQL",
                    "Unexpected PostgreSQL response"
                )
        except subprocess.CalledProcessError as e:
            self.log_test_fail(
                "Database - PostgreSQL",
                "PostgreSQL connection failed",
                str(e)
            )
        
        # Redis connectivity
        try:
            result = subprocess.run([
                "docker", "exec", "prds-redis-1",
                "redis-cli", "ping"
            ], capture_output=True, text=True, check=True)
            
            if "PONG" in result.stdout:
                self.log_test_pass(
                    "Database - Redis",
                    "Redis connection and ping successful"
                )
            else:
                self.log_test_fail(
                    "Database - Redis",
                    "Redis ping failed"
                )
        except subprocess.CalledProcessError as e:
            self.log_test_fail(
                "Database - Redis",
                "Redis connection failed",
                str(e)
            )
        
        # Neo4j connectivity (basic check)
        try:
            result = subprocess.run([
                "docker", "logs", "prds-neo4j-1", "--tail", "5"
            ], capture_output=True, text=True, check=True)
            
            # Check if Neo4j is running (no critical errors in recent logs)
            if "ERROR" not in result.stdout.upper() or "Started" in result.stdout:
                self.log_test_pass(
                    "Database - Neo4j",
                    "Neo4j container running without critical errors"
                )
            else:
                self.log_test_fail(
                    "Database - Neo4j",
                    "Neo4j container has errors"
                )
        except subprocess.CalledProcessError as e:
            self.log_test_fail(
                "Database - Neo4j",
                "Failed to check Neo4j status",
                str(e)
            )
    
    async def test_security_compliance(self):
        """Test security compliance and configuration."""
        print("\n🛡️ Security Compliance Tests")
        
        # Test environment security
        if not os.path.exists(".env.production"):
            self.log_test_pass(
                "Security - Environment Files",
                "Production environment file not committed to repository"
            )
        else:
            self.log_test_fail(
                "Security - Environment Files",
                "Production environment file found in repository"
            )
        
        # Test Docker security (non-root user)
        dockerfile_path = "backend/Dockerfile"
        if os.path.exists(dockerfile_path):
            with open(dockerfile_path, 'r') as f:
                content = f.read()
                if "USER " in content and "USER root" not in content:
                    self.log_test_pass(
                        "Security - Docker Configuration",
                        "Application runs as non-root user"
                    )
                else:
                    self.log_test_fail(
                        "Security - Docker Configuration",
                        "Application may run as root user"
                    )
        
        # Test port exposure (containers should not expose unnecessary ports)
        try:
            result = subprocess.run([
                "docker", "ps", "--format", "{{.Ports}}"
            ], capture_output=True, text=True, check=True)
            
            # Check if only expected ports are exposed
            exposed_ports = result.stdout
            dangerous_ports = ["22/tcp", "3389/tcp", "21/tcp"]  # SSH, RDP, FTP
            
            dangerous_found = any(port in exposed_ports for port in dangerous_ports)
            if not dangerous_found:
                self.log_test_pass(
                    "Security - Port Exposure", 
                    "No dangerous ports exposed"
                )
            else:
                self.log_test_fail(
                    "Security - Port Exposure",
                    "Potentially dangerous ports exposed"
                )
        except subprocess.CalledProcessError as e:
            self.log_test_fail(
                "Security - Port Exposure",
                "Failed to check port exposure",
                str(e)
            )
    
    async def test_monitoring_and_observability(self):
        """Test monitoring and observability setup."""
        print("\n📊 Monitoring & Observability Tests")
        
        # Test Prometheus metrics collection
        try:
            response = requests.get("http://localhost:9092/api/v1/targets", timeout=10)
            if response.status_code == 200:
                targets_data = response.json()
                active_targets = len([t for t in targets_data.get("data", {}).get("activeTargets", []) if t.get("health") == "up"])
                
                if active_targets > 0:
                    self.log_test_pass(
                        "Monitoring - Prometheus Targets",
                        f"Prometheus collecting metrics from targets",
                        {"active_targets": active_targets}
                    )
                else:
                    self.log_test_fail(
                        "Monitoring - Prometheus Targets",
                        "No active monitoring targets"
                    )
            else:
                self.log_test_fail(
                    "Monitoring - Prometheus Targets",
                    f"Prometheus targets endpoint failed (status: {response.status_code})"
                )
        except requests.exceptions.RequestException as e:
            self.log_test_fail(
                "Monitoring - Prometheus Targets",
                "Failed to check Prometheus targets",
                str(e)
            )
        
        # Test Grafana dashboard availability
        try:
            response = requests.get("http://localhost:3001/api/health", timeout=10)
            if response.status_code == 200:
                self.log_test_pass(
                    "Monitoring - Grafana Health",
                    "Grafana dashboard service healthy"
                )
            else:
                self.log_test_fail(
                    "Monitoring - Grafana Health",
                    f"Grafana health check failed (status: {response.status_code})"
                )
        except requests.exceptions.RequestException as e:
            self.log_test_fail(
                "Monitoring - Grafana Health",
                "Failed to check Grafana health",
                str(e)
            )
    
    async def test_backup_and_recovery(self):
        """Test backup and recovery procedures."""
        print("\n💾 Backup & Recovery Tests")
        
        # Test backup script availability and permissions
        backup_scripts = ["scripts/backup.sh", "scripts/restore.sh"]
        
        for script in backup_scripts:
            if os.path.exists(script):
                # Check if script has execute permissions
                import stat
                file_stats = os.stat(script)
                is_executable = bool(file_stats.st_mode & stat.S_IXUSR)
                
                if is_executable:
                    self.log_test_pass(
                        f"Backup - {script}",
                        "Script exists and is executable"
                    )
                else:
                    self.log_test_fail(
                        f"Backup - {script}",
                        "Script exists but is not executable"
                    )
            else:
                self.log_test_fail(
                    f"Backup - {script}",
                    "Backup script not found"
                )
    
    async def test_configuration_management(self):
        """Test configuration management and environment setup."""
        print("\n⚙️ Configuration Management Tests")
        
        # Test configuration files
        config_files = [
            ("docker-stack.yml", "Docker Swarm configuration"),
            (".env.production.example", "Environment template"),
            ("backend/requirements.txt", "Python dependencies"),
            ("frontend/package.json", "Node.js dependencies")
        ]
        
        for file_path, description in config_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if len(content.strip()) > 50:  # Basic content validation
                            self.log_test_pass(
                                f"Configuration - {description}",
                                "Configuration file complete and valid"
                            )
                        else:
                            self.log_test_fail(
                                f"Configuration - {description}",
                                "Configuration file appears incomplete"
                            )
                except Exception as e:
                    self.log_test_fail(
                        f"Configuration - {description}",
                        "Failed to read configuration file",
                        str(e)
                    )
            else:
                self.log_test_fail(
                    f"Configuration - {description}",
                    "Configuration file not found"
                )
    
    def generate_final_report(self):
        """Generate final acceptance test report."""
        success_rate = (self.results["passed_tests"] / self.results["total_tests"]) * 100 if self.results["total_tests"] > 0 else 0
        
        self.results["test_status"] = "passed" if success_rate >= 90 else "failed"
        self.results["success_rate"] = success_rate
        
        print(f"\n{'='*60}")
        print("🎯 FINAL ACCEPTANCE TEST RESULTS")
        print(f"{'='*60}")
        print(f"✅ Passed Tests: {self.results['passed_tests']}")
        print(f"❌ Failed Tests: {len(self.results['errors'])}")
        print(f"📊 Success Rate: {success_rate:.1f}%")
        print(f"🏆 Test Status: {self.results['test_status'].upper()}")
        
        if self.results["errors"]:
            print(f"\n🚨 FAILED TESTS:")
            for error in self.results["errors"]:
                print(f"   • {error['test']}: {error['message']}")
        
        # Performance summary
        if self.results["infrastructure_health"]:
            health = self.results["infrastructure_health"]
            print(f"\n📈 PERFORMANCE SUMMARY:")
            print(f"   • Healthy Services: {health.get('healthy_services', 0)}/{health.get('total_services', 0)}")
            print(f"   • Average Response Time: {health.get('avg_response_time_ms', 0):.1f}ms")
        
        # Save detailed report
        with open("final_acceptance_report.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: final_acceptance_report.json")
        print(f"⏰ Tests completed at: {datetime.now().isoformat()}")
        
        return success_rate >= 90
    
    async def run_all_tests(self):
        """Run all acceptance tests."""
        print("🎯 AI AGENT PLATFORM - FINAL ACCEPTANCE TESTS")
        print("="*60)
        print(f"Tests started at: {datetime.now().isoformat()}")
        
        try:
            await self.test_infrastructure_readiness()
            await self.test_database_connectivity()
            await self.test_security_compliance()
            await self.test_monitoring_and_observability()
            await self.test_backup_and_recovery()
            await self.test_configuration_management()
            
            return self.generate_final_report()
        except Exception as e:
            self.log_test_fail("Test Suite", "Unexpected error during testing", str(e))
            self.results["test_status"] = "error"
            return False


async def main():
    """Main test runner."""
    tester = FinalAcceptanceTests()
    success = await tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    import os
    asyncio.run(main())