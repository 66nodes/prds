#!/usr/bin/env python3
"""
AI Agent Platform - Infrastructure Validation Tests
Comprehensive health checks for Hybrid RAG infrastructure
"""

import requests
import time
import sys
import subprocess
from typing import Dict, List, Tuple

class InfrastructureValidator:
    def __init__(self):
        self.results = []
        self.endpoints = {
            "milvus": "http://localhost:9091/healthz",
            "neo4j": "http://localhost:7474",
            "pulsar": "http://localhost:8080/admin/v2/namespaces/public",
            "prometheus": "http://localhost:9092/-/healthy", 
            "grafana": "http://localhost:3001/api/health",
            "minio": "http://localhost:9000/minio/health/live"
        }
    
    def test_endpoint(self, name: str, url: str, timeout: int = 10) -> Tuple[bool, str]:
        """Test a single HTTP endpoint"""
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return True, f"✅ {name}: Healthy (HTTP {response.status_code})"
            else:
                return False, f"❌ {name}: Unhealthy (HTTP {response.status_code})"
        except requests.exceptions.ConnectionError:
            return False, f"❌ {name}: Connection failed"
        except requests.exceptions.Timeout:
            return False, f"❌ {name}: Timeout"
        except Exception as e:
            return False, f"❌ {name}: Error - {str(e)}"
    
    def test_database_connections(self) -> List[Tuple[bool, str]]:
        """Test database connectivity"""
        results = []
        
        # Test Redis
        try:
            result = subprocess.run(
                ["docker", "exec", "prds-redis-1", "redis-cli", "ping"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and "PONG" in result.stdout:
                results.append((True, "✅ Redis: Connection successful"))
            else:
                results.append((False, f"❌ Redis: Command failed - {result.stderr}"))
        except Exception as e:
            results.append((False, f"❌ Redis: Test failed - {str(e)}"))
        
        # Test PostgreSQL
        try:
            result = subprocess.run(
                ["docker", "exec", "prds-postgres-1", "pg_isready", "-U", "postgres", "-d", "aiplatform"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                results.append((True, "✅ PostgreSQL: Connection successful"))
            else:
                results.append((False, f"❌ PostgreSQL: Connection failed - {result.stderr}"))
        except Exception as e:
            results.append((False, f"❌ PostgreSQL: Test failed - {str(e)}"))
        
        # Test Neo4j cypher connection
        try:
            result = subprocess.run([
                "docker", "exec", "prds-neo4j-1", "cypher-shell", "-u", "neo4j", "-p", "development", 
                "RETURN 1 AS test;"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and "1" in result.stdout:
                results.append((True, "✅ Neo4j: Cypher query successful"))
            else:
                results.append((False, f"❌ Neo4j: Cypher query failed - {result.stderr}"))
        except Exception as e:
            results.append((False, f"❌ Neo4j: Test failed - {str(e)}"))
        
        return results
    
    def test_milvus_functionality(self) -> Tuple[bool, str]:
        """Test Milvus vector database functionality"""
        try:
            # Test basic health endpoint first
            health_ok, health_msg = self.test_endpoint("Milvus Health", "http://localhost:9091/healthz")
            if not health_ok:
                return False, health_msg
            
            # Test metrics endpoint  
            metrics_response = requests.get("http://localhost:9091/metrics", timeout=10)
            if metrics_response.status_code == 200:
                return True, "✅ Milvus: Health check and metrics accessible"
            else:
                return False, f"❌ Milvus: Metrics endpoint failed (HTTP {metrics_response.status_code})"
        except Exception as e:
            return False, f"❌ Milvus: Functionality test failed - {str(e)}"
    
    def test_monitoring_stack(self) -> List[Tuple[bool, str]]:
        """Test Prometheus and Grafana monitoring"""
        results = []
        
        # Test Prometheus metrics
        try:
            response = requests.get("http://localhost:9092/api/v1/query?query=up", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    results.append((True, "✅ Prometheus: Metrics query successful"))
                else:
                    results.append((False, "❌ Prometheus: Metrics query failed"))
            else:
                results.append((False, f"❌ Prometheus: HTTP {response.status_code}"))
        except Exception as e:
            results.append((False, f"❌ Prometheus: Test failed - {str(e)}"))
        
        # Test Grafana API
        try:
            response = requests.get("http://localhost:3001/api/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("database") == "ok":
                    results.append((True, "✅ Grafana: API and database healthy"))
                else:
                    results.append((False, "❌ Grafana: Database unhealthy"))
            else:
                results.append((False, f"❌ Grafana: HTTP {response.status_code}"))
        except Exception as e:
            results.append((False, f"❌ Grafana: Test failed - {str(e)}"))
        
        return results
    
    def test_pulsar_functionality(self) -> Tuple[bool, str]:
        """Test Pulsar message broker functionality"""
        try:
            # Test admin API
            response = requests.get("http://localhost:8080/admin/v2/namespaces/public", timeout=10)
            if response.status_code == 200:
                return True, "✅ Pulsar: Admin API accessible"
            else:
                return False, f"❌ Pulsar: Admin API failed (HTTP {response.status_code})"
        except Exception as e:
            return False, f"❌ Pulsar: Test failed - {str(e)}"
    
    def test_storage_services(self) -> List[Tuple[bool, str]]:
        """Test MinIO and other storage services"""
        results = []
        
        # Test MinIO health
        try:
            response = requests.get("http://localhost:9000/minio/health/live", timeout=10)
            if response.status_code == 200:
                results.append((True, "✅ MinIO: Health check successful"))
            else:
                results.append((False, f"❌ MinIO: Health check failed (HTTP {response.status_code})"))
        except Exception as e:
            results.append((False, f"❌ MinIO: Test failed - {str(e)}"))
        
        return results
    
    def run_comprehensive_tests(self) -> Dict:
        """Run all infrastructure validation tests"""
        print("🚀 AI Agent Platform - Infrastructure Validation Tests")
        print("=" * 60)
        
        all_results = []
        
        # Test HTTP endpoints
        print("\n📡 Testing HTTP Endpoints...")
        for service, url in self.endpoints.items():
            success, message = self.test_endpoint(service, url)
            all_results.append((success, message))
            print(f"  {message}")
        
        # Test database connections
        print("\n🗄️  Testing Database Connections...")
        db_results = self.test_database_connections()
        for success, message in db_results:
            all_results.append((success, message))
            print(f"  {message}")
        
        # Test Milvus functionality
        print("\n🔍 Testing Vector Database...")
        milvus_success, milvus_message = self.test_milvus_functionality()
        all_results.append((milvus_success, milvus_message))
        print(f"  {milvus_message}")
        
        # Test monitoring
        print("\n📊 Testing Monitoring Stack...")
        monitoring_results = self.test_monitoring_stack()
        for success, message in monitoring_results:
            all_results.append((success, message))
            print(f"  {message}")
        
        # Test Pulsar
        print("\n📨 Testing Message Broker...")
        pulsar_success, pulsar_message = self.test_pulsar_functionality()
        all_results.append((pulsar_success, pulsar_message))
        print(f"  {pulsar_message}")
        
        # Test storage services
        print("\n💾 Testing Storage Services...")
        storage_results = self.test_storage_services()
        for success, message in storage_results:
            all_results.append((success, message))
            print(f"  {message}")
        
        # Summary
        passed = sum(1 for success, _ in all_results if success)
        total = len(all_results)
        
        print("\n" + "=" * 60)
        print(f"📋 VALIDATION SUMMARY")
        print(f"   Passed: {passed}/{total} tests")
        print(f"   Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            print("🎉 ALL INFRASTRUCTURE TESTS PASSED!")
            status = "success"
        else:
            print("⚠️  Some infrastructure tests failed")
            status = "partial"
        
        print("\n🔗 Service Access URLs:")
        print("   • Neo4j Browser: http://localhost:7474 (neo4j/development)")
        print("   • Milvus Admin: http://localhost:9091")
        print("   • Pulsar Admin: http://localhost:8080")
        print("   • Prometheus: http://localhost:9092")
        print("   • Grafana: http://localhost:3001 (admin/development)")
        print("   • MinIO Console: http://localhost:9001 (minioadmin/minioadmin)")
        
        return {
            "status": status,
            "passed": passed,
            "total": total,
            "success_rate": (passed/total)*100,
            "results": all_results
        }

def main():
    validator = InfrastructureValidator()
    results = validator.run_comprehensive_tests()
    
    if results["status"] == "success":
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()