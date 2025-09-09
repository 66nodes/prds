"""
Locust performance testing for Strategic Planning Platform API.
"""

import json
import random
from locust import HttpUser, task, between, events
import time
from faker import Faker
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

fake = Faker()


class PerformanceMetrics:
    """Track custom performance metrics."""
    
    def __init__(self):
        self.prd_generation_times = []
        self.validation_times = []
        self.api_response_times = []
        self.error_counts = {"auth": 0, "prd": 0, "validation": 0, "general": 0}
        
    def record_prd_generation(self, response_time):
        self.prd_generation_times.append(response_time)
        
    def record_validation(self, response_time):
        self.validation_times.append(response_time)
        
    def record_api_response(self, response_time):
        self.api_response_times.append(response_time)
        
    def record_error(self, category):
        self.error_counts[category] += 1

# Global metrics instance
metrics = PerformanceMetrics()


@events.request.add_listener
def request_handler(request_type, name, response_time, response_length, exception, context, **kwargs):
    """Custom request handler to track metrics."""
    metrics.record_api_response(response_time)
    
    if exception:
        if "auth" in name.lower():
            metrics.record_error("auth")
        elif "prd" in name.lower():
            metrics.record_error("prd")
        elif "validation" in name.lower():
            metrics.record_error("validation")
        else:
            metrics.record_error("general")


@events.test_stop.add_listener
def test_stop_handler(environment, **kwargs):
    """Generate performance report at test completion."""
    logger.info("=== PERFORMANCE TEST RESULTS ===")
    
    if metrics.prd_generation_times:
        avg_prd_time = sum(metrics.prd_generation_times) / len(metrics.prd_generation_times)
        logger.info(f"PRD Generation - Avg: {avg_prd_time:.2f}ms, Samples: {len(metrics.prd_generation_times)}")
    
    if metrics.validation_times:
        avg_validation_time = sum(metrics.validation_times) / len(metrics.validation_times)
        logger.info(f"Validation - Avg: {avg_validation_time:.2f}ms, Samples: {len(metrics.validation_times)}")
    
    if metrics.api_response_times:
        avg_api_time = sum(metrics.api_response_times) / len(metrics.api_response_times)
        logger.info(f"Overall API - Avg: {avg_api_time:.2f}ms, Samples: {len(metrics.api_response_times)}")
    
    logger.info(f"Error Counts: {metrics.error_counts}")


class APIUser(HttpUser):
    """Base API user with authentication capabilities."""
    
    wait_time = between(1, 3)
    weight = 1
    
    def on_start(self):
        """Initialize user session with authentication."""
        self.access_token = None
        self.project_id = None
        self.user_data = {
            "email": fake.email(),
            "password": "TestPassword123!",
            "name": fake.name()
        }
        
        # Register and authenticate user
        self.register_user()
        self.login_user()
        self.create_test_project()
    
    def register_user(self):
        """Register a new test user."""
        with self.client.post(
            "/api/auth/register",
            json=self.user_data,
            catch_response=True,
            name="auth_register"
        ) as response:
            if response.status_code == 201:
                response.success()
            elif response.status_code == 409:
                # User already exists, that's fine
                response.success()
            else:
                response.failure(f"Registration failed: {response.status_code}")
    
    def login_user(self):
        """Authenticate user and get access token."""
        with self.client.post(
            "/api/auth/login",
            json={
                "email": self.user_data["email"],
                "password": self.user_data["password"]
            },
            catch_response=True,
            name="auth_login"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("success") and "data" in data:
                    self.access_token = data["data"]["access_token"]
                    response.success()
                else:
                    response.failure("Login response missing token")
            else:
                response.failure(f"Login failed: {response.status_code}")
    
    def create_test_project(self):
        """Create a test project for PRD generation."""
        if not self.access_token:
            return
            
        headers = {"Authorization": f"Bearer {self.access_token}"}
        project_data = {
            "name": f"Load Test Project {fake.uuid4()[:8]}",
            "description": "Project created for performance testing"
        }
        
        with self.client.post(
            "/api/projects",
            json=project_data,
            headers=headers,
            catch_response=True,
            name="create_project"
        ) as response:
            if response.status_code == 201:
                data = response.json()
                if data.get("success") and "data" in data:
                    self.project_id = data["data"]["id"]
                    response.success()
                else:
                    response.failure("Project creation response missing ID")
            else:
                response.failure(f"Project creation failed: {response.status_code}")
    
    def get_auth_headers(self):
        """Get headers with authentication."""
        if self.access_token:
            return {"Authorization": f"Bearer {self.access_token}"}
        return {}


class PRDGenerationUser(APIUser):
    """User focused on PRD generation workflows."""
    
    weight = 3  # Higher weight for primary use case
    
    @task(5)
    def generate_prd(self):
        """Generate a PRD - main performance test."""
        if not self.access_token or not self.project_id:
            return
        
        prd_request = {
            "title": f"Performance Test PRD - {fake.catch_phrase()}",
            "description": fake.text(max_nb_chars=500),
            "project_id": self.project_id,
            "requirements": [
                fake.sentence() for _ in range(random.randint(3, 8))
            ],
            "constraints": [
                fake.sentence() for _ in range(random.randint(2, 5))
            ],
            "target_audience": fake.job(),
            "success_metrics": [
                f"{fake.word()} increased by {random.randint(10, 50)}%" 
                for _ in range(random.randint(2, 5))
            ]
        }
        
        headers = self.get_auth_headers()
        start_time = time.time()
        
        with self.client.post(
            f"/api/projects/{self.project_id}/prds/generate",
            json=prd_request,
            headers=headers,
            catch_response=True,
            name="prd_generate",
            timeout=120  # 2 minute timeout
        ) as response:
            end_time = time.time()
            generation_time = (end_time - start_time) * 1000  # Convert to ms
            
            if response.status_code == 201:
                data = response.json()
                if data.get("success"):
                    metrics.record_prd_generation(generation_time)
                    
                    # Store PRD ID for potential follow-up actions
                    if hasattr(self, 'generated_prds'):
                        self.generated_prds.append(data["data"]["id"])
                    else:
                        self.generated_prds = [data["data"]["id"]]
                    
                    response.success()
                    
                    # Log performance metrics
                    hallucination_rate = data["data"].get("hallucination_rate", 0)
                    validation_score = data["data"].get("validation_score", 0)
                    logger.info(f"PRD generated in {generation_time:.0f}ms, "
                              f"hallucination: {hallucination_rate:.3f}, "
                              f"validation: {validation_score:.3f}")
                else:
                    response.failure("PRD generation response not successful")
            else:
                response.failure(f"PRD generation failed: {response.status_code}")
    
    @task(2)
    def validate_prd(self):
        """Validate an existing PRD."""
        if not hasattr(self, 'generated_prds') or not self.generated_prds:
            return
        
        prd_id = random.choice(self.generated_prds)
        headers = self.get_auth_headers()
        start_time = time.time()
        
        with self.client.post(
            f"/api/projects/{self.project_id}/prds/{prd_id}/validate",
            headers=headers,
            catch_response=True,
            name="prd_validate"
        ) as response:
            end_time = time.time()
            validation_time = (end_time - start_time) * 1000
            
            if response.status_code == 200:
                metrics.record_validation(validation_time)
                response.success()
            else:
                response.failure(f"PRD validation failed: {response.status_code}")
    
    @task(1)
    def list_prds(self):
        """List PRDs for a project."""
        if not self.project_id:
            return
        
        headers = self.get_auth_headers()
        
        with self.client.get(
            f"/api/projects/{self.project_id}/prds",
            headers=headers,
            catch_response=True,
            name="prd_list"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"PRD list failed: {response.status_code}")
    
    @task(1)
    def get_prd_details(self):
        """Get details of a specific PRD."""
        if not hasattr(self, 'generated_prds') or not self.generated_prds:
            return
        
        prd_id = random.choice(self.generated_prds)
        headers = self.get_auth_headers()
        
        with self.client.get(
            f"/api/projects/{self.project_id}/prds/{prd_id}",
            headers=headers,
            catch_response=True,
            name="prd_details"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"PRD details failed: {response.status_code}")


class AuthenticationUser(APIUser):
    """User focused on authentication workflows."""
    
    weight = 1
    
    @task(3)
    def login_logout_cycle(self):
        """Test login/logout performance."""
        # Logout
        headers = self.get_auth_headers()
        with self.client.post(
            "/api/auth/logout",
            headers=headers,
            catch_response=True,
            name="auth_logout"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Logout failed: {response.status_code}")
        
        # Login again
        self.login_user()
    
    @task(1)
    def refresh_token(self):
        """Test token refresh."""
        if not self.access_token:
            return
            
        # This would require having a refresh token
        # For now, just test getting current user info
        headers = self.get_auth_headers()
        with self.client.get(
            "/api/auth/me",
            headers=headers,
            catch_response=True,
            name="auth_me"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Get current user failed: {response.status_code}")


class DashboardUser(APIUser):
    """User focused on dashboard and analytics."""
    
    weight = 2
    
    @task(2)
    def view_dashboard(self):
        """View dashboard analytics."""
        headers = self.get_auth_headers()
        
        with self.client.get(
            "/api/dashboard",
            headers=headers,
            catch_response=True,
            name="dashboard_view"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Dashboard view failed: {response.status_code}")
    
    @task(1)
    def project_analytics(self):
        """View project analytics."""
        if not self.project_id:
            return
        
        headers = self.get_auth_headers()
        
        with self.client.get(
            f"/api/projects/{self.project_id}/analytics",
            headers=headers,
            catch_response=True,
            name="project_analytics"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Project analytics failed: {response.status_code}")


class ConcurrentUser(APIUser):
    """User simulating concurrent operations."""
    
    weight = 2
    
    def on_start(self):
        """Initialize with multiple projects."""
        super().on_start()
        self.project_ids = []
        
        # Create multiple projects for concurrent testing
        for _ in range(3):
            self.create_additional_project()
    
    def create_additional_project(self):
        """Create additional test project."""
        if not self.access_token:
            return
            
        headers = {"Authorization": f"Bearer {self.access_token}"}
        project_data = {
            "name": f"Concurrent Test Project {fake.uuid4()[:8]}",
            "description": "Project for concurrent testing"
        }
        
        with self.client.post(
            "/api/projects",
            json=project_data,
            headers=headers,
            catch_response=True,
            name="create_concurrent_project"
        ) as response:
            if response.status_code == 201:
                data = response.json()
                if data.get("success") and "data" in data:
                    self.project_ids.append(data["data"]["id"])
                    response.success()
    
    @task(3)
    def concurrent_prd_generation(self):
        """Generate PRDs across multiple projects."""
        if not self.project_ids:
            return
        
        project_id = random.choice(self.project_ids)
        
        prd_request = {
            "title": f"Concurrent PRD - {fake.catch_phrase()}",
            "description": fake.text(max_nb_chars=300),
            "project_id": project_id,
            "requirements": [fake.sentence() for _ in range(random.randint(2, 5))],
            "constraints": [fake.sentence() for _ in range(random.randint(1, 3))]
        }
        
        headers = self.get_auth_headers()
        
        with self.client.post(
            f"/api/projects/{project_id}/prds/generate",
            json=prd_request,
            headers=headers,
            catch_response=True,
            name="concurrent_prd_generate",
            timeout=120
        ) as response:
            if response.status_code == 201:
                response.success()
            else:
                response.failure(f"Concurrent PRD generation failed: {response.status_code}")


class StressTestUser(HttpUser):
    """High-intensity stress testing user."""
    
    wait_time = between(0.1, 0.5)  # Very short wait time
    weight = 1
    
    def on_start(self):
        """Minimal setup for stress testing."""
        self.access_token = None
        # Use a shared test account for stress testing
        self.login_shared_account()
    
    def login_shared_account(self):
        """Login with shared stress test account."""
        with self.client.post(
            "/api/auth/login",
            json={
                "email": "stress.testexample.com",
                "password": "StressTest123!"
            },
            catch_response=True,
            name="stress_login"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("success") and "data" in data:
                    self.access_token = data["data"]["access_token"]
                    response.success()
    
    @task(5)
    def rapid_api_calls(self):
        """Make rapid API calls to test rate limiting."""
        if not self.access_token:
            return
        
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        # Test various lightweight endpoints rapidly
        endpoints = [
            "/api/auth/me",
            "/api/dashboard",
            "/api/projects",
            "/api/health"
        ]
        
        endpoint = random.choice(endpoints)
        
        with self.client.get(
            endpoint,
            headers=headers,
            catch_response=True,
            name="rapid_api_call"
        ) as response:
            if response.status_code in [200, 429]:  # 429 = Rate Limited (expected)
                response.success()
            else:
                response.failure(f"Rapid API call failed: {response.status_code}")


# Custom Locust tasks for specific scenarios
class LoadTestScenarios:
    """Define specific load test scenarios."""
    
    @staticmethod
    def peak_traffic_simulation():
        """Simulate peak traffic conditions."""
        return {
            "PRDGenerationUser": 10,
            "AuthenticationUser": 5,
            "DashboardUser": 8,
            "ConcurrentUser": 3
        }
    
    @staticmethod
    def steady_state_simulation():
        """Simulate normal steady-state traffic."""
        return {
            "PRDGenerationUser": 5,
            "AuthenticationUser": 2,
            "DashboardUser": 4,
            "ConcurrentUser": 1
        }
    
    @staticmethod
    def stress_test_simulation():
        """High-intensity stress testing."""
        return {
            "PRDGenerationUser": 15,
            "StressTestUser": 10,
            "ConcurrentUser": 5
        }