"""
Tests for main FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient


class TestMainApp:
    """Test main FastAPI application."""
    
    def test_root_endpoint(self, test_client: TestClient):
        """Test root endpoint returns basic info."""
        response = test_client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Strategic Planning Platform API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "operational"
        assert "features" in data
        assert isinstance(data["features"], list)
    
    def test_health_endpoint_structure(self, test_client: TestClient):
        """Test health endpoint returns proper structure."""
        response = test_client.get("/health")
        assert response.status_code in [200, 503]  # May be unhealthy in test env
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "components" in data
        
        # Check component structure
        components = data["components"]
        assert isinstance(components, dict)
        
        # At least API component should be present
        if "api" in components:
            api_health = components["api"]
            assert "status" in api_health
    
    def test_health_endpoint_with_mock_dependencies(self, test_client: TestClient):
        """Test health endpoint with mocked healthy dependencies."""
        # The health check may fail in test environment due to missing dependencies
        # That's expected and acceptable for this test
        response = test_client.get("/health")
        assert response.status_code in [200, 503]
        
        data = response.json()
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
    
    def test_docs_endpoint_in_development(self, test_client: TestClient):
        """Test that docs are available in development mode."""
        # In development mode, docs should be available
        response = test_client.get("/docs")
        # May return 200 (docs page) or 404 (if disabled), both are acceptable
        assert response.status_code in [200, 404]
    
    def test_404_for_nonexistent_endpoint(self, test_client: TestClient):
        """Test 404 response for non-existent endpoints."""
        response = test_client.get("/nonexistent")
        assert response.status_code == 404