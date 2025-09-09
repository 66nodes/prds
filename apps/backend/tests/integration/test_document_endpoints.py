"""
Integration tests for document generation API endpoints.
"""

import pytest
import json
from unittest.mock import patch, AsyncMock, Mock
from fastapi.testclient import TestClient
from httpx import AsyncClient

from main import app
from services.document.export_service import ExportFormat


class TestDocumentGenerationEndpoints:
    """Integration tests for document generation endpoints."""
    
    @pytest.fixture
    def client(self):
        """Test client for FastAPI app."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_auth(self):
        """Mock authentication for testing."""
        with patch('core.auth.get_current_user') as mock:
            mock.return_value = {"user_id": "test_user", "email": "user@company.local"}
            yield mock
    
    @pytest.fixture
    def mock_document_generator(self):
        """Mock document generator service."""
        with patch('api.endpoints.documents.get_document_generator') as mock:
            generator = AsyncMock()
            
            # Mock document generation response
            mock_response = Mock()
            mock_response.id = "doc_20240120_123456"
            mock_response.title = "Test Document"
            mock_response.document_type = "prd"
            mock_response.content = {
                "overview": "Test project overview",
                "requirements": ["Requirement 1", "Requirement 2"],
                "implementation": {
                    "approach": "Agile development",
                    "technologies": ["Python", "React"]
                }
            }
            mock_response.metadata = {"version": "1.0"}
            mock_response.wbs = Mock()
            mock_response.wbs.dict = Mock(return_value={
                "project_name": "Test Project",
                "total_estimated_hours": 240.0
            })
            mock_response.estimates = Mock()
            mock_response.estimates.dict = Mock(return_value={
                "total_cost": 150000,
                "team_size": 5
            })
            mock_response.exports = {ExportFormat.JSON: "/tmp/test_doc.json"}
            mock_response.created_at.isoformat = Mock(return_value="2024-01-20T12:34:56")
            mock_response.generation_time_ms = 2500
            
            generator.generate_document.return_value = mock_response
            
            # Mock templates response
            generator.get_document_templates.return_value = [
                {
                    "id": "prd_standard",
                    "name": "Standard PRD",
                    "type": "prd",
                    "description": "Standard Product Requirements Document",
                    "sections": ["overview", "requirements", "implementation"]
                }
            ]
            
            # Mock WBS analysis
            generator.wbs_generator = Mock()
            generator.wbs_generator.generate_wbs = AsyncMock()
            wbs_mock = Mock()
            wbs_mock.total_estimated_hours = 240.0
            wbs_mock.total_estimated_days = 30
            wbs_mock.phases = []
            wbs_mock.critical_path = ["task_1"]
            wbs_mock.risk_assessment = {"overall_risk_level": "medium"}
            generator.wbs_generator.generate_wbs.return_value = wbs_mock
            
            # Mock resource analysis
            generator.resource_estimator = Mock()
            generator.resource_estimator.estimate_resources = AsyncMock()
            estimate_mock = Mock()
            estimate_mock.team_composition = Mock(
                total_team_size=5,
                roles={"Developer": 3, "Manager": 1},
                skill_distribution={"senior": 3, "intermediate": 2},
                estimated_monthly_cost=40000,
                recommended_duration_months=3
            )
            estimate_mock.cost_estimate = Mock(
                total_cost=150000,
                human_resources=120000,
                infrastructure=20000,
                software_licenses=5000,
                contingency=5000,
                confidence_level=0.8
            )
            estimate_mock.timeline_estimate = Mock(
                total_duration_days=90,
                total_duration_months=3.0,
                critical_path_duration=75,
                buffer_days=15,
                risk_adjusted_duration=105
            )
            estimate_mock.assumptions = ["Standard team productivity"]
            estimate_mock.risks = [{"type": "timeline", "level": "medium"}]
            estimate_mock.recommendations = ["Regular progress reviews"]
            generator.resource_estimator.estimate_resources.return_value = estimate_mock
            
            mock.return_value = generator
            yield generator
    
    def test_generate_document_basic(self, client, mock_auth, mock_document_generator):
        """Test basic document generation endpoint."""
        request_data = {
            "title": "Test Document",
            "document_type": "prd",
            "context": "Test context",
            "sections": ["overview", "requirements"],
            "export_formats": ["json"],
            "include_wbs": True,
            "include_estimates": True
        }
        
        response = client.post("/api/v1/documents/generate", json=request_data)
        
        assert response.status_code == 201
        data = response.json()
        
        # Verify response structure
        assert data["id"] == "doc_20240120_123456"
        assert data["title"] == "Test Document"
        assert data["document_type"] == "prd"
        assert "content" in data
        assert "wbs" in data
        assert "estimates" in data
        assert data["generation_time_ms"] == 2500
        
        # Verify content structure
        assert "overview" in data["content"]
        assert "requirements" in data["content"]
        assert "implementation" in data["content"]
    
    def test_generate_document_minimal(self, client, mock_auth, mock_document_generator):
        """Test document generation with minimal parameters."""
        request_data = {
            "title": "Minimal Document",
            "include_wbs": False,
            "include_estimates": False
        }
        
        response = client.post("/api/v1/documents/generate", json=request_data)
        
        assert response.status_code == 201
        data = response.json()
        
        assert data["title"] == "Minimal Document"
        # Should still have content even with minimal request
        assert "content" in data
    
    def test_generate_document_invalid_type(self, client, mock_auth):
        """Test document generation with invalid document type."""
        request_data = {
            "title": "Invalid Document",
            "document_type": "invalid_type"
        }
        
        response = client.post("/api/v1/documents/generate", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_generate_document_missing_title(self, client, mock_auth):
        """Test document generation without required title."""
        request_data = {
            "document_type": "prd"
            # Missing title
        }
        
        response = client.post("/api/v1/documents/generate", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_generate_document_unauthorized(self, client):
        """Test document generation without authentication."""
        request_data = {
            "title": "Unauthorized Test",
            "document_type": "prd"
        }
        
        response = client.post("/api/v1/documents/generate", json=request_data)
        
        # Should be unauthorized (depends on your auth implementation)
        assert response.status_code in [401, 403]
    
    def test_get_document_templates(self, client, mock_auth, mock_document_generator):
        """Test getting available document templates."""
        response = client.get("/api/v1/documents/templates")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) > 0
        
        template = data[0]
        assert "id" in template
        assert "name" in template
        assert "type" in template
        assert "description" in template
        assert "sections" in template
        assert template["id"] == "prd_standard"
        assert template["name"] == "Standard PRD"
    
    def test_analyze_wbs_complexity(self, client, mock_auth, mock_document_generator):
        """Test WBS complexity analysis endpoint."""
        request_data = {
            "title": "Test Project",
            "requirements": ["Requirement 1", "Requirement 2"],
            "context": "Test project context",
            "project_type": "software"
        }
        
        response = client.post("/api/v1/documents/analyze/wbs", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify analysis structure
        assert "complexity_summary" in data
        assert "phase_breakdown" in data
        assert "critical_path" in data
        assert "risk_assessment" in data
        assert "recommendations" in data
        
        # Verify complexity summary
        summary = data["complexity_summary"]
        assert "total_hours" in summary
        assert "total_days" in summary
        assert "risk_level" in summary
    
    def test_analyze_resource_requirements(self, client, mock_auth, mock_document_generator):
        """Test resource requirements analysis endpoint."""
        request_data = {
            "content": {
                "overview": "Test project",
                "requirements": ["Req 1", "Req 2"],
                "implementation": {
                    "technologies": ["Python", "React"]
                }
            },
            "project_context": {
                "remote_team": True
            }
        }
        
        response = client.post("/api/v1/documents/analyze/resources", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify analysis structure
        assert "team_composition" in data
        assert "cost_breakdown" in data
        assert "timeline_projection" in data
        assert "assumptions" in data
        assert "risks" in data
        assert "recommendations" in data
        
        # Verify team composition
        team = data["team_composition"]
        assert "total_team_size" in team
        assert "roles" in team
        assert "monthly_cost" in team
        
        # Verify cost breakdown
        cost = data["cost_breakdown"]
        assert "total_cost" in cost
        assert "human_resources" in cost
        assert "confidence_level" in cost
    
    def test_export_document_not_implemented(self, client, mock_auth):
        """Test document export endpoint (not yet implemented)."""
        export_data = {
            "format": "pdf"
        }
        
        response = client.post("/api/v1/documents/export/test_doc", json=export_data)
        
        assert response.status_code == 501  # Not implemented
        data = response.json()
        assert "not yet implemented" in data["error"].lower()
    
    def test_download_document_not_implemented(self, client, mock_auth):
        """Test document download endpoint (not yet implemented)."""
        response = client.get("/api/v1/documents/download/test_doc/pdf")
        
        assert response.status_code == 501  # Not implemented
    
    def test_health_check(self, client, mock_document_generator):
        """Test document service health check."""
        response = client.get("/api/v1/documents/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["service"] == "document_generation"
        assert "components" in data
        
        components = data["components"]
        assert components["document_generator"] == "operational"
        assert components["wbs_generator"] == "operational"
        assert components["resource_estimator"] == "operational"
        assert components["export_service"] == "operational"
    
    def test_get_supported_formats(self, client):
        """Test getting supported export formats."""
        response = client.get("/api/v1/documents/config/formats")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify format information
        assert "pdf" in data
        assert "docx" in data
        assert "json" in data
        assert "html" in data
        assert "markdown" in data
        
        # Verify format details
        pdf_info = data["pdf"]
        assert pdf_info["name"] == "PDF"
        assert pdf_info["file_extension"] == "pdf"
        assert "mime_type" in pdf_info
    
    def test_get_complexity_levels(self, client):
        """Test getting complexity levels configuration."""
        response = client.get("/api/v1/documents/config/complexity-levels")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify complexity levels
        assert "simple" in data
        assert "moderate" in data
        assert "complex" in data
        assert "expert" in data
        
        # Verify level details
        simple_info = data["simple"]
        assert "name" in simple_info
        assert "description" in simple_info
        assert "typical_duration" in simple_info
        assert "skill_level_required" in simple_info
    
    def test_get_skill_levels(self, client):
        """Test getting skill levels configuration."""
        response = client.get("/api/v1/documents/config/skill-levels")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify skill levels
        assert "junior" in data
        assert "intermediate" in data
        assert "senior" in data
        assert "expert" in data
        
        # Verify level details
        senior_info = data["senior"]
        assert "name" in senior_info
        assert "description" in senior_info
        assert "typical_hourly_rate_usd" in senior_info
        assert "responsibilities" in senior_info
    
    def test_document_generation_with_multiple_exports(self, client, mock_auth, mock_document_generator):
        """Test document generation with multiple export formats."""
        request_data = {
            "title": "Multi-Export Document",
            "document_type": "technical_spec",
            "export_formats": ["json", "html", "markdown"],
            "sections": ["overview", "architecture", "implementation"]
        }
        
        response = client.post("/api/v1/documents/generate", json=request_data)
        
        assert response.status_code == 201
        data = response.json()
        
        assert data["title"] == "Multi-Export Document"
        assert data["document_type"] == "technical_spec"
        assert "exports" in data
    
    def test_document_generation_with_project_context(self, client, mock_auth, mock_document_generator):
        """Test document generation with project context."""
        request_data = {
            "title": "Context-Aware Document",
            "document_type": "project_plan",
            "context": "Agile project with remote team",
            "project_context": {
                "methodology": "agile",
                "team_location": "remote",
                "timeline_constraint": "aggressive",
                "budget_constraint": "limited"
            },
            "include_wbs": True,
            "include_estimates": True
        }
        
        response = client.post("/api/v1/documents/generate", json=request_data)
        
        assert response.status_code == 201
        data = response.json()
        
        assert data["document_type"] == "project_plan"
        assert data["wbs"] is not None
        assert data["estimates"] is not None


class TestDocumentGenerationErrorHandling:
    """Test error handling in document generation endpoints."""
    
    @pytest.fixture
    def client(self):
        """Test client for FastAPI app."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_auth(self):
        """Mock authentication."""
        with patch('core.auth.get_current_user') as mock:
            mock.return_value = {"user_id": "test_user"}
            yield mock
    
    def test_document_generation_service_error(self, client, mock_auth):
        """Test handling of document generation service errors."""
        with patch('api.endpoints.documents.get_document_generator') as mock:
            generator = AsyncMock()
            generator.generate_document.side_effect = Exception("Service error")
            mock.return_value = generator
            
            request_data = {
                "title": "Error Test Document",
                "document_type": "prd"
            }
            
            response = client.post("/api/v1/documents/generate", json=request_data)
            
            assert response.status_code == 500
            data = response.json()
            assert "failed" in data["error"].lower()
    
    def test_wbs_analysis_error(self, client, mock_auth):
        """Test handling of WBS analysis errors."""
        with patch('api.endpoints.documents.get_document_generator') as mock:
            generator = AsyncMock()
            generator.wbs_generator.generate_wbs.side_effect = Exception("WBS error")
            mock.return_value = generator
            
            request_data = {
                "title": "Error Test",
                "requirements": ["Req 1"]
            }
            
            response = client.post("/api/v1/documents/analyze/wbs", json=request_data)
            
            assert response.status_code == 500
    
    def test_resource_analysis_error(self, client, mock_auth):
        """Test handling of resource analysis errors."""
        with patch('api.endpoints.documents.get_document_generator') as mock:
            generator = AsyncMock()
            generator.resource_estimator.estimate_resources.side_effect = Exception("Resource error")
            mock.return_value = generator
            
            request_data = {
                "content": {"overview": "Test"}
            }
            
            response = client.post("/api/v1/documents/analyze/resources", json=request_data)
            
            assert response.status_code == 500
    
    def test_health_check_unhealthy(self, client):
        """Test health check when service is unhealthy."""
        with patch('api.endpoints.documents.get_document_generator') as mock:
            mock.side_effect = Exception("Initialization failed")
            
            response = client.get("/api/v1/documents/health")
            
            assert response.status_code == 200  # Health check always returns 200
            data = response.json()
            
            assert data["status"] == "unhealthy"
            assert "error" in data
    
    def test_invalid_json_request(self, client, mock_auth):
        """Test handling of invalid JSON in request."""
        response = client.post(
            "/api/v1/documents/generate",
            data="invalid json",
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_empty_request_body(self, client, mock_auth):
        """Test handling of empty request body."""
        response = client.post("/api/v1/documents/generate", json={})
        
        assert response.status_code == 422  # Validation error - missing title
    
    def test_oversized_context(self, client, mock_auth):
        """Test handling of oversized context field."""
        large_context = "x" * 3000  # Exceeds 2000 char limit
        
        request_data = {
            "title": "Large Context Test",
            "context": large_context
        }
        
        response = client.post("/api/v1/documents/generate", json=request_data)
        
        assert response.status_code == 422  # Validation error