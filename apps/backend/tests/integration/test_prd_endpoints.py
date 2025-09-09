"""
Integration tests for PRD endpoints.
"""
import uuid
from tests.utilities.test_data_factory import test_data_factory

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from main import app


class TestPRDEndpoints:
    """Integration tests for PRD API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def authenticated_client(self, client):
        """Create authenticated client with valid token."""
        # Register and login user
        user_data = {
            email=f"testuser{str(uuid.uuid4())[:8]}@{self.fake.domain_name()}",",
            "password": "SecureTestPassword123!",
            "name": "Test User"
        }
        client.post("/api/auth/register", json=user_data)
        
        login_response = client.post("/api/auth/login", json={
            "email": user_data["email"],
            "password": user_data["password"]
        })
        
        access_token = login_response.json()["data"]["access_token"]
        client.headers.update({"Authorization": f"Bearer {access_token}"})
        
        return client

    @pytest.fixture
    def sample_project_id(self):
        """Sample project ID for testing."""
        return "project-123"

    @pytest.fixture
    def sample_prd_request(self, sample_project_id):
        """Sample PRD generation request."""
        return {
            "title": "AI-Powered Task Management System",
            "description": "A comprehensive task management system using AI for prioritization and organization",
            "project_id": sample_project_id,
            "requirements": [
                "User authentication and authorization",
                "AI-powered task prioritization",
                "Real-time collaboration features",
                "Mobile app support"
            ],
            "constraints": [
                "GDPR compliance required",
                "Response time under 200ms",
                "Support 10,000 concurrent users"
            ],
            "target_audience": "Small to medium-sized businesses",
            "success_metrics": [
                "User engagement increased by 30%",
                "Task completion rate improved by 25%",
                "User satisfaction score > 4.5/5"
            ]
        }

    @pytest.fixture
    def mock_generated_prd(self):
        """Mock generated PRD response."""
        return {
            "id": "prd-456",
            "title": "AI-Powered Task Management System",
            "content": """
            # AI-Powered Task Management System

            ## Overview
            This document outlines requirements for an AI-powered task management system.

            ## Features
            - User authentication and role-based access control
            - AI-powered task prioritization algorithm
            - Real-time collaboration workspace
            - Cross-platform mobile applications
            """,
            "hallucination_rate": 0.015,
            "validation_score": 0.985,
            "graph_evidence": [
                {"node_id": "concept_1", "confidence": 0.95},
                {"node_id": "concept_2", "confidence": 0.88}
            ],
            "metadata": {
                "version": "1.0",
                "status": "draft",
                author=f"user{str(uuid.uuid4())[:8]}@{self.fake.domain_name()}",",
                "reviewers": [],
                "tags": ["AI", "task-management"],
                "estimated_effort": "3 months",
                "priority": "high"
            },
            "created_at": "2025-01-20T10:00:00Z",
            "updated_at": "2025-01-20T10:00:00Z"
        }

    def test_generate_prd_success(self, authenticated_client, sample_prd_request, mock_generated_prd):
        """Test successful PRD generation."""
        with patch('services.prd_pipeline.PRDPipeline.generate_prd', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = type('PRDResponse', (), mock_generated_prd)()
            
            response = authenticated_client.post(
                f"/api/projects/{sample_prd_request['project_id']}/prds/generate",
                json=sample_prd_request
            )
            
            assert response.status_code == 201
            data = response.json()
            
            assert data["success"] is True
            assert "data" in data
            assert data["data"]["id"] == mock_generated_prd["id"]
            assert data["data"]["title"] == sample_prd_request["title"]
            assert data["data"]["hallucination_rate"] == 0.015
            assert data["data"]["validation_score"] == 0.985

    def test_generate_prd_unauthenticated(self, client, sample_prd_request):
        """Test PRD generation without authentication."""
        response = client.post(
            f"/api/projects/{sample_prd_request['project_id']}/prds/generate",
            json=sample_prd_request
        )
        
        assert response.status_code == 401
        data = response.json()
        assert data["success"] is False

    def test_generate_prd_invalid_request(self, authenticated_client, sample_project_id):
        """Test PRD generation with invalid request data."""
        invalid_request = {
            "title": "",  # Empty title
            "description": "A" * 5000,  # Too long description
            "project_id": sample_project_id
            # Missing required fields
        }
        
        response = authenticated_client.post(
            f"/api/projects/{sample_project_id}/prds/generate",
            json=invalid_request
        )
        
        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False

    def test_generate_prd_high_hallucination(self, authenticated_client, sample_prd_request):
        """Test PRD generation with high hallucination rate."""
        with patch('services.prd_pipeline.PRDPipeline.generate_prd', new_callable=AsyncMock) as mock_generate:
            mock_generate.side_effect = ValueError("PRD hallucination rate (4.5%) exceeds threshold (2%)")
            
            response = authenticated_client.post(
                f"/api/projects/{sample_prd_request['project_id']}/prds/generate",
                json=sample_prd_request
            )
            
            assert response.status_code == 400
            data = response.json()
            assert data["success"] is False
            assert "hallucination rate" in data["message"].lower()

    def test_get_prds_list_success(self, authenticated_client, sample_project_id):
        """Test retrieving PRDs list for a project."""
        mock_prds = [
            {
                "id": "prd-1",
                "title": "First PRD",
                "status": "published",
                "created_at": "2025-01-20T10:00:00Z",
                "hallucination_rate": 0.01,
                "validation_score": 0.95
            },
            {
                "id": "prd-2", 
                "title": "Second PRD",
                "status": "draft",
                "created_at": "2025-01-20T11:00:00Z",
                "hallucination_rate": 0.02,
                "validation_score": 0.88
            }
        ]
        
        with patch('services.prd_pipeline.PRDPipeline.get_prds_by_project', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_prds
            
            response = authenticated_client.get(f"/api/projects/{sample_project_id}/prds")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert len(data["data"]) == 2
            assert data["data"][0]["id"] == "prd-1"
            assert data["data"][1]["id"] == "prd-2"

    def test_get_prds_list_with_filters(self, authenticated_client, sample_project_id):
        """Test retrieving PRDs list with status filter."""
        mock_published_prds = [
            {
                "id": "prd-1",
                "title": "Published PRD",
                "status": "published",
                "created_at": "2025-01-20T10:00:00Z"
            }
        ]
        
        with patch('services.prd_pipeline.PRDPipeline.get_prds_by_project', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_published_prds
            
            response = authenticated_client.get(
                f"/api/projects/{sample_project_id}/prds?status=published"
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert len(data["data"]) == 1
            assert data["data"][0]["status"] == "published"

    def test_get_prd_by_id_success(self, authenticated_client, sample_project_id, mock_generated_prd):
        """Test retrieving a specific PRD by ID."""
        prd_id = mock_generated_prd["id"]
        
        with patch('services.prd_pipeline.PRDPipeline.get_prd_by_id', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = type('PRDResponse', (), mock_generated_prd)()
            
            response = authenticated_client.get(
                f"/api/projects/{sample_project_id}/prds/{prd_id}"
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["data"]["id"] == prd_id
            assert data["data"]["title"] == mock_generated_prd["title"]

    def test_get_prd_by_id_not_found(self, authenticated_client, sample_project_id):
        """Test retrieving non-existent PRD."""
        non_existent_id = "prd-999"
        
        with patch('services.prd_pipeline.PRDPipeline.get_prd_by_id', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None
            
            response = authenticated_client.get(
                f"/api/projects/{sample_project_id}/prds/{non_existent_id}"
            )
            
            assert response.status_code == 404
            data = response.json()
            assert data["success"] is False

    def test_validate_prd_success(self, authenticated_client, sample_project_id):
        """Test PRD validation endpoint."""
        prd_id = "prd-123"
        
        mock_validation_result = {
            "content": "PRD content to validate",
            "hallucination_rate": 0.018,
            "validation_score": 0.982,
            "is_valid": True,
            "graph_evidence": [
                {"node_id": "concept_1", "confidence": 0.95}
            ],
            "issues": []
        }
        
        with patch('services.prd_pipeline.PRDPipeline.validate_prd_content', new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = type('ValidationResult', (), mock_validation_result)()
            
            response = authenticated_client.post(
                f"/api/projects/{sample_project_id}/prds/{prd_id}/validate"
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["data"]["hallucination_rate"] == 0.018
            assert data["data"]["validation_score"] == 0.982
            assert data["data"]["is_valid"] is True

    def test_validate_prd_high_hallucination(self, authenticated_client, sample_project_id):
        """Test PRD validation with high hallucination rate."""
        prd_id = "prd-123"
        
        mock_validation_result = {
            "content": "PRD content with hallucinations",
            "hallucination_rate": 0.055,  # 5.5%
            "validation_score": 0.72,
            "is_valid": False,
            "graph_evidence": [],
            "issues": [
                {"type": "hallucination", "severity": "error", "message": "Unsupported claims detected"},
                {"type": "missing_evidence", "severity": "warning", "message": "Lack of supporting evidence"}
            ]
        }
        
        with patch('services.prd_pipeline.PRDPipeline.validate_prd_content', new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = type('ValidationResult', (), mock_validation_result)()
            
            response = authenticated_client.post(
                f"/api/projects/{sample_project_id}/prds/{prd_id}/validate"
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["data"]["is_valid"] is False
            assert data["data"]["hallucination_rate"] > 0.05
            assert len(data["data"]["issues"]) > 0

    def test_update_prd_status_success(self, authenticated_client, sample_project_id):
        """Test updating PRD status."""
        prd_id = "prd-123"
        new_status = "published"
        
        with patch('services.prd_pipeline.PRDPipeline.update_prd_status', new_callable=AsyncMock) as mock_update:
            mock_update.return_value = True
            
            response = authenticated_client.patch(
                f"/api/projects/{sample_project_id}/prds/{prd_id}",
                json={"status": new_status}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_update_prd_status_invalid(self, authenticated_client, sample_project_id):
        """Test updating PRD with invalid status."""
        prd_id = "prd-123"
        invalid_status = "invalid_status"
        
        response = authenticated_client.patch(
            f"/api/projects/{sample_project_id}/prds/{prd_id}",
            json={"status": invalid_status}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False

    def test_delete_prd_success(self, authenticated_client, sample_project_id):
        """Test PRD deletion."""
        prd_id = "prd-123"
        
        with patch('services.prd_pipeline.PRDPipeline.delete_prd', new_callable=AsyncMock) as mock_delete:
            mock_delete.return_value = True
            
            response = authenticated_client.delete(
                f"/api/projects/{sample_project_id}/prds/{prd_id}"
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_delete_prd_not_found(self, authenticated_client, sample_project_id):
        """Test deleting non-existent PRD."""
        prd_id = "prd-999"
        
        with patch('services.prd_pipeline.PRDPipeline.delete_prd', new_callable=AsyncMock) as mock_delete:
            mock_delete.return_value = False
            
            response = authenticated_client.delete(
                f"/api/projects/{sample_project_id}/prds/{prd_id}"
            )
            
            assert response.status_code == 404
            data = response.json()
            assert data["success"] is False

    def test_export_prd_pdf(self, authenticated_client, sample_project_id):
        """Test exporting PRD to PDF format."""
        prd_id = "prd-123"
        
        mock_pdf_content = b"%PDF-1.4 mock pdf content"
        
        with patch('services.prd_pipeline.PRDPipeline.export_prd_to_format', new_callable=AsyncMock) as mock_export:
            mock_export.return_value = mock_pdf_content
            
            response = authenticated_client.get(
                f"/api/projects/{sample_project_id}/prds/{prd_id}/export?format=pdf"
            )
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/pdf"
            assert response.content == mock_pdf_content

    def test_export_prd_markdown(self, authenticated_client, sample_project_id):
        """Test exporting PRD to Markdown format."""
        prd_id = "prd-123"
        
        mock_markdown_content = b"# PRD Title\n\n## Overview\n\nThis is the PRD content."
        
        with patch('services.prd_pipeline.PRDPipeline.export_prd_to_format', new_callable=AsyncMock) as mock_export:
            mock_export.return_value = mock_markdown_content
            
            response = authenticated_client.get(
                f"/api/projects/{sample_project_id}/prds/{prd_id}/export?format=markdown"
            )
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/markdown"
            assert response.content == mock_markdown_content

    def test_export_prd_invalid_format(self, authenticated_client, sample_project_id):
        """Test exporting PRD with invalid format."""
        prd_id = "prd-123"
        
        response = authenticated_client.get(
            f"/api/projects/{sample_project_id}/prds/{prd_id}/export?format=invalid"
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "format" in data["message"].lower()

    def test_prd_analytics_success(self, authenticated_client, sample_project_id):
        """Test retrieving PRD analytics for a project."""
        mock_analytics = {
            "total_prds": 15,
            "average_hallucination_rate": 0.018,
            "average_validation_score": 0.924,
            "status_distribution": {
                "draft": 5,
                "in_review": 3,
                "published": 7
            },
            "quality_trends": {
                "last_30_days": [0.92, 0.94, 0.89, 0.95],
                "trend": "improving"
            },
            "generation_time_avg": 45.2,
            "most_common_topics": ["API", "authentication", "performance"]
        }
        
        with patch('services.prd_pipeline.PRDPipeline.get_project_analytics', new_callable=AsyncMock) as mock_analytics_fn:
            mock_analytics_fn.return_value = mock_analytics
            
            response = authenticated_client.get(
                f"/api/projects/{sample_project_id}/prds/analytics"
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["data"]["total_prds"] == 15
            assert data["data"]["average_hallucination_rate"] < 0.02
            assert "quality_trends" in data["data"]

    def test_bulk_prd_operations(self, authenticated_client, sample_project_id):
        """Test bulk operations on PRDs."""
        prd_ids = ["prd-1", "prd-2", "prd-3"]
        operation_data = {
            "prd_ids": prd_ids,
            "operation": "update_status",
            "parameters": {"status": "published"}
        }
        
        with patch('services.prd_pipeline.PRDPipeline.bulk_update_status', new_callable=AsyncMock) as mock_bulk:
            mock_bulk.return_value = {"updated": 3, "failed": 0}
            
            response = authenticated_client.post(
                f"/api/projects/{sample_project_id}/prds/bulk",
                json=operation_data
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["data"]["updated"] == 3
            assert data["data"]["failed"] == 0

    def test_prd_comparison(self, authenticated_client, sample_project_id):
        """Test comparing two PRDs."""
        prd_id_1 = "prd-123"
        prd_id_2 = "prd-456"
        
        mock_comparison = {
            "similarity_score": 0.73,
            "differences": [
                {"type": "content", "description": "Different target audience"},
                {"type": "requirements", "description": "Additional security requirement in PRD-456"}
            ],
            "common_topics": ["authentication", "performance"],
            "unique_to_first": ["mobile support"],
            "unique_to_second": ["security audit"]
        }
        
        with patch('services.prd_pipeline.PRDPipeline.compare_prds', new_callable=AsyncMock) as mock_compare:
            mock_compare.return_value = mock_comparison
            
            response = authenticated_client.get(
                f"/api/projects/{sample_project_id}/prds/{prd_id_1}/compare/{prd_id_2}"
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["data"]["similarity_score"] == 0.73
            assert len(data["data"]["differences"]) == 2

    def test_prd_generation_with_websocket_updates(self, authenticated_client, sample_prd_request):
        """Test PRD generation with WebSocket progress updates."""
        # This would require WebSocket testing setup
        # For now, we test that the endpoint accepts the request correctly
        
        with patch('services.prd_pipeline.PRDPipeline.generate_prd_with_progress', new_callable=AsyncMock) as mock_generate:
            mock_result = {
                "task_id": "task-789",
                "status": "started",
                "estimated_completion": "2025-01-20T10:05:00Z"
            }
            mock_generate.return_value = mock_result
            
            response = authenticated_client.post(
                f"/api/projects/{sample_prd_request['project_id']}/prds/generate-async",
                json=sample_prd_request
            )
            
            assert response.status_code == 202  # Accepted
            data = response.json()
            
            assert data["success"] is True
            assert data["data"]["task_id"] == "task-789"
            assert data["data"]["status"] == "started"

    @pytest.mark.parametrize("invalid_project_id", [
        "",
        "a" * 100,  # Too long
        "invalid-chars!@#",
        "../../../etc/passwd"  # Path traversal
    ])
    def test_invalid_project_ids(self, authenticated_client, invalid_project_id):
        """Test endpoints with invalid project IDs."""
        response = authenticated_client.get(f"/api/projects/{invalid_project_id}/prds")
        
        assert response.status_code in [400, 422, 404]
        data = response.json()
        assert data["success"] is False

    def test_rate_limiting(self, authenticated_client, sample_prd_request):
        """Test rate limiting on PRD generation endpoint."""
        # This would require actual rate limiting implementation
        # For now, test that multiple rapid requests are handled gracefully
        
        responses = []
        for _ in range(10):
            response = authenticated_client.post(
                f"/api/projects/{sample_prd_request['project_id']}/prds/generate",
                json=sample_prd_request
            )
            responses.append(response)
        
        # At least some requests should succeed
        success_count = sum(1 for r in responses if r.status_code == 201)
        assert success_count > 0

    def test_prd_versioning(self, authenticated_client, sample_project_id):
        """Test PRD versioning functionality."""
        prd_id = "prd-123"
        
        mock_versions = [
            {
                "version": "1.0",
                "created_at": "2025-01-20T10:00:00Z",
                author=f"user{str(uuid.uuid4())[:8]}@{self.fake.domain_name()}",",
                "changes": "Initial version"
            },
            {
                "version": "1.1",
                "created_at": "2025-01-20T11:00:00Z", 
                author=f"user{str(uuid.uuid4())[:8]}@{self.fake.domain_name()}",",
                "changes": "Added security requirements"
            }
        ]
        
        with patch('services.prd_pipeline.PRDPipeline.get_prd_versions', new_callable=AsyncMock) as mock_versions_fn:
            mock_versions_fn.return_value = mock_versions
            
            response = authenticated_client.get(
                f"/api/projects/{sample_project_id}/prds/{prd_id}/versions"
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert len(data["data"]) == 2
            assert data["data"][0]["version"] == "1.0"
            assert data["data"][1]["version"] == "1.1"