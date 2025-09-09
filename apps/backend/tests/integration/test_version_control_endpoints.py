"""
Integration tests for version control API endpoints.
"""

import pytest
from httpx import AsyncClient
from unittest.mock import patch, MagicMock
from datetime import datetime
import json

from main import app
from models.version_control import DocumentVersion


@pytest.fixture
async def client():
    """Test client fixture."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_auth_user():
    """Mock authenticated user."""
    return {
        "id": "auth_user_456",
        "sub": "auth_user_456", 
        "email": "alice.smith@company.com",
        "role": "user",
        "name": "Alice Smith"
    }


@pytest.fixture
def ecommerce_prd_content():
    """E-commerce PRD content for testing."""
    return {
        "title": "E-Commerce Platform PRD",
        "description": "Product requirements for e-commerce platform development",
        "sections": {
            "overview": "Building a scalable e-commerce platform for B2B customers",
            "requirements": [
                "Implement multi-tenant architecture for scalability",
                "Provide secure payment processing integration"
            ],
            "technical_specs": {
                "framework": "FastAPI",
                "database": "PostgreSQL",
                "auth": "JWT"
            }
        },
        "metadata": {
            "priority": "high",
            "estimated_hours": 240,
            "tags": ["e-commerce", "scalability"]
        }
    }


@pytest.fixture
def mock_version_service():
    """Mock version control service."""
    service = MagicMock()
    
    # Mock create_version
    service.create_version.return_value = DocumentVersion(
        id="version_ecom_001",
        document_id="doc_ecommerce_123",
        document_type="prd",
        version_number=1,
        title="E-Commerce Platform PRD",
        content={"title": "E-Commerce Platform PRD"},
        metadata={"content_hash": "sha256_hash_abc123"},
        created_by="auth_user_456",
        created_at=datetime.utcnow(),
        comment="Initial platform specification",
        changes_summary={"total_changes": 0},
        is_validated=True,
        validation_score=8.5
    )
    
    return service


class TestVersionControlEndpoints:
    """Test cases for version control API endpoints."""

    @pytest.mark.asyncio
    async def test_create_version_success(self, client, mock_auth_user, ecommerce_prd_content, mock_version_service):
        """Test successful version creation via API."""
        
        with patch('api.dependencies.auth.get_current_user', return_value=mock_auth_user):
            with patch('services.version_control_service.get_version_control_service', return_value=mock_version_service):
                
                request_data = {
                    "document_id": "doc_ecommerce_123",
                    "document_type": "prd",
                    "content": ecommerce_prd_content,
                    "comment": "Initial platform specification"
                }
                
                response = await client.post(
                    "/api/v1/versions/create",
                    json=request_data,
                    headers={"Authorization": "Bearer mock_token"}
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["id"] == "version_ecom_001"
                assert data["document_id"] == "doc_ecommerce_123"
                assert data["version_number"] == 1
                assert data["title"] == "E-Commerce Platform PRD"
                assert data["created_by"] == "auth_user_456"
                assert data["is_validated"] is True
                assert data["validation_score"] == 8.5

    @pytest.mark.asyncio
    async def test_create_version_unauthorized(self, client, ecommerce_prd_content):
        """Test version creation without authentication."""
        
        request_data = {
            "document_id": "test_doc_123",
            "document_type": "prd",
            "content": ecommerce_prd_content
        }
        
        response = await client.post("/api/v1/versions/create", json=request_data)
        
        assert response.status_code == 403  # Forbidden due to missing auth

    @pytest.mark.asyncio
    async def test_get_version_success(self, client, mock_auth_user, mock_version_service):
        """Test successful version retrieval."""
        
        version_id = "test_version_123"
        expected_version = DocumentVersion(
            id=version_id,
            document_id="test_doc_123",
            document_type="prd",
            version_number=2,
            title="Retrieved Version",
            content={"title": "Retrieved Version", "content": "test"},
            metadata={},
            created_by="test_user_123",
            created_at=datetime.utcnow(),
            changes_summary={"total_changes": 3},
            is_validated=True
        )
        
        mock_version_service.get_version.return_value = expected_version
        
        with patch('api.dependencies.auth.get_current_user', return_value=mock_auth_user):
            with patch('services.version_control_service.get_version_control_service', return_value=mock_version_service):
                
                response = await client.get(
                    f"/api/v1/versions/{version_id}",
                    headers={"Authorization": "Bearer mock_token"}
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["id"] == version_id
                assert data["version_number"] == 2
                assert data["title"] == "Retrieved Version"
                assert data["changes_summary"]["total_changes"] == 3

    @pytest.mark.asyncio
    async def test_get_version_not_found(self, client, mock_auth_user, mock_version_service):
        """Test version retrieval when version doesn't exist."""
        
        version_id = "nonexistent_version"
        mock_version_service.get_version.return_value = None
        
        with patch('api.dependencies.auth.get_current_user', return_value=mock_auth_user):
            with patch('services.version_control_service.get_version_control_service', return_value=mock_version_service):
                
                response = await client.get(
                    f"/api/v1/versions/{version_id}",
                    headers={"Authorization": "Bearer mock_token"}
                )
                
                assert response.status_code == 404
                data = response.json()
                assert "not found" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_list_document_versions(self, client, mock_auth_user, mock_version_service):
        """Test listing document versions."""
        
        document_id = "test_doc_123"
        
        from models.version_control import VersionListResponse
        mock_response = VersionListResponse(
            document_id=document_id,
            versions=[
                DocumentVersion(
                    id="version_3",
                    document_id=document_id,
                    document_type="prd",
                    version_number=3,
                    title="Latest Version",
                    content={"title": "Latest"},
                    metadata={},
                    created_by="user1",
                    created_at=datetime.utcnow(),
                    changes_summary={"total_changes": 5},
                    is_validated=True
                ),
                DocumentVersion(
                    id="version_2",
                    document_id=document_id,
                    document_type="prd", 
                    version_number=2,
                    title="Previous Version",
                    content={"title": "Previous"},
                    metadata={},
                    created_by="user2",
                    created_at=datetime.utcnow(),
                    changes_summary={"total_changes": 2},
                    is_validated=True
                )
            ],
            total_count=3,
            current_version_id="version_3",
            page=1,
            page_size=20
        )
        
        mock_version_service.list_versions.return_value = mock_response
        
        with patch('api.dependencies.auth.get_current_user', return_value=mock_auth_user):
            with patch('services.version_control_service.get_version_control_service', return_value=mock_version_service):
                
                response = await client.get(
                    f"/api/v1/versions/document/{document_id}?page=1&page_size=20",
                    headers={"Authorization": "Bearer mock_token"}
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["document_id"] == document_id
                assert len(data["versions"]) == 2
                assert data["total_count"] == 3
                assert data["current_version_id"] == "version_3"
                assert data["versions"][0]["version_number"] == 3

    @pytest.mark.asyncio
    async def test_restore_version_success(self, client, mock_auth_user, mock_version_service):
        """Test successful version restoration."""
        
        restored_version = DocumentVersion(
            id="new_version_id",
            document_id="test_doc_123",
            document_type="prd",
            version_number=4,
            title="Restored Version",
            content={"title": "Restored content"},
            metadata={},
            created_by="test_user_123",
            created_at=datetime.utcnow(),
            comment="Restored from version 2",
            changes_summary={"restored_from": "version_2"},
            is_validated=True
        )
        
        mock_version_service.restore_version.return_value = restored_version
        
        with patch('api.dependencies.auth.get_current_user', return_value=mock_auth_user):
            with patch('services.version_control_service.get_version_control_service', return_value=mock_version_service):
                
                request_data = {
                    "document_id": "test_doc_123",
                    "version_id": "version_2",
                    "comment": "Restore to stable version"
                }
                
                response = await client.post(
                    "/api/v1/versions/restore",
                    json=request_data,
                    headers={"Authorization": "Bearer mock_token"}
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["id"] == "new_version_id"
                assert data["version_number"] == 4
                assert data["comment"] == "Restored from version 2"
                assert data["created_by"] == "test_user_123"

    @pytest.mark.asyncio
    async def test_compare_versions_success(self, client, mock_auth_user, mock_version_service):
        """Test successful version comparison."""
        
        from models.version_control import DocumentDiff
        mock_diff = DocumentDiff(
            from_version_id="version_1",
            to_version_id="version_2", 
            from_version_number=1,
            to_version_number=2,
            additions=[
                {"field": "new_section", "value": "New content added"}
            ],
            deletions=[
                {"field": "old_section", "value": "Removed content"}
            ],
            modifications=[
                {
                    "field": "title",
                    "old_value": "Old Title",
                    "new_value": "New Title"
                }
            ],
            total_changes=3,
            lines_added=10,
            lines_deleted=5,
            generated_at=datetime.utcnow(),
            generated_by="test_user_123"
        )
        
        mock_version_service.generate_diff.return_value = mock_diff
        
        with patch('api.dependencies.auth.get_current_user', return_value=mock_auth_user):
            with patch('services.version_control_service.get_version_control_service', return_value=mock_version_service):
                
                request_data = {
                    "document_id": "test_doc_123",
                    "from_version_id": "version_1",
                    "to_version_id": "version_2",
                    "include_metadata": False
                }
                
                response = await client.post(
                    "/api/v1/versions/compare",
                    json=request_data,
                    headers={"Authorization": "Bearer mock_token"}
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["from_version_id"] == "version_1"
                assert data["to_version_id"] == "version_2"
                assert data["total_changes"] == 3
                assert data["lines_added"] == 10
                assert data["lines_deleted"] == 5
                assert len(data["additions"]) == 1
                assert len(data["deletions"]) == 1
                assert len(data["modifications"]) == 1

    @pytest.mark.asyncio
    async def test_get_change_history_success(self, client, mock_auth_user, mock_version_service):
        """Test successful change history retrieval."""
        
        from models.version_control import ChangeHistoryResponse, ChangeHistoryEntry, ChangeType
        
        mock_history = ChangeHistoryResponse(
            document_id="test_doc_123",
            changes=[
                ChangeHistoryEntry(
                    id="change_1",
                    document_id="test_doc_123",
                    version_id="version_2",
                    change_type=ChangeType.UPDATE,
                    changed_by="user1",
                    changed_at=datetime.utcnow(),
                    comment="Updated document content"
                ),
                ChangeHistoryEntry(
                    id="change_2", 
                    document_id="test_doc_123",
                    version_id="version_1",
                    change_type=ChangeType.CREATE,
                    changed_by="user2",
                    changed_at=datetime.utcnow(),
                    comment="Initial document creation"
                )
            ],
            total_count=2,
            page=1,
            page_size=50
        )
        
        mock_version_service.get_change_history.return_value = mock_history
        
        with patch('api.dependencies.auth.get_current_user', return_value=mock_auth_user):
            with patch('services.version_control_service.get_version_control_service', return_value=mock_version_service):
                
                response = await client.get(
                    "/api/v1/versions/history/test_doc_123?page=1&page_size=50",
                    headers={"Authorization": "Bearer mock_token"}
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["document_id"] == "test_doc_123"
                assert len(data["changes"]) == 2
                assert data["total_count"] == 2
                assert data["changes"][0]["change_type"] == "update"
                assert data["changes"][1]["change_type"] == "create"

    @pytest.mark.asyncio 
    async def test_delete_version_admin_only(self, client, mock_auth_user, mock_version_service):
        """Test version deletion requires admin role."""
        
        # Test with regular user
        with patch('api.dependencies.auth.get_current_user', return_value=mock_auth_user):
            response = await client.delete(
                "/api/v1/versions/version_123",
                headers={"Authorization": "Bearer mock_token"}
            )
            
            assert response.status_code == 403
            data = response.json()
            assert "administrator" in data["error"].lower()
        
        # Test with admin user
        admin_user = mock_auth_user.copy()
        admin_user["role"] = "admin"
        
        with patch('api.dependencies.auth.get_current_user', return_value=admin_user):
            response = await client.delete(
                "/api/v1/versions/version_123",
                headers={"Authorization": "Bearer mock_token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Version version_123 deleted successfully"

    @pytest.mark.asyncio
    async def test_get_latest_version_success(self, client, mock_auth_user, mock_version_service):
        """Test getting latest version of a document."""
        
        from models.version_control import VersionListResponse
        mock_response = VersionListResponse(
            document_id="test_doc_123",
            versions=[
                DocumentVersion(
                    id="latest_version",
                    document_id="test_doc_123",
                    document_type="prd",
                    version_number=5,
                    title="Latest Version",
                    content={"title": "Latest", "updated": True},
                    metadata={},
                    created_by="user1",
                    created_at=datetime.utcnow(),
                    changes_summary={"total_changes": 2},
                    is_validated=True,
                    validation_score=9.2
                )
            ],
            total_count=5,
            current_version_id="latest_version",
            page=1,
            page_size=1
        )
        
        mock_version_service.list_versions.return_value = mock_response
        
        with patch('api.dependencies.auth.get_current_user', return_value=mock_auth_user):
            with patch('services.version_control_service.get_version_control_service', return_value=mock_version_service):
                
                response = await client.get(
                    "/api/v1/versions/latest/test_doc_123",
                    headers={"Authorization": "Bearer mock_token"}
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["id"] == "latest_version"
                assert data["version_number"] == 5
                assert data["title"] == "Latest Version"
                assert data["validation_score"] == 9.2

    @pytest.mark.asyncio
    async def test_get_latest_version_no_versions(self, client, mock_auth_user, mock_version_service):
        """Test getting latest version when no versions exist."""
        
        from models.version_control import VersionListResponse
        mock_response = VersionListResponse(
            document_id="test_doc_123",
            versions=[],
            total_count=0,
            current_version_id="",
            page=1,
            page_size=1
        )
        
        mock_version_service.list_versions.return_value = mock_response
        
        with patch('api.dependencies.auth.get_current_user', return_value=mock_auth_user):
            with patch('services.version_control_service.get_version_control_service', return_value=mock_version_service):
                
                response = await client.get(
                    "/api/v1/versions/latest/test_doc_123",
                    headers={"Authorization": "Bearer mock_token"}
                )
                
                assert response.status_code == 404
                data = response.json()
                assert "no versions found" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_api_error_handling(self, client, mock_auth_user, mock_version_service):
        """Test API error handling for version control endpoints."""
        
        # Test service error handling
        mock_version_service.create_version.side_effect = Exception("Database connection failed")
        
        with patch('api.dependencies.auth.get_current_user', return_value=mock_auth_user):
            with patch('services.version_control_service.get_version_control_service', return_value=mock_version_service):
                
                request_data = {
                    "document_id": "test_doc_123",
                    "document_type": "prd",
                    "content": {"title": "Test"},
                    "comment": "Test version"
                }
                
                response = await client.post(
                    "/api/v1/versions/create",
                    json=request_data,
                    headers={"Authorization": "Bearer mock_token"}
                )
                
                assert response.status_code == 500
                data = response.json()
                assert "failed to create version" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_request_validation(self, client, mock_auth_user):
        """Test request validation for version control endpoints."""
        
        with patch('api.dependencies.auth.get_current_user', return_value=mock_auth_user):
            # Test missing required fields
            response = await client.post(
                "/api/v1/versions/create",
                json={"document_id": "test_doc"},  # Missing required fields
                headers={"Authorization": "Bearer mock_token"}
            )
            
            assert response.status_code == 422  # Validation error
            
            # Test invalid data types
            response = await client.post(
                "/api/v1/versions/create",
                json={
                    "document_id": 123,  # Should be string
                    "document_type": "prd",
                    "content": "not_a_dict"  # Should be dict
                },
                headers={"Authorization": "Bearer mock_token"}
            )
            
            assert response.status_code == 422


class TestVersionControlPerformance:
    """Performance-related tests for version control."""

    @pytest.mark.asyncio
    async def test_large_content_handling(self, client, mock_auth_user, mock_version_service):
        """Test handling of large document content."""
        
        # Create large content (simulating a complex PRD)
        large_content = {
            "title": "Enterprise Platform Architecture PRD",
            "sections": {}
        }
        
        # Add many sections with substantial content
        for i in range(100):
            large_content["sections"][f"section_{i}"] = {
                "title": f"Platform Component {i}",
                "content": "Detailed architectural specifications and design patterns for enterprise-scale deployment. " * 50,
                "requirements": [f"Scalability requirement {j}" for j in range(10)],
                "technical_details": {
                    "implementation": f"Microservice implementation strategy for component {i}",
                    "dependencies": [f"service_dependency_{j}" for j in range(5)],
                    "testing_strategy": "Automated testing with integration coverage"
                }
            }
        
        # Mock successful handling
        mock_version_service.create_version.return_value = DocumentVersion(
            id="enterprise_arch_v001",
            document_id="enterprise_platform_456", 
            document_type="prd",
            version_number=1,
            title="Enterprise Platform Architecture PRD",
            content=large_content,
            metadata={"size_bytes": len(str(large_content))},
            created_by="auth_user_456",
            created_at=datetime.utcnow(),
            changes_summary={"total_changes": 0},
            is_validated=True
        )
        
        with patch('api.dependencies.auth.get_current_user', return_value=mock_auth_user):
            with patch('services.version_control_service.get_version_control_service', return_value=mock_version_service):
                
                request_data = {
                    "document_id": "enterprise_platform_456",
                    "document_type": "prd", 
                    "content": large_content,
                    "comment": "Enterprise architecture specification"
                }
                
                response = await client.post(
                    "/api/v1/versions/create",
                    json=request_data,
                    headers={"Authorization": "Bearer mock_token"}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["title"] == "Enterprise Platform Architecture PRD"
                assert len(str(data["content"])) > 10000  # Verify large content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])