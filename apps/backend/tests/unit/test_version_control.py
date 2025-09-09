"""
Unit tests for version control functionality.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from services.version_control_service import VersionControlService
from models.version_control import (
    VersionCreateRequest, 
    VersionRestoreRequest, 
    VersionComparisonRequest,
    DocumentVersion, 
    DocumentDiff,
    ChangeType
)


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    redis_mock = MagicMock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.setex.return_value = True
    redis_mock.llen.return_value = 0
    redis_mock.lrange.return_value = []
    redis_mock.rpush.return_value = 1
    return redis_mock


@pytest.fixture
def version_service(mock_redis):
    """Version control service instance for testing."""
    with patch('services.version_control_service.get_redis_client', return_value=mock_redis):
        service = VersionControlService(redis_client=mock_redis)
        return service


@pytest.fixture
def mobile_app_prd_content():
    """Mobile app PRD content for testing."""
    return {
        "title": "Mobile App Development PRD",
        "description": "Requirements for mobile application development project",
        "sections": {
            "introduction": "Cross-platform mobile application for customer engagement",
            "requirements": ["Native performance requirements", "Offline functionality support"],
            "implementation": {
                "approach": "React Native with TypeScript",
                "timeline": "12 weeks"
            }
        },
        "metadata": {
            "created_by": "product_manager",
            "tags": ["mobile", "react-native"]
        }
    }


@pytest.fixture
def mobile_app_version_request(mobile_app_prd_content):
    """Mobile app version creation request."""
    return VersionCreateRequest(
        document_id="mobile_app_doc_789",
        document_type="prd",
        content=mobile_app_prd_content,
        comment="Initial mobile app specification",
        user_id="product_manager"
    )


class TestVersionControlService:
    """Test cases for VersionControlService."""

    @pytest.mark.asyncio
    async def test_create_version_success(self, version_service, mobile_app_version_request, mock_redis):
        """Test successful version creation."""
        # Mock Redis responses
        mock_redis.get.return_value = None  # No existing current version
        mock_redis.llen.return_value = 0    # No existing versions
        
        # Create version
        version = await version_service.create_version(mobile_app_version_request)
        
        # Assertions
        assert version is not None
        assert version.document_id == mobile_app_version_request.document_id
        assert version.version_number == 1
        assert version.title == "Mobile App Development PRD"
        assert version.content == mobile_app_version_request.content
        assert version.created_by == mobile_app_version_request.user_id
        assert version.comment == mobile_app_version_request.comment
        assert version.changes_summary == {}  # First version has no changes
        
        # Verify Redis calls
        mock_redis.set.assert_called()  # Version stored
        mock_redis.rpush.assert_called()  # Added to version list
    
    @pytest.mark.asyncio
    async def test_create_version_incremental(self, version_service, mobile_app_version_request, mock_redis):
        """Test creating subsequent versions."""
        # Mock existing version
        mock_redis.llen.return_value = 1
        mock_redis.get.side_effect = lambda key: b'"test_version_1"' if "current" in key else None
        
        # Mock parent version
        parent_version = DocumentVersion(
            id="test_version_1",
            document_id=mobile_app_version_request.document_id,
            document_type="prd",
            version_number=1,
            title="Original Document",
            content={"title": "Original Document", "description": "Original description"},
            metadata={},
            created_by="original_user",
            created_at=datetime.utcnow(),
            changes_summary={},
            is_validated=True
        )
        
        with patch.object(version_service, 'get_version', return_value=parent_version):
            version = await version_service.create_version(mobile_app_version_request)
            
            # Assertions
            assert version.version_number == 2
            assert version.parent_version_id == "test_version_1"
            assert version.changes_summary["total_changes"] > 0
            assert "title" in version.changes_summary["fields_modified"]
    
    @pytest.mark.asyncio
    async def test_create_version_no_changes(self, version_service, mobile_app_version_request, mock_redis):
        """Test that identical content doesn't create a new version."""
        # Mock existing version with same content
        mock_redis.llen.return_value = 1
        mock_redis.get.side_effect = lambda key: b'"test_version_1"' if "current" in key else None
        
        parent_version = DocumentVersion(
            id="test_version_1",
            document_id=mobile_app_version_request.document_id,
            document_type="prd",
            version_number=1,
            title="Mobile App Development PRD",
            content=mobile_app_version_request.content,  # Same content
            metadata={"content_hash": version_service._calculate_content_hash(mobile_app_version_request.content)},
            created_by="test_user",
            created_at=datetime.utcnow(),
            changes_summary={},
            is_validated=True
        )
        
        with patch.object(version_service, 'get_version', return_value=parent_version):
            with patch.object(version_service, '_get_version_hash', return_value=version_service._calculate_content_hash(mobile_app_version_request.content)):
                version = await version_service.create_version(mobile_app_version_request)
                
                # Should return the existing version
                assert version.id == "test_version_1"
                assert version.version_number == 1
    
    @pytest.mark.asyncio
    async def test_restore_version_success(self, version_service, mock_redis):
        """Test successful version restoration."""
        document_id = "test_doc_123"
        version_id = "test_version_2"
        user_id = "restore_user"
        
        # Mock version to restore
        restore_version = DocumentVersion(
            id=version_id,
            document_id=document_id,
            document_type="prd",
            version_number=2,
            title="Version 2",
            content={"title": "Version 2", "description": "Content from version 2"},
            metadata={},
            created_by="original_user",
            created_at=datetime.utcnow(),
            changes_summary={},
            is_validated=True
        )
        
        mock_redis.llen.return_value = 3  # Existing versions count
        
        with patch.object(version_service, 'get_version', return_value=restore_version):
            with patch.object(version_service, 'create_version') as mock_create:
                mock_create.return_value = DocumentVersion(
                    id="new_version_id",
                    document_id=document_id,
                    document_type="prd",
                    version_number=4,
                    title="Version 2",
                    content=restore_version.content,
                    metadata={},
                    created_by=user_id,
                    created_at=datetime.utcnow(),
                    changes_summary={},
                    is_validated=True
                )
                
                request = VersionRestoreRequest(
                    document_id=document_id,
                    version_id=version_id,
                    comment="Restore to working version",
                    user_id=user_id
                )
                
                result = await version_service.restore_version(request)
                
                # Assertions
                assert result.version_number == 4
                assert result.content == restore_version.content
                assert result.created_by == user_id
                mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_diff_success(self, version_service):
        """Test successful diff generation."""
        document_id = "test_doc_123"
        
        # Mock versions
        old_version = DocumentVersion(
            id="version_1",
            document_id=document_id,
            document_type="prd",
            version_number=1,
            title="Old Version",
            content={
                "title": "Old Title",
                "description": "Old description",
                "status": "draft"
            },
            metadata={},
            created_by="user1",
            created_at=datetime.utcnow(),
            changes_summary={},
            is_validated=True
        )
        
        new_version = DocumentVersion(
            id="version_2",
            document_id=document_id,
            document_type="prd",
            version_number=2,
            title="New Version",
            content={
                "title": "New Title",
                "description": "Updated description",
                "status": "review",
                "new_field": "new value"
            },
            metadata={},
            created_by="user2",
            created_at=datetime.utcnow(),
            changes_summary={},
            is_validated=True
        )
        
        with patch.object(version_service, 'get_version') as mock_get:
            mock_get.side_effect = [old_version, new_version]
            
            request = VersionComparisonRequest(
                document_id=document_id,
                from_version_id="version_1",
                to_version_id="version_2",
                include_metadata=False
            )
            
            diff = await version_service.generate_diff(request)
            
            # Assertions
            assert diff.from_version_id == "version_1"
            assert diff.to_version_id == "version_2"
            assert diff.total_changes > 0
            
            # Check for expected changes
            modified_fields = [mod["field"] for mod in diff.modifications if "field" in mod]
            added_fields = [add["field"] for add in diff.additions if "field" in add]
            
            assert "title" in modified_fields or any("title" in str(mod) for mod in diff.modifications)
            assert "new_field" in added_fields or any("new_field" in str(add) for add in diff.additions)
    
    @pytest.mark.asyncio
    async def test_list_versions_success(self, version_service, mock_redis):
        """Test successful version listing."""
        document_id = "test_doc_123"
        
        # Mock version IDs in Redis
        mock_redis.llen.return_value = 3
        mock_redis.lrange.return_value = [b"version_3", b"version_2", b"version_1"]
        mock_redis.get.side_effect = lambda key: {
            "db:document:test_doc_123:current": b"version_3",
            "db:version:version_3": DocumentVersion(
                id="version_3", document_id=document_id, document_type="prd",
                version_number=3, title="Version 3", content={}, metadata={},
                created_by="user", created_at=datetime.utcnow(), changes_summary={},
                is_validated=True
            ).json().encode(),
            "db:version:version_2": DocumentVersion(
                id="version_2", document_id=document_id, document_type="prd",
                version_number=2, title="Version 2", content={}, metadata={},
                created_by="user", created_at=datetime.utcnow(), changes_summary={},
                is_validated=True
            ).json().encode(),
            "db:version:version_1": DocumentVersion(
                id="version_1", document_id=document_id, document_type="prd",
                version_number=1, title="Version 1", content={}, metadata={},
                created_by="user", created_at=datetime.utcnow(), changes_summary={},
                is_validated=True
            ).json().encode()
        }.get(key)
        
        response = await version_service.list_versions(document_id, page=1, page_size=10)
        
        # Assertions
        assert response.document_id == document_id
        assert len(response.versions) == 3
        assert response.total_count == 3
        assert response.current_version_id == "version_3"
        assert response.versions[0].version_number == 3  # Latest first
    
    def test_calculate_content_hash(self, version_service):
        """Test content hash calculation."""
        content1 = {"title": "Test", "description": "Content"}
        content2 = {"description": "Content", "title": "Test"}  # Same content, different order
        content3 = {"title": "Test", "description": "Different content"}
        
        hash1 = version_service._calculate_content_hash(content1)
        hash2 = version_service._calculate_content_hash(content2)
        hash3 = version_service._calculate_content_hash(content3)
        
        # Same content should have same hash regardless of key order
        assert hash1 == hash2
        # Different content should have different hash
        assert hash1 != hash3
        # Hash should be consistent
        assert len(hash1) == 64  # SHA256 hex digest length
    
    def test_generate_changes_summary(self, version_service):
        """Test changes summary generation."""
        old_content = {
            "title": "Old Title",
            "description": "Old description",
            "status": "draft",
            "old_field": "to be removed"
        }
        
        new_content = {
            "title": "New Title",  # Modified
            "description": "Old description",  # Unchanged
            "status": "review",  # Modified
            "new_field": "added field"  # Added
            # old_field removed
        }
        
        summary = version_service._generate_changes_summary(old_content, new_content)
        
        # Assertions
        assert "new_field" in summary["fields_added"]
        assert "old_field" in summary["fields_removed"]
        assert "title" in summary["fields_modified"]
        assert "status" in summary["fields_modified"]
        assert "description" not in summary["fields_modified"]  # Unchanged
        assert summary["total_changes"] == 4  # 1 added + 1 removed + 2 modified
    
    def test_calculate_diff_detailed(self, version_service):
        """Test detailed diff calculation."""
        old_content = {
            "title": "Original Title",
            "sections": ["intro", "body"],
            "metadata": {"version": 1}
        }
        
        new_content = {
            "title": "Updated Title",
            "sections": ["intro", "body", "conclusion"],
            "metadata": {"version": 2},
            "new_section": "additional content"
        }
        
        diff = version_service._calculate_diff(old_content, new_content, include_metadata=True)
        
        # Assertions
        assert diff["total_changes"] > 0
        assert len(diff["modifications"]) >= 1  # title, sections, metadata.version changed
        assert len(diff["additions"]) >= 1  # new_section added
        
        # Check specific changes
        field_names = [mod.get("field", "") for mod in diff["modifications"]]
        assert "title" in field_names
        
        added_fields = [add.get("field", "") for add in diff["additions"]]
        assert "new_section" in added_fields


class TestVersionControlModels:
    """Test cases for version control models."""
    
    def test_version_create_request_validation(self):
        """Test VersionCreateRequest validation."""
        # Valid request
        request = VersionCreateRequest(
            document_id="doc_123",
            document_type="prd",
            content={"title": "Test"},
            user_id="user_123"
        )
        assert request.document_id == "doc_123"
        assert request.comment is None
        
        # With comment
        request_with_comment = VersionCreateRequest(
            document_id="doc_123",
            document_type="prd",
            content={"title": "Test"},
            comment="Test version",
            user_id="user_123"
        )
        assert request_with_comment.comment == "Test version"
    
    def test_document_version_model(self):
        """Test DocumentVersion model."""
        now = datetime.utcnow()
        
        version = DocumentVersion(
            id="version_123",
            document_id="doc_123",
            document_type="prd",
            version_number=1,
            title="Test Document",
            content={"title": "Test"},
            metadata={"size": 100},
            created_by="user_123",
            created_at=now,
            comment="Initial version",
            changes_summary={"total_changes": 0},
            is_validated=True,
            validation_score=8.5
        )
        
        assert version.id == "version_123"
        assert version.version_number == 1
        assert version.is_validated is True
        assert version.validation_score == 8.5
        assert version.created_at == now
    
    def test_document_diff_model(self):
        """Test DocumentDiff model."""
        now = datetime.utcnow()
        
        diff = DocumentDiff(
            from_version_id="v1",
            to_version_id="v2",
            from_version_number=1,
            to_version_number=2,
            additions=[{"field": "new_field", "value": "new_value"}],
            deletions=[{"field": "old_field", "value": "old_value"}],
            modifications=[{"field": "title", "old_value": "old", "new_value": "new"}],
            total_changes=3,
            lines_added=5,
            lines_deleted=2,
            generated_at=now,
            generated_by="user_123"
        )
        
        assert diff.total_changes == 3
        assert len(diff.additions) == 1
        assert len(diff.deletions) == 1
        assert len(diff.modifications) == 1
        assert diff.lines_added == 5
        assert diff.lines_deleted == 2


@pytest.mark.asyncio
class TestVersionControlIntegration:
    """Integration tests for version control."""
    
    async def test_full_version_workflow(self, version_service, mobile_app_prd_content, mock_redis):
        """Test complete version control workflow."""
        document_id = "integration_test_doc"
        user_id = "integration_user"
        
        # Mock Redis for clean state
        mock_redis.llen.return_value = 0
        mock_redis.get.return_value = None
        
        # 1. Create initial version
        request1 = VersionCreateRequest(
            document_id=document_id,
            document_type="prd",
            content=mobile_app_prd_content,
            comment="Initial version",
            user_id=user_id
        )
        
        version1 = await version_service.create_version(request1)
        assert version1.version_number == 1
        
        # 2. Create second version with changes
        updated_content = mobile_app_prd_content.copy()
        updated_content["title"] = "Updated Test Document"
        updated_content["new_field"] = "new value"
        
        # Mock existing version for incremental creation
        mock_redis.llen.return_value = 1
        mock_redis.get.side_effect = lambda key: b'"version_1"' if "current" in key else None
        
        with patch.object(version_service, 'get_version', return_value=version1):
            request2 = VersionCreateRequest(
                document_id=document_id,
                document_type="prd",
                content=updated_content,
                comment="Added new field and updated title",
                user_id=user_id
            )
            
            version2 = await version_service.create_version(request2)
            assert version2.version_number == 2
            assert version2.parent_version_id == version1.id
        
        # 3. Compare versions
        with patch.object(version_service, 'get_version') as mock_get:
            mock_get.side_effect = [version1, version2]
            
            comparison_request = VersionComparisonRequest(
                document_id=document_id,
                from_version_id=version1.id,
                to_version_id=version2.id
            )
            
            diff = await version_service.generate_diff(comparison_request)
            assert diff.total_changes > 0
            assert diff.from_version_number == 1
            assert diff.to_version_number == 2
        
        # 4. Restore to previous version
        mock_redis.llen.return_value = 2
        
        with patch.object(version_service, 'get_version', return_value=version1):
            with patch.object(version_service, 'create_version') as mock_create:
                restored_version = DocumentVersion(
                    id="version_3",
                    document_id=document_id,
                    document_type="prd",
                    version_number=3,
                    title=version1.title,
                    content=version1.content,
                    metadata={},
                    created_by=user_id,
                    created_at=datetime.utcnow(),
                    changes_summary={},
                    is_validated=True
                )
                mock_create.return_value = restored_version
                
                restore_request = VersionRestoreRequest(
                    document_id=document_id,
                    version_id=version1.id,
                    comment="Restore to initial version",
                    user_id=user_id
                )
                
                result = await version_service.restore_version(restore_request)
                assert result.version_number == 3
                assert result.content == version1.content