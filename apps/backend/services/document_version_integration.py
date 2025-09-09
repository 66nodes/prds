"""
Document versioning integration service.

Provides automatic versioning for document operations across different document types.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from ..core.logging_config import get_logger
from ..models.version_control import VersionCreateRequest
from ..services.version_control_service import get_version_control_service

logger = get_logger(__name__)


class DocumentVersioningService:
    """Service for automatic document versioning integration."""
    
    def __init__(self):
        """Initialize the versioning service."""
        self.version_service = get_version_control_service()
        
    async def create_document_version(
        self,
        document_id: str,
        document_type: str,
        content: Dict[str, Any],
        user_id: str,
        comment: Optional[str] = None,
        auto_validate: bool = True
    ) -> str:
        """
        Create a new version of a document.
        
        Args:
            document_id: Unique document identifier
            document_type: Type of document (prd, specification, etc.)
            content: Document content
            user_id: User making the change
            comment: Optional comment describing the change
            auto_validate: Whether to auto-validate the content
            
        Returns:
            Version ID of the created version
        """
        try:
            request = VersionCreateRequest(
                document_id=document_id,
                document_type=document_type,
                content=content,
                comment=comment,
                user_id=user_id
            )
            
            version = await self.version_service.create_version(
                request, 
                auto_validate=auto_validate
            )
            
            logger.info(
                f"Created version {version.id} for document {document_id}",
                extra={
                    "document_id": document_id,
                    "document_type": document_type,
                    "version_id": version.id,
                    "version_number": version.version_number,
                    "user_id": user_id
                }
            )
            
            return version.id
            
        except Exception as e:
            logger.error(
                f"Failed to create version for document {document_id}: {e}",
                extra={
                    "document_id": document_id,
                    "document_type": document_type,
                    "user_id": user_id,
                    "error": str(e)
                }
            )
            raise
    
    async def version_on_create(
        self,
        document_id: str,
        document_type: str,
        content: Dict[str, Any],
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create initial version when a document is created.
        
        Args:
            document_id: Document identifier
            document_type: Type of document
            content: Initial document content
            user_id: User creating the document
            metadata: Additional metadata
            
        Returns:
            Version ID of the initial version
        """
        return await self.create_document_version(
            document_id=document_id,
            document_type=document_type,
            content=content,
            user_id=user_id,
            comment="Initial version",
            auto_validate=True
        )
    
    async def version_on_update(
        self,
        document_id: str,
        document_type: str,
        content: Dict[str, Any],
        user_id: str,
        change_comment: Optional[str] = None
    ) -> str:
        """
        Create version when a document is updated.
        
        Args:
            document_id: Document identifier
            document_type: Type of document
            content: Updated document content
            user_id: User making the update
            change_comment: Description of changes made
            
        Returns:
            Version ID of the new version
        """
        return await self.create_document_version(
            document_id=document_id,
            document_type=document_type,
            content=content,
            user_id=user_id,
            comment=change_comment or "Document updated",
            auto_validate=True
        )
    
    async def version_on_phase_completion(
        self,
        document_id: str,
        document_type: str,
        content: Dict[str, Any],
        user_id: str,
        phase: str,
        phase_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create version when a document phase is completed.
        
        Args:
            document_id: Document identifier
            document_type: Type of document
            content: Document content after phase completion
            user_id: User completing the phase
            phase: Phase name (e.g., "phase_1", "review", "approval")
            phase_data: Additional phase-specific data
            
        Returns:
            Version ID of the new version
        """
        comment = f"Completed {phase}"
        if phase_data and phase_data.get("score"):
            comment += f" (Score: {phase_data['score']})"
            
        return await self.create_document_version(
            document_id=document_id,
            document_type=document_type,
            content=content,
            user_id=user_id,
            comment=comment,
            auto_validate=True
        )
    
    async def get_current_version(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current version of a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Current version data or None if no versions exist
        """
        try:
            versions = await self.version_service.list_versions(document_id, page=1, page_size=1)
            if versions.versions:
                return versions.versions[0].content
            return None
        except Exception as e:
            logger.error(f"Failed to get current version for document {document_id}: {e}")
            return None
    
    async def restore_document_version(
        self,
        document_id: str,
        version_id: str,
        user_id: str,
        restore_comment: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Restore a document to a previous version.
        
        Args:
            document_id: Document identifier
            version_id: Version to restore
            user_id: User performing the restore
            restore_comment: Comment for the restore operation
            
        Returns:
            Restored document content
        """
        try:
            from ..models.version_control import VersionRestoreRequest
            
            request = VersionRestoreRequest(
                document_id=document_id,
                version_id=version_id,
                comment=restore_comment,
                user_id=user_id
            )
            
            restored_version = await self.version_service.restore_version(request)
            
            logger.info(
                f"Restored document {document_id} to version {version_id}",
                extra={
                    "document_id": document_id,
                    "restored_from_version": version_id,
                    "new_version_id": restored_version.id,
                    "user_id": user_id
                }
            )
            
            return restored_version.content
            
        except Exception as e:
            logger.error(f"Failed to restore document {document_id}: {e}")
            raise
    
    async def batch_version_documents(
        self,
        document_updates: List[Dict[str, Any]],
        user_id: str,
        batch_comment: Optional[str] = None
    ) -> List[str]:
        """
        Create versions for multiple documents in a batch operation.
        
        Args:
            document_updates: List of document update data
            user_id: User making the batch update
            batch_comment: Comment for the batch operation
            
        Returns:
            List of version IDs created
        """
        version_ids = []
        batch_id = str(uuid4())[:8]
        
        for doc_update in document_updates:
            try:
                comment = batch_comment or f"Batch update ({batch_id})"
                version_id = await self.create_document_version(
                    document_id=doc_update["document_id"],
                    document_type=doc_update["document_type"],
                    content=doc_update["content"],
                    user_id=user_id,
                    comment=comment,
                    auto_validate=doc_update.get("auto_validate", True)
                )
                version_ids.append(version_id)
            except Exception as e:
                logger.error(f"Failed to version document in batch: {e}")
                # Continue with other documents even if one fails
                continue
        
        logger.info(
            f"Batch versioned {len(version_ids)} documents",
            extra={
                "batch_id": batch_id,
                "user_id": user_id,
                "total_documents": len(document_updates),
                "successful_versions": len(version_ids)
            }
        )
        
        return version_ids


# Decorators for automatic versioning

def version_on_create(document_type: str = "document"):
    """
    Decorator to automatically create a version when a document is created.
    
    Args:
        document_type: Type of document being created
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Extract necessary data from result or kwargs
            document_id = result.get("id") or kwargs.get("document_id")
            content = result.get("content") or result
            user_id = kwargs.get("user_id") or "system"
            
            if document_id and content:
                try:
                    versioning_service = DocumentVersioningService()
                    await versioning_service.version_on_create(
                        document_id=str(document_id),
                        document_type=document_type,
                        content=content if isinstance(content, dict) else {"data": content},
                        user_id=user_id
                    )
                except Exception as e:
                    logger.warning(f"Failed to create version for new document: {e}")
            
            return result
        return wrapper
    return decorator


def version_on_update(document_type: str = "document"):
    """
    Decorator to automatically create a version when a document is updated.
    
    Args:
        document_type: Type of document being updated
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Extract necessary data from result or kwargs
            document_id = kwargs.get("document_id") or result.get("id")
            content = result.get("content") or result
            user_id = kwargs.get("user_id") or "system"
            comment = kwargs.get("comment") or kwargs.get("change_comment")
            
            if document_id and content:
                try:
                    versioning_service = DocumentVersioningService()
                    await versioning_service.version_on_update(
                        document_id=str(document_id),
                        document_type=document_type,
                        content=content if isinstance(content, dict) else {"data": content},
                        user_id=user_id,
                        change_comment=comment
                    )
                except Exception as e:
                    logger.warning(f"Failed to create version for updated document: {e}")
            
            return result
        return wrapper
    return decorator


# Singleton instance
_versioning_service: Optional[DocumentVersioningService] = None


def get_document_versioning_service() -> DocumentVersioningService:
    """Get or create the document versioning service singleton."""
    global _versioning_service
    if _versioning_service is None:
        _versioning_service = DocumentVersioningService()
    return _versioning_service