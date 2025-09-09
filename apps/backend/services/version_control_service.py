"""
Version control service for document versioning and change tracking.
"""

import difflib
import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from redis import Redis

from ..core.database import get_db
from ..core.logging_config import get_logger
from ..core.redis import get_redis_client
from ..models.version_control import (
    AuditLog,
    ChangeHistoryEntry,
    ChangeHistoryResponse,
    ChangeType,
    DocumentDiff,
    DocumentVersion,
    VersionComparisonRequest,
    VersionCreateRequest,
    VersionListResponse,
    VersionRestoreRequest,
)

logger = get_logger(__name__)


class VersionControlService:
    """Service for managing document versions and change history."""
    
    def __init__(self, redis_client: Optional[Redis] = None):
        """Initialize version control service."""
        self.redis = redis_client or get_redis_client()
        self.version_cache_ttl = 3600  # 1 hour cache
        
    async def create_version(
        self,
        request: VersionCreateRequest,
        auto_validate: bool = True
    ) -> DocumentVersion:
        """
        Create a new version of a document.
        
        Args:
            request: Version creation request
            auto_validate: Whether to validate the content automatically
            
        Returns:
            Created document version
        """
        try:
            # Generate version ID
            version_id = f"ver_{uuid4().hex[:12]}"
            
            # Get current version number
            current_version_num = await self._get_latest_version_number(request.document_id)
            new_version_num = current_version_num + 1
            
            # Get parent version ID
            parent_version_id = await self._get_latest_version_id(request.document_id)
            
            # Calculate content hash for deduplication
            content_hash = self._calculate_content_hash(request.content)
            
            # Check if content has actually changed
            if parent_version_id:
                parent_hash = await self._get_version_hash(parent_version_id)
                if parent_hash == content_hash:
                    logger.info(f"No changes detected for document {request.document_id}")
                    return await self.get_version(parent_version_id)
            
            # Extract changes summary
            changes_summary = {}
            if parent_version_id:
                parent_version = await self.get_version(parent_version_id)
                if parent_version:
                    changes_summary = self._generate_changes_summary(
                        parent_version.content,
                        request.content
                    )
            
            # Create version object
            version = DocumentVersion(
                id=version_id,
                document_id=request.document_id,
                document_type=request.document_type,
                version_number=new_version_num,
                title=request.content.get("title", "Untitled"),
                content=request.content,
                metadata={
                    "content_hash": content_hash,
                    "size_bytes": len(json.dumps(request.content)),
                },
                created_by=request.user_id,
                created_at=datetime.utcnow(),
                comment=request.comment,
                changes_summary=changes_summary,
                parent_version_id=parent_version_id,
                is_validated=False,
            )
            
            # Store version in database
            await self._store_version(version)
            
            # Create change history entry
            await self._create_change_history(
                document_id=request.document_id,
                version_id=version_id,
                change_type=ChangeType.CREATE if new_version_num == 1 else ChangeType.UPDATE,
                user_id=request.user_id,
                comment=request.comment,
                changes=changes_summary
            )
            
            # Auto-validate if enabled
            if auto_validate:
                version.is_validated = True
                version.validation_score = await self._validate_content(request.content)
            
            # Cache version
            await self._cache_version(version)
            
            # Create audit log
            await self._create_audit_log(
                entity_type="document",
                entity_id=request.document_id,
                action="version_created",
                user_id=request.user_id,
                details={
                    "version_id": version_id,
                    "version_number": new_version_num,
                    "comment": request.comment
                }
            )
            
            logger.info(f"Created version {version_id} for document {request.document_id}")
            return version
            
        except Exception as e:
            logger.error(f"Error creating version: {e}")
            raise
    
    async def get_version(self, version_id: str) -> Optional[DocumentVersion]:
        """Get a specific version by ID."""
        try:
            # Check cache first
            cached = await self._get_cached_version(version_id)
            if cached:
                return cached
            
            # Fetch from database
            version = await self._fetch_version(version_id)
            
            if version:
                # Cache for future use
                await self._cache_version(version)
            
            return version
            
        except Exception as e:
            logger.error(f"Error fetching version {version_id}: {e}")
            return None
    
    async def list_versions(
        self,
        document_id: str,
        page: int = 1,
        page_size: int = 20
    ) -> VersionListResponse:
        """
        List all versions of a document.
        
        Args:
            document_id: Document ID
            page: Page number
            page_size: Items per page
            
        Returns:
            Version list response
        """
        try:
            # Fetch versions from database
            versions = await self._fetch_document_versions(
                document_id, 
                offset=(page - 1) * page_size,
                limit=page_size
            )
            
            # Get total count
            total_count = await self._count_document_versions(document_id)
            
            # Get current version ID
            current_version_id = await self._get_latest_version_id(document_id)
            
            return VersionListResponse(
                document_id=document_id,
                versions=versions,
                total_count=total_count,
                current_version_id=current_version_id or "",
                page=page,
                page_size=page_size
            )
            
        except Exception as e:
            logger.error(f"Error listing versions for document {document_id}: {e}")
            raise
    
    async def restore_version(
        self,
        request: VersionRestoreRequest
    ) -> DocumentVersion:
        """
        Restore a previous version of a document.
        
        Args:
            request: Version restore request
            
        Returns:
            New version created from restoration
        """
        try:
            # Fetch the version to restore
            version_to_restore = await self.get_version(request.version_id)
            if not version_to_restore:
                raise ValueError(f"Version {request.version_id} not found")
            
            # Create a new version with the restored content
            restore_comment = request.comment or f"Restored from version {version_to_restore.version_number}"
            
            new_version_request = VersionCreateRequest(
                document_id=request.document_id,
                document_type=version_to_restore.document_type,
                content=version_to_restore.content,
                comment=restore_comment,
                user_id=request.user_id
            )
            
            # Create the new version
            new_version = await self.create_version(new_version_request)
            
            # Create change history entry for restoration
            await self._create_change_history(
                document_id=request.document_id,
                version_id=new_version.id,
                change_type=ChangeType.RESTORE,
                user_id=request.user_id,
                comment=restore_comment,
                changes={"restored_from": request.version_id}
            )
            
            # Audit log
            await self._create_audit_log(
                entity_type="document",
                entity_id=request.document_id,
                action="version_restored",
                user_id=request.user_id,
                details={
                    "restored_version_id": request.version_id,
                    "new_version_id": new_version.id,
                    "comment": restore_comment
                }
            )
            
            logger.info(f"Restored version {request.version_id} as new version {new_version.id}")
            return new_version
            
        except Exception as e:
            logger.error(f"Error restoring version: {e}")
            raise
    
    async def generate_diff(
        self,
        request: VersionComparisonRequest
    ) -> DocumentDiff:
        """
        Generate a diff between two versions.
        
        Args:
            request: Version comparison request
            
        Returns:
            Document diff
        """
        try:
            # Determine versions to compare
            from_version_id = request.from_version_id
            to_version_id = request.to_version_id
            
            # If not specified, use latest and its parent
            if not to_version_id:
                to_version_id = await self._get_latest_version_id(request.document_id)
            
            if not from_version_id and to_version_id:
                to_version = await self.get_version(to_version_id)
                if to_version:
                    from_version_id = to_version.parent_version_id
            
            if not from_version_id or not to_version_id:
                raise ValueError("Cannot determine versions to compare")
            
            # Fetch both versions
            from_version = await self.get_version(from_version_id)
            to_version = await self.get_version(to_version_id)
            
            if not from_version or not to_version:
                raise ValueError("One or both versions not found")
            
            # Generate diff
            diff_result = self._calculate_diff(
                from_version.content,
                to_version.content,
                include_metadata=request.include_metadata
            )
            
            # Create diff object
            diff = DocumentDiff(
                from_version_id=from_version_id,
                to_version_id=to_version_id,
                from_version_number=from_version.version_number,
                to_version_number=to_version.version_number,
                additions=diff_result["additions"],
                deletions=diff_result["deletions"],
                modifications=diff_result["modifications"],
                total_changes=diff_result["total_changes"],
                lines_added=diff_result["lines_added"],
                lines_deleted=diff_result["lines_deleted"],
                generated_at=datetime.utcnow(),
                generated_by=request.document_id  # Should be user_id in real implementation
            )
            
            return diff
            
        except Exception as e:
            logger.error(f"Error generating diff: {e}")
            raise
    
    async def get_change_history(
        self,
        document_id: str,
        page: int = 1,
        page_size: int = 50
    ) -> ChangeHistoryResponse:
        """
        Get change history for a document.
        
        Args:
            document_id: Document ID
            page: Page number
            page_size: Items per page
            
        Returns:
            Change history response
        """
        try:
            # Fetch change history from database
            changes = await self._fetch_change_history(
                document_id,
                offset=(page - 1) * page_size,
                limit=page_size
            )
            
            # Get total count
            total_count = await self._count_change_history(document_id)
            
            return ChangeHistoryResponse(
                document_id=document_id,
                changes=changes,
                total_count=total_count,
                page=page,
                page_size=page_size
            )
            
        except Exception as e:
            logger.error(f"Error fetching change history: {e}")
            raise
    
    # Private helper methods
    
    def _calculate_content_hash(self, content: Dict[str, Any]) -> str:
        """Calculate hash of content for deduplication."""
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def _generate_changes_summary(
        self,
        old_content: Dict[str, Any],
        new_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a summary of changes between two content versions."""
        summary = {
            "fields_added": [],
            "fields_removed": [],
            "fields_modified": [],
            "total_changes": 0
        }
        
        old_keys = set(old_content.keys())
        new_keys = set(new_content.keys())
        
        # Fields added
        summary["fields_added"] = list(new_keys - old_keys)
        
        # Fields removed
        summary["fields_removed"] = list(old_keys - new_keys)
        
        # Fields modified
        for key in old_keys & new_keys:
            if old_content[key] != new_content[key]:
                summary["fields_modified"].append(key)
        
        summary["total_changes"] = (
            len(summary["fields_added"]) +
            len(summary["fields_removed"]) +
            len(summary["fields_modified"])
        )
        
        return summary
    
    def _calculate_diff(
        self,
        old_content: Dict[str, Any],
        new_content: Dict[str, Any],
        include_metadata: bool = False
    ) -> Dict[str, Any]:
        """Calculate detailed diff between two content versions."""
        result = {
            "additions": [],
            "deletions": [],
            "modifications": [],
            "total_changes": 0,
            "lines_added": 0,
            "lines_deleted": 0
        }
        
        # Convert to strings for line-by-line diff
        old_str = json.dumps(old_content, indent=2, sort_keys=True)
        new_str = json.dumps(new_content, indent=2, sort_keys=True)
        
        # Generate unified diff
        diff_lines = list(difflib.unified_diff(
            old_str.splitlines(keepends=True),
            new_str.splitlines(keepends=True),
            lineterm=""
        ))
        
        # Parse diff lines
        for line in diff_lines:
            if line.startswith("+") and not line.startswith("+++"):
                result["additions"].append({"line": line[1:].strip()})
                result["lines_added"] += 1
            elif line.startswith("-") and not line.startswith("---"):
                result["deletions"].append({"line": line[1:].strip()})
                result["lines_deleted"] += 1
        
        # Field-level comparison
        old_keys = set(old_content.keys())
        new_keys = set(new_content.keys())
        
        # Added fields
        for key in new_keys - old_keys:
            if include_metadata or not key.startswith("_"):
                result["additions"].append({
                    "field": key,
                    "value": new_content[key]
                })
        
        # Removed fields
        for key in old_keys - new_keys:
            if include_metadata or not key.startswith("_"):
                result["deletions"].append({
                    "field": key,
                    "value": old_content[key]
                })
        
        # Modified fields
        for key in old_keys & new_keys:
            if old_content[key] != new_content[key]:
                if include_metadata or not key.startswith("_"):
                    result["modifications"].append({
                        "field": key,
                        "old_value": old_content[key],
                        "new_value": new_content[key]
                    })
        
        result["total_changes"] = (
            len(result["additions"]) +
            len(result["deletions"]) +
            len(result["modifications"])
        )
        
        return result
    
    async def _validate_content(self, content: Dict[str, Any]) -> float:
        """Validate content and return a score."""
        # Placeholder for actual validation logic
        # In real implementation, this would call the GraphRAG validator
        score = 8.5  # Default score
        
        # Basic validation checks
        if "title" in content and len(content["title"]) > 10:
            score += 0.5
        if "description" in content and len(content["description"]) > 100:
            score += 0.5
        if "success_criteria" in content and len(content["success_criteria"]) > 0:
            score += 0.5
        
        return min(score, 10.0)
    
    # Cache management methods
    
    async def _cache_version(self, version: DocumentVersion) -> None:
        """Cache a version in Redis."""
        if self.redis:
            key = f"version:{version.id}"
            self.redis.setex(
                key,
                self.version_cache_ttl,
                version.json()
            )
    
    async def _get_cached_version(self, version_id: str) -> Optional[DocumentVersion]:
        """Get a cached version from Redis."""
        if self.redis:
            key = f"version:{version_id}"
            data = self.redis.get(key)
            if data:
                return DocumentVersion.parse_raw(data)
        return None
    
    # Database methods (placeholders - would use actual DB in production)
    
    async def _store_version(self, version: DocumentVersion) -> None:
        """Store version in database."""
        # In production, this would use SQLAlchemy or similar
        if self.redis:
            # Using Redis as a simple store for demo
            key = f"db:version:{version.id}"
            self.redis.set(key, version.json())
            
            # Update document's current version
            doc_key = f"db:document:{version.document_id}:current"
            self.redis.set(doc_key, version.id)
            
            # Add to version list
            list_key = f"db:document:{version.document_id}:versions"
            self.redis.rpush(list_key, version.id)
    
    async def _fetch_version(self, version_id: str) -> Optional[DocumentVersion]:
        """Fetch version from database."""
        if self.redis:
            key = f"db:version:{version_id}"
            data = self.redis.get(key)
            if data:
                return DocumentVersion.parse_raw(data)
        return None
    
    async def _get_latest_version_number(self, document_id: str) -> int:
        """Get the latest version number for a document."""
        if self.redis:
            list_key = f"db:document:{document_id}:versions"
            count = self.redis.llen(list_key)
            return count
        return 0
    
    async def _get_latest_version_id(self, document_id: str) -> Optional[str]:
        """Get the latest version ID for a document."""
        if self.redis:
            doc_key = f"db:document:{document_id}:current"
            version_id = self.redis.get(doc_key)
            return version_id.decode() if version_id else None
        return None
    
    async def _get_version_hash(self, version_id: str) -> Optional[str]:
        """Get content hash for a version."""
        version = await self.get_version(version_id)
        if version and version.metadata:
            return version.metadata.get("content_hash")
        return None
    
    async def _fetch_document_versions(
        self,
        document_id: str,
        offset: int = 0,
        limit: int = 20
    ) -> List[DocumentVersion]:
        """Fetch versions for a document."""
        versions = []
        if self.redis:
            list_key = f"db:document:{document_id}:versions"
            version_ids = self.redis.lrange(list_key, offset, offset + limit - 1)
            
            for vid in version_ids:
                version = await self._fetch_version(vid.decode())
                if version:
                    versions.append(version)
        
        return versions
    
    async def _count_document_versions(self, document_id: str) -> int:
        """Count versions for a document."""
        if self.redis:
            list_key = f"db:document:{document_id}:versions"
            return self.redis.llen(list_key)
        return 0
    
    async def _create_change_history(
        self,
        document_id: str,
        version_id: str,
        change_type: ChangeType,
        user_id: str,
        comment: Optional[str] = None,
        changes: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create a change history entry."""
        entry = ChangeHistoryEntry(
            id=f"ch_{uuid4().hex[:12]}",
            document_id=document_id,
            version_id=version_id,
            change_type=change_type,
            changed_by=user_id,
            changed_at=datetime.utcnow(),
            comment=comment
        )
        
        if self.redis:
            # Store in Redis for demo
            key = f"db:change:{entry.id}"
            self.redis.set(key, entry.json())
            
            # Add to document's change history
            list_key = f"db:document:{document_id}:changes"
            self.redis.rpush(list_key, entry.id)
    
    async def _fetch_change_history(
        self,
        document_id: str,
        offset: int = 0,
        limit: int = 50
    ) -> List[ChangeHistoryEntry]:
        """Fetch change history for a document."""
        changes = []
        if self.redis:
            list_key = f"db:document:{document_id}:changes"
            change_ids = self.redis.lrange(list_key, offset, offset + limit - 1)
            
            for cid in change_ids:
                key = f"db:change:{cid.decode()}"
                data = self.redis.get(key)
                if data:
                    changes.append(ChangeHistoryEntry.parse_raw(data))
        
        return changes
    
    async def _count_change_history(self, document_id: str) -> int:
        """Count change history entries for a document."""
        if self.redis:
            list_key = f"db:document:{document_id}:changes"
            return self.redis.llen(list_key)
        return 0
    
    async def _create_audit_log(
        self,
        entity_type: str,
        entity_id: str,
        action: str,
        user_id: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create an audit log entry."""
        audit = AuditLog(
            id=f"audit_{uuid4().hex[:12]}",
            entity_type=entity_type,
            entity_id=entity_id,
            action=action,
            user_id=user_id,
            details=details or {},
            result="success",
            timestamp=datetime.utcnow(),
            compliance_tags=["SOC2", "GDPR"]
        )
        
        if self.redis:
            # Store audit log
            key = f"db:audit:{audit.id}"
            self.redis.set(key, audit.json())
            
            # Add to audit list
            list_key = f"db:audit:list"
            self.redis.rpush(list_key, audit.id)


# Singleton instance
_version_control_service = None


def get_version_control_service() -> VersionControlService:
    """Get or create the version control service singleton."""
    global _version_control_service
    if _version_control_service is None:
        _version_control_service = VersionControlService()
    return _version_control_service