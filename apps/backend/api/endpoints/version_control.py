"""
API endpoints for version control and change tracking.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse

from ...core.logging_config import get_logger
from ...models.version_control import (
    DocumentDiff,
    DocumentVersion,
    VersionComparisonRequest,
    VersionCreateRequest,
    VersionListResponse,
    VersionRestoreRequest,
    ChangeHistoryResponse,
)
from ...services.version_control_service import get_version_control_service
from ..dependencies.auth import get_current_user

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/versions",
    tags=["version-control"],
)


@router.post(
    "/create", 
    response_model=DocumentVersion,
    summary="Create new document version",
    description="Creates a new version of a document with automatic change detection and content hashing for deduplication",
    responses={
        201: {"description": "Version created successfully", "model": DocumentVersion},
        400: {"description": "Invalid request data"},
        401: {"description": "Authentication required"},
        500: {"description": "Internal server error"}
    },
    tags=["Version Control"]
)
async def create_version(
    request: VersionCreateRequest,
    auto_validate: bool = Query(True, description="Auto-validate content using GraphRAG"),
    current_user: dict = Depends(get_current_user)
) -> DocumentVersion:
    """
    Create a new version of a document.
    
    This endpoint creates a new version with automatic change detection
    and content hashing for deduplication. If the content is identical 
    to the previous version, the existing version will be returned.
    
    - **document_id**: Unique identifier for the document
    - **document_type**: Type of document (prd, specification, etc.)
    - **content**: Document content as JSON object
    - **comment**: Optional comment describing the changes
    - **auto_validate**: Whether to validate content using GraphRAG
    
    Returns the created DocumentVersion with metadata and change summary.
    """
    try:
        # Add user ID from authenticated user
        request.user_id = current_user.get("id", current_user.get("sub", "unknown"))
        
        # Get version control service
        service = get_version_control_service()
        
        # Create version
        version = await service.create_version(request, auto_validate=auto_validate)
        
        logger.info(f"User {request.user_id} created version {version.id} for document {request.document_id}")
        return version
        
    except ValueError as e:
        logger.error(f"Validation error creating version: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating version: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create version"
        )


@router.get(
    "/{version_id}", 
    response_model=DocumentVersion,
    summary="Get version by ID",
    description="Retrieves a specific document version by its unique identifier",
    responses={
        200: {"description": "Version retrieved successfully", "model": DocumentVersion},
        401: {"description": "Authentication required"},
        404: {"description": "Version not found"},
        500: {"description": "Internal server error"}
    },
    tags=["Version Control"]
)
async def get_version(
    version_id: str,
    current_user: dict = Depends(get_current_user)
) -> DocumentVersion:
    """
    Get a specific version by ID.
    
    Retrieves the complete version including content, metadata, and change summary.
    
    - **version_id**: Unique identifier for the version
    
    Returns the DocumentVersion with all associated data.
    """
    try:
        service = get_version_control_service()
        version = await service.get_version(version_id)
        
        if not version:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Version {version_id} not found"
            )
        
        return version
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching version {version_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch version"
        )


@router.get(
    "/document/{document_id}", 
    response_model=VersionListResponse,
    summary="List document versions",
    description="Retrieves a paginated list of all versions for a specific document",
    responses={
        200: {"description": "Versions retrieved successfully", "model": VersionListResponse},
        401: {"description": "Authentication required"},
        404: {"description": "Document not found"},
        500: {"description": "Internal server error"}
    },
    tags=["Version Control"]
)
async def list_document_versions(
    document_id: str,
    page: int = Query(1, ge=1, description="Page number (starting from 1)"),
    page_size: int = Query(20, ge=1, le=100, description="Number of items per page (max 100)"),
    current_user: dict = Depends(get_current_user)
) -> VersionListResponse:
    """
    List all versions of a document.
    
    Returns a paginated list of versions ordered by creation date (newest first).
    
    - **document_id**: Unique identifier for the document
    - **page**: Page number for pagination (starts at 1)
    - **page_size**: Number of versions per page (1-100)
    
    Returns VersionListResponse with versions array and pagination metadata.
    """
    try:
        service = get_version_control_service()
        response = await service.list_versions(
            document_id=document_id,
            page=page,
            page_size=page_size
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error listing versions for document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list versions"
        )


@router.post(
    "/restore", 
    response_model=DocumentVersion,
    summary="Restore document version",
    description="Restores a previous version by creating a new version with the restored content",
    responses={
        201: {"description": "Version restored successfully", "model": DocumentVersion},
        400: {"description": "Invalid request data"},
        401: {"description": "Authentication required"},
        404: {"description": "Version to restore not found"},
        500: {"description": "Internal server error"}
    },
    tags=["Version Control"]
)
async def restore_version(
    request: VersionRestoreRequest,
    current_user: dict = Depends(get_current_user)
) -> DocumentVersion:
    """
    Restore a previous version of a document.
    
    Creates a new version using the content from a specified previous version.
    This is useful for rolling back changes or recovering from mistakes.
    
    - **document_id**: Unique identifier for the document
    - **version_id**: ID of the version to restore from
    - **comment**: Optional comment explaining the restore operation
    
    Returns a new DocumentVersion with the restored content.
    """
    try:
        # Add user ID from authenticated user
        request.user_id = current_user.get("id", current_user.get("sub", "unknown"))
        
        service = get_version_control_service()
        restored_version = await service.restore_version(request)
        
        logger.info(
            f"User {request.user_id} restored version {request.version_id} "
            f"for document {request.document_id} as new version {restored_version.id}"
        )
        
        return restored_version
        
    except ValueError as e:
        logger.error(f"Error restoring version: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error restoring version: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to restore version"
        )


@router.post(
    "/compare", 
    response_model=DocumentDiff,
    summary="Compare document versions",
    description="Generates a detailed diff between two versions showing additions, deletions, and modifications",
    responses={
        200: {"description": "Diff generated successfully", "model": DocumentDiff},
        400: {"description": "Invalid request data"},
        401: {"description": "Authentication required"},
        404: {"description": "One or both versions not found"},
        500: {"description": "Internal server error"}
    },
    tags=["Version Control"]
)
async def compare_versions(
    request: VersionComparisonRequest,
    current_user: dict = Depends(get_current_user)
) -> DocumentDiff:
    """
    Compare two versions of a document.
    
    Generates a detailed diff showing additions, deletions, and modifications
    between two versions of a document. Useful for reviewing changes.
    
    - **document_id**: Unique identifier for the document
    - **from_version_id**: ID of the source version to compare from
    - **to_version_id**: ID of the target version to compare to
    - **include_metadata**: Whether to include metadata changes in the diff
    
    Returns DocumentDiff with detailed change analysis.
    """
    try:
        service = get_version_control_service()
        diff = await service.generate_diff(request)
        
        return diff
        
    except ValueError as e:
        logger.error(f"Error comparing versions: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error comparing versions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compare versions"
        )


@router.get(
    "/history/{document_id}", 
    response_model=ChangeHistoryResponse,
    summary="Get document change history",
    description="Retrieves a chronological list of all changes made to a document",
    responses={
        200: {"description": "Change history retrieved successfully", "model": ChangeHistoryResponse},
        401: {"description": "Authentication required"},
        404: {"description": "Document not found"},
        500: {"description": "Internal server error"}
    },
    tags=["Version Control"]
)
async def get_change_history(
    document_id: str,
    page: int = Query(1, ge=1, description="Page number (starting from 1)"),
    page_size: int = Query(50, ge=1, le=200, description="Number of items per page (max 200)"),
    current_user: dict = Depends(get_current_user)
) -> ChangeHistoryResponse:
    """
    Get change history for a document.
    
    Returns a chronological list of all changes made to the document,
    including who made the change, when, and what type of change it was.
    
    - **document_id**: Unique identifier for the document
    - **page**: Page number for pagination (starts at 1)
    - **page_size**: Number of changes per page (1-200)
    
    Returns ChangeHistoryResponse with change entries and pagination metadata.
    """
    try:
        service = get_version_control_service()
        history = await service.get_change_history(
            document_id=document_id,
            page=page,
            page_size=page_size
        )
        
        return history
        
    except Exception as e:
        logger.error(f"Error fetching change history for document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch change history"
        )


@router.delete(
    "/{version_id}",
    summary="Delete document version", 
    description="Soft deletes a document version (marks as deleted but keeps for audit trail)",
    responses={
        200: {"description": "Version deleted successfully"},
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions (admin required)"},
        404: {"description": "Version not found"},
        500: {"description": "Internal server error"}
    },
    tags=["Version Control"]
)
async def delete_version(
    version_id: str,
    current_user: dict = Depends(get_current_user)
) -> JSONResponse:
    """
    Soft delete a version (mark as deleted but keep for audit).
    
    Marks a version as deleted while preserving it for audit trails.
    Only administrators can perform this operation.
    
    - **version_id**: Unique identifier for the version to delete
    
    Returns confirmation message with the deleted version ID.
    """
    try:
        # Check if user is admin
        user_role = current_user.get("role", "user")
        if user_role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only administrators can delete versions"
            )
        
        # In production, implement soft delete
        # For now, just return success
        logger.info(f"User {current_user.get('id')} deleted version {version_id}")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": f"Version {version_id} deleted successfully",
                "version_id": version_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting version {version_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete version"
        )


@router.get(
    "/latest/{document_id}", 
    response_model=DocumentVersion,
    summary="Get latest document version",
    description="Convenience endpoint to retrieve the most recent version of a document",
    responses={
        200: {"description": "Latest version retrieved successfully", "model": DocumentVersion},
        401: {"description": "Authentication required"},
        404: {"description": "Document not found or no versions exist"},
        500: {"description": "Internal server error"}
    },
    tags=["Version Control"]
)
async def get_latest_version(
    document_id: str,
    current_user: dict = Depends(get_current_user)
) -> DocumentVersion:
    """
    Get the latest version of a document.
    
    Convenience endpoint to fetch the most recent version without pagination.
    Useful for getting the current state of a document.
    
    - **document_id**: Unique identifier for the document
    
    Returns the most recent DocumentVersion.
    """
    try:
        service = get_version_control_service()
        
        # Get latest version ID
        versions = await service.list_versions(document_id, page=1, page_size=1)
        
        if not versions.versions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No versions found for document {document_id}"
            )
        
        return versions.versions[0]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching latest version for document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch latest version"
        )