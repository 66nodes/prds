"""
API endpoints for Human-in-the-Loop validation system
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from datetime import datetime

from ...core.database import get_db_session
from ...services.auth_service import get_current_user, User
from ...services.human_in_the_loop import (
    HumanInTheLoopService,
    HumanValidationPrompt,
    HumanValidationType,
    ValidationResponse,
    ValidationEvent
)
from ...services.websocket_manager import WebSocketManager
from ...services.graphrag.validation_pipeline import ValidationPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/human-validation", tags=["human-validation"])
security = HTTPBearer()

# Request/Response Models

class ValidationRequestModel(BaseModel):
    """Model for requesting human validation"""
    conversation_id: str
    validation_type: HumanValidationType
    question: str
    context: str
    options: Optional[List[Dict[str, str]]] = None
    required: bool = True
    timeout_ms: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class ValidationResponseModel(BaseModel):
    """Model for validation response"""
    validation_id: str
    response: Any
    approved: bool
    feedback: Optional[str] = None

class ValidationHistoryResponse(BaseModel):
    """Response model for validation history"""
    id: str
    type: str
    conversation_id: str
    user_id: str
    request_data: Dict[str, Any]
    response_data: Optional[Dict[str, Any]]
    status: str
    created_at: str
    updated_at: str
    expires_at: Optional[str]

class ActiveValidationResponse(BaseModel):
    """Response model for active validations"""
    id: str
    prompt: Dict[str, Any]
    conversation_id: str
    created_at: str

# Dependency injection

async def get_validation_service(
    db: Session = Depends(get_db_session),
    websocket_manager: WebSocketManager = Depends(lambda: WebSocketManager()),
    validation_pipeline: ValidationPipeline = Depends(lambda: ValidationPipeline())
) -> HumanInTheLoopService:
    """Get human-in-the-loop validation service instance"""
    return HumanInTheLoopService(
        websocket_manager=websocket_manager,
        validation_pipeline=validation_pipeline,
        db_session=db
    )

# API Endpoints

@router.post("/request", response_model=Dict[str, str])
async def request_validation(
    request: ValidationRequestModel,
    current_user: User = Depends(get_current_user),
    validation_service: HumanInTheLoopService = Depends(get_validation_service)
):
    """
    Request human validation for a specific workflow step
    """
    try:
        # Create validation prompt
        options = None
        if request.options:
            options = [
                {"label": opt["label"], "value": opt["value"], "description": opt.get("description")}
                for opt in request.options
            ]
        
        prompt = HumanValidationPrompt(
            type=request.validation_type,
            question=request.question,
            context=request.context,
            options=options,
            required=request.required,
            timeout=request.timeout_ms,
            metadata=request.metadata
        )
        
        # Request validation
        validation_id = await validation_service.request_human_validation(
            conversation_id=request.conversation_id,
            user_id=str(current_user.id),
            prompt=prompt,
            step_context={"request_metadata": request.metadata}
        )
        
        logger.info(f"Validation requested by user {current_user.id}: {validation_id}")
        
        return {
            "validation_id": validation_id,
            "status": "requested",
            "message": "Validation request created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to request validation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to request validation: {str(e)}"
        )

@router.post("/respond", response_model=Dict[str, str])
async def submit_validation_response(
    response: ValidationResponseModel,
    current_user: User = Depends(get_current_user),
    validation_service: HumanInTheLoopService = Depends(get_validation_service)
):
    """
    Submit a response to a validation request
    """
    try:
        success = await validation_service.submit_validation_response(
            validation_id=response.validation_id,
            user_id=str(current_user.id),
            response=response.response,
            approved=response.approved,
            feedback=response.feedback
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to submit validation response. Request may not exist or be expired."
            )
        
        logger.info(f"Validation response submitted by user {current_user.id}: {response.validation_id}")
        
        return {
            "validation_id": response.validation_id,
            "status": "completed",
            "message": "Validation response submitted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit validation response: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit validation response: {str(e)}"
        )

@router.delete("/{validation_id}", response_model=Dict[str, str])
async def cancel_validation(
    validation_id: str,
    current_user: User = Depends(get_current_user),
    validation_service: HumanInTheLoopService = Depends(get_validation_service)
):
    """
    Cancel an active validation request
    """
    try:
        success = await validation_service.cancel_validation_request(
            validation_id=validation_id,
            user_id=str(current_user.id)
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to cancel validation request. Request may not exist or be completed."
            )
        
        logger.info(f"Validation cancelled by user {current_user.id}: {validation_id}")
        
        return {
            "validation_id": validation_id,
            "status": "cancelled",
            "message": "Validation request cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel validation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel validation: {str(e)}"
        )

@router.get("/history", response_model=List[ValidationHistoryResponse])
async def get_validation_history(
    conversation_id: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    validation_service: HumanInTheLoopService = Depends(get_validation_service)
):
    """
    Get validation history for the current user
    """
    try:
        history = await validation_service.get_validation_history(
            conversation_id=conversation_id,
            user_id=str(current_user.id),
            limit=limit
        )
        
        return [
            ValidationHistoryResponse(
                id=event["id"],
                type=event["type"],
                conversation_id=event["conversation_id"],
                user_id=event["user_id"],
                request_data=event["request"],
                response_data=event["response"],
                status=event["status"],
                created_at=event["created_at"],
                updated_at=event["updated_at"],
                expires_at=event["expires_at"]
            )
            for event in history
        ]
        
    except Exception as e:
        logger.error(f"Failed to get validation history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get validation history: {str(e)}"
        )

@router.get("/active", response_model=List[ActiveValidationResponse])
async def get_active_validations(
    current_user: User = Depends(get_current_user),
    validation_service: HumanInTheLoopService = Depends(get_validation_service)
):
    """
    Get active validation requests for the current user
    """
    try:
        active_validations = await validation_service.get_active_validations(
            user_id=str(current_user.id)
        )
        
        return [
            ActiveValidationResponse(
                id=validation["id"],
                prompt=validation["prompt"],
                conversation_id=validation["conversation_id"],
                created_at=validation["created_at"]
            )
            for validation in active_validations
        ]
        
    except Exception as e:
        logger.error(f"Failed to get active validations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get active validations: {str(e)}"
        )

@router.get("/{validation_id}", response_model=ValidationHistoryResponse)
async def get_validation_details(
    validation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Get details for a specific validation request
    """
    try:
        # Query validation event from database
        validation_event = db.query(ValidationEvent).filter(
            ValidationEvent.id == validation_id,
            ValidationEvent.user_id == str(current_user.id)
        ).first()
        
        if not validation_event:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Validation request not found"
            )
        
        return ValidationHistoryResponse(
            id=validation_event.id,
            type=validation_event.type,
            conversation_id=validation_event.conversation_id,
            user_id=validation_event.user_id,
            request_data=validation_event.request_data,
            response_data=validation_event.response_data,
            status=validation_event.status,
            created_at=validation_event.created_at.isoformat(),
            updated_at=validation_event.updated_at.isoformat(),
            expires_at=validation_event.expires_at.isoformat() if validation_event.expires_at else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get validation details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get validation details: {str(e)}"
        )

# Background task endpoints

@router.post("/cleanup", response_model=Dict[str, str])
async def cleanup_expired_validations(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    validation_service: HumanInTheLoopService = Depends(get_validation_service)
):
    """
    Trigger cleanup of expired validation requests
    """
    try:
        # Add cleanup task to background tasks
        background_tasks.add_task(validation_service.cleanup_expired_requests)
        
        return {
            "status": "scheduled",
            "message": "Expired validation cleanup scheduled"
        }
        
    except Exception as e:
        logger.error(f"Failed to schedule cleanup: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to schedule cleanup: {str(e)}"
        )

# Utility endpoints for testing

@router.post("/test/approval", response_model=Dict[str, Any])
async def test_approval_validation(
    conversation_id: str,
    question: str,
    context: str,
    timeout_ms: Optional[int] = 30000,
    current_user: User = Depends(get_current_user),
    validation_service: HumanInTheLoopService = Depends(get_validation_service)
):
    """
    Test endpoint for approval validation
    """
    try:
        from ...services.human_in_the_loop import request_approval
        
        approved = await request_approval(
            service=validation_service,
            conversation_id=conversation_id,
            user_id=str(current_user.id),
            question=question,
            context=context,
            timeout_ms=timeout_ms
        )
        
        return {
            "approved": approved,
            "message": f"Approval validation completed: {'Approved' if approved else 'Rejected'}"
        }
        
    except Exception as e:
        logger.error(f"Test approval validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Test approval validation failed: {str(e)}"
        )

@router.post("/test/choice", response_model=Dict[str, Any])
async def test_choice_validation(
    conversation_id: str,
    question: str,
    context: str,
    options: List[Dict[str, str]],
    timeout_ms: Optional[int] = 30000,
    current_user: User = Depends(get_current_user),
    validation_service: HumanInTheLoopService = Depends(get_validation_service)
):
    """
    Test endpoint for choice validation
    """
    try:
        from ...services.human_in_the_loop import request_choice, HumanValidationOption
        
        choice_options = [
            HumanValidationOption(
                label=opt["label"],
                value=opt["value"],
                description=opt.get("description")
            )
            for opt in options
        ]
        
        choice = await request_choice(
            service=validation_service,
            conversation_id=conversation_id,
            user_id=str(current_user.id),
            question=question,
            options=choice_options,
            context=context,
            timeout_ms=timeout_ms
        )
        
        return {
            "choice": choice,
            "message": f"Choice validation completed: {choice or 'No choice made'}"
        }
        
    except Exception as e:
        logger.error(f"Test choice validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Test choice validation failed: {str(e)}"
        )

# Health check

@router.get("/health", response_model=Dict[str, str])
async def validation_health_check():
    """
    Health check endpoint for validation service
    """
    return {
        "status": "healthy",
        "service": "human-in-the-loop-validation",
        "timestamp": datetime.utcnow().isoformat()
    }