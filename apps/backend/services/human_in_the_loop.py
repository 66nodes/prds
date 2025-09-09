"""
Human-in-the-Loop Validation Service

Enables manual validation checkpoints in AI-driven workflows for critical decisions.
Provides UI prompts for user validation at key workflow stages and stores validation events.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from uuid import uuid4, UUID
from enum import Enum

from pydantic import BaseModel, Field
from sqlalchemy import Column, String, DateTime, Boolean, Text, JSON, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

from ..core.database import get_db_session
from ..services.websocket_manager import WebSocketManager
from ..services.graphrag.validation_pipeline import ValidationPipeline

logger = logging.getLogger(__name__)

Base = declarative_base()

class HumanValidationType(str, Enum):
    """Types of human validation prompts"""
    APPROVAL = "approval"
    CHOICE = "choice"
    INPUT = "input"
    REVIEW = "review"
    CONFIRMATION = "confirmation"

class ValidationEventType(str, Enum):
    """Types of validation events"""
    REQUESTED = "requested"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

class ValidationEventStatus(str, Enum):
    """Status of validation events"""
    PENDING = "pending"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

class HumanValidationOption(BaseModel):
    """Option for choice-type validations"""
    label: str
    value: str
    description: Optional[str] = None

class HumanValidationPrompt(BaseModel):
    """Human validation prompt definition"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: HumanValidationType
    question: str
    context: str
    options: Optional[List[HumanValidationOption]] = None
    required: bool = True
    timeout: Optional[int] = None  # milliseconds
    metadata: Optional[Dict[str, Any]] = None

class ValidationRequest(BaseModel):
    """Validation request model"""
    conversation_id: str
    step_id: str
    type: HumanValidationType
    prompt: str
    context: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

class ValidationResponse(BaseModel):
    """Validation response model"""
    request_id: str
    response: Any
    approved: bool
    feedback: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ValidationEvent(Base):
    """Database model for validation events"""
    __tablename__ = "validation_events"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    type = Column(String, nullable=False)  # ValidationEventType
    conversation_id = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    request_data = Column(JSON, nullable=False)
    response_data = Column(JSON, nullable=True)
    status = Column(String, nullable=False, default=ValidationEventStatus.PENDING)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    timeout_ms = Column(Integer, nullable=True)

class HumanInTheLoopService:
    """Service for managing human-in-the-loop validation workflows"""
    
    def __init__(
        self,
        websocket_manager: WebSocketManager,
        validation_pipeline: ValidationPipeline,
        db_session: Session = None
    ):
        self.websocket_manager = websocket_manager
        self.validation_pipeline = validation_pipeline
        self.db_session = db_session or next(get_db_session())
        
        # Active validation requests
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        
        # Timeout handlers
        self.timeout_tasks: Dict[str, asyncio.Task] = {}
        
        # Validation callbacks
        self.validation_callbacks: Dict[str, Callable] = {}
    
    async def request_human_validation(
        self,
        conversation_id: str,
        user_id: str,
        prompt: HumanValidationPrompt,
        step_context: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[str, Any, bool], None]] = None
    ) -> str:
        """
        Request human validation for a specific step in the workflow
        
        Args:
            conversation_id: ID of the conversation
            user_id: ID of the user who needs to validate
            prompt: Validation prompt configuration
            step_context: Additional context for the validation step
            callback: Optional callback function for validation result
            
        Returns:
            Validation request ID
        """
        try:
            # Create validation request
            request = ValidationRequest(
                conversation_id=conversation_id,
                step_id=prompt.id,
                type=prompt.type,
                prompt=prompt.question,
                context=step_context or {}
            )
            
            # Set expiration if timeout is specified
            if prompt.timeout:
                request.expires_at = datetime.utcnow() + timedelta(milliseconds=prompt.timeout)
            
            # Store in database
            validation_event = ValidationEvent(
                id=prompt.id,
                type=ValidationEventType.REQUESTED,
                conversation_id=conversation_id,
                user_id=user_id,
                request_data=request.dict(),
                timeout_ms=prompt.timeout,
                expires_at=request.expires_at
            )
            
            self.db_session.add(validation_event)
            self.db_session.commit()
            
            # Store active request
            self.active_requests[prompt.id] = {
                "prompt": prompt,
                "request": request,
                "user_id": user_id,
                "created_at": datetime.utcnow()
            }
            
            # Register callback if provided
            if callback:
                self.validation_callbacks[prompt.id] = callback
            
            # Send validation request via WebSocket
            await self.websocket_manager.send_to_user(
                user_id,
                {
                    "type": "human_validation_request",
                    "payload": {
                        "id": prompt.id,
                        "conversation_id": conversation_id,
                        "prompt": prompt.dict(),
                        "context": step_context
                    }
                }
            )
            
            # Set up timeout handler if specified
            if prompt.timeout:
                timeout_task = asyncio.create_task(
                    self._handle_validation_timeout(prompt.id, prompt.timeout)
                )
                self.timeout_tasks[prompt.id] = timeout_task
            
            logger.info(f"Human validation requested: {prompt.id} for conversation {conversation_id}")
            return prompt.id
            
        except Exception as e:
            logger.error(f"Failed to request human validation: {str(e)}")
            raise
    
    async def submit_validation_response(
        self,
        validation_id: str,
        user_id: str,
        response: Any,
        approved: bool,
        feedback: Optional[str] = None
    ) -> bool:
        """
        Submit a response to a human validation request
        
        Args:
            validation_id: ID of the validation request
            user_id: ID of the user submitting the response
            response: The validation response data
            approved: Whether the validation was approved
            feedback: Optional feedback from the user
            
        Returns:
            Success status
        """
        try:
            # Check if validation request exists
            if validation_id not in self.active_requests:
                logger.warning(f"Validation request not found: {validation_id}")
                return False
            
            active_request = self.active_requests[validation_id]
            
            # Verify user authorization
            if active_request["user_id"] != user_id:
                logger.warning(f"Unauthorized validation response from user {user_id}")
                return False
            
            # Create validation response
            validation_response = ValidationResponse(
                request_id=validation_id,
                response=response,
                approved=approved,
                feedback=feedback
            )
            
            # Update database record
            validation_event = self.db_session.query(ValidationEvent).filter(
                ValidationEvent.id == validation_id
            ).first()
            
            if validation_event:
                validation_event.response_data = validation_response.dict()
                validation_event.status = ValidationEventStatus.COMPLETED
                validation_event.type = ValidationEventType.APPROVED if approved else ValidationEventType.REJECTED
                validation_event.updated_at = datetime.utcnow()
                self.db_session.commit()
            
            # Cancel timeout handler
            if validation_id in self.timeout_tasks:
                self.timeout_tasks[validation_id].cancel()
                del self.timeout_tasks[validation_id]
            
            # Execute callback if registered
            if validation_id in self.validation_callbacks:
                callback = self.validation_callbacks[validation_id]
                try:
                    await callback(validation_id, response, approved)
                except Exception as e:
                    logger.error(f"Validation callback error: {str(e)}")
                finally:
                    del self.validation_callbacks[validation_id]
            
            # Send confirmation via WebSocket
            await self.websocket_manager.send_to_user(
                user_id,
                {
                    "type": "validation_response_acknowledged",
                    "payload": {
                        "validation_id": validation_id,
                        "approved": approved,
                        "message": "Validation response received successfully"
                    }
                }
            )
            
            # Integrate with GraphRAG pipeline for override logic
            await self._handle_graphrag_integration(validation_id, validation_response, active_request)
            
            # Clean up active request
            del self.active_requests[validation_id]
            
            logger.info(f"Validation response submitted: {validation_id} - {'Approved' if approved else 'Rejected'}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit validation response: {str(e)}")
            return False
    
    async def _handle_validation_timeout(self, validation_id: str, timeout_ms: int) -> None:
        """Handle validation timeout"""
        try:
            # Wait for timeout duration
            await asyncio.sleep(timeout_ms / 1000)
            
            # Check if validation is still active
            if validation_id in self.active_requests:
                active_request = self.active_requests[validation_id]
                user_id = active_request["user_id"]
                
                # Update database record
                validation_event = self.db_session.query(ValidationEvent).filter(
                    ValidationEvent.id == validation_id
                ).first()
                
                if validation_event:
                    validation_event.status = ValidationEventStatus.EXPIRED
                    validation_event.type = ValidationEventType.TIMEOUT
                    validation_event.updated_at = datetime.utcnow()
                    self.db_session.commit()
                
                # Execute callback with timeout result
                if validation_id in self.validation_callbacks:
                    callback = self.validation_callbacks[validation_id]
                    try:
                        await callback(validation_id, {"reason": "timeout"}, False)
                    except Exception as e:
                        logger.error(f"Timeout callback error: {str(e)}")
                    finally:
                        del self.validation_callbacks[validation_id]
                
                # Notify user of timeout
                await self.websocket_manager.send_to_user(
                    user_id,
                    {
                        "type": "validation_timeout",
                        "payload": {
                            "validation_id": validation_id,
                            "message": "Validation request has timed out"
                        }
                    }
                )
                
                # Clean up
                del self.active_requests[validation_id]
                
                logger.warning(f"Validation request timed out: {validation_id}")
                
        except asyncio.CancelledError:
            # Timeout was cancelled (validation was completed)
            logger.debug(f"Validation timeout cancelled: {validation_id}")
        except Exception as e:
            logger.error(f"Error handling validation timeout: {str(e)}")
    
    async def _handle_graphrag_integration(
        self,
        validation_id: str,
        response: ValidationResponse,
        active_request: Dict[str, Any]
    ) -> None:
        """Integrate validation response with GraphRAG pipeline"""
        try:
            conversation_id = active_request["request"].conversation_id
            prompt = active_request["prompt"]
            
            # Create validation context for GraphRAG
            validation_context = {
                "validation_id": validation_id,
                "validation_type": prompt.type.value,
                "question": prompt.question,
                "user_response": response.response,
                "approved": response.approved,
                "feedback": response.feedback,
                "conversation_id": conversation_id,
                "timestamp": response.timestamp.isoformat()
            }
            
            # If validation was approved, update GraphRAG knowledge
            if response.approved:
                await self.validation_pipeline.record_human_approval(
                    validation_context,
                    confidence_boost=0.2  # Boost confidence for human-approved content
                )
            else:
                # Record human rejection for future reference
                await self.validation_pipeline.record_human_rejection(
                    validation_context,
                    confidence_penalty=0.3  # Reduce confidence for rejected content
                )
            
            logger.debug(f"GraphRAG integration completed for validation: {validation_id}")
            
        except Exception as e:
            logger.error(f"Failed to integrate with GraphRAG: {str(e)}")
    
    async def cancel_validation_request(self, validation_id: str, user_id: str) -> bool:
        """Cancel an active validation request"""
        try:
            if validation_id not in self.active_requests:
                return False
            
            active_request = self.active_requests[validation_id]
            
            # Verify user authorization
            if active_request["user_id"] != user_id:
                logger.warning(f"Unauthorized validation cancellation from user {user_id}")
                return False
            
            # Update database record
            validation_event = self.db_session.query(ValidationEvent).filter(
                ValidationEvent.id == validation_id
            ).first()
            
            if validation_event:
                validation_event.status = ValidationEventStatus.CANCELLED
                validation_event.type = ValidationEventType.CANCELLED
                validation_event.updated_at = datetime.utcnow()
                self.db_session.commit()
            
            # Cancel timeout handler
            if validation_id in self.timeout_tasks:
                self.timeout_tasks[validation_id].cancel()
                del self.timeout_tasks[validation_id]
            
            # Clean up callback
            if validation_id in self.validation_callbacks:
                del self.validation_callbacks[validation_id]
            
            # Clean up active request
            del self.active_requests[validation_id]
            
            logger.info(f"Validation request cancelled: {validation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel validation request: {str(e)}")
            return False
    
    async def get_validation_history(
        self,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get validation history with optional filters"""
        try:
            query = self.db_session.query(ValidationEvent)
            
            if conversation_id:
                query = query.filter(ValidationEvent.conversation_id == conversation_id)
            
            if user_id:
                query = query.filter(ValidationEvent.user_id == user_id)
            
            events = query.order_by(ValidationEvent.created_at.desc()).limit(limit).all()
            
            return [
                {
                    "id": event.id,
                    "type": event.type,
                    "conversation_id": event.conversation_id,
                    "user_id": event.user_id,
                    "request": event.request_data,
                    "response": event.response_data,
                    "status": event.status,
                    "created_at": event.created_at.isoformat(),
                    "updated_at": event.updated_at.isoformat(),
                    "expires_at": event.expires_at.isoformat() if event.expires_at else None
                }
                for event in events
            ]
            
        except Exception as e:
            logger.error(f"Failed to get validation history: {str(e)}")
            return []
    
    async def get_active_validations(self, user_id: str) -> List[Dict[str, Any]]:
        """Get active validation requests for a user"""
        try:
            active_validations = []
            
            for validation_id, request_data in self.active_requests.items():
                if request_data["user_id"] == user_id:
                    active_validations.append({
                        "id": validation_id,
                        "prompt": request_data["prompt"].dict(),
                        "conversation_id": request_data["request"].conversation_id,
                        "created_at": request_data["created_at"].isoformat()
                    })
            
            return active_validations
            
        except Exception as e:
            logger.error(f"Failed to get active validations: {str(e)}")
            return []
    
    def cleanup_expired_requests(self) -> None:
        """Clean up expired validation requests"""
        try:
            current_time = datetime.utcnow()
            expired_requests = []
            
            for validation_id, request_data in self.active_requests.items():
                request = request_data["request"]
                if request.expires_at and request.expires_at < current_time:
                    expired_requests.append(validation_id)
            
            for validation_id in expired_requests:
                # Cancel timeout handler
                if validation_id in self.timeout_tasks:
                    self.timeout_tasks[validation_id].cancel()
                    del self.timeout_tasks[validation_id]
                
                # Clean up callback
                if validation_id in self.validation_callbacks:
                    del self.validation_callbacks[validation_id]
                
                # Clean up active request
                del self.active_requests[validation_id]
                
                logger.debug(f"Cleaned up expired validation request: {validation_id}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired requests: {str(e)}")

# Utility functions for common validation patterns

async def request_approval(
    service: HumanInTheLoopService,
    conversation_id: str,
    user_id: str,
    question: str,
    context: str,
    timeout_ms: Optional[int] = None
) -> bool:
    """Request simple approval validation"""
    prompt = HumanValidationPrompt(
        type=HumanValidationType.APPROVAL,
        question=question,
        context=context,
        timeout=timeout_ms
    )
    
    result = {"approved": False}
    
    def callback(validation_id: str, response: Any, approved: bool):
        result["approved"] = approved
    
    await service.request_human_validation(
        conversation_id, user_id, prompt, callback=callback
    )
    
    # Wait for response (with timeout)
    timeout_seconds = (timeout_ms / 1000) if timeout_ms else 300
    for _ in range(int(timeout_seconds)):
        if validation_id not in service.active_requests:
            break
        await asyncio.sleep(1)
    
    return result["approved"]

async def request_choice(
    service: HumanInTheLoopService,
    conversation_id: str,
    user_id: str,
    question: str,
    options: List[HumanValidationOption],
    context: str,
    timeout_ms: Optional[int] = None
) -> Optional[str]:
    """Request choice validation"""
    prompt = HumanValidationPrompt(
        type=HumanValidationType.CHOICE,
        question=question,
        context=context,
        options=options,
        timeout=timeout_ms
    )
    
    result = {"choice": None}
    
    def callback(validation_id: str, response: Any, approved: bool):
        if approved and isinstance(response, dict):
            result["choice"] = response.get("choice")
    
    await service.request_human_validation(
        conversation_id, user_id, prompt, callback=callback
    )
    
    # Wait for response
    timeout_seconds = (timeout_ms / 1000) if timeout_ms else 300
    for _ in range(int(timeout_seconds)):
        if validation_id not in service.active_requests:
            break
        await asyncio.sleep(1)
    
    return result["choice"]