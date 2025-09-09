"""
LLM API endpoints for AI model interactions.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.dependencies.auth import get_current_user
from models.user import User
from services.llm import (
    LLMService,
    LLMRequest,
    ChatMessage,
    LLMResponse,
    get_llm_service
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["llm"])


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""
    messages: List[Dict[str, str]] = Field(..., description="Chat messages with role and content")
    model: Optional[str] = Field(default=None, description="Preferred model")
    task_type: str = Field(default="general", description="Type of task")
    complexity: str = Field(default="standard", description="Task complexity") 
    temperature: Optional[float] = Field(default=None, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens")
    require_high_confidence: bool = Field(default=False, description="Require high confidence")


class PRDGenerationRequest(BaseModel):
    """Request for PRD content generation."""
    prompt: str = Field(..., min_length=10, description="Content generation prompt")
    context: Optional[str] = Field(default=None, description="Optional context")
    section_type: str = Field(default="general", description="PRD section type")


class ContentValidationRequest(BaseModel):
    """Request for content validation."""
    content: str = Field(..., min_length=10, description="Content to validate")
    validation_type: str = Field(default="general", description="Type of validation")
    criteria: Optional[List[str]] = Field(default=None, description="Validation criteria")


class RequirementsAnalysisRequest(BaseModel):
    """Request for requirements analysis."""
    requirements_text: str = Field(..., min_length=10, description="Requirements to analyze")
    analysis_type: str = Field(default="comprehensive", description="Type of analysis")


class SummaryRequest(BaseModel):
    """Request for content summarization."""
    content: str = Field(..., min_length=50, description="Content to summarize")
    summary_type: str = Field(default="executive", description="Type of summary")
    max_length: str = Field(default="medium", description="Summary length")


@router.post("/chat/completions", response_model=LLMResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    current_user: User = Depends(get_current_user),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Generate chat completion using AI models with intelligent fallback.
    
    This endpoint provides access to multiple AI models through OpenRouter
    with automatic model selection and fallback strategies based on:
    - Task complexity and type
    - Model availability and performance
    - Confidence requirements
    
    Returns the generated response with metadata including which model
    was used and confidence score.
    """
    try:
        # Convert request to internal format
        chat_messages = [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in request.messages
        ]
        
        llm_request = LLMRequest(
            messages=chat_messages,
            model=request.model,
            task_type=request.task_type,
            complexity=request.complexity,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            require_high_confidence=request.require_high_confidence
        )
        
        logger.info(f"User {current_user.id} requesting chat completion for task '{request.task_type}'")
        
        response = await llm_service.generate_completion(llm_request)
        
        logger.info(f"Chat completion generated for user {current_user.id}: "
                   f"model={response.model}, confidence={response.confidence}")
        
        return response
        
    except Exception as e:
        logger.error(f"Chat completion failed for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate completion: {str(e)}"
        )


@router.post("/prd/generate", response_model=LLMResponse)
async def generate_prd_content(
    request: PRDGenerationRequest,
    current_user: User = Depends(get_current_user),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Generate Product Requirements Document content.
    
    Specialized endpoint for generating high-quality PRD content
    using models optimized for technical writing and product management.
    
    Automatically uses high-confidence models and includes specialized
    prompting for PRD generation best practices.
    """
    try:
        logger.info(f"User {current_user.id} requesting PRD content generation")
        
        response = await llm_service.generate_prd_content(
            prompt=request.prompt,
            context=request.context,
            section_type=request.section_type
        )
        
        logger.info(f"PRD content generated for user {current_user.id}: "
                   f"model={response.model}, confidence={response.confidence}")
        
        return response
        
    except Exception as e:
        logger.error(f"PRD generation failed for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate PRD content: {str(e)}"
        )


@router.post("/validate", response_model=LLMResponse)
async def validate_content(
    request: ContentValidationRequest,
    current_user: User = Depends(get_current_user),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Validate content for quality, accuracy, and completeness.
    
    Provides AI-powered content validation with structured feedback
    and recommendations for improvement. Useful for validating:
    - PRD sections and technical documentation
    - Business requirements and specifications  
    - Content quality and clarity
    """
    try:
        logger.info(f"User {current_user.id} requesting content validation")
        
        response = await llm_service.validate_content(
            content=request.content,
            validation_type=request.validation_type,
            criteria=request.criteria
        )
        
        logger.info(f"Content validation completed for user {current_user.id}: "
                   f"model={response.model}, confidence={response.confidence}")
        
        return response
        
    except Exception as e:
        logger.error(f"Content validation failed for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate content: {str(e)}"
        )


@router.post("/analyze/requirements", response_model=LLMResponse)
async def analyze_requirements(
    request: RequirementsAnalysisRequest,
    current_user: User = Depends(get_current_user),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Analyze requirements for gaps, conflicts, and improvements.
    
    Provides comprehensive analysis of business and technical requirements
    including:
    - Gap analysis and completeness assessment
    - Conflict detection and resolution suggestions
    - Risk assessment and mitigation strategies
    - Prioritization and implementation recommendations
    """
    try:
        logger.info(f"User {current_user.id} requesting requirements analysis")
        
        response = await llm_service.analyze_requirements(
            requirements_text=request.requirements_text,
            analysis_type=request.analysis_type
        )
        
        logger.info(f"Requirements analysis completed for user {current_user.id}: "
                   f"model={response.model}, confidence={response.confidence}")
        
        return response
        
    except Exception as e:
        logger.error(f"Requirements analysis failed for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze requirements: {str(e)}"
        )


@router.post("/summarize", response_model=LLMResponse)
async def summarize_content(
    request: SummaryRequest,
    current_user: User = Depends(get_current_user),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Generate summaries of content.
    
    Creates professional summaries optimized for different audiences:
    - Executive summaries for leadership
    - Technical summaries for development teams
    - Bullet-point summaries for quick reference
    
    Configurable length and focus based on intended use case.
    """
    try:
        logger.info(f"User {current_user.id} requesting content summarization")
        
        response = await llm_service.generate_summary(
            content=request.content,
            summary_type=request.summary_type,
            max_length=request.max_length
        )
        
        logger.info(f"Content summarization completed for user {current_user.id}: "
                   f"model={response.model}, confidence={response.confidence}")
        
        return response
        
    except Exception as e:
        logger.error(f"Content summarization failed for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to summarize content: {str(e)}"
        )


@router.get("/health")
async def health_check(
    current_user: User = Depends(get_current_user),
    llm_service: LLMService = Depends(get_llm_service)
) -> Dict[str, Any]:
    """
    Check LLM service health and connectivity.
    
    Returns status of:
    - LLM service availability
    - OpenRouter API connectivity  
    - Model availability and response times
    - Configuration validation
    """
    try:
        health_status = await llm_service.health_check()
        return health_status
        
    except Exception as e:
        logger.error(f"LLM health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/models")
async def get_model_info(
    current_user: User = Depends(get_current_user),
    llm_service: LLMService = Depends(get_llm_service)
) -> Dict[str, Any]:
    """
    Get information about available AI models.
    
    Returns details about:
    - Configured models and their capabilities
    - Model tiers (premium, standard, budget)
    - Performance characteristics
    - Current default and fallback models
    """
    try:
        model_info = await llm_service.get_model_info()
        return model_info
        
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model information: {str(e)}"
        )


@router.get("/usage/stats")
async def get_usage_statistics(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get LLM usage statistics for the current user.
    
    Returns:
    - Token usage by model and time period
    - Request counts and success rates
    - Average response times
    - Cost estimates (if available)
    """
    try:
        # TODO: Implement usage tracking in future iteration
        return {
            "message": "Usage statistics feature coming soon",
            "user_id": current_user.id,
            "current_session": {
                "requests_made": 0,
                "tokens_used": 0,
                "estimated_cost": 0.0
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get usage statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get usage statistics: {str(e)}"
        )