"""
API endpoints for Context-Aware Agent Selection.

Provides intelligent agent selection and orchestration capabilities
for multi-agent workflow optimization.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field, validator

from ...core.database import get_db_session
from ...services.auth_service import get_current_user, User
from ...services.agent_orchestrator import get_orchestrator, AgentOrchestrator, AgentType, TaskPriority
from ...services.context_aware_agent_selector import TaskContext, get_context_aware_selector
from ...services.agent_registry import CapabilityType, ComplexityLevel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent-selection", tags=["agent-selection"])
security = HTTPBearer()


# Request/Response Models

class AgentSelectionRequest(BaseModel):
    """Request model for intelligent agent selection."""
    task_context: TaskContext = Field(..., description="Context type for the task")
    required_capabilities: List[CapabilityType] = Field(..., description="Required agent capabilities")
    complexity_level: ComplexityLevel = Field(default=ComplexityLevel.MODERATE, description="Task complexity level")
    estimated_tokens: int = Field(default=5000, ge=100, le=100000, description="Estimated token requirement")
    max_execution_time_minutes: int = Field(default=10, ge=1, le=120, description="Maximum execution time in minutes")
    max_agents: int = Field(default=5, ge=1, le=20, description="Maximum number of agents to select")
    domain_knowledge: Optional[List[str]] = Field(default=None, description="Required domain knowledge areas")
    preferred_agents: Optional[List[AgentType]] = Field(default=None, description="Preferred agent types")
    excluded_agents: Optional[List[AgentType]] = Field(default=None, description="Agents to exclude from selection")
    
    @validator('required_capabilities')
    def validate_capabilities(cls, v):
        if not v:
            raise ValueError("At least one capability must be specified")
        return v


class WorkflowCreationRequest(BaseModel):
    """Request model for creating intelligent workflows."""
    task_context: TaskContext = Field(..., description="Context type for the workflow")
    required_capabilities: List[CapabilityType] = Field(..., description="Required agent capabilities")
    project_id: Optional[str] = Field(default=None, description="Associated project ID")
    complexity_level: ComplexityLevel = Field(default=ComplexityLevel.MODERATE, description="Task complexity level")
    estimated_tokens: int = Field(default=5000, ge=100, le=100000, description="Estimated token requirement")
    max_execution_time_minutes: int = Field(default=10, ge=1, le=120, description="Maximum execution time in minutes")
    max_agents: int = Field(default=5, ge=1, le=20, description="Maximum number of agents to select")
    domain_knowledge: Optional[List[str]] = Field(default=None, description="Required domain knowledge areas")
    initial_context: Optional[Dict[str, Any]] = Field(default=None, description="Initial workflow context")


class TaskAdditionRequest(BaseModel):
    """Request model for adding intelligent tasks to workflows."""
    workflow_id: str = Field(..., description="Target workflow ID")
    operation: str = Field(..., description="Operation to perform")
    task_context: TaskContext = Field(..., description="Context type for the task")
    required_capabilities: List[CapabilityType] = Field(..., description="Required agent capabilities")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Task parameters")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Task context")
    complexity_level: ComplexityLevel = Field(default=ComplexityLevel.MODERATE, description="Task complexity level")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Task priority")


class AgentScoreResponse(BaseModel):
    """Response model for agent scores."""
    agent_type: str
    total_score: float
    capability_score: float
    performance_score: float
    availability_score: float
    context_score: float
    confidence: float
    reasoning: List[str]


class AgentSelectionResponse(BaseModel):
    """Response model for agent selection results."""
    selected_agents: List[str]
    agent_scores: List[AgentScoreResponse]
    execution_plan: Dict[str, Any]
    resource_allocation: Dict[str, Any]
    estimated_completion_time_seconds: float
    confidence_level: float
    fallback_options: List[str]
    selection_reasoning: List[str]
    requirements: Dict[str, Any]


class WorkflowCreationResponse(BaseModel):
    """Response model for workflow creation."""
    workflow_context: Dict[str, Any]
    agent_selection: Dict[str, Any]


class TaskAdditionResponse(BaseModel):
    """Response model for task addition."""
    task_id: str
    selected_agent: str
    selection_confidence: float
    fallback_options: List[str]
    task_details: Dict[str, Any]


class PerformanceUpdateRequest(BaseModel):
    """Request model for updating agent performance."""
    agent_type: AgentType = Field(..., description="Agent type to update")
    execution_time_ms: int = Field(..., ge=1, description="Execution time in milliseconds")
    success: bool = Field(..., description="Whether the execution was successful")
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Quality score (0-1)")


# API Endpoints

@router.post("/select", response_model=AgentSelectionResponse)
async def select_agents_intelligently(
    request: AgentSelectionRequest,
    current_user: User = Depends(get_current_user),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """
    Intelligently select agents based on task requirements.
    
    Uses context-aware selection algorithms to find the optimal combination
    of agents for the specified task context and requirements.
    """
    try:
        max_execution_time = timedelta(minutes=request.max_execution_time_minutes)
        
        result = await orchestrator.select_agents_intelligently(
            task_context=request.task_context,
            required_capabilities=request.required_capabilities,
            complexity_level=request.complexity_level,
            estimated_tokens=request.estimated_tokens,
            max_execution_time=max_execution_time,
            max_agents=request.max_agents,
            domain_knowledge=request.domain_knowledge,
            preferred_agents=request.preferred_agents,
            excluded_agents=request.excluded_agents
        )
        
        logger.info(
            f"Agent selection completed for user {current_user.id}",
            extra={
                "selected_agents": result["selected_agents"],
                "confidence": result["confidence_level"],
                "task_context": request.task_context
            }
        )
        
        return AgentSelectionResponse(**result)
        
    except Exception as e:
        logger.error(f"Agent selection failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to select agents: {str(e)}"
        )


@router.post("/workflows/create", response_model=WorkflowCreationResponse)
async def create_intelligent_workflow(
    request: WorkflowCreationRequest,
    current_user: User = Depends(get_current_user),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """
    Create a workflow with intelligently selected agents.
    
    Combines agent selection and workflow creation into a single operation
    for streamlined multi-agent orchestration setup.
    """
    try:
        max_execution_time = timedelta(minutes=request.max_execution_time_minutes)
        
        result = await orchestrator.create_intelligent_workflow(
            user_id=str(current_user.id),
            task_context=request.task_context,
            required_capabilities=request.required_capabilities,
            project_id=request.project_id,
            complexity_level=request.complexity_level,
            estimated_tokens=request.estimated_tokens,
            max_execution_time=max_execution_time,
            max_agents=request.max_agents,
            domain_knowledge=request.domain_knowledge,
            initial_context=request.initial_context
        )
        
        logger.info(
            f"Intelligent workflow created for user {current_user.id}",
            extra={
                "workflow_id": result["workflow_context"]["workflow_id"],
                "selected_agents": result["agent_selection"]["selected_agents"],
                "task_context": request.task_context
            }
        )
        
        return WorkflowCreationResponse(**result)
        
    except Exception as e:
        logger.error(f"Intelligent workflow creation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create intelligent workflow: {str(e)}"
        )


@router.post("/workflows/tasks/add", response_model=TaskAdditionResponse)
async def add_intelligent_task(
    request: TaskAdditionRequest,
    current_user: User = Depends(get_current_user),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """
    Add a task to a workflow with intelligent agent selection.
    
    Selects the best agent for the specific task and adds it to the workflow
    with appropriate context and fallback options.
    """
    try:
        result = await orchestrator.add_intelligent_task(
            workflow_id=request.workflow_id,
            operation=request.operation,
            task_context=request.task_context,
            required_capabilities=request.required_capabilities,
            parameters=request.parameters,
            context=request.context,
            complexity_level=request.complexity_level,
            priority=request.priority
        )
        
        logger.info(
            f"Intelligent task added for user {current_user.id}",
            extra={
                "task_id": result["task_id"],
                "workflow_id": request.workflow_id,
                "selected_agent": result["selected_agent"],
                "operation": request.operation
            }
        )
        
        return TaskAdditionResponse(**result)
        
    except ValueError as e:
        logger.warning(f"Task addition failed - no suitable agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Intelligent task addition failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add intelligent task: {str(e)}"
        )


@router.post("/performance/update", response_model=Dict[str, str])
async def update_agent_performance(
    request: PerformanceUpdateRequest,
    current_user: User = Depends(get_current_user),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """
    Update agent performance metrics after task execution.
    
    Updates the agent's historical performance data to improve future
    selection decisions and track system effectiveness.
    """
    try:
        await orchestrator.update_agent_performance(
            agent_type=request.agent_type,
            execution_time_ms=request.execution_time_ms,
            success=request.success,
            quality_score=request.quality_score
        )
        
        logger.info(
            f"Agent performance updated for user {current_user.id}",
            extra={
                "agent_type": request.agent_type,
                "success": request.success,
                "execution_time": request.execution_time_ms,
                "quality_score": request.quality_score
            }
        )
        
        return {
            "status": "success",
            "message": f"Performance metrics updated for agent {request.agent_type.value}"
        }
        
    except Exception as e:
        logger.error(f"Performance update failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update agent performance: {str(e)}"
        )


@router.get("/capabilities", response_model=List[str])
async def list_available_capabilities(
    current_user: User = Depends(get_current_user)
):
    """
    List all available agent capabilities.
    
    Returns the complete list of capability types that can be used
    for agent selection requirements.
    """
    try:
        capabilities = [capability.value for capability in CapabilityType]
        
        logger.info(f"Capabilities listed for user {current_user.id}")
        
        return capabilities
        
    except Exception as e:
        logger.error(f"Failed to list capabilities: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list capabilities: {str(e)}"
        )


@router.get("/contexts", response_model=List[str])
async def list_available_task_contexts(
    current_user: User = Depends(get_current_user)
):
    """
    List all available task contexts.
    
    Returns the complete list of task context types that can be used
    for context-aware agent selection.
    """
    try:
        contexts = [context.value for context in TaskContext]
        
        logger.info(f"Task contexts listed for user {current_user.id}")
        
        return contexts
        
    except Exception as e:
        logger.error(f"Failed to list task contexts: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list task contexts: {str(e)}"
        )


@router.get("/agents", response_model=List[str])
async def list_available_agent_types(
    current_user: User = Depends(get_current_user)
):
    """
    List all available agent types.
    
    Returns the complete list of agent types that can be selected
    for workflow execution.
    """
    try:
        agent_types = [agent_type.value for agent_type in AgentType]
        
        logger.info(f"Agent types listed for user {current_user.id}")
        
        return agent_types
        
    except Exception as e:
        logger.error(f"Failed to list agent types: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list agent types: {str(e)}"
        )


@router.get("/complexity-levels", response_model=List[str])
async def list_complexity_levels(
    current_user: User = Depends(get_current_user)
):
    """
    List all available complexity levels.
    
    Returns the complete list of complexity levels that can be used
    for task requirements specification.
    """
    try:
        complexity_levels = [level.value for level in ComplexityLevel]
        
        logger.info(f"Complexity levels listed for user {current_user.id}")
        
        return complexity_levels
        
    except Exception as e:
        logger.error(f"Failed to list complexity levels: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list complexity levels: {str(e)}"
        )


@router.get("/health", response_model=Dict[str, Any])
async def agent_selection_health_check():
    """
    Health check endpoint for agent selection service.
    
    Provides status information about the agent selection system
    and its dependent services.
    """
    try:
        # Get selector instance to check initialization
        selector = await get_context_aware_selector()
        orchestrator = await get_orchestrator()
        
        orchestrator_health = await orchestrator.health_check()
        
        return {
            "status": "healthy",
            "service": "agent-selection",
            "timestamp": datetime.utcnow().isoformat(),
            "selector_initialized": selector.is_initialized,
            "orchestrator_status": orchestrator_health.get("status", "unknown"),
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "agent-selection",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "version": "1.0.0"
        }