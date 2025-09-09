"""
PRD Generation API endpoints
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from core.config import get_settings
from services.graphrag.graph_service import GraphRAGService
from services.pydantic_agents.prd_agent import PRDCreationAgent, PRDInput, PRDResult

logger = structlog.get_logger(__name__)
settings = get_settings()

router = APIRouter()


# Request/Response Models
class Phase0Request(BaseModel):
    """Phase 0: Project invitation input."""
    initial_description: str = Field(..., min_length=50, max_length=2000, description="Project concept description")
    user_id: str = Field(..., description="User identifier")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")


class ClarificationQuestion(BaseModel):
    """Generated clarification question."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    question: str = Field(..., description="Clarification question text")
    category: str = Field(..., description="Question category")
    required: bool = Field(default=True, description="Is this question required?")
    help_text: Optional[str] = Field(None, description="Help text for the question")


class Phase0Response(BaseModel):
    """Phase 0 response with questions and similar projects."""
    prd_id: str = Field(..., description="Generated PRD session ID")
    questions: List[ClarificationQuestion] = Field(..., description="Clarification questions")
    similar_projects: List[Dict[str, Any]] = Field(default_factory=list, description="Similar existing projects")
    concepts: List[str] = Field(default_factory=list, description="Extracted concepts")


class Phase1Request(BaseModel):
    """Phase 1: Objective clarification input."""
    prd_id: str = Field(..., description="PRD session ID")
    answers: Dict[str, str] = Field(..., description="Answers to clarification questions")


class ValidationSummary(BaseModel):
    """Validation summary for phase responses."""
    question_id: str = Field(..., description="Question ID")
    answer: str = Field(..., description="User answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Validation confidence")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")


class Phase1Response(BaseModel):
    """Phase 1 response with validation results."""
    prd_id: str = Field(..., description="PRD session ID")
    validations: List[ValidationSummary] = Field(..., description="Answer validations")
    ready_for_phase2: bool = Field(..., description="Ready to proceed to next phase")
    overall_confidence: float = Field(..., description="Overall validation confidence")


class Phase2Request(BaseModel):
    """Phase 2: Objective drafting input."""
    prd_id: str = Field(..., description="PRD session ID") 
    user_refinements: Optional[str] = Field(None, description="User refinements to objectives")


class ObjectiveDraft(BaseModel):
    """Generated objective draft."""
    content: str = Field(..., description="Objective text")
    confidence: float = Field(..., description="GraphRAG validation confidence")
    is_smart: bool = Field(..., description="Meets SMART criteria")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")


class Phase2Response(BaseModel):
    """Phase 2 response with objective draft."""
    prd_id: str = Field(..., description="PRD session ID")
    objective: ObjectiveDraft = Field(..., description="Generated objective")
    ready_for_phase3: bool = Field(..., description="Ready for section creation")


class PRDGenerationRequest(BaseModel):
    """Complete PRD generation request."""
    title: str = Field(..., min_length=10, max_length=200)
    description: str = Field(..., min_length=100)
    business_context: Optional[str] = None
    target_audience: Optional[str] = None
    success_criteria: Optional[List[str]] = None
    constraints: Optional[List[str]] = None
    priority: str = Field(default="medium", regex="^(low|medium|high|critical)$")
    user_id: str = Field(..., description="User identifier")


class PRDGenerationResponse(BaseModel):
    """Complete PRD generation response."""
    prd_id: str = Field(..., description="Generated PRD ID")
    title: str = Field(..., description="PRD title")
    status: str = Field(..., description="Generation status")
    quality_score: float = Field(..., description="Overall quality score")
    sections_count: int = Field(..., description="Number of sections generated")
    validation_summary: Dict[str, Any] = Field(..., description="Validation results summary")
    estimated_completion_time: Optional[str] = Field(None, description="Estimated completion time")


# Dependency injection
async def get_graphrag_service() -> GraphRAGService:
    """Get GraphRAG service from app state."""
    # This would typically come from FastAPI app state
    # For now, create a new instance (in production, use dependency injection)
    service = GraphRAGService()
    if not service.is_initialized:
        await service.initialize()
    return service


async def get_prd_agent() -> PRDCreationAgent:
    """Get PRD creation agent."""
    agent = PRDCreationAgent()
    graphrag_service = await get_graphrag_service()
    await agent.initialize(graphrag_service)
    return agent


# Phase 0: Project Invitation
@router.post("/phase0/initiate", response_model=Phase0Response)
async def initiate_prd(
    request: Phase0Request,
    prd_agent: PRDCreationAgent = Depends(get_prd_agent)
):
    """Phase 0: Process initial project description and generate clarification questions."""
    try:
        logger.info(
            "Starting Phase 0 PRD initiation",
            user_id=request.user_id,
            description_length=len(request.initial_description)
        )
        
        # Generate PRD session ID
        prd_id = str(uuid4())
        
        # Extract concepts from description
        concepts = await extract_concepts(request.initial_description)
        
        # Find similar projects
        similar_projects = await find_similar_projects(concepts, prd_agent)
        
        # Generate clarification questions
        questions = await generate_clarifying_questions(
            request.initial_description, 
            similar_projects,
            prd_agent
        )
        
        # Store phase 0 data (implement session storage)
        await store_phase0_data(prd_id, request, concepts, questions)
        
        logger.info(
            "Phase 0 completed successfully",
            prd_id=prd_id,
            questions_generated=len(questions),
            similar_projects=len(similar_projects)
        )
        
        return Phase0Response(
            prd_id=prd_id,
            questions=questions,
            similar_projects=similar_projects,
            concepts=concepts
        )
        
    except Exception as e:
        logger.error("Phase 0 initiation failed", error=str(e), user_id=request.user_id)
        raise HTTPException(status_code=500, detail=f"Failed to initiate PRD: {str(e)}")


# Phase 1: Objective Clarification
@router.post("/phase1/clarify", response_model=Phase1Response)
async def clarify_objectives(
    request: Phase1Request,
    graphrag_service: GraphRAGService = Depends(get_graphrag_service)
):
    """Phase 1: Process clarification answers and validate with GraphRAG."""
    try:
        logger.info(
            "Starting Phase 1 objective clarification",
            prd_id=request.prd_id,
            answers_count=len(request.answers)
        )
        
        validations = []
        confidence_scores = []
        
        # Validate each answer against GraphRAG
        for question_id, answer in request.answers.items():
            validation_result = await graphrag_service.validate_content(
                answer, 
                {"phase": "clarification", "question_id": question_id}
            )
            
            suggestions = []
            if validation_result.confidence < 0.8:
                suggestions = validation_result.corrections
            
            validations.append(ValidationSummary(
                question_id=question_id,
                answer=answer,
                confidence=validation_result.confidence,
                suggestions=suggestions
            ))
            
            confidence_scores.append(validation_result.confidence)
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        ready_for_phase2 = overall_confidence >= 0.7
        
        # Store phase 1 data
        await store_phase1_data(request.prd_id, validations, overall_confidence)
        
        logger.info(
            "Phase 1 completed successfully", 
            prd_id=request.prd_id,
            overall_confidence=overall_confidence,
            ready_for_phase2=ready_for_phase2
        )
        
        return Phase1Response(
            prd_id=request.prd_id,
            validations=validations,
            ready_for_phase2=ready_for_phase2,
            overall_confidence=overall_confidence
        )
        
    except Exception as e:
        logger.error("Phase 1 clarification failed", error=str(e), prd_id=request.prd_id)
        raise HTTPException(status_code=500, detail=f"Failed to process clarifications: {str(e)}")


# Phase 2: Objective Drafting
@router.post("/phase2/draft", response_model=Phase2Response)
async def draft_objectives(
    request: Phase2Request,
    graphrag_service: GraphRAGService = Depends(get_graphrag_service)
):
    """Phase 2: Generate and validate SMART objectives."""
    try:
        logger.info("Starting Phase 2 objective drafting", prd_id=request.prd_id)
        
        # Retrieve phase 1 data
        phase1_data = await get_phase1_data(request.prd_id)
        if not phase1_data:
            raise HTTPException(status_code=404, detail="Phase 1 data not found")
        
        # Generate SMART objective
        objective_text = await generate_smart_objective(phase1_data, request.user_refinements)
        
        # Validate with GraphRAG
        validation_result = await graphrag_service.validate_content(
            objective_text,
            {"phase": "objectives", "prd_id": request.prd_id}
        )
        
        # Check SMART criteria
        is_smart = await validate_smart_criteria(objective_text)
        
        objective = ObjectiveDraft(
            content=objective_text,
            confidence=validation_result.confidence,
            is_smart=is_smart,
            suggestions=validation_result.corrections if validation_result.confidence < 0.8 else []
        )
        
        # Store phase 2 data
        await store_phase2_data(request.prd_id, objective)
        
        ready_for_phase3 = objective.confidence >= 0.8 and is_smart
        
        logger.info(
            "Phase 2 completed successfully",
            prd_id=request.prd_id,
            confidence=objective.confidence,
            is_smart=is_smart
        )
        
        return Phase2Response(
            prd_id=request.prd_id,
            objective=objective,
            ready_for_phase3=ready_for_phase3
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Phase 2 drafting failed", error=str(e), prd_id=request.prd_id)
        raise HTTPException(status_code=500, detail=f"Failed to draft objectives: {str(e)}")


# Complete PRD Generation
@router.post("/generate", response_model=PRDGenerationResponse)
async def generate_complete_prd(
    request: PRDGenerationRequest,
    background_tasks: BackgroundTasks,
    prd_agent: PRDCreationAgent = Depends(get_prd_agent)
):
    """Generate a complete PRD using PydanticAI agent."""
    try:
        logger.info(
            "Starting complete PRD generation",
            title=request.title,
            user_id=request.user_id
        )
        
        # Convert to PRD input
        prd_input = PRDInput(
            title=request.title,
            description=request.description,
            business_context=request.business_context,
            target_audience=request.target_audience,
            success_criteria=request.success_criteria,
            constraints=request.constraints,
            priority=request.priority
        )
        
        # Generate PRD
        prd_result = await prd_agent.create_prd(
            prd_input=prd_input,
            user_id=request.user_id,
            context={"generation_mode": "complete"}
        )
        
        # Calculate validation summary
        validation_summary = {
            "overall_confidence": prd_result.overall_quality_score / 10,
            "sections_validated": len(prd_result.sections),
            "validation_details": prd_result.validation_results
        }
        
        # Add background tasks for post-processing
        background_tasks.add_task(
            post_process_prd,
            prd_result.id,
            request.user_id
        )
        
        logger.info(
            "PRD generation completed successfully",
            prd_id=prd_result.id,
            quality_score=prd_result.overall_quality_score,
            sections_count=len(prd_result.sections)
        )
        
        return PRDGenerationResponse(
            prd_id=prd_result.id,
            title=prd_result.title,
            status=prd_result.status,
            quality_score=prd_result.overall_quality_score,
            sections_count=len(prd_result.sections),
            validation_summary=validation_summary,
            estimated_completion_time="5-10 minutes"
        )
        
    except Exception as e:
        logger.error("Complete PRD generation failed", error=str(e), user_id=request.user_id)
        raise HTTPException(status_code=500, detail=f"Failed to generate PRD: {str(e)}")


# Helper Functions (simplified implementations)
async def extract_concepts(description: str) -> List[str]:
    """Extract key concepts from project description."""
    # Simplified concept extraction (in production, use NLP)
    words = description.lower().split()
    concepts = [word for word in words if len(word) > 4]
    return list(set(concepts))[:10]  # Return top 10 unique concepts


async def find_similar_projects(concepts: List[str], prd_agent: PRDCreationAgent) -> List[Dict[str, Any]]:
    """Find similar existing projects."""
    try:
        # Use the agent's research tool
        similar_prds = []
        for concept in concepts[:3]:  # Check top 3 concepts
            results = await prd_agent.agent.run_tool(
                "research_similar_prds",
                query=concept,
                limit=2
            )
            if hasattr(results, 'data'):
                similar_prds.extend(results.data)
        
        return similar_prds[:5]  # Return top 5 unique results
    except Exception as e:
        logger.warning("Similar project search failed", error=str(e))
        return []


async def generate_clarifying_questions(
    description: str, 
    similar_projects: List[Dict[str, Any]],
    prd_agent: PRDCreationAgent
) -> List[ClarificationQuestion]:
    """Generate AI-powered clarifying questions."""
    
    # Standard questions based on PRD best practices
    questions = [
        ClarificationQuestion(
            question="What specific business problem does this project solve?",
            category="business_context",
            help_text="Describe the pain point or opportunity this addresses"
        ),
        ClarificationQuestion(
            question="Who are the primary users or stakeholders?",
            category="target_audience", 
            help_text="Be specific about user personas and their needs"
        ),
        ClarificationQuestion(
            question="How will you measure success for this project?",
            category="success_metrics",
            help_text="Include quantifiable metrics and timeframes"
        ),
        ClarificationQuestion(
            question="What are the key technical or business constraints?",
            category="constraints",
            required=False,
            help_text="Budget, timeline, technical limitations, etc."
        ),
        ClarificationQuestion(
            question="What is the expected timeline and priority level?",
            category="timeline_priority",
            help_text="When do you need this delivered and how critical is it?"
        )
    ]
    
    return questions


# Session storage functions (implement with Redis or database)
async def store_phase0_data(prd_id: str, request: Phase0Request, concepts: List[str], questions: List[ClarificationQuestion]) -> None:
    """Store Phase 0 data for session management."""
    # Implementation would store in Redis or database
    pass


async def store_phase1_data(prd_id: str, validations: List[ValidationSummary], confidence: float) -> None:
    """Store Phase 1 validation data."""
    pass


async def get_phase1_data(prd_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve Phase 1 data for Phase 2."""
    # Mock implementation
    return {"answers": {}, "confidence": 0.8}


async def store_phase2_data(prd_id: str, objective: ObjectiveDraft) -> None:
    """Store Phase 2 objective data."""
    pass


async def generate_smart_objective(phase1_data: Dict[str, Any], refinements: Optional[str] = None) -> str:
    """Generate SMART objective from clarification data."""
    # Simplified implementation
    return "Develop a comprehensive project management solution that increases team productivity by 25% within 6 months through automated task tracking and intelligent resource allocation."


async def validate_smart_criteria(objective: str) -> bool:
    """Validate if objective meets SMART criteria."""
    # Simplified validation (in production, use NLP analysis)
    smart_indicators = ["increase", "improve", "reduce", "within", "by", "%", "month", "year"]
    return any(indicator in objective.lower() for indicator in smart_indicators)


async def post_process_prd(prd_id: str, user_id: str) -> None:
    """Background task for PRD post-processing."""
    try:
        # Add analytics, notifications, etc.
        logger.info("PRD post-processing completed", prd_id=prd_id, user_id=user_id)
    except Exception as e:
        logger.error("PRD post-processing failed", error=str(e), prd_id=prd_id)