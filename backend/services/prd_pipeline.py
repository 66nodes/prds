"""
PRD Generation Pipeline - Orchestrates the 4-phase PRD creation workflow.

This service coordinates the entire PRD generation process from initial concept
to final validated document using the agent orchestration system.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
import uuid
import json

import structlog
from pydantic import BaseModel, Field

from services.agent_orchestrator import (
    AgentOrchestrator, 
    AgentType, 
    TaskPriority,
    get_orchestrator
)
from services.hybrid_rag import HybridRAGService
from models.prd import (
    PRDPhase, 
    PRDStatus,
    Phase0Request,
    Phase0Response,
    Phase1Request, 
    Phase1Response,
    ClarificationQuestion
)
from core.config import get_settings
from core.database import get_neo4j

logger = structlog.get_logger(__name__)
settings = get_settings()


class ValidationStage(str, Enum):
    """PRD validation stages."""
    PHASE_VALIDATION = "phase_validation"
    CONTENT_VALIDATION = "content_validation" 
    COMPREHENSIVE_REVIEW = "comprehensive_review"
    FINAL_VALIDATION = "final_validation"


class PRDGenerationStatus(str, Enum):
    """PRD generation status."""
    INITIALIZING = "initializing"
    PHASE_0_IN_PROGRESS = "phase_0_in_progress"
    PHASE_0_COMPLETE = "phase_0_complete"
    AWAITING_STAKEHOLDER_INPUT = "awaiting_stakeholder_input"
    PHASE_1_IN_PROGRESS = "phase_1_in_progress"
    PHASE_1_COMPLETE = "phase_1_complete"
    PHASE_2_IN_PROGRESS = "phase_2_in_progress"
    PHASE_2_COMPLETE = "phase_2_complete"
    PHASE_3_IN_PROGRESS = "phase_3_in_progress"
    PHASE_3_COMPLETE = "phase_3_complete"
    PHASE_4_IN_PROGRESS = "phase_4_in_progress"
    FINAL_REVIEW = "final_review"
    COMPLETED = "completed"
    FAILED = "failed"


class PRDGenerationRequest(BaseModel):
    """Request model for PRD generation pipeline."""
    user_id: str = Field(..., description="User initiating the PRD generation")
    project_id: Optional[str] = Field(None, description="Associated project ID")
    initial_description: str = Field(..., min_length=50, description="Initial project concept")
    complexity_level: str = Field(default="medium", description="Project complexity level")
    priority: str = Field(default="medium", description="Generation priority")
    validation_level: str = Field(default="standard", description="Validation strictness")
    include_implementation_plan: bool = Field(default=True, description="Include implementation planning")
    target_completion_hours: int = Field(default=24, description="Target completion time in hours")


class PRDGenerationResult(BaseModel):
    """Result model for PRD generation pipeline."""
    prd_id: str = Field(..., description="Generated PRD identifier")
    status: PRDGenerationStatus = Field(..., description="Generation status")
    current_phase: PRDPhase = Field(..., description="Current PRD phase")
    quality_score: float = Field(..., ge=0.0, le=10.0, description="Overall quality score")
    validation_results: List[Dict[str, Any]] = Field(default_factory=list, description="Validation results")
    generated_content: Dict[str, Any] = Field(default_factory=dict, description="Generated PRD content")
    stakeholder_questions: List[ClarificationQuestion] = Field(default_factory=list, description="Questions for stakeholders")
    recommendations: List[str] = Field(default_factory=list, description="Process recommendations")
    estimated_completion_time: Optional[str] = Field(None, description="Estimated completion time")
    next_steps: List[str] = Field(default_factory=list, description="Required next steps")


class PRDGenerationPipeline:
    """
    Comprehensive PRD generation pipeline implementing the 4-phase workflow:
    
    Phase 0: Project invitation and clarification questions
    Phase 1: Objective clarification and validation
    Phase 2: Detailed requirement drafting
    Phase 3: Section co-creation and enhancement  
    Phase 4: Final synthesis and validation
    """
    
    def __init__(self):
        self.orchestrator: Optional[AgentOrchestrator] = None
        self.hybrid_rag = HybridRAGService()
        self.neo4j = None
        self.active_generations: Dict[str, Dict[str, Any]] = {}
        self.is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the PRD generation pipeline."""
        try:
            self.orchestrator = await get_orchestrator()
            await self.hybrid_rag.initialize()
            self.neo4j = await get_neo4j()
            
            self.is_initialized = True
            logger.info("PRD generation pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PRD generation pipeline: {str(e)}")
            raise
    
    async def start_prd_generation(
        self, 
        request: PRDGenerationRequest
    ) -> PRDGenerationResult:
        """Start the complete PRD generation process."""
        
        if not self.is_initialized:
            raise RuntimeError("PRD generation pipeline not initialized")
        
        prd_id = str(uuid.uuid4())
        
        try:
            # Create workflow context for this PRD generation
            workflow_context = await self.orchestrator.create_workflow(
                user_id=request.user_id,
                project_id=request.project_id,
                initial_context={
                    "prd_generation_request": request.model_dump(),
                    "complexity_level": request.complexity_level,
                    "validation_level": request.validation_level
                }
            )
            
            # Store generation state
            self.active_generations[prd_id] = {
                "workflow_id": workflow_context.workflow_id,
                "request": request,
                "status": PRDGenerationStatus.INITIALIZING,
                "current_phase": PRDPhase.PHASE_0,
                "start_time": datetime.utcnow(),
                "quality_scores": [],
                "validation_history": []
            }
            
            # Start with Phase 0
            phase0_result = await self._execute_phase_0(prd_id, request)
            
            result = PRDGenerationResult(
                prd_id=prd_id,
                status=PRDGenerationStatus.PHASE_0_COMPLETE,
                current_phase=PRDPhase.PHASE_0,
                quality_score=phase0_result.get("quality_score", 7.0),
                validation_results=phase0_result.get("validation_results", []),
                generated_content=phase0_result.get("generated_content", {}),
                stakeholder_questions=phase0_result.get("stakeholder_questions", []),
                recommendations=phase0_result.get("recommendations", []),
                estimated_completion_time=self._estimate_completion_time(request),
                next_steps=["Complete stakeholder questions to proceed to Phase 1"]
            )
            
            logger.info(
                "PRD generation started",
                prd_id=prd_id,
                user_id=request.user_id,
                complexity=request.complexity_level
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to start PRD generation: {str(e)}", prd_id=prd_id)
            raise
    
    async def _execute_phase_0(
        self, 
        prd_id: str, 
        request: PRDGenerationRequest
    ) -> Dict[str, Any]:
        """Execute Phase 0: Project invitation and clarification questions."""
        
        generation_state = self.active_generations[prd_id]
        generation_state["status"] = PRDGenerationStatus.PHASE_0_IN_PROGRESS
        
        try:
            # Add Phase 0 tasks to workflow
            analysis_task = await self.orchestrator.add_task(
                workflow_id=generation_state["workflow_id"],
                agent_type=AgentType.PRD_GENERATOR,
                operation="analyze_initial_requirements",
                parameters={
                    "initial_description": request.initial_description,
                    "user_context": {
                        "user_id": request.user_id,
                        "complexity_level": request.complexity_level,
                        "validation_level": request.validation_level
                    }
                },
                priority=TaskPriority.HIGH
            )
            
            phase0_task = await self.orchestrator.add_task(
                workflow_id=generation_state["workflow_id"],
                agent_type=AgentType.PRD_GENERATOR,
                operation="generate_phase0", 
                parameters={
                    "initial_description": request.initial_description
                },
                dependencies=[analysis_task.id],
                priority=TaskPriority.HIGH
            )
            
            validation_task = await self.orchestrator.add_task(
                workflow_id=generation_state["workflow_id"],
                agent_type=AgentType.JUDGE_AGENT,
                operation="validate_prd_phase",
                parameters={
                    "phase": "phase_0"
                },
                dependencies=[phase0_task.id],
                priority=TaskPriority.HIGH
            )
            
            # Execute workflow
            execution_result = await self.orchestrator.execute_workflow(
                workflow_id=generation_state["workflow_id"],
                parallel_execution=False,  # Sequential for PRD phases
                max_concurrent_tasks=1
            )
            
            if not execution_result.get("successful_tasks", 0) >= 2:
                raise RuntimeError("Phase 0 execution failed")
            
            # Process results
            phase0_content = execution_result["results"].get(phase0_task.id, {}).get("result", {})
            validation_result = execution_result["results"].get(validation_task.id, {}).get("result", {})
            
            # Extract clarification questions
            stakeholder_questions = self._extract_clarification_questions(phase0_content)
            
            # Calculate quality score
            quality_score = validation_result.get("phase_validation", {}).get("overall_score", 7.0)
            
            result = {
                "generated_content": phase0_content,
                "validation_results": [validation_result],
                "stakeholder_questions": stakeholder_questions,
                "quality_score": quality_score,
                "recommendations": self._generate_phase0_recommendations(phase0_content, validation_result)
            }
            
            # Store validation results
            await self._store_phase_validation(prd_id, PRDPhase.PHASE_0, result)
            
            return result
            
        except Exception as e:
            generation_state["status"] = PRDGenerationStatus.FAILED
            logger.error(f"Phase 0 execution failed: {str(e)}", prd_id=prd_id)
            raise
    
    async def process_phase_1(
        self,
        prd_id: str,
        stakeholder_answers: Dict[str, str]
    ) -> PRDGenerationResult:
        """Process Phase 1: Objective clarification and validation."""
        
        if prd_id not in self.active_generations:
            raise ValueError(f"PRD generation {prd_id} not found")
        
        generation_state = self.active_generations[prd_id]
        generation_state["status"] = PRDGenerationStatus.PHASE_1_IN_PROGRESS
        
        try:
            # Get Phase 0 content
            phase0_content = await self._get_phase_content(prd_id, PRDPhase.PHASE_0)
            
            # Add Phase 1 tasks
            phase1_task = await self.orchestrator.add_task(
                workflow_id=generation_state["workflow_id"],
                agent_type=AgentType.PRD_GENERATOR,
                operation="generate_phase1",
                parameters={
                    "phase0_content": phase0_content,
                    "stakeholder_answers": stakeholder_answers
                },
                priority=TaskPriority.HIGH
            )
            
            validation_task = await self.orchestrator.add_task(
                workflow_id=generation_state["workflow_id"],
                agent_type=AgentType.JUDGE_AGENT,
                operation="validate_prd_phase",
                parameters={
                    "phase": "phase_1"
                },
                dependencies=[phase1_task.id],
                priority=TaskPriority.HIGH
            )
            
            # Execute workflow
            execution_result = await self.orchestrator.execute_workflow(
                workflow_id=generation_state["workflow_id"],
                parallel_execution=False,
                max_concurrent_tasks=1
            )
            
            if not execution_result.get("successful_tasks", 0) >= 1:
                raise RuntimeError("Phase 1 execution failed")
            
            # Process results
            phase1_content = execution_result["results"].get(phase1_task.id, {}).get("result", {})
            validation_result = execution_result["results"].get(validation_task.id, {}).get("result", {})
            
            quality_score = validation_result.get("phase_validation", {}).get("overall_score", 7.0)
            ready_for_phase2 = validation_result.get("passes_validation", False) and quality_score >= 7.0
            
            # Update generation state
            generation_state["status"] = PRDGenerationStatus.PHASE_1_COMPLETE if ready_for_phase2 else PRDGenerationStatus.PHASE_1_IN_PROGRESS
            generation_state["current_phase"] = PRDPhase.PHASE_1
            generation_state["quality_scores"].append(quality_score)
            
            result = PRDGenerationResult(
                prd_id=prd_id,
                status=generation_state["status"],
                current_phase=PRDPhase.PHASE_1,
                quality_score=quality_score,
                validation_results=[validation_result],
                generated_content=phase1_content,
                recommendations=self._generate_phase1_recommendations(phase1_content, validation_result),
                next_steps=["Proceed to Phase 2: Detailed requirements"] if ready_for_phase2 else ["Improve Phase 1 content based on validation feedback"]
            )
            
            # Store validation results
            await self._store_phase_validation(prd_id, PRDPhase.PHASE_1, result.model_dump())
            
            return result
            
        except Exception as e:
            generation_state["status"] = PRDGenerationStatus.FAILED
            logger.error(f"Phase 1 processing failed: {str(e)}", prd_id=prd_id)
            raise
    
    async def continue_to_phase_2(self, prd_id: str) -> PRDGenerationResult:
        """Continue to Phase 2: Detailed requirement drafting."""
        
        if prd_id not in self.active_generations:
            raise ValueError(f"PRD generation {prd_id} not found")
        
        generation_state = self.active_generations[prd_id]
        
        if generation_state["status"] != PRDGenerationStatus.PHASE_1_COMPLETE:
            raise ValueError("Phase 1 must be completed before proceeding to Phase 2")
        
        generation_state["status"] = PRDGenerationStatus.PHASE_2_IN_PROGRESS
        
        try:
            # Get previous phase content
            phase1_content = await self._get_phase_content(prd_id, PRDPhase.PHASE_1)
            
            # Add Phase 2 tasks
            phase2_task = await self.orchestrator.add_task(
                workflow_id=generation_state["workflow_id"],
                agent_type=AgentType.PRD_GENERATOR,
                operation="generate_phase2",
                parameters={
                    "phase1_content": phase1_content
                },
                priority=TaskPriority.HIGH
            )
            
            validation_task = await self.orchestrator.add_task(
                workflow_id=generation_state["workflow_id"],
                agent_type=AgentType.JUDGE_AGENT,
                operation="comprehensive_quality_review",
                parameters={
                    "content_type": "detailed_requirements"
                },
                dependencies=[phase2_task.id],
                priority=TaskPriority.HIGH
            )
            
            # Execute workflow
            execution_result = await self.orchestrator.execute_workflow(
                workflow_id=generation_state["workflow_id"],
                parallel_execution=False,
                max_concurrent_tasks=1
            )
            
            # Process results
            phase2_content = execution_result["results"].get(phase2_task.id, {}).get("result", {})
            validation_result = execution_result["results"].get(validation_task.id, {}).get("result", {})
            
            quality_score = validation_result.get("comprehensive_review", {}).get("overall_score", 7.0)
            
            # Update state
            generation_state["status"] = PRDGenerationStatus.PHASE_2_COMPLETE
            generation_state["current_phase"] = PRDPhase.PHASE_2
            generation_state["quality_scores"].append(quality_score)
            
            result = PRDGenerationResult(
                prd_id=prd_id,
                status=PRDGenerationStatus.PHASE_2_COMPLETE,
                current_phase=PRDPhase.PHASE_2,
                quality_score=quality_score,
                validation_results=[validation_result],
                generated_content=phase2_content,
                recommendations=["Proceed to Phase 3 for section enhancement"],
                next_steps=["Continue to Phase 3: Section co-creation"]
            )
            
            await self._store_phase_validation(prd_id, PRDPhase.PHASE_2, result.model_dump())
            
            return result
            
        except Exception as e:
            generation_state["status"] = PRDGenerationStatus.FAILED  
            logger.error(f"Phase 2 processing failed: {str(e)}", prd_id=prd_id)
            raise
    
    async def complete_prd_generation(self, prd_id: str) -> PRDGenerationResult:
        """Complete Phase 3 and 4: Section co-creation and final synthesis."""
        
        if prd_id not in self.active_generations:
            raise ValueError(f"PRD generation {prd_id} not found")
        
        generation_state = self.active_generations[prd_id]
        
        if generation_state["status"] != PRDGenerationStatus.PHASE_2_COMPLETE:
            raise ValueError("Phase 2 must be completed before final completion")
        
        try:
            # Execute Phase 3
            generation_state["status"] = PRDGenerationStatus.PHASE_3_IN_PROGRESS
            phase3_result = await self._execute_phase_3(prd_id)
            
            # Execute Phase 4  
            generation_state["status"] = PRDGenerationStatus.PHASE_4_IN_PROGRESS
            phase4_result = await self._execute_phase_4(prd_id, phase3_result)
            
            # Final validation and quality review
            generation_state["status"] = PRDGenerationStatus.FINAL_REVIEW
            final_validation = await self._execute_final_validation(prd_id, phase4_result)
            
            # Calculate final quality score
            final_quality_score = self._calculate_final_quality_score(generation_state["quality_scores"], final_validation)
            
            # Mark as completed
            generation_state["status"] = PRDGenerationStatus.COMPLETED
            generation_state["completion_time"] = datetime.utcnow()
            
            result = PRDGenerationResult(
                prd_id=prd_id,
                status=PRDGenerationStatus.COMPLETED,
                current_phase=PRDPhase.PHASE_4,
                quality_score=final_quality_score,
                validation_results=[final_validation],
                generated_content=phase4_result,
                recommendations=["PRD generation completed successfully"],
                next_steps=["Review final PRD and proceed with implementation"]
            )
            
            # Store final PRD
            await self._store_final_prd(prd_id, result)
            
            logger.info(
                "PRD generation completed successfully",
                prd_id=prd_id,
                final_quality_score=final_quality_score,
                completion_time=generation_state["completion_time"]
            )
            
            return result
            
        except Exception as e:
            generation_state["status"] = PRDGenerationStatus.FAILED
            logger.error(f"PRD completion failed: {str(e)}", prd_id=prd_id)
            raise
    
    async def _execute_phase_3(self, prd_id: str) -> Dict[str, Any]:
        """Execute Phase 3: Section co-creation."""
        
        generation_state = self.active_generations[prd_id]
        phase2_content = await self._get_phase_content(prd_id, PRDPhase.PHASE_2)
        
        # Add Phase 3 task
        phase3_task = await self.orchestrator.add_task(
            workflow_id=generation_state["workflow_id"],
            agent_type=AgentType.PRD_GENERATOR,
            operation="generate_phase3",
            parameters={
                "phase2_content": phase2_content
            },
            priority=TaskPriority.HIGH
        )
        
        # Execute
        execution_result = await self.orchestrator.execute_workflow(
            workflow_id=generation_state["workflow_id"],
            parallel_execution=False
        )
        
        phase3_content = execution_result["results"].get(phase3_task.id, {}).get("result", {})
        return phase3_content
    
    async def _execute_phase_4(self, prd_id: str, phase3_content: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Phase 4: Final synthesis."""
        
        generation_state = self.active_generations[prd_id]
        
        # Add Phase 4 task
        phase4_task = await self.orchestrator.add_task(
            workflow_id=generation_state["workflow_id"],
            agent_type=AgentType.PRD_GENERATOR,
            operation="generate_phase4",
            parameters={
                "phase3_content": phase3_content
            },
            priority=TaskPriority.CRITICAL
        )
        
        # Execute
        execution_result = await self.orchestrator.execute_workflow(
            workflow_id=generation_state["workflow_id"],
            parallel_execution=False
        )
        
        phase4_content = execution_result["results"].get(phase4_task.id, {}).get("result", {})
        return phase4_content
    
    async def _execute_final_validation(self, prd_id: str, final_content: Dict[str, Any]) -> Dict[str, Any]:
        """Execute final comprehensive validation."""
        
        generation_state = self.active_generations[prd_id]
        
        # Add final validation task
        validation_task = await self.orchestrator.add_task(
            workflow_id=generation_state["workflow_id"],
            agent_type=AgentType.JUDGE_AGENT,
            operation="comprehensive_quality_review",
            parameters={
                "content": final_content,
                "content_type": "complete_prd",
                "review_scope": "comprehensive"
            },
            priority=TaskPriority.CRITICAL
        )
        
        # Execute
        execution_result = await self.orchestrator.execute_workflow(
            workflow_id=generation_state["workflow_id"],
            parallel_execution=False
        )
        
        validation_result = execution_result["results"].get(validation_task.id, {}).get("result", {})
        return validation_result
    
    def _extract_clarification_questions(self, phase0_content: Dict[str, Any]) -> List[ClarificationQuestion]:
        """Extract clarification questions from Phase 0 content."""
        
        questions = []
        phase0_data = phase0_content.get("phase0_content", {})
        
        # Try to extract questions from different possible structures
        question_data = phase0_data.get("questions", [])
        if isinstance(question_data, list):
            for i, q in enumerate(question_data):
                if isinstance(q, dict):
                    question = ClarificationQuestion(
                        id=q.get("id", f"q{i+1:03d}"),
                        question=q.get("question", str(q)),
                        category=q.get("category", "general"),
                        required=q.get("required", True),
                        help_text=q.get("help_text")
                    )
                    questions.append(question)
                else:
                    question = ClarificationQuestion(
                        id=f"q{i+1:03d}",
                        question=str(q),
                        category="general",
                        required=True
                    )
                    questions.append(question)
        
        return questions[:12]  # Limit to 12 questions max
    
    def _generate_phase0_recommendations(self, phase0_content: Dict[str, Any], validation_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on Phase 0 results."""
        
        recommendations = []
        
        quality_score = validation_result.get("phase_validation", {}).get("overall_score", 7.0)
        if quality_score < 8.0:
            recommendations.append("Consider refining project description for better clarity")
        
        questions = phase0_content.get("phase0_content", {}).get("questions", [])
        if len(questions) < 8:
            recommendations.append("Add more comprehensive clarification questions")
        
        recommendations.append("Engage stakeholders to answer clarification questions thoroughly")
        recommendations.append("Review similar projects for additional insights")
        
        return recommendations
    
    def _generate_phase1_recommendations(self, phase1_content: Dict[str, Any], validation_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on Phase 1 results."""
        
        recommendations = []
        
        quality_score = validation_result.get("phase_validation", {}).get("overall_score", 7.0)
        if quality_score < 8.0:
            recommendations.append("Improve objective clarity and measurability")
            
        recommendations.append("Validate requirements with key stakeholders")
        recommendations.append("Proceed to detailed requirements specification")
        
        return recommendations
    
    def _calculate_final_quality_score(self, phase_scores: List[float], final_validation: Dict[str, Any]) -> float:
        """Calculate final quality score from all phases."""
        
        # Weight: Phase scores (60%) + Final validation (40%)
        phase_avg = sum(phase_scores) / len(phase_scores) if phase_scores else 7.0
        final_score = final_validation.get("comprehensive_review", {}).get("overall_score", 7.0)
        
        weighted_score = (phase_avg * 0.6) + (final_score * 0.4)
        return round(weighted_score, 2)
    
    def _estimate_completion_time(self, request: PRDGenerationRequest) -> str:
        """Estimate completion time based on request parameters."""
        
        base_hours = {
            "low": 2,
            "medium": 6, 
            "high": 12
        }
        
        hours = base_hours.get(request.complexity_level, 6)
        
        if request.validation_level == "strict":
            hours *= 1.5
        
        return f"{hours}-{hours + 2} hours"
    
    async def _get_phase_content(self, prd_id: str, phase: PRDPhase) -> Dict[str, Any]:
        """Retrieve content from a specific phase."""
        
        try:
            # In a real implementation, this would query Neo4j for stored phase content
            # For now, return empty dict as placeholder
            return {}
            
        except Exception as e:
            logger.warning(f"Failed to retrieve phase content: {str(e)}")
            return {}
    
    async def _store_phase_validation(self, prd_id: str, phase: PRDPhase, validation_data: Dict[str, Any]) -> None:
        """Store phase validation results in Neo4j."""
        
        try:
            if not self.neo4j:
                return
                
            query = """
            CREATE (v:PhaseValidation {
                prd_id: $prd_id,
                phase: $phase,
                validation_data: $validation_data,
                timestamp: datetime(),
                quality_score: $quality_score
            })
            """
            
            parameters = {
                "prd_id": prd_id,
                "phase": phase.value,
                "validation_data": json.dumps(validation_data),
                "quality_score": validation_data.get("quality_score", 0.0)
            }
            
            await self.neo4j.execute_write(query, parameters)
            
        except Exception as e:
            logger.warning(f"Failed to store phase validation: {str(e)}")
    
    async def _store_final_prd(self, prd_id: str, result: PRDGenerationResult) -> None:
        """Store final PRD in Neo4j."""
        
        try:
            if not self.neo4j:
                return
                
            query = """
            CREATE (p:FinalPRD {
                id: $prd_id,
                status: $status,
                quality_score: $quality_score,
                generated_content: $content,
                completion_timestamp: datetime(),
                user_id: $user_id
            })
            """
            
            generation_state = self.active_generations[prd_id]
            parameters = {
                "prd_id": prd_id,
                "status": result.status.value,
                "quality_score": result.quality_score,
                "content": json.dumps(result.generated_content),
                "user_id": generation_state["request"].user_id
            }
            
            await self.neo4j.execute_write(query, parameters)
            
        except Exception as e:
            logger.warning(f"Failed to store final PRD: {str(e)}")
    
    async def get_generation_status(self, prd_id: str) -> Dict[str, Any]:
        """Get current status of PRD generation."""
        
        if prd_id not in self.active_generations:
            raise ValueError(f"PRD generation {prd_id} not found")
        
        generation_state = self.active_generations[prd_id]
        
        return {
            "prd_id": prd_id,
            "status": generation_state["status"].value,
            "current_phase": generation_state["current_phase"].value,
            "start_time": generation_state["start_time"].isoformat(),
            "quality_scores": generation_state["quality_scores"],
            "completion_time": generation_state.get("completion_time", {}).isoformat() if generation_state.get("completion_time") else None
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check pipeline health status."""
        
        try:
            orchestrator_health = await self.orchestrator.health_check() if self.orchestrator else {"status": "not_initialized"}
            hybrid_rag_health = await self.hybrid_rag.health_check()
            
            return {
                "status": "healthy" if self.is_initialized else "initializing",
                "initialized": self.is_initialized,
                "active_generations": len(self.active_generations),
                "orchestrator_status": orchestrator_health.get("status", "unknown"),
                "hybrid_rag_status": hybrid_rag_health.get("status", "unknown")
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Global pipeline instance
pipeline = PRDGenerationPipeline()


async def get_prd_pipeline() -> PRDGenerationPipeline:
    """Get the global PRD pipeline instance."""
    if not pipeline.is_initialized:
        await pipeline.initialize()
    return pipeline