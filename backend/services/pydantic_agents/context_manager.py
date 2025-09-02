"""
Context Manager Agent - Central orchestration agent for complex multi-stage operations.

This agent coordinates multiple other agents to accomplish complex workflows like
comprehensive PRD generation, multi-phase validation, and strategic planning tasks.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import json

from pydantic_ai import Agent as PydanticAIAgent
from pydantic import BaseModel, Field
import structlog

from .base_agent import BaseAgent, AgentResult
from core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class TaskDefinition(BaseModel):
    """Definition of a task to be coordinated."""
    task_id: str = Field(..., description="Unique task identifier")
    agent_type: str = Field(..., description="Type of agent to handle the task")
    operation: str = Field(..., description="Operation to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")
    validation_required: bool = Field(default=True, description="Whether GraphRAG validation is required")


class WorkflowPlan(BaseModel):
    """Plan for executing a complex workflow."""
    workflow_name: str = Field(..., description="Name of the workflow")
    description: str = Field(..., description="Workflow description")
    tasks: List[TaskDefinition] = Field(..., description="List of tasks to execute")
    estimated_duration_minutes: int = Field(..., description="Estimated completion time")
    quality_gates: List[str] = Field(default_factory=list, description="Quality checkpoints")


class ContextManagerAgent(BaseAgent):
    """
    Context Manager Agent for orchestrating complex multi-agent workflows.
    
    Capabilities:
    - Workflow planning and decomposition
    - Task coordination and dependency management
    - Context sharing between agents
    - Quality gate enforcement
    - Progress monitoring and reporting
    """
    
    def _initialize_agent(self) -> None:
        """Initialize the PydanticAI agent for context management."""
        self.pydantic_agent = PydanticAIAgent(
            model_name='openai:gpt-4o',
            system_prompt="""You are the Context Manager Agent for an AI-powered strategic planning platform.

Your role is to orchestrate complex workflows by coordinating multiple specialized agents:

1. **PRD Generator Agent**: Creates Product Requirements Documents
2. **Draft Agent**: Generates initial content drafts 
3. **Judge Agent**: Validates and scores content quality
4. **Task Executor Agent**: Handles implementation tasks
5. **Documentation Librarian Agent**: Manages documentation and knowledge

Key Responsibilities:
- Break down complex requests into manageable tasks
- Plan optimal execution sequences considering dependencies
- Coordinate agent interactions and context sharing
- Ensure quality gates are met throughout execution
- Monitor progress and adjust plans as needed

Quality Standards:
- All strategic documents must achieve >95% GraphRAG validation confidence
- Task dependencies must be clearly defined and respected
- Context must be preserved and shared appropriately between agents
- Progress must be tracked and reported in real-time

Response Format:
Always respond with structured JSON containing:
- planned_tasks: Array of task definitions
- execution_strategy: Sequential or parallel approach
- quality_gates: Validation checkpoints
- estimated_timeline: Duration estimates
- context_requirements: Shared data needs""",
            deps_type=Dict[str, Any]
        )
    
    async def execute(self, operation: str, context: Dict[str, Any]) -> AgentResult:
        """Execute a context management operation."""
        start_time = self._log_operation_start(operation, context)
        
        try:
            if operation == "plan_prd_generation":
                result = await self._plan_prd_generation(context)
            elif operation == "coordinate_workflow":
                result = await self._coordinate_workflow(context)
            elif operation == "monitor_progress":
                result = await self._monitor_progress(context)
            elif operation == "handle_quality_gate":
                result = await self._handle_quality_gate(context)
            elif operation == "update_context":
                result = await self._update_context(context)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            processing_time_ms = self._log_operation_complete(operation, start_time, True)
            
            return self._create_success_result(
                result=result,
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            processing_time_ms = self._log_operation_complete(operation, start_time, False, str(e))
            return self._create_error_result(
                error=str(e),
                metadata={"processing_time_ms": processing_time_ms}
            )
    
    async def _plan_prd_generation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan a comprehensive PRD generation workflow."""
        
        # Extract requirements from context
        initial_description = self._extract_context_parameter(context, "initial_description")
        user_id = self._extract_context_parameter(context, "user_id")
        project_id = self._extract_context_parameter(context, "project_id", required=False)
        complexity_level = self._extract_context_parameter(context, "complexity_level", required=False, default="medium")
        
        # Use PydanticAI to create execution plan
        planning_prompt = f"""Plan a comprehensive PRD generation workflow for the following requirements:

Initial Description: {initial_description}
User ID: {user_id}
Project ID: {project_id or 'New Project'}
Complexity Level: {complexity_level}

Create a detailed workflow plan that includes:
1. Initial analysis and concept extraction
2. Stakeholder identification and requirements gathering
3. Technical feasibility assessment
4. Draft PRD generation with multiple iterations
5. GraphRAG validation at each stage
6. Quality assurance and final review
7. Documentation and knowledge storage

Consider the complexity level when determining the number of iterations and validation stages required."""

        result = await self.pydantic_agent.run(
            planning_prompt,
            deps={
                "hybrid_rag": self.hybrid_rag,
                "context": context
            }
        )
        
        # Parse the result and create structured workflow plan
        workflow_plan = self._create_prd_workflow_plan(result.data, context)
        
        # Validate the plan using GraphRAG
        plan_validation = await self.validate_with_graphrag(
            content=json.dumps(workflow_plan.model_dump(), indent=2),
            context={"section_type": "workflow_plan"}
        )
        
        return {
            "workflow_plan": workflow_plan.model_dump(),
            "validation_results": plan_validation,
            "ready_for_execution": plan_validation.get("passes_threshold", False),
            "estimated_duration_minutes": workflow_plan.estimated_duration_minutes
        }
    
    def _create_prd_workflow_plan(self, ai_result: str, context: Dict[str, Any]) -> WorkflowPlan:
        """Create a structured workflow plan from AI result."""
        
        complexity_level = context.get("complexity_level", "medium")
        
        # Define tasks based on complexity
        base_tasks = [
            TaskDefinition(
                task_id="analyze_requirements",
                agent_type="prd_generator",
                operation="analyze_initial_requirements",
                parameters={"initial_description": context.get("initial_description")}
            ),
            TaskDefinition(
                task_id="generate_phase0",
                agent_type="prd_generator", 
                operation="generate_phase0",
                parameters={"analysis_result": "analyze_requirements"},
                dependencies=["analyze_requirements"]
            ),
            TaskDefinition(
                task_id="validate_phase0",
                agent_type="judge_agent",
                operation="validate_prd_phase",
                parameters={"phase": "phase_0", "content": "generate_phase0"},
                dependencies=["generate_phase0"]
            )
        ]
        
        if complexity_level in ["medium", "high"]:
            base_tasks.extend([
                TaskDefinition(
                    task_id="generate_phase1",
                    agent_type="prd_generator",
                    operation="generate_phase1",
                    dependencies=["validate_phase0"]
                ),
                TaskDefinition(
                    task_id="validate_phase1", 
                    agent_type="judge_agent",
                    operation="validate_prd_phase",
                    parameters={"phase": "phase_1"},
                    dependencies=["generate_phase1"]
                )
            ])
        
        if complexity_level == "high":
            base_tasks.extend([
                TaskDefinition(
                    task_id="comprehensive_review",
                    agent_type="judge_agent",
                    operation="comprehensive_quality_review",
                    dependencies=["validate_phase1"]
                ),
                TaskDefinition(
                    task_id="store_knowledge",
                    agent_type="documentation_librarian",
                    operation="store_prd_knowledge",
                    dependencies=["comprehensive_review"]
                )
            ])
        
        # Estimate duration based on complexity
        duration_map = {"low": 30, "medium": 60, "high": 120}
        estimated_duration = duration_map.get(complexity_level, 60)
        
        return WorkflowPlan(
            workflow_name=f"PRD Generation - {complexity_level.title()} Complexity",
            description=f"Complete PRD generation workflow for {complexity_level} complexity project",
            tasks=base_tasks,
            estimated_duration_minutes=estimated_duration,
            quality_gates=[
                "phase0_validation_confidence_>_95%",
                "phase1_validation_confidence_>_95%",
                "final_graphrag_validation_passes"
            ]
        )
    
    async def _coordinate_workflow(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate execution of a planned workflow."""
        
        workflow_plan = self._extract_context_parameter(context, "workflow_plan")
        execution_mode = self._extract_context_parameter(context, "execution_mode", required=False, default="sequential")
        
        # Create coordination plan
        coordination_prompt = f"""Coordinate the execution of this workflow plan:

Workflow: {json.dumps(workflow_plan, indent=2)}
Execution Mode: {execution_mode}

Provide detailed coordination instructions including:
1. Task execution order (respecting dependencies)
2. Context sharing requirements between agents
3. Quality gate checkpoints
4. Error handling and recovery procedures
5. Progress monitoring milestones

Consider optimal resource utilization and execution efficiency."""

        result = await self.pydantic_agent.run(
            coordination_prompt,
            deps={
                "workflow_plan": workflow_plan,
                "execution_mode": execution_mode
            }
        )
        
        return {
            "coordination_plan": result.data,
            "execution_ready": True,
            "monitoring_enabled": True
        }
    
    async def _monitor_progress(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor workflow execution progress."""
        
        workflow_id = self._extract_context_parameter(context, "workflow_id")
        current_tasks = self._extract_context_parameter(context, "current_tasks")
        completed_tasks = self._extract_context_parameter(context, "completed_tasks", required=False, default=[])
        
        # Analyze progress
        progress_prompt = f"""Analyze the current progress of workflow {workflow_id}:

Current Tasks: {json.dumps(current_tasks, indent=2)}
Completed Tasks: {json.dumps(completed_tasks, indent=2)}

Provide progress analysis including:
1. Completion percentage
2. Task execution status
3. Quality metrics
4. Timeline adherence
5. Risk assessment
6. Recommendations for optimization"""

        result = await self.pydantic_agent.run(
            progress_prompt,
            deps={
                "workflow_id": workflow_id,
                "current_tasks": current_tasks,
                "completed_tasks": completed_tasks
            }
        )
        
        return {
            "progress_analysis": result.data,
            "workflow_id": workflow_id,
            "monitoring_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_quality_gate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle quality gate validation."""
        
        gate_name = self._extract_context_parameter(context, "gate_name")
        gate_criteria = self._extract_context_parameter(context, "gate_criteria")
        current_results = self._extract_context_parameter(context, "current_results")
        
        # Evaluate quality gate
        gate_prompt = f"""Evaluate quality gate '{gate_name}' with the following criteria:

Gate Criteria: {json.dumps(gate_criteria, indent=2)}
Current Results: {json.dumps(current_results, indent=2)}

Determine if the quality gate passes and provide:
1. Pass/fail status with detailed reasoning
2. Quality metrics analysis
3. Areas needing improvement (if any)
4. Recommendations for proceeding
5. Risk assessment if gate fails"""

        result = await self.pydantic_agent.run(
            gate_prompt,
            deps={
                "gate_name": gate_name,
                "gate_criteria": gate_criteria,
                "current_results": current_results
            }
        )
        
        return {
            "gate_evaluation": result.data,
            "gate_name": gate_name,
            "evaluation_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _update_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Update shared workflow context."""
        
        context_updates = self._extract_context_parameter(context, "context_updates")
        merge_strategy = self._extract_context_parameter(context, "merge_strategy", required=False, default="merge")
        
        # Process context updates
        update_prompt = f"""Process context updates for workflow:

Context Updates: {json.dumps(context_updates, indent=2)}
Merge Strategy: {merge_strategy}

Process the updates and provide:
1. Validated context changes
2. Impact analysis on ongoing tasks
3. Required notifications to other agents
4. Updated shared context structure"""

        result = await self.pydantic_agent.run(
            update_prompt,
            deps={
                "context_updates": context_updates,
                "merge_strategy": merge_strategy
            }
        )
        
        return {
            "context_updates_processed": result.data,
            "update_timestamp": datetime.utcnow().isoformat()
        }