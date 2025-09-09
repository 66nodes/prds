"""
Agent Orchestration System for Strategic Planning Platform.

Coordinates PydanticAI agents for PRD generation, task management, and validation.
Implements Context Manager pattern for multi-agent workflows.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import uuid
import json

import structlog
from pydantic import BaseModel, Field

from core.config import get_settings
from services.hybrid_rag import HybridRAGService
from core.database import get_neo4j
from services.context_aware_agent_selector import (
    ContextAwareAgentSelector,
    TaskRequirements,
    TaskContext,
    get_context_aware_selector
)
from services.agent_registry import CapabilityType, ComplexityLevel
from services.enhanced_parallel_executor import get_enhanced_executor, PriorityLevel

logger = structlog.get_logger(__name__)
settings = get_settings()


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentType(str, Enum):
    """Available agent types."""
    # Core Orchestration
    CONTEXT_MANAGER = "context_manager"
    TASK_ORCHESTRATOR = "task_orchestrator"
    TASK_EXECUTOR = "task_executor"
    TASK_CHECKER = "task_checker"
    
    # Strategic & Planning
    PROJECT_ARCHITECT = "project_architect"
    BUSINESS_ANALYST = "business_analyst"
    STRATEGY_CONSULTANT = "strategy_consultant"
    PRODUCT_MANAGER = "product_manager"
    TECHNICAL_LEAD = "technical_lead"
    CHANGE_MANAGER = "change_manager"
    
    # Content & Documentation
    DRAFT_AGENT = "draft_agent"
    DOCUMENTATION_LIBRARIAN = "documentation_librarian"
    TECHNICAL_WRITER = "technical_writer"
    CONTENT_MARKETER = "content_marketer"
    COPYWRITER = "copywriter"
    EDITOR = "editor"
    SEO_WRITER = "seo_writer"
    GRANT_WRITER = "grant_writer"
    LEGAL_WRITER = "legal_writer"
    ACADEMIC_WRITER = "academic_writer"
    BLOGGER = "blogger"
    SOCIAL_MEDIA_MANAGER = "social_media_manager"
    
    # Development & Engineering
    FULLSTACK_DEVELOPER = "fullstack_developer"
    FRONTEND_DEVELOPER = "frontend_developer"
    BACKEND_DEVELOPER = "backend_developer"
    MOBILE_DEVELOPER = "mobile_developer"
    DEVOPS_ENGINEER = "devops_engineer"
    CLOUD_ARCHITECT = "cloud_architect"
    DATABASE_ARCHITECT = "database_architect"
    API_ARCHITECT = "api_architect"
    SECURITY_ENGINEER = "security_engineer"
    PERFORMANCE_ENGINEER = "performance_engineer"
    QA_ENGINEER = "qa_engineer"
    TEST_AUTOMATION = "test_automation"
    CODE_REVIEWER = "code_reviewer"
    LEGACY_MODERNIZER = "legacy_modernizer"
    MICROSERVICES_ARCHITECT = "microservices_architect"
    KUBERNETES_SPECIALIST = "kubernetes_specialist"
    CI_CD_ENGINEER = "ci_cd_engineer"
    MONITORING_ENGINEER = "monitoring_engineer"
    SRE_ENGINEER = "sre_engineer"
    DATA_ENGINEER = "data_engineer"
    ML_ENGINEER = "ml_engineer"
    BLOCKCHAIN_DEVELOPER = "blockchain_developer"
    GAME_DEVELOPER = "game_developer"
    EMBEDDED_DEVELOPER = "embedded_developer"
    PLATFORM_ENGINEER = "platform_engineer"
    
    # AI & Machine Learning
    AI_ENGINEER = "ai_engineer"
    ML_RESEARCHER = "ml_researcher"
    DATA_SCIENTIST = "data_scientist"
    NLP_ENGINEER = "nlp_engineer"
    COMPUTER_VISION = "computer_vision"
    PROMPT_ENGINEER = "prompt_engineer"
    LLM_ARCHITECT = "llm_architect"
    AI_TRAINER = "ai_trainer"
    ML_OPS = "ml_ops"
    AI_ETHICIST = "ai_ethicist"
    RESEARCH_ASSISTANT = "research_assistant"
    AI_PRODUCT_MANAGER = "ai_product_manager"
    CONVERSATIONAL_AI = "conversational_ai"
    RECOMMENDATION_ENGINE = "recommendation_engine"
    ANOMALY_DETECTION = "anomaly_detection"
    
    # Analysis & Investigation
    BUSINESS_INTELLIGENCE = "business_intelligence"
    DATA_ANALYST = "data_analyst"
    FINANCIAL_ANALYST = "financial_analyst"
    MARKET_RESEARCHER = "market_researcher"
    USER_RESEARCHER = "user_researcher"
    COMPLIANCE_AUDITOR = "compliance_auditor"
    SECURITY_AUDITOR = "security_auditor"
    PERFORMANCE_ANALYST = "performance_analyst"
    COST_OPTIMIZER = "cost_optimizer"
    RISK_ANALYST = "risk_analyst"
    PROCESS_ANALYST = "process_analyst"
    COMPETITIVE_ANALYST = "competitive_analyst"
    TREND_ANALYST = "trend_analyst"
    SENTIMENT_ANALYST = "sentiment_analyst"
    CONVERSION_OPTIMIZER = "conversion_optimizer"
    SUPPLY_CHAIN_ANALYST = "supply_chain_analyst"
    ENVIRONMENTAL_ANALYST = "environmental_analyst"
    SOCIAL_IMPACT_ANALYST = "social_impact_analyst"
    
    # Creative & Design
    UI_DESIGNER = "ui_designer"
    UX_DESIGNER = "ux_designer"
    GRAPHIC_DESIGNER = "graphic_designer"
    WEB_DESIGNER = "web_designer"
    MOTION_DESIGNER = "motion_designer"
    BRAND_DESIGNER = "brand_designer"
    PRODUCT_DESIGNER = "product_designer"
    DESIGN_SYSTEMS = "design_systems"
    ACCESSIBILITY_SPECIALIST = "accessibility_specialist"
    CREATIVE_DIRECTOR = "creative_director"
    
    # Specialized Domains
    LEGAL_ADVISOR = "legal_advisor"
    HR_SPECIALIST = "hr_specialist"
    FINANCE_CONTROLLER = "finance_controller"
    SALES_ENGINEER = "sales_engineer"
    CUSTOMER_SUCCESS = "customer_success"
    SUPPORT_SPECIALIST = "support_specialist"
    TRAINING_SPECIALIST = "training_specialist"
    LOCALIZATION_SPECIALIST = "localization_specialist"
    HEALTHCARE_ANALYST = "healthcare_analyst"
    FINTECH_SPECIALIST = "fintech_specialist"
    ECOMMERCE_SPECIALIST = "ecommerce_specialist"
    LOGISTICS_COORDINATOR = "logistics_coordinator"
    
    # Core Content Agents
    PRD_GENERATOR = "prd_generator"
    JUDGE_AGENT = "judge_agent"
    
    # Additional Strategic Agents
    WBS_STRUCTURING_AGENT = "wbs_structuring_agent"
    RISK_MANAGER = "risk_manager"
    STRATEGIC_PLANNER = "strategic_planner"
    
    # Additional Content Agents
    API_DOCUMENTER = "api_documenter"
    DOCS_ARCHITECT = "docs_architect"
    MERMAID_EXPERT = "mermaid_expert"
    TUTORIAL_ENGINEER = "tutorial_engineer"
    REFERENCE_BUILDER = "reference_builder"
    
    # Additional Development Agents
    BACKEND_ARCHITECT = "backend_architect"
    TYPESCRIPT_PRO = "typescript_pro"
    JAVASCRIPT_PRO = "javascript_pro"
    PYTHON_PRO = "python_pro"
    GOLANG_PRO = "golang_pro"
    RUST_PRO = "rust_pro"
    JAVA_PRO = "java_pro"
    CSHARP_PRO = "csharp_pro"
    PHP_PRO = "php_pro"
    RUBY_PRO = "ruby_pro"
    C_PRO = "c_pro"
    CPP_PRO = "cpp_pro"
    ELIXIR_PRO = "elixir_pro"
    SCALA_PRO = "scala_pro"
    SQL_PRO = "sql_pro"
    
    # Additional Infrastructure Agents
    HYBRID_CLOUD_ARCHITECT = "hybrid_cloud_architect"
    DEPLOYMENT_ENGINEER = "deployment_engineer"
    DEVOPS_TROUBLESHOOTER = "devops_troubleshooter"
    TERRAFORM_SPECIALIST = "terraform_specialist"
    INCIDENT_RESPONDER = "incident_responder"
    NETWORK_ENGINEER = "network_engineer"
    
    # Additional Database Agents
    DATABASE_ADMIN = "database_admin"
    DATABASE_OPTIMIZER = "database_optimizer"
    POSTGRES_PRO = "postgres_pro"
    
    # Additional Quality & Security Agents
    SECURITY_AUDITOR = "security_auditor"
    CODE_REVIEWER_SPECIALIST = "code_reviewer_specialist"
    TEST_AUTOMATOR = "test_automator"
    DEBUGGER = "debugger"
    ERROR_DETECTIVE = "error_detective"
    
    # Additional Design Agents
    UI_UX_DESIGNER = "ui_ux_designer"
    VUE_EXPERT = "vue_expert"
    
    # Additional AI/ML Agents
    LLM_ARCHITECT_SPECIALIST = "llm_architect_specialist"
    ML_ENGINEER_SPECIALIST = "ml_engineer_specialist"
    MLOPS_ENGINEER = "mlops_engineer"
    
    # Specialized Domain Agents
    UNITY_DEVELOPER = "unity_developer"
    FLUTTER_EXPERT = "flutter_expert"
    IOS_DEVELOPER = "ios_developer"
    WORDPRESS_MASTER = "wordpress_master"
    MINECRAFT_BUKKIT_PRO = "minecraft_bukkit_pro"
    PAYMENT_INTEGRATION = "payment_integration"
    
    # SEO & Marketing Specialists
    SEO_CONTENT_WRITER = "seo_content_writer"
    SEO_CONTENT_AUDITOR = "seo_content_auditor"
    SEO_KEYWORD_STRATEGIST = "seo_keyword_strategist"
    SEO_META_OPTIMIZER = "seo_meta_optimizer"
    SEO_STRUCTURE_ARCHITECT = "seo_structure_architect"
    SEO_SNIPPET_HUNTER = "seo_snippet_hunter"
    SEO_CONTENT_PLANNER = "seo_content_planner"
    SEO_CONTENT_REFRESHER = "seo_content_refresher"
    SEO_AUTHORITY_BUILDER = "seo_authority_builder"
    SEO_CANNIBALIZATION_DETECTOR = "seo_cannibalization_detector"
    
    # Business Function Specialists
    QUANT_ANALYST = "quant_analyst"
    LEGACY_MODERNIZER = "legacy_modernizer"
    REFACTORING_SPECIALIST = "refactoring_specialist"
    SEARCH_SPECIALIST = "search_specialist"
    DX_OPTIMIZER = "dx_optimizer"
    
    # Platform Specific Agents
    HALLUCINATION_TRACE_AGENT = "hallucination_trace_agent"
    PROVENANCE_AUDITOR = "provenance_auditor"
    FEEDBACK_LOOP_TRACKER = "feedback_loop_tracker"
    COST_OPTIMIZATION_AGENT = "cost_optimization_agent"
    COMPLIANCE_OFFICER_AGENT = "compliance_officer_agent"
    CHANGE_MANAGEMENT_AGENT = "change_management_agent"
    AI_AGENT_PERFORMANCE_PROFILER = "ai_agent_performance_profiler"
    USER_BEHAVIOR_ANALYST = "user_behavior_analyst"
    HUMAN_IN_THE_LOOP_HANDLER = "human_in_the_loop_handler"
    API_SCHEMA_AUTO_MIGRATOR = "api_schema_auto_migrator"
    TRAINING_DATA_STEWARD = "training_data_steward"


class AgentTask(BaseModel):
    """Individual task for agent execution."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: AgentType = Field(..., description="Type of agent to execute task")
    operation: str = Field(..., description="Operation to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    context: Dict[str, Any] = Field(default_factory=dict, description="Task context")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Task priority")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Task status")
    result: Optional[Dict[str, Any]] = Field(None, description="Task execution result")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(None, description="Task start time")
    completed_at: Optional[datetime] = Field(None, description="Task completion time")
    timeout_seconds: int = Field(default=300, description="Task timeout in seconds")


class WorkflowContext(BaseModel):
    """Context shared across workflow execution."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(..., description="User initiating the workflow")
    project_id: Optional[str] = Field(None, description="Associated project ID")
    session_data: Dict[str, Any] = Field(default_factory=dict, description="Session data")
    shared_context: Dict[str, Any] = Field(default_factory=dict, description="Shared context")
    validation_results: List[Dict[str, Any]] = Field(default_factory=list, description="Validation results")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AgentOrchestrator:
    """
    Central orchestrator for managing PydanticAI agents and workflow execution.
    Implements Context Manager pattern for complex multi-agent workflows.
    """
    
    def __init__(self):
        self.hybrid_rag = HybridRAGService()
        self.neo4j = None  # Will be set in initialize()
        self.active_workflows: Dict[str, WorkflowContext] = {}
        self.task_registry: Dict[str, AgentTask] = {}
        self.agent_pool = {}  # Lazy-loaded agent instances
        self.context_aware_selector: Optional[ContextAwareAgentSelector] = None
        self.enhanced_executor = None
        self.is_initialized = False
        
    async def initialize(self) -> None:
        """Initialize the orchestrator with database connections."""
        try:
            await self.hybrid_rag.initialize()
            self.neo4j = await get_neo4j()
            
            # Initialize agent pool
            await self._initialize_agent_pool()
            
            # Initialize context-aware selector
            self.context_aware_selector = await get_context_aware_selector()
            
            # Initialize enhanced parallel executor
            self.enhanced_executor = await get_enhanced_executor()
            
            self.is_initialized = True
            logger.info("Agent orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent orchestrator: {str(e)}")
            raise
    
    async def _initialize_agent_pool(self) -> None:
        """Initialize the pool of available agents."""
        try:
            # Import agents here to avoid circular imports
            from services.pydantic_agents.context_manager import ContextManagerAgent
            from services.pydantic_agents.prd_generator import PRDGeneratorAgent
            from services.pydantic_agents.draft_agent import DraftAgent
            from services.pydantic_agents.judge_agent import JudgeAgent
            from services.pydantic_agents.task_executor import TaskExecutorAgent
            from services.pydantic_agents.documentation_librarian import DocumentationLibrarianAgent
            
            self.agent_pool = {
                AgentType.CONTEXT_MANAGER: ContextManagerAgent(self.hybrid_rag),
                AgentType.PRD_GENERATOR: PRDGeneratorAgent(self.hybrid_rag),
                AgentType.DRAFT_AGENT: DraftAgent(self.hybrid_rag),
                AgentType.JUDGE_AGENT: JudgeAgent(self.hybrid_rag),
                AgentType.TASK_EXECUTOR: TaskExecutorAgent(self.hybrid_rag),
                AgentType.DOCUMENTATION_LIBRARIAN: DocumentationLibrarianAgent(self.hybrid_rag)
            }
            
            logger.info(f"Initialized {len(self.agent_pool)} agents in pool")
            
        except ImportError as e:
            logger.warning(f"Some agents not available: {str(e)}")
            # Initialize with available agents only
            self.agent_pool = {}
    
    async def create_workflow(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> WorkflowContext:
        """Create a new workflow context."""
        if not self.is_initialized:
            raise RuntimeError("Agent orchestrator not initialized")
        
        context = WorkflowContext(
            user_id=user_id,
            project_id=project_id,
            shared_context=initial_context or {}
        )
        
        self.active_workflows[context.workflow_id] = context
        
        # Store workflow in Neo4j for persistence
        await self._store_workflow_context(context)
        
        logger.info(
            "Created new workflow",
            workflow_id=context.workflow_id,
            user_id=user_id,
            project_id=project_id
        )
        
        return context
    
    async def add_task(
        self,
        workflow_id: str,
        agent_type: AgentType,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        priority: TaskPriority = TaskPriority.MEDIUM
    ) -> AgentTask:
        """Add a task to the workflow."""
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        task = AgentTask(
            agent_type=agent_type,
            operation=operation,
            parameters=parameters or {},
            context=context or {},
            dependencies=dependencies or [],
            priority=priority
        )
        
        self.task_registry[task.id] = task
        
        logger.info(
            "Added task to workflow",
            workflow_id=workflow_id,
            task_id=task.id,
            agent_type=agent_type.value,
            operation=operation
        )
        
        return task
    
    async def execute_workflow(
        self,
        workflow_id: str,
        parallel_execution: bool = True,
        max_concurrent_tasks: int = 5
    ) -> Dict[str, Any]:
        """Execute all tasks in a workflow."""
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.active_workflows[workflow_id]
        
        # Get all tasks for this workflow
        workflow_tasks = [
            task for task in self.task_registry.values()
            if task.status == TaskStatus.PENDING
        ]
        
        if not workflow_tasks:
            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "tasks_executed": 0,
                "results": {}
            }
        
        logger.info(
            "Starting workflow execution",
            workflow_id=workflow_id,
            task_count=len(workflow_tasks),
            parallel_execution=parallel_execution
        )
        
        try:
            if parallel_execution:
                results = await self._execute_tasks_parallel(
                    workflow_tasks, workflow, max_concurrent_tasks
                )
            else:
                results = await self._execute_tasks_sequential(workflow_tasks, workflow)
            
            # Update workflow completion status
            workflow.updated_at = datetime.utcnow()
            await self._update_workflow_context(workflow)
            
            execution_summary = {
                "workflow_id": workflow_id,
                "status": "completed",
                "tasks_executed": len(workflow_tasks),
                "successful_tasks": len([r for r in results.values() if r.get("success", False)]),
                "failed_tasks": len([r for r in results.values() if not r.get("success", False)]),
                "results": results,
                "execution_time_seconds": (datetime.utcnow() - workflow.created_at).total_seconds()
            }
            
            logger.info(
                "Workflow execution completed",
                workflow_id=workflow_id,
                **{k: v for k, v in execution_summary.items() if k != "results"}
            )
            
            return execution_summary
            
        except Exception as e:
            logger.error(
                "Workflow execution failed",
                workflow_id=workflow_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def _execute_tasks_parallel(
        self,
        tasks: List[AgentTask],
        workflow: WorkflowContext,
        max_concurrent: int
    ) -> Dict[str, Any]:
        """Execute tasks in parallel with dependency management."""
        results = {}
        remaining_tasks = tasks.copy()
        semaphore = asyncio.Semaphore(max_concurrent)
        
        while remaining_tasks:
            # Find tasks with resolved dependencies
            ready_tasks = []
            for task in remaining_tasks:
                if all(dep_id in results for dep_id in task.dependencies):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # Check for circular dependencies
                unresolved_deps = set()
                for task in remaining_tasks:
                    unresolved_deps.update(task.dependencies)
                
                task_ids = {task.id for task in remaining_tasks}
                circular_deps = unresolved_deps.intersection(task_ids)
                
                if circular_deps:
                    raise RuntimeError(f"Circular dependency detected: {circular_deps}")
                else:
                    raise RuntimeError("No tasks ready for execution - dependency issue")
            
            # Execute ready tasks in parallel
            execution_tasks = []
            for task in ready_tasks:
                execution_tasks.append(
                    self._execute_single_task_with_semaphore(task, workflow, semaphore)
                )
            
            # Wait for completion
            task_results = await asyncio.gather(*execution_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(task_results):
                task = ready_tasks[i]
                if isinstance(result, Exception):
                    results[task.id] = {
                        "success": False,
                        "error": str(result),
                        "task_id": task.id
                    }
                    task.status = TaskStatus.FAILED
                    task.error = str(result)
                else:
                    results[task.id] = result
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                
                task.completed_at = datetime.utcnow()
                remaining_tasks.remove(task)
        
        return results
    
    async def _execute_tasks_sequential(
        self,
        tasks: List[AgentTask],
        workflow: WorkflowContext
    ) -> Dict[str, Any]:
        """Execute tasks sequentially in dependency order."""
        results = {}
        
        # Sort tasks by dependencies (topological sort)
        sorted_tasks = self._topological_sort_tasks(tasks)
        
        for task in sorted_tasks:
            try:
                result = await self._execute_single_task(task, workflow)
                results[task.id] = result
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.completed_at = datetime.utcnow()
                
            except Exception as e:
                logger.error(
                    "Task execution failed",
                    task_id=task.id,
                    agent_type=task.agent_type.value,
                    error=str(e)
                )
                results[task.id] = {
                    "success": False,
                    "error": str(e),
                    "task_id": task.id
                }
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = datetime.utcnow()
                
                # Decide whether to continue or stop on failure
                if task.priority == TaskPriority.CRITICAL:
                    raise RuntimeError(f"Critical task {task.id} failed: {str(e)}")
        
        return results
    
    async def _execute_single_task_with_semaphore(
        self,
        task: AgentTask,
        workflow: WorkflowContext,
        semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        """Execute a single task with semaphore protection."""
        async with semaphore:
            return await self._execute_single_task(task, workflow)
    
    async def _execute_single_task(
        self,
        task: AgentTask,
        workflow: WorkflowContext
    ) -> Dict[str, Any]:
        """Execute a single agent task."""
        if task.agent_type not in self.agent_pool:
            raise RuntimeError(f"Agent type {task.agent_type.value} not available")
        
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.utcnow()
        
        logger.info(
            "Executing task",
            task_id=task.id,
            agent_type=task.agent_type.value,
            operation=task.operation
        )
        
        agent = self.agent_pool[task.agent_type]
        
        # Prepare execution context
        execution_context = {
            **task.context,
            "workflow_context": workflow.shared_context,
            "task_parameters": task.parameters,
            "workflow_id": workflow.workflow_id,
            "user_id": workflow.user_id,
            "project_id": workflow.project_id
        }
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                agent.execute(task.operation, execution_context),
                timeout=task.timeout_seconds
            )
            
            # Update shared context with results if needed
            if result.get("update_context"):
                workflow.shared_context.update(result["update_context"])
                workflow.updated_at = datetime.utcnow()
            
            return {
                "success": True,
                "result": result,
                "task_id": task.id,
                "agent_type": task.agent_type.value
            }
            
        except asyncio.TimeoutError:
            raise RuntimeError(f"Task {task.id} timed out after {task.timeout_seconds} seconds")
        except Exception as e:
            logger.error(
                "Agent execution failed",
                task_id=task.id,
                agent_type=task.agent_type.value,
                error=str(e),
                exc_info=True
            )
            raise
    
    def _topological_sort_tasks(self, tasks: List[AgentTask]) -> List[AgentTask]:
        """Sort tasks in dependency order using topological sort."""
        # Create adjacency list
        graph = {task.id: task.dependencies for task in tasks}
        task_map = {task.id: task for task in tasks}
        
        # Kahn's algorithm
        in_degree = {task.id: 0 for task in tasks}
        for task_id, deps in graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[task_id] += 1
        
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(task_map[current])
            
            for task_id, deps in graph.items():
                if current in deps:
                    in_degree[task_id] -= 1
                    if in_degree[task_id] == 0:
                        queue.append(task_id)
        
        if len(result) != len(tasks):
            raise RuntimeError("Circular dependency detected in task graph")
        
        return result
    
    async def _store_workflow_context(self, context: WorkflowContext) -> None:
        """Store workflow context in Neo4j."""
        try:
            query = """
            CREATE (w:Workflow {
                id: $workflow_id,
                user_id: $user_id,
                project_id: $project_id,
                session_data: $session_data,
                shared_context: $shared_context,
                created_at: datetime(),
                updated_at: datetime()
            })
            """
            
            parameters = {
                'workflow_id': context.workflow_id,
                'user_id': context.user_id,
                'project_id': context.project_id,
                'session_data': json.dumps(context.session_data),
                'shared_context': json.dumps(context.shared_context)
            }
            
            await self.neo4j.execute_write(query, parameters)
            
        except Exception as e:
            logger.warning(f"Failed to store workflow context: {str(e)}")
    
    async def _update_workflow_context(self, context: WorkflowContext) -> None:
        """Update workflow context in Neo4j."""
        try:
            query = """
            MATCH (w:Workflow {id: $workflow_id})
            SET w.shared_context = $shared_context,
                w.updated_at = datetime()
            """
            
            parameters = {
                'workflow_id': context.workflow_id,
                'shared_context': json.dumps(context.shared_context)
            }
            
            await self.neo4j.execute_write(query, parameters)
            
        except Exception as e:
            logger.warning(f"Failed to update workflow context: {str(e)}")
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of a workflow."""
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.active_workflows[workflow_id]
        
        # Get task statuses
        workflow_tasks = [
            task for task in self.task_registry.values()
        ]
        
        task_summary = {
            "total_tasks": len(workflow_tasks),
            "pending": len([t for t in workflow_tasks if t.status == TaskStatus.PENDING]),
            "in_progress": len([t for t in workflow_tasks if t.status == TaskStatus.IN_PROGRESS]),
            "completed": len([t for t in workflow_tasks if t.status == TaskStatus.COMPLETED]),
            "failed": len([t for t in workflow_tasks if t.status == TaskStatus.FAILED]),
            "cancelled": len([t for t in workflow_tasks if t.status == TaskStatus.CANCELLED])
        }
        
        return {
            "workflow_id": workflow_id,
            "user_id": workflow.user_id,
            "project_id": workflow.project_id,
            "created_at": workflow.created_at.isoformat(),
            "updated_at": workflow.updated_at.isoformat(),
            "task_summary": task_summary,
            "shared_context": workflow.shared_context
        }
    
    async def cancel_workflow(self, workflow_id: str, reason: str = "User cancelled") -> None:
        """Cancel a workflow and all its pending tasks."""
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Cancel all pending tasks
        workflow_tasks = [
            task for task in self.task_registry.values()
            if task.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]
        ]
        
        for task in workflow_tasks:
            task.status = TaskStatus.CANCELLED
            task.error = reason
            task.completed_at = datetime.utcnow()
        
        logger.info(
            "Workflow cancelled",
            workflow_id=workflow_id,
            cancelled_tasks=len(workflow_tasks),
            reason=reason
        )
    
    async def select_agents_intelligently(
        self,
        task_context: TaskContext,
        required_capabilities: List[CapabilityType],
        complexity_level: ComplexityLevel = ComplexityLevel.MODERATE,
        estimated_tokens: int = 5000,
        max_execution_time: timedelta = timedelta(minutes=10),
        max_agents: int = 5,
        domain_knowledge: Optional[List[str]] = None,
        preferred_agents: Optional[List[AgentType]] = None,
        excluded_agents: Optional[List[AgentType]] = None
    ) -> Dict[str, Any]:
        """
        Intelligently select agents for a task using context-aware selection.
        
        Args:
            task_context: Context type for the task
            required_capabilities: List of required capabilities
            complexity_level: Task complexity level
            estimated_tokens: Estimated token requirement
            max_execution_time: Maximum execution time allowed
            max_agents: Maximum number of agents to select
            domain_knowledge: Required domain knowledge areas
            preferred_agents: Preferred agent types
            excluded_agents: Agents to exclude from selection
            
        Returns:
            Dictionary containing selection results and execution plan
        """
        if not self.context_aware_selector:
            raise RuntimeError("Context-aware selector not initialized")
        
        # Create task requirements
        requirements = TaskRequirements(
            task_context=task_context,
            required_capabilities=required_capabilities,
            complexity_level=complexity_level,
            estimated_tokens=estimated_tokens,
            max_execution_time=max_execution_time,
            domain_knowledge=domain_knowledge or [],
            preferred_agents=preferred_agents or [],
            excluded_agents=excluded_agents or []
        )
        
        logger.info(
            "Starting intelligent agent selection",
            task_context=task_context,
            capabilities=[cap.value for cap in required_capabilities],
            complexity=complexity_level,
            max_agents=max_agents
        )
        
        # Use context-aware selector to find optimal agents
        selection_result = await self.context_aware_selector.select_agents(
            requirements=requirements,
            max_agents=max_agents,
            include_fallbacks=True
        )
        
        logger.info(
            "Agent selection completed",
            selected_agents=[agent.value for agent in selection_result.selected_agents],
            confidence=selection_result.confidence_level,
            estimated_time=selection_result.estimated_completion_time.total_seconds()
        )
        
        return {
            "selected_agents": [agent.value for agent in selection_result.selected_agents],
            "agent_scores": [
                {
                    "agent_type": score.agent_type.value,
                    "total_score": score.total_score,
                    "capability_score": score.capability_score,
                    "performance_score": score.performance_score,
                    "availability_score": score.availability_score,
                    "context_score": score.context_score,
                    "confidence": score.confidence,
                    "reasoning": score.reasoning
                }
                for score in selection_result.scores
            ],
            "execution_plan": selection_result.execution_plan,
            "resource_allocation": selection_result.resource_allocation,
            "estimated_completion_time_seconds": selection_result.estimated_completion_time.total_seconds(),
            "confidence_level": selection_result.confidence_level,
            "fallback_options": [agent.value for agent in selection_result.fallback_options],
            "selection_reasoning": selection_result.selection_reasoning,
            "requirements": {
                "task_context": task_context.value,
                "required_capabilities": [cap.value for cap in required_capabilities],
                "complexity_level": complexity_level.value,
                "estimated_tokens": estimated_tokens,
                "max_execution_time_minutes": max_execution_time.total_seconds() / 60
            }
        }
    
    async def create_intelligent_workflow(
        self,
        user_id: str,
        task_context: TaskContext,
        required_capabilities: List[CapabilityType],
        project_id: Optional[str] = None,
        complexity_level: ComplexityLevel = ComplexityLevel.MODERATE,
        estimated_tokens: int = 5000,
        max_execution_time: timedelta = timedelta(minutes=10),
        max_agents: int = 5,
        domain_knowledge: Optional[List[str]] = None,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a workflow with intelligently selected agents.
        
        Returns workflow context and selected agent information.
        """
        # Select agents intelligently
        selection_result = await self.select_agents_intelligently(
            task_context=task_context,
            required_capabilities=required_capabilities,
            complexity_level=complexity_level,
            estimated_tokens=estimated_tokens,
            max_execution_time=max_execution_time,
            max_agents=max_agents,
            domain_knowledge=domain_knowledge
        )
        
        # Create workflow context
        workflow_context = await self.create_workflow(
            user_id=user_id,
            project_id=project_id,
            initial_context=initial_context
        )
        
        # Store agent selection information in workflow context
        workflow_context.shared_context.update({
            "intelligent_selection": selection_result,
            "task_context": task_context.value,
            "complexity_level": complexity_level.value
        })
        
        return {
            "workflow_context": {
                "workflow_id": workflow_context.workflow_id,
                "user_id": workflow_context.user_id,
                "project_id": workflow_context.project_id,
                "created_at": workflow_context.created_at.isoformat()
            },
            "agent_selection": selection_result
        }
    
    async def add_intelligent_task(
        self,
        workflow_id: str,
        operation: str,
        task_context: TaskContext,
        required_capabilities: List[CapabilityType],
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        complexity_level: ComplexityLevel = ComplexityLevel.MODERATE,
        priority: TaskPriority = TaskPriority.MEDIUM
    ) -> Dict[str, Any]:
        """
        Add a task to workflow with intelligent agent selection for that specific task.
        """
        # Select best agent for this specific task
        requirements = TaskRequirements(
            task_context=task_context,
            required_capabilities=required_capabilities,
            complexity_level=complexity_level,
            estimated_tokens=parameters.get("estimated_tokens", 2000) if parameters else 2000,
            max_execution_time=timedelta(minutes=5)
        )
        
        selection_result = await self.context_aware_selector.select_agents(
            requirements=requirements,
            max_agents=1,  # Single best agent for this task
            include_fallbacks=True
        )
        
        if not selection_result.selected_agents:
            raise ValueError("No suitable agent found for task requirements")
        
        selected_agent = selection_result.selected_agents[0]
        
        # Add task with selected agent
        task = await self.add_task(
            workflow_id=workflow_id,
            agent_type=selected_agent,
            operation=operation,
            parameters=parameters,
            context={
                **(context or {}),
                "intelligent_selection": {
                    "agent_score": next(
                        (score for score in selection_result.scores 
                         if score.agent_type == selected_agent), None
                    ),
                    "selection_confidence": selection_result.confidence_level,
                    "fallback_options": [agent.value for agent in selection_result.fallback_options]
                }
            },
            priority=priority
        )
        
        return {
            "task_id": task.id,
            "selected_agent": selected_agent.value,
            "selection_confidence": selection_result.confidence_level,
            "fallback_options": [agent.value for agent in selection_result.fallback_options],
            "task_details": {
                "operation": operation,
                "priority": priority.value,
                "status": task.status.value,
                "created_at": task.created_at.isoformat()
            }
        }
    
    async def update_agent_performance(
        self,
        agent_type: AgentType,
        execution_time_ms: int,
        success: bool,
        quality_score: Optional[float] = None
    ) -> None:
        """Update agent performance metrics after task execution."""
        if self.context_aware_selector:
            await self.context_aware_selector.update_performance_metrics(
                agent_type=agent_type,
                execution_time_ms=execution_time_ms,
                success=success,
                quality_score=quality_score
            )
    
    async def execute_tasks_enhanced(
        self,
        tasks: List[AgentTask],
        workflow: WorkflowContext,
        priority: PriorityLevel = PriorityLevel.NORMAL,
        timeout: Optional[float] = None,
        use_enhanced_executor: bool = True
    ) -> Dict[str, Any]:
        """
        Execute tasks using the enhanced parallel executor with advanced features.
        
        Args:
            tasks: List of agent tasks to execute
            workflow: Workflow context
            priority: Execution priority level
            timeout: Optional timeout in seconds
            use_enhanced_executor: Whether to use enhanced executor (fallback to standard if False)
            
        Returns:
            Dict containing execution results and analytics
        """
        if not self.is_initialized:
            raise RuntimeError("Orchestrator not initialized")
        
        if not use_enhanced_executor or not self.enhanced_executor:
            logger.warning("Enhanced executor not available, falling back to standard execution")
            return await self._execute_tasks_parallel(tasks, workflow, 5)
        
        try:
            logger.info(
                "Starting enhanced task execution",
                task_count=len(tasks),
                priority=priority.value,
                workflow_id=workflow.workflow_id
            )
            
            result = await self.enhanced_executor.execute_parallel(
                tasks=tasks,
                workflow=workflow,
                priority=priority,
                timeout=timeout
            )
            
            # Update workflow status
            workflow.status = WorkflowStatus.COMPLETED if result.get("results") else WorkflowStatus.FAILED
            workflow.end_time = datetime.utcnow()
            workflow.result = result.get("results", {})
            
            # Update performance metrics for agents
            if "results" in result:
                for task_id, task_result in result["results"].items():
                    if "metrics" in task_result:
                        task = next((t for t in tasks if t.task_id == task_id), None)
                        if task:
                            await self.update_agent_performance(
                                agent_type=task.agent_type,
                                execution_time_ms=task_result["metrics"]["duration_ms"],
                                success=task_result["status"] == "completed"
                            )
            
            logger.info(
                "Enhanced task execution completed",
                workflow_id=workflow.workflow_id,
                results_count=len(result.get("results", {})),
                analytics=result.get("analytics", {})
            )
            
            return result
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.end_time = datetime.utcnow()
            workflow.error_message = str(e)
            
            logger.error(
                f"Enhanced task execution failed",
                workflow_id=workflow.workflow_id,
                error=str(e)
            )
            raise
    
    async def get_enhanced_executor_status(self) -> Dict[str, Any]:
        """Get status and analytics from the enhanced executor."""
        if not self.enhanced_executor:
            return {"error": "Enhanced executor not available"}
        
        return await self.enhanced_executor.get_execution_status()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check orchestrator health status."""
        try:
            hybrid_rag_health = await self.hybrid_rag.health_check()
            enhanced_executor_status = await self.get_enhanced_executor_status() if self.enhanced_executor else {"status": "not_available"}
            
            return {
                "status": "healthy" if self.is_initialized else "initializing",
                "initialized": self.is_initialized,
                "active_workflows": len(self.active_workflows),
                "total_tasks": len(self.task_registry),
                "available_agents": list(self.agent_pool.keys()),
                "hybrid_rag_status": hybrid_rag_health.get("status", "unknown"),
                "enhanced_executor": {
                    "available": self.enhanced_executor is not None,
                    "active_tasks": enhanced_executor_status.get("active_tasks", {}),
                    "optimal_concurrency": enhanced_executor_status.get("optimal_concurrency", 0),
                    "resource_usage": enhanced_executor_status.get("resource_usage", {}),
                    "analytics": enhanced_executor_status.get("analytics", {})
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Global orchestrator instance
orchestrator = AgentOrchestrator()


async def get_orchestrator() -> AgentOrchestrator:
    """Get the global orchestrator instance."""
    if not orchestrator.is_initialized:
        await orchestrator.initialize()
    return orchestrator