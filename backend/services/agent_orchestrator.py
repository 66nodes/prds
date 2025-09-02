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
    CONTEXT_MANAGER = "context_manager"
    PRD_GENERATOR = "prd_generator"
    DRAFT_AGENT = "draft_agent"
    JUDGE_AGENT = "judge_agent"
    TASK_EXECUTOR = "task_executor"
    DOCUMENTATION_LIBRARIAN = "documentation_librarian"


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
        self.is_initialized = False
        
    async def initialize(self) -> None:
        """Initialize the orchestrator with database connections."""
        try:
            await self.hybrid_rag.initialize()
            self.neo4j = await get_neo4j()
            
            # Initialize agent pool
            await self._initialize_agent_pool()
            
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
    
    async def health_check(self) -> Dict[str, Any]:
        """Check orchestrator health status."""
        try:
            hybrid_rag_health = await self.hybrid_rag.health_check()
            
            return {
                "status": "healthy" if self.is_initialized else "initializing",
                "initialized": self.is_initialized,
                "active_workflows": len(self.active_workflows),
                "total_tasks": len(self.task_registry),
                "available_agents": list(self.agent_pool.keys()),
                "hybrid_rag_status": hybrid_rag_health.get("status", "unknown")
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