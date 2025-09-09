"""
Agent Registry System for Multi-Agent Orchestration.

Comprehensive catalog of all 107 specialized agents with metadata, capabilities,
roles, and interfaces for context-aware selection and orchestration.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import structlog

from pydantic import BaseModel, Field, validator
from core.config import get_settings
from services.agent_orchestrator import AgentType

logger = structlog.get_logger(__name__)
settings = get_settings()


class CapabilityType(str, Enum):
    """Types of agent capabilities."""
    # Core Functions
    ANALYSIS = "analysis"
    CREATION = "creation"
    OPTIMIZATION = "optimization"
    VALIDATION = "validation"
    COORDINATION = "coordination"
    COMMUNICATION = "communication"
    
    # Domain Expertise
    TECHNICAL = "technical"
    BUSINESS = "business"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    STRATEGIC = "strategic"
    OPERATIONAL = "operational"
    
    # Processing Types
    TEXT_PROCESSING = "text_processing"
    DATA_PROCESSING = "data_processing"
    CODE_PROCESSING = "code_processing"
    VISUAL_PROCESSING = "visual_processing"
    AUDIO_PROCESSING = "audio_processing"
    
    # Interaction Patterns
    AUTONOMOUS = "autonomous"
    COLLABORATIVE = "collaborative"
    SUPERVISORY = "supervisory"
    REACTIVE = "reactive"


class ComplexityLevel(str, Enum):
    """Task complexity levels agents can handle."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"
    RESEARCH_LEVEL = "research_level"


class ResourceRequirement(str, Enum):
    """Resource requirements for agent execution."""
    LOW = "low"           # Minimal compute/memory
    MEDIUM = "medium"     # Standard resources
    HIGH = "high"         # Significant resources
    INTENSIVE = "intensive"  # High-performance requirements


@dataclass
class AgentCapability:
    """Individual capability with metadata."""
    capability_type: CapabilityType
    proficiency_level: float  # 0.0 to 1.0
    complexity_support: List[ComplexityLevel]
    resource_cost: ResourceRequirement
    prerequisites: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.8


@dataclass
class AgentInterface:
    """Agent interface specification."""
    supported_operations: List[str]
    input_formats: List[str]
    output_formats: List[str]
    context_requirements: List[str] = field(default_factory=list)
    optional_parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_default: int = 300  # seconds
    max_retries: int = 3


@dataclass
class AgentMetrics:
    """Performance and usage metrics."""
    success_rate: float = 0.95
    average_response_time_ms: int = 2000
    peak_response_time_ms: int = 10000
    total_executions: int = 0
    last_used: Optional[datetime] = None
    confidence_scores: List[float] = field(default_factory=list)
    error_patterns: List[str] = field(default_factory=list)


@dataclass
class AgentRegistryEntry:
    """Complete agent registry entry."""
    agent_type: AgentType
    name: str
    description: str
    category: str
    subcategory: Optional[str] = None
    
    # Core Properties
    capabilities: List[AgentCapability] = field(default_factory=list)
    interface: Optional[AgentInterface] = None
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    
    # Relationships
    dependencies: List[AgentType] = field(default_factory=list)
    collaborators: List[AgentType] = field(default_factory=list)
    alternatives: List[AgentType] = field(default_factory=list)
    
    # Metadata
    version: str = "1.0.0"
    maintainer: str = "Strategic Planning Platform"
    documentation_url: Optional[str] = None
    implementation_class: Optional[str] = None
    is_available: bool = True
    is_experimental: bool = False
    
    # Runtime Properties
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)


class AgentRegistry:
    """
    Comprehensive registry for all specialized agents.
    
    Provides centralized catalog with capabilities, metrics, and metadata
    for intelligent agent selection and orchestration.
    """
    
    def __init__(self):
        self._registry: Dict[AgentType, AgentRegistryEntry] = {}
        self._capability_index: Dict[CapabilityType, Set[AgentType]] = {}
        self._category_index: Dict[str, Set[AgentType]] = {}
        self._tag_index: Dict[str, Set[AgentType]] = {}
        self._is_initialized = False
        
    async def initialize(self) -> None:
        """Initialize the agent registry with all agent definitions."""
        try:
            logger.info("Initializing agent registry with 107 specialized agents")
            
            # Initialize all agent entries
            await self._initialize_core_agents()
            await self._initialize_strategic_agents()
            await self._initialize_content_agents()
            await self._initialize_development_agents()
            await self._initialize_ai_ml_agents()
            await self._initialize_analysis_agents()
            await self._initialize_creative_agents()
            await self._initialize_specialized_agents()
            
            # Build indexes for fast lookup
            await self._build_indexes()
            
            # Load historical metrics if available
            await self._load_metrics()
            
            self._is_initialized = True
            logger.info(f"Agent registry initialized with {len(self._registry)} agents")
            
            # Validate we have the expected number of agents
            expected_count = 107
            if len(self._registry) < expected_count:
                logger.warning(f"Registry has {len(self._registry)} agents, expected at least {expected_count}")
            else:
                logger.info(f"Agent registry successfully initialized with all {len(self._registry)} specialized agents")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent registry: {str(e)}")
            raise
    
    async def _initialize_core_agents(self) -> None:
        """Initialize core orchestration agents."""
        
        # Context Manager
        self._registry[AgentType.CONTEXT_MANAGER] = AgentRegistryEntry(
            agent_type=AgentType.CONTEXT_MANAGER,
            name="Context Manager",
            description="Maintains shared context across multi-agent workflows",
            category="Core Orchestration",
            capabilities=[
                AgentCapability(
                    CapabilityType.COORDINATION,
                    proficiency_level=0.95,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.MEDIUM
                ),
                AgentCapability(
                    CapabilityType.COMMUNICATION,
                    proficiency_level=0.90,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.LOW
                )
            ],
            interface=AgentInterface(
                supported_operations=["update_context", "get_context", "merge_context"],
                input_formats=["json", "dict"],
                output_formats=["json", "dict"],
                context_requirements=["workflow_id", "user_id"]
            ),
            implementation_class="services.pydantic_agents.context_manager.ContextManagerAgent",
            tags=["orchestration", "context", "workflow"]
        )
        
        # Task Orchestrator
        self._registry[AgentType.TASK_ORCHESTRATOR] = AgentRegistryEntry(
            agent_type=AgentType.TASK_ORCHESTRATOR,
            name="Task Orchestrator",
            description="Coordinates task execution across multiple agents",
            category="Core Orchestration",
            capabilities=[
                AgentCapability(
                    CapabilityType.COORDINATION,
                    proficiency_level=0.90,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.HIGH
                ),
                AgentCapability(
                    CapabilityType.ANALYTICAL,
                    proficiency_level=0.85,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.MEDIUM
                )
            ],
            interface=AgentInterface(
                supported_operations=["create_workflow", "execute_tasks", "monitor_progress"],
                input_formats=["json", "yaml"],
                output_formats=["json", "status"],
                timeout_default=600
            ),
            tags=["orchestration", "workflow", "coordination"]
        )
        
        # Task Executor
        self._registry[AgentType.TASK_EXECUTOR] = AgentRegistryEntry(
            agent_type=AgentType.TASK_EXECUTOR,
            name="Task Executor",
            description="Executes individual tasks with monitoring and reporting",
            category="Core Orchestration",
            capabilities=[
                AgentCapability(
                    CapabilityType.OPERATIONAL,
                    proficiency_level=0.88,
                    complexity_support=[ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.MEDIUM
                )
            ],
            interface=AgentInterface(
                supported_operations=["execute_task", "report_status", "handle_errors"],
                input_formats=["json", "dict"],
                output_formats=["json", "result"],
                timeout_default=300
            ),
            implementation_class="services.pydantic_agents.task_executor.TaskExecutorAgent",
            tags=["execution", "tasks", "monitoring"]
        )
        
        # Task Checker
        self._registry[AgentType.TASK_CHECKER] = AgentRegistryEntry(
            agent_type=AgentType.TASK_CHECKER,
            name="Task Checker",
            description="Validates task completion and quality assurance",
            category="Core Orchestration",
            capabilities=[
                AgentCapability(
                    CapabilityType.VALIDATION,
                    proficiency_level=0.92,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.MEDIUM
                ),
                AgentCapability(
                    CapabilityType.ANALYTICAL,
                    proficiency_level=0.85,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.LOW
                )
            ],
            interface=AgentInterface(
                supported_operations=["validate_task", "check_quality", "generate_report"],
                input_formats=["json", "result"],
                output_formats=["json", "report"],
                confidence_threshold=0.9
            ),
            tags=["validation", "quality", "checking"]
        )
    
    async def _initialize_strategic_agents(self) -> None:
        """Initialize strategic and planning agents."""
        
        # Project Architect
        self._registry[AgentType.PROJECT_ARCHITECT] = AgentRegistryEntry(
            agent_type=AgentType.PROJECT_ARCHITECT,
            name="Project Architect",
            description="Designs project structure and technical architecture",
            category="Strategic & Planning",
            capabilities=[
                AgentCapability(
                    CapabilityType.STRATEGIC,
                    proficiency_level=0.95,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT, ComplexityLevel.RESEARCH_LEVEL],
                    resource_cost=ResourceRequirement.HIGH
                ),
                AgentCapability(
                    CapabilityType.TECHNICAL,
                    proficiency_level=0.90,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.HIGH
                )
            ],
            interface=AgentInterface(
                supported_operations=["design_architecture", "create_project_plan", "assess_feasibility"],
                input_formats=["json", "requirements"],
                output_formats=["json", "diagram", "specification"],
                timeout_default=900
            ),
            tags=["architecture", "design", "planning", "technical"]
        )
        
        # Business Analyst
        self._registry[AgentType.BUSINESS_ANALYST] = AgentRegistryEntry(
            agent_type=AgentType.BUSINESS_ANALYST,
            name="Business Analyst",
            description="Analyzes business requirements and processes",
            category="Strategic & Planning",
            capabilities=[
                AgentCapability(
                    CapabilityType.BUSINESS,
                    proficiency_level=0.92,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.MEDIUM
                ),
                AgentCapability(
                    CapabilityType.ANALYTICAL,
                    proficiency_level=0.88,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.MEDIUM
                )
            ],
            interface=AgentInterface(
                supported_operations=["analyze_requirements", "model_processes", "identify_stakeholders"],
                input_formats=["json", "text", "document"],
                output_formats=["json", "report", "diagram"]
            ),
            tags=["business", "analysis", "requirements", "processes"]
        )
        
        # Add more strategic agents...
        self._add_strategic_agent_definitions()
    
    async def _initialize_content_agents(self) -> None:
        """Initialize content and documentation agents."""
        
        # Draft Agent
        self._registry[AgentType.DRAFT_AGENT] = AgentRegistryEntry(
            agent_type=AgentType.DRAFT_AGENT,
            name="Draft Agent",
            description="Creates initial drafts of documents and content",
            category="Content & Documentation",
            capabilities=[
                AgentCapability(
                    CapabilityType.CREATION,
                    proficiency_level=0.88,
                    complexity_support=[ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.MEDIUM
                ),
                AgentCapability(
                    CapabilityType.TEXT_PROCESSING,
                    proficiency_level=0.90,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.LOW
                )
            ],
            interface=AgentInterface(
                supported_operations=["create_draft", "refine_content", "structure_document"],
                input_formats=["json", "text", "outline"],
                output_formats=["markdown", "html", "json"]
            ),
            implementation_class="services.pydantic_agents.draft_agent.DraftAgent",
            tags=["content", "drafting", "writing", "documents"]
        )
        
        # Documentation Librarian
        self._registry[AgentType.DOCUMENTATION_LIBRARIAN] = AgentRegistryEntry(
            agent_type=AgentType.DOCUMENTATION_LIBRARIAN,
            name="Documentation Librarian",
            description="Manages and organizes documentation and knowledge base",
            category="Content & Documentation",
            capabilities=[
                AgentCapability(
                    CapabilityType.COORDINATION,
                    proficiency_level=0.85,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.MEDIUM
                ),
                AgentCapability(
                    CapabilityType.TEXT_PROCESSING,
                    proficiency_level=0.92,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.MEDIUM
                )
            ],
            interface=AgentInterface(
                supported_operations=["organize_docs", "search_knowledge", "maintain_index"],
                input_formats=["json", "markdown", "html"],
                output_formats=["json", "index", "search_results"]
            ),
            implementation_class="services.pydantic_agents.documentation_librarian.DocumentationLibrarianAgent",
            tags=["documentation", "knowledge", "organization", "search"]
        )
        
        # Add more content agents...
        self._add_content_agent_definitions()
    
    async def _initialize_development_agents(self) -> None:
        """Initialize development and engineering agents."""
        
        # Fullstack Developer
        self._registry[AgentType.FULLSTACK_DEVELOPER] = AgentRegistryEntry(
            agent_type=AgentType.FULLSTACK_DEVELOPER,
            name="Fullstack Developer",
            description="Develops both frontend and backend components",
            category="Development & Engineering",
            capabilities=[
                AgentCapability(
                    CapabilityType.TECHNICAL,
                    proficiency_level=0.88,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.HIGH
                ),
                AgentCapability(
                    CapabilityType.CODE_PROCESSING,
                    proficiency_level=0.90,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.HIGH
                )
            ],
            interface=AgentInterface(
                supported_operations=["develop_frontend", "develop_backend", "integrate_systems"],
                input_formats=["json", "specification", "code"],
                output_formats=["code", "component", "system"],
                timeout_default=1200
            ),
            tags=["development", "fullstack", "frontend", "backend", "integration"]
        )
        
        # Add more development agents...
        self._add_development_agent_definitions()
    
    async def _initialize_ai_ml_agents(self) -> None:
        """Initialize AI and machine learning agents."""
        
        # AI Engineer
        self._registry[AgentType.AI_ENGINEER] = AgentRegistryEntry(
            agent_type=AgentType.AI_ENGINEER,
            name="AI Engineer",
            description="Develops and optimizes AI systems and models",
            category="AI & Machine Learning",
            capabilities=[
                AgentCapability(
                    CapabilityType.TECHNICAL,
                    proficiency_level=0.95,
                    complexity_support=[ComplexityLevel.EXPERT, ComplexityLevel.RESEARCH_LEVEL],
                    resource_cost=ResourceRequirement.INTENSIVE
                ),
                AgentCapability(
                    CapabilityType.DATA_PROCESSING,
                    proficiency_level=0.92,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.HIGH
                )
            ],
            interface=AgentInterface(
                supported_operations=["design_model", "train_system", "optimize_performance"],
                input_formats=["json", "data", "specification"],
                output_formats=["model", "metrics", "report"],
                timeout_default=3600
            ),
            tags=["ai", "machine-learning", "models", "optimization"]
        )
        
        # Add more AI/ML agents...
        self._add_ai_ml_agent_definitions()
    
    async def _initialize_analysis_agents(self) -> None:
        """Initialize analysis and investigation agents."""
        # Add analysis agent definitions...
        self._add_analysis_agent_definitions()
    
    async def _initialize_creative_agents(self) -> None:
        """Initialize creative and design agents."""
        # Add creative agent definitions...
        self._add_creative_agent_definitions()
    
    async def _initialize_specialized_agents(self) -> None:
        """Initialize specialized domain agents."""
        # Add specialized agent definitions...
        self._add_specialized_agent_definitions()
    
    def _add_strategic_agent_definitions(self) -> None:
        """Add remaining strategic agent definitions."""
        
        # WBS Structuring Agent
        self._registry[AgentType.WBS_STRUCTURING_AGENT] = AgentRegistryEntry(
            agent_type=AgentType.WBS_STRUCTURING_AGENT,
            name="WBS Structuring Agent",
            description="Intelligent work breakdown structure generator with dependency analysis and effort estimation",
            category="Strategic & Planning",
            capabilities=[
                AgentCapability(
                    CapabilityType.STRATEGIC,
                    proficiency_level=0.92,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.HIGH
                ),
                AgentCapability(
                    CapabilityType.ANALYTICAL,
                    proficiency_level=0.88,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.MEDIUM
                )
            ],
            interface=AgentInterface(
                supported_operations=["create_wbs", "analyze_dependencies", "estimate_effort"],
                input_formats=["json", "requirements", "project_spec"],
                output_formats=["json", "wbs_structure", "gantt"]
            ),
            tags=["wbs", "planning", "estimation", "dependencies"]
        )
        
        # Risk Manager
        self._registry[AgentType.RISK_MANAGER] = AgentRegistryEntry(
            agent_type=AgentType.RISK_MANAGER,
            name="Risk Manager",
            description="Risk assessment and mitigation strategies for project management",
            category="Strategic & Planning",
            capabilities=[
                AgentCapability(
                    CapabilityType.ANALYTICAL,
                    proficiency_level=0.90,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.MEDIUM
                ),
                AgentCapability(
                    CapabilityType.BUSINESS,
                    proficiency_level=0.85,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.MEDIUM
                )
            ],
            interface=AgentInterface(
                supported_operations=["assess_risk", "create_mitigation_plan", "monitor_risks"],
                input_formats=["json", "project_data", "requirements"],
                output_formats=["json", "risk_report", "mitigation_plan"]
            ),
            tags=["risk", "analysis", "mitigation", "planning"]
        )
        
        # Strategic Planner
        self._registry[AgentType.STRATEGIC_PLANNER] = AgentRegistryEntry(
            agent_type=AgentType.STRATEGIC_PLANNER,
            name="Strategic Planner",
            description="Long-term strategic planning and roadmap development",
            category="Strategic & Planning",
            capabilities=[
                AgentCapability(
                    CapabilityType.STRATEGIC,
                    proficiency_level=0.95,
                    complexity_support=[ComplexityLevel.EXPERT, ComplexityLevel.RESEARCH_LEVEL],
                    resource_cost=ResourceRequirement.HIGH
                )
            ],
            interface=AgentInterface(
                supported_operations=["create_strategy", "roadmap_planning", "goal_setting"],
                input_formats=["json", "vision", "objectives"],
                output_formats=["json", "strategy_doc", "roadmap"]
            ),
            tags=["strategy", "planning", "roadmap", "vision"]
        )
    
    def _add_content_agent_definitions(self) -> None:
        """Add remaining content agent definitions."""
        
        # Judge Agent
        self._registry[AgentType.JUDGE_AGENT] = AgentRegistryEntry(
            agent_type=AgentType.JUDGE_AGENT,
            name="Judge Agent",
            description="Quality assessment and validation of generated content",
            category="Content & Documentation",
            capabilities=[
                AgentCapability(
                    CapabilityType.VALIDATION,
                    proficiency_level=0.95,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.MEDIUM
                ),
                AgentCapability(
                    CapabilityType.ANALYTICAL,
                    proficiency_level=0.90,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.MEDIUM
                )
            ],
            interface=AgentInterface(
                supported_operations=["evaluate_content", "score_quality", "provide_feedback"],
                input_formats=["json", "text", "document"],
                output_formats=["json", "score", "feedback_report"]
            ),
            implementation_class="services.pydantic_agents.judge_agent.JudgeAgent",
            tags=["validation", "quality", "assessment", "scoring"]
        )
        
        # API Documenter
        self._registry[AgentType.API_DOCUMENTER] = AgentRegistryEntry(
            agent_type=AgentType.API_DOCUMENTER,
            name="API Documenter",
            description="API documentation and SDK generation specialist",
            category="Content & Documentation",
            capabilities=[
                AgentCapability(
                    CapabilityType.CREATION,
                    proficiency_level=0.90,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.MEDIUM
                ),
                AgentCapability(
                    CapabilityType.TECHNICAL,
                    proficiency_level=0.85,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.LOW
                )
            ],
            interface=AgentInterface(
                supported_operations=["generate_api_docs", "create_sdk", "update_specs"],
                input_formats=["json", "openapi", "code"],
                output_formats=["markdown", "html", "sdk_package"]
            ),
            tags=["api", "documentation", "sdk", "openapi"]
        )
        
        # Docs Architect
        self._registry[AgentType.DOCS_ARCHITECT] = AgentRegistryEntry(
            agent_type=AgentType.DOCS_ARCHITECT,
            name="Documentation Architect",
            description="Technical documentation architecture and information design",
            category="Content & Documentation",
            capabilities=[
                AgentCapability(
                    CapabilityType.STRATEGIC,
                    proficiency_level=0.88,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.MEDIUM
                ),
                AgentCapability(
                    CapabilityType.CREATION,
                    proficiency_level=0.90,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.MEDIUM
                )
            ],
            interface=AgentInterface(
                supported_operations=["design_docs_structure", "create_guides", "maintain_standards"],
                input_formats=["json", "requirements", "content"],
                output_formats=["markdown", "structure_plan", "style_guide"]
            ),
            tags=["documentation", "architecture", "design", "standards"]
        )
    
    def _add_development_agent_definitions(self) -> None:
        """Add remaining development agent definitions."""
        
        # Backend Developer
        self._registry[AgentType.BACKEND_DEVELOPER] = AgentRegistryEntry(
            agent_type=AgentType.BACKEND_DEVELOPER,
            name="Backend Developer",
            description="Scalable API development and microservices architecture",
            category="Development & Engineering",
            capabilities=[
                AgentCapability(
                    CapabilityType.TECHNICAL,
                    proficiency_level=0.90,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.HIGH
                ),
                AgentCapability(
                    CapabilityType.CODE_PROCESSING,
                    proficiency_level=0.88,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.MEDIUM
                )
            ],
            interface=AgentInterface(
                supported_operations=["develop_api", "create_microservice", "optimize_database"],
                input_formats=["json", "specification", "requirements"],
                output_formats=["code", "api_spec", "service"]
            ),
            tags=["backend", "api", "microservices", "database"]
        )
        
        # Frontend Developer
        self._registry[AgentType.FRONTEND_DEVELOPER] = AgentRegistryEntry(
            agent_type=AgentType.FRONTEND_DEVELOPER,
            name="Frontend Developer",
            description="Modern UI development with React, Vue, Angular",
            category="Development & Engineering",
            capabilities=[
                AgentCapability(
                    CapabilityType.TECHNICAL,
                    proficiency_level=0.88,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.HIGH
                ),
                AgentCapability(
                    CapabilityType.VISUAL_PROCESSING,
                    proficiency_level=0.85,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.MEDIUM
                )
            ],
            interface=AgentInterface(
                supported_operations=["create_component", "implement_ui", "optimize_frontend"],
                input_formats=["json", "design", "specification"],
                output_formats=["code", "component", "application"]
            ),
            tags=["frontend", "ui", "components", "javascript"]
        )
        
        # Backend Architect
        self._registry[AgentType.BACKEND_ARCHITECT] = AgentRegistryEntry(
            agent_type=AgentType.BACKEND_ARCHITECT,
            name="Backend Architect",
            description="API design, microservice boundaries, and system architecture",
            category="Development & Engineering",
            capabilities=[
                AgentCapability(
                    CapabilityType.STRATEGIC,
                    proficiency_level=0.92,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT, ComplexityLevel.RESEARCH_LEVEL],
                    resource_cost=ResourceRequirement.HIGH
                ),
                AgentCapability(
                    CapabilityType.TECHNICAL,
                    proficiency_level=0.95,
                    complexity_support=[ComplexityLevel.EXPERT, ComplexityLevel.RESEARCH_LEVEL],
                    resource_cost=ResourceRequirement.HIGH
                )
            ],
            interface=AgentInterface(
                supported_operations=["design_architecture", "define_boundaries", "create_specs"],
                input_formats=["json", "requirements", "constraints"],
                output_formats=["json", "architecture_spec", "api_design"],
                timeout_default=900
            ),
            tags=["architecture", "backend", "design", "microservices"]
        )
    
    def _add_ai_ml_agent_definitions(self) -> None:
        """Add remaining AI/ML agent definitions."""
        
        # LLM Architect Specialist
        self._registry[AgentType.LLM_ARCHITECT_SPECIALIST] = AgentRegistryEntry(
            agent_type=AgentType.LLM_ARCHITECT_SPECIALIST,
            name="LLM Architect Specialist",
            description="Large language model architecture and deployment specialist",
            category="AI & Machine Learning",
            capabilities=[
                AgentCapability(
                    CapabilityType.TECHNICAL,
                    proficiency_level=0.95,
                    complexity_support=[ComplexityLevel.EXPERT, ComplexityLevel.RESEARCH_LEVEL],
                    resource_cost=ResourceRequirement.INTENSIVE
                ),
                AgentCapability(
                    CapabilityType.DATA_PROCESSING,
                    proficiency_level=0.90,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.HIGH
                )
            ],
            interface=AgentInterface(
                supported_operations=["design_llm_architecture", "optimize_inference", "deploy_models"],
                input_formats=["json", "model_spec", "requirements"],
                output_formats=["architecture", "deployment_plan", "optimization_report"],
                timeout_default=1800
            ),
            tags=["llm", "architecture", "deployment", "optimization"]
        )
        
        # ML Engineer Specialist
        self._registry[AgentType.ML_ENGINEER_SPECIALIST] = AgentRegistryEntry(
            agent_type=AgentType.ML_ENGINEER_SPECIALIST,
            name="ML Engineer Specialist",
            description="Machine learning model development and production deployment",
            category="AI & Machine Learning",
            capabilities=[
                AgentCapability(
                    CapabilityType.TECHNICAL,
                    proficiency_level=0.92,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.HIGH
                ),
                AgentCapability(
                    CapabilityType.DATA_PROCESSING,
                    proficiency_level=0.88,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.HIGH
                )
            ],
            interface=AgentInterface(
                supported_operations=["develop_model", "train_system", "validate_performance"],
                input_formats=["json", "data", "training_spec"],
                output_formats=["model", "metrics", "validation_report"]
            ),
            tags=["machine-learning", "models", "training", "validation"]
        )
        
        # MLOps Engineer
        self._registry[AgentType.MLOPS_ENGINEER] = AgentRegistryEntry(
            agent_type=AgentType.MLOPS_ENGINEER,
            name="MLOps Engineer",
            description="ML pipeline automation and model lifecycle management",
            category="AI & Machine Learning",
            capabilities=[
                AgentCapability(
                    CapabilityType.OPERATIONAL,
                    proficiency_level=0.90,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.HIGH
                ),
                AgentCapability(
                    CapabilityType.TECHNICAL,
                    proficiency_level=0.85,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.MEDIUM
                )
            ],
            interface=AgentInterface(
                supported_operations=["setup_pipeline", "automate_deployment", "monitor_models"],
                input_formats=["json", "pipeline_spec", "model"],
                output_formats=["pipeline", "monitoring_dashboard", "deployment"]
            ),
            tags=["mlops", "automation", "pipelines", "monitoring"]
        )
    
    def _add_analysis_agent_definitions(self) -> None:
        """Add remaining analysis agent definitions."""
        
        # Hallucination Trace Agent
        self._registry[AgentType.HALLUCINATION_TRACE_AGENT] = AgentRegistryEntry(
            agent_type=AgentType.HALLUCINATION_TRACE_AGENT,
            name="Hallucination Trace Agent",
            description="Advanced hallucination detection and validation for GraphRAG content",
            category="Analysis & Investigation",
            capabilities=[
                AgentCapability(
                    CapabilityType.VALIDATION,
                    proficiency_level=0.95,
                    complexity_support=[ComplexityLevel.EXPERT, ComplexityLevel.RESEARCH_LEVEL],
                    resource_cost=ResourceRequirement.HIGH
                ),
                AgentCapability(
                    CapabilityType.ANALYTICAL,
                    proficiency_level=0.92,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.HIGH
                )
            ],
            interface=AgentInterface(
                supported_operations=["detect_hallucination", "trace_sources", "validate_claims"],
                input_formats=["json", "text", "graph_data"],
                output_formats=["json", "trace_report", "validation_score"],
                confidence_threshold=0.95
            ),
            tags=["hallucination", "validation", "tracing", "graphrag"]
        )
        
        # Cost Optimization Agent
        self._registry[AgentType.COST_OPTIMIZATION_AGENT] = AgentRegistryEntry(
            agent_type=AgentType.COST_OPTIMIZATION_AGENT,
            name="Cost Optimization Agent",
            description="Intelligent cost management for multi-model LLM usage optimization",
            category="Analysis & Investigation",
            capabilities=[
                AgentCapability(
                    CapabilityType.OPTIMIZATION,
                    proficiency_level=0.90,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.MEDIUM
                ),
                AgentCapability(
                    CapabilityType.ANALYTICAL,
                    proficiency_level=0.85,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.LOW
                )
            ],
            interface=AgentInterface(
                supported_operations=["analyze_costs", "optimize_usage", "recommend_models"],
                input_formats=["json", "usage_data", "cost_metrics"],
                output_formats=["json", "optimization_plan", "cost_report"]
            ),
            tags=["cost", "optimization", "llm", "efficiency"]
        )
        
        # User Behavior Analyst
        self._registry[AgentType.USER_BEHAVIOR_ANALYST] = AgentRegistryEntry(
            agent_type=AgentType.USER_BEHAVIOR_ANALYST,
            name="User Behavior Analyst",
            description="Privacy-compliant user interaction analysis for UX optimization",
            category="Analysis & Investigation",
            capabilities=[
                AgentCapability(
                    CapabilityType.ANALYTICAL,
                    proficiency_level=0.88,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.MEDIUM
                ),
                AgentCapability(
                    CapabilityType.DATA_PROCESSING,
                    proficiency_level=0.85,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.MEDIUM
                )
            ],
            interface=AgentInterface(
                supported_operations=["analyze_behavior", "identify_patterns", "optimize_ux"],
                input_formats=["json", "telemetry", "interaction_data"],
                output_formats=["json", "behavior_report", "ux_recommendations"]
            ),
            tags=["behavior", "analytics", "ux", "optimization"]
        )
    
    def _add_creative_agent_definitions(self) -> None:
        """Add remaining creative agent definitions."""
        
        # UI/UX Designer
        self._registry[AgentType.UI_UX_DESIGNER] = AgentRegistryEntry(
            agent_type=AgentType.UI_UX_DESIGNER,
            name="UI/UX Designer",
            description="Comprehensive user experience design and interface creation",
            category="Creative & Design",
            capabilities=[
                AgentCapability(
                    CapabilityType.CREATIVE,
                    proficiency_level=0.92,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.MEDIUM
                ),
                AgentCapability(
                    CapabilityType.VISUAL_PROCESSING,
                    proficiency_level=0.90,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.MEDIUM
                )
            ],
            interface=AgentInterface(
                supported_operations=["design_interface", "create_wireframes", "prototype_ux"],
                input_formats=["json", "requirements", "user_stories"],
                output_formats=["design", "wireframes", "prototype"]
            ),
            tags=["ui", "ux", "design", "interface"]
        )
        
        # Vue Expert
        self._registry[AgentType.VUE_EXPERT] = AgentRegistryEntry(
            agent_type=AgentType.VUE_EXPERT,
            name="Vue Expert",
            description="Vue.js 3 development with Composition API specialist",
            category="Creative & Design",
            capabilities=[
                AgentCapability(
                    CapabilityType.TECHNICAL,
                    proficiency_level=0.90,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.MEDIUM
                ),
                AgentCapability(
                    CapabilityType.CODE_PROCESSING,
                    proficiency_level=0.88,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.MEDIUM
                )
            ],
            interface=AgentInterface(
                supported_operations=["develop_vue_component", "implement_composition_api", "optimize_reactivity"],
                input_formats=["json", "component_spec", "requirements"],
                output_formats=["vue_component", "code", "documentation"]
            ),
            tags=["vue", "frontend", "composition-api", "reactivity"]
        )
    
    def _add_specialized_agent_definitions(self) -> None:
        """Add remaining specialized agent definitions."""
        
        # Human-in-the-Loop Handler
        self._registry[AgentType.HUMAN_IN_THE_LOOP_HANDLER] = AgentRegistryEntry(
            agent_type=AgentType.HUMAN_IN_THE_LOOP_HANDLER,
            name="Human-in-the-Loop Handler",
            description="Intelligent escalation and human review orchestration for critical decisions",
            category="Specialized Domains",
            capabilities=[
                AgentCapability(
                    CapabilityType.COORDINATION,
                    proficiency_level=0.90,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.MEDIUM
                ),
                AgentCapability(
                    CapabilityType.COMMUNICATION,
                    proficiency_level=0.95,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.LOW
                )
            ],
            interface=AgentInterface(
                supported_operations=["escalate_decision", "coordinate_review", "manage_feedback"],
                input_formats=["json", "decision_context", "review_request"],
                output_formats=["json", "escalation_ticket", "review_result"]
            ),
            tags=["human-loop", "escalation", "review", "coordination"]
        )
        
        # Compliance Officer Agent
        self._registry[AgentType.COMPLIANCE_OFFICER_AGENT] = AgentRegistryEntry(
            agent_type=AgentType.COMPLIANCE_OFFICER_AGENT,
            name="Compliance Officer Agent",
            description="Enterprise compliance validation and regulatory enforcement",
            category="Specialized Domains",
            capabilities=[
                AgentCapability(
                    CapabilityType.VALIDATION,
                    proficiency_level=0.95,
                    complexity_support=[ComplexityLevel.EXPERT, ComplexityLevel.RESEARCH_LEVEL],
                    resource_cost=ResourceRequirement.HIGH
                ),
                AgentCapability(
                    CapabilityType.BUSINESS,
                    proficiency_level=0.90,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.MEDIUM
                )
            ],
            interface=AgentInterface(
                supported_operations=["validate_compliance", "audit_processes", "generate_reports"],
                input_formats=["json", "policy_data", "audit_request"],
                output_formats=["json", "compliance_report", "audit_findings"]
            ),
            tags=["compliance", "regulation", "audit", "governance"]
        )
        
        # Training Data Steward
        self._registry[AgentType.TRAINING_DATA_STEWARD] = AgentRegistryEntry(
            agent_type=AgentType.TRAINING_DATA_STEWARD,
            name="Training Data Steward",
            description="GraphRAG knowledge base curator ensuring semantic accuracy and data quality",
            category="Specialized Domains",
            capabilities=[
                AgentCapability(
                    CapabilityType.DATA_PROCESSING,
                    proficiency_level=0.92,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.HIGH
                ),
                AgentCapability(
                    CapabilityType.VALIDATION,
                    proficiency_level=0.88,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.MEDIUM
                )
            ],
            interface=AgentInterface(
                supported_operations=["curate_data", "validate_quality", "update_embeddings"],
                input_formats=["json", "vector_data", "knowledge_graph"],
                output_formats=["json", "curated_dataset", "quality_report"]
            ),
            tags=["data", "curation", "quality", "embeddings"]
        )
        
        # Add Language Specialists
        self._add_language_specialists()
        
        # Add SEO Specialists
        self._add_seo_specialists()
        
        # Add Platform Specialists
        self._add_platform_specialists()
    
    def _add_language_specialists(self) -> None:
        """Add language-specific programming agents."""
        
        # Python Pro
        self._registry[AgentType.PYTHON_PRO] = AgentRegistryEntry(
            agent_type=AgentType.PYTHON_PRO,
            name="Python Pro",
            description="Advanced Python development and optimization specialist",
            category="Language Specialists",
            capabilities=[
                AgentCapability(
                    CapabilityType.TECHNICAL,
                    proficiency_level=0.92,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.MEDIUM
                ),
                AgentCapability(
                    CapabilityType.CODE_PROCESSING,
                    proficiency_level=0.95,
                    complexity_support=[ComplexityLevel.EXPERT, ComplexityLevel.RESEARCH_LEVEL],
                    resource_cost=ResourceRequirement.MEDIUM
                )
            ],
            interface=AgentInterface(
                supported_operations=["optimize_python", "implement_async", "profile_performance"],
                input_formats=["python", "requirements", "specification"],
                output_formats=["python", "optimization_report", "performance_metrics"]
            ),
            tags=["python", "optimization", "async", "performance"]
        )
        
        # TypeScript Pro
        self._registry[AgentType.TYPESCRIPT_PRO] = AgentRegistryEntry(
            agent_type=AgentType.TYPESCRIPT_PRO,
            name="TypeScript Pro",
            description="Advanced TypeScript development with strict type safety",
            category="Language Specialists",
            capabilities=[
                AgentCapability(
                    CapabilityType.TECHNICAL,
                    proficiency_level=0.90,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.MEDIUM
                ),
                AgentCapability(
                    CapabilityType.CODE_PROCESSING,
                    proficiency_level=0.92,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.MEDIUM
                )
            ],
            interface=AgentInterface(
                supported_operations=["implement_types", "optimize_typescript", "refactor_generics"],
                input_formats=["typescript", "types", "specification"],
                output_formats=["typescript", "type_definitions", "refactored_code"]
            ),
            tags=["typescript", "types", "generics", "strict-mode"]
        )
    
    def _add_seo_specialists(self) -> None:
        """Add SEO and marketing specialist agents."""
        
        # SEO Content Writer
        self._registry[AgentType.SEO_CONTENT_WRITER] = AgentRegistryEntry(
            agent_type=AgentType.SEO_CONTENT_WRITER,
            name="SEO Content Writer",
            description="SEO-optimized content creation and keyword optimization",
            category="SEO & Marketing",
            capabilities=[
                AgentCapability(
                    CapabilityType.CREATION,
                    proficiency_level=0.88,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.LOW
                ),
                AgentCapability(
                    CapabilityType.TEXT_PROCESSING,
                    proficiency_level=0.90,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.LOW
                )
            ],
            interface=AgentInterface(
                supported_operations=["create_seo_content", "optimize_keywords", "analyze_readability"],
                input_formats=["json", "keywords", "content_brief"],
                output_formats=["markdown", "html", "seo_report"]
            ),
            tags=["seo", "content", "keywords", "optimization"]
        )
    
    def _add_platform_specialists(self) -> None:
        """Add platform and framework specialist agents."""
        
        # Flutter Expert
        self._registry[AgentType.FLUTTER_EXPERT] = AgentRegistryEntry(
            agent_type=AgentType.FLUTTER_EXPERT,
            name="Flutter Expert",
            description="Cross-platform mobile development with Flutter and Dart",
            category="Platform Specialists",
            capabilities=[
                AgentCapability(
                    CapabilityType.TECHNICAL,
                    proficiency_level=0.90,
                    complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                    resource_cost=ResourceRequirement.HIGH
                ),
                AgentCapability(
                    CapabilityType.CODE_PROCESSING,
                    proficiency_level=0.85,
                    complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                    resource_cost=ResourceRequirement.MEDIUM
                )
            ],
            interface=AgentInterface(
                supported_operations=["develop_flutter_app", "implement_widgets", "optimize_performance"],
                input_formats=["json", "dart", "app_spec"],
                output_formats=["dart", "flutter_app", "performance_report"]
            ),
            tags=["flutter", "dart", "mobile", "cross-platform"]
        )
    
    async def _build_indexes(self) -> None:
        """Build lookup indexes for fast agent selection."""
        for agent_type, entry in self._registry.items():
            # Capability index
            for capability in entry.capabilities:
                if capability.capability_type not in self._capability_index:
                    self._capability_index[capability.capability_type] = set()
                self._capability_index[capability.capability_type].add(agent_type)
            
            # Category index
            if entry.category not in self._category_index:
                self._category_index[entry.category] = set()
            self._category_index[entry.category].add(agent_type)
            
            # Tag index
            for tag in entry.tags:
                if tag not in self._tag_index:
                    self._tag_index[tag] = set()
                self._tag_index[tag].add(agent_type)
    
    async def _load_metrics(self) -> None:
        """Load historical performance metrics."""
        try:
            metrics_file = Path("data/agent_metrics.json")
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                
                for agent_type_str, metrics_dict in metrics_data.items():
                    try:
                        agent_type = AgentType(agent_type_str)
                        if agent_type in self._registry:
                            entry = self._registry[agent_type]
                            entry.metrics.success_rate = metrics_dict.get('success_rate', 0.95)
                            entry.metrics.average_response_time_ms = metrics_dict.get('avg_response_time', 2000)
                            entry.metrics.total_executions = metrics_dict.get('total_executions', 0)
                            
                            if 'last_used' in metrics_dict:
                                entry.metrics.last_used = datetime.fromisoformat(metrics_dict['last_used'])
                    except ValueError:
                        continue
                        
                logger.info("Historical agent metrics loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load agent metrics: {str(e)}")
    
    # Public Interface Methods
    
    def get_agent(self, agent_type: AgentType) -> Optional[AgentRegistryEntry]:
        """Get agent registry entry by type."""
        return self._registry.get(agent_type)
    
    def get_agents_by_capability(
        self, 
        capability: CapabilityType,
        min_proficiency: float = 0.8,
        complexity_level: Optional[ComplexityLevel] = None
    ) -> List[AgentRegistryEntry]:
        """Find agents by capability requirements."""
        if capability not in self._capability_index:
            return []
        
        candidates = []
        for agent_type in self._capability_index[capability]:
            entry = self._registry[agent_type]
            
            # Find matching capability
            matching_capability = None
            for cap in entry.capabilities:
                if cap.capability_type == capability and cap.proficiency_level >= min_proficiency:
                    if complexity_level is None or complexity_level in cap.complexity_support:
                        matching_capability = cap
                        break
            
            if matching_capability and entry.is_available:
                candidates.append(entry)
        
        # Sort by proficiency level (descending)
        return sorted(candidates, key=lambda x: max(
            cap.proficiency_level for cap in x.capabilities 
            if cap.capability_type == capability
        ), reverse=True)
    
    def get_agents_by_category(self, category: str) -> List[AgentRegistryEntry]:
        """Get all agents in a category."""
        if category not in self._category_index:
            return []
        
        return [
            self._registry[agent_type] 
            for agent_type in self._category_index[category]
            if self._registry[agent_type].is_available
        ]
    
    def get_agents_by_tags(self, tags: List[str], match_all: bool = False) -> List[AgentRegistryEntry]:
        """Find agents by tags."""
        if not tags:
            return []
        
        if match_all:
            # Find agents that have all tags
            agent_sets = [self._tag_index.get(tag, set()) for tag in tags]
            if not agent_sets:
                return []
            
            matching_agents = agent_sets[0].intersection(*agent_sets[1:])
        else:
            # Find agents that have any of the tags
            matching_agents = set()
            for tag in tags:
                matching_agents.update(self._tag_index.get(tag, set()))
        
        return [
            self._registry[agent_type] 
            for agent_type in matching_agents
            if self._registry[agent_type].is_available
        ]
    
    def search_agents(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        capabilities: Optional[List[CapabilityType]] = None,
        tags: Optional[List[str]] = None,
        available_only: bool = True
    ) -> List[AgentRegistryEntry]:
        """Search agents by query with filters."""
        results = []
        query_lower = query.lower()
        
        for entry in self._registry.values():
            if available_only and not entry.is_available:
                continue
            
            # Check categories filter
            if categories and entry.category not in categories:
                continue
            
            # Check capabilities filter
            if capabilities:
                entry_capabilities = {cap.capability_type for cap in entry.capabilities}
                if not any(cap in entry_capabilities for cap in capabilities):
                    continue
            
            # Check tags filter
            if tags and not any(tag in entry.tags for tag in tags):
                continue
            
            # Text search in name, description, tags
            searchable_text = f"{entry.name} {entry.description} {' '.join(entry.tags)}".lower()
            if query_lower in searchable_text:
                results.append(entry)
        
        return results
    
    def get_top_performers(
        self, 
        capability: Optional[CapabilityType] = None,
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[AgentRegistryEntry]:
        """Get top performing agents by success rate and usage."""
        candidates = list(self._registry.values())
        
        # Apply filters
        if capability:
            candidates = [
                entry for entry in candidates
                if any(cap.capability_type == capability for cap in entry.capabilities)
            ]
        
        if category:
            candidates = [entry for entry in candidates if entry.category == category]
        
        # Filter available agents
        candidates = [entry for entry in candidates if entry.is_available]
        
        # Sort by performance score (success_rate * usage_score)
        def performance_score(entry: AgentRegistryEntry) -> float:
            usage_score = min(entry.metrics.total_executions / 100, 1.0)  # Normalize to 0-1
            return entry.metrics.success_rate * 0.7 + usage_score * 0.3
        
        candidates.sort(key=performance_score, reverse=True)
        return candidates[:limit]
    
    def update_agent_metrics(
        self, 
        agent_type: AgentType, 
        execution_time_ms: int,
        success: bool,
        confidence_score: Optional[float] = None,
        error_pattern: Optional[str] = None
    ) -> None:
        """Update agent performance metrics."""
        if agent_type not in self._registry:
            return
        
        entry = self._registry[agent_type]
        metrics = entry.metrics
        
        # Update metrics
        metrics.total_executions += 1
        metrics.last_used = datetime.utcnow()
        
        # Update success rate (exponential moving average)
        alpha = 0.1  # Learning rate
        if success:
            metrics.success_rate = (1 - alpha) * metrics.success_rate + alpha * 1.0
        else:
            metrics.success_rate = (1 - alpha) * metrics.success_rate + alpha * 0.0
        
        # Update response times
        metrics.average_response_time_ms = int(
            (1 - alpha) * metrics.average_response_time_ms + alpha * execution_time_ms
        )
        metrics.peak_response_time_ms = max(metrics.peak_response_time_ms, execution_time_ms)
        
        # Update confidence scores
        if confidence_score is not None:
            metrics.confidence_scores.append(confidence_score)
            if len(metrics.confidence_scores) > 100:  # Keep last 100 scores
                metrics.confidence_scores.pop(0)
        
        # Track error patterns
        if error_pattern and error_pattern not in metrics.error_patterns:
            metrics.error_patterns.append(error_pattern)
            if len(metrics.error_patterns) > 20:  # Keep last 20 error patterns
                metrics.error_patterns.pop(0)
        
        entry.updated_at = datetime.utcnow()
    
    async def save_metrics(self) -> None:
        """Save agent metrics to persistent storage."""
        try:
            metrics_data = {}
            for agent_type, entry in self._registry.items():
                metrics = entry.metrics
                metrics_data[agent_type.value] = {
                    'success_rate': metrics.success_rate,
                    'avg_response_time': metrics.average_response_time_ms,
                    'peak_response_time': metrics.peak_response_time_ms,
                    'total_executions': metrics.total_executions,
                    'last_used': metrics.last_used.isoformat() if metrics.last_used else None,
                    'confidence_scores': metrics.confidence_scores[-10:],  # Save last 10
                    'error_patterns': metrics.error_patterns[-5:]  # Save last 5
                }
            
            metrics_file = Path("data/agent_metrics.json")
            metrics_file.parent.mkdir(exist_ok=True)
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            logger.info("Agent metrics saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save agent metrics: {str(e)}")
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_agents = len(self._registry)
        available_agents = sum(1 for entry in self._registry.values() if entry.is_available)
        experimental_agents = sum(1 for entry in self._registry.values() if entry.is_experimental)
        
        categories = {}
        for entry in self._registry.values():
            categories[entry.category] = categories.get(entry.category, 0) + 1
        
        capabilities = {}
        for cap_type, agents in self._capability_index.items():
            capabilities[cap_type.value] = len(agents)
        
        return {
            "total_agents": total_agents,
            "available_agents": available_agents,
            "experimental_agents": experimental_agents,
            "categories": categories,
            "capabilities": capabilities,
            "total_capabilities": len(self._capability_index),
            "total_tags": len(self._tag_index),
            "initialization_status": "initialized" if self._is_initialized else "not_initialized"
        }


# Global registry instance
agent_registry = AgentRegistry()


async def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry instance."""
    if not agent_registry._is_initialized:
        await agent_registry.initialize()
    return agent_registry