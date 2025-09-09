"""
Context-Aware Agent Selection Logic for Multi-Agent Orchestration System.

Implements intelligent agent selection based on task context, agent capabilities,
system constraints, and performance metrics for optimal workflow execution.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from enum import Enum
import math
import structlog
from dataclasses import dataclass, field
from collections import defaultdict

from pydantic import BaseModel, Field
from core.config import get_settings
from services.agent_registry import (
    AgentRegistry, 
    AgentRegistryEntry, 
    CapabilityType, 
    ComplexityLevel,
    ResourceRequirement,
    get_agent_registry
)
from services.agent_orchestrator import AgentType, TaskPriority

logger = structlog.get_logger(__name__)
settings = get_settings()


class TaskContext(str, Enum):
    """Types of task contexts for agent selection."""
    PRD_GENERATION = "prd_generation"
    TECHNICAL_ANALYSIS = "technical_analysis"
    BUSINESS_ANALYSIS = "business_analysis"
    CONTENT_CREATION = "content_creation"
    CODE_DEVELOPMENT = "code_development"
    SYSTEM_DESIGN = "system_design"
    QUALITY_ASSURANCE = "quality_assurance"
    DOCUMENTATION = "documentation"
    STRATEGIC_PLANNING = "strategic_planning"
    RESEARCH = "research"
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"


class SelectionCriteria(str, Enum):
    """Criteria for agent selection prioritization."""
    CAPABILITY_MATCH = "capability_match"
    PERFORMANCE_HISTORY = "performance_history"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    AVAILABILITY = "availability"
    SPECIALIZATION = "specialization"
    COLLABORATIVE_AFFINITY = "collaborative_affinity"


@dataclass
class TaskRequirements:
    """Requirements for task execution."""
    task_context: TaskContext
    required_capabilities: List[CapabilityType]
    complexity_level: ComplexityLevel
    estimated_tokens: int
    max_execution_time: timedelta
    quality_threshold: float = 0.85
    resource_constraints: Optional[Dict[str, Any]] = None
    domain_knowledge: List[str] = field(default_factory=list)
    preferred_agents: List[AgentType] = field(default_factory=list)
    excluded_agents: List[AgentType] = field(default_factory=list)
    collaboration_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentScore:
    """Scoring information for agent selection."""
    agent_type: AgentType
    total_score: float
    capability_score: float
    performance_score: float
    availability_score: float
    resource_score: float
    context_score: float
    collaboration_score: float
    confidence: float
    reasoning: List[str] = field(default_factory=list)


@dataclass 
class SelectionResult:
    """Result of agent selection process."""
    selected_agents: List[AgentType]
    scores: List[AgentScore]
    execution_plan: Dict[str, Any]
    resource_allocation: Dict[str, Any]
    estimated_completion_time: timedelta
    confidence_level: float
    fallback_options: List[AgentType] = field(default_factory=list)
    selection_reasoning: List[str] = field(default_factory=list)


class SystemResourceMonitor:
    """Monitors system resources and constraints."""
    
    def __init__(self):
        self.active_agents: Dict[AgentType, int] = defaultdict(int)
        self.resource_usage: Dict[str, float] = defaultdict(float)
        self.performance_history: Dict[AgentType, List[float]] = defaultdict(list)
        self.last_update: datetime = datetime.utcnow()
    
    def get_agent_availability(self, agent_type: AgentType) -> float:
        """Get availability score for an agent (0.0 - 1.0)."""
        active_count = self.active_agents.get(agent_type, 0)
        # Assume max 3 concurrent instances per agent type
        max_concurrent = 3
        availability = max(0.0, (max_concurrent - active_count) / max_concurrent)
        return availability
    
    def get_resource_pressure(self) -> float:
        """Get current resource pressure (0.0 - 1.0)."""
        cpu_usage = self.resource_usage.get('cpu', 0.0)
        memory_usage = self.resource_usage.get('memory', 0.0)
        return max(cpu_usage, memory_usage)
    
    def update_agent_status(self, agent_type: AgentType, is_active: bool):
        """Update agent activity status."""
        if is_active:
            self.active_agents[agent_type] += 1
        else:
            self.active_agents[agent_type] = max(0, self.active_agents[agent_type] - 1)
        
        self.last_update = datetime.utcnow()


class ContextAwareAgentSelector:
    """
    Context-aware agent selection system for intelligent multi-agent orchestration.
    
    Selects optimal agents based on task requirements, system constraints,
    and performance characteristics using sophisticated scoring algorithms.
    """
    
    def __init__(self):
        self.registry: Optional[AgentRegistry] = None
        self.resource_monitor = SystemResourceMonitor()
        self.context_patterns: Dict[TaskContext, Dict[str, Any]] = {}
        self.collaboration_history: Dict[Tuple[AgentType, AgentType], float] = {}
        self.is_initialized = False
        
        # Selection weights for different criteria
        self.selection_weights = {
            SelectionCriteria.CAPABILITY_MATCH: 0.25,
            SelectionCriteria.PERFORMANCE_HISTORY: 0.20,
            SelectionCriteria.RESOURCE_EFFICIENCY: 0.15,
            SelectionCriteria.AVAILABILITY: 0.15,
            SelectionCriteria.SPECIALIZATION: 0.15,
            SelectionCriteria.COLLABORATIVE_AFFINITY: 0.10
        }
    
    async def initialize(self) -> None:
        """Initialize the agent selector with registry and patterns."""
        try:
            self.registry = await get_agent_registry()
            await self._initialize_context_patterns()
            await self._load_collaboration_history()
            
            self.is_initialized = True
            logger.info("Context-aware agent selector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent selector: {str(e)}")
            raise
    
    async def _initialize_context_patterns(self) -> None:
        """Initialize context-specific selection patterns."""
        self.context_patterns = {
            TaskContext.PRD_GENERATION: {
                "preferred_capabilities": [
                    CapabilityType.CREATION,
                    CapabilityType.STRATEGIC,
                    CapabilityType.BUSINESS
                ],
                "collaboration_patterns": ["sequential", "validation"],
                "quality_emphasis": 0.9
            },
            TaskContext.TECHNICAL_ANALYSIS: {
                "preferred_capabilities": [
                    CapabilityType.ANALYTICAL,
                    CapabilityType.TECHNICAL,
                    CapabilityType.VALIDATION
                ],
                "collaboration_patterns": ["parallel", "peer_review"],
                "quality_emphasis": 0.95
            },
            TaskContext.CODE_DEVELOPMENT: {
                "preferred_capabilities": [
                    CapabilityType.TECHNICAL,
                    CapabilityType.CODE_PROCESSING,
                    CapabilityType.CREATION
                ],
                "collaboration_patterns": ["pipeline", "review"],
                "quality_emphasis": 0.92
            },
            TaskContext.CONTENT_CREATION: {
                "preferred_capabilities": [
                    CapabilityType.CREATION,
                    CapabilityType.TEXT_PROCESSING,
                    CapabilityType.CREATIVE
                ],
                "collaboration_patterns": ["draft_review", "iterative"],
                "quality_emphasis": 0.85
            }
        }
    
    async def _load_collaboration_history(self) -> None:
        """Load historical collaboration performance data."""
        # In a real implementation, this would load from persistent storage
        # For now, initialize with some default patterns
        high_synergy_pairs = [
            (AgentType.BUSINESS_ANALYST, AgentType.PROJECT_ARCHITECT),
            (AgentType.DRAFT_AGENT, AgentType.JUDGE_AGENT),
            (AgentType.FRONTEND_DEVELOPER, AgentType.UI_UX_DESIGNER),
            (AgentType.BACKEND_DEVELOPER, AgentType.BACKEND_ARCHITECT),
            (AgentType.AI_ENGINEER, AgentType.ML_ENGINEER_SPECIALIST)
        ]
        
        for agent1, agent2 in high_synergy_pairs:
            self.collaboration_history[(agent1, agent2)] = 0.9
            self.collaboration_history[(agent2, agent1)] = 0.9
    
    async def select_agents(
        self,
        requirements: TaskRequirements,
        max_agents: int = 5,
        include_fallbacks: bool = True
    ) -> SelectionResult:
        """
        Select optimal agents for task execution based on requirements.
        
        Args:
            requirements: Task requirements and constraints
            max_agents: Maximum number of agents to select
            include_fallbacks: Whether to include fallback options
            
        Returns:
            SelectionResult with selected agents and execution plan
        """
        if not self.is_initialized:
            raise RuntimeError("Agent selector not initialized")
        
        logger.info(
            "Starting agent selection",
            task_context=requirements.task_context,
            complexity=requirements.complexity_level,
            max_agents=max_agents
        )
        
        # Step 1: Get candidate agents based on capabilities
        candidates = await self._get_candidate_agents(requirements)
        
        if not candidates:
            raise ValueError("No suitable candidate agents found for requirements")
        
        # Step 2: Score all candidates
        scored_candidates = await self._score_candidates(candidates, requirements)
        
        # Step 3: Select optimal combination
        selection = await self._select_optimal_combination(
            scored_candidates, requirements, max_agents
        )
        
        # Step 4: Generate execution plan
        execution_plan = await self._generate_execution_plan(
            selection.selected_agents, requirements
        )
        
        # Step 5: Add fallback options if requested
        fallbacks = []
        if include_fallbacks:
            fallbacks = await self._identify_fallback_options(
                selection.selected_agents, scored_candidates
            )
        
        result = SelectionResult(
            selected_agents=selection.selected_agents,
            scores=selection.scores,
            execution_plan=execution_plan,
            resource_allocation=await self._calculate_resource_allocation(selection.selected_agents),
            estimated_completion_time=await self._estimate_completion_time(
                selection.selected_agents, requirements
            ),
            confidence_level=selection.confidence_level,
            fallback_options=fallbacks,
            selection_reasoning=selection.selection_reasoning
        )
        
        logger.info(
            "Agent selection completed",
            selected_agents=[agent.value for agent in result.selected_agents],
            confidence=result.confidence_level,
            estimated_time=result.estimated_completion_time
        )
        
        return result
    
    async def _get_candidate_agents(
        self, 
        requirements: TaskRequirements
    ) -> List[AgentRegistryEntry]:
        """Get candidate agents based on capability requirements."""
        candidates = []
        
        # Get agents for each required capability
        for capability in requirements.required_capabilities:
            capability_agents = self.registry.get_agents_by_capability(
                capability=capability,
                min_proficiency=0.7,
                complexity_level=requirements.complexity_level
            )
            candidates.extend(capability_agents)
        
        # Remove duplicates
        unique_candidates = {}
        for agent in candidates:
            if agent.agent_type not in unique_candidates:
                unique_candidates[agent.agent_type] = agent
        
        # Apply filters
        filtered_candidates = []
        for agent in unique_candidates.values():
            # Check exclusions
            if agent.agent_type in requirements.excluded_agents:
                continue
            
            # Check availability
            if not agent.is_available:
                continue
            
            # Check domain knowledge if specified
            if requirements.domain_knowledge:
                agent_domains = set(agent.tags + [agent.category.lower()])
                required_domains = set(requirements.domain_knowledge)
                if not agent_domains.intersection(required_domains):
                    continue
            
            filtered_candidates.append(agent)
        
        return filtered_candidates
    
    async def _score_candidates(
        self,
        candidates: List[AgentRegistryEntry],
        requirements: TaskRequirements
    ) -> List[AgentScore]:
        """Score all candidate agents based on selection criteria."""
        scores = []
        
        for agent in candidates:
            # Calculate individual criterion scores
            capability_score = await self._calculate_capability_score(agent, requirements)
            performance_score = await self._calculate_performance_score(agent, requirements)
            availability_score = await self._calculate_availability_score(agent)
            resource_score = await self._calculate_resource_score(agent, requirements)
            context_score = await self._calculate_context_score(agent, requirements)
            collaboration_score = await self._calculate_collaboration_score(agent, requirements)
            
            # Calculate weighted total score
            total_score = (
                capability_score * self.selection_weights[SelectionCriteria.CAPABILITY_MATCH] +
                performance_score * self.selection_weights[SelectionCriteria.PERFORMANCE_HISTORY] +
                availability_score * self.selection_weights[SelectionCriteria.AVAILABILITY] +
                resource_score * self.selection_weights[SelectionCriteria.RESOURCE_EFFICIENCY] +
                context_score * self.selection_weights[SelectionCriteria.SPECIALIZATION] +
                collaboration_score * self.selection_weights[SelectionCriteria.COLLABORATIVE_AFFINITY]
            )
            
            # Calculate confidence based on score distribution
            scores_list = [capability_score, performance_score, availability_score, 
                          resource_score, context_score, collaboration_score]
            confidence = 1.0 - (max(scores_list) - min(scores_list))
            
            # Generate reasoning
            reasoning = []
            if capability_score > 0.8:
                reasoning.append("Strong capability match")
            if performance_score > 0.8:
                reasoning.append("Excellent performance history")
            if availability_score > 0.9:
                reasoning.append("High availability")
            if context_score > 0.8:
                reasoning.append("Well-suited for task context")
            
            agent_score = AgentScore(
                agent_type=agent.agent_type,
                total_score=total_score,
                capability_score=capability_score,
                performance_score=performance_score,
                availability_score=availability_score,
                resource_score=resource_score,
                context_score=context_score,
                collaboration_score=collaboration_score,
                confidence=confidence,
                reasoning=reasoning
            )
            
            scores.append(agent_score)
        
        # Sort by total score (descending)
        scores.sort(key=lambda x: x.total_score, reverse=True)
        
        return scores
    
    async def _calculate_capability_score(
        self, 
        agent: AgentRegistryEntry, 
        requirements: TaskRequirements
    ) -> float:
        """Calculate capability match score for an agent."""
        if not agent.capabilities:
            return 0.0
        
        total_score = 0.0
        capability_count = 0
        
        for required_cap in requirements.required_capabilities:
            best_match_score = 0.0
            
            for agent_cap in agent.capabilities:
                if agent_cap.capability_type == required_cap:
                    # Base score from proficiency
                    score = agent_cap.proficiency_level
                    
                    # Bonus for complexity support
                    if requirements.complexity_level in agent_cap.complexity_support:
                        score *= 1.2
                    
                    # Bonus for resource efficiency match
                    if (requirements.estimated_tokens < 5000 and 
                        agent_cap.resource_cost in [ResourceRequirement.LOW, ResourceRequirement.MEDIUM]):
                        score *= 1.1
                    
                    best_match_score = max(best_match_score, score)
            
            total_score += best_match_score
            capability_count += 1
        
        return min(total_score / max(capability_count, 1), 1.0)
    
    async def _calculate_performance_score(
        self, 
        agent: AgentRegistryEntry, 
        requirements: TaskRequirements
    ) -> float:
        """Calculate performance history score for an agent."""
        metrics = agent.metrics
        
        # Base success rate score
        success_score = metrics.success_rate
        
        # Response time score (faster is better)
        target_time = requirements.max_execution_time.total_seconds() * 1000  # Convert to ms
        if target_time > 0:
            time_score = max(0.0, 1.0 - (metrics.average_response_time_ms / target_time))
        else:
            time_score = 0.5  # Neutral if no time requirement
        
        # Usage experience score (more experience is better, up to a point)
        experience_score = min(metrics.total_executions / 100, 1.0)
        
        # Confidence score from historical data
        confidence_score = 0.5
        if metrics.confidence_scores:
            avg_confidence = sum(metrics.confidence_scores) / len(metrics.confidence_scores)
            confidence_score = avg_confidence
        
        # Combined performance score
        performance_score = (
            success_score * 0.4 +
            time_score * 0.3 +
            experience_score * 0.2 +
            confidence_score * 0.1
        )
        
        return min(performance_score, 1.0)
    
    async def _calculate_availability_score(self, agent: AgentRegistryEntry) -> float:
        """Calculate availability score for an agent."""
        if not agent.is_available:
            return 0.0
        
        # Check system resource monitor
        availability = self.resource_monitor.get_agent_availability(agent.agent_type)
        
        # Factor in maintenance windows or experimental status
        if agent.is_experimental:
            availability *= 0.8
        
        return availability
    
    async def _calculate_resource_score(
        self, 
        agent: AgentRegistryEntry, 
        requirements: TaskRequirements
    ) -> float:
        """Calculate resource efficiency score for an agent."""
        # Get current resource pressure
        resource_pressure = self.resource_monitor.get_resource_pressure()
        
        # Score based on agent's resource requirements vs current pressure
        resource_score = 1.0
        
        for capability in agent.capabilities:
            if capability.capability_type in requirements.required_capabilities:
                cost_factor = {
                    ResourceRequirement.LOW: 0.9,
                    ResourceRequirement.MEDIUM: 0.7,
                    ResourceRequirement.HIGH: 0.5,
                    ResourceRequirement.INTENSIVE: 0.3
                }.get(capability.resource_cost, 0.5)
                
                # Adjust for current resource pressure
                adjusted_score = cost_factor * (1.0 - resource_pressure * 0.5)
                resource_score = min(resource_score, adjusted_score)
        
        return max(resource_score, 0.1)
    
    async def _calculate_context_score(
        self, 
        agent: AgentRegistryEntry, 
        requirements: TaskRequirements
    ) -> float:
        """Calculate context specialization score for an agent."""
        context_patterns = self.context_patterns.get(requirements.task_context, {})
        
        if not context_patterns:
            return 0.5  # Neutral score if no pattern defined
        
        score = 0.5  # Base score
        
        # Check for preferred capabilities
        preferred_caps = context_patterns.get("preferred_capabilities", [])
        agent_caps = [cap.capability_type for cap in agent.capabilities]
        
        matching_caps = len(set(preferred_caps).intersection(set(agent_caps)))
        if preferred_caps:
            score += (matching_caps / len(preferred_caps)) * 0.3
        
        # Check domain alignment
        context_domains = {
            TaskContext.PRD_GENERATION: ["strategic", "business", "planning"],
            TaskContext.TECHNICAL_ANALYSIS: ["technical", "analysis", "engineering"],
            TaskContext.CODE_DEVELOPMENT: ["development", "engineering", "technical"],
            TaskContext.CONTENT_CREATION: ["content", "creative", "writing"]
        }.get(requirements.task_context, [])
        
        agent_domains = set(tag.lower() for tag in agent.tags)
        domain_overlap = len(set(context_domains).intersection(agent_domains))
        if context_domains:
            score += (domain_overlap / len(context_domains)) * 0.2
        
        return min(score, 1.0)
    
    async def _calculate_collaboration_score(
        self, 
        agent: AgentRegistryEntry, 
        requirements: TaskRequirements
    ) -> float:
        """Calculate collaboration affinity score for an agent."""
        if not requirements.preferred_agents:
            return 0.5  # Neutral if no collaboration context
        
        collaboration_scores = []
        
        for preferred_agent in requirements.preferred_agents:
            if preferred_agent == agent.agent_type:
                collaboration_scores.append(1.0)  # Perfect self-match
                continue
            
            # Check historical collaboration performance
            pair_key = (agent.agent_type, preferred_agent)
            reverse_pair_key = (preferred_agent, agent.agent_type)
            
            historical_score = (
                self.collaboration_history.get(pair_key, 0.5) +
                self.collaboration_history.get(reverse_pair_key, 0.5)
            ) / 2
            
            collaboration_scores.append(historical_score)
        
        # Check for known conflicts
        conflict_penalty = 0.0
        for capability in agent.capabilities:
            # This could be expanded to include actual conflict data
            pass
        
        avg_collaboration_score = sum(collaboration_scores) / len(collaboration_scores)
        return max(avg_collaboration_score - conflict_penalty, 0.0)
    
    async def _select_optimal_combination(
        self,
        scored_candidates: List[AgentScore],
        requirements: TaskRequirements,
        max_agents: int
    ) -> SelectionResult:
        """Select optimal combination of agents using constraint optimization."""
        if not scored_candidates:
            raise ValueError("No scored candidates available")
        
        # For simple cases, use greedy selection
        if len(scored_candidates) <= max_agents:
            selected_agents = [score.agent_type for score in scored_candidates]
            confidence = sum(score.confidence for score in scored_candidates) / len(scored_candidates)
            reasoning = ["Selected all available qualified candidates"]
        else:
            # Use more sophisticated selection for larger candidate pools
            selected_agents, confidence, reasoning = await self._optimize_agent_selection(
                scored_candidates, requirements, max_agents
            )
        
        # Get scores for selected agents
        selected_scores = [
            score for score in scored_candidates 
            if score.agent_type in selected_agents
        ]
        
        return SelectionResult(
            selected_agents=selected_agents,
            scores=selected_scores,
            execution_plan={},  # Will be filled by caller
            resource_allocation={},  # Will be filled by caller
            estimated_completion_time=timedelta(),  # Will be filled by caller
            confidence_level=confidence,
            selection_reasoning=reasoning
        )
    
    async def _optimize_agent_selection(
        self,
        scored_candidates: List[AgentScore],
        requirements: TaskRequirements,
        max_agents: int
    ) -> Tuple[List[AgentType], float, List[str]]:
        """Optimize agent selection using advanced algorithms."""
        
        # Simple greedy approach: select top-scoring agents with diversity
        selected = []
        used_categories = set()
        reasoning = []
        
        for candidate in scored_candidates:
            if len(selected) >= max_agents:
                break
            
            agent_entry = self.registry.get_agent(candidate.agent_type)
            if not agent_entry:
                continue
            
            # Prefer diversity in categories for better coverage
            category_bonus = 0.0 if agent_entry.category in used_categories else 0.1
            adjusted_score = candidate.total_score + category_bonus
            
            if len(selected) == 0 or adjusted_score > 0.7:
                selected.append(candidate.agent_type)
                used_categories.add(agent_entry.category)
                
                if category_bonus > 0:
                    reasoning.append(f"Selected {candidate.agent_type.value} for category diversity")
                else:
                    reasoning.append(f"Selected {candidate.agent_type.value} for high score ({adjusted_score:.2f})")
        
        # Calculate overall confidence
        selected_scores = [
            score for score in scored_candidates 
            if score.agent_type in selected
        ]
        confidence = sum(score.confidence for score in selected_scores) / len(selected_scores)
        
        return selected, confidence, reasoning
    
    async def _generate_execution_plan(
        self,
        selected_agents: List[AgentType],
        requirements: TaskRequirements
    ) -> Dict[str, Any]:
        """Generate execution plan for selected agents."""
        plan = {
            "execution_strategy": "parallel" if len(selected_agents) > 1 else "sequential",
            "phases": [],
            "coordination_points": [],
            "quality_gates": [],
            "resource_requirements": {}
        }
        
        # Analyze task context to determine execution strategy
        context_patterns = self.context_patterns.get(requirements.task_context, {})
        collaboration_patterns = context_patterns.get("collaboration_patterns", ["parallel"])
        
        if "sequential" in collaboration_patterns:
            plan["execution_strategy"] = "sequential"
            for i, agent in enumerate(selected_agents):
                plan["phases"].append({
                    "phase": i + 1,
                    "agent": agent.value,
                    "dependencies": [i] if i > 0 else []
                })
        elif "pipeline" in collaboration_patterns:
            plan["execution_strategy"] = "pipeline"
            # Create pipeline phases
        else:
            plan["execution_strategy"] = "parallel"
            plan["phases"] = [{
                "phase": 1,
                "agents": [agent.value for agent in selected_agents],
                "dependencies": []
            }]
        
        return plan
    
    async def _calculate_resource_allocation(
        self, 
        selected_agents: List[AgentType]
    ) -> Dict[str, Any]:
        """Calculate resource allocation for selected agents."""
        allocation = {
            "cpu_cores": 0.0,
            "memory_mb": 0.0,
            "concurrent_limit": len(selected_agents),
            "estimated_cost": 0.0
        }
        
        for agent_type in selected_agents:
            agent_entry = self.registry.get_agent(agent_type)
            if not agent_entry:
                continue
            
            # Estimate resource needs based on capabilities
            for capability in agent_entry.capabilities:
                cost_multiplier = {
                    ResourceRequirement.LOW: 0.5,
                    ResourceRequirement.MEDIUM: 1.0,
                    ResourceRequirement.HIGH: 2.0,
                    ResourceRequirement.INTENSIVE: 4.0
                }.get(capability.resource_cost, 1.0)
                
                allocation["cpu_cores"] += 1.0 * cost_multiplier
                allocation["memory_mb"] += 512 * cost_multiplier
                allocation["estimated_cost"] += 0.01 * cost_multiplier
        
        return allocation
    
    async def _estimate_completion_time(
        self,
        selected_agents: List[AgentType],
        requirements: TaskRequirements
    ) -> timedelta:
        """Estimate completion time for selected agents."""
        if not selected_agents:
            return timedelta(minutes=5)  # Default estimate
        
        # Get average response times
        total_time_ms = 0
        for agent_type in selected_agents:
            agent_entry = self.registry.get_agent(agent_type)
            if agent_entry:
                total_time_ms += agent_entry.metrics.average_response_time_ms
        
        avg_time_ms = total_time_ms / len(selected_agents)
        
        # Apply complexity multiplier
        complexity_multiplier = {
            ComplexityLevel.SIMPLE: 0.5,
            ComplexityLevel.MODERATE: 1.0,
            ComplexityLevel.COMPLEX: 2.0,
            ComplexityLevel.EXPERT: 3.0,
            ComplexityLevel.RESEARCH_LEVEL: 5.0
        }.get(requirements.complexity_level, 1.0)
        
        estimated_ms = avg_time_ms * complexity_multiplier
        return timedelta(milliseconds=estimated_ms)
    
    async def _identify_fallback_options(
        self,
        selected_agents: List[AgentType],
        all_scored_candidates: List[AgentScore]
    ) -> List[AgentType]:
        """Identify fallback options in case primary selection fails."""
        fallbacks = []
        
        for candidate in all_scored_candidates:
            if candidate.agent_type not in selected_agents:
                if candidate.total_score > 0.6 and len(fallbacks) < 3:
                    fallbacks.append(candidate.agent_type)
        
        return fallbacks
    
    async def update_performance_metrics(
        self,
        agent_type: AgentType,
        execution_time_ms: int,
        success: bool,
        quality_score: Optional[float] = None
    ) -> None:
        """Update performance metrics for an agent after execution."""
        if self.registry:
            confidence_score = quality_score if quality_score is not None else (0.9 if success else 0.3)
            
            self.registry.update_agent_metrics(
                agent_type=agent_type,
                execution_time_ms=execution_time_ms,
                success=success,
                confidence_score=confidence_score
            )
            
            # Update resource monitor
            self.resource_monitor.update_agent_status(agent_type, False)
    
    async def update_collaboration_history(
        self,
        agent_pairs: List[Tuple[AgentType, AgentType]],
        success_rate: float
    ) -> None:
        """Update collaboration history for agent pairs."""
        for agent1, agent2 in agent_pairs:
            current_score = self.collaboration_history.get((agent1, agent2), 0.5)
            # Exponential moving average update
            alpha = 0.1
            new_score = (1 - alpha) * current_score + alpha * success_rate
            self.collaboration_history[(agent1, agent2)] = new_score


# Global selector instance
context_aware_selector = ContextAwareAgentSelector()


async def get_context_aware_selector() -> ContextAwareAgentSelector:
    """Get the global context-aware agent selector instance."""
    if not context_aware_selector.is_initialized:
        await context_aware_selector.initialize()
    return context_aware_selector