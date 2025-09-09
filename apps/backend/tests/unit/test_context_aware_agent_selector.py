"""
Unit tests for Context-Aware Agent Selector.

Tests intelligent agent selection logic, scoring algorithms,
and optimization strategies for multi-agent orchestration.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from services.context_aware_agent_selector import (
    ContextAwareAgentSelector,
    TaskRequirements,
    TaskContext,
    AgentScore,
    SelectionResult,
    SystemResourceMonitor,
    get_context_aware_selector
)
from services.agent_registry import (
    AgentRegistryEntry,
    AgentCapability,
    AgentInterface,
    AgentMetrics,
    CapabilityType,
    ComplexityLevel,
    ResourceRequirement
)
from services.agent_orchestrator import AgentType, TaskPriority


@pytest.fixture
def mock_agent_registry():
    """Mock agent registry with test agents."""
    registry = MagicMock()
    
    # Create test agent entries
    draft_agent = AgentRegistryEntry(
        agent_type=AgentType.DRAFT_AGENT,
        name="Draft Agent",
        description="Creates initial drafts of documents",
        category="Content & Documentation",
        capabilities=[
            AgentCapability(
                CapabilityType.CREATION,
                proficiency_level=0.88,
                complexity_support=[ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE],
                resource_cost=ResourceRequirement.MEDIUM
            ),
            AgentCapability(
                CapabilityType.TEXT_PROCESSING,
                proficiency_level=0.90,
                complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                resource_cost=ResourceRequirement.LOW
            )
        ],
        metrics=AgentMetrics(
            success_rate=0.92,
            average_response_time_ms=2500,
            total_executions=150,
            confidence_scores=[0.85, 0.88, 0.91]
        ),
        tags=["content", "drafting", "writing"]
    )
    
    judge_agent = AgentRegistryEntry(
        agent_type=AgentType.JUDGE_AGENT,
        name="Judge Agent",
        description="Quality assessment and validation",
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
        metrics=AgentMetrics(
            success_rate=0.96,
            average_response_time_ms=1800,
            total_executions=200,
            confidence_scores=[0.92, 0.94, 0.96]
        ),
        tags=["validation", "quality", "assessment"]
    )
    
    business_analyst = AgentRegistryEntry(
        agent_type=AgentType.BUSINESS_ANALYST,
        name="Business Analyst",
        description="Business requirements analysis",
        category="Strategic & Planning",
        capabilities=[
            AgentCapability(
                CapabilityType.BUSINESS,
                proficiency_level=0.92,
                complexity_support=[ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT],
                resource_cost=ResourceRequirement.MEDIUM
            ),
            AgentCapability(
                CapabilityType.ANALYTICAL,
                proficiency_level=0.88,
                complexity_support=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
                resource_cost=ResourceRequirement.MEDIUM
            )
        ],
        metrics=AgentMetrics(
            success_rate=0.89,
            average_response_time_ms=3200,
            total_executions=120,
            confidence_scores=[0.87, 0.89, 0.91]
        ),
        tags=["business", "analysis", "requirements"]
    )
    
    # Configure mock registry methods
    registry.get_agents_by_capability.side_effect = lambda capability, **kwargs: {
        CapabilityType.CREATION: [draft_agent],
        CapabilityType.VALIDATION: [judge_agent],
        CapabilityType.BUSINESS: [business_analyst],
        CapabilityType.ANALYTICAL: [judge_agent, business_analyst],
        CapabilityType.TEXT_PROCESSING: [draft_agent]
    }.get(capability, [])
    
    registry.get_agent.side_effect = lambda agent_type: {
        AgentType.DRAFT_AGENT: draft_agent,
        AgentType.JUDGE_AGENT: judge_agent,
        AgentType.BUSINESS_ANALYST: business_analyst
    }.get(agent_type)
    
    registry.update_agent_metrics = MagicMock()
    
    return registry


@pytest.fixture
def agent_selector(mock_agent_registry):
    """Create agent selector with mocked registry."""
    selector = ContextAwareAgentSelector()
    selector.registry = mock_agent_registry
    selector.is_initialized = True
    return selector


@pytest.fixture
def sample_requirements():
    """Sample task requirements for testing."""
    return TaskRequirements(
        task_context=TaskContext.CONTENT_CREATION,
        required_capabilities=[CapabilityType.CREATION, CapabilityType.VALIDATION],
        complexity_level=ComplexityLevel.MODERATE,
        estimated_tokens=5000,
        max_execution_time=timedelta(minutes=10),
        quality_threshold=0.85,
        domain_knowledge=["content", "writing"],
        preferred_agents=[AgentType.DRAFT_AGENT]
    )


class TestSystemResourceMonitor:
    """Test cases for SystemResourceMonitor."""
    
    def test_init(self):
        """Test monitor initialization."""
        monitor = SystemResourceMonitor()
        
        assert monitor.active_agents == {}
        assert monitor.resource_usage == {}
        assert monitor.performance_history == {}
        assert isinstance(monitor.last_update, datetime)
    
    def test_agent_availability_tracking(self):
        """Test agent availability tracking."""
        monitor = SystemResourceMonitor()
        
        # Initially available
        availability = monitor.get_agent_availability(AgentType.DRAFT_AGENT)
        assert availability == 1.0
        
        # Mark as active
        monitor.update_agent_status(AgentType.DRAFT_AGENT, is_active=True)
        availability = monitor.get_agent_availability(AgentType.DRAFT_AGENT)
        assert availability == 2/3  # 2 out of 3 max instances available
        
        # Mark another as active
        monitor.update_agent_status(AgentType.DRAFT_AGENT, is_active=True)
        availability = monitor.get_agent_availability(AgentType.DRAFT_AGENT)
        assert availability == 1/3
        
        # Mark as inactive
        monitor.update_agent_status(AgentType.DRAFT_AGENT, is_active=False)
        availability = monitor.get_agent_availability(AgentType.DRAFT_AGENT)
        assert availability == 2/3
    
    def test_resource_pressure_calculation(self):
        """Test resource pressure calculation."""
        monitor = SystemResourceMonitor()
        
        # No resource usage
        pressure = monitor.get_resource_pressure()
        assert pressure == 0.0
        
        # Set resource usage
        monitor.resource_usage['cpu'] = 0.7
        monitor.resource_usage['memory'] = 0.5
        pressure = monitor.get_resource_pressure()
        assert pressure == 0.7  # Max of cpu and memory


class TestContextAwareAgentSelector:
    """Test cases for ContextAwareAgentSelector."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test selector initialization."""
        selector = ContextAwareAgentSelector()
        
        assert not selector.is_initialized
        assert selector.registry is None
        assert len(selector.context_patterns) == 0
        assert len(selector.collaboration_history) == 0
    
    @pytest.mark.asyncio
    async def test_initialize_context_patterns(self, agent_selector):
        """Test context pattern initialization."""
        await agent_selector._initialize_context_patterns()
        
        assert TaskContext.PRD_GENERATION in agent_selector.context_patterns
        assert TaskContext.TECHNICAL_ANALYSIS in agent_selector.context_patterns
        assert TaskContext.CODE_DEVELOPMENT in agent_selector.context_patterns
        assert TaskContext.CONTENT_CREATION in agent_selector.context_patterns
        
        # Check specific pattern structure
        prd_pattern = agent_selector.context_patterns[TaskContext.PRD_GENERATION]
        assert "preferred_capabilities" in prd_pattern
        assert CapabilityType.CREATION in prd_pattern["preferred_capabilities"]
        assert "quality_emphasis" in prd_pattern
    
    @pytest.mark.asyncio
    async def test_load_collaboration_history(self, agent_selector):
        """Test collaboration history loading."""
        await agent_selector._load_collaboration_history()
        
        assert len(agent_selector.collaboration_history) > 0
        
        # Check for high-synergy pairs
        ba_pa_key = (AgentType.BUSINESS_ANALYST, AgentType.PROJECT_ARCHITECT)
        if ba_pa_key in agent_selector.collaboration_history:
            assert agent_selector.collaboration_history[ba_pa_key] == 0.9
    
    @pytest.mark.asyncio
    async def test_get_candidate_agents(self, agent_selector, sample_requirements):
        """Test candidate agent retrieval."""
        candidates = await agent_selector._get_candidate_agents(sample_requirements)
        
        assert len(candidates) >= 1
        agent_types = [agent.agent_type for agent in candidates]
        assert AgentType.DRAFT_AGENT in agent_types  # Should match CREATION capability
        assert AgentType.JUDGE_AGENT in agent_types  # Should match VALIDATION capability
    
    @pytest.mark.asyncio
    async def test_get_candidate_agents_with_exclusions(self, agent_selector, sample_requirements):
        """Test candidate retrieval with exclusions."""
        # Add exclusion
        sample_requirements.excluded_agents = [AgentType.DRAFT_AGENT]
        
        candidates = await agent_selector._get_candidate_agents(sample_requirements)
        agent_types = [agent.agent_type for agent in candidates]
        
        assert AgentType.DRAFT_AGENT not in agent_types
        assert AgentType.JUDGE_AGENT in agent_types  # Should still be included
    
    @pytest.mark.asyncio
    async def test_score_candidates(self, agent_selector, sample_requirements):
        """Test candidate scoring."""
        # Get candidates first
        candidates = await agent_selector._get_candidate_agents(sample_requirements)
        
        # Score them
        scores = await agent_selector._score_candidates(candidates, sample_requirements)
        
        assert len(scores) == len(candidates)
        
        for score in scores:
            assert isinstance(score, AgentScore)
            assert 0.0 <= score.total_score <= 1.0
            assert 0.0 <= score.capability_score <= 1.0
            assert 0.0 <= score.performance_score <= 1.0
            assert 0.0 <= score.availability_score <= 1.0
            assert 0.0 <= score.resource_score <= 1.0
            assert 0.0 <= score.context_score <= 1.0
            assert 0.0 <= score.collaboration_score <= 1.0
            assert 0.0 <= score.confidence <= 1.0
        
        # Scores should be sorted by total_score (descending)
        for i in range(len(scores) - 1):
            assert scores[i].total_score >= scores[i + 1].total_score
    
    @pytest.mark.asyncio
    async def test_capability_score_calculation(self, agent_selector, sample_requirements):
        """Test capability score calculation."""
        draft_agent = agent_selector.registry.get_agent(AgentType.DRAFT_AGENT)
        
        score = await agent_selector._calculate_capability_score(draft_agent, sample_requirements)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should have good match for CREATION capability
    
    @pytest.mark.asyncio
    async def test_performance_score_calculation(self, agent_selector, sample_requirements):
        """Test performance score calculation."""
        judge_agent = agent_selector.registry.get_agent(AgentType.JUDGE_AGENT)
        
        score = await agent_selector._calculate_performance_score(judge_agent, sample_requirements)
        
        assert 0.0 <= score <= 1.0
        # Judge agent has high success rate and good metrics
        assert score > 0.7
    
    @pytest.mark.asyncio
    async def test_availability_score_calculation(self, agent_selector):
        """Test availability score calculation."""
        draft_agent = agent_selector.registry.get_agent(AgentType.DRAFT_AGENT)
        
        score = await agent_selector._calculate_availability_score(draft_agent)
        
        assert 0.0 <= score <= 1.0
        # Should be high availability initially
        assert score > 0.8
    
    @pytest.mark.asyncio
    async def test_context_score_calculation(self, agent_selector, sample_requirements):
        """Test context specialization score calculation."""
        draft_agent = agent_selector.registry.get_agent(AgentType.DRAFT_AGENT)
        
        # Initialize context patterns first
        await agent_selector._initialize_context_patterns()
        
        score = await agent_selector._calculate_context_score(draft_agent, sample_requirements)
        
        assert 0.0 <= score <= 1.0
        # Draft agent should be good for content creation context
        assert score > 0.5
    
    @pytest.mark.asyncio
    async def test_collaboration_score_calculation(self, agent_selector, sample_requirements):
        """Test collaboration score calculation."""
        draft_agent = agent_selector.registry.get_agent(AgentType.DRAFT_AGENT)
        
        score = await agent_selector._calculate_collaboration_score(draft_agent, sample_requirements)
        
        assert 0.0 <= score <= 1.0
        # Should have good collaboration score since DRAFT_AGENT is preferred
        assert score > 0.8
    
    @pytest.mark.asyncio
    async def test_select_optimal_combination(self, agent_selector, sample_requirements):
        """Test optimal agent combination selection."""
        # Create scored candidates
        scored_candidates = [
            AgentScore(
                agent_type=AgentType.DRAFT_AGENT,
                total_score=0.85,
                capability_score=0.9,
                performance_score=0.8,
                availability_score=1.0,
                resource_score=0.7,
                context_score=0.9,
                collaboration_score=0.8,
                confidence=0.85
            ),
            AgentScore(
                agent_type=AgentType.JUDGE_AGENT,
                total_score=0.82,
                capability_score=0.95,
                performance_score=0.85,
                availability_score=1.0,
                resource_score=0.6,
                context_score=0.7,
                collaboration_score=0.7,
                confidence=0.82
            )
        ]
        
        selection = await agent_selector._select_optimal_combination(
            scored_candidates, sample_requirements, max_agents=2
        )
        
        assert len(selection.selected_agents) <= 2
        assert AgentType.DRAFT_AGENT in selection.selected_agents
        assert AgentType.JUDGE_AGENT in selection.selected_agents
        assert 0.0 <= selection.confidence_level <= 1.0
        assert len(selection.selection_reasoning) > 0
    
    @pytest.mark.asyncio
    async def test_generate_execution_plan(self, agent_selector, sample_requirements):
        """Test execution plan generation."""
        await agent_selector._initialize_context_patterns()
        
        selected_agents = [AgentType.DRAFT_AGENT, AgentType.JUDGE_AGENT]
        
        plan = await agent_selector._generate_execution_plan(selected_agents, sample_requirements)
        
        assert "execution_strategy" in plan
        assert "phases" in plan
        assert "coordination_points" in plan
        assert "quality_gates" in plan
        assert "resource_requirements" in plan
        
        assert plan["execution_strategy"] in ["sequential", "parallel", "pipeline"]
    
    @pytest.mark.asyncio
    async def test_calculate_resource_allocation(self, agent_selector):
        """Test resource allocation calculation."""
        selected_agents = [AgentType.DRAFT_AGENT, AgentType.JUDGE_AGENT]
        
        allocation = await agent_selector._calculate_resource_allocation(selected_agents)
        
        assert "cpu_cores" in allocation
        assert "memory_mb" in allocation
        assert "concurrent_limit" in allocation
        assert "estimated_cost" in allocation
        
        assert allocation["cpu_cores"] > 0
        assert allocation["memory_mb"] > 0
        assert allocation["concurrent_limit"] == len(selected_agents)
        assert allocation["estimated_cost"] > 0
    
    @pytest.mark.asyncio
    async def test_estimate_completion_time(self, agent_selector, sample_requirements):
        """Test completion time estimation."""
        selected_agents = [AgentType.DRAFT_AGENT, AgentType.JUDGE_AGENT]
        
        estimated_time = await agent_selector._estimate_completion_time(
            selected_agents, sample_requirements
        )
        
        assert isinstance(estimated_time, timedelta)
        assert estimated_time.total_seconds() > 0
    
    @pytest.mark.asyncio
    async def test_identify_fallback_options(self, agent_selector):
        """Test fallback option identification."""
        selected_agents = [AgentType.DRAFT_AGENT]
        all_candidates = [
            AgentScore(
                agent_type=AgentType.DRAFT_AGENT,
                total_score=0.85,
                capability_score=0.9,
                performance_score=0.8,
                availability_score=1.0,
                resource_score=0.7,
                context_score=0.9,
                collaboration_score=0.8,
                confidence=0.85
            ),
            AgentScore(
                agent_type=AgentType.JUDGE_AGENT,
                total_score=0.75,
                capability_score=0.8,
                performance_score=0.7,
                availability_score=1.0,
                resource_score=0.6,
                context_score=0.8,
                collaboration_score=0.7,
                confidence=0.75
            ),
            AgentScore(
                agent_type=AgentType.BUSINESS_ANALYST,
                total_score=0.65,
                capability_score=0.7,
                performance_score=0.6,
                availability_score=1.0,
                resource_score=0.5,
                context_score=0.7,
                collaboration_score=0.6,
                confidence=0.65
            )
        ]
        
        fallbacks = await agent_selector._identify_fallback_options(
            selected_agents, all_candidates
        )
        
        assert AgentType.DRAFT_AGENT not in fallbacks  # Already selected
        assert AgentType.JUDGE_AGENT in fallbacks  # Good score, not selected
        assert len(fallbacks) <= 3  # Max 3 fallbacks
    
    @pytest.mark.asyncio
    async def test_full_agent_selection_workflow(self, agent_selector, sample_requirements):
        """Test complete agent selection workflow."""
        result = await agent_selector.select_agents(
            requirements=sample_requirements,
            max_agents=2,
            include_fallbacks=True
        )
        
        assert isinstance(result, SelectionResult)
        assert len(result.selected_agents) <= 2
        assert len(result.selected_agents) > 0
        assert len(result.scores) == len(result.selected_agents)
        assert 0.0 <= result.confidence_level <= 1.0
        assert isinstance(result.estimated_completion_time, timedelta)
        assert "execution_strategy" in result.execution_plan
        
        # Should include some fallback options
        assert len(result.fallback_options) >= 0
        assert len(result.selection_reasoning) > 0
    
    @pytest.mark.asyncio
    async def test_performance_metrics_update(self, agent_selector):
        """Test performance metrics updating."""
        await agent_selector.update_performance_metrics(
            agent_type=AgentType.DRAFT_AGENT,
            execution_time_ms=2000,
            success=True,
            quality_score=0.9
        )
        
        # Should call registry update
        agent_selector.registry.update_agent_metrics.assert_called_once_with(
            agent_type=AgentType.DRAFT_AGENT,
            execution_time_ms=2000,
            success=True,
            confidence_score=0.9
        )
    
    @pytest.mark.asyncio
    async def test_collaboration_history_update(self, agent_selector):
        """Test collaboration history updating."""
        agent_pairs = [(AgentType.DRAFT_AGENT, AgentType.JUDGE_AGENT)]
        success_rate = 0.95
        
        await agent_selector.update_collaboration_history(agent_pairs, success_rate)
        
        # Check that collaboration history was updated
        pair_key = (AgentType.DRAFT_AGENT, AgentType.JUDGE_AGENT)
        if pair_key in agent_selector.collaboration_history:
            assert agent_selector.collaboration_history[pair_key] > 0.5
    
    @pytest.mark.asyncio
    async def test_no_candidates_error(self, agent_selector):
        """Test error handling when no candidates are found."""
        # Create requirements that no agents can fulfill
        impossible_requirements = TaskRequirements(
            task_context=TaskContext.CONTENT_CREATION,
            required_capabilities=[],  # No required capabilities
            complexity_level=ComplexityLevel.SIMPLE,
            estimated_tokens=1000,
            max_execution_time=timedelta(minutes=5),
            excluded_agents=[AgentType.DRAFT_AGENT, AgentType.JUDGE_AGENT, AgentType.BUSINESS_ANALYST]
        )
        
        # Mock empty candidate list
        agent_selector.registry.get_agents_by_capability.return_value = []
        
        with pytest.raises(ValueError, match="No suitable candidate agents found"):
            await agent_selector.select_agents(impossible_requirements)
    
    @pytest.mark.asyncio
    async def test_uninitialized_selector_error(self):
        """Test error when using uninitialized selector."""
        selector = ContextAwareAgentSelector()
        requirements = TaskRequirements(
            task_context=TaskContext.CONTENT_CREATION,
            required_capabilities=[CapabilityType.CREATION],
            complexity_level=ComplexityLevel.SIMPLE,
            estimated_tokens=1000,
            max_execution_time=timedelta(minutes=5)
        )
        
        with pytest.raises(RuntimeError, match="Agent selector not initialized"):
            await selector.select_agents(requirements)


class TestTaskRequirements:
    """Test cases for TaskRequirements model."""
    
    def test_basic_requirements_creation(self):
        """Test basic requirements creation."""
        requirements = TaskRequirements(
            task_context=TaskContext.PRD_GENERATION,
            required_capabilities=[CapabilityType.CREATION, CapabilityType.STRATEGIC],
            complexity_level=ComplexityLevel.COMPLEX,
            estimated_tokens=10000,
            max_execution_time=timedelta(minutes=15)
        )
        
        assert requirements.task_context == TaskContext.PRD_GENERATION
        assert CapabilityType.CREATION in requirements.required_capabilities
        assert requirements.complexity_level == ComplexityLevel.COMPLEX
        assert requirements.estimated_tokens == 10000
        assert requirements.max_execution_time == timedelta(minutes=15)
        assert requirements.quality_threshold == 0.85  # Default
    
    def test_requirements_with_constraints(self):
        """Test requirements with additional constraints."""
        requirements = TaskRequirements(
            task_context=TaskContext.TECHNICAL_ANALYSIS,
            required_capabilities=[CapabilityType.ANALYTICAL, CapabilityType.TECHNICAL],
            complexity_level=ComplexityLevel.EXPERT,
            estimated_tokens=25000,
            max_execution_time=timedelta(hours=1),
            quality_threshold=0.95,
            domain_knowledge=["api", "architecture", "scalability"],
            preferred_agents=[AgentType.BACKEND_ARCHITECT, AgentType.PROJECT_ARCHITECT],
            excluded_agents=[AgentType.DRAFT_AGENT],
            collaboration_requirements={"review_required": True, "parallel_execution": False}
        )
        
        assert requirements.quality_threshold == 0.95
        assert "api" in requirements.domain_knowledge
        assert AgentType.BACKEND_ARCHITECT in requirements.preferred_agents
        assert AgentType.DRAFT_AGENT in requirements.excluded_agents
        assert requirements.collaboration_requirements["review_required"] is True


@pytest.mark.asyncio
async def test_get_context_aware_selector():
    """Test global selector instance getter."""
    with patch('services.context_aware_agent_selector.context_aware_selector.initialize') as mock_init:
        mock_init.return_value = None
        
        selector = await get_context_aware_selector()
        
        assert isinstance(selector, ContextAwareAgentSelector)
        mock_init.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])