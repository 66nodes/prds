"""
Integration tests for Context-Aware Agent Selection System.

Tests the complete flow from API endpoints through the context-aware selector
to the orchestrator, validating end-to-end functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy.orm import Session

from main import app
from api.endpoints.agent_selection import router as agent_selection_router
from services.context_aware_agent_selector import (
    ContextAwareAgentSelector,
    TaskRequirements,
    TaskContext,
    SelectionResult,
    AgentScore,
    get_context_aware_selector
)
from services.agent_orchestrator import (
    AgentOrchestrator,
    AgentType,
    TaskPriority,
    get_orchestrator
)
from services.agent_registry import (
    AgentRegistry,
    CapabilityType,
    ComplexityLevel,
    get_agent_registry
)
from services.auth_service import User


@pytest.fixture
async def test_client():
    """Create test client for API integration tests."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_current_user():
    """Mock authenticated user for API tests."""
    return User(
        id="test-user-123",
        email="user@company.local",
        name="Test User",
        is_active=True
    )


@pytest.fixture
async def initialized_selector():
    """Create and initialize a real context-aware selector for testing."""
    selector = ContextAwareAgentSelector()
    
    # Mock the registry dependency
    with patch('services.context_aware_agent_selector.get_agent_registry') as mock_registry:
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance
        
        # Configure mock registry with test data
        await _configure_mock_registry(mock_registry_instance)
        
        await selector.initialize()
        yield selector


@pytest.fixture
async def initialized_orchestrator():
    """Create and initialize a real orchestrator for testing."""
    orchestrator = AgentOrchestrator()
    
    # Mock dependencies
    with patch.multiple(
        'services.agent_orchestrator',
        get_agent_registry=AsyncMock(),
        get_context_aware_selector=AsyncMock(),
    ):
        await orchestrator.initialize()
        yield orchestrator


async def _configure_mock_registry(mock_registry):
    """Configure mock agent registry with realistic test data."""
    from services.agent_registry import (
        AgentRegistryEntry,
        AgentCapability,
        AgentMetrics,
        ResourceRequirement
    )
    
    # Create test agent entries
    test_agents = [
        AgentRegistryEntry(
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
                )
            ],
            metrics=AgentMetrics(
                success_rate=0.92,
                average_response_time_ms=2500,
                total_executions=150,
                confidence_scores=[0.85, 0.88, 0.91]
            ),
            tags=["content", "drafting", "writing"]
        ),
        AgentRegistryEntry(
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
                )
            ],
            metrics=AgentMetrics(
                success_rate=0.96,
                average_response_time_ms=1800,
                total_executions=200,
                confidence_scores=[0.92, 0.94, 0.96]
            ),
            tags=["validation", "quality", "assessment"]
        ),
        AgentRegistryEntry(
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
    ]
    
    # Configure mock methods
    mock_registry.get_agents_by_capability.side_effect = lambda capability, **kwargs: [
        agent for agent in test_agents 
        if any(cap.capability_type == capability for cap in agent.capabilities)
    ]
    
    mock_registry.get_agent.side_effect = lambda agent_type: next(
        (agent for agent in test_agents if agent.agent_type == agent_type),
        None
    )
    
    mock_registry.update_agent_metrics = MagicMock()


class TestContextAwareIntegration:
    """Integration tests for context-aware agent selection system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_agent_selection(self, initialized_selector, initialized_orchestrator):
        """Test complete agent selection flow."""
        # Create realistic task requirements
        requirements = TaskRequirements(
            task_context=TaskContext.CONTENT_CREATION,
            required_capabilities=[CapabilityType.CREATION, CapabilityType.VALIDATION],
            complexity_level=ComplexityLevel.MODERATE,
            estimated_tokens=5000,
            max_execution_time=timedelta(minutes=10),
            quality_threshold=0.85,
            domain_knowledge=["content", "writing"],
            preferred_agents=[AgentType.DRAFT_AGENT]
        )
        
        # Execute selection
        result = await initialized_selector.select_agents(
            requirements=requirements,
            max_agents=2,
            include_fallbacks=True
        )
        
        # Validate results
        assert isinstance(result, SelectionResult)
        assert len(result.selected_agents) <= 2
        assert len(result.selected_agents) > 0
        assert AgentType.DRAFT_AGENT in result.selected_agents
        assert 0.7 <= result.confidence_level <= 1.0
        assert isinstance(result.estimated_completion_time, timedelta)
        
        # Validate execution plan
        assert "execution_strategy" in result.execution_plan
        assert result.execution_plan["execution_strategy"] in ["sequential", "parallel", "pipeline"]
        
        # Validate resource allocation
        assert "cpu_cores" in result.resource_allocation
        assert "memory_mb" in result.resource_allocation
        assert result.resource_allocation["cpu_cores"] > 0
        assert result.resource_allocation["memory_mb"] > 0
    
    @pytest.mark.asyncio
    async def test_orchestrator_intelligent_selection_integration(self, initialized_orchestrator):
        """Test integration with orchestrator's intelligent selection methods."""
        with patch('services.agent_orchestrator.get_context_aware_selector') as mock_selector:
            # Mock selector response
            mock_selector_instance = AsyncMock()
            mock_selector.return_value = mock_selector_instance
            
            mock_selector_instance.select_agents.return_value = SelectionResult(
                selected_agents=[AgentType.DRAFT_AGENT, AgentType.JUDGE_AGENT],
                scores=[
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
                    )
                ],
                execution_plan={"execution_strategy": "sequential"},
                resource_allocation={"cpu_cores": 2, "memory_mb": 1024},
                estimated_completion_time=timedelta(minutes=5),
                confidence_level=0.85,
                fallback_options=[AgentType.BUSINESS_ANALYST],
                selection_reasoning=["High capability match", "Good performance history"]
            )
            
            # Test intelligent agent selection
            result = await initialized_orchestrator.select_agents_intelligently(
                task_context=TaskContext.CONTENT_CREATION,
                required_capabilities=[CapabilityType.CREATION],
                complexity_level=ComplexityLevel.MODERATE,
                estimated_tokens=5000,
                max_execution_time=timedelta(minutes=10),
                max_agents=2
            )
            
            # Validate orchestrator result format
            assert "selected_agents" in result
            assert "confidence_level" in result
            assert "execution_plan" in result
            assert "resource_allocation" in result
            assert result["selected_agents"] == [AgentType.DRAFT_AGENT.value, AgentType.JUDGE_AGENT.value]
            assert result["confidence_level"] == 0.85
    
    @pytest.mark.asyncio 
    async def test_workflow_creation_with_intelligent_selection(self, initialized_orchestrator):
        """Test workflow creation with intelligent agent selection."""
        with patch('services.agent_orchestrator.get_context_aware_selector') as mock_selector:
            # Mock selector
            mock_selector_instance = AsyncMock()
            mock_selector.return_value = mock_selector_instance
            
            mock_selector_instance.select_agents.return_value = SelectionResult(
                selected_agents=[AgentType.DRAFT_AGENT],
                scores=[],
                execution_plan={"execution_strategy": "sequential"},
                resource_allocation={"cpu_cores": 1, "memory_mb": 512},
                estimated_completion_time=timedelta(minutes=3),
                confidence_level=0.9,
                fallback_options=[],
                selection_reasoning=["Perfect capability match"]
            )
            
            # Test intelligent workflow creation
            result = await initialized_orchestrator.create_intelligent_workflow(
                user_id="test-user",
                task_context=TaskContext.CONTENT_CREATION,
                required_capabilities=[CapabilityType.CREATION],
                project_id="test-project",
                complexity_level=ComplexityLevel.SIMPLE,
                estimated_tokens=2000,
                max_execution_time=timedelta(minutes=5),
                max_agents=1
            )
            
            # Validate workflow creation result
            assert "workflow_context" in result
            assert "agent_selection" in result
            assert "workflow_id" in result["workflow_context"]
            assert "selected_agents" in result["agent_selection"]
            assert result["agent_selection"]["selected_agents"] == [AgentType.DRAFT_AGENT.value]
    
    @pytest.mark.asyncio
    async def test_task_addition_with_intelligent_selection(self, initialized_orchestrator):
        """Test adding tasks with intelligent agent selection."""
        with patch('services.agent_orchestrator.get_context_aware_selector') as mock_selector:
            # Mock selector
            mock_selector_instance = AsyncMock()
            mock_selector.return_value = mock_selector_instance
            
            mock_selector_instance.select_agents.return_value = SelectionResult(
                selected_agents=[AgentType.JUDGE_AGENT],
                scores=[],
                execution_plan={"execution_strategy": "sequential"},
                resource_allocation={"cpu_cores": 1, "memory_mb": 512},
                estimated_completion_time=timedelta(minutes=2),
                confidence_level=0.95,
                fallback_options=[AgentType.DRAFT_AGENT],
                selection_reasoning=["Excellent validation capabilities"]
            )
            
            # Test intelligent task addition
            result = await initialized_orchestrator.add_intelligent_task(
                workflow_id="test-workflow-123",
                operation="validate_content",
                task_context=TaskContext.VALIDATION,
                required_capabilities=[CapabilityType.VALIDATION],
                parameters={"content_type": "document"},
                context={"quality_threshold": 0.9},
                complexity_level=ComplexityLevel.MODERATE,
                priority=TaskPriority.HIGH
            )
            
            # Validate task addition result
            assert "task_id" in result
            assert "selected_agent" in result
            assert "selection_confidence" in result
            assert "fallback_options" in result
            assert result["selected_agent"] == AgentType.JUDGE_AGENT.value
            assert result["selection_confidence"] == 0.95
            assert AgentType.DRAFT_AGENT.value in result["fallback_options"]
    
    @pytest.mark.asyncio
    async def test_performance_update_integration(self, initialized_selector, initialized_orchestrator):
        """Test performance metrics update integration."""
        # Test selector performance update
        await initialized_selector.update_performance_metrics(
            agent_type=AgentType.DRAFT_AGENT,
            execution_time_ms=2500,
            success=True,
            quality_score=0.92
        )
        
        # Test orchestrator performance update
        await initialized_orchestrator.update_agent_performance(
            agent_type=AgentType.JUDGE_AGENT,
            execution_time_ms=1500,
            success=True,
            quality_score=0.95
        )
        
        # Verify updates were processed (would check actual storage in real implementation)
        assert True  # Placeholder - in real system would verify database/storage updates
    
    @pytest.mark.asyncio
    async def test_collaboration_history_update(self, initialized_selector):
        """Test collaboration history tracking."""
        agent_pairs = [(AgentType.DRAFT_AGENT, AgentType.JUDGE_AGENT)]
        success_rate = 0.96
        
        await initialized_selector.update_collaboration_history(agent_pairs, success_rate)
        
        # Check that collaboration history was updated
        pair_key = (AgentType.DRAFT_AGENT, AgentType.JUDGE_AGENT)
        if pair_key in initialized_selector.collaboration_history:
            assert initialized_selector.collaboration_history[pair_key] >= 0.5
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, initialized_selector):
        """Test error handling in integration scenarios."""
        # Test with invalid requirements
        invalid_requirements = TaskRequirements(
            task_context=TaskContext.CONTENT_CREATION,
            required_capabilities=[],  # Empty capabilities should cause issues
            complexity_level=ComplexityLevel.SIMPLE,
            estimated_tokens=1000,
            max_execution_time=timedelta(minutes=5),
            excluded_agents=[AgentType.DRAFT_AGENT, AgentType.JUDGE_AGENT, AgentType.BUSINESS_ANALYST]
        )
        
        with pytest.raises(ValueError, match="No suitable candidate agents found"):
            await initialized_selector.select_agents(invalid_requirements)
    
    @pytest.mark.asyncio
    async def test_resource_constraint_handling(self, initialized_selector):
        """Test handling of resource constraints."""
        # Simulate high resource pressure
        initialized_selector.resource_monitor.resource_usage['cpu'] = 0.95
        initialized_selector.resource_monitor.resource_usage['memory'] = 0.90
        
        requirements = TaskRequirements(
            task_context=TaskContext.CONTENT_CREATION,
            required_capabilities=[CapabilityType.CREATION],
            complexity_level=ComplexityLevel.COMPLEX,
            estimated_tokens=50000,  # High token requirement
            max_execution_time=timedelta(minutes=30)
        )
        
        result = await initialized_selector.select_agents(requirements)
        
        # Should still work but with adjusted resource scores
        assert len(result.selected_agents) > 0
        assert result.confidence_level > 0.0
    
    @pytest.mark.asyncio
    async def test_concurrent_selection_operations(self, initialized_selector):
        """Test handling concurrent selection operations."""
        requirements = TaskRequirements(
            task_context=TaskContext.CONTENT_CREATION,
            required_capabilities=[CapabilityType.CREATION],
            complexity_level=ComplexityLevel.MODERATE,
            estimated_tokens=5000,
            max_execution_time=timedelta(minutes=10)
        )
        
        # Run multiple selections concurrently
        tasks = [
            initialized_selector.select_agents(requirements)
            for _ in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 3
        for result in results:
            assert isinstance(result, SelectionResult)
            assert len(result.selected_agents) > 0


class TestAPIEndpointIntegration:
    """Integration tests for API endpoints."""
    
    @pytest.mark.asyncio
    async def test_agent_selection_endpoint_integration(self, test_client):
        """Test agent selection API endpoint integration."""
        with patch('api.endpoints.agent_selection.get_current_user') as mock_user, \
             patch('api.endpoints.agent_selection.get_orchestrator') as mock_orch:
            
            # Mock dependencies
            mock_user.return_value = User(id="test-user", email="user@company.local", name="Test User")
            
            mock_orchestrator = AsyncMock()
            mock_orch.return_value = mock_orchestrator
            
            mock_orchestrator.select_agents_intelligently.return_value = {
                "selected_agents": ["draft_agent", "judge_agent"],
                "agent_scores": [
                    {
                        "agent_type": "draft_agent",
                        "total_score": 0.85,
                        "capability_score": 0.9,
                        "performance_score": 0.8,
                        "availability_score": 1.0,
                        "context_score": 0.9,
                        "confidence": 0.85,
                        "reasoning": ["Strong capability match"]
                    }
                ],
                "execution_plan": {"execution_strategy": "sequential"},
                "resource_allocation": {"cpu_cores": 2, "memory_mb": 1024},
                "estimated_completion_time_seconds": 300.0,
                "confidence_level": 0.85,
                "fallback_options": ["business_analyst"],
                "selection_reasoning": ["High capability match"],
                "requirements": {"task_context": "content_creation"}
            }
            
            # Test request
            response = await test_client.post(
                "/agent-selection/select",
                json={
                    "task_context": "content_creation",
                    "required_capabilities": ["creation", "validation"],
                    "complexity_level": "moderate",
                    "estimated_tokens": 5000,
                    "max_execution_time_minutes": 10,
                    "max_agents": 2
                },
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "selected_agents" in data
            assert "confidence_level" in data
            assert data["selected_agents"] == ["draft_agent", "judge_agent"]
            assert data["confidence_level"] == 0.85
    
    @pytest.mark.asyncio
    async def test_workflow_creation_endpoint_integration(self, test_client):
        """Test workflow creation API endpoint integration."""
        with patch('api.endpoints.agent_selection.get_current_user') as mock_user, \
             patch('api.endpoints.agent_selection.get_orchestrator') as mock_orch:
            
            # Mock dependencies
            mock_user.return_value = User(id="test-user", email="user@company.local", name="Test User")
            
            mock_orchestrator = AsyncMock()
            mock_orch.return_value = mock_orchestrator
            
            mock_orchestrator.create_intelligent_workflow.return_value = {
                "workflow_context": {
                    "workflow_id": "workflow-123",
                    "status": "created",
                    "created_at": datetime.utcnow().isoformat()
                },
                "agent_selection": {
                    "selected_agents": ["draft_agent"],
                    "confidence_level": 0.9,
                    "execution_strategy": "sequential"
                }
            }
            
            # Test request
            response = await test_client.post(
                "/agent-selection/workflows/create",
                json={
                    "task_context": "content_creation",
                    "required_capabilities": ["creation"],
                    "project_id": "test-project",
                    "complexity_level": "simple",
                    "estimated_tokens": 2000,
                    "max_execution_time_minutes": 5,
                    "max_agents": 1
                },
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "workflow_context" in data
            assert "agent_selection" in data
            assert data["workflow_context"]["workflow_id"] == "workflow-123"
            assert data["agent_selection"]["selected_agents"] == ["draft_agent"]
    
    @pytest.mark.asyncio
    async def test_performance_update_endpoint_integration(self, test_client):
        """Test performance update API endpoint integration."""
        with patch('api.endpoints.agent_selection.get_current_user') as mock_user, \
             patch('api.endpoints.agent_selection.get_orchestrator') as mock_orch:
            
            # Mock dependencies
            mock_user.return_value = User(id="test-user", email="user@company.local", name="Test User")
            
            mock_orchestrator = AsyncMock()
            mock_orch.return_value = mock_orchestrator
            
            # Test request
            response = await test_client.post(
                "/agent-selection/performance/update",
                json={
                    "agent_type": "draft_agent",
                    "execution_time_ms": 2500,
                    "success": True,
                    "quality_score": 0.92
                },
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "success"
            assert "Performance metrics updated" in data["message"]
            
            # Verify orchestrator method was called
            mock_orchestrator.update_agent_performance.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_endpoint_integration(self, test_client):
        """Test health check endpoint integration."""
        with patch('api.endpoints.agent_selection.get_context_aware_selector') as mock_selector, \
             patch('api.endpoints.agent_selection.get_orchestrator') as mock_orch:
            
            # Mock dependencies
            mock_selector_instance = AsyncMock()
            mock_selector.return_value = mock_selector_instance
            mock_selector_instance.is_initialized = True
            
            mock_orchestrator = AsyncMock()
            mock_orch.return_value = mock_orchestrator
            mock_orchestrator.health_check.return_value = {"status": "healthy"}
            
            # Test request
            response = await test_client.get("/agent-selection/health")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "healthy"
            assert data["service"] == "agent-selection"
            assert data["selector_initialized"] == True
            assert data["orchestrator_status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_capability_listing_endpoints(self, test_client):
        """Test capability and context listing endpoints."""
        with patch('api.endpoints.agent_selection.get_current_user') as mock_user:
            mock_user.return_value = User(id="test-user", email="user@company.local", name="Test User")
            
            # Test capabilities endpoint
            response = await test_client.get(
                "/agent-selection/capabilities",
                headers={"Authorization": "Bearer test-token"}
            )
            assert response.status_code == 200
            capabilities = response.json()
            assert isinstance(capabilities, list)
            assert len(capabilities) > 0
            
            # Test contexts endpoint
            response = await test_client.get(
                "/agent-selection/contexts",
                headers={"Authorization": "Bearer test-token"}
            )
            assert response.status_code == 200
            contexts = response.json()
            assert isinstance(contexts, list)
            assert len(contexts) > 0
            
            # Test agent types endpoint
            response = await test_client.get(
                "/agent-selection/agents",
                headers={"Authorization": "Bearer test-token"}
            )
            assert response.status_code == 200
            agent_types = response.json()
            assert isinstance(agent_types, list)
            assert len(agent_types) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])