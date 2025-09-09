"""
Simplified unit tests for OpenRouter LLM components.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import os

from services.llm.openrouter_client import (
    OpenRouterClient,
    ChatMessage,
    ModelConfig,
    ModelTier,
    ModelProvider
)
from services.llm.llm_service import LLMService, LLMRequest


class TestOpenRouterClientUnit:
    """Unit tests for OpenRouter client."""

    def test_client_initialization(self):
        """Test client initialization."""
        with patch.dict(os.environ, {
            'OPENROUTER_API_KEY': 'test-key', 
            'ENVIRONMENT': 'testing',
            'SECRET_KEY': 'test-secret',
            'NEO4J_PASSWORD': 'test-neo4j'
        }):
            client = OpenRouterClient()
            
            assert client is not None
            assert len(client.models) > 0
            assert "anthropic/claude-3.5-sonnet" in client.models
            assert "openai/gpt-4o" in client.models

    def test_model_selection_complex_task(self):
        """Test model selection for complex tasks."""
        with patch.dict(os.environ, {
            'OPENROUTER_API_KEY': 'test-key',
            'ENVIRONMENT': 'testing',
            'SECRET_KEY': 'test-secret',
            'NEO4J_PASSWORD': 'test-neo4j'
        }):
            client = OpenRouterClient()
            models = client.select_models_for_request(task_complexity="complex")
            
            assert len(models) > 0
            # Premium models should be first for complex tasks
            assert models[0].tier == ModelTier.PREMIUM

    def test_model_selection_simple_task(self):
        """Test model selection for simple tasks."""
        with patch.dict(os.environ, {
            'OPENROUTER_API_KEY': 'test-key',
            'ENVIRONMENT': 'testing',
            'SECRET_KEY': 'test-secret',
            'NEO4J_PASSWORD': 'test-neo4j'
        }):
            client = OpenRouterClient()
            models = client.select_models_for_request(task_complexity="simple")
            
            assert len(models) > 0
            # Budget models should be prioritized for simple tasks
            assert models[0].tier == ModelTier.BUDGET

    def test_confidence_calculation(self):
        """Test response confidence calculation."""
        with patch.dict(os.environ, {
            'OPENROUTER_API_KEY': 'test-key',
            'ENVIRONMENT': 'testing',
            'SECRET_KEY': 'test-secret',
            'NEO4J_PASSWORD': 'test-neo4j'
        }):
            client = OpenRouterClient()
            model_config = client.models["anthropic/claude-3.5-sonnet"]
            
            # Test high confidence response
            high_confidence = client._calculate_response_confidence(
                content="This is a detailed and comprehensive response.",
                finish_reason="stop",
                model_config=model_config
            )
            assert high_confidence >= model_config.confidence_threshold
            
            # Test low confidence response
            low_confidence = client._calculate_response_confidence(
                content="I don't know",
                finish_reason="stop", 
                model_config=model_config
            )
            assert low_confidence < model_config.confidence_threshold


class TestLLMServiceUnit:
    """Unit tests for LLM service."""

    def test_service_initialization(self):
        """Test LLM service initialization."""
        with patch.dict(os.environ, {
            'OPENROUTER_API_KEY': 'test-key',
            'ENVIRONMENT': 'testing',
            'SECRET_KEY': 'test-secret',
            'NEO4J_PASSWORD': 'test-neo4j'
        }):
            service = LLMService()
            assert service is not None
            assert service._openrouter_client is None  # Lazy loading

    def test_task_complexity_determination(self):
        """Test task complexity determination."""
        with patch.dict(os.environ, {
            'OPENROUTER_API_KEY': 'test-key',
            'ENVIRONMENT': 'testing',
            'SECRET_KEY': 'test-secret',
            'NEO4J_PASSWORD': 'test-neo4j'
        }):
            service = LLMService()
            
            # Explicit complexity
            request = LLMRequest(
                messages=[ChatMessage(role="user", content="Test")],
                complexity="complex"
            )
            assert service._determine_task_complexity(request) == "complex"
            
            # Task type based complexity
            prd_request = LLMRequest(
                messages=[ChatMessage(role="user", content="Test")],
                task_type="prd"
            )
            assert service._determine_task_complexity(prd_request) == "complex"

    def test_model_selection_for_task(self):
        """Test model selection based on task type."""
        with patch.dict(os.environ, {
            'OPENROUTER_API_KEY': 'test-key',
            'ENVIRONMENT': 'testing',
            'SECRET_KEY': 'test-secret',
            'NEO4J_PASSWORD': 'test-neo4j'
        }):
            service = LLMService()
            
            # PRD task should select Claude
            prd_request = LLMRequest(
                messages=[ChatMessage(role="user", content="Test")],
                task_type="prd",
                complexity="complex"
            )
            selected = service._select_model_for_task(prd_request)
            assert "claude" in selected.lower()
            
            # Validation task should select GPT-4
            validation_request = LLMRequest(
                messages=[ChatMessage(role="user", content="Test")],
                task_type="validation"
            )
            selected = service._select_model_for_task(validation_request)
            assert "gpt-4" in selected.lower()


class TestModelConfiguration:
    """Test model configuration and fallback logic."""

    def test_model_config_creation(self):
        """Test model configuration creation."""
        config = ModelConfig(
            name="test/model",
            provider=ModelProvider.OPENAI,
            tier=ModelTier.STANDARD,
            max_tokens=2000,
            confidence_threshold=0.8
        )
        
        assert config.name == "test/model"
        assert config.provider == ModelProvider.OPENAI
        assert config.tier == ModelTier.STANDARD
        assert config.max_tokens == 2000
        assert config.confidence_threshold == 0.8

    def test_chat_message_creation(self):
        """Test chat message creation."""
        message = ChatMessage(role="user", content="Hello, world!")
        
        assert message.role == "user"
        assert message.content == "Hello, world!"

    def test_llm_request_creation(self):
        """Test LLM request creation."""
        messages = [ChatMessage(role="user", content="Test")]
        request = LLMRequest(
            messages=messages,
            task_type="prd",
            complexity="complex",
            require_high_confidence=True
        )
        
        assert len(request.messages) == 1
        assert request.task_type == "prd"
        assert request.complexity == "complex"
        assert request.require_high_confidence is True

    def test_llm_request_defaults(self):
        """Test LLM request default values."""
        messages = [ChatMessage(role="user", content="Test")]
        request = LLMRequest(messages=messages)
        
        assert request.model is None
        assert request.task_type == "general"
        assert request.complexity == "standard"
        assert request.temperature is None
        assert request.max_tokens is None
        assert request.require_high_confidence is False


@pytest.mark.asyncio
async def test_openrouter_health_check():
    """Test OpenRouter health check without real API call."""
    with patch.dict(os.environ, {
        'OPENROUTER_API_KEY': 'test-key',
        'ENVIRONMENT': 'testing',
        'SECRET_KEY': 'test-secret',
        'NEO4J_PASSWORD': 'test-neo4j'
    }):
        client = OpenRouterClient()
        
        with patch.object(client, 'chat_completion', new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = Mock(
                model="test-model",
                confidence=0.95
            )
            
            health = await client.health_check()
            assert health["status"] == "healthy"
            assert health["confidence"] == 0.95

        await client.close()


@pytest.mark.asyncio
async def test_llm_service_health_check():
    """Test LLM service health check without real API call."""
    with patch.dict(os.environ, {
        'OPENROUTER_API_KEY': 'test-key',
        'ENVIRONMENT': 'testing',
        'SECRET_KEY': 'test-secret',
        'NEO4J_PASSWORD': 'test-neo4j'
    }):
        service = LLMService()
        
        with patch.object(service, 'generate_completion', new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = Mock(
                content="Service operational",
                model="test-model",
                confidence=0.99,
                response_time_ms=80
            )
            
            health = await service.health_check()
            assert health["status"] == "healthy"
            assert health["service"] == "LLMService"
            assert health["test_response_confidence"] == 0.99

        await service.close()


def test_openrouter_api_key_requirement():
    """Test that OpenRouter API key is required."""
    # Without API key should not raise during initialization
    # but should fail during actual API calls
    with patch.dict(os.environ, {
        'ENVIRONMENT': 'testing',
        'SECRET_KEY': 'test-secret',
        'NEO4J_PASSWORD': 'test-neo4j'
    }, clear=True):
        client = OpenRouterClient()
        assert client.headers["Authorization"] == "Bearer None"


def test_model_enumeration():
    """Test model provider and tier enumerations."""
    # Test ModelProvider enum
    assert ModelProvider.OPENAI == "openai"
    assert ModelProvider.ANTHROPIC == "anthropic"
    assert ModelProvider.GOOGLE == "google"
    
    # Test ModelTier enum
    assert ModelTier.PREMIUM == "premium"
    assert ModelTier.STANDARD == "standard"
    assert ModelTier.BUDGET == "budget"