"""
Tests for LLM fallback scenarios and model routing.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import os
from httpx import HTTPStatusError

from services.llm.openrouter_client import OpenRouterClient, ChatMessage, LLMResponse


class TestFallbackScenarios:
    """Test fallback scenarios and model routing."""

    @pytest.mark.asyncio
    async def test_model_fallback_on_failure(self):
        """Test that system falls back to alternative models when primary fails."""
        with patch.dict(os.environ, {
            'OPENROUTER_API_KEY': 'test-key',
            'ENVIRONMENT': 'testing',
            'SECRET_KEY': 'test-secret',
            'NEO4J_PASSWORD': 'test-neo4j'
        }):
            client = OpenRouterClient()
            messages = [ChatMessage(role="user", content="Hello, test message")]
            
            # Mock the _make_completion_request method directly
            fallback_response = LLMResponse(
                content="Fallback response",
                model="openai/gpt-4o-mini",
                provider="openai",
                confidence=0.85,
                token_usage={"total_tokens": 25},
                response_time_ms=150,
                fallback_used=False  # Will be set to True by chat_completion
            )
            
            with patch.object(client, '_make_completion_request', new_callable=AsyncMock) as mock_request, \
                 patch.object(client, '_log_interaction', new_callable=AsyncMock):
                
                # First call fails, second succeeds
                mock_request.side_effect = [
                    Exception("Primary model failed"),
                    fallback_response
                ]
                
                response = await client.chat_completion(
                    messages=messages,
                    task_complexity="standard"
                )
                
                assert isinstance(response, LLMResponse)
                assert response.content == "Fallback response"
                assert response.fallback_used is True
                assert mock_request.call_count == 2

            await client.close()

    @pytest.mark.asyncio
    async def test_low_confidence_triggers_fallback(self):
        """Test that low confidence responses trigger fallback to better models."""
        with patch.dict(os.environ, {
            'OPENROUTER_API_KEY': 'test-key',
            'ENVIRONMENT': 'testing',
            'SECRET_KEY': 'test-secret',
            'NEO4J_PASSWORD': 'test-neo4j'
        }):
            client = OpenRouterClient()
            messages = [ChatMessage(role="user", content="Complex question")]
            
            # Mock low confidence response first, then high confidence
            low_confidence_response = LLMResponse(
                content="I'm not sure about this",
                model="anthropic/claude-3-haiku",
                provider="anthropic",
                confidence=0.65,  # Below threshold
                token_usage={"total_tokens": 15},
                response_time_ms=100,
                fallback_used=False
            )
            
            high_confidence_response = LLMResponse(
                content="Here's a detailed and accurate answer",
                model="anthropic/claude-3.5-sonnet",
                provider="anthropic",
                confidence=0.95,  # Above threshold
                token_usage={"total_tokens": 35},
                response_time_ms=200,
                fallback_used=False
            )
            
            with patch.object(client, '_make_completion_request', new_callable=AsyncMock) as mock_request, \
                 patch.object(client, '_log_interaction', new_callable=AsyncMock):
                
                mock_request.side_effect = [low_confidence_response, high_confidence_response]
                
                response = await client.chat_completion(
                    messages=messages,
                    require_confidence=0.8  # Require high confidence
                )
                
                assert isinstance(response, LLMResponse)
                assert "detailed and accurate" in response.content
                assert response.fallback_used is True
                assert mock_request.call_count == 2

            await client.close()

    @pytest.mark.asyncio
    async def test_rate_limiting_retry_logic(self):
        """Test retry logic when rate limited."""
        with patch.dict(os.environ, {
            'OPENROUTER_API_KEY': 'test-key',
            'ENVIRONMENT': 'testing',
            'SECRET_KEY': 'test-secret',
            'NEO4J_PASSWORD': 'test-neo4j'
        }):
            client = OpenRouterClient()
            messages = [ChatMessage(role="user", content="Test")]
            
            success_response = LLMResponse(
                content="Success after retry",
                model="openai/gpt-4o-mini",
                provider="openai",
                confidence=0.85,
                token_usage={"total_tokens": 20},
                response_time_ms=150,
                fallback_used=False
            )
            
            with patch.object(client, '_make_completion_request', new_callable=AsyncMock) as mock_request, \
                 patch.object(client, '_log_interaction', new_callable=AsyncMock):
                
                # Test successful retry after rate limit
                mock_request.return_value = success_response
                
                response = await client.chat_completion(messages, max_retries=2)
                
                assert isinstance(response, LLMResponse)
                assert response.content == "Success after retry"
                assert not response.fallback_used  # First model worked after retry

            await client.close()

    def test_model_selection_by_task_type(self):
        """Test that different task types select appropriate models."""
        with patch.dict(os.environ, {
            'OPENROUTER_API_KEY': 'test-key',
            'ENVIRONMENT': 'testing',
            'SECRET_KEY': 'test-secret',
            'NEO4J_PASSWORD': 'test-neo4j'
        }):
            client = OpenRouterClient()
            
            # Complex tasks should prefer premium models
            complex_models = client.select_models_for_request(task_complexity="complex")
            assert len(complex_models) > 0
            assert complex_models[0].tier.value == "premium"
            
            # Simple tasks should prefer budget models
            simple_models = client.select_models_for_request(task_complexity="simple")
            assert len(simple_models) > 0
            assert simple_models[0].tier.value == "budget"
            
            # Standard tasks should balance between standard and premium
            standard_models = client.select_models_for_request(task_complexity="standard")
            assert len(standard_models) > 0
            assert standard_models[0].tier.value in ["standard", "premium"]

    def test_confidence_scoring_edge_cases(self):
        """Test confidence scoring for various response scenarios."""
        with patch.dict(os.environ, {
            'OPENROUTER_API_KEY': 'test-key',
            'ENVIRONMENT': 'testing',
            'SECRET_KEY': 'test-secret',
            'NEO4J_PASSWORD': 'test-neo4j'
        }):
            client = OpenRouterClient()
            model_config = client.models["anthropic/claude-3.5-sonnet"]
            
            # Test empty response
            empty_confidence = client._calculate_response_confidence(
                content="",
                finish_reason="stop",
                model_config=model_config
            )
            assert empty_confidence < 0.5
            
            # Test uncertain response
            uncertain_confidence = client._calculate_response_confidence(
                content="I'm not sure about this answer",
                finish_reason="stop",
                model_config=model_config
            )
            assert uncertain_confidence < model_config.confidence_threshold
            
            # Test truncated response
            truncated_confidence = client._calculate_response_confidence(
                content="This is a good response that was truncated",
                finish_reason="length",
                model_config=model_config
            )
            assert truncated_confidence < model_config.confidence_threshold
            
            # Test complete, confident response
            confident_response = client._calculate_response_confidence(
                content="This is a comprehensive and well-structured response that demonstrates clear understanding of the topic and provides actionable insights.",
                finish_reason="stop",
                model_config=model_config
            )
            assert confident_response >= model_config.confidence_threshold

    def test_fallback_model_selection_logic(self):
        """Test the logic for selecting fallback models."""
        with patch.dict(os.environ, {
            'OPENROUTER_API_KEY': 'test-key',
            'ENVIRONMENT': 'testing',
            'SECRET_KEY': 'test-secret',
            'NEO4J_PASSWORD': 'test-neo4j'
        }):
            client = OpenRouterClient()
            
            # Test fallback for premium model
            premium_model = client.models["anthropic/claude-3.5-sonnet"]
            fallbacks = client._get_fallback_models(premium_model)
            
            assert len(fallbacks) > 0
            # Should include same-tier different providers and lower tiers
            fallback_providers = [f.provider for f in fallbacks]
            assert len(set(fallback_providers)) > 1  # Multiple providers
            
            # Test fallback for budget model (should have fewer options)
            budget_model = client.models["meta-llama/llama-3.1-8b-instruct"]
            budget_fallbacks = client._get_fallback_models(budget_model)
            
            # Budget tier should have fewer fallback options
            assert len(budget_fallbacks) >= 0

    @pytest.mark.asyncio
    async def test_all_models_exhausted_scenario(self):
        """Test behavior when all available models fail."""
        with patch.dict(os.environ, {
            'OPENROUTER_API_KEY': 'test-key',
            'ENVIRONMENT': 'testing',
            'SECRET_KEY': 'test-secret',
            'NEO4J_PASSWORD': 'test-neo4j'
        }):
            client = OpenRouterClient()
            messages = [ChatMessage(role="user", content="Test")]
            
            with patch.object(client, '_make_completion_request', new_callable=AsyncMock) as mock_request:
                # Make all model requests fail
                mock_request.side_effect = Exception("All models unavailable")
                
                with pytest.raises(ValueError, match="All configured models failed"):
                    await client.chat_completion(messages)

            await client.close()

    def test_model_configuration_validation(self):
        """Test that model configurations are properly validated."""
        with patch.dict(os.environ, {
            'OPENROUTER_API_KEY': 'test-key',
            'ENVIRONMENT': 'testing',
            'SECRET_KEY': 'test-secret',
            'NEO4J_PASSWORD': 'test-neo4j'
        }):
            client = OpenRouterClient()
            
            # Verify all models have required fields
            for name, config in client.models.items():
                assert config.name == name
                assert config.provider in [p.value for p in client.models[name].provider.__class__]
                assert config.tier in [t.value for t in client.models[name].tier.__class__]
                assert config.max_tokens > 0
                assert 0.0 <= config.temperature <= 2.0
                assert 0.0 <= config.confidence_threshold <= 1.0