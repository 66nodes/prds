"""
Integration tests for OpenRouter LLM service.
"""

import pytest
import asyncio
from unittest.mock import patch, Mock, AsyncMock
from datetime import datetime

from services.llm.openrouter_client import (
    OpenRouterClient,
    ChatMessage,
    LLMResponse,
    ModelConfig,
    ModelTier,
    ModelProvider
)
from services.llm.llm_service import LLMService, LLMRequest


class TestOpenRouterIntegration:
    """Test suite for OpenRouter integration."""

    @pytest.fixture
    def mock_openrouter_response(self):
        """Mock successful OpenRouter API response."""
        return {
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "anthropic/claude-3.5-sonnet",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response from the AI model."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 15,
                "total_tokens": 35
            }
        }

    @pytest.fixture
    def openrouter_client(self):
        """Create OpenRouter client for testing."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key-12345', 'ENVIRONMENT': 'testing'}):
            return OpenRouterClient()

    @pytest.fixture
    def llm_service(self):
        """Create LLM service for testing."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key-12345', 'ENVIRONMENT': 'testing'}):
            return LLMService()

    @pytest.mark.asyncio
    async def test_openrouter_client_initialization(self, openrouter_client):
        """Test OpenRouter client initialization."""
        assert openrouter_client is not None
        assert len(openrouter_client.models) > 0
        assert "anthropic/claude-3.5-sonnet" in openrouter_client.models
        assert "openai/gpt-4o" in openrouter_client.models

    @pytest.mark.asyncio
    async def test_model_selection_for_complex_task(self, openrouter_client):
        """Test model selection for complex tasks."""
        models = openrouter_client.select_models_for_request(
            task_complexity="complex"
        )
        
        assert len(models) > 0
        # Premium models should be first for complex tasks
        assert models[0].tier == ModelTier.PREMIUM

    @pytest.mark.asyncio
    async def test_model_selection_for_simple_task(self, openrouter_client):
        """Test model selection for simple tasks."""
        models = openrouter_client.select_models_for_request(
            task_complexity="simple"
        )
        
        assert len(models) > 0
        # Budget models should be prioritized for simple tasks
        assert models[0].tier == ModelTier.BUDGET

    @pytest.mark.asyncio
    async def test_model_selection_with_specific_model(self, openrouter_client):
        """Test model selection with specific model request."""
        specific_model = "anthropic/claude-3.5-sonnet"
        models = openrouter_client.select_models_for_request(
            requested_model=specific_model
        )
        
        assert len(models) > 0
        assert models[0].name == specific_model

    @pytest.mark.asyncio
    async def test_chat_completion_success(self, openrouter_client, mock_openrouter_response):
        """Test successful chat completion."""
        messages = [
            ChatMessage(role="user", content="Hello, test message")
        ]
        
        with patch.object(openrouter_client.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = mock_openrouter_response
            mock_post.return_value = mock_response
            
            response = await openrouter_client.chat_completion(
                messages=messages,
                task_complexity="standard"
            )
            
            assert isinstance(response, LLMResponse)
            assert response.content == "This is a test response from the AI model."
            assert response.model == "anthropic/claude-3.5-sonnet"
            assert response.confidence > 0
            assert response.token_usage["total_tokens"] == 35
            assert not response.fallback_used

    @pytest.mark.asyncio
    async def test_chat_completion_with_fallback(self, openrouter_client, mock_openrouter_response):
        """Test chat completion with fallback when primary model fails."""
        messages = [
            ChatMessage(role="user", content="Hello, test message")
        ]
        
        with patch.object(openrouter_client.client, 'post', new_callable=AsyncMock) as mock_post:
            # First call fails, second succeeds
            mock_post.side_effect = [
                Exception("Primary model failed"),
                Mock(raise_for_status=Mock(), json=Mock(return_value=mock_openrouter_response))
            ]
            
            response = await openrouter_client.chat_completion(
                messages=messages,
                task_complexity="standard"
            )
            
            assert isinstance(response, LLMResponse)
            assert response.fallback_used is True
            assert mock_post.call_count == 2

    @pytest.mark.asyncio
    async def test_confidence_calculation(self, openrouter_client):
        """Test response confidence calculation."""
        model_config = openrouter_client.models["anthropic/claude-3.5-sonnet"]
        
        # Test high confidence response
        high_confidence = openrouter_client._calculate_response_confidence(
            content="This is a detailed and comprehensive response with substantial content.",
            finish_reason="stop",
            model_config=model_config
        )
        assert high_confidence >= model_config.confidence_threshold
        
        # Test low confidence response
        low_confidence = openrouter_client._calculate_response_confidence(
            content="I don't know",
            finish_reason="stop", 
            model_config=model_config
        )
        assert low_confidence < model_config.confidence_threshold
        
        # Test truncated response
        truncated_confidence = openrouter_client._calculate_response_confidence(
            content="This response was cut off due to length",
            finish_reason="length",
            model_config=model_config
        )
        assert truncated_confidence < model_config.confidence_threshold

    @pytest.mark.asyncio
    async def test_health_check_success(self, openrouter_client, mock_openrouter_response):
        """Test successful health check."""
        with patch.object(openrouter_client, 'chat_completion', new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = LLMResponse(
                content="OK",
                model="openai/gpt-4o-mini",
                provider="openai",
                confidence=0.95,
                token_usage={"total_tokens": 10},
                response_time_ms=150,
                fallback_used=False
            )
            
            health_status = await openrouter_client.health_check()
            
            assert health_status["status"] == "healthy"
            assert "response_time_ms" in health_status
            assert health_status["confidence"] == 0.95
            assert health_status["available_models"] > 0

    @pytest.mark.asyncio
    async def test_health_check_failure(self, openrouter_client):
        """Test health check failure."""
        with patch.object(openrouter_client, 'chat_completion', new_callable=AsyncMock) as mock_completion:
            mock_completion.side_effect = Exception("Service unavailable")
            
            health_status = await openrouter_client.health_check()
            
            assert health_status["status"] == "unhealthy"
            assert "error" in health_status


class TestLLMServiceIntegration:
    """Test suite for LLM service integration."""

    @pytest.fixture
    def llm_service_integration(self):
        """Create LLM service for testing."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key-12345', 'ENVIRONMENT': 'testing'}):
            return LLMService()

    @pytest.mark.asyncio
    async def test_llm_service_initialization(self, llm_service):
        """Test LLM service initialization."""
        assert llm_service is not None
        assert llm_service._openrouter_client is None  # Lazy loading

    @pytest.mark.asyncio
    async def test_task_complexity_determination(self, llm_service):
        """Test task complexity determination."""
        # Explicit complexity
        request = LLMRequest(
            messages=[ChatMessage(role="user", content="Test")],
            complexity="complex"
        )
        assert llm_service._determine_task_complexity(request) == "complex"
        
        # Task type based complexity
        prd_request = LLMRequest(
            messages=[ChatMessage(role="user", content="Test")],
            task_type="prd"
        )
        assert llm_service._determine_task_complexity(prd_request) == "complex"
        
        qa_request = LLMRequest(
            messages=[ChatMessage(role="user", content="Test")],
            task_type="qa"
        )
        assert llm_service._determine_task_complexity(qa_request) == "simple"

    @pytest.mark.asyncio
    async def test_model_selection_for_task(self, llm_service):
        """Test model selection based on task type."""
        # PRD task should select Claude
        prd_request = LLMRequest(
            messages=[ChatMessage(role="user", content="Test")],
            task_type="prd",
            complexity="complex"
        )
        selected = llm_service._select_model_for_task(prd_request)
        assert "claude" in selected.lower()
        
        # Validation task should select GPT-4
        validation_request = LLMRequest(
            messages=[ChatMessage(role="user", content="Test")],
            task_type="validation"
        )
        selected = llm_service._select_model_for_task(validation_request)
        assert "gpt-4" in selected.lower()

    @pytest.mark.asyncio
    async def test_generate_completion_success(self, llm_service):
        """Test successful completion generation."""
        messages = [ChatMessage(role="user", content="Generate a test response")]
        request = LLMRequest(messages=messages, task_type="general")
        
        with patch.object(llm_service, '_get_openrouter_client', new_callable=AsyncMock) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.chat_completion.return_value = LLMResponse(
                content="Test response generated successfully",
                model="openai/gpt-4o",
                provider="openai",
                confidence=0.92,
                token_usage={"total_tokens": 25},
                response_time_ms=180,
                fallback_used=False
            )
            mock_get_client.return_value = mock_client
            
            response = await llm_service.generate_completion(request)
            
            assert isinstance(response, LLMResponse)
            assert response.content == "Test response generated successfully"
            assert response.confidence >= 0.8
            mock_client.chat_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_prd_content(self, llm_service):
        """Test PRD content generation."""
        prompt = "Create a PRD section for user authentication"
        
        with patch.object(llm_service, 'generate_completion', new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = LLMResponse(
                content="## User Authentication Requirements\n\nThe system shall provide...",
                model="anthropic/claude-3.5-sonnet",
                provider="anthropic",
                confidence=0.95,
                token_usage={"total_tokens": 150},
                response_time_ms=250,
                fallback_used=False
            )
            
            response = await llm_service.generate_prd_content(prompt)
            
            assert isinstance(response, LLMResponse)
            assert "authentication" in response.content.lower()
            assert response.confidence >= 0.85  # High confidence required for PRD
            
            # Verify PRD-specific parameters were used
            call_args = mock_completion.call_args[0][0]
            assert call_args.task_type == "prd"
            assert call_args.complexity == "complex"
            assert call_args.require_high_confidence is True

    @pytest.mark.asyncio
    async def test_validate_content(self, llm_service):
        """Test content validation."""
        content = "This is a sample document that needs validation for quality and accuracy."
        
        with patch.object(llm_service, 'generate_completion', new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = LLMResponse(
                content="## Validation Results\n\nOverall: PASS\nIssues: None found\nRecommendations: Consider adding more detail",
                model="openai/gpt-4o",
                provider="openai",
                confidence=0.88,
                token_usage={"total_tokens": 120},
                response_time_ms=200,
                fallback_used=False
            )
            
            response = await llm_service.validate_content(content)
            
            assert isinstance(response, LLMResponse)
            assert "validation" in response.content.lower()
            
            # Verify validation-specific parameters
            call_args = mock_completion.call_args[0][0]
            assert call_args.task_type == "validation"
            assert call_args.require_high_confidence is True

    @pytest.mark.asyncio
    async def test_analyze_requirements(self, llm_service):
        """Test requirements analysis."""
        requirements = "User must be able to login with email and password. System should be secure."
        
        with patch.object(llm_service, 'generate_completion', new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = LLMResponse(
                content="## Requirements Analysis\n\nGaps identified: MFA not specified\nRisks: Password policy undefined",
                model="anthropic/claude-3-haiku",
                provider="anthropic",
                confidence=0.83,
                token_usage={"total_tokens": 200},
                response_time_ms=300,
                fallback_used=False
            )
            
            response = await llm_service.analyze_requirements(requirements)
            
            assert isinstance(response, LLMResponse)
            assert "analysis" in response.content.lower()
            
            # Verify analysis-specific parameters
            call_args = mock_completion.call_args[0][0]
            assert call_args.task_type == "analysis"
            assert call_args.complexity == "complex"

    @pytest.mark.asyncio 
    async def test_generate_summary(self, llm_service):
        """Test content summarization."""
        content = "Long content that needs to be summarized for executive review. " * 20
        
        with patch.object(llm_service, 'generate_completion', new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = LLMResponse(
                content="Executive Summary: Key points identified and consolidated for decision-making.",
                model="openai/gpt-4o-mini",
                provider="openai", 
                confidence=0.79,
                token_usage={"total_tokens": 80},
                response_time_ms=120,
                fallback_used=False
            )
            
            response = await llm_service.generate_summary(content)
            
            assert isinstance(response, LLMResponse)
            assert len(response.content) < len(content)  # Should be shorter
            
            # Verify summary-specific parameters
            call_args = mock_completion.call_args[0][0]
            assert call_args.task_type == "summary"
            assert call_args.complexity == "simple"

    @pytest.mark.asyncio
    async def test_llm_service_health_check(self, llm_service):
        """Test LLM service health check."""
        with patch.object(llm_service, 'generate_completion', new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = LLMResponse(
                content="Service operational",
                model="openai/gpt-4o-mini",
                provider="openai",
                confidence=0.99,
                token_usage={"total_tokens": 5},
                response_time_ms=80,
                fallback_used=False
            )
            
            health_status = await llm_service.health_check()
            
            assert health_status["status"] == "healthy"
            assert health_status["service"] == "LLMService"
            assert health_status["test_response_confidence"] == 0.99

    @pytest.mark.asyncio
    async def test_get_model_info(self, llm_service):
        """Test getting model information."""
        with patch.object(llm_service, '_get_openrouter_client', new_callable=AsyncMock) as mock_get_client:
            mock_client = Mock()
            mock_client.models = {
                "anthropic/claude-3.5-sonnet": Mock(
                    provider=ModelProvider.ANTHROPIC,
                    tier=ModelTier.PREMIUM,
                    max_tokens=8000,
                    confidence_threshold=0.9
                ),
                "openai/gpt-4o": Mock(
                    provider=ModelProvider.OPENAI,
                    tier=ModelTier.PREMIUM,
                    max_tokens=4000,
                    confidence_threshold=0.85
                )
            }
            mock_get_client.return_value = mock_client
            
            model_info = await llm_service.get_model_info()
            
            assert "configured_models" in model_info
            assert model_info["configured_models"] == 2
            assert "model_details" in model_info
            assert "anthropic/claude-3.5-sonnet" in model_info["model_details"]


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_openrouter_api_error_handling(self):
        """Test handling of OpenRouter API errors."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key-12345'}):
            client = OpenRouterClient()
            
            messages = [ChatMessage(role="user", content="Test")]
            
            with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
                # Simulate API error
                mock_response = Mock()
                mock_response.status_code = 429
                mock_response.text = "Rate limit exceeded"
                
                from httpx import HTTPStatusError
                mock_post.side_effect = HTTPStatusError(
                    "Rate limit exceeded", 
                    request=Mock(),
                    response=mock_response
                )
                
                with pytest.raises(HTTPStatusError):
                    await client.chat_completion(messages, max_retries=1)
                
            await client.close()

    @pytest.mark.asyncio
    async def test_all_models_fail_scenario(self):
        """Test scenario where all models fail."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key-12345'}):
            client = OpenRouterClient()
            
            messages = [ChatMessage(role="user", content="Test")]
            
            with patch.object(client, '_make_completion_request', new_callable=AsyncMock) as mock_request:
                mock_request.side_effect = Exception("All models unavailable")
                
                with pytest.raises(ValueError, match="All configured models failed"):
                    await client.chat_completion(messages)
                
            await client.close()

    @pytest.mark.asyncio
    async def test_invalid_api_response_format(self):
        """Test handling of invalid API response format."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key-12345'}):
            client = OpenRouterClient()
            
            with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
                # Invalid response format
                mock_response = Mock()
                mock_response.raise_for_status.return_value = None
                mock_response.json.return_value = {"invalid": "response"}
                mock_post.return_value = mock_response
                
                model_config = client.models["openai/gpt-4o-mini"]
                messages = [ChatMessage(role="user", content="Test")]
                
                with pytest.raises(ValueError, match="Invalid response format"):
                    await client._make_completion_request(messages, model_config)
                    
            await client.close()