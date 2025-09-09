"""
Unit tests for LLM API endpoints.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from fastapi.testclient import TestClient
from fastapi import status

from services.llm import LLMResponse
from models.user import User


class TestLLMEndpoints:
    """Test suite for LLM API endpoints."""

    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user."""
        return User(
            id="user-123",
            email="user@company.local",
            full_name="Test User",
            role="user",
            is_active=True
        )

    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response."""
        return LLMResponse(
            content="This is a test response from the AI model.",
            model="anthropic/claude-3.5-sonnet",
            provider="anthropic",
            confidence=0.92,
            token_usage={
                "prompt_tokens": 20,
                "completion_tokens": 15,
                "total_tokens": 35
            },
            response_time_ms=180,
            fallback_used=False
        )

    @pytest.mark.asyncio
    async def test_create_chat_completion_success(self, test_client, mock_user, mock_llm_response):
        """Test successful chat completion creation."""
        with patch('api.endpoints.llm.get_current_user', return_value=mock_user), \
             patch('api.endpoints.llm.get_llm_service') as mock_get_service:
            
            mock_service = AsyncMock()
            mock_service.generate_completion.return_value = mock_llm_response
            mock_get_service.return_value = mock_service
            
            response = test_client.post(
                "/api/v1/llm/chat/completions",
                json={
                    "messages": [
                        {"role": "user", "content": "Hello, how are you?"}
                    ],
                    "task_type": "general",
                    "complexity": "standard"
                }
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["content"] == mock_llm_response.content
            assert data["model"] == mock_llm_response.model
            assert data["confidence"] == mock_llm_response.confidence
            assert data["fallback_used"] == mock_llm_response.fallback_used

    @pytest.mark.asyncio
    async def test_create_chat_completion_unauthenticated(self, test_client):
        """Test chat completion without authentication."""
        response = test_client.post(
            "/api/v1/llm/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "Hello"}
                ]
            }
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_create_chat_completion_invalid_payload(self, test_client, mock_user):
        """Test chat completion with invalid payload."""
        with patch('api.endpoints.llm.get_current_user', return_value=mock_user):
            
            response = test_client.post(
                "/api/v1/llm/chat/completions",
                json={
                    "messages": "invalid_format"  # Should be list
                }
            )
            
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_create_chat_completion_service_error(self, test_client, mock_user):
        """Test chat completion when service fails."""
        with patch('api.endpoints.llm.get_current_user', return_value=mock_user), \
             patch('api.endpoints.llm.get_llm_service') as mock_get_service:
            
            mock_service = AsyncMock()
            mock_service.generate_completion.side_effect = Exception("Service unavailable")
            mock_get_service.return_value = mock_service
            
            response = test_client.post(
                "/api/v1/llm/chat/completions",
                json={
                    "messages": [
                        {"role": "user", "content": "Hello"}
                    ]
                }
            )
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            data = response.json()
            assert "Failed to generate completion" in data["detail"]

    @pytest.mark.asyncio
    async def test_generate_prd_content_success(self, test_client, mock_user, mock_llm_response):
        """Test successful PRD content generation."""
        prd_response = LLMResponse(
            content="## User Authentication\n\nThe system shall provide secure authentication...",
            model="anthropic/claude-3.5-sonnet",
            provider="anthropic",
            confidence=0.95,
            token_usage={"total_tokens": 150},
            response_time_ms=250,
            fallback_used=False
        )
        
        with patch('api.endpoints.llm.get_current_user', return_value=mock_user), \
             patch('api.endpoints.llm.get_llm_service') as mock_get_service:
            
            mock_service = AsyncMock()
            mock_service.generate_prd_content.return_value = prd_response
            mock_get_service.return_value = mock_service
            
            response = test_client.post(
                "/api/v1/llm/prd/generate",
                json={
                    "prompt": "Create a PRD section for user authentication",
                    "context": "Enterprise application",
                    "section_type": "requirements"
                }
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "authentication" in data["content"].lower()
            assert data["confidence"] >= 0.9
            mock_service.generate_prd_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_prd_content_short_prompt(self, test_client, mock_user):
        """Test PRD generation with too short prompt."""
        with patch('api.endpoints.llm.get_current_user', return_value=mock_user):
            
            response = test_client.post(
                "/api/v1/llm/prd/generate",
                json={
                    "prompt": "Short"  # Below 10 character minimum
                }
            )
            
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_validate_content_success(self, test_client, mock_user):
        """Test successful content validation."""
        validation_response = LLMResponse(
            content="## Validation Results\n\nOverall: PASS\nIssues: None identified",
            model="openai/gpt-4o",
            provider="openai",
            confidence=0.88,
            token_usage={"total_tokens": 120},
            response_time_ms=200,
            fallback_used=False
        )
        
        with patch('api.endpoints.llm.get_current_user', return_value=mock_user), \
             patch('api.endpoints.llm.get_llm_service') as mock_get_service:
            
            mock_service = AsyncMock()
            mock_service.validate_content.return_value = validation_response
            mock_get_service.return_value = mock_service
            
            response = test_client.post(
                "/api/v1/llm/validate",
                json={
                    "content": "This is content that needs validation for quality.",
                    "validation_type": "quality",
                    "criteria": ["clarity", "completeness", "accuracy"]
                }
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "validation" in data["content"].lower()
            assert data["confidence"] > 0.8

    @pytest.mark.asyncio
    async def test_analyze_requirements_success(self, test_client, mock_user):
        """Test successful requirements analysis."""
        analysis_response = LLMResponse(
            content="## Requirements Analysis\n\nGaps: MFA requirements missing\nRisks: Password policy not defined",
            model="anthropic/claude-3-haiku",
            provider="anthropic",
            confidence=0.83,
            token_usage={"total_tokens": 200},
            response_time_ms=300,
            fallback_used=False
        )
        
        with patch('api.endpoints.llm.get_current_user', return_value=mock_user), \
             patch('api.endpoints.llm.get_llm_service') as mock_get_service:
            
            mock_service = AsyncMock()
            mock_service.analyze_requirements.return_value = analysis_response
            mock_get_service.return_value = mock_service
            
            response = test_client.post(
                "/api/v1/llm/analyze/requirements",
                json={
                    "requirements_text": "Users must login with email and password. System should be secure.",
                    "analysis_type": "comprehensive"
                }
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "analysis" in data["content"].lower()
            mock_service.analyze_requirements.assert_called_once_with(
                requirements_text="Users must login with email and password. System should be secure.",
                analysis_type="comprehensive"
            )

    @pytest.mark.asyncio
    async def test_summarize_content_success(self, test_client, mock_user):
        """Test successful content summarization."""
        summary_response = LLMResponse(
            content="Executive Summary: This document covers key authentication requirements for the platform.",
            model="openai/gpt-4o-mini",
            provider="openai",
            confidence=0.79,
            token_usage={"total_tokens": 80},
            response_time_ms=120,
            fallback_used=False
        )
        
        with patch('api.endpoints.llm.get_current_user', return_value=mock_user), \
             patch('api.endpoints.llm.get_llm_service') as mock_get_service:
            
            mock_service = AsyncMock()
            mock_service.generate_summary.return_value = summary_response
            mock_get_service.return_value = mock_service
            
            long_content = "This is a very long document that needs summarization. " * 20
            
            response = test_client.post(
                "/api/v1/llm/summarize",
                json={
                    "content": long_content,
                    "summary_type": "executive",
                    "max_length": "medium"
                }
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "summary" in data["content"].lower()
            assert len(data["content"]) < len(long_content)

    @pytest.mark.asyncio
    async def test_summarize_content_too_short(self, test_client, mock_user):
        """Test summarization with content too short."""
        with patch('api.endpoints.llm.get_current_user', return_value=mock_user):
            
            response = test_client.post(
                "/api/v1/llm/summarize",
                json={
                    "content": "Short content"  # Below 50 character minimum
                }
            )
            
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_health_check_success(self, test_client, mock_user):
        """Test successful health check."""
        health_status = {
            "status": "healthy",
            "service": "LLMService",
            "openrouter_status": "connected",
            "test_response_confidence": 0.95,
            "model_used": "openai/gpt-4o-mini",
            "response_time_ms": 100
        }
        
        with patch('api.endpoints.llm.get_current_user', return_value=mock_user), \
             patch('api.endpoints.llm.get_llm_service') as mock_get_service:
            
            mock_service = AsyncMock()
            mock_service.health_check.return_value = health_status
            mock_get_service.return_value = mock_service
            
            response = test_client.get("/api/v1/llm/health")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["status"] == "healthy"
            assert data["service"] == "LLMService"
            assert data["test_response_confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_health_check_failure(self, test_client, mock_user):
        """Test health check failure."""
        with patch('api.endpoints.llm.get_current_user', return_value=mock_user), \
             patch('api.endpoints.llm.get_llm_service') as mock_get_service:
            
            mock_service = AsyncMock()
            mock_service.health_check.side_effect = Exception("Health check failed")
            mock_get_service.return_value = mock_service
            
            response = test_client.get("/api/v1/llm/health")
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            data = response.json()
            assert "Health check failed" in data["detail"]

    @pytest.mark.asyncio
    async def test_get_model_info_success(self, test_client, mock_user):
        """Test successful model info retrieval."""
        model_info = {
            "configured_models": 8,
            "model_details": {
                "anthropic/claude-3.5-sonnet": {
                    "provider": "anthropic",
                    "tier": "premium",
                    "max_tokens": 8000,
                    "confidence_threshold": 0.9
                },
                "openai/gpt-4o": {
                    "provider": "openai",
                    "tier": "premium", 
                    "max_tokens": 4000,
                    "confidence_threshold": 0.85
                }
            },
            "default_model": "gpt-4o",
            "fallback_model": "gpt-4o-mini"
        }
        
        with patch('api.endpoints.llm.get_current_user', return_value=mock_user), \
             patch('api.endpoints.llm.get_llm_service') as mock_get_service:
            
            mock_service = AsyncMock()
            mock_service.get_model_info.return_value = model_info
            mock_get_service.return_value = mock_service
            
            response = test_client.get("/api/v1/llm/models")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["configured_models"] == 8
            assert "anthropic/claude-3.5-sonnet" in data["model_details"]
            assert data["default_model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_get_usage_statistics_placeholder(self, test_client, mock_user):
        """Test usage statistics placeholder endpoint."""
        with patch('api.endpoints.llm.get_current_user', return_value=mock_user):
            
            response = test_client.get("/api/v1/llm/usage/stats")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "message" in data
            assert "coming soon" in data["message"]
            assert data["user_id"] == mock_user.id

    @pytest.mark.asyncio
    async def test_complex_chat_completion_with_options(self, test_client, mock_user, mock_llm_response):
        """Test chat completion with all optional parameters."""
        with patch('api.endpoints.llm.get_current_user', return_value=mock_user), \
             patch('api.endpoints.llm.get_llm_service') as mock_get_service:
            
            mock_service = AsyncMock()
            mock_service.generate_completion.return_value = mock_llm_response
            mock_get_service.return_value = mock_service
            
            response = test_client.post(
                "/api/v1/llm/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Generate a comprehensive analysis."}
                    ],
                    "model": "anthropic/claude-3.5-sonnet",
                    "task_type": "analysis",
                    "complexity": "complex",
                    "temperature": 0.2,
                    "max_tokens": 6000,
                    "require_high_confidence": True
                }
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["model"] == mock_llm_response.model
            assert data["confidence"] == mock_llm_response.confidence
            
            # Verify the service was called with correct parameters
            call_args = mock_service.generate_completion.call_args[0][0]
            assert call_args.model == "anthropic/claude-3.5-sonnet"
            assert call_args.task_type == "analysis"
            assert call_args.complexity == "complex"
            assert call_args.temperature == 0.2
            assert call_args.max_tokens == 6000
            assert call_args.require_high_confidence is True