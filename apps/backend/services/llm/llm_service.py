"""
LLM Service - Main interface for AI model interactions with fallback strategies.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .openrouter_client import (
    OpenRouterClient,
    ChatMessage,
    LLMResponse,
    get_openrouter_client
)
from core.config import get_settings
from ..cache_service import get_cache_service, CacheNamespace

logger = logging.getLogger(__name__)


class LLMRequest(BaseModel):
    """Standard LLM request format."""
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    model: Optional[str] = Field(default=None, description="Preferred model")
    task_type: str = Field(default="general", description="Type of task: general, prd, validation, etc.")
    complexity: str = Field(default="standard", description="Task complexity: simple, standard, complex")
    temperature: Optional[float] = Field(default=None, description="Override temperature")
    max_tokens: Optional[int] = Field(default=None, description="Override max tokens")
    require_high_confidence: bool = Field(default=False, description="Require high confidence response")


class LLMService:
    """
    Main LLM service providing unified interface for AI operations.
    Handles model selection, fallback strategies, and request routing.
    """

    def __init__(self):
        self.settings = get_settings()
        self._openrouter_client: Optional[OpenRouterClient] = None
        self._cache_service = get_cache_service()

    async def _get_openrouter_client(self) -> OpenRouterClient:
        """Get OpenRouter client instance."""
        if self._openrouter_client is None:
            self._openrouter_client = await get_openrouter_client()
        return self._openrouter_client

    async def close(self):
        """Close all client connections."""
        if self._openrouter_client:
            await self._openrouter_client.close()
            self._openrouter_client = None

    def _determine_task_complexity(self, request: LLMRequest) -> str:
        """Determine task complexity based on request parameters."""
        if request.complexity != "standard":
            return request.complexity

        # Auto-detect complexity based on task type and content
        complexity_mapping = {
            "prd": "complex",
            "validation": "complex", 
            "graphrag": "complex",
            "planning": "complex",
            "analysis": "standard",
            "summary": "simple",
            "qa": "simple",
            "general": "standard"
        }
        
        return complexity_mapping.get(request.task_type, "standard")

    def _select_model_for_task(self, request: LLMRequest) -> Optional[str]:
        """Select appropriate model based on task type and complexity."""
        if request.model:
            return request.model

        task_complexity = self._determine_task_complexity(request)
        
        # Model selection based on task type and complexity
        if request.task_type in ["prd", "planning", "graphrag"]:
            if task_complexity == "complex":
                return "anthropic/claude-3.5-sonnet"
            else:
                return "anthropic/claude-3-haiku"
                
        elif request.task_type in ["validation", "analysis"]:
            return "openai/gpt-4o"
            
        elif request.task_type in ["summary", "qa", "general"]:
            if task_complexity == "simple":
                return "openai/gpt-4o-mini"
            else:
                return "openai/gpt-4o"
        
        return None  # Let OpenRouter client decide

    async def generate_completion(self, request: LLMRequest) -> LLMResponse:
        """
        Generate completion using optimal model selection and fallback.
        
        Args:
            request: LLM request with messages and configuration
            
        Returns:
            LLM response with metadata
        """
        try:
            selected_model = self._select_model_for_task(request)
            task_complexity = self._determine_task_complexity(request)
            
            # Check cache first for non-creative tasks
            use_cache = request.task_type not in ["creative", "brainstorm"] and request.temperature <= 0.3
            
            if use_cache and self._cache_service.is_available:
                # Generate prompt text for cache key
                prompt_text = "\n".join([msg.content for msg in request.messages])
                
                cached_response = await self._cache_service.get_llm_response(
                    prompt=prompt_text,
                    model=selected_model or self.settings.default_model
                )
                
                if cached_response:
                    logger.info(f"Cache hit for LLM request (task={request.task_type})")
                    # Convert cached response back to LLMResponse
                    return LLMResponse(
                        content=cached_response["response"],
                        model=cached_response["model"],
                        confidence=cached_response.get("confidence", 0.9),
                        fallback_used=False,
                        metadata=cached_response.get("metadata", {})
                    )
            
            # No cache hit, proceed with LLM call
            client = await self._get_openrouter_client()
            
            # Determine confidence requirement
            min_confidence = 0.9 if request.require_high_confidence else 0.8
            if request.task_type in ["prd", "validation", "graphrag"]:
                min_confidence = max(min_confidence, 0.85)
            
            logger.info(f"Generating completion for task '{request.task_type}' "
                       f"with complexity '{task_complexity}' using model '{selected_model}'")
            
            response = await client.chat_completion(
                messages=request.messages,
                model=selected_model,
                task_complexity=task_complexity,
                require_confidence=min_confidence
            )
            
            logger.info(f"Completion generated: model={response.model}, "
                       f"confidence={response.confidence}, "
                       f"fallback_used={response.fallback_used}")
            
            # Cache the response if eligible
            if use_cache and self._cache_service.is_available:
                prompt_text = "\n".join([msg.content for msg in request.messages])
                await self._cache_service.cache_llm_response(
                    prompt=prompt_text,
                    model=response.model,
                    response=response.content,
                    metadata={
                        "confidence": response.confidence,
                        "task_type": request.task_type,
                        "complexity": task_complexity,
                        "fallback_used": response.fallback_used
                    }
                )
                logger.debug(f"Cached LLM response for future use")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate completion: {str(e)}")
            raise

    async def generate_prd_content(
        self,
        prompt: str,
        context: Optional[str] = None,
        section_type: str = "general"
    ) -> LLMResponse:
        """
        Generate PRD content with specialized prompting.
        
        Args:
            prompt: Content generation prompt
            context: Optional context information
            section_type: Type of PRD section (overview, requirements, etc.)
            
        Returns:
            Generated PRD content
        """
        messages = []
        
        # System prompt for PRD generation
        system_prompt = """You are a senior product manager and technical writer specializing in creating comprehensive Product Requirements Documents (PRDs). 

Your PRDs should be:
- Clear, detailed, and actionable
- Structured with proper sections and subsections
- Include specific technical requirements
- Consider user experience, business goals, and technical constraints
- Follow industry best practices for PRD documentation

Always provide complete, professional content that stakeholders can use to implement features."""

        messages.append(ChatMessage(role="system", content=system_prompt))
        
        if context:
            messages.append(ChatMessage(role="user", content=f"Context: {context}"))
        
        messages.append(ChatMessage(role="user", content=prompt))
        
        request = LLMRequest(
            messages=messages,
            task_type="prd",
            complexity="complex",
            require_high_confidence=True
        )
        
        return await self.generate_completion(request)

    async def validate_content(
        self,
        content: str,
        validation_type: str = "general",
        criteria: Optional[List[str]] = None
    ) -> LLMResponse:
        """
        Validate content for quality, accuracy, and completeness.
        
        Args:
            content: Content to validate
            validation_type: Type of validation (prd, technical, business)
            criteria: Optional specific validation criteria
            
        Returns:
            Validation results with recommendations
        """
        system_prompt = f"""You are a quality assurance specialist performing {validation_type} validation. 

Analyze the provided content for:
- Accuracy and factual correctness
- Completeness and coverage
- Clarity and readability
- Technical feasibility (if applicable)
- Business alignment (if applicable)

Provide a structured validation report with:
1. Overall assessment (Pass/Fail/Needs Review)
2. Specific issues found
3. Recommendations for improvement
4. Confidence level in your assessment"""

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=f"Content to validate:\n\n{content}")
        ]
        
        if criteria:
            criteria_text = "\n".join(f"- {criterion}" for criterion in criteria)
            messages.append(ChatMessage(
                role="user", 
                content=f"Additional validation criteria:\n{criteria_text}"
            ))
        
        request = LLMRequest(
            messages=messages,
            task_type="validation",
            complexity="complex",
            require_high_confidence=True
        )
        
        return await self.generate_completion(request)

    async def analyze_requirements(
        self,
        requirements_text: str,
        analysis_type: str = "comprehensive"
    ) -> LLMResponse:
        """
        Analyze requirements for gaps, conflicts, and improvements.
        
        Args:
            requirements_text: Requirements to analyze
            analysis_type: Type of analysis (comprehensive, technical, business)
            
        Returns:
            Analysis results with insights and recommendations
        """
        system_prompt = f"""You are a business analyst and requirements engineer performing {analysis_type} analysis.

Analyze the provided requirements for:
- Completeness and coverage
- Potential conflicts or contradictions
- Missing dependencies
- Technical feasibility
- Business value alignment
- Risk factors

Provide a structured analysis with:
1. Summary of findings
2. Identified gaps or issues
3. Risk assessment
4. Recommendations for improvement
5. Priority suggestions"""

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=f"Requirements to analyze:\n\n{requirements_text}")
        ]
        
        request = LLMRequest(
            messages=messages,
            task_type="analysis",
            complexity="complex"
        )
        
        return await self.generate_completion(request)

    async def generate_summary(
        self,
        content: str,
        summary_type: str = "executive",
        max_length: str = "medium"
    ) -> LLMResponse:
        """
        Generate summaries of content.
        
        Args:
            content: Content to summarize
            summary_type: Type of summary (executive, technical, bullet-points)
            max_length: Summary length (short, medium, long)
            
        Returns:
            Generated summary
        """
        length_guidance = {
            "short": "1-2 paragraphs",
            "medium": "3-4 paragraphs",
            "long": "5-6 paragraphs with detailed points"
        }
        
        system_prompt = f"""You are a professional summarizer creating {summary_type} summaries.

Create a {max_length} summary ({length_guidance[max_length]}) that captures:
- Key points and main themes
- Important details and conclusions
- Action items or next steps (if applicable)
- Critical insights or implications

The summary should be clear, concise, and valuable for decision-makers."""

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=f"Content to summarize:\n\n{content}")
        ]
        
        request = LLMRequest(
            messages=messages,
            task_type="summary",
            complexity="simple"
        )
        
        return await self.generate_completion(request)

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on LLM service."""
        try:
            client = await self._get_openrouter_client()
            
            # Test basic functionality
            test_request = LLMRequest(
                messages=[ChatMessage(role="user", content="Respond with 'Service operational' if working correctly.")],
                task_type="general",
                complexity="simple"
            )
            
            response = await self.generate_completion(test_request)
            
            return {
                "status": "healthy",
                "service": "LLMService",
                "openrouter_status": "connected",
                "test_response_confidence": response.confidence,
                "model_used": response.model,
                "response_time_ms": response.response_time_ms
            }
            
        except Exception as e:
            logger.error(f"LLM service health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "service": "LLMService", 
                "error": str(e)
            }

    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models."""
        try:
            client = await self._get_openrouter_client()
            
            return {
                "configured_models": len(client.models),
                "model_details": {
                    name: {
                        "provider": config.provider.value,
                        "tier": config.tier.value,
                        "max_tokens": config.max_tokens,
                        "confidence_threshold": config.confidence_threshold
                    }
                    for name, config in client.models.items()
                },
                "default_model": self.settings.default_model,
                "fallback_model": self.settings.fallback_model
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            return {"error": str(e)}


# Global service instance
_llm_service: Optional[LLMService] = None


async def get_llm_service() -> LLMService:
    """Get or create LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


async def close_llm_service():
    """Close LLM service."""
    global _llm_service
    if _llm_service:
        await _llm_service.close()
        _llm_service = None