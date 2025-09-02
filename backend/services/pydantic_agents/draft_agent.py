"""
Draft Agent - Rapid content generation and iterative drafting agent.

This agent specializes in quickly generating initial drafts, prototypes,
and iterative improvements to content with fast turnaround times.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import json

from pydantic_ai import Agent as PydanticAIAgent
from pydantic import BaseModel, Field
import structlog

from .base_agent import BaseAgent, AgentResult
from core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class DraftSpecification(BaseModel):
    """Specification for draft content generation."""
    content_type: str = Field(..., description="Type of content to generate")
    target_length: str = Field(..., description="Target length (brief, medium, detailed)")
    tone: str = Field(..., description="Desired tone (formal, conversational, technical)")
    audience: str = Field(..., description="Target audience")
    key_points: List[str] = Field(..., description="Key points to cover")
    constraints: List[str] = Field(default_factory=list, description="Content constraints")


class IterationFeedback(BaseModel):
    """Feedback for iterative improvements."""
    feedback_type: str = Field(..., description="Type of feedback")
    specific_areas: List[str] = Field(..., description="Specific areas to improve")
    priority: str = Field(..., description="Priority level")
    suggestions: List[str] = Field(..., description="Specific suggestions")


class DraftAgent(BaseAgent):
    """
    Draft Agent specializing in rapid content generation and iteration.
    
    Capabilities:
    - Quick first-draft generation for any content type
    - Iterative content improvement based on feedback
    - Content outline and structure creation
    - Rapid prototyping of ideas and concepts
    - Multi-format content adaptation
    - Speed-optimized generation with quality balance
    """
    
    def _initialize_agent(self) -> None:
        """Initialize the PydanticAI agent for content drafting."""
        self.pydantic_agent = PydanticAIAgent(
            model_name='openai:gpt-4o',
            system_prompt="""You are the Draft Agent for an AI-powered strategic planning platform.

You specialize in rapid, high-quality content generation and iterative improvement. Your role is to quickly produce initial drafts that can be refined through iteration.

**Core Capabilities:**

1. **Rapid Generation**: Create first drafts quickly while maintaining quality
2. **Iterative Improvement**: Enhance content based on specific feedback
3. **Format Flexibility**: Adapt content to different formats and audiences
4. **Structure Creation**: Build solid content frameworks and outlines
5. **Idea Development**: Transform concepts into structured content

**Content Types You Handle:**
- Executive summaries and business overviews
- Technical specifications and requirements
- User stories and acceptance criteria
- Project plans and timelines
- Risk assessments and mitigation strategies
- Marketing and communication materials
- Process documentation and procedures

**Quality Standards:**
- Clear, well-structured content organization
- Appropriate tone and style for target audience
- Comprehensive coverage of required topics
- Professional presentation and formatting
- Actionable and practical recommendations

**Drafting Principles:**
- Speed with quality: Generate quickly without sacrificing clarity
- Iterative mindset: Design content for easy revision and improvement
- Audience awareness: Tailor content to specific stakeholder needs
- Structure first: Establish solid frameworks before detail
- Practical focus: Ensure content is actionable and implementable

**Response Format:**
Always provide structured content with:
- Clear headings and organization
- Bullet points for key information
- Logical flow and transitions
- Specific, actionable items
- Professional but accessible language

You prioritize getting useful content quickly while maintaining professional quality.""",
            deps_type=Dict[str, Any]
        )
    
    async def execute(self, operation: str, context: Dict[str, Any]) -> AgentResult:
        """Execute a content drafting operation."""
        start_time = self._log_operation_start(operation, context)
        
        try:
            if operation == "generate_initial_draft":
                result = await self._generate_initial_draft(context)
            elif operation == "iterate_content":
                result = await self._iterate_content(context)
            elif operation == "create_content_outline":
                result = await self._create_content_outline(context)
            elif operation == "adapt_content_format":
                result = await self._adapt_content_format(context)
            elif operation == "expand_content_section":
                result = await self._expand_content_section(context)
            elif operation == "summarize_content":
                result = await self._summarize_content(context)
            elif operation == "generate_multiple_variants":
                result = await self._generate_multiple_variants(context)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            processing_time_ms = self._log_operation_complete(operation, start_time, True)
            
            # Light validation for draft content (faster than full GraphRAG)
            validation_results = []
            if result.get("generated_content"):
                # Quick validation for obvious issues
                content_str = str(result["generated_content"])
                basic_validation = self._basic_content_validation(content_str)
                validation_results.append(basic_validation)
            
            return self._create_success_result(
                result=result,
                validation_results=validation_results,
                processing_time_ms=processing_time_ms,
                confidence_score=validation_results[0].get("confidence", 0.8) if validation_results else 0.8
            )
            
        except Exception as e:
            processing_time_ms = self._log_operation_complete(operation, start_time, False, str(e))
            return self._create_error_result(
                error=str(e),
                metadata={"processing_time_ms": processing_time_ms}
            )
    
    async def _generate_initial_draft(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an initial draft based on requirements."""
        
        content_type = self._extract_context_parameter(context, "content_type")
        requirements = self._extract_context_parameter(context, "requirements")
        specifications = self._extract_context_parameter(context, "specifications", required=False, default={})
        
        draft_prompt = f"""Generate an initial draft for the following content:

Content Type: {content_type}
Requirements: {json.dumps(requirements, indent=2)}
Specifications: {json.dumps(specifications, indent=2)}

Create a comprehensive initial draft that includes:

1. **Clear Structure**: Well-organized sections with logical flow
2. **Complete Coverage**: Address all key requirements and topics
3. **Professional Quality**: Business-appropriate tone and presentation
4. **Actionable Content**: Specific, implementable recommendations
5. **Appropriate Detail**: Right level of detail for the content type

Key areas to cover based on content type:
- Executive summary content: Business context, objectives, recommendations
- Technical specifications: Requirements, architecture, implementation details
- Project plans: Timeline, resources, milestones, deliverables
- User stories: User needs, acceptance criteria, success measures
- Risk assessments: Identified risks, impact analysis, mitigation strategies

Focus on creating a solid foundation that can be refined through iteration.
Prioritize clarity, completeness, and actionability."""

        result = await self.pydantic_agent.run(
            draft_prompt,
            deps={
                "content_type": content_type,
                "requirements": requirements,
                "specifications": specifications,
                "hybrid_rag": self.hybrid_rag
            }
        )
        
        return {
            "initial_draft": result.data,
            "content_type": content_type,
            "draft_version": "1.0",
            "generation_timestamp": datetime.utcnow().isoformat(),
            "generated_content": result.data
        }
    
    async def _iterate_content(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Improve content based on feedback."""
        
        current_content = self._extract_context_parameter(context, "current_content")
        feedback = self._extract_context_parameter(context, "feedback")
        iteration_goals = self._extract_context_parameter(context, "iteration_goals", required=False, default=[])
        
        iteration_prompt = f"""Improve the following content based on specific feedback:

Current Content: {json.dumps(current_content, indent=2)}
Feedback: {json.dumps(feedback, indent=2)}
Iteration Goals: {json.dumps(iteration_goals, indent=2)}

Apply the feedback to create an improved version that:

1. **Addresses Specific Feedback**: Directly address each piece of feedback provided
2. **Maintains Strengths**: Keep the good elements from the current version
3. **Enhances Weak Areas**: Significantly improve areas identified as needing work
4. **Improves Overall Quality**: Raise the overall quality and effectiveness
5. **Preserves Intent**: Maintain the original purpose and key messages

For each piece of feedback:
- Identify the specific area to improve
- Determine the best approach for improvement
- Implement changes while maintaining coherence
- Ensure changes align with overall goals

Iteration approach:
- Prioritize high-impact improvements first
- Make targeted changes rather than wholesale rewrites
- Enhance clarity and specificity
- Strengthen weak arguments or sections
- Add missing elements identified in feedback

Provide the improved content with clear indication of what was changed and why."""

        result = await self.pydantic_agent.run(
            iteration_prompt,
            deps={
                "current_content": current_content,
                "feedback": feedback,
                "iteration_goals": iteration_goals,
                "hybrid_rag": self.hybrid_rag
            }
        )
        
        return {
            "improved_content": result.data,
            "iteration_applied": feedback,
            "improvement_timestamp": datetime.utcnow().isoformat(),
            "generated_content": result.data
        }
    
    async def _create_content_outline(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a detailed content outline and structure."""
        
        topic = self._extract_context_parameter(context, "topic")
        content_type = self._extract_context_parameter(context, "content_type")
        target_audience = self._extract_context_parameter(context, "target_audience", required=False, default="general")
        depth_level = self._extract_context_parameter(context, "depth_level", required=False, default="medium")
        
        outline_prompt = f"""Create a comprehensive content outline for:

Topic: {topic}
Content Type: {content_type}
Target Audience: {target_audience}
Depth Level: {depth_level}

Develop a detailed outline that includes:

1. **Main Structure**: Primary sections and subsections
2. **Key Points**: Important points to cover in each section
3. **Flow Logic**: Logical progression and transitions between sections
4. **Content Elements**: Specific types of content for each section (text, examples, data, etc.)
5. **Estimated Scope**: Rough length and complexity for each section

Outline format:
- Use hierarchical structure (1.0, 1.1, 1.1.1)
- Include brief descriptions for each section
- Note key messages and takeaways
- Identify areas requiring research or data
- Specify audience-appropriate tone and style

Consider content type best practices:
- Executive summaries: Brief, high-level, decision-focused
- Technical documents: Detailed, structured, implementation-focused
- User guides: Step-by-step, practical, user-friendly
- Business plans: Strategic, data-driven, comprehensive

Create an outline that serves as a clear roadmap for content development."""

        result = await self.pydantic_agent.run(
            outline_prompt,
            deps={
                "topic": topic,
                "content_type": content_type,
                "target_audience": target_audience,
                "depth_level": depth_level
            }
        )
        
        return {
            "content_outline": result.data,
            "topic": topic,
            "content_type": content_type,
            "outline_timestamp": datetime.utcnow().isoformat(),
            "generated_content": result.data
        }
    
    async def _adapt_content_format(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt existing content to a different format or audience."""
        
        source_content = self._extract_context_parameter(context, "source_content")
        target_format = self._extract_context_parameter(context, "target_format")
        target_audience = self._extract_context_parameter(context, "target_audience", required=False)
        adaptation_requirements = self._extract_context_parameter(context, "requirements", required=False, default={})
        
        adaptation_prompt = f"""Adapt the following content to a new format and/or audience:

Source Content: {json.dumps(source_content, indent=2)}
Target Format: {target_format}
Target Audience: {target_audience or 'Not specified'}
Adaptation Requirements: {json.dumps(adaptation_requirements, indent=2)}

Adapt the content while:

1. **Preserving Core Message**: Keep the essential information and key messages
2. **Adjusting Format**: Restructure to match target format conventions
3. **Audience Alignment**: Modify tone, terminology, and detail level for new audience
4. **Format Optimization**: Use format-specific best practices and structures
5. **Adding/Removing Content**: Include or exclude content appropriate for new context

Format-specific adaptations:
- Executive summary: High-level, decision-focused, brief
- Technical documentation: Detailed, structured, implementation-focused
- Presentation slides: Concise, visual, key points only
- Email communication: Professional, actionable, scannable
- User guide: Step-by-step, practical, user-friendly
- Report: Comprehensive, data-driven, analytical

Audience-specific adaptations:
- Technical audience: More detail, technical terminology, implementation focus
- Executive audience: Strategic focus, business impact, high-level overview
- General audience: Accessible language, clear explanations, practical relevance

Ensure the adapted content is effective and appropriate for its new purpose."""

        result = await self.pydantic_agent.run(
            adaptation_prompt,
            deps={
                "source_content": source_content,
                "target_format": target_format,
                "target_audience": target_audience,
                "requirements": adaptation_requirements
            }
        )
        
        return {
            "adapted_content": result.data,
            "source_format": "original",
            "target_format": target_format,
            "adaptation_timestamp": datetime.utcnow().isoformat(),
            "generated_content": result.data
        }
    
    async def _expand_content_section(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Expand a specific section of content with more detail."""
        
        section_content = self._extract_context_parameter(context, "section_content")
        section_topic = self._extract_context_parameter(context, "section_topic")
        expansion_type = self._extract_context_parameter(context, "expansion_type", required=False, default="detailed")
        focus_areas = self._extract_context_parameter(context, "focus_areas", required=False, default=[])
        
        expansion_prompt = f"""Expand the following content section with additional detail:

Current Section Content: {json.dumps(section_content, indent=2)}
Section Topic: {section_topic}
Expansion Type: {expansion_type}
Focus Areas: {json.dumps(focus_areas, indent=2)}

Expand the section by:

1. **Adding Depth**: Provide more detailed explanations and analysis
2. **Including Examples**: Add relevant examples, case studies, or scenarios  
3. **Expanding Context**: Provide broader context and background information
4. **Adding Supporting Information**: Include data, research, or evidence
5. **Enhancing Structure**: Improve organization and flow of information

Expansion approaches by type:
- Detailed: Add comprehensive information and thorough coverage
- Examples-focused: Include multiple relevant examples and use cases
- Analysis-focused: Provide deeper analysis and interpretation
- Practical-focused: Add implementation steps and practical guidance
- Research-focused: Include supporting data and evidence

If focus areas are specified, prioritize expansion in those areas while maintaining overall coherence.

Ensure the expanded content maintains the original intent while significantly enhancing value and usefulness."""

        result = await self.pydantic_agent.run(
            expansion_prompt,
            deps={
                "section_content": section_content,
                "section_topic": section_topic,
                "expansion_type": expansion_type,
                "focus_areas": focus_areas
            }
        )
        
        return {
            "expanded_content": result.data,
            "section_topic": section_topic,
            "expansion_type": expansion_type,
            "expansion_timestamp": datetime.utcnow().isoformat(),
            "generated_content": result.data
        }
    
    async def _summarize_content(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of existing content."""
        
        source_content = self._extract_context_parameter(context, "source_content")
        summary_type = self._extract_context_parameter(context, "summary_type", required=False, default="executive")
        target_length = self._extract_context_parameter(context, "target_length", required=False, default="medium")
        key_focus = self._extract_context_parameter(context, "key_focus", required=False, default="main_points")
        
        summary_prompt = f"""Create a summary of the following content:

Source Content: {json.dumps(source_content, indent=2)}
Summary Type: {summary_type}
Target Length: {target_length}
Key Focus: {key_focus}

Create a high-quality summary that:

1. **Captures Key Points**: Include all essential information and main messages
2. **Maintains Clarity**: Use clear, concise language appropriate for summary format
3. **Preserves Context**: Keep important context and relationships between ideas
4. **Highlights Priorities**: Emphasize the most important and actionable items
5. **Provides Structure**: Organize summary logically and coherently

Summary type guidelines:
- Executive: High-level overview focusing on decisions and business impact
- Technical: Key technical details and implementation considerations
- Action-oriented: Focus on required actions and next steps
- Analytical: Key insights, findings, and conclusions
- Overview: Comprehensive but concise coverage of all main areas

Length guidelines:
- Brief: 2-3 paragraphs, key points only
- Medium: 4-6 paragraphs, main points with context
- Detailed: Multiple sections, comprehensive coverage while still condensed

Focus area priorities:
- Main points: Core messages and key information
- Actions: Required actions and next steps
- Decisions: Key decisions and recommendations
- Outcomes: Expected results and success measures

Ensure the summary stands alone and provides clear value to readers."""

        result = await self.pydantic_agent.run(
            summary_prompt,
            deps={
                "source_content": source_content,
                "summary_type": summary_type,
                "target_length": target_length,
                "key_focus": key_focus
            }
        )
        
        return {
            "content_summary": result.data,
            "summary_type": summary_type,
            "target_length": target_length,
            "summary_timestamp": datetime.utcnow().isoformat(),
            "generated_content": result.data
        }
    
    async def _generate_multiple_variants(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate multiple variants of content for comparison."""
        
        base_requirements = self._extract_context_parameter(context, "base_requirements")
        variant_count = self._extract_context_parameter(context, "variant_count", required=False, default=3)
        variation_aspects = self._extract_context_parameter(context, "variation_aspects", required=False, default=["tone", "approach"])
        
        variants_prompt = f"""Generate {variant_count} different variants of content based on:

Base Requirements: {json.dumps(base_requirements, indent=2)}
Variation Aspects: {json.dumps(variation_aspects, indent=2)}

Create {variant_count} distinct variants that differ in:
{', '.join(variation_aspects)}

For each variant:

1. **Unique Approach**: Take a different approach to addressing the requirements
2. **Distinct Style**: Use different tone, structure, or presentation style
3. **Alternative Focus**: Emphasize different aspects of the requirements
4. **Varied Structure**: Organize information differently while covering same topics
5. **Different Strengths**: Each variant should excel in different areas

Variation strategies:
- Tone: Formal vs. conversational vs. technical
- Approach: Strategic vs. tactical vs. operational
- Structure: Narrative vs. analytical vs. practical
- Focus: Business-focused vs. user-focused vs. technical-focused
- Detail level: High-level overview vs. detailed specifications

Label each variant clearly (Variant A, B, C) and provide:
- Brief description of the variant's approach
- The content itself
- Key strengths and intended use case

This will allow for comparison and selection of the best approach."""

        result = await self.pydantic_agent.run(
            variants_prompt,
            deps={
                "base_requirements": base_requirements,
                "variant_count": variant_count,
                "variation_aspects": variation_aspects
            }
        )
        
        return {
            "content_variants": result.data,
            "variant_count": variant_count,
            "variation_aspects": variation_aspects,
            "variants_timestamp": datetime.utcnow().isoformat(),
            "generated_content": result.data
        }
    
    def _basic_content_validation(self, content: str) -> Dict[str, Any]:
        """Perform basic validation checks on content."""
        
        validation_result = {
            "validation_type": "basic_draft_validation",
            "confidence": 0.8,
            "passes_threshold": True,
            "issues": [],
            "strengths": []
        }
        
        # Basic length check
        if len(content.strip()) < 100:
            validation_result["issues"].append("Content appears too short")
            validation_result["confidence"] -= 0.2
        
        # Basic structure check
        if "\n" not in content or content.count("\n") < 3:
            validation_result["issues"].append("Content lacks clear structure")
            validation_result["confidence"] -= 0.1
        else:
            validation_result["strengths"].append("Content has clear structure")
        
        # Check for placeholder text
        placeholders = ["[placeholder]", "[TODO]", "[TBD]", "replace-me", "fill-in"]
        for placeholder in placeholders:
            if placeholder.lower() in content.lower():
                validation_result["issues"].append(f"Contains placeholder text: {placeholder}")
                validation_result["confidence"] -= 0.2
        
        # Check for reasonable quality
        if len(content.split()) < 50:
            validation_result["issues"].append("Content appears incomplete")
            validation_result["confidence"] -= 0.1
        else:
            validation_result["strengths"].append("Content has reasonable depth")
        
        # Update pass/fail based on issues
        validation_result["passes_threshold"] = validation_result["confidence"] >= 0.6 and len(validation_result["issues"]) <= 2
        
        return validation_result