"""
PRD Generator Agent - Specialized agent for creating Product Requirements Documents.

This agent handles all phases of PRD creation from initial concept analysis
to comprehensive document generation with GraphRAG validation.
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


class RequirementAnalysis(BaseModel):
    """Analysis of initial requirements."""
    extracted_concepts: List[str] = Field(..., description="Key concepts identified")
    stakeholder_types: List[str] = Field(..., description="Identified stakeholder types")
    complexity_assessment: str = Field(..., description="Complexity level assessment")
    technical_feasibility: float = Field(..., ge=0.0, le=1.0, description="Technical feasibility score")
    business_impact: str = Field(..., description="Expected business impact")
    recommended_approach: str = Field(..., description="Recommended development approach")


class ClarificationQuestion(BaseModel):
    """Clarification question for stakeholders."""
    id: str = Field(..., description="Question ID")
    question: str = Field(..., description="Question text")
    category: str = Field(..., description="Question category")
    priority: str = Field(..., description="Question priority")
    context: str = Field(..., description="Context for the question")


class PRDGeneratorAgent(BaseAgent):
    """
    PRD Generator Agent specializing in comprehensive PRD creation.
    
    Capabilities:
    - Initial requirements analysis and concept extraction
    - Phase 0: Project invitation and clarification questions
    - Phase 1: Objective clarification and validation
    - Phase 2: Detailed requirement drafting
    - Phase 3: Section co-creation and enhancement
    - Phase 4: Final synthesis and validation
    """
    
    def _initialize_agent(self) -> None:
        """Initialize the PydanticAI agent for PRD generation."""
        self.pydantic_agent = PydanticAIAgent(
            model_name='openai:gpt-4o',
            system_prompt="""You are the PRD Generator Agent for an AI-powered strategic planning platform.

You specialize in creating comprehensive, high-quality Product Requirements Documents (PRDs) through a structured 4-phase approach:

**Phase 0 - Project Invitation**: Analyze initial concepts and generate clarification questions
**Phase 1 - Objective Clarification**: Process stakeholder answers and validate requirements
**Phase 2 - Requirement Drafting**: Create detailed requirements and specifications  
**Phase 3 - Section Co-creation**: Develop comprehensive PRD sections collaboratively
**Phase 4 - Final Synthesis**: Synthesize all elements into final validated PRD

Core Principles:
- Every PRD must be backed by solid business justification
- Technical feasibility must be realistically assessed
- User needs and stakeholder requirements are paramount
- All content must achieve >95% GraphRAG validation confidence
- Requirements must be measurable and testable

PRD Structure Standards:
1. Executive Summary (business context, objectives)
2. Product Overview (vision, scope, success metrics)
3. User Stories & Requirements (detailed functionality)
4. Technical Specifications (architecture, constraints)
5. Implementation Timeline (phases, milestones)
6. Risk Assessment (technical, business, operational)
7. Success Metrics (KPIs, measurement methods)

Quality Standards:
- Requirements must be SMART (Specific, Measurable, Achievable, Relevant, Time-bound)
- Technical specifications must be implementable
- Business cases must be data-driven and realistic
- User experience must be prioritized throughout

Always respond with structured data that can be validated through GraphRAG.""",
            deps_type=Dict[str, Any]
        )
    
    async def execute(self, operation: str, context: Dict[str, Any]) -> AgentResult:
        """Execute a PRD generation operation."""
        start_time = self._log_operation_start(operation, context)
        
        try:
            if operation == "analyze_initial_requirements":
                result = await self._analyze_initial_requirements(context)
            elif operation == "generate_phase0":
                result = await self._generate_phase0(context)
            elif operation == "generate_phase1":
                result = await self._generate_phase1(context)
            elif operation == "generate_phase2":
                result = await self._generate_phase2(context)
            elif operation == "generate_phase3":
                result = await self._generate_phase3(context)
            elif operation == "generate_phase4":
                result = await self._generate_phase4(context)
            elif operation == "generate_full_prd":
                result = await self._generate_full_prd(context)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            processing_time_ms = self._log_operation_complete(operation, start_time, True)
            
            # Validate result with GraphRAG if it contains generated content
            validation_results = []
            if result.get("generated_content"):
                validation = await self.validate_with_graphrag(
                    content=str(result["generated_content"]),
                    context={"section_type": operation}
                )
                validation_results.append(validation)
            
            return self._create_success_result(
                result=result,
                validation_results=validation_results,
                processing_time_ms=processing_time_ms,
                confidence_score=validation_results[0].get("confidence", 0.9) if validation_results else None
            )
            
        except Exception as e:
            processing_time_ms = self._log_operation_complete(operation, start_time, False, str(e))
            return self._create_error_result(
                error=str(e),
                metadata={"processing_time_ms": processing_time_ms}
            )
    
    async def _analyze_initial_requirements(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze initial requirements and extract key concepts."""
        
        initial_description = self._extract_context_parameter(context, "initial_description")
        user_context = self._extract_context_parameter(context, "user_context", required=False, default={})
        
        analysis_prompt = f"""Analyze the following product concept and provide a comprehensive requirements analysis:

Initial Description: {initial_description}
User Context: {json.dumps(user_context, indent=2)}

Provide a detailed analysis including:

1. **Extracted Concepts**: Key technical and business concepts identified
2. **Stakeholder Types**: Primary and secondary stakeholders who would be involved
3. **Complexity Assessment**: Technical complexity level (low/medium/high) with reasoning
4. **Technical Feasibility**: Score from 0.0 to 1.0 with detailed assessment
5. **Business Impact**: Expected business value and impact assessment
6. **Recommended Approach**: Suggested development methodology and approach

Focus on identifying potential challenges, opportunities, and critical success factors.
Be realistic about technical constraints and business viability."""

        result = await self.pydantic_agent.run(
            analysis_prompt,
            deps={
                "initial_description": initial_description,
                "user_context": user_context,
                "hybrid_rag": self.hybrid_rag
            }
        )
        
        return {
            "analysis_result": result.data,
            "initial_description": initial_description,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "generated_content": result.data
        }
    
    async def _generate_phase0(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Phase 0: Project invitation with clarification questions."""
        
        analysis_result = self._extract_context_parameter(context, "analysis_result", required=False)
        initial_description = self._extract_context_parameter(context, "initial_description")
        
        phase0_prompt = f"""Generate Phase 0 (Project Invitation) content for a PRD based on:

Initial Description: {initial_description}
Analysis Result: {json.dumps(analysis_result, indent=2) if analysis_result else "Not provided"}

Create comprehensive Phase 0 content including:

1. **Project Summary**: Clear, concise overview of the proposed project
2. **Initial Scope**: High-level scope definition and boundaries  
3. **Clarification Questions**: 8-12 strategic questions to gather essential information:
   - Business Context & Objectives (2-3 questions)
   - Target Users & Use Cases (2-3 questions)  
   - Technical Requirements & Constraints (2-3 questions)
   - Success Metrics & Timeline (2-3 questions)

Each clarification question should:
- Have a unique ID (q001, q002, etc.)
- Include clear context and rationale
- Be categorized appropriately
- Include priority level (high/medium/low)
- Provide guidance for stakeholders

4. **Similar Project References**: Identify 2-3 similar projects or products for context
5. **Key Concepts**: List of important concepts that stakeholders should understand

Format the response as structured JSON that can be easily processed."""

        result = await self.pydantic_agent.run(
            phase0_prompt,
            deps={
                "initial_description": initial_description,
                "analysis_result": analysis_result,
                "hybrid_rag": self.hybrid_rag
            }
        )
        
        return {
            "phase0_content": result.data,
            "phase": "phase_0",
            "generation_timestamp": datetime.utcnow().isoformat(),
            "generated_content": result.data
        }
    
    async def _generate_phase1(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Phase 1: Objective clarification and validation."""
        
        phase0_content = self._extract_context_parameter(context, "phase0_content")
        stakeholder_answers = self._extract_context_parameter(context, "stakeholder_answers")
        
        phase1_prompt = f"""Process Phase 1 (Objective Clarification) based on stakeholder responses:

Phase 0 Content: {json.dumps(phase0_content, indent=2)}
Stakeholder Answers: {json.dumps(stakeholder_answers, indent=2)}

Analyze the stakeholder responses and generate:

1. **Validated Objectives**: Clear, validated project objectives based on answers
2. **Requirements Synthesis**: Synthesized requirements from stakeholder input
3. **Scope Refinement**: Refined project scope with clear inclusions/exclusions
4. **User Personas**: Detailed user personas based on target audience answers
5. **Success Criteria**: Specific, measurable success criteria and KPIs
6. **Risk Assessment**: Initial risks identified from stakeholder responses
7. **Technical Approach**: Preliminary technical approach recommendations
8. **Answer Validation**: For each answer provided:
   - Confidence score (0.0 to 1.0)
   - Completeness assessment
   - Follow-up questions if needed
   - Recommendations for improvement

9. **Readiness Assessment**: Determination if ready to proceed to Phase 2

Ensure all content is specific, actionable, and aligned with business objectives."""

        result = await self.pydantic_agent.run(
            phase1_prompt,
            deps={
                "phase0_content": phase0_content,
                "stakeholder_answers": stakeholder_answers,
                "hybrid_rag": self.hybrid_rag
            }
        )
        
        return {
            "phase1_content": result.data,
            "phase": "phase_1",
            "generation_timestamp": datetime.utcnow().isoformat(),
            "generated_content": result.data
        }
    
    async def _generate_phase2(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Phase 2: Detailed requirement drafting."""
        
        phase1_content = self._extract_context_parameter(context, "phase1_content")
        additional_context = self._extract_context_parameter(context, "additional_context", required=False, default={})
        
        phase2_prompt = f"""Generate Phase 2 (Requirement Drafting) content:

Phase 1 Content: {json.dumps(phase1_content, indent=2)}
Additional Context: {json.dumps(additional_context, indent=2)}

Create comprehensive requirement specifications:

1. **Functional Requirements**: 
   - Core features and capabilities
   - User workflows and use cases
   - System behaviors and interactions
   - Data requirements and processing

2. **Non-Functional Requirements**:
   - Performance requirements (response times, throughput)
   - Scalability requirements (user load, data volume)
   - Security requirements (authentication, authorization, data protection)
   - Reliability requirements (uptime, error handling, recovery)

3. **Technical Specifications**:
   - Architecture recommendations
   - Technology stack suggestions
   - Integration requirements
   - Data models and schemas

4. **User Experience Requirements**:
   - UI/UX principles and guidelines
   - Accessibility requirements
   - Mobile/responsive design needs
   - User interaction patterns

5. **Acceptance Criteria**:
   - Testable acceptance criteria for each requirement
   - Definition of done for features
   - Quality gates and validation methods

All requirements should be SMART (Specific, Measurable, Achievable, Relevant, Time-bound)."""

        result = await self.pydantic_agent.run(
            phase2_prompt,
            deps={
                "phase1_content": phase1_content,
                "additional_context": additional_context,
                "hybrid_rag": self.hybrid_rag
            }
        )
        
        return {
            "phase2_content": result.data,
            "phase": "phase_2",
            "generation_timestamp": datetime.utcnow().isoformat(),
            "generated_content": result.data
        }
    
    async def _generate_phase3(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Phase 3: Section co-creation and enhancement."""
        
        phase2_content = self._extract_context_parameter(context, "phase2_content")
        collaboration_input = self._extract_context_parameter(context, "collaboration_input", required=False, default={})
        
        phase3_prompt = f"""Generate Phase 3 (Section Co-creation) enhanced content:

Phase 2 Content: {json.dumps(phase2_content, indent=2)}
Collaboration Input: {json.dumps(collaboration_input, indent=2)}

Enhance and expand the PRD with detailed sections:

1. **Executive Summary**:
   - Business context and opportunity
   - Solution overview
   - Expected outcomes and ROI

2. **Product Strategy**:
   - Product vision and positioning
   - Competitive analysis
   - Go-to-market strategy

3. **Implementation Roadmap**:
   - Development phases and milestones
   - Resource requirements
   - Timeline and dependencies
   - Risk mitigation strategies

4. **Operations and Maintenance**:
   - Deployment strategy
   - Monitoring and analytics
   - Support and maintenance plans
   - Performance optimization

5. **Stakeholder Communication**:
   - Communication plan
   - Change management
   - Training requirements
   - Success measurement

6. **Appendices**:
   - Technical diagrams and mockups
   - Research findings and references
   - Glossary of terms
   - Additional resources

Ensure content is professional, comprehensive, and actionable."""

        result = await self.pydantic_agent.run(
            phase3_prompt,
            deps={
                "phase2_content": phase2_content,
                "collaboration_input": collaboration_input,
                "hybrid_rag": self.hybrid_rag
            }
        )
        
        return {
            "phase3_content": result.data,
            "phase": "phase_3", 
            "generation_timestamp": datetime.utcnow().isoformat(),
            "generated_content": result.data
        }
    
    async def _generate_phase4(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Phase 4: Final synthesis and validation."""
        
        phase3_content = self._extract_context_parameter(context, "phase3_content")
        validation_feedback = self._extract_context_parameter(context, "validation_feedback", required=False, default={})
        
        phase4_prompt = f"""Generate Phase 4 (Final Synthesis) - complete and validated PRD:

Phase 3 Content: {json.dumps(phase3_content, indent=2)}
Validation Feedback: {json.dumps(validation_feedback, indent=2)}

Create the final, polished PRD document with:

1. **Document Quality Assurance**:
   - Consistent formatting and structure
   - Clear section organization and navigation
   - Professional presentation quality
   - Complete cross-references and links

2. **Content Validation**:
   - Verify all requirements are complete and testable
   - Ensure technical feasibility of all specifications
   - Validate business alignment and ROI calculations
   - Confirm stakeholder needs are fully addressed

3. **Final Integration**:
   - Synthesize all previous phases into cohesive document
   - Resolve any inconsistencies or conflicts
   - Add executive summary and conclusions
   - Include implementation recommendations

4. **Quality Metrics**:
   - Document completeness score
   - Technical feasibility assessment
   - Business value score
   - Stakeholder alignment rating

5. **Next Steps**:
   - Recommended immediate actions
   - Approval and sign-off process
   - Implementation kickoff plan
   - Success measurement framework

The final PRD should be ready for stakeholder approval and project initiation."""

        result = await self.pydantic_agent.run(
            phase4_prompt,
            deps={
                "phase3_content": phase3_content,
                "validation_feedback": validation_feedback,
                "hybrid_rag": self.hybrid_rag
            }
        )
        
        return {
            "phase4_content": result.data,
            "phase": "phase_4",
            "generation_timestamp": datetime.utcnow().isoformat(),
            "generated_content": result.data,
            "final_prd": True
        }
    
    async def _generate_full_prd(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a complete PRD in a single operation (for simpler cases)."""
        
        requirements = self._extract_context_parameter(context, "requirements")
        complexity_level = self._extract_context_parameter(context, "complexity_level", required=False, default="medium")
        
        full_prd_prompt = f"""Generate a complete, comprehensive PRD document:

Requirements: {json.dumps(requirements, indent=2)}
Complexity Level: {complexity_level}

Create a full PRD including all standard sections:

1. **Executive Summary**
2. **Product Overview** 
3. **Business Context**
4. **User Requirements & Stories**
5. **Technical Specifications**
6. **Implementation Plan**
7. **Risk Assessment**
8. **Success Metrics**
9. **Appendices**

The PRD should be production-ready and suitable for immediate use by development teams."""

        result = await self.pydantic_agent.run(
            full_prd_prompt,
            deps={
                "requirements": requirements,
                "complexity_level": complexity_level,
                "hybrid_rag": self.hybrid_rag
            }
        )
        
        return {
            "full_prd_content": result.data,
            "complexity_level": complexity_level,
            "generation_timestamp": datetime.utcnow().isoformat(),
            "generated_content": result.data
        }