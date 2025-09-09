"""
Judge Agent - Content validation and quality assessment agent.

This agent evaluates content quality, validates against standards,
and provides scoring and improvement recommendations.
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


class QualityAssessment(BaseModel):
    """Quality assessment result."""
    overall_score: float = Field(..., ge=0.0, le=10.0, description="Overall quality score")
    dimension_scores: Dict[str, float] = Field(..., description="Scores by quality dimension")
    strengths: List[str] = Field(..., description="Content strengths identified")
    weaknesses: List[str] = Field(..., description="Areas needing improvement")
    recommendations: List[str] = Field(..., description="Specific improvement recommendations")
    passes_threshold: bool = Field(..., description="Whether content passes quality threshold")


class ValidationCriteria(BaseModel):
    """Validation criteria definition."""
    criterion_name: str = Field(..., description="Name of the criterion")
    weight: float = Field(..., ge=0.0, le=1.0, description="Weight of this criterion")
    minimum_score: float = Field(..., ge=0.0, le=10.0, description="Minimum required score")
    description: str = Field(..., description="Detailed criterion description")


class JudgeAgent(BaseAgent):
    """
    Judge Agent specializing in content quality assessment and validation.
    
    Capabilities:
    - Content quality scoring across multiple dimensions
    - PRD phase validation with specific criteria
    - GraphRAG validation result interpretation
    - Comparative content analysis
    - Improvement recommendation generation
    - Standards compliance checking
    """
    
    def _initialize_agent(self) -> None:
        """Initialize the PydanticAI agent for content judgment."""
        self.pydantic_agent = PydanticAIAgent(
            model_name='openai:gpt-4o',
            system_prompt="""You are the Judge Agent for an AI-powered strategic planning platform.

You specialize in evaluating content quality, validating documents against standards, and providing actionable improvement recommendations.

**Core Evaluation Dimensions:**

1. **Content Quality** (0-10 scale):
   - Clarity and coherence of communication
   - Logical structure and organization
   - Completeness and comprehensiveness
   - Professional writing quality

2. **Technical Accuracy** (0-10 scale):
   - Technical feasibility and realism
   - Accuracy of specifications and requirements
   - Alignment with industry best practices
   - Implementation viability

3. **Business Alignment** (0-10 scale):
   - Alignment with business objectives
   - Market relevance and competitive positioning
   - ROI and value proposition clarity
   - Stakeholder needs satisfaction

4. **Requirements Quality** (0-10 scale):
   - SMART criteria compliance (Specific, Measurable, Achievable, Relevant, Time-bound)
   - Testability and verifiability
   - Completeness of acceptance criteria
   - Clear definition of success

5. **Risk Management** (0-10 scale):
   - Identification of potential risks
   - Mitigation strategies adequacy
   - Contingency planning completeness
   - Risk-benefit analysis quality

**Quality Thresholds:**
- Minimum acceptable score: 7.0/10 overall
- Critical dimensions (Technical Accuracy, Requirements Quality): minimum 8.0/10
- GraphRAG validation confidence: minimum 95%

**Judgment Principles:**
- Be objective and evidence-based in all assessments
- Provide specific, actionable feedback
- Consider context and intended audience
- Balance thoroughness with practical constraints
- Maintain consistent evaluation standards

Always provide detailed reasoning for scores and specific recommendations for improvement.""",
            deps_type=Dict[str, Any]
        )
    
    async def execute(self, operation: str, context: Dict[str, Any]) -> AgentResult:
        """Execute a content judgment operation."""
        start_time = self._log_operation_start(operation, context)
        
        try:
            if operation == "validate_prd_phase":
                result = await self._validate_prd_phase(context)
            elif operation == "comprehensive_quality_review":
                result = await self._comprehensive_quality_review(context)
            elif operation == "compare_content_versions":
                result = await self._compare_content_versions(context)
            elif operation == "validate_requirements":
                result = await self._validate_requirements(context)
            elif operation == "assess_graphrag_results":
                result = await self._assess_graphrag_results(context)
            elif operation == "standards_compliance_check":
                result = await self._standards_compliance_check(context)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            processing_time_ms = self._log_operation_complete(operation, start_time, True)
            
            return self._create_success_result(
                result=result,
                processing_time_ms=processing_time_ms,
                confidence_score=result.get("quality_assessment", {}).get("overall_score", 0.0) / 10.0
            )
            
        except Exception as e:
            processing_time_ms = self._log_operation_complete(operation, start_time, False, str(e))
            return self._create_error_result(
                error=str(e),
                metadata={"processing_time_ms": processing_time_ms}
            )
    
    async def _validate_prd_phase(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a specific PRD phase against criteria."""
        
        phase = self._extract_context_parameter(context, "phase")
        content = self._extract_context_parameter(context, "content")
        custom_criteria = self._extract_context_parameter(context, "criteria", required=False)
        
        # Define phase-specific validation criteria
        phase_criteria = self._get_phase_validation_criteria(phase, custom_criteria)
        
        validation_prompt = f"""Validate PRD {phase} content against specific criteria:

Phase: {phase}
Content to Validate: {json.dumps(content, indent=2)}
Validation Criteria: {json.dumps(phase_criteria, indent=2)}

Perform detailed validation and provide:

1. **Overall Assessment**:
   - Overall quality score (0-10)
   - Pass/fail determination
   - Confidence level in assessment

2. **Criteria-by-Criteria Evaluation**:
   For each validation criterion:
   - Score (0-10)
   - Detailed evaluation reasoning
   - Specific examples from content
   - Compliance status (pass/fail)

3. **Content Analysis**:
   - Strengths identified
   - Critical weaknesses or gaps
   - Missing required elements
   - Quality of existing content

4. **Improvement Recommendations**:
   - Specific actionable recommendations
   - Priority level for each recommendation
   - Expected impact of improvements
   - Implementation guidance

5. **Next Steps**:
   - Readiness for next phase (if applicable)
   - Required revisions before proceeding
   - Approval recommendations

Be thorough and provide specific examples from the content to support your assessment."""

        result = await self.pydantic_agent.run(
            validation_prompt,
            deps={
                "phase": phase,
                "content": content,
                "criteria": phase_criteria,
                "hybrid_rag": self.hybrid_rag
            }
        )
        
        # Perform GraphRAG validation on the content
        graphrag_validation = await self.validate_with_graphrag(
            content=str(content),
            context={"section_type": f"prd_{phase}"}
        )
        
        return {
            "phase_validation": result.data,
            "phase": phase,
            "validation_criteria": phase_criteria,
            "graphrag_validation": graphrag_validation,
            "validation_timestamp": datetime.utcnow().isoformat(),
            "passes_validation": self._determine_validation_pass(result.data, graphrag_validation)
        }
    
    def _get_phase_validation_criteria(self, phase: str, custom_criteria: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get validation criteria for a specific PRD phase."""
        
        base_criteria = {
            "phase_0": [
                {
                    "name": "Clarity of Project Description",
                    "weight": 0.25,
                    "minimum_score": 8.0,
                    "description": "Project concept is clearly articulated and understandable"
                },
                {
                    "name": "Quality of Clarification Questions",
                    "weight": 0.35,
                    "minimum_score": 8.0,
                    "description": "Questions are strategic, comprehensive, and will elicit useful information"
                },
                {
                    "name": "Stakeholder Identification",
                    "weight": 0.20,
                    "minimum_score": 7.0,
                    "description": "Key stakeholders and user types are properly identified"
                },
                {
                    "name": "Scope Definition",
                    "weight": 0.20,
                    "minimum_score": 7.0,
                    "description": "Initial scope is well-defined with clear boundaries"
                }
            ],
            "phase_1": [
                {
                    "name": "Answer Integration Quality",
                    "weight": 0.30,
                    "minimum_score": 8.0,
                    "description": "Stakeholder answers are properly integrated and synthesized"
                },
                {
                    "name": "Objective Clarity",
                    "weight": 0.25,
                    "minimum_score": 8.0,
                    "description": "Project objectives are clear, measurable, and achievable"
                },
                {
                    "name": "Requirements Synthesis",
                    "weight": 0.25,
                    "minimum_score": 7.0,
                    "description": "Initial requirements are well-synthesized from stakeholder input"
                },
                {
                    "name": "Risk Identification", 
                    "weight": 0.20,
                    "minimum_score": 7.0,
                    "description": "Key risks are identified and assessed"
                }
            ],
            "phase_2": [
                {
                    "name": "Functional Requirements Quality",
                    "weight": 0.30,
                    "minimum_score": 8.0,
                    "description": "Functional requirements are specific, complete, and testable"
                },
                {
                    "name": "Technical Specifications",
                    "weight": 0.25,
                    "minimum_score": 8.0,
                    "description": "Technical specs are detailed, feasible, and implementable"
                },
                {
                    "name": "Non-Functional Requirements",
                    "weight": 0.25,
                    "minimum_score": 7.0,
                    "description": "Performance, security, and scalability requirements are defined"
                },
                {
                    "name": "Acceptance Criteria",
                    "weight": 0.20,
                    "minimum_score": 8.0,
                    "description": "Clear, testable acceptance criteria for all requirements"
                }
            ]
        }
        
        criteria = base_criteria.get(phase, [])
        
        # Merge with custom criteria if provided
        if custom_criteria:
            criteria.extend(custom_criteria.get("additional_criteria", []))
        
        return criteria
    
    async def _comprehensive_quality_review(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive quality review of complete content."""
        
        content = self._extract_context_parameter(context, "content")
        content_type = self._extract_context_parameter(context, "content_type", required=False, default="prd")
        review_scope = self._extract_context_parameter(context, "review_scope", required=False, default="complete")
        
        review_prompt = f"""Perform comprehensive quality review of the following {content_type}:

Content: {json.dumps(content, indent=2)}
Content Type: {content_type}
Review Scope: {review_scope}

Conduct thorough quality assessment across all dimensions:

1. **Content Quality Analysis** (0-10 each):
   - Clarity and coherence
   - Structure and organization  
   - Completeness and depth
   - Professional presentation

2. **Technical Assessment** (0-10 each):
   - Technical accuracy and feasibility
   - Implementation viability
   - Best practices compliance
   - Innovation and efficiency

3. **Business Value Assessment** (0-10 each):
   - Business alignment
   - Market relevance
   - ROI and value proposition
   - Stakeholder value delivery

4. **Requirements Quality** (0-10 each):
   - SMART criteria compliance
   - Testability and measurability  
   - Completeness of specifications
   - Clear success definition

5. **Risk and Compliance** (0-10 each):
   - Risk identification and mitigation
   - Compliance with standards
   - Security and privacy considerations
   - Operational readiness

6. **Overall Assessment**:
   - Weighted overall score
   - Critical strengths
   - Major weaknesses
   - Priority improvement areas
   - Readiness for implementation

Provide specific evidence and examples for all assessments."""

        result = await self.pydantic_agent.run(
            review_prompt,
            deps={
                "content": content,
                "content_type": content_type,
                "review_scope": review_scope,
                "hybrid_rag": self.hybrid_rag
            }
        )
        
        # Perform GraphRAG validation
        graphrag_validation = await self.validate_with_graphrag(
            content=str(content),
            context={"section_type": f"comprehensive_{content_type}"}
        )
        
        return {
            "comprehensive_review": result.data,
            "content_type": content_type,
            "review_scope": review_scope,
            "graphrag_validation": graphrag_validation,
            "review_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _compare_content_versions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compare different versions of content for quality improvements."""
        
        version_a = self._extract_context_parameter(context, "version_a")
        version_b = self._extract_context_parameter(context, "version_b")
        comparison_criteria = self._extract_context_parameter(context, "criteria", required=False)
        
        comparison_prompt = f"""Compare two versions of content and assess improvements:

Version A: {json.dumps(version_a, indent=2)}
Version B: {json.dumps(version_b, indent=2)}
Comparison Criteria: {json.dumps(comparison_criteria, indent=2) if comparison_criteria else "Standard quality dimensions"}

Provide detailed comparison analysis:

1. **Side-by-Side Quality Scores**:
   - Overall quality scores for each version
   - Dimension-by-dimension comparison
   - Improvement/regression analysis

2. **Content Comparison**:
   - Added content in Version B
   - Removed content from Version A
   - Modified sections analysis
   - Quality impact of changes

3. **Improvement Assessment**:
   - Specific improvements identified
   - Areas where quality declined
   - Net quality change analysis
   - Recommendation on preferred version

4. **Detailed Analysis**:
   - Strengths of each version
   - Unique value in each version
   - Integration recommendations
   - Further improvement suggestions

Focus on objective, evidence-based comparison with specific examples."""

        result = await self.pydantic_agent.run(
            comparison_prompt,
            deps={
                "version_a": version_a,
                "version_b": version_b,
                "criteria": comparison_criteria,
                "hybrid_rag": self.hybrid_rag
            }
        )
        
        return {
            "version_comparison": result.data,
            "comparison_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _validate_requirements(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate requirements against SMART criteria and best practices."""
        
        requirements = self._extract_context_parameter(context, "requirements")
        validation_standards = self._extract_context_parameter(context, "standards", required=False)
        
        requirements_prompt = f"""Validate requirements against SMART criteria and industry best practices:

Requirements: {json.dumps(requirements, indent=2)}
Validation Standards: {json.dumps(validation_standards, indent=2) if validation_standards else "Industry standard SMART criteria"}

Perform detailed requirements validation:

1. **SMART Criteria Assessment** (for each requirement):
   - Specific: Is the requirement clearly defined?
   - Measurable: Can success be measured?
   - Achievable: Is it technically and practically feasible?
   - Relevant: Does it align with business objectives?
   - Time-bound: Are there clear deadlines/timelines?

2. **Quality Assessment** (for each requirement):
   - Clarity and unambiguity
   - Testability and verifiability  
   - Completeness of specification
   - Consistency with other requirements
   - Traceability to business needs

3. **Best Practices Compliance**:
   - Industry standards alignment
   - Architecture best practices
   - Security and privacy considerations
   - Performance and scalability factors

4. **Overall Assessment**:
   - Requirements quality score
   - Critical gaps or issues
   - Implementation risk assessment
   - Recommendations for improvement

Provide specific feedback for each requirement with improvement suggestions."""

        result = await self.pydantic_agent.run(
            requirements_prompt,
            deps={
                "requirements": requirements,
                "standards": validation_standards,
                "hybrid_rag": self.hybrid_rag
            }
        )
        
        return {
            "requirements_validation": result.data,
            "validation_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _assess_graphrag_results(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess and interpret GraphRAG validation results."""
        
        graphrag_results = self._extract_context_parameter(context, "graphrag_results")
        content = self._extract_context_parameter(context, "content")
        
        assessment_prompt = f"""Assess GraphRAG validation results and provide interpretation:

GraphRAG Results: {json.dumps(graphrag_results, indent=2)}
Original Content: {json.dumps(content, indent=2)}

Provide detailed GraphRAG assessment:

1. **Confidence Analysis**:
   - Overall confidence interpretation
   - Entity validation assessment
   - Community validation assessment  
   - Global validation assessment
   - Weighted score analysis

2. **Validation Quality**:
   - Reliability of validation results
   - Potential false positives/negatives
   - Areas of uncertainty
   - Validation coverage assessment

3. **Content Quality Implications**:
   - What the scores indicate about content quality
   - Specific areas flagged by validation
   - Recommendations based on validation
   - Risk assessment for proceeding

4. **Improvement Guidance**:
   - Specific steps to improve validation scores
   - Content modifications recommended
   - Additional validation needed
   - Quality assurance recommendations

Focus on practical interpretation and actionable recommendations."""

        result = await self.pydantic_agent.run(
            assessment_prompt,
            deps={
                "graphrag_results": graphrag_results,
                "content": content,
                "hybrid_rag": self.hybrid_rag
            }
        )
        
        return {
            "graphrag_assessment": result.data,
            "assessment_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _standards_compliance_check(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check content compliance against specific standards."""
        
        content = self._extract_context_parameter(context, "content")
        standards = self._extract_context_parameter(context, "standards")
        compliance_level = self._extract_context_parameter(context, "compliance_level", required=False, default="standard")
        
        compliance_prompt = f"""Check content compliance against specified standards:

Content: {json.dumps(content, indent=2)}
Standards: {json.dumps(standards, indent=2)}
Compliance Level: {compliance_level}

Perform thorough compliance assessment:

1. **Standards Compliance** (for each standard):
   - Compliance status (compliant/non-compliant/partial)
   - Specific violations or gaps identified
   - Evidence supporting assessment
   - Severity of non-compliance

2. **Gap Analysis**:
   - Missing elements required by standards
   - Areas of partial compliance
   - Critical vs. non-critical gaps
   - Effort required to achieve compliance

3. **Risk Assessment**:
   - Risk of proceeding with current compliance level
   - Potential impact of non-compliance
   - Mitigation strategies available
   - Timeline implications

4. **Remediation Plan**:
   - Specific actions to achieve compliance
   - Priority order for addressing gaps
   - Resource requirements
   - Timeline for compliance achievement

Provide specific, actionable compliance guidance."""

        result = await self.pydantic_agent.run(
            compliance_prompt,
            deps={
                "content": content,
                "standards": standards,
                "compliance_level": compliance_level,
                "hybrid_rag": self.hybrid_rag
            }
        )
        
        return {
            "compliance_assessment": result.data,
            "standards": standards,
            "compliance_level": compliance_level,
            "assessment_timestamp": datetime.utcnow().isoformat()
        }
    
    def _determine_validation_pass(self, validation_result: Any, graphrag_result: Dict[str, Any]) -> bool:
        """Determine if content passes validation based on multiple factors."""
        
        try:
            # Extract scores from validation result (assuming it's structured)
            if isinstance(validation_result, dict):
                overall_score = validation_result.get("overall_score", 0.0)
            else:
                # If it's a string, assume moderate quality
                overall_score = 7.0
            
            # Check GraphRAG confidence
            graphrag_confidence = graphrag_result.get("confidence", 0.0)
            graphrag_passes = graphrag_result.get("passes_threshold", False)
            
            # Determine pass/fail
            quality_pass = overall_score >= 7.0
            validation_pass = quality_pass and graphrag_passes and graphrag_confidence >= 0.95
            
            return validation_pass
            
        except Exception as e:
            logger.warning(f"Error determining validation pass: {str(e)}")
            return False