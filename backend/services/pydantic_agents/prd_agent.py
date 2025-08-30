"""
PydanticAI agent for PRD creation and processing
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import KnownModelName

from core.config import get_settings
from services.graphrag.graph_service import GraphRAGService

logger = structlog.get_logger(__name__)
settings = get_settings()


# Pydantic Models for PRD Processing
class PRDInput(BaseModel):
    """Input model for PRD creation request."""
    title: str = Field(..., min_length=10, max_length=200, description="Clear, actionable PRD title")
    description: str = Field(..., min_length=100, description="Detailed feature description")
    business_context: Optional[str] = Field(None, description="Business justification and impact")
    target_audience: Optional[str] = Field(None, description="Primary users and stakeholders") 
    success_criteria: Optional[List[str]] = Field(None, description="Measurable success metrics")
    constraints: Optional[List[str]] = Field(None, description="Technical or business limitations")
    priority: str = Field(default="medium", description="Priority level: low, medium, high, critical")


class PRDSection(BaseModel):
    """Model for individual PRD section."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")
    validation_score: float = Field(..., ge=0.0, le=1.0, description="GraphRAG validation score")
    status: str = Field(default="draft", description="Section status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class PRDResult(BaseModel):
    """Complete PRD generation result."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(..., description="PRD title")
    executive_summary: str = Field(..., description="Executive summary")
    sections: List[PRDSection] = Field(..., description="PRD sections")
    overall_quality_score: float = Field(..., ge=0.0, le=10.0, description="Overall quality score")
    validation_results: Dict[str, Any] = Field(..., description="GraphRAG validation results")
    github_integration: Optional[Dict[str, Any]] = Field(None, description="GitHub project info")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field(default="generated", description="PRD status")


class AgentContext(BaseModel):
    """Context for agent operations."""
    user_id: str = Field(..., description="User ID for request")
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    graphrag_service: Optional[GraphRAGService] = Field(None, description="GraphRAG service instance")
    project_context: Dict[str, Any] = Field(default_factory=dict, description="Project context")


# PydanticAI Agent Definition
prd_creation_agent = Agent(
    model=settings.default_model,
    result_type=PRDResult,
    system_prompt="""You are an expert Product Requirements Document (PRD) creator specializing in generating 
    comprehensive, actionable documents through AI-human collaboration.

    Your core responsibilities:
    1. Transform high-level product concepts into detailed, implementable PRDs
    2. Apply GraphRAG validation to ensure accuracy and consistency 
    3. Generate structured sections with clear acceptance criteria
    4. Maintain enterprise-quality standards with <2% hallucination rate
    5. Create actionable deliverables ready for development teams

    Key principles:
    - Always validate content against organizational knowledge using GraphRAG
    - Prioritize clarity and specificity over complexity
    - Include measurable success criteria and acceptance criteria
    - Consider technical feasibility and business constraints
    - Maintain professional, enterprise-appropriate tone

    Output quality standards:
    - Validation confidence score ≥8.0/10 required
    - All technical claims must be verifiable
    - Business impact must be quantifiable where possible
    - Risk assessments should be comprehensive but realistic
    """,
    deps_type=AgentContext
)


class PRDCreationAgent:
    """High-level interface for PRD creation agent."""
    
    def __init__(self):
        self.agent = prd_creation_agent
        self.graphrag_service = None
    
    async def initialize(self, graphrag_service: GraphRAGService) -> None:
        """Initialize agent with GraphRAG service."""
        self.graphrag_service = graphrag_service
        logger.info("PRD Creation Agent initialized")
    
    async def create_prd(
        self,
        prd_input: PRDInput,
        user_id: str,
        context: Dict[str, Any] = None
    ) -> PRDResult:
        """Generate complete PRD from input specifications."""
        if not context:
            context = {}
        
        try:
            # Create agent context
            agent_context = AgentContext(
                user_id=user_id,
                graphrag_service=self.graphrag_service,
                project_context=context
            )
            
            # Run PRD generation
            result = await self.agent.run(
                user_prompt=self._create_user_prompt(prd_input),
                deps=agent_context
            )
            
            logger.info(
                "PRD generated successfully",
                prd_id=result.data.id,
                quality_score=result.data.overall_quality_score,
                sections=len(result.data.sections)
            )
            
            return result.data
            
        except Exception as e:
            logger.error("PRD creation failed", error=str(e), user_id=user_id)
            raise
    
    def _create_user_prompt(self, prd_input: PRDInput) -> str:
        """Create user prompt from PRD input."""
        prompt_parts = [
            f"Create a comprehensive PRD for: {prd_input.title}",
            f"\nDescription: {prd_input.description}"
        ]
        
        if prd_input.business_context:
            prompt_parts.append(f"\nBusiness Context: {prd_input.business_context}")
        
        if prd_input.target_audience:
            prompt_parts.append(f"\nTarget Audience: {prd_input.target_audience}")
        
        if prd_input.success_criteria:
            prompt_parts.append(f"\nSuccess Criteria: {', '.join(prd_input.success_criteria)}")
        
        if prd_input.constraints:
            prompt_parts.append(f"\nConstraints: {', '.join(prd_input.constraints)}")
        
        prompt_parts.append(f"\nPriority: {prd_input.priority}")
        
        prompt_parts.append("""
        
        Generate a PRD with the following sections:
        1. Executive Summary - High-level overview and business impact
        2. Problem Statement - Clear definition of the problem being solved
        3. Solution Overview - Proposed solution approach
        4. Requirements - Detailed functional and non-functional requirements
        5. Success Metrics - Measurable outcomes and KPIs
        6. Implementation Plan - High-level implementation approach
        7. Risks and Mitigation - Identified risks and mitigation strategies
        8. Timeline - Project phases and milestones
        
        Each section must include:
        - Clear, actionable content
        - Specific acceptance criteria where applicable
        - Business value justification
        - Technical feasibility assessment
        """)
        
        return "\n".join(prompt_parts)


# Agent Tools for Enhanced Functionality
@prd_creation_agent.tool
async def validate_content_with_graphrag(
    ctx: RunContext[AgentContext],
    content: str,
    section_type: str
) -> Dict[str, Any]:
    """Validate content against GraphRAG knowledge base."""
    if not ctx.deps.graphrag_service:
        return {"error": "GraphRAG service not available", "confidence": 0.0}
    
    try:
        validation_result = await ctx.deps.graphrag_service.validate_content(
            content, 
            {"section_type": section_type, "user_id": ctx.deps.user_id}
        )
        
        return validation_result.to_dict()
        
    except Exception as e:
        logger.error("GraphRAG validation failed", error=str(e))
        return {"error": str(e), "confidence": 0.0}


@prd_creation_agent.tool
async def research_similar_prds(
    ctx: RunContext[AgentContext],
    query: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """Research similar PRDs from knowledge base."""
    if not ctx.deps.graphrag_service:
        return []
    
    try:
        # Use Neo4j connection to find similar PRDs
        neo4j_conn = await ctx.deps.graphrag_service.neo4j_conn
        
        similar_prds_query = """
        MATCH (p:PRD)
        WHERE toLower(p.title) CONTAINS toLower($query) OR 
              toLower(p.description) CONTAINS toLower($query)
        RETURN p.title as title, p.description as description, 
               p.id as id, p.created_at as created_at
        ORDER BY p.created_at DESC
        LIMIT $limit
        """
        
        results = await neo4j_conn.execute_query(
            similar_prds_query, 
            {"query": query, "limit": limit}
        )
        
        return results
        
    except Exception as e:
        logger.error("Similar PRD research failed", error=str(e))
        return []


@prd_creation_agent.tool
async def calculate_quality_score(
    ctx: RunContext[AgentContext],
    sections: List[Dict[str, Any]]
) -> float:
    """Calculate overall quality score based on validation results."""
    try:
        if not sections:
            return 0.0
        
        # Extract validation scores from sections
        validation_scores = []
        for section in sections:
            if isinstance(section, dict) and "validation_score" in section:
                validation_scores.append(section["validation_score"])
        
        if not validation_scores:
            return 5.0  # Default neutral score
        
        # Calculate weighted average (scale to 10-point system)
        avg_validation = sum(validation_scores) / len(validation_scores)
        quality_score = min(avg_validation * 10, 10.0)
        
        # Apply quality bonuses/penalties
        if quality_score >= 8.0:
            quality_score = min(quality_score + 0.5, 10.0)  # Bonus for high quality
        elif quality_score < 6.0:
            quality_score = max(quality_score - 0.5, 1.0)  # Penalty for low quality
        
        return round(quality_score, 1)
        
    except Exception as e:
        logger.error("Quality score calculation failed", error=str(e))
        return 5.0  # Default score on error


@prd_creation_agent.tool
async def store_prd_in_graph(
    ctx: RunContext[AgentContext],
    prd_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Store generated PRD in Neo4j knowledge graph."""
    if not ctx.deps.graphrag_service:
        return {"error": "GraphRAG service not available"}
    
    try:
        neo4j_conn = ctx.deps.graphrag_service.neo4j_conn
        
        # Store PRD in graph
        store_query = """
        CREATE (p:PRD {
            id: $prd_id,
            title: $title,
            description: $description,
            executive_summary: $executive_summary,
            quality_score: $quality_score,
            status: $status,
            created_by: $user_id,
            created_at: datetime()
        })
        
        WITH p
        UNWIND $sections as section
        CREATE (s:Section {
            id: section.id,
            title: section.title,
            content: section.content,
            validation_score: section.validation_score,
            status: section.status,
            created_at: datetime()
        })
        CREATE (p)-[:CONTAINS]->(s)
        
        RETURN p.id as prd_id, count(s) as sections_created
        """
        
        result = await neo4j_conn.execute_write(
            store_query,
            {
                "prd_id": prd_data["id"],
                "title": prd_data["title"],
                "description": prd_data.get("description", ""),
                "executive_summary": prd_data["executive_summary"],
                "quality_score": prd_data["overall_quality_score"],
                "status": prd_data["status"],
                "user_id": ctx.deps.user_id,
                "sections": [
                    {
                        "id": section["id"],
                        "title": section["title"],
                        "content": section["content"],
                        "validation_score": section["validation_score"],
                        "status": section["status"]
                    }
                    for section in prd_data["sections"]
                ]
            }
        )
        
        return {"stored": True, "result": result}
        
    except Exception as e:
        logger.error("PRD storage failed", error=str(e))
        return {"error": str(e)}