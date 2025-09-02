"""
Documentation Librarian Agent - Knowledge management and documentation specialist.

This agent manages documentation storage, organization, and retrieval across
the platform's knowledge base with GraphRAG integration.
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


class DocumentMetadata(BaseModel):
    """Metadata for stored documents."""
    document_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    document_type: str = Field(..., description="Type of document")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    author: str = Field(..., description="Document author")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    version: str = Field(default="1.0", description="Document version")
    status: str = Field(default="draft", description="Document status")


class KnowledgeEntry(BaseModel):
    """Entry for the knowledge base."""
    entry_id: str = Field(..., description="Unique entry identifier")
    content: str = Field(..., description="Entry content")
    category: str = Field(..., description="Knowledge category")
    subcategory: Optional[str] = Field(None, description="Knowledge subcategory")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Content confidence")
    source_documents: List[str] = Field(default_factory=list, description="Source document IDs")
    validation_status: str = Field(..., description="GraphRAG validation status")


class DocumentationLibrarianAgent(BaseAgent):
    """
    Documentation Librarian Agent specializing in knowledge management.
    
    Capabilities:
    - Document storage and organization
    - Knowledge base management and curation
    - Document versioning and lifecycle management
    - Search and retrieval optimization
    - Content validation and quality assurance
    - Cross-reference and relationship management
    """
    
    def _initialize_agent(self) -> None:
        """Initialize the PydanticAI agent for documentation management."""
        self.pydantic_agent = PydanticAIAgent(
            model_name='openai:gpt-4o',
            system_prompt="""You are the Documentation Librarian Agent for an AI-powered strategic planning platform.

You specialize in managing, organizing, and curating the platform's knowledge base and documentation ecosystem. Your role is to ensure high-quality, accessible, and well-organized information management.

**Core Responsibilities:**

1. **Document Management**:
   - Organize and categorize documents systematically
   - Maintain document metadata and relationships
   - Version control and lifecycle management
   - Quality assurance and validation

2. **Knowledge Base Curation**:
   - Extract and structure knowledge from documents
   - Create cross-references and relationships
   - Maintain knowledge taxonomy and ontology
   - Ensure information accuracy and currency

3. **Search and Retrieval Optimization**:
   - Optimize content for searchability
   - Create effective tagging and categorization
   - Maintain search indexes and facets
   - Enable semantic and contextual search

4. **Content Validation and Quality**:
   - Validate content against standards
   - Ensure GraphRAG compatibility and optimization
   - Maintain quality scores and metrics
   - Identify and resolve content conflicts

5. **Integration and Relationships**:
   - Maintain document relationships and dependencies
   - Create topic clusters and knowledge maps
   - Enable cross-document navigation
   - Support collaborative editing and review

**Quality Standards:**
- All stored content must achieve >95% GraphRAG validation confidence
- Documents must be properly categorized and tagged
- Metadata must be complete and accurate
- Cross-references must be maintained and validated
- Version control must be comprehensive and traceable

**Organization Principles:**
- Hierarchical categorization with flexible tagging
- Semantic relationships and topic clustering
- Time-based and version-based organization
- User role-based access and organization
- Project and context-based grouping

**Response Format:**
Always provide structured responses with:
- Clear categorization and organization
- Complete metadata specification
- Validation and quality indicators
- Relationship and cross-reference information
- Search optimization suggestions

You maintain the highest standards for information organization and accessibility.""",
            deps_type=Dict[str, Any]
        )
    
    async def execute(self, operation: str, context: Dict[str, Any]) -> AgentResult:
        """Execute a documentation management operation."""
        start_time = self._log_operation_start(operation, context)
        
        try:
            if operation == "store_document":
                result = await self._store_document(context)
            elif operation == "organize_knowledge_base":
                result = await self._organize_knowledge_base(context)
            elif operation == "create_document_index":
                result = await self._create_document_index(context)
            elif operation == "extract_knowledge":
                result = await self._extract_knowledge(context)
            elif operation == "manage_document_relationships":
                result = await self._manage_document_relationships(context)
            elif operation == "validate_documentation":
                result = await self._validate_documentation(context)
            elif operation == "store_prd_knowledge":
                result = await self._store_prd_knowledge(context)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            processing_time_ms = self._log_operation_complete(operation, start_time, True)
            
            # Validate stored content with GraphRAG if applicable
            validation_results = []
            if result.get("stored_content"):
                validation = await self.validate_with_graphrag(
                    content=str(result["stored_content"]),
                    context={"section_type": "knowledge_base_entry"}
                )
                validation_results.append(validation)
            
            return self._create_success_result(
                result=result,
                validation_results=validation_results,
                processing_time_ms=processing_time_ms,
                confidence_score=validation_results[0].get("confidence", 0.95) if validation_results else 0.95
            )
            
        except Exception as e:
            processing_time_ms = self._log_operation_complete(operation, start_time, False, str(e))
            return self._create_error_result(
                error=str(e),
                metadata={"processing_time_ms": processing_time_ms}
            )
    
    async def _store_document(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Store a document in the knowledge base with proper organization."""
        
        document_content = self._extract_context_parameter(context, "document_content")
        document_type = self._extract_context_parameter(context, "document_type")
        title = self._extract_context_parameter(context, "title", required=False)
        author = self._extract_context_parameter(context, "author", required=False, default="system")
        project_context = self._extract_context_parameter(context, "project_context", required=False, default={})
        
        storage_prompt = f"""Process and store the following document in the knowledge base:

Document Content: {json.dumps(document_content, indent=2)}
Document Type: {document_type}
Title: {title or 'Auto-generated title needed'}
Author: {author}
Project Context: {json.dumps(project_context, indent=2)}

Perform comprehensive document processing including:

1. **Content Analysis**:
   - Analyze document content and structure
   - Extract key topics and concepts
   - Identify main themes and subjects
   - Assess content quality and completeness

2. **Metadata Generation**:
   - Generate appropriate title if not provided
   - Create comprehensive tag set for searchability
   - Determine appropriate category and subcategory
   - Set quality and confidence scores

3. **Organization Structure**:
   - Determine optimal placement in knowledge hierarchy
   - Identify related documents and cross-references
   - Create topic clusters and associations
   - Establish document relationships

4. **Search Optimization**:
   - Generate search keywords and phrases
   - Create content summaries and abstracts
   - Identify key entities and concepts
   - Optimize for semantic search

5. **Validation and Quality Assurance**:
   - Validate content structure and completeness
   - Check for consistency with existing knowledge
   - Identify potential conflicts or duplicates
   - Assess GraphRAG compatibility

6. **Storage Specification**:
   - Specify storage location and organization
   - Define access permissions and visibility
   - Set up version control and history
   - Configure backup and recovery

Provide detailed storage instructions and metadata for optimal knowledge base integration."""

        result = await self.pydantic_agent.run(
            storage_prompt,
            deps={
                "document_content": document_content,
                "document_type": document_type,
                "title": title,
                "author": author,
                "project_context": project_context,
                "hybrid_rag": self.hybrid_rag
            }
        )
        
        return {
            "storage_result": result.data,
            "document_type": document_type,
            "storage_timestamp": datetime.utcnow().isoformat(),
            "stored_content": document_content
        }
    
    async def _organize_knowledge_base(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Organize and restructure the knowledge base for optimal access."""
        
        current_structure = self._extract_context_parameter(context, "current_structure", required=False, default={})
        optimization_goals = self._extract_context_parameter(context, "optimization_goals", required=False, default=[])
        user_access_patterns = self._extract_context_parameter(context, "user_access_patterns", required=False, default={})
        
        organization_prompt = f"""Analyze and optimize the knowledge base organization:

Current Structure: {json.dumps(current_structure, indent=2)}
Optimization Goals: {json.dumps(optimization_goals, indent=2)}
User Access Patterns: {json.dumps(user_access_patterns, indent=2)}

Create comprehensive organization plan including:

1. **Structural Analysis**:
   - Evaluate current organization effectiveness
   - Identify structural gaps and inefficiencies
   - Assess user navigation patterns and pain points
   - Analyze content distribution and clustering

2. **Taxonomy Development**:
   - Design hierarchical category structure
   - Create flexible tagging system
   - Establish topic clustering approach
   - Define semantic relationships

3. **Navigation Optimization**:
   - Design intuitive navigation paths
   - Create effective search facets and filters
   - Establish cross-reference systems
   - Optimize for different user types and needs

4. **Content Organization**:
   - Group related content effectively
   - Create topic-based collections
   - Establish content lifecycle management
   - Design version and history organization

5. **Search and Discovery**:
   - Optimize search indexing and ranking
   - Create recommendation systems
   - Design contextual content suggestions
   - Enable semantic and conceptual search

6. **Access and Permissions**:
   - Design role-based access control
   - Create collaborative editing workflows
   - Establish review and approval processes
   - Manage sensitive content protection

Provide specific implementation recommendations for optimal knowledge organization."""

        result = await self.pydantic_agent.run(
            organization_prompt,
            deps={
                "current_structure": current_structure,
                "optimization_goals": optimization_goals,
                "user_access_patterns": user_access_patterns,
                "hybrid_rag": self.hybrid_rag
            }
        )
        
        return {
            "organization_plan": result.data,
            "optimization_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _create_document_index(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive document index for efficient retrieval."""
        
        document_collection = self._extract_context_parameter(context, "document_collection")
        indexing_strategy = self._extract_context_parameter(context, "indexing_strategy", required=False, default="comprehensive")
        search_requirements = self._extract_context_parameter(context, "search_requirements", required=False, default={})
        
        indexing_prompt = f"""Create comprehensive document index for efficient search and retrieval:

Document Collection: {json.dumps(document_collection, indent=2)}
Indexing Strategy: {indexing_strategy}
Search Requirements: {json.dumps(search_requirements, indent=2)}

Develop complete indexing solution including:

1. **Index Structure Design**:
   - Primary index organization and hierarchy
   - Secondary index structures for specialized searches
   - Cross-reference indexes and relationship maps
   - Temporal and version-based indexes

2. **Content Extraction and Processing**:
   - Extract key terms and concepts from documents
   - Create content abstracts and summaries
   - Identify entities, topics, and themes
   - Process metadata and structural information

3. **Search Optimization**:
   - Create full-text search capabilities
   - Enable semantic and contextual search
   - Design faceted search and filtering
   - Optimize ranking and relevance algorithms

4. **Relationship Mapping**:
   - Map document relationships and dependencies
   - Create topic clusters and concept maps
   - Establish citation and reference networks
   - Enable collaborative filtering and recommendations

5. **Performance Optimization**:
   - Design efficient index storage and retrieval
   - Create caching and pre-computation strategies
   - Optimize for common search patterns
   - Enable incremental index updates

6. **Quality Assurance**:
   - Validate index accuracy and completeness
   - Test search performance and relevance
   - Monitor index freshness and currency
   - Ensure GraphRAG compatibility

Provide detailed implementation specifications for high-performance document indexing."""

        result = await self.pydantic_agent.run(
            indexing_prompt,
            deps={
                "document_collection": document_collection,
                "indexing_strategy": indexing_strategy,
                "search_requirements": search_requirements,
                "hybrid_rag": self.hybrid_rag
            }
        )
        
        return {
            "index_specification": result.data,
            "indexing_strategy": indexing_strategy,
            "index_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _extract_knowledge(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured knowledge from documents for the knowledge base."""
        
        source_documents = self._extract_context_parameter(context, "source_documents")
        knowledge_categories = self._extract_context_parameter(context, "knowledge_categories", required=False, default=[])
        extraction_depth = self._extract_context_parameter(context, "extraction_depth", required=False, default="comprehensive")
        
        extraction_prompt = f"""Extract structured knowledge from the provided documents:

Source Documents: {json.dumps(source_documents, indent=2)}
Knowledge Categories: {json.dumps(knowledge_categories, indent=2)}
Extraction Depth: {extraction_depth}

Perform comprehensive knowledge extraction including:

1. **Concept Identification**:
   - Extract key concepts and definitions
   - Identify domain-specific terminology
   - Capture business rules and principles
   - Extract process flows and procedures

2. **Relationship Mapping**:
   - Map relationships between concepts
   - Identify cause-and-effect relationships
   - Capture hierarchical and taxonomic relationships
   - Extract temporal and sequential relationships

3. **Fact and Data Extraction**:
   - Extract factual information and data points
   - Capture quantitative metrics and benchmarks
   - Identify best practices and recommendations
   - Extract case studies and examples

4. **Knowledge Structuring**:
   - Organize knowledge into coherent structures
   - Create knowledge hierarchies and taxonomies
   - Establish semantic networks and ontologies
   - Design knowledge representation formats

5. **Quality Validation**:
   - Validate extracted knowledge accuracy
   - Check for consistency and completeness
   - Resolve conflicts and ambiguities
   - Ensure GraphRAG compatibility

6. **Knowledge Base Integration**:
   - Prepare knowledge for storage and indexing
   - Create appropriate metadata and tags
   - Establish connections to existing knowledge
   - Enable search and retrieval optimization

Focus on creating high-quality, reusable knowledge assets that enhance the platform's intelligence."""

        result = await self.pydantic_agent.run(
            extraction_prompt,
            deps={
                "source_documents": source_documents,
                "knowledge_categories": knowledge_categories,
                "extraction_depth": extraction_depth,
                "hybrid_rag": self.hybrid_rag
            }
        )
        
        return {
            "extracted_knowledge": result.data,
            "extraction_depth": extraction_depth,
            "extraction_timestamp": datetime.utcnow().isoformat(),
            "stored_content": result.data
        }
    
    async def _manage_document_relationships(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Manage relationships and dependencies between documents."""
        
        document_set = self._extract_context_parameter(context, "document_set")
        relationship_types = self._extract_context_parameter(context, "relationship_types", required=False, default=[])
        analysis_depth = self._extract_context_parameter(context, "analysis_depth", required=False, default="standard")
        
        relationships_prompt = f"""Analyze and manage relationships between documents:

Document Set: {json.dumps(document_set, indent=2)}
Relationship Types: {json.dumps(relationship_types, indent=2)}
Analysis Depth: {analysis_depth}

Perform comprehensive relationship analysis including:

1. **Relationship Discovery**:
   - Identify explicit references and citations
   - Discover implicit topical relationships
   - Map hierarchical and dependency relationships
   - Find temporal and sequential relationships

2. **Relationship Classification**:
   - Categorize relationship types and strengths
   - Assign confidence scores to relationships
   - Identify bidirectional vs unidirectional relationships
   - Classify relationship contexts and purposes

3. **Network Analysis**:
   - Create document relationship networks
   - Identify central and peripheral documents
   - Find clusters and communities of related documents
   - Analyze information flow patterns

4. **Dependency Management**:
   - Map document dependencies and prerequisites
   - Identify circular dependencies and conflicts
   - Create dependency resolution strategies
   - Manage version and update propagation

5. **Navigation Enhancement**:
   - Create intelligent cross-reference systems
   - Design contextual navigation paths
   - Enable relationship-based recommendations
   - Optimize user discovery workflows

6. **Maintenance and Validation**:
   - Validate relationship accuracy and currency
   - Monitor relationship changes over time
   - Maintain relationship integrity during updates
   - Ensure GraphRAG relationship optimization

Provide actionable relationship management strategies for enhanced knowledge connectivity."""

        result = await self.pydantic_agent.run(
            relationships_prompt,
            deps={
                "document_set": document_set,
                "relationship_types": relationship_types,
                "analysis_depth": analysis_depth,
                "hybrid_rag": self.hybrid_rag
            }
        )
        
        return {
            "relationship_analysis": result.data,
            "analysis_depth": analysis_depth,
            "relationship_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _validate_documentation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate documentation quality and standards compliance."""
        
        documentation_set = self._extract_context_parameter(context, "documentation_set")
        validation_standards = self._extract_context_parameter(context, "validation_standards", required=False, default={})
        validation_scope = self._extract_context_parameter(context, "validation_scope", required=False, default="comprehensive")
        
        validation_prompt = f"""Validate documentation quality and standards compliance:

Documentation Set: {json.dumps(documentation_set, indent=2)}
Validation Standards: {json.dumps(validation_standards, indent=2)}
Validation Scope: {validation_scope}

Perform thorough documentation validation including:

1. **Content Quality Assessment**:
   - Evaluate clarity, accuracy, and completeness
   - Check for consistency and coherence
   - Assess technical accuracy and currency
   - Validate examples and code samples

2. **Standards Compliance**:
   - Check against documentation standards
   - Validate formatting and structure
   - Ensure style guide compliance
   - Verify accessibility and usability requirements

3. **GraphRAG Compatibility**:
   - Validate content for GraphRAG optimization
   - Check entity recognition and extraction
   - Ensure relationship mapping capability
   - Validate semantic search optimization

4. **Organizational Requirements**:
   - Verify proper categorization and tagging
   - Check metadata completeness and accuracy
   - Validate cross-references and relationships
   - Ensure appropriate access controls

5. **User Experience Validation**:
   - Assess navigation and discoverability
   - Check search effectiveness and relevance
   - Validate user workflow support
   - Ensure appropriate detail levels for audiences

6. **Maintenance and Currency**:
   - Check content freshness and relevance
   - Validate version control and history
   - Assess update and maintenance requirements
   - Identify obsolete or conflicting content

Provide detailed validation results with specific recommendations for improvement."""

        result = await self.pydantic_agent.run(
            validation_prompt,
            deps={
                "documentation_set": documentation_set,
                "validation_standards": validation_standards,
                "validation_scope": validation_scope,
                "hybrid_rag": self.hybrid_rag
            }
        )
        
        return {
            "validation_results": result.data,
            "validation_scope": validation_scope,
            "validation_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _store_prd_knowledge(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Store PRD-specific knowledge in the knowledge base."""
        
        prd_content = self._extract_context_parameter(context, "prd_content")
        project_context = self._extract_context_parameter(context, "project_context", required=False, default={})
        quality_metrics = self._extract_context_parameter(context, "quality_metrics", required=False, default={})
        
        prd_storage_prompt = f"""Store PRD-specific knowledge in the knowledge base:

PRD Content: {json.dumps(prd_content, indent=2)}
Project Context: {json.dumps(project_context, indent=2)}
Quality Metrics: {json.dumps(quality_metrics, indent=2)}

Process PRD for optimal knowledge storage including:

1. **PRD Knowledge Extraction**:
   - Extract business requirements and objectives
   - Capture user stories and acceptance criteria
   - Identify technical specifications and constraints
   - Extract success metrics and KPIs

2. **Project Context Integration**:
   - Link to project timelines and milestones
   - Connect to team and stakeholder information
   - Associate with related projects and initiatives
   - Map to organizational goals and strategies

3. **Requirements Knowledge Management**:
   - Create searchable requirements database
   - Establish requirement traceability
   - Map requirement dependencies and relationships
   - Enable requirement impact analysis

4. **Best Practices Capture**:
   - Extract proven patterns and approaches
   - Capture lessons learned and insights
   - Identify successful solutions and decisions
   - Document risk mitigation strategies

5. **Reusability Optimization**:
   - Create reusable requirement templates
   - Extract generalizable patterns and frameworks
   - Enable cross-project knowledge sharing
   - Build organizational requirement libraries

6. **Quality and Validation Integration**:
   - Store validation results and quality scores
   - Maintain GraphRAG compatibility
   - Enable quality trend analysis
   - Support continuous improvement processes

Focus on creating valuable, reusable knowledge assets that improve future PRD development."""

        result = await self.pydantic_agent.run(
            prd_storage_prompt,
            deps={
                "prd_content": prd_content,
                "project_context": project_context,
                "quality_metrics": quality_metrics,
                "hybrid_rag": self.hybrid_rag
            }
        )
        
        return {
            "prd_knowledge_storage": result.data,
            "project_context": project_context,
            "prd_storage_timestamp": datetime.utcnow().isoformat(),
            "stored_content": prd_content
        }