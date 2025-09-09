"""
Main document generation service.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Literal
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.schema import TextNode
from llama_index.core.llms import LLM
from llama_index.core.query_engine import BaseQueryEngine

from core.config import get_settings
from services.llm.llm_service import LLMService
from .wbs_generator import WBSGenerator
from .resource_estimator import ResourceEstimator
from .export_service import ExportService, ExportFormat

logger = logging.getLogger(__name__)
settings = get_settings()


class DocumentRequest(BaseModel):
    """Request for document generation."""
    
    title: str = Field(..., description="Document title")
    document_type: Literal["prd", "technical_spec", "wbs", "project_plan"] = Field(
        default="prd", description="Type of document to generate"
    )
    context: Optional[str] = Field(None, description="Additional context or requirements")
    sections: List[str] = Field(
        default_factory=lambda: ["overview", "requirements", "implementation", "timeline"],
        description="Sections to include in document"
    )
    export_formats: List[ExportFormat] = Field(
        default_factory=lambda: [ExportFormat.JSON],
        description="Output formats"
    )
    include_wbs: bool = Field(True, description="Include Work Breakdown Structure")
    include_estimates: bool = Field(True, description="Include resource estimates")
    project_context: Optional[Dict[str, Any]] = Field(None, description="Project context data")


class DocumentResponse(BaseModel):
    """Response from document generation."""
    
    id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    document_type: str = Field(..., description="Document type")
    content: Dict[str, Any] = Field(..., description="Generated content structure")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    wbs: Optional[Dict[str, Any]] = Field(None, description="Work Breakdown Structure")
    estimates: Optional[Dict[str, Any]] = Field(None, description="Resource estimates")
    exports: Dict[ExportFormat, str] = Field(
        default_factory=dict, description="Export file paths by format"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    generation_time_ms: int = Field(..., description="Generation time in milliseconds")


class DocumentGenerator:
    """Main service for document generation with LlamaIndex integration."""
    
    def __init__(self):
        self.llm_service = LLMService()
        self.wbs_generator = WBSGenerator()
        self.resource_estimator = ResourceEstimator()
        self.export_service = ExportService()
        self._query_engine: Optional[BaseQueryEngine] = None
        
    async def initialize(self) -> None:
        """Initialize the document generation service."""
        try:
            # Initialize LlamaIndex settings
            await self._setup_llama_index()
            logger.info("Document generation service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize document generator: {e}")
            raise

    async def _setup_llama_index(self) -> None:
        """Set up LlamaIndex for structured document generation."""
        try:
            # Create sample documents for context (could be loaded from database)
            documents = [
                Document(
                    text="PRD Template: Product Requirements Document should include overview, user stories, functional requirements, technical requirements, acceptance criteria, and timeline.",
                    metadata={"type": "template", "document_type": "prd"}
                ),
                Document(
                    text="WBS Structure: Work Breakdown Structure should decompose project into phases, tasks, and subtasks with clear dependencies and resource allocation.",
                    metadata={"type": "template", "document_type": "wbs"}
                ),
                Document(
                    text="Technical Specification: Should include architecture overview, system components, API specifications, data models, and implementation details.",
                    metadata={"type": "template", "document_type": "technical_spec"}
                )
            ]
            
            # Create vector index
            index = VectorStoreIndex.from_documents(documents)
            self._query_engine = index.as_query_engine(
                response_mode="tree_summarize",
                streaming=False
            )
            
            logger.info("LlamaIndex setup completed")
            
        except Exception as e:
            logger.error(f"LlamaIndex setup failed: {e}")
            raise

    async def generate_document(self, request: DocumentRequest) -> DocumentResponse:
        """Generate a document based on the request."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"Generating {request.document_type} document: {request.title}")
            
            # Generate document content
            content = await self._generate_content(request)
            
            # Generate WBS if requested
            wbs = None
            if request.include_wbs:
                wbs = await self.wbs_generator.generate_wbs(
                    title=request.title,
                    requirements=content.get("requirements", []),
                    context=request.context
                )
            
            # Generate estimates if requested
            estimates = None
            if request.include_estimates:
                estimates = await self.resource_estimator.estimate_resources(
                    content=content,
                    wbs=wbs,
                    project_context=request.project_context
                )
            
            # Create document response
            doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            response = DocumentResponse(
                id=doc_id,
                title=request.title,
                document_type=request.document_type,
                content=content,
                metadata={
                    "sections": request.sections,
                    "generated_by": "DocumentGenerator",
                    "version": "1.0"
                },
                wbs=wbs,
                estimates=estimates,
                generation_time_ms=int((asyncio.get_event_loop().time() - start_time) * 1000)
            )
            
            # Export to requested formats
            exports = {}
            for export_format in request.export_formats:
                try:
                    file_path = await self.export_service.export_document(
                        document=response,
                        format=export_format
                    )
                    exports[export_format] = file_path
                except Exception as e:
                    logger.error(f"Export failed for format {export_format}: {e}")
                    # Continue with other formats
            
            response.exports = exports
            
            logger.info(f"Document generated successfully: {doc_id} in {response.generation_time_ms}ms")
            return response
            
        except Exception as e:
            logger.error(f"Document generation failed: {e}")
            raise

    async def _generate_content(self, request: DocumentRequest) -> Dict[str, Any]:
        """Generate the main document content."""
        try:
            # Build prompt for document generation
            prompt = self._build_generation_prompt(request)
            
            # Use LLM service for content generation
            response = await self.llm_service.generate_structured_content(
                prompt=prompt,
                context=request.context,
                max_tokens=4000
            )
            
            # Parse and structure the content
            content = await self._structure_content(response, request)
            
            return content
            
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            raise

    def _build_generation_prompt(self, request: DocumentRequest) -> str:
        """Build prompt for document generation."""
        base_prompt = f"""
Generate a comprehensive {request.document_type.replace('_', ' ')} titled "{request.title}".

Document Type: {request.document_type}
Required Sections: {', '.join(request.sections)}

Context: {request.context or 'No additional context provided'}

Please structure the output as a JSON object with the following format:
{{
    "overview": "Brief overview of the project/product",
    "requirements": ["Requirement 1", "Requirement 2", "..."],
    "implementation": {{
        "approach": "Implementation approach",
        "phases": ["Phase 1", "Phase 2", "..."],
        "technologies": ["Technology 1", "Technology 2", "..."]
    }},
    "timeline": {{
        "total_duration": "X weeks/months",
        "milestones": [
            {{"name": "Milestone 1", "date": "Date", "description": "Description"}},
            {{"name": "Milestone 2", "date": "Date", "description": "Description"}}
        ]
    }}
}}

Focus on providing actionable, detailed content that can be used for project planning and execution.
"""
        
        return base_prompt

    async def _structure_content(
        self, 
        raw_content: str, 
        request: DocumentRequest
    ) -> Dict[str, Any]:
        """Structure the generated content into a proper format."""
        try:
            # Try to parse as JSON first
            import json
            try:
                content = json.loads(raw_content)
                if isinstance(content, dict):
                    return content
            except json.JSONDecodeError:
                pass
            
            # Fallback: structure content manually based on sections
            content = {
                "overview": self._extract_section(raw_content, "overview"),
                "requirements": self._extract_requirements(raw_content),
                "implementation": {
                    "approach": self._extract_section(raw_content, "implementation"),
                    "phases": self._extract_phases(raw_content),
                    "technologies": self._extract_technologies(raw_content)
                },
                "timeline": {
                    "total_duration": self._extract_duration(raw_content),
                    "milestones": self._extract_milestones(raw_content)
                }
            }
            
            return content
            
        except Exception as e:
            logger.error(f"Content structuring failed: {e}")
            # Return basic structure with raw content
            return {
                "overview": "Generated document content",
                "raw_content": raw_content,
                "requirements": [],
                "implementation": {
                    "approach": "To be defined",
                    "phases": [],
                    "technologies": []
                },
                "timeline": {
                    "total_duration": "TBD",
                    "milestones": []
                }
            }

    def _extract_section(self, content: str, section_name: str) -> str:
        """Extract a specific section from raw content."""
        # Simple extraction logic - could be enhanced with NLP
        lines = content.split('\n')
        section_content = []
        in_section = False
        
        for line in lines:
            if section_name.lower() in line.lower():
                in_section = True
                continue
            elif in_section and (line.strip() == '' or any(s in line.lower() for s in ['requirements', 'implementation', 'timeline'])):
                break
            elif in_section:
                section_content.append(line.strip())
        
        return ' '.join(section_content) if section_content else f"Content for {section_name} section"

    def _extract_requirements(self, content: str) -> List[str]:
        """Extract requirements from content."""
        requirements = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('- ') or line.startswith('* '):
                requirements.append(line[2:])
            elif line.startswith(('1.', '2.', '3.', '4.', '5.')):
                requirements.append(line[3:])
        
        return requirements[:10]  # Limit to 10 requirements

    def _extract_phases(self, content: str) -> List[str]:
        """Extract project phases from content."""
        phases = ["Planning", "Development", "Testing", "Deployment"]
        return phases  # Default phases - could be enhanced

    def _extract_technologies(self, content: str) -> List[str]:
        """Extract technologies from content."""
        # Common technology keywords
        tech_keywords = ["Python", "FastAPI", "React", "PostgreSQL", "Redis", "Docker"]
        technologies = []
        
        for tech in tech_keywords:
            if tech.lower() in content.lower():
                technologies.append(tech)
        
        return technologies

    def _extract_duration(self, content: str) -> str:
        """Extract project duration from content."""
        # Simple extraction - could be enhanced
        import re
        duration_match = re.search(r'(\d+)\s*(weeks?|months?)', content.lower())
        if duration_match:
            return f"{duration_match.group(1)} {duration_match.group(2)}"
        return "12 weeks"  # Default duration

    def _extract_milestones(self, content: str) -> List[Dict[str, str]]:
        """Extract milestones from content."""
        # Default milestones - could be enhanced with NLP
        milestones = [
            {
                "name": "Project Kickoff",
                "date": "Week 1",
                "description": "Project initiation and planning"
            },
            {
                "name": "Development Complete",
                "date": "Week 8",
                "description": "Core development completed"
            },
            {
                "name": "Testing Complete",
                "date": "Week 10",
                "description": "All testing phases completed"
            },
            {
                "name": "Go Live",
                "date": "Week 12",
                "description": "Production deployment"
            }
        ]
        
        return milestones

    async def get_document_templates(self) -> List[Dict[str, Any]]:
        """Get available document templates."""
        templates = [
            {
                "id": "prd_standard",
                "name": "Standard PRD",
                "type": "prd",
                "description": "Standard Product Requirements Document template",
                "sections": ["overview", "user_stories", "requirements", "acceptance_criteria", "timeline"]
            },
            {
                "id": "technical_spec",
                "name": "Technical Specification",
                "type": "technical_spec",
                "description": "Detailed technical specification template",
                "sections": ["architecture", "components", "api_spec", "data_models", "implementation"]
            },
            {
                "id": "project_plan",
                "name": "Project Plan",
                "type": "project_plan",
                "description": "Comprehensive project planning template",
                "sections": ["overview", "scope", "timeline", "resources", "risks", "deliverables"]
            },
            {
                "id": "wbs_only",
                "name": "Work Breakdown Structure",
                "type": "wbs",
                "description": "Focused WBS document",
                "sections": ["structure", "tasks", "dependencies", "estimates"]
            }
        ]
        
        return templates