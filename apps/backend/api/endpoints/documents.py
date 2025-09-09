"""
FastAPI endpoints for document generation service.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path
from fastapi.responses import FileResponse
from starlette import status
from pydantic import BaseModel, Field, validator

from core.auth import get_current_user
from services.document import DocumentGenerator
from services.document.export_service import ExportFormat, ExportOptions
from services.document.wbs_generator import TaskComplexity, TaskPriority
from services.document.resource_estimator import SkillLevel, ResourceType

logger = logging.getLogger(__name__)

# Global document generator instance
_document_generator: Optional[DocumentGenerator] = None

async def get_document_generator() -> DocumentGenerator:
    """Get document generator instance."""
    global _document_generator
    if _document_generator is None:
        _document_generator = DocumentGenerator()
        await _document_generator.initialize()
    return _document_generator

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


# Request/Response Models
class DocumentGenerationRequest(BaseModel):
    """Request model for document generation."""
    
    title: str = Field(..., min_length=1, max_length=200, description="Document title")
    document_type: str = Field(
        default="prd", 
        description="Document type: prd, technical_spec, wbs, project_plan"
    )
    context: Optional[str] = Field(None, max_length=2000, description="Additional context")
    sections: List[str] = Field(
        default_factory=lambda: ["overview", "requirements", "implementation", "timeline"],
        description="Document sections to include"
    )
    export_formats: List[ExportFormat] = Field(
        default_factory=lambda: [ExportFormat.JSON],
        description="Export formats"
    )
    include_wbs: bool = Field(True, description="Include Work Breakdown Structure")
    include_estimates: bool = Field(True, description="Include resource estimates")
    project_context: Optional[Dict[str, Any]] = Field(None, description="Project context")

    @validator('document_type')
    def validate_document_type(cls, v):
        valid_types = ["prd", "technical_spec", "wbs", "project_plan"]
        if v not in valid_types:
            raise ValueError(f"Document type must be one of: {', '.join(valid_types)}")
        return v

    @validator('sections')
    def validate_sections(cls, v):
        if len(v) == 0:
            raise ValueError("At least one section must be specified")
        return v[:10]  # Limit to 10 sections


class DocumentResponse(BaseModel):
    """Response model for document generation."""
    
    id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    document_type: str = Field(..., description="Document type")
    content: Dict[str, Any] = Field(..., description="Generated content")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    wbs: Optional[Dict[str, Any]] = Field(None, description="Work Breakdown Structure")
    estimates: Optional[Dict[str, Any]] = Field(None, description="Resource estimates")
    exports: Dict[str, str] = Field(..., description="Export file paths by format")
    created_at: str = Field(..., description="Creation timestamp")
    generation_time_ms: int = Field(..., description="Generation time in milliseconds")


class ExportRequest(BaseModel):
    """Request model for document export."""
    
    format: ExportFormat = Field(..., description="Export format")
    options: Optional[ExportOptions] = Field(None, description="Export options")


class TemplateResponse(BaseModel):
    """Response model for document templates."""
    
    id: str = Field(..., description="Template ID")
    name: str = Field(..., description="Template name")
    type: str = Field(..., description="Document type")
    description: str = Field(..., description="Template description")
    sections: List[str] = Field(..., description="Default sections")


class DocumentListResponse(BaseModel):
    """Response model for document list."""
    
    documents: List[Dict[str, Any]] = Field(..., description="Document list")
    total_count: int = Field(..., description="Total document count")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Page size")


# Document Generation Endpoints
@router.post("/generate", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def generate_document(
    request: DocumentGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user),
    document_generator: DocumentGenerator = Depends(get_document_generator)
):
    """
    Generate a new document with AI assistance.
    
    This endpoint creates comprehensive documents including:
    - Content generation using LLM
    - Work Breakdown Structure (WBS) if requested
    - Resource and cost estimates if requested  
    - Export to multiple formats
    """
    try:
        logger.info(f"Generating document: {request.title} for user {current_user.get('user_id')}")
        
        # Convert request to document generator format
        from services.document.document_generator import DocumentRequest
        
        doc_request = DocumentRequest(
            title=request.title,
            document_type=request.document_type,
            context=request.context,
            sections=request.sections,
            export_formats=request.export_formats,
            include_wbs=request.include_wbs,
            include_estimates=request.include_estimates,
            project_context=request.project_context
        )
        
        # Generate document
        document = await document_generator.generate_document(doc_request)
        
        # Convert response to API format
        response = DocumentResponse(
            id=document.id,
            title=document.title,
            document_type=document.document_type,
            content=document.content,
            metadata=document.metadata,
            wbs=document.wbs.dict() if document.wbs else None,
            estimates=document.estimates.dict() if document.estimates else None,
            exports=document.exports,
            created_at=document.created_at.isoformat(),
            generation_time_ms=document.generation_time_ms
        )
        
        logger.info(f"Document generated successfully: {document.id}")
        return response
        
    except ValueError as e:
        logger.warning(f"Invalid request for document generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Document generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document generation failed"
        )


@router.get("/templates", response_model=List[TemplateResponse])
async def get_document_templates(
    current_user: Dict = Depends(get_current_user),
    document_generator: DocumentGenerator = Depends(get_document_generator)
):
    """
    Get available document templates.
    
    Returns a list of pre-configured document templates with their
    sections and configuration options.
    """
    try:
        templates = await document_generator.get_document_templates()
        
        response = [
            TemplateResponse(
                id=template["id"],
                name=template["name"],
                type=template["type"],
                description=template["description"],
                sections=template["sections"]
            ) for template in templates
        ]
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get document templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document templates"
        )


# Export Endpoints
@router.post("/export/{document_id}", response_model=Dict[str, str])
async def export_document(
    document_id: str = Path(..., description="Document ID"),
    export_request: ExportRequest = None,
    current_user: Dict = Depends(get_current_user),
    document_generator: DocumentGenerator = Depends(get_document_generator)
):
    """
    Export an existing document to specified format.
    
    Supports multiple export formats:
    - PDF: Professional PDF documents
    - Word: Microsoft Word format
    - HTML: Web-ready HTML format
    - Markdown: Markdown format
    - JSON: Structured JSON format
    """
    try:
        # For now, return a placeholder since we don't have document storage
        # In a real implementation, you would retrieve the document from database
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Document export from storage not yet implemented. Use direct generation with export formats."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document export failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document export failed"
        )


@router.get("/download/{document_id}/{format}")
async def download_document(
    document_id: str = Path(..., description="Document ID"),
    format: str = Path(..., description="File format"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Download a previously generated document file.
    
    Returns the actual file for download with proper content-type headers.
    """
    try:
        # For now, return an error since we don't have persistent storage
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Document download not yet implemented. Files are available immediately after generation."
        )
        
        # In a real implementation:
        # file_path = await get_document_file_path(document_id, format)
        # return FileResponse(
        #     path=file_path,
        #     filename=f"{document_id}.{format}",
        #     media_type=get_media_type(format)
        # )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document download failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document download failed"
        )


# Analysis and Estimation Endpoints
@router.post("/analyze/wbs")
async def analyze_wbs_complexity(
    request: Dict[str, Any],
    current_user: Dict = Depends(get_current_user),
    document_generator: DocumentGenerator = Depends(get_document_generator)
):
    """
    Analyze Work Breakdown Structure complexity and provide recommendations.
    
    Input should include project requirements and context.
    Returns complexity analysis and optimization suggestions.
    """
    try:
        title = request.get("title", "Project Analysis")
        requirements = request.get("requirements", [])
        context = request.get("context")
        project_type = request.get("project_type", "software")
        
        # Generate WBS for analysis
        wbs = await document_generator.wbs_generator.generate_wbs(
            title=title,
            requirements=requirements,
            context=context,
            project_type=project_type
        )
        
        # Return analysis results
        analysis = {
            "complexity_summary": {
                "total_hours": wbs.total_estimated_hours,
                "total_days": wbs.total_estimated_days,
                "total_phases": len(wbs.phases),
                "total_tasks": sum(len(phase.tasks) for phase in wbs.phases),
                "risk_level": wbs.risk_assessment.get("overall_risk_level", "medium")
            },
            "phase_breakdown": [
                {
                    "name": phase.name,
                    "duration_days": phase.estimated_duration_days,
                    "task_count": len(phase.tasks),
                    "complexity_distribution": {
                        "simple": sum(1 for task in phase.tasks if task.complexity.value == "simple"),
                        "moderate": sum(1 for task in phase.tasks if task.complexity.value == "moderate"), 
                        "complex": sum(1 for task in phase.tasks if task.complexity.value == "complex"),
                        "expert": sum(1 for task in phase.tasks if task.complexity.value == "expert")
                    }
                } for phase in wbs.phases
            ],
            "critical_path": wbs.critical_path,
            "risk_assessment": wbs.risk_assessment,
            "recommendations": wbs.risk_assessment.get("recommended_actions", [])
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"WBS analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="WBS analysis failed"
        )


@router.post("/analyze/resources")
async def analyze_resource_requirements(
    request: Dict[str, Any],
    current_user: Dict = Depends(get_current_user),
    document_generator: DocumentGenerator = Depends(get_document_generator)
):
    """
    Analyze resource requirements and provide cost estimates.
    
    Input should include project content and optional WBS data.
    Returns detailed resource analysis and cost breakdown.
    """
    try:
        content = request.get("content", {})
        wbs_data = request.get("wbs")
        project_context = request.get("project_context", {})
        
        # Convert WBS data if provided
        wbs = None
        if wbs_data:
            # In a real implementation, you'd reconstruct the WBS object from data
            pass
        
        # Generate resource estimates
        estimates = await document_generator.resource_estimator.estimate_resources(
            content=content,
            wbs=wbs,
            project_context=project_context
        )
        
        # Return analysis results
        analysis = {
            "team_composition": {
                "total_team_size": estimates.team_composition.total_team_size,
                "roles": estimates.team_composition.roles,
                "skill_distribution": estimates.team_composition.skill_distribution,
                "monthly_cost": estimates.team_composition.estimated_monthly_cost,
                "duration_months": estimates.team_composition.recommended_duration_months
            },
            "cost_breakdown": {
                "total_cost": estimates.cost_estimate.total_cost,
                "human_resources": estimates.cost_estimate.human_resources,
                "infrastructure": estimates.cost_estimate.infrastructure,
                "software_licenses": estimates.cost_estimate.software_licenses,
                "contingency": estimates.cost_estimate.contingency,
                "confidence_level": estimates.cost_estimate.confidence_level
            },
            "timeline_projection": {
                "total_days": estimates.timeline_estimate.total_duration_days,
                "total_months": estimates.timeline_estimate.total_duration_months,
                "critical_path_days": estimates.timeline_estimate.critical_path_duration,
                "buffer_days": estimates.timeline_estimate.buffer_days,
                "risk_adjusted_days": estimates.timeline_estimate.risk_adjusted_duration
            },
            "assumptions": estimates.assumptions,
            "risks": estimates.risks,
            "recommendations": estimates.recommendations
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Resource analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Resource analysis failed"
        )


# Health and Status Endpoints
@router.get("/health")
async def health_check():
    """
    Health check endpoint for document generation service.
    """
    try:
        # Test document generator initialization
        doc_gen = await get_document_generator()
        
        return {
            "status": "healthy",
            "service": "document_generation",
            "timestamp": asyncio.get_event_loop().time(),
            "components": {
                "document_generator": "operational",
                "wbs_generator": "operational", 
                "resource_estimator": "operational",
                "export_service": "operational"
            }
        }
        
    except Exception as e:
        logger.error(f"Document service health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "document_generation",
            "error": str(e),
            "timestamp": asyncio.get_event_loop().time()
        }


# Configuration and Metadata Endpoints
@router.get("/config/formats")
async def get_supported_formats():
    """
    Get supported export formats and their capabilities.
    """
    formats = {
        "pdf": {
            "name": "PDF",
            "description": "Portable Document Format",
            "supports_images": True,
            "supports_tables": True,
            "supports_charts": False,
            "file_extension": "pdf",
            "mime_type": "application/pdf"
        },
        "docx": {
            "name": "Microsoft Word",
            "description": "Word document format",
            "supports_images": True,
            "supports_tables": True,
            "supports_charts": False,
            "file_extension": "docx",
            "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        },
        "html": {
            "name": "HTML",
            "description": "Web document format",
            "supports_images": True,
            "supports_tables": True,
            "supports_charts": True,
            "file_extension": "html",
            "mime_type": "text/html"
        },
        "markdown": {
            "name": "Markdown",
            "description": "Lightweight markup format",
            "supports_images": True,
            "supports_tables": True,
            "supports_charts": False,
            "file_extension": "md",
            "mime_type": "text/markdown"
        },
        "json": {
            "name": "JSON",
            "description": "Structured data format",
            "supports_images": False,
            "supports_tables": False,
            "supports_charts": False,
            "file_extension": "json",
            "mime_type": "application/json"
        }
    }
    
    return formats


@router.get("/config/complexity-levels")
async def get_complexity_levels():
    """
    Get available task complexity levels and their descriptions.
    """
    return {
        "simple": {
            "name": "Simple",
            "description": "Straightforward tasks requiring basic skills",
            "typical_duration": "1-8 hours",
            "skill_level_required": "junior-intermediate"
        },
        "moderate": {
            "name": "Moderate", 
            "description": "Standard tasks requiring solid understanding",
            "typical_duration": "1-3 days",
            "skill_level_required": "intermediate"
        },
        "complex": {
            "name": "Complex",
            "description": "Advanced tasks requiring deep expertise",
            "typical_duration": "3-10 days", 
            "skill_level_required": "senior"
        },
        "expert": {
            "name": "Expert",
            "description": "Highly specialized tasks requiring rare expertise",
            "typical_duration": "1-4 weeks",
            "skill_level_required": "expert"
        }
    }


@router.get("/config/skill-levels")
async def get_skill_levels():
    """
    Get available skill levels and their hourly rates.
    """
    return {
        "junior": {
            "name": "Junior",
            "description": "0-2 years experience",
            "typical_hourly_rate_usd": 50,
            "responsibilities": ["Basic implementation", "Code reviews", "Bug fixes"]
        },
        "intermediate": {
            "name": "Intermediate", 
            "description": "2-5 years experience",
            "typical_hourly_rate_usd": 75,
            "responsibilities": ["Feature development", "System design", "Mentoring"]
        },
        "senior": {
            "name": "Senior",
            "description": "5-8 years experience", 
            "typical_hourly_rate_usd": 120,
            "responsibilities": ["Architecture decisions", "Technical leadership", "Complex problem solving"]
        },
        "expert": {
            "name": "Expert",
            "description": "8+ years experience",
            "typical_hourly_rate_usd": 180,
            "responsibilities": ["Strategic planning", "Innovation", "Cross-team leadership"]
        }
    }