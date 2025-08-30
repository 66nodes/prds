"""
GraphRAG validation endpoints
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.config import get_settings
from services.graphrag.graph_service import GraphRAGService, ValidationResult
from .auth import get_current_user, User

logger = structlog.get_logger(__name__)
settings = get_settings()

router = APIRouter()


# Request/Response Models
class ValidationRequest(BaseModel):
    """Content validation request."""
    content: str = Field(..., min_length=10, max_length=5000, description="Content to validate")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Validation context")
    section_type: Optional[str] = Field(None, description="Type of content section")
    project_id: Optional[str] = Field(None, description="Associated project ID")


class ValidationResponse(BaseModel):
    """Content validation response."""
    validation_id: str = Field(..., description="Validation result ID")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    passes_threshold: bool = Field(..., description="Meets validation threshold")
    entity_validation: Dict[str, Any] = Field(..., description="Entity-level validation results")
    community_validation: Dict[str, Any] = Field(..., description="Community-level validation results")
    global_validation: Dict[str, Any] = Field(..., description="Global-level validation results")
    corrections: List[str] = Field(default_factory=list, description="Suggested corrections")
    requires_human_review: bool = Field(..., description="Requires manual review")
    timestamp: str = Field(..., description="Validation timestamp")


class BatchValidationRequest(BaseModel):
    """Batch validation request for multiple content pieces."""
    items: List[ValidationRequest] = Field(..., min_items=1, max_items=10, description="Content items to validate")
    parallel: bool = Field(default=True, description="Process items in parallel")


class BatchValidationResponse(BaseModel):
    """Batch validation response."""
    results: List[ValidationResponse] = Field(..., description="Validation results")
    summary: Dict[str, Any] = Field(..., description="Batch validation summary")


class ConfidenceScoreRequest(BaseModel):
    """Confidence score calculation request."""
    prd_id: str = Field(..., description="PRD ID to analyze")


class ConfidenceScoreResponse(BaseModel):
    """Confidence score response."""
    prd_id: str = Field(..., description="PRD ID")
    overall_confidence: float = Field(..., description="Overall confidence score")
    section_scores: List[Dict[str, Any]] = Field(..., description="Individual section scores")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    validation_history: List[Dict[str, Any]] = Field(..., description="Historical validation data")


# Dependency injection
async def get_graphrag_service() -> GraphRAGService:
    """Get GraphRAG service instance."""
    service = GraphRAGService()
    if not service.is_initialized:
        await service.initialize()
    return service


# Validation Endpoints
@router.post("/validate-content", response_model=ValidationResponse)
async def validate_content(
    request: ValidationRequest,
    graphrag_service: GraphRAGService = Depends(get_graphrag_service),
    current_user: User = Depends(get_current_user)
):
    """Validate content using GraphRAG three-tier validation."""
    try:
        logger.info(
            "Content validation requested",
            user_id=current_user.id,
            content_length=len(request.content),
            section_type=request.section_type
        )
        
        # Enhance context with user information
        enhanced_context = {
            **request.context,
            "user_id": current_user.id,
            "section_type": request.section_type,
            "project_id": request.project_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Perform GraphRAG validation
        validation_result = await graphrag_service.validate_content(
            request.content,
            enhanced_context
        )
        
        # Convert to response model
        response = ValidationResponse(
            validation_id=f"val-{validation_result.timestamp.strftime('%Y%m%d%H%M%S')}-{current_user.id[:8]}",
            confidence=validation_result.confidence,
            passes_threshold=validation_result.confidence >= settings.validation_threshold,
            entity_validation=validation_result.entity_validation,
            community_validation=validation_result.community_validation,
            global_validation=validation_result.global_validation,
            corrections=validation_result.corrections,
            requires_human_review=validation_result.requires_human_review,
            timestamp=validation_result.timestamp.isoformat()
        )
        
        logger.info(
            "Content validation completed",
            user_id=current_user.id,
            confidence=validation_result.confidence,
            passes_threshold=response.passes_threshold
        )
        
        return response
        
    except Exception as e:
        logger.error("Content validation failed", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.post("/validate-batch", response_model=BatchValidationResponse)
async def validate_batch_content(
    request: BatchValidationRequest,
    graphrag_service: GraphRAGService = Depends(get_graphrag_service),
    current_user: User = Depends(get_current_user)
):
    """Validate multiple content pieces in batch."""
    try:
        logger.info(
            "Batch validation requested",
            user_id=current_user.id,
            items_count=len(request.items),
            parallel=request.parallel
        )
        
        validation_results = []
        
        if request.parallel:
            # Process items in parallel
            import asyncio
            
            async def validate_item(item: ValidationRequest) -> ValidationResponse:
                enhanced_context = {
                    **item.context,
                    "user_id": current_user.id,
                    "section_type": item.section_type,
                    "project_id": item.project_id,
                    "batch_validation": True
                }
                
                result = await graphrag_service.validate_content(item.content, enhanced_context)
                
                return ValidationResponse(
                    validation_id=f"val-batch-{result.timestamp.strftime('%Y%m%d%H%M%S')}-{len(validation_results)}",
                    confidence=result.confidence,
                    passes_threshold=result.confidence >= settings.validation_threshold,
                    entity_validation=result.entity_validation,
                    community_validation=result.community_validation,
                    global_validation=result.global_validation,
                    corrections=result.corrections,
                    requires_human_review=result.requires_human_review,
                    timestamp=result.timestamp.isoformat()
                )
            
            validation_results = await asyncio.gather(
                *[validate_item(item) for item in request.items],
                return_exceptions=True
            )
            
            # Handle exceptions
            validation_results = [
                result if not isinstance(result, Exception) else 
                ValidationResponse(
                    validation_id=f"val-error-{i}",
                    confidence=0.0,
                    passes_threshold=False,
                    entity_validation={"error": str(result)},
                    community_validation={"error": str(result)},
                    global_validation={"error": str(result)},
                    corrections=["Validation failed - manual review required"],
                    requires_human_review=True,
                    timestamp=datetime.utcnow().isoformat()
                )
                for i, result in enumerate(validation_results)
            ]
        
        else:
            # Process items sequentially
            for i, item in enumerate(request.items):
                enhanced_context = {
                    **item.context,
                    "user_id": current_user.id,
                    "section_type": item.section_type,
                    "project_id": item.project_id,
                    "batch_validation": True,
                    "sequence_number": i
                }
                
                result = await graphrag_service.validate_content(item.content, enhanced_context)
                
                validation_results.append(ValidationResponse(
                    validation_id=f"val-seq-{result.timestamp.strftime('%Y%m%d%H%M%S')}-{i}",
                    confidence=result.confidence,
                    passes_threshold=result.confidence >= settings.validation_threshold,
                    entity_validation=result.entity_validation,
                    community_validation=result.community_validation,
                    global_validation=result.global_validation,
                    corrections=result.corrections,
                    requires_human_review=result.requires_human_review,
                    timestamp=result.timestamp.isoformat()
                ))
        
        # Calculate batch summary
        confidences = [result.confidence for result in validation_results]
        summary = {
            "total_items": len(validation_results),
            "average_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "passed_threshold": sum(1 for result in validation_results if result.passes_threshold),
            "requires_review": sum(1 for result in validation_results if result.requires_human_review),
            "processing_time": "batch",
            "parallel_processing": request.parallel
        }
        
        logger.info(
            "Batch validation completed",
            user_id=current_user.id,
            total_items=summary["total_items"],
            average_confidence=summary["average_confidence"],
            passed_threshold=summary["passed_threshold"]
        )
        
        return BatchValidationResponse(
            results=validation_results,
            summary=summary
        )
        
    except Exception as e:
        logger.error("Batch validation failed", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail=f"Batch validation failed: {str(e)}")


@router.get("/confidence-score/{prd_id}", response_model=ConfidenceScoreResponse)
async def get_confidence_score(
    prd_id: str,
    graphrag_service: GraphRAGService = Depends(get_graphrag_service),
    current_user: User = Depends(get_current_user)
):
    """Get overall confidence score for a PRD."""
    try:
        logger.info("Confidence score requested", prd_id=prd_id, user_id=current_user.id)
        
        # Get PRD data from Neo4j
        neo4j_conn = await graphrag_service.neo4j_conn
        
        prd_query = """
        MATCH (p:PRD {id: $prd_id})-[:CONTAINS]->(s:Section)
        OPTIONAL MATCH (p)-[:VALIDATED_BY]->(v:ValidationResult)
        RETURN p.title as title, p.quality_score as quality_score,
               collect({
                   id: s.id,
                   title: s.title,
                   validation_score: s.validation_score,
                   status: s.status
               }) as sections,
               collect({
                   confidence: v.confidence,
                   created_at: v.created_at,
                   passes_threshold: v.passes_threshold
               }) as validation_history
        """
        
        result = await neo4j_conn.execute_query(prd_query, {"prd_id": prd_id})
        
        if not result:
            raise HTTPException(status_code=404, detail="PRD not found")
        
        prd_data = result[0]
        sections = prd_data["sections"]
        validation_history = prd_data["validation_history"]
        
        # Calculate overall confidence
        section_scores = []
        total_confidence = 0.0
        
        for section in sections:
            validation_score = section.get("validation_score", 0.0)
            section_scores.append({
                "section_id": section["id"],
                "title": section["title"],
                "confidence": validation_score,
                "status": section["status"],
                "passes_threshold": validation_score >= settings.validation_threshold
            })
            total_confidence += validation_score
        
        overall_confidence = total_confidence / len(sections) if sections else 0.0
        
        # Generate recommendations
        recommendations = []
        low_confidence_sections = [s for s in section_scores if s["confidence"] < settings.validation_threshold]
        
        if low_confidence_sections:
            recommendations.append(f"{len(low_confidence_sections)} sections need improvement")
            
        if overall_confidence < 0.7:
            recommendations.append("Consider adding more specific details and context")
            
        if overall_confidence < 0.5:
            recommendations.append("Manual review recommended before proceeding")
        
        logger.info(
            "Confidence score calculated",
            prd_id=prd_id,
            overall_confidence=overall_confidence,
            sections_count=len(sections)
        )
        
        return ConfidenceScoreResponse(
            prd_id=prd_id,
            overall_confidence=overall_confidence,
            section_scores=section_scores,
            recommendations=recommendations,
            validation_history=validation_history
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Confidence score calculation failed", error=str(e), prd_id=prd_id)
        raise HTTPException(status_code=500, detail=f"Failed to calculate confidence score: {str(e)}")


@router.get("/validation-stats")
async def get_validation_statistics(
    current_user: User = Depends(get_current_user),
    graphrag_service: GraphRAGService = Depends(get_graphrag_service)
):
    """Get validation statistics for the current user."""
    try:
        # Get validation statistics from Neo4j
        neo4j_conn = graphrag_service.neo4j_conn
        
        stats_query = """
        MATCH (v:ValidationResult)
        WHERE v.created_at >= datetime() - duration('P30D')
        RETURN 
            count(v) as total_validations,
            avg(v.confidence) as average_confidence,
            count(CASE WHEN v.passes_threshold THEN 1 END) as passed_validations,
            count(CASE WHEN v.confidence < 0.5 THEN 1 END) as low_confidence_validations
        """
        
        result = await neo4j_conn.execute_query(stats_query)
        stats = result[0] if result else {}
        
        return {
            "period": "Last 30 days",
            "total_validations": stats.get("total_validations", 0),
            "average_confidence": round(stats.get("average_confidence", 0.0), 3),
            "passed_validations": stats.get("passed_validations", 0),
            "low_confidence_validations": stats.get("low_confidence_validations", 0),
            "pass_rate": round(
                (stats.get("passed_validations", 0) / stats.get("total_validations", 1)) * 100, 2
            ) if stats.get("total_validations", 0) > 0 else 0.0
        }
        
    except Exception as e:
        logger.error("Validation statistics failed", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail=f"Failed to get validation statistics: {str(e)}")