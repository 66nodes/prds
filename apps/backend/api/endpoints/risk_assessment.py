"""
Risk Assessment API endpoints for project risk analysis and historical insights.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
import io
import json

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
import structlog

from core.config import get_settings
from core.database import get_neo4j
from api.dependencies.auth import get_current_user
from services.risk_assessment_service import (
    get_risk_assessment_service, 
    RiskAssessmentResult,
    LessonsLearned
)
from services.pattern_recognition_service import (
    get_pattern_recognition_service,
    PatternAnalysisResult
)
from services.risk_scoring_algorithm import (
    get_risk_scoring_algorithm,
    RiskScoreResult
)

logger = structlog.get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/risk-assessment", tags=["Risk Assessment"])


# Request/Response Models
class RiskAssessmentRequest(BaseModel):
    """Risk assessment request model."""
    project_description: str = Field(..., min_length=50, max_length=5000, description="Project description")
    project_id: Optional[str] = Field(None, description="Project identifier")
    project_category: Optional[str] = Field(None, description="Project category")
    include_historical: bool = Field(default=True, description="Include historical analysis")
    include_templates: bool = Field(default=True, description="Include template recommendations")
    include_lessons: bool = Field(default=True, description="Include lessons learned")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

    @validator('project_description')
    def validate_description(cls, v):
        if len(v.strip().split()) < 10:
            raise ValueError('Description must contain at least 10 words')
        return v.strip()


class PatternAnalysisRequest(BaseModel):
    """Pattern analysis request model."""
    project_description: str = Field(..., min_length=50, max_length=5000)
    context: Dict[str, Any] = Field(default_factory=dict)


class RiskScoringRequest(BaseModel):
    """Risk scoring request model."""
    project_description: str = Field(..., min_length=50, max_length=5000)
    context: Dict[str, Any] = Field(default_factory=dict)


class HistoricalComparisonRequest(BaseModel):
    """Historical comparison request model."""
    risk_score: float = Field(..., ge=0.0, le=1.0)
    project_description: str = Field(..., min_length=50)
    project_category: Optional[str] = None


class LessonsLearnedRequest(BaseModel):
    """Lessons learned request model."""
    category: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=50)


class ExportRequest(BaseModel):
    """Export request model."""
    assessment: Dict[str, Any] = Field(..., description="Risk assessment data")
    format: str = Field(default="pdf", description="Export format")
    include_charts: bool = Field(default=True, description="Include charts and visualizations")


class RiskAssessmentResponse(BaseModel):
    """Risk assessment response model."""
    assessment: RiskAssessmentResult
    pattern_analysis: Optional[PatternAnalysisResult] = None
    risk_scoring: Optional[RiskScoreResult] = None
    lessons_learned: List[LessonsLearned] = Field(default_factory=list)
    processing_time: float = Field(..., description="Processing time in seconds")
    cached: bool = Field(default=False, description="Whether result was cached")


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    services: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# API Endpoints
@router.post("/", response_model=RiskAssessmentResponse)
async def run_risk_assessment(
    request: RiskAssessmentRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
) -> RiskAssessmentResponse:
    """
    Run comprehensive risk assessment for a project.
    
    Analyzes project description to identify risks, patterns, and provide
    historical insights and recommendations.
    """
    start_time = datetime.utcnow()
    
    try:
        logger.info(
            "Starting risk assessment",
            user_id=current_user.get("id"),
            project_id=request.project_id,
            description_length=len(request.project_description)
        )
        
        # Get services
        risk_service = await get_risk_assessment_service()
        pattern_service = await get_pattern_recognition_service()
        scoring_service = await get_risk_scoring_algorithm()
        
        # Run assessments in parallel
        tasks = [
            risk_service.assess_project_risks(
                request.project_description,
                request.project_category,
                request.context
            )
        ]
        
        # Add optional analyses
        if request.include_historical:
            tasks.append(
                pattern_service.analyze_patterns(request.project_description, request.context)
            )
            tasks.append(
                scoring_service.calculate_risk_score(request.project_description, request.context)
            )
        
        # Execute tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        risk_assessment = results[0]
        if isinstance(risk_assessment, Exception):
            logger.error("Risk assessment failed", error=str(risk_assessment))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Risk assessment failed: {str(risk_assessment)}"
            )
        
        pattern_analysis = None
        risk_scoring = None
        
        if request.include_historical and len(results) > 1:
            if not isinstance(results[1], Exception):
                pattern_analysis = results[1]
            if len(results) > 2 and not isinstance(results[2], Exception):
                risk_scoring = results[2]
        
        # Get lessons learned if requested
        lessons_learned = []
        if request.include_lessons:
            try:
                lessons_learned = await risk_service.get_lessons_learned(
                    category=request.project_category,
                    limit=5
                )
            except Exception as e:
                logger.warning("Failed to get lessons learned", error=str(e))
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Log assessment for analytics (background task)
        background_tasks.add_task(
            _log_assessment_usage,
            current_user.get("id"),
            request.project_id,
            risk_assessment.overall_risk_score,
            processing_time
        )
        
        response = RiskAssessmentResponse(
            assessment=risk_assessment,
            pattern_analysis=pattern_analysis,
            risk_scoring=risk_scoring,
            lessons_learned=lessons_learned,
            processing_time=processing_time,
            cached=False  # TODO: Implement caching detection
        )
        
        logger.info(
            "Risk assessment completed",
            user_id=current_user.get("id"),
            risk_score=risk_assessment.overall_risk_score,
            processing_time=processing_time
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Risk assessment failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during risk assessment"
        )


@router.post("/patterns", response_model=PatternAnalysisResult)
async def analyze_patterns(
    request: PatternAnalysisRequest,
    current_user = Depends(get_current_user)
) -> PatternAnalysisResult:
    """
    Analyze project patterns for risk detection and template suggestions.
    """
    try:
        logger.info(
            "Starting pattern analysis",
            user_id=current_user.get("id"),
            description_length=len(request.project_description)
        )
        
        pattern_service = await get_pattern_recognition_service()
        result = await pattern_service.analyze_patterns(
            request.project_description,
            request.context
        )
        
        logger.info(
            "Pattern analysis completed",
            user_id=current_user.get("id"),
            patterns_count=len(result.detected_patterns),
            templates_count=len(result.template_recommendations)
        )
        
        return result
        
    except Exception as e:
        logger.error("Pattern analysis failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Pattern analysis failed"
        )


@router.post("/scoring", response_model=RiskScoreResult)
async def calculate_risk_score(
    request: RiskScoringRequest,
    current_user = Depends(get_current_user)
) -> RiskScoreResult:
    """
    Calculate detailed risk score using advanced algorithms.
    """
    try:
        logger.info(
            "Starting risk scoring",
            user_id=current_user.get("id"),
            description_length=len(request.project_description)
        )
        
        scoring_service = await get_risk_scoring_algorithm()
        result = await scoring_service.calculate_risk_score(
            request.project_description,
            request.context
        )
        
        logger.info(
            "Risk scoring completed",
            user_id=current_user.get("id"),
            risk_score=result.overall_score,
            confidence=result.confidence
        )
        
        return result
        
    except Exception as e:
        logger.error("Risk scoring failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Risk scoring failed"
        )


@router.post("/historical-comparison")
async def get_historical_comparison(
    request: HistoricalComparisonRequest,
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get historical comparison data for similar projects.
    """
    try:
        neo4j_conn = await get_neo4j()
        
        # Find projects with similar risk scores
        comparison_query = """
        MATCH (p:Project)
        WHERE p.risk_score IS NOT NULL 
        AND p.risk_score >= $min_score 
        AND p.risk_score <= $max_score
        AND p.success_score IS NOT NULL
        WITH p, 
             CASE WHEN p.success_score >= 0.7 THEN 1 ELSE 0 END as success_flag
        RETURN 
            COUNT(p) as total_projects,
            AVG(p.success_score) as avg_success_rate,
            SUM(success_flag) as successful_projects,
            AVG(p.risk_score) as avg_risk_score,
            STDDEV(p.success_score) as success_stddev,
            MIN(p.success_score) as min_success,
            MAX(p.success_score) as max_success
        """
        
        min_score = max(0.0, request.risk_score - 0.1)
        max_score = min(1.0, request.risk_score + 0.1)
        
        results = await neo4j_conn.execute_query(
            comparison_query,
            {
                "min_score": min_score,
                "max_score": max_score
            }
        )
        
        if results and results[0]["total_projects"] > 0:
            data = results[0]
            success_rate = data["successful_projects"] / data["total_projects"]
            
            comparison = {
                "similar_projects_count": data["total_projects"],
                "average_success_rate": data["avg_success_rate"],
                "success_rate": success_rate,
                "average_risk_score": data["avg_risk_score"],
                "score_range": f"{min_score:.2f}-{max_score:.2f}",
                "confidence_interval": {
                    "min": data["min_success"],
                    "max": data["max_success"]
                },
                "statistical_significance": data["total_projects"] >= 30
            }
        else:
            comparison = {
                "similar_projects_count": 0,
                "message": "Limited historical data for this risk score range",
                "score_range": f"{min_score:.2f}-{max_score:.2f}"
            }
        
        return {"comparison": comparison}
        
    except Exception as e:
        logger.error("Historical comparison failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Historical comparison failed"
        )


@router.get("/lessons-learned")
async def get_lessons_learned(
    category: Optional[str] = Query(None, description="Lesson category filter"),
    limit: int = Query(10, ge=1, le=50, description="Number of lessons to return"),
    current_user = Depends(get_current_user)
) -> Dict[str, List[LessonsLearned]]:
    """
    Get lessons learned from historical projects.
    """
    try:
        risk_service = await get_risk_assessment_service()
        lessons = await risk_service.get_lessons_learned(category, limit)
        
        return {"lessons": lessons}
        
    except Exception as e:
        logger.error("Failed to get lessons learned", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve lessons learned"
        )


@router.post("/export")
async def export_assessment(
    request: ExportRequest,
    current_user = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Export risk assessment report in various formats.
    """
    try:
        # Generate export file based on format
        if request.format.lower() == "pdf":
            file_content = await _generate_pdf_report(
                request.assessment,
                request.include_charts
            )
            content_type = "application/pdf"
            file_extension = "pdf"
        elif request.format.lower() == "json":
            file_content = json.dumps(request.assessment, indent=2).encode()
            content_type = "application/json"
            file_extension = "json"
        elif request.format.lower() == "csv":
            file_content = await _generate_csv_report(request.assessment)
            content_type = "text/csv"
            file_extension = "csv"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported export format"
            )
        
        # Store file temporarily (in production, use cloud storage)
        file_id = str(uuid4())
        file_path = f"/tmp/risk_assessment_{file_id}.{file_extension}"
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        # Return download URL (in production, use signed URLs)
        download_url = f"/api/risk-assessment/download/{file_id}.{file_extension}"
        
        logger.info(
            "Assessment exported",
            user_id=current_user.get("id"),
            format=request.format,
            file_size=len(file_content)
        )
        
        return {
            "download_url": download_url,
            "file_size": len(file_content),
            "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Export failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Export failed"
        )


@router.get("/download/{file_name}")
async def download_file(
    file_name: str,
    current_user = Depends(get_current_user)
):
    """
    Download exported risk assessment file.
    """
    try:
        file_path = f"/tmp/risk_assessment_{file_name}"
        
        # Check if file exists
        import os
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found or expired"
            )
        
        # Determine content type
        if file_name.endswith('.pdf'):
            content_type = "application/pdf"
        elif file_name.endswith('.json'):
            content_type = "application/json"
        elif file_name.endswith('.csv'):
            content_type = "text/csv"
        else:
            content_type = "application/octet-stream"
        
        # Stream file
        def generate_file():
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    yield chunk
        
        return StreamingResponse(
            generate_file(),
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={file_name}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("File download failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File download failed"
        )


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """
    Health check for risk assessment services.
    """
    try:
        # Check all services
        services_status = {}
        
        # Risk Assessment Service
        try:
            risk_service = await get_risk_assessment_service()
            risk_health = await risk_service.health_check()
            services_status["risk_assessment"] = risk_health
        except Exception as e:
            services_status["risk_assessment"] = {"status": "unhealthy", "error": str(e)}
        
        # Pattern Recognition Service
        try:
            pattern_service = await get_pattern_recognition_service()
            pattern_health = await pattern_service.health_check()
            services_status["pattern_recognition"] = pattern_health
        except Exception as e:
            services_status["pattern_recognition"] = {"status": "unhealthy", "error": str(e)}
        
        # Risk Scoring Algorithm
        try:
            scoring_service = await get_risk_scoring_algorithm()
            scoring_health = await scoring_service.health_check()
            services_status["risk_scoring"] = scoring_health
        except Exception as e:
            services_status["risk_scoring"] = {"status": "unhealthy", "error": str(e)}
        
        # Overall status
        overall_status = "healthy"
        if any(service.get("status") != "healthy" for service in services_status.values()):
            overall_status = "degraded"
        
        return HealthCheckResponse(
            status=overall_status,
            services=services_status
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return HealthCheckResponse(
            status="unhealthy",
            services={"error": str(e)}
        )


# Helper functions
async def _log_assessment_usage(
    user_id: str,
    project_id: Optional[str],
    risk_score: float,
    processing_time: float
) -> None:
    """Log assessment usage for analytics."""
    try:
        neo4j_conn = await get_neo4j()
        
        log_query = """
        CREATE (u:AssessmentUsage {
            id: randomUUID(),
            user_id: $user_id,
            project_id: $project_id,
            risk_score: $risk_score,
            processing_time: $processing_time,
            created_at: datetime()
        })
        """
        
        await neo4j_conn.execute_write(
            log_query,
            {
                "user_id": user_id,
                "project_id": project_id,
                "risk_score": risk_score,
                "processing_time": processing_time
            }
        )
        
    except Exception as e:
        logger.warning("Failed to log assessment usage", error=str(e))


async def _generate_pdf_report(
    assessment_data: Dict[str, Any],
    include_charts: bool = True
) -> bytes:
    """Generate PDF report from assessment data."""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        import io
        
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        
        # Title
        p.setFont("Helvetica-Bold", 16)
        p.drawString(100, 750, "Risk Assessment Report")
        
        # Risk Score
        p.setFont("Helvetica", 12)
        risk_score = assessment_data.get("overall_risk_score", 0)
        p.drawString(100, 720, f"Overall Risk Score: {risk_score:.2%}")
        
        # Risk Level
        risk_level = assessment_data.get("risk_level", "Unknown")
        p.drawString(100, 700, f"Risk Level: {risk_level}")
        
        # Confidence
        confidence = assessment_data.get("confidence", 0)
        p.drawString(100, 680, f"Assessment Confidence: {confidence:.2%}")
        
        # Risk Factors
        p.setFont("Helvetica-Bold", 14)
        p.drawString(100, 650, "Risk Factors:")
        
        y_pos = 630
        risk_factors = assessment_data.get("risk_factors", [])
        for i, factor in enumerate(risk_factors[:10]):  # Limit to first 10
            p.setFont("Helvetica", 10)
            factor_text = f"{i+1}. {factor.get('name', 'Unknown')} ({factor.get('level', 'Unknown')})"
            p.drawString(120, y_pos, factor_text)
            y_pos -= 20
            
            if y_pos < 100:  # Start new page if needed
                p.showPage()
                y_pos = 750
        
        # Actionable Insights
        insights = assessment_data.get("actionable_insights", [])
        if insights:
            p.setFont("Helvetica-Bold", 14)
            p.drawString(100, y_pos - 20, "Key Recommendations:")
            y_pos -= 40
            
            for insight in insights:
                p.setFont("Helvetica", 10)
                p.drawString(120, y_pos, f"• {insight}")
                y_pos -= 15
        
        p.save()
        buffer.seek(0)
        return buffer.getvalue()
        
    except ImportError:
        # Fallback to simple text format if reportlab is not available
        content = f"""Risk Assessment Report
========================

Overall Risk Score: {assessment_data.get('overall_risk_score', 0):.2%}
Risk Level: {assessment_data.get('risk_level', 'Unknown')}
Confidence: {assessment_data.get('confidence', 0):.2%}

Risk Factors:
{chr(10).join([f"- {f.get('name', 'Unknown')}" for f in assessment_data.get('risk_factors', [])])}

Recommendations:
{chr(10).join([f"- {insight}" for insight in assessment_data.get('actionable_insights', [])])}
"""
        return content.encode('utf-8')


async def _generate_csv_report(assessment_data: Dict[str, Any]) -> bytes:
    """Generate CSV report from assessment data."""
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(["Risk Assessment Report"])
    writer.writerow([])
    
    # Overall metrics
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Overall Risk Score", f"{assessment_data.get('overall_risk_score', 0):.2%}"])
    writer.writerow(["Risk Level", assessment_data.get('risk_level', 'Unknown')])
    writer.writerow(["Confidence", f"{assessment_data.get('confidence', 0):.2%}"])
    writer.writerow([])
    
    # Risk factors
    writer.writerow(["Risk Factors"])
    writer.writerow(["Name", "Category", "Level", "Score", "Probability", "Impact"])
    
    for factor in assessment_data.get('risk_factors', []):
        writer.writerow([
            factor.get('name', ''),
            factor.get('category', ''),
            factor.get('level', ''),
            f"{factor.get('risk_score', 0):.2f}",
            f"{factor.get('probability', 0):.2f}",
            f"{factor.get('impact', 0):.2f}"
        ])
    
    # Insights
    writer.writerow([])
    writer.writerow(["Actionable Insights"])
    for insight in assessment_data.get('actionable_insights', []):
        writer.writerow([insight])
    
    return output.getvalue().encode('utf-8')