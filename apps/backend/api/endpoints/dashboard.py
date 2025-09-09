"""
Dashboard and analytics endpoints
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from core.config import get_settings
from core.database import get_neo4j
from .auth import get_current_user, User

logger = structlog.get_logger(__name__)
settings = get_settings()

router = APIRouter()


# Response Models
class DashboardMetrics(BaseModel):
    """Dashboard key metrics."""
    active_prds: int = Field(..., description="Number of active PRDs")
    completed_prds: int = Field(..., description="Number of completed PRDs")
    average_quality_score: float = Field(..., description="Average quality score")
    time_saved_hours: float = Field(..., description="Estimated time saved in hours")
    total_users: int = Field(..., description="Total number of users")
    validation_accuracy: float = Field(..., description="GraphRAG validation accuracy")


class PRDSummary(BaseModel):
    """PRD summary for dashboard list."""
    id: str = Field(..., description="PRD ID")
    title: str = Field(..., description="PRD title")
    status: str = Field(..., description="PRD status")
    quality_score: float = Field(..., description="Quality score")
    created_at: str = Field(..., description="Creation date")
    created_by: str = Field(..., description="Creator name")
    sections_count: int = Field(..., description="Number of sections")
    last_updated: str = Field(..., description="Last update date")


class PRDListResponse(BaseModel):
    """PRD list response with pagination."""
    prds: List[PRDSummary] = Field(..., description="List of PRDs")
    total_count: int = Field(..., description="Total number of PRDs")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Page size")
    has_more: bool = Field(..., description="More pages available")


class QualityTrend(BaseModel):
    """Quality score trend data."""
    date: str = Field(..., description="Date")
    average_score: float = Field(..., description="Average quality score")
    prd_count: int = Field(..., description="Number of PRDs created")
    validation_accuracy: float = Field(..., description="Validation accuracy")


class ActivitySummary(BaseModel):
    """User activity summary."""
    user_id: str = Field(..., description="User ID")
    user_name: str = Field(..., description="User name")
    prds_created: int = Field(..., description="PRDs created")
    total_quality_score: float = Field(..., description="Total quality score")
    last_activity: str = Field(..., description="Last activity date")


class ValidationInsights(BaseModel):
    """GraphRAG validation insights."""
    total_validations: int = Field(..., description="Total validations performed")
    average_confidence: float = Field(..., description="Average confidence score")
    hallucination_rate: float = Field(..., description="Detected hallucination rate")
    entity_accuracy: float = Field(..., description="Entity validation accuracy")
    community_accuracy: float = Field(..., description="Community validation accuracy")
    global_accuracy: float = Field(..., description="Global validation accuracy")


class DashboardData(BaseModel):
    """Complete dashboard data."""
    metrics: DashboardMetrics = Field(..., description="Key metrics")
    recent_prds: List[PRDSummary] = Field(..., description="Recent PRDs")
    quality_trends: List[QualityTrend] = Field(..., description="Quality trends over time")
    validation_insights: ValidationInsights = Field(..., description="Validation insights")
    top_users: List[ActivitySummary] = Field(..., description="Most active users")


# Dashboard Endpoints
@router.get("/metrics", response_model=DashboardMetrics)
async def get_dashboard_metrics(
    current_user: User = Depends(get_current_user)
):
    """Get key dashboard metrics."""
    try:
        logger.info("Dashboard metrics requested", user_id=current_user.id)
        
        neo4j_conn = await get_neo4j()
        
        # Get PRD metrics
        prd_metrics_query = """
        MATCH (p:PRD)
        WITH 
            count(CASE WHEN p.status = 'active' THEN 1 END) as active_prds,
            count(CASE WHEN p.status = 'completed' THEN 1 END) as completed_prds,
            avg(p.quality_score) as avg_quality,
            count(p) as total_prds
        
        MATCH (u:User)
        WITH active_prds, completed_prds, avg_quality, total_prds, count(u) as total_users
        
        MATCH (v:ValidationResult)
        WHERE v.created_at >= datetime() - duration('P30D')
        WITH active_prds, completed_prds, avg_quality, total_prds, total_users,
             avg(v.confidence) as validation_accuracy
        
        RETURN active_prds, completed_prds, avg_quality, total_prds, 
               total_users, validation_accuracy
        """
        
        result = await neo4j_conn.execute_query(prd_metrics_query)
        metrics = result[0] if result else {}
        
        # Calculate time saved (assuming 2 weeks manual vs 2 hours AI-assisted)
        manual_hours_per_prd = 80  # 2 weeks * 40 hours
        ai_hours_per_prd = 2       # 2 hours with AI
        total_prds = metrics.get("total_prds", 0)
        time_saved = total_prds * (manual_hours_per_prd - ai_hours_per_prd)
        
        dashboard_metrics = DashboardMetrics(
            active_prds=metrics.get("active_prds", 0),
            completed_prds=metrics.get("completed_prds", 0),
            average_quality_score=round(metrics.get("avg_quality", 0.0), 2),
            time_saved_hours=time_saved,
            total_users=metrics.get("total_users", 0),
            validation_accuracy=round(metrics.get("validation_accuracy", 0.0), 3)
        )
        
        logger.info(
            "Dashboard metrics retrieved",
            user_id=current_user.id,
            active_prds=dashboard_metrics.active_prds,
            avg_quality=dashboard_metrics.average_quality_score
        )
        
        return dashboard_metrics
        
    except Exception as e:
        logger.error("Dashboard metrics failed", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/prds", response_model=PRDListResponse)
async def get_prds_list(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    status: Optional[str] = Query(None, description="Filter by status"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    current_user: User = Depends(get_current_user)
):
    """Get paginated list of PRDs."""
    try:
        logger.info(
            "PRD list requested",
            user_id=current_user.id,
            page=page,
            page_size=page_size,
            status=status
        )
        
        neo4j_conn = await get_neo4j()
        
        # Build query with filters
        where_clause = ""
        if status:
            where_clause = f"WHERE p.status = '{status}'"
        
        # Count total
        count_query = f"""
        MATCH (p:PRD)
        {where_clause}
        RETURN count(p) as total
        """
        
        total_result = await neo4j_conn.execute_query(count_query)
        total_count = total_result[0]["total"] if total_result else 0
        
        # Get paginated PRDs
        skip = (page - 1) * page_size
        
        prds_query = f"""
        MATCH (p:PRD)-[:CONTAINS]->(s:Section)
        MATCH (p)<-[:CREATED]-(u:User)
        {where_clause}
        WITH p, u, count(s) as sections_count
        ORDER BY p.{sort_by} {'DESC' if sort_order == 'desc' else 'ASC'}
        SKIP {skip} LIMIT {page_size}
        RETURN p.id as id, p.title as title, p.status as status,
               p.quality_score as quality_score, p.created_at as created_at,
               p.updated_at as updated_at, u.full_name as created_by,
               sections_count
        """
        
        prds_result = await neo4j_conn.execute_query(prds_query)
        
        # Convert to response models
        prds = []
        for prd in prds_result:
            prds.append(PRDSummary(
                id=prd["id"],
                title=prd["title"],
                status=prd["status"],
                quality_score=round(prd["quality_score"], 2),
                created_at=prd["created_at"].isoformat() if prd.get("created_at") else "",
                created_by=prd.get("created_by", "Unknown"),
                sections_count=prd["sections_count"],
                last_updated=prd["updated_at"].isoformat() if prd.get("updated_at") else prd["created_at"].isoformat()
            ))
        
        has_more = (page * page_size) < total_count
        
        logger.info(
            "PRD list retrieved",
            user_id=current_user.id,
            total_count=total_count,
            returned_count=len(prds)
        )
        
        return PRDListResponse(
            prds=prds,
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_more=has_more
        )
        
    except Exception as e:
        logger.error("PRD list retrieval failed", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail=f"Failed to get PRDs: {str(e)}")


@router.get("/quality-trends", response_model=List[QualityTrend])
async def get_quality_trends(
    days: int = Query(30, ge=7, le=365, description="Number of days to analyze"),
    current_user: User = Depends(get_current_user)
):
    """Get quality score trends over time."""
    try:
        logger.info("Quality trends requested", user_id=current_user.id, days=days)
        
        neo4j_conn = await get_neo4j()
        
        trends_query = """
        MATCH (p:PRD)
        WHERE p.created_at >= datetime() - duration($period)
        WITH date(p.created_at) as creation_date, p.quality_score as quality_score
        
        OPTIONAL MATCH (v:ValidationResult)
        WHERE date(v.created_at) = creation_date
        
        WITH creation_date, 
             avg(quality_score) as avg_score,
             count(quality_score) as prd_count,
             avg(v.confidence) as validation_accuracy
        ORDER BY creation_date
        
        RETURN creation_date, avg_score, prd_count, validation_accuracy
        """
        
        period = f"P{days}D"
        result = await neo4j_conn.execute_query(trends_query, {"period": period})
        
        trends = []
        for row in result:
            trends.append(QualityTrend(
                date=row["creation_date"].isoformat() if row.get("creation_date") else "",
                average_score=round(row.get("avg_score", 0.0), 2),
                prd_count=row.get("prd_count", 0),
                validation_accuracy=round(row.get("validation_accuracy", 0.0), 3)
            ))
        
        logger.info(
            "Quality trends retrieved",
            user_id=current_user.id,
            data_points=len(trends)
        )
        
        return trends
        
    except Exception as e:
        logger.error("Quality trends failed", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail=f"Failed to get quality trends: {str(e)}")


@router.get("/validation-insights", response_model=ValidationInsights)
async def get_validation_insights(
    days: int = Query(30, ge=7, le=365, description="Analysis period in days"),
    current_user: User = Depends(get_current_user)
):
    """Get GraphRAG validation insights."""
    try:
        logger.info("Validation insights requested", user_id=current_user.id, days=days)
        
        neo4j_conn = await get_neo4j()
        
        insights_query = """
        MATCH (v:ValidationResult)
        WHERE v.created_at >= datetime() - duration($period)
        
        WITH 
            count(v) as total_validations,
            avg(v.confidence) as avg_confidence,
            avg(v.entity_confidence) as entity_accuracy,
            avg(v.community_confidence) as community_accuracy,
            avg(v.global_confidence) as global_accuracy,
            count(CASE WHEN v.confidence < 0.8 THEN 1 END) as low_confidence
        
        RETURN total_validations, avg_confidence, entity_accuracy,
               community_accuracy, global_accuracy, low_confidence
        """
        
        period = f"P{days}D"
        result = await neo4j_conn.execute_query(insights_query, {"period": period})
        
        if result:
            data = result[0]
            hallucination_rate = (data.get("low_confidence", 0) / data.get("total_validations", 1)) * 100
            
            insights = ValidationInsights(
                total_validations=data.get("total_validations", 0),
                average_confidence=round(data.get("avg_confidence", 0.0), 3),
                hallucination_rate=round(hallucination_rate, 2),
                entity_accuracy=round(data.get("entity_accuracy", 0.0), 3),
                community_accuracy=round(data.get("community_accuracy", 0.0), 3),
                global_accuracy=round(data.get("global_accuracy", 0.0), 3)
            )
        else:
            # Default values if no data
            insights = ValidationInsights(
                total_validations=0,
                average_confidence=0.0,
                hallucination_rate=0.0,
                entity_accuracy=0.0,
                community_accuracy=0.0,
                global_accuracy=0.0
            )
        
        logger.info(
            "Validation insights retrieved",
            user_id=current_user.id,
            total_validations=insights.total_validations,
            hallucination_rate=insights.hallucination_rate
        )
        
        return insights
        
    except Exception as e:
        logger.error("Validation insights failed", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail=f"Failed to get validation insights: {str(e)}")


@router.get("/dashboard", response_model=DashboardData)
async def get_complete_dashboard(
    current_user: User = Depends(get_current_user)
):
    """Get complete dashboard data in a single request."""
    try:
        logger.info("Complete dashboard requested", user_id=current_user.id)
        
        # Fetch all dashboard data in parallel
        import asyncio
        
        metrics_task = get_dashboard_metrics(current_user)
        prds_task = get_prds_list(page=1, page_size=10, current_user=current_user)
        trends_task = get_quality_trends(days=30, current_user=current_user)
        insights_task = get_validation_insights(days=30, current_user=current_user)
        
        metrics, prds_response, trends, insights = await asyncio.gather(
            metrics_task, prds_task, trends_task, insights_task
        )
        
        # Get top users
        neo4j_conn = await get_neo4j()
        
        top_users_query = """
        MATCH (u:User)-[:CREATED]->(p:PRD)
        WITH u, count(p) as prds_created, sum(p.quality_score) as total_quality
        ORDER BY prds_created DESC, total_quality DESC
        LIMIT 5
        RETURN u.id as user_id, u.full_name as user_name,
               prds_created, total_quality,
               max(p.created_at) as last_activity
        """
        
        users_result = await neo4j_conn.execute_query(top_users_query)
        
        top_users = []
        for user in users_result:
            top_users.append(ActivitySummary(
                user_id=user["user_id"],
                user_name=user.get("user_name", "Unknown"),
                prds_created=user["prds_created"],
                total_quality_score=round(user.get("total_quality", 0.0), 2),
                last_activity=user["last_activity"].isoformat() if user.get("last_activity") else ""
            ))
        
        dashboard_data = DashboardData(
            metrics=metrics,
            recent_prds=prds_response.prds,
            quality_trends=trends,
            validation_insights=insights,
            top_users=top_users
        )
        
        logger.info("Complete dashboard data retrieved", user_id=current_user.id)
        
        return dashboard_data
        
    except Exception as e:
        logger.error("Complete dashboard failed", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")