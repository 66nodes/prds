"""
Risk Assessment and Historical Analysis Service

Provides risk scoring and lessons learned based on historical project data.
Integrates with Neo4j for historical project queries and pattern recognition.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

from core.config import get_settings
from core.database import get_neo4j
from .cache_service import get_cache_service, CacheNamespace, cached
from .graphrag.graph_service import GraphRAGService

logger = structlog.get_logger(__name__)
settings = get_settings()


class RiskLevel(str, Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(str, Enum):
    """Risk category enumeration."""
    TECHNICAL = "technical"
    SCHEDULE = "schedule"
    BUDGET = "budget"
    SCOPE = "scope"
    QUALITY = "quality"
    TEAM = "team"
    EXTERNAL = "external"


class RiskFactor(BaseModel):
    """Individual risk factor model."""
    id: str = Field(..., description="Risk factor ID")
    category: RiskCategory = Field(..., description="Risk category")
    name: str = Field(..., description="Risk factor name")
    description: str = Field(..., description="Risk factor description")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of occurrence")
    impact: float = Field(..., ge=0.0, le=1.0, description="Impact severity")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Calculated risk score")
    level: RiskLevel = Field(..., description="Risk level")
    mitigation_strategies: List[str] = Field(default_factory=list, description="Suggested mitigations")
    historical_frequency: float = Field(default=0.0, description="Historical occurrence frequency")


class HistoricalPattern(BaseModel):
    """Historical pattern model."""
    pattern_id: str = Field(..., description="Pattern ID")
    pattern_type: str = Field(..., description="Pattern type")
    description: str = Field(..., description="Pattern description")
    frequency: float = Field(..., description="Pattern frequency")
    success_rate: float = Field(..., description="Success rate when pattern present")
    projects_count: int = Field(..., description="Number of projects with this pattern")
    template_suggestions: List[str] = Field(default_factory=list, description="Template suggestions")


class ProjectTemplate(BaseModel):
    """Project template model."""
    template_id: str = Field(..., description="Template ID")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    category: str = Field(..., description="Template category")
    success_rate: float = Field(..., description="Historical success rate")
    risk_reduction: float = Field(..., description="Risk reduction factor")
    applicable_scenarios: List[str] = Field(default_factory=list, description="Applicable scenarios")
    content: Dict[str, Any] = Field(default_factory=dict, description="Template content")


class RiskAssessmentResult(BaseModel):
    """Risk assessment result model."""
    project_id: str = Field(..., description="Project ID")
    overall_risk_score: float = Field(..., ge=0.0, le=1.0, description="Overall risk score")
    risk_level: RiskLevel = Field(..., description="Overall risk level")
    risk_factors: List[RiskFactor] = Field(default_factory=list, description="Identified risk factors")
    historical_patterns: List[HistoricalPattern] = Field(default_factory=list, description="Relevant patterns")
    recommended_templates: List[ProjectTemplate] = Field(default_factory=list, description="Recommended templates")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Assessment confidence")
    assessment_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Assessment timestamp")
    actionable_insights: List[str] = Field(default_factory=list, description="Actionable insights")


class LessonsLearned(BaseModel):
    """Lessons learned model."""
    lesson_id: str = Field(..., description="Lesson ID")
    category: str = Field(..., description="Lesson category")
    title: str = Field(..., description="Lesson title")
    description: str = Field(..., description="Lesson description")
    impact: str = Field(..., description="Impact description")
    recommendation: str = Field(..., description="Recommendation")
    source_projects: List[str] = Field(default_factory=list, description="Source project IDs")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Lesson confidence")
    frequency: int = Field(default=1, description="Occurrence frequency")


class RiskAssessmentService:
    """Risk Assessment and Historical Analysis Service."""
    
    def __init__(self):
        self.neo4j_conn = None
        self.graphrag_service = None
        self._cache_service = get_cache_service()
        self.is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the risk assessment service."""
        try:
            # Get Neo4j connection
            self.neo4j_conn = await get_neo4j()
            
            # Initialize GraphRAG service for pattern analysis
            self.graphrag_service = GraphRAGService()
            if not self.graphrag_service.is_initialized:
                await self.graphrag_service.initialize()
            
            # Create risk assessment indexes if they don't exist
            await self._create_risk_indexes()
            
            self.is_initialized = True
            logger.info("Risk assessment service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize risk assessment service", error=str(e))
            raise
    
    async def _create_risk_indexes(self) -> None:
        """Create necessary Neo4j indexes for risk assessment."""
        try:
            indexes = [
                "CREATE INDEX risk_project_idx IF NOT EXISTS FOR (p:Project) ON (p.id, p.status, p.created_at)",
                "CREATE INDEX risk_factor_idx IF NOT EXISTS FOR (r:RiskFactor) ON (r.category, r.level)",
                "CREATE INDEX pattern_idx IF NOT EXISTS FOR (p:Pattern) ON (p.type, p.frequency)",
                "CREATE INDEX template_idx IF NOT EXISTS FOR (t:Template) ON (t.category, t.success_rate)",
                "CREATE INDEX lesson_idx IF NOT EXISTS FOR (l:Lesson) ON (l.category, l.confidence)"
            ]
            
            for index_query in indexes:
                await self.neo4j_conn.execute_write(index_query)
                
            logger.info("Risk assessment indexes created successfully")
            
        except Exception as e:
            logger.warning("Failed to create some indexes", error=str(e))
    
    @cached(CacheNamespace.RISK_ASSESSMENT, ttl=3600)
    async def assess_project_risks(
        self,
        project_description: str,
        project_category: str = None,
        context: Dict[str, Any] = None
    ) -> RiskAssessmentResult:
        """
        Perform comprehensive risk assessment for a project.
        
        Args:
            project_description: Project description text
            project_category: Project category (optional)
            context: Additional context information
            
        Returns:
            RiskAssessmentResult with comprehensive risk analysis
        """
        if not self.is_initialized:
            raise RuntimeError("Risk assessment service not initialized")
        
        if not context:
            context = {}
            
        project_id = context.get("project_id", str(uuid4()))
        
        try:
            logger.info("Starting risk assessment", project_id=project_id)
            
            # Run risk analysis components in parallel
            tasks = [
                self._analyze_technical_risks(project_description, context),
                self._analyze_schedule_risks(project_description, context),
                self._analyze_scope_risks(project_description, context),
                self._identify_historical_patterns(project_description, project_category),
                self._recommend_templates(project_description, project_category)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            technical_risks = results[0] if not isinstance(results[0], Exception) else []
            schedule_risks = results[1] if not isinstance(results[1], Exception) else []
            scope_risks = results[2] if not isinstance(results[2], Exception) else []
            patterns = results[3] if not isinstance(results[3], Exception) else []
            templates = results[4] if not isinstance(results[4], Exception) else []
            
            # Combine all risk factors
            all_risks = technical_risks + schedule_risks + scope_risks
            
            # Calculate overall risk score
            overall_score = self._calculate_overall_risk_score(all_risks)
            risk_level = self._determine_risk_level(overall_score)
            
            # Generate actionable insights
            insights = await self._generate_actionable_insights(
                all_risks, patterns, templates, project_description
            )
            
            # Calculate confidence based on historical data availability
            confidence = await self._calculate_assessment_confidence(
                project_description, patterns, len(all_risks)
            )
            
            result = RiskAssessmentResult(
                project_id=project_id,
                overall_risk_score=overall_score,
                risk_level=risk_level,
                risk_factors=all_risks,
                historical_patterns=patterns,
                recommended_templates=templates,
                confidence=confidence,
                actionable_insights=insights
            )
            
            # Store assessment result for future learning
            await self._store_assessment_result(result)
            
            logger.info(
                "Risk assessment completed",
                project_id=project_id,
                risk_score=overall_score,
                risk_level=risk_level.value,
                factors_count=len(all_risks)
            )
            
            return result
            
        except Exception as e:
            logger.error("Risk assessment failed", project_id=project_id, error=str(e))
            raise
    
    async def _analyze_technical_risks(
        self, 
        project_description: str, 
        context: Dict[str, Any]
    ) -> List[RiskFactor]:
        """Analyze technical risks based on project description and historical data."""
        try:
            # Query for similar technical challenges from historical projects
            tech_risk_query = """
            MATCH (p:Project)-[:HAS_RISK]->(r:RiskFactor)
            WHERE r.category = 'technical' 
            AND p.status IN ['completed', 'failed']
            AND any(keyword IN split(toLower($description), ' ') WHERE 
                toLower(p.description) CONTAINS keyword OR
                toLower(r.description) CONTAINS keyword)
            WITH r, COUNT(p) as frequency, AVG(r.impact) as avg_impact
            WHERE frequency >= 2
            RETURN r.name as name, r.description as description,
                   AVG(r.probability) as avg_probability, avg_impact,
                   frequency, collect(DISTINCT r.mitigation_strategies) as mitigations
            ORDER BY frequency DESC, avg_impact DESC
            LIMIT 10
            """
            
            tech_risks_data = await self.neo4j_conn.execute_query(
                tech_risk_query, {"description": project_description}
            )
            
            # Convert to RiskFactor objects
            risk_factors = []
            for risk_data in tech_risks_data:
                probability = min(risk_data["avg_probability"], 1.0)
                impact = min(risk_data["avg_impact"], 1.0)
                risk_score = probability * impact
                
                # Flatten mitigation strategies
                mitigations = []
                for mitigation_list in risk_data["mitigations"]:
                    if isinstance(mitigation_list, list):
                        mitigations.extend(mitigation_list)
                    else:
                        mitigations.append(str(mitigation_list))
                
                risk_factor = RiskFactor(
                    id=str(uuid4()),
                    category=RiskCategory.TECHNICAL,
                    name=risk_data["name"],
                    description=risk_data["description"],
                    probability=probability,
                    impact=impact,
                    risk_score=risk_score,
                    level=self._determine_risk_level(risk_score),
                    mitigation_strategies=mitigations[:5],  # Limit to top 5
                    historical_frequency=risk_data["frequency"] / 100.0  # Normalize
                )
                risk_factors.append(risk_factor)
            
            # Add common technical risks if not found in historical data
            if len(risk_factors) < 3:
                common_risks = await self._get_common_technical_risks(project_description)
                risk_factors.extend(common_risks)
            
            return risk_factors[:10]  # Return top 10 risks
            
        except Exception as e:
            logger.warning("Technical risk analysis failed", error=str(e))
            return []
    
    async def _analyze_schedule_risks(
        self, 
        project_description: str, 
        context: Dict[str, Any]
    ) -> List[RiskFactor]:
        """Analyze schedule-related risks."""
        try:
            # Query for schedule-related issues from historical projects
            schedule_risk_query = """
            MATCH (p:Project)-[:HAS_RISK]->(r:RiskFactor)
            WHERE r.category = 'schedule'
            AND p.status IN ['completed', 'failed']
            WITH r, COUNT(p) as frequency, AVG(r.impact) as avg_impact
            WHERE frequency >= 2
            RETURN r.name as name, r.description as description,
                   AVG(r.probability) as avg_probability, avg_impact,
                   frequency, collect(DISTINCT r.mitigation_strategies) as mitigations
            ORDER BY frequency DESC, avg_impact DESC
            LIMIT 5
            """
            
            schedule_risks_data = await self.neo4j_conn.execute_query(schedule_risk_query)
            
            risk_factors = []
            for risk_data in schedule_risks_data:
                probability = min(risk_data["avg_probability"], 1.0)
                impact = min(risk_data["avg_impact"], 1.0)
                risk_score = probability * impact
                
                mitigations = []
                for mitigation_list in risk_data["mitigations"]:
                    if isinstance(mitigation_list, list):
                        mitigations.extend(mitigation_list)
                    else:
                        mitigations.append(str(mitigation_list))
                
                risk_factor = RiskFactor(
                    id=str(uuid4()),
                    category=RiskCategory.SCHEDULE,
                    name=risk_data["name"],
                    description=risk_data["description"],
                    probability=probability,
                    impact=impact,
                    risk_score=risk_score,
                    level=self._determine_risk_level(risk_score),
                    mitigation_strategies=mitigations[:3],
                    historical_frequency=risk_data["frequency"] / 100.0
                )
                risk_factors.append(risk_factor)
            
            return risk_factors
            
        except Exception as e:
            logger.warning("Schedule risk analysis failed", error=str(e))
            return []
    
    async def _analyze_scope_risks(
        self, 
        project_description: str, 
        context: Dict[str, Any]
    ) -> List[RiskFactor]:
        """Analyze scope-related risks."""
        try:
            # Analyze scope clarity and complexity indicators
            scope_indicators = await self._analyze_scope_clarity(project_description)
            
            risk_factors = []
            
            # Scope creep risk
            if scope_indicators.get("clarity_score", 0.5) < 0.7:
                risk_factors.append(RiskFactor(
                    id=str(uuid4()),
                    category=RiskCategory.SCOPE,
                    name="Scope Creep",
                    description="Unclear requirements may lead to scope expansion",
                    probability=0.6,
                    impact=0.7,
                    risk_score=0.42,
                    level=RiskLevel.MEDIUM,
                    mitigation_strategies=[
                        "Define clear acceptance criteria",
                        "Implement change control process",
                        "Regular stakeholder reviews"
                    ]
                ))
            
            # Complexity risk
            if scope_indicators.get("complexity_score", 0.5) > 0.8:
                risk_factors.append(RiskFactor(
                    id=str(uuid4()),
                    category=RiskCategory.SCOPE,
                    name="High Complexity",
                    description="Project complexity may lead to underestimation",
                    probability=0.7,
                    impact=0.8,
                    risk_score=0.56,
                    level=RiskLevel.HIGH,
                    mitigation_strategies=[
                        "Break down into smaller phases",
                        "Use iterative development",
                        "Increase buffer time"
                    ]
                ))
            
            return risk_factors
            
        except Exception as e:
            logger.warning("Scope risk analysis failed", error=str(e))
            return []
    
    async def _analyze_scope_clarity(self, project_description: str) -> Dict[str, float]:
        """Analyze scope clarity indicators."""
        # Simple heuristic-based analysis
        # In production, this would use more sophisticated NLP
        
        clarity_indicators = {
            "specific_terms": ["must", "shall", "will", "should"],
            "vague_terms": ["might", "could", "may", "possibly", "approximately"],
            "measurable_terms": ["increase", "decrease", "reduce", "improve", "achieve"],
            "complexity_terms": ["integrate", "complex", "multiple", "various", "across"]
        }
        
        description_lower = project_description.lower()
        
        specific_count = sum(1 for term in clarity_indicators["specific_terms"] 
                           if term in description_lower)
        vague_count = sum(1 for term in clarity_indicators["vague_terms"] 
                         if term in description_lower)
        measurable_count = sum(1 for term in clarity_indicators["measurable_terms"] 
                              if term in description_lower)
        complexity_count = sum(1 for term in clarity_indicators["complexity_terms"] 
                              if term in description_lower)
        
        # Calculate clarity score (0-1, higher is clearer)
        clarity_score = min((specific_count + measurable_count) / max(vague_count + 2, 1), 1.0)
        
        # Calculate complexity score (0-1, higher is more complex)
        complexity_score = min(complexity_count / 10.0, 1.0)
        
        return {
            "clarity_score": clarity_score,
            "complexity_score": complexity_score,
            "specific_count": specific_count,
            "vague_count": vague_count,
            "measurable_count": measurable_count,
            "complexity_count": complexity_count
        }
    
    async def _get_common_technical_risks(self, project_description: str) -> List[RiskFactor]:
        """Get common technical risks when historical data is limited."""
        common_risks = []
        
        # Integration complexity
        if any(term in project_description.lower() for term in ["integrate", "api", "system", "third-party"]):
            common_risks.append(RiskFactor(
                id=str(uuid4()),
                category=RiskCategory.TECHNICAL,
                name="Integration Complexity",
                description="Integration with external systems may pose technical challenges",
                probability=0.5,
                impact=0.6,
                risk_score=0.3,
                level=RiskLevel.MEDIUM,
                mitigation_strategies=[
                    "Early integration testing",
                    "API documentation review",
                    "Fallback mechanisms"
                ],
                historical_frequency=0.4
            ))
        
        # Performance requirements
        if any(term in project_description.lower() for term in ["performance", "scale", "load", "speed"]):
            common_risks.append(RiskFactor(
                id=str(uuid4()),
                category=RiskCategory.TECHNICAL,
                name="Performance Requirements",
                description="Meeting performance requirements may be challenging",
                probability=0.4,
                impact=0.7,
                risk_score=0.28,
                level=RiskLevel.MEDIUM,
                mitigation_strategies=[
                    "Performance testing early",
                    "Architecture review",
                    "Monitoring implementation"
                ],
                historical_frequency=0.3
            ))
        
        return common_risks
    
    async def _identify_historical_patterns(
        self, 
        project_description: str, 
        project_category: str = None
    ) -> List[HistoricalPattern]:
        """Identify relevant historical patterns."""
        try:
            # Query for patterns in similar projects
            pattern_query = """
            MATCH (p:Project)-[:HAS_PATTERN]->(pat:Pattern)
            WHERE ($category IS NULL OR p.category = $category)
            AND any(keyword IN split(toLower($description), ' ') WHERE 
                toLower(p.description) CONTAINS keyword OR
                toLower(pat.description) CONTAINS keyword)
            WITH pat, COUNT(p) as frequency, 
                 AVG(CASE WHEN p.status = 'completed' THEN 1.0 ELSE 0.0 END) as success_rate
            WHERE frequency >= 2
            RETURN pat.id as pattern_id, pat.type as pattern_type,
                   pat.description as description, frequency,
                   success_rate, pat.template_suggestions as suggestions
            ORDER BY frequency DESC, success_rate DESC
            LIMIT 5
            """
            
            patterns_data = await self.neo4j_conn.execute_query(
                pattern_query, 
                {"description": project_description, "category": project_category}
            )
            
            patterns = []
            for pattern_data in patterns_data:
                pattern = HistoricalPattern(
                    pattern_id=pattern_data["pattern_id"],
                    pattern_type=pattern_data["pattern_type"],
                    description=pattern_data["description"],
                    frequency=pattern_data["frequency"] / 100.0,  # Normalize
                    success_rate=pattern_data["success_rate"],
                    projects_count=pattern_data["frequency"],
                    template_suggestions=pattern_data.get("suggestions", [])
                )
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.warning("Historical pattern identification failed", error=str(e))
            return []
    
    async def _recommend_templates(
        self, 
        project_description: str, 
        project_category: str = None
    ) -> List[ProjectTemplate]:
        """Recommend project templates based on historical success."""
        try:
            # Query for successful templates
            template_query = """
            MATCH (t:Template)
            WHERE ($category IS NULL OR t.category = $category OR t.category = 'general')
            AND any(keyword IN split(toLower($description), ' ') WHERE 
                any(scenario IN t.applicable_scenarios WHERE 
                    toLower(scenario) CONTAINS keyword))
            RETURN t.id as template_id, t.name as name, t.description as description,
                   t.category as category, t.success_rate as success_rate,
                   t.risk_reduction as risk_reduction, 
                   t.applicable_scenarios as scenarios,
                   t.content as content
            ORDER BY t.success_rate DESC, t.risk_reduction DESC
            LIMIT 3
            """
            
            templates_data = await self.neo4j_conn.execute_query(
                template_query, 
                {"description": project_description, "category": project_category}
            )
            
            templates = []
            for template_data in templates_data:
                template = ProjectTemplate(
                    template_id=template_data["template_id"],
                    name=template_data["name"],
                    description=template_data["description"],
                    category=template_data["category"],
                    success_rate=template_data["success_rate"],
                    risk_reduction=template_data["risk_reduction"],
                    applicable_scenarios=template_data.get("scenarios", []),
                    content=template_data.get("content", {})
                )
                templates.append(template)
            
            return templates
            
        except Exception as e:
            logger.warning("Template recommendation failed", error=str(e))
            return []
    
    async def _generate_actionable_insights(
        self,
        risks: List[RiskFactor],
        patterns: List[HistoricalPattern],
        templates: List[ProjectTemplate],
        project_description: str
    ) -> List[str]:
        """Generate actionable insights based on risk analysis."""
        insights = []
        
        # Risk-based insights
        high_risks = [r for r in risks if r.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        if high_risks:
            insights.append(f"Address {len(high_risks)} high-priority risks before project start")
            
            # Get most critical risk
            critical_risk = max(high_risks, key=lambda r: r.risk_score)
            insights.append(f"Primary concern: {critical_risk.name} - {critical_risk.mitigation_strategies[0] if critical_risk.mitigation_strategies else 'requires attention'}")
        
        # Pattern-based insights
        successful_patterns = [p for p in patterns if p.success_rate > 0.7]
        if successful_patterns:
            best_pattern = max(successful_patterns, key=lambda p: p.success_rate)
            insights.append(f"Consider following {best_pattern.pattern_type} approach (success rate: {best_pattern.success_rate:.0%})")
        
        # Template-based insights
        if templates:
            best_template = max(templates, key=lambda t: t.success_rate)
            insights.append(f"Use {best_template.name} template to reduce risks by {best_template.risk_reduction:.0%}")
        
        # General insights
        total_risk_score = sum(r.risk_score for r in risks) / len(risks) if risks else 0
        if total_risk_score > 0.6:
            insights.append("Consider breaking project into smaller phases to reduce overall risk")
        elif total_risk_score < 0.3:
            insights.append("Project shows low risk profile - good opportunity for aggressive timeline")
        
        return insights[:5]  # Limit to top 5 insights
    
    async def _calculate_assessment_confidence(
        self, 
        project_description: str, 
        patterns: List[HistoricalPattern],
        risks_count: int
    ) -> float:
        """Calculate confidence in the risk assessment."""
        # Base confidence factors
        description_length_factor = min(len(project_description) / 500.0, 1.0)
        patterns_factor = min(len(patterns) / 5.0, 1.0)
        risks_factor = min(risks_count / 10.0, 1.0)
        
        # Historical data availability
        historical_factor = sum(p.frequency for p in patterns) / len(patterns) if patterns else 0.3
        
        # Calculate weighted confidence
        confidence = (
            description_length_factor * 0.2 +
            patterns_factor * 0.3 +
            risks_factor * 0.2 +
            historical_factor * 0.3
        )
        
        return min(confidence, 0.95)  # Cap at 95%
    
    def _calculate_overall_risk_score(self, risks: List[RiskFactor]) -> float:
        """Calculate overall risk score from individual risks."""
        if not risks:
            return 0.0
        
        # Weight risks by category
        category_weights = {
            RiskCategory.TECHNICAL: 1.0,
            RiskCategory.SCHEDULE: 0.9,
            RiskCategory.SCOPE: 0.8,
            RiskCategory.QUALITY: 0.9,
            RiskCategory.BUDGET: 0.7,
            RiskCategory.TEAM: 0.8,
            RiskCategory.EXTERNAL: 0.6
        }
        
        weighted_scores = []
        for risk in risks:
            weight = category_weights.get(risk.category, 0.7)
            weighted_scores.append(risk.risk_score * weight)
        
        # Use root mean square to avoid linear addition
        overall_score = np.sqrt(np.mean([score ** 2 for score in weighted_scores]))
        
        return min(overall_score, 1.0)
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from risk score."""
        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    async def _store_assessment_result(self, result: RiskAssessmentResult) -> None:
        """Store risk assessment result for future learning."""
        try:
            store_query = """
            CREATE (a:RiskAssessment {
                id: $result_id,
                project_id: $project_id,
                overall_risk_score: $overall_score,
                risk_level: $risk_level,
                confidence: $confidence,
                factors_count: $factors_count,
                patterns_count: $patterns_count,
                templates_count: $templates_count,
                created_at: datetime()
            })
            """
            
            await self.neo4j_conn.execute_write(
                store_query,
                {
                    "result_id": str(uuid4()),
                    "project_id": result.project_id,
                    "overall_score": result.overall_risk_score,
                    "risk_level": result.risk_level.value,
                    "confidence": result.confidence,
                    "factors_count": len(result.risk_factors),
                    "patterns_count": len(result.historical_patterns),
                    "templates_count": len(result.recommended_templates)
                }
            )
            
        except Exception as e:
            logger.warning("Failed to store assessment result", error=str(e))
    
    @cached(CacheNamespace.RISK_ASSESSMENT, ttl=7200)
    async def get_lessons_learned(
        self, 
        category: str = None, 
        limit: int = 10
    ) -> List[LessonsLearned]:
        """Get lessons learned from historical projects."""
        try:
            lessons_query = """
            MATCH (l:Lesson)
            WHERE ($category IS NULL OR l.category = $category)
            AND l.confidence >= 0.6
            RETURN l.id as lesson_id, l.category as category,
                   l.title as title, l.description as description,
                   l.impact as impact, l.recommendation as recommendation,
                   l.source_projects as source_projects,
                   l.confidence as confidence, l.frequency as frequency
            ORDER BY l.confidence DESC, l.frequency DESC
            LIMIT $limit
            """
            
            lessons_data = await self.neo4j_conn.execute_query(
                lessons_query, {"category": category, "limit": limit}
            )
            
            lessons = []
            for lesson_data in lessons_data:
                lesson = LessonsLearned(
                    lesson_id=lesson_data["lesson_id"],
                    category=lesson_data["category"],
                    title=lesson_data["title"],
                    description=lesson_data["description"],
                    impact=lesson_data["impact"],
                    recommendation=lesson_data["recommendation"],
                    source_projects=lesson_data.get("source_projects", []),
                    confidence=lesson_data["confidence"],
                    frequency=lesson_data.get("frequency", 1)
                )
                lessons.append(lesson)
            
            return lessons
            
        except Exception as e:
            logger.error("Failed to get lessons learned", error=str(e))
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for risk assessment service."""
        try:
            if not self.is_initialized:
                return {"status": "unhealthy", "error": "Service not initialized"}
            
            # Test database connection
            test_query = "RETURN 'ok' as status"
            await self.neo4j_conn.execute_query(test_query)
            
            return {
                "status": "healthy",
                "initialized": self.is_initialized,
                "components": {
                    "neo4j": self.neo4j_conn.is_connected if self.neo4j_conn else False,
                    "graphrag": self.graphrag_service.is_initialized if self.graphrag_service else False
                }
            }
            
        except Exception as e:
            logger.error("Risk assessment health check failed", error=str(e))
            return {"status": "unhealthy", "error": str(e)}


# Global service instance
_risk_service = None

async def get_risk_assessment_service() -> RiskAssessmentService:
    """Get or create risk assessment service instance."""
    global _risk_service
    if _risk_service is None:
        _risk_service = RiskAssessmentService()
        await _risk_service.initialize()
    return _risk_service