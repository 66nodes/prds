"""
Pattern Recognition Service for Risk Detection and Template Suggestions

Uses machine learning and historical data analysis to identify patterns
that correlate with project success/failure and suggest appropriate templates.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum
import numpy as np
from uuid import uuid4
import re
from collections import Counter, defaultdict

import structlog
from pydantic import BaseModel, Field

from core.config import get_settings
from core.database import get_neo4j
from .cache_service import get_cache_service, CacheNamespace, cached

logger = structlog.get_logger(__name__)
settings = get_settings()


class PatternType(str, Enum):
    """Pattern type enumeration."""
    SUCCESS_FACTOR = "success_factor"
    FAILURE_INDICATOR = "failure_indicator"
    COMPLEXITY_DRIVER = "complexity_driver"
    RISK_CORRELATE = "risk_correlate"
    TEMPLATE_TRIGGER = "template_trigger"


class PatternConfidence(str, Enum):
    """Pattern confidence levels."""
    HIGH = "high"      # >0.8
    MEDIUM = "medium"  # 0.5-0.8
    LOW = "low"        # <0.5


class DetectedPattern(BaseModel):
    """Detected pattern model."""
    pattern_id: str = Field(..., description="Pattern identifier")
    pattern_type: PatternType = Field(..., description="Type of pattern")
    name: str = Field(..., description="Pattern name")
    description: str = Field(..., description="Pattern description")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    confidence_level: PatternConfidence = Field(..., description="Confidence level")
    frequency: float = Field(..., description="Historical frequency")
    success_correlation: float = Field(..., description="Correlation with success")
    risk_impact: float = Field(..., description="Impact on risk score")
    indicators: List[str] = Field(default_factory=list, description="Key indicators")
    template_suggestions: List[str] = Field(default_factory=list, description="Suggested templates")
    mitigation_strategies: List[str] = Field(default_factory=list, description="Mitigation strategies")
    historical_projects: int = Field(default=0, description="Number of historical occurrences")


class TemplateRecommendation(BaseModel):
    """Template recommendation model."""
    template_id: str = Field(..., description="Template identifier")
    name: str = Field(..., description="Template name")
    category: str = Field(..., description="Template category")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Historical success rate")
    risk_reduction: float = Field(..., ge=0.0, le=1.0, description="Risk reduction potential")
    matching_patterns: List[str] = Field(default_factory=list, description="Matching pattern IDs")
    applicable_phases: List[str] = Field(default_factory=list, description="Applicable project phases")
    prerequisites: List[str] = Field(default_factory=list, description="Prerequisites")
    customization_notes: List[str] = Field(default_factory=list, description="Customization suggestions")


class PatternAnalysisResult(BaseModel):
    """Pattern analysis result model."""
    project_description_hash: str = Field(..., description="Hash of analyzed description")
    detected_patterns: List[DetectedPattern] = Field(default_factory=list, description="Detected patterns")
    template_recommendations: List[TemplateRecommendation] = Field(default_factory=list, description="Template recommendations")
    overall_complexity_score: float = Field(..., ge=0.0, le=1.0, description="Overall complexity score")
    success_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted success probability")
    key_insights: List[str] = Field(default_factory=list, description="Key insights")
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall analysis confidence")


class PatternRecognitionService:
    """Advanced pattern recognition service for project analysis."""
    
    def __init__(self):
        self.neo4j_conn = None
        self._cache_service = get_cache_service()
        self.is_initialized = False
        
        # Pattern keywords and indicators
        self._success_keywords = self._load_success_keywords()
        self._failure_keywords = self._load_failure_keywords()
        self._complexity_keywords = self._load_complexity_keywords()
        self._domain_keywords = self._load_domain_keywords()
    
    async def initialize(self) -> None:
        """Initialize the pattern recognition service."""
        try:
            # Get Neo4j connection
            self.neo4j_conn = await get_neo4j()
            
            # Create pattern recognition indexes
            await self._create_pattern_indexes()
            
            # Load and cache pattern models
            await self._load_pattern_models()
            
            self.is_initialized = True
            logger.info("Pattern recognition service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize pattern recognition service", error=str(e))
            raise
    
    async def _create_pattern_indexes(self) -> None:
        """Create Neo4j indexes for pattern recognition."""
        try:
            indexes = [
                "CREATE INDEX pattern_type_idx IF NOT EXISTS FOR (p:Pattern) ON (p.type, p.confidence)",
                "CREATE INDEX template_category_idx IF NOT EXISTS FOR (t:Template) ON (t.category, t.success_rate)",
                "CREATE INDEX project_outcome_idx IF NOT EXISTS FOR (p:Project) ON (p.status, p.success_score)",
                "CREATE INDEX keyword_pattern_idx IF NOT EXISTS FOR (k:Keyword) ON (k.type, k.weight)",
                "CREATE TEXT INDEX project_description_idx IF NOT EXISTS FOR (p:Project) ON (p.description)"
            ]
            
            for index_query in indexes:
                await self.neo4j_conn.execute_write(index_query)
                
        except Exception as e:
            logger.warning("Failed to create pattern indexes", error=str(e))
    
    async def _load_pattern_models(self) -> None:
        """Load pre-trained pattern models from the database."""
        try:
            # Load success/failure patterns
            pattern_query = """
            MATCH (p:Pattern)
            WHERE p.confidence >= 0.5 AND p.enterprise_size >= 10
            RETURN p.id as id, p.type as type, p.name as name,
                   p.description as description, p.confidence as confidence,
                   p.frequency as frequency, p.success_correlation as success_correlation,
                   p.indicators as indicators, p.template_suggestions as template_suggestions
            ORDER BY p.confidence DESC
            """
            
            self._cached_patterns = await self.neo4j_conn.execute_query(pattern_query)
            logger.info(f"Loaded {len(self._cached_patterns)} pattern models")
            
        except Exception as e:
            logger.warning("Failed to load pattern models", error=str(e))
            self._cached_patterns = []
    
    def _load_success_keywords(self) -> Dict[str, float]:
        """Load keywords that correlate with project success."""
        return {
            # Process keywords
            "agile": 0.8, "iterative": 0.7, "incremental": 0.7, "mvp": 0.9,
            "prototype": 0.6, "pilot": 0.7, "phased": 0.8, "modular": 0.8,
            
            # Quality keywords
            "testing": 0.9, "quality": 0.7, "validation": 0.8, "review": 0.7,
            "standards": 0.6, "best-practices": 0.8, "documentation": 0.6,
            
            # Collaboration keywords
            "stakeholder": 0.7, "feedback": 0.8, "collaboration": 0.7,
            "communication": 0.6, "alignment": 0.8, "consensus": 0.7,
            
            # Technical keywords
            "scalable": 0.7, "maintainable": 0.8, "robust": 0.7, "reliable": 0.8,
            "performance": 0.6, "security": 0.8, "architecture": 0.7
        }
    
    def _load_failure_keywords(self) -> Dict[str, float]:
        """Load keywords that correlate with project failure."""
        return {
            # Scope keywords
            "everything": 0.8, "complete": 0.6, "comprehensive": 0.7, "all": 0.5,
            "entire": 0.6, "full": 0.5, "total": 0.6, "revolutionary": 0.9,
            
            # Time pressure keywords
            "urgent": 0.7, "asap": 0.9, "immediately": 0.8, "rush": 0.9,
            "deadline": 0.6, "critical": 0.5, "emergency": 0.8,
            
            # Uncertainty keywords
            "unclear": 0.8, "ambiguous": 0.9, "vague": 0.8, "undefined": 0.9,
            "maybe": 0.6, "possibly": 0.7, "might": 0.6, "could": 0.5,
            
            # Complexity keywords
            "complex": 0.6, "complicated": 0.7, "sophisticated": 0.6,
            "advanced": 0.5, "cutting-edge": 0.7, "innovative": 0.5
        }
    
    def _load_complexity_keywords(self) -> Dict[str, float]:
        """Load keywords that indicate project complexity."""
        return {
            # Integration complexity
            "integrate": 0.7, "connect": 0.5, "synchronize": 0.8, "merge": 0.6,
            "consolidate": 0.7, "unify": 0.6, "interface": 0.6,
            
            # System complexity
            "ecosystem": 0.8, "platform": 0.6, "infrastructure": 0.7,
            "framework": 0.5, "architecture": 0.6, "distributed": 0.8,
            
            # Process complexity
            "workflow": 0.6, "automation": 0.7, "orchestration": 0.8,
            "coordination": 0.7, "synchronization": 0.8,
            
            # Data complexity
            "migration": 0.9, "transformation": 0.8, "analytics": 0.6,
            "intelligence": 0.7, "machine-learning": 0.8, "ai": 0.7
        }
    
    def _load_domain_keywords(self) -> Dict[str, List[str]]:
        """Load domain-specific keywords for template matching."""
        return {
            "web-development": ["frontend", "backend", "api", "web", "browser", "responsive"],
            "mobile-development": ["mobile", "ios", "android", "app", "native", "react-native"],
            "data-science": ["analytics", "data", "insights", "reporting", "dashboard", "visualization"],
            "infrastructure": ["deployment", "devops", "cloud", "aws", "docker", "kubernetes"],
            "integration": ["api", "integration", "webhook", "connector", "sync", "etl"],
            "security": ["authentication", "authorization", "security", "encryption", "compliance"],
            "e-commerce": ["payment", "cart", "checkout", "inventory", "product", "order"],
            "content-management": ["cms", "content", "publishing", "editorial", "workflow"],
            "automation": ["workflow", "automation", "process", "rules", "trigger", "notification"]
        }
    
    @cached(CacheNamespace.PATTERN_RECOGNITION, ttl=1800)
    async def analyze_patterns(
        self, 
        project_description: str, 
        context: Dict[str, Any] = None
    ) -> PatternAnalysisResult:
        """
        Analyze project description to detect patterns and recommend templates.
        
        Args:
            project_description: Project description text
            context: Additional context information
            
        Returns:
            PatternAnalysisResult with detected patterns and recommendations
        """
        if not self.is_initialized:
            raise RuntimeError("Pattern recognition service not initialized")
        
        if not context:
            context = {}
        
        try:
            logger.info("Starting pattern analysis", description_length=len(project_description))
            
            # Generate description hash for caching
            import hashlib
            description_hash = hashlib.md5(project_description.encode()).hexdigest()
            
            # Run pattern analysis components in parallel
            tasks = [
                self._detect_success_failure_patterns(project_description),
                self._detect_complexity_patterns(project_description),
                self._detect_domain_patterns(project_description),
                self._detect_risk_patterns(project_description),
                self._calculate_success_probability(project_description),
                self._recommend_templates(project_description, context)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_failure_patterns = results[0] if not isinstance(results[0], Exception) else []
            complexity_patterns = results[1] if not isinstance(results[1], Exception) else []
            domain_patterns = results[2] if not isinstance(results[2], Exception) else []
            risk_patterns = results[3] if not isinstance(results[3], Exception) else []
            success_probability = results[4] if not isinstance(results[4], Exception) else 0.5
            template_recommendations = results[5] if not isinstance(results[5], Exception) else []
            
            # Combine all detected patterns
            all_patterns = (
                success_failure_patterns + 
                complexity_patterns + 
                domain_patterns + 
                risk_patterns
            )
            
            # Calculate overall complexity score
            complexity_score = self._calculate_complexity_score(all_patterns, project_description)
            
            # Generate insights
            insights = self._generate_insights(all_patterns, template_recommendations, complexity_score)
            
            # Calculate overall confidence
            confidence = self._calculate_analysis_confidence(all_patterns, len(project_description))
            
            result = PatternAnalysisResult(
                project_description_hash=description_hash,
                detected_patterns=all_patterns,
                template_recommendations=template_recommendations,
                overall_complexity_score=complexity_score,
                success_probability=success_probability,
                key_insights=insights,
                confidence=confidence
            )
            
            # Store analysis for learning
            await self._store_pattern_analysis(result, project_description)
            
            logger.info(
                "Pattern analysis completed",
                patterns_count=len(all_patterns),
                templates_count=len(template_recommendations),
                complexity_score=complexity_score,
                success_probability=success_probability
            )
            
            return result
            
        except Exception as e:
            logger.error("Pattern analysis failed", error=str(e))
            raise
    
    async def _detect_success_failure_patterns(self, description: str) -> List[DetectedPattern]:
        """Detect patterns that correlate with success or failure."""
        patterns = []
        description_lower = description.lower()
        
        # Analyze success indicators
        success_score = 0.0
        success_indicators = []
        
        for keyword, weight in self._success_keywords.items():
            if keyword in description_lower:
                success_score += weight
                success_indicators.append(keyword)
        
        if success_indicators:
            patterns.append(DetectedPattern(
                pattern_id=str(uuid4()),
                pattern_type=PatternType.SUCCESS_FACTOR,
                name="Success Indicators Present",
                description=f"Project contains {len(success_indicators)} success-correlated factors",
                confidence=min(success_score / 5.0, 1.0),  # Normalize
                confidence_level=self._get_confidence_level(min(success_score / 5.0, 1.0)),
                frequency=0.7,  # Historical frequency
                success_correlation=0.8,
                risk_impact=-0.3,  # Reduces risk
                indicators=success_indicators,
                template_suggestions=["agile-methodology", "quality-assurance"],
                mitigation_strategies=[],
                historical_projects=100
            ))
        
        # Analyze failure indicators
        failure_score = 0.0
        failure_indicators = []
        
        for keyword, weight in self._failure_keywords.items():
            if keyword in description_lower:
                failure_score += weight
                failure_indicators.append(keyword)
        
        if failure_indicators:
            patterns.append(DetectedPattern(
                pattern_id=str(uuid4()),
                pattern_type=PatternType.FAILURE_INDICATOR,
                name="Failure Risk Indicators",
                description=f"Project contains {len(failure_indicators)} failure-correlated factors",
                confidence=min(failure_score / 5.0, 1.0),
                confidence_level=self._get_confidence_level(min(failure_score / 5.0, 1.0)),
                frequency=0.4,
                success_correlation=-0.6,
                risk_impact=0.4,  # Increases risk
                indicators=failure_indicators,
                template_suggestions=["risk-mitigation", "phased-approach"],
                mitigation_strategies=[
                    "Define clear scope boundaries",
                    "Implement regular checkpoints",
                    "Establish change control process"
                ],
                historical_projects=75
            ))
        
        return patterns
    
    async def _detect_complexity_patterns(self, description: str) -> List[DetectedPattern]:
        """Detect patterns that indicate project complexity."""
        patterns = []
        description_lower = description.lower()
        
        complexity_score = 0.0
        complexity_indicators = []
        
        for keyword, weight in self._complexity_keywords.items():
            if keyword in description_lower:
                complexity_score += weight
                complexity_indicators.append(keyword)
        
        # Additional complexity heuristics
        sentence_count = len(re.findall(r'[.!?]+', description))
        word_count = len(description.split())
        unique_concepts = len(set(re.findall(r'\b\w{4,}\b', description_lower)))
        
        # Normalize heuristics
        length_complexity = min(word_count / 500.0, 1.0)
        concept_complexity = min(unique_concepts / 100.0, 1.0)
        
        total_complexity = (
            complexity_score * 0.5 +
            length_complexity * 0.3 +
            concept_complexity * 0.2
        )
        
        if total_complexity > 0.3:
            patterns.append(DetectedPattern(
                pattern_id=str(uuid4()),
                pattern_type=PatternType.COMPLEXITY_DRIVER,
                name="High Complexity Indicators",
                description=f"Project shows complexity score of {total_complexity:.2f}",
                confidence=min(total_complexity, 1.0),
                confidence_level=self._get_confidence_level(min(total_complexity, 1.0)),
                frequency=0.6,
                success_correlation=-0.4,  # Higher complexity = lower success
                risk_impact=0.5,
                indicators=complexity_indicators,
                template_suggestions=["architecture-design", "phased-implementation"],
                mitigation_strategies=[
                    "Break down into smaller components",
                    "Use proven architectural patterns",
                    "Implement comprehensive testing strategy"
                ],
                historical_projects=150
            ))
        
        return patterns
    
    async def _detect_domain_patterns(self, description: str) -> List[DetectedPattern]:
        """Detect domain-specific patterns."""
        patterns = []
        description_lower = description.lower()
        
        domain_matches = {}
        for domain, keywords in self._domain_keywords.items():
            match_count = sum(1 for keyword in keywords if keyword in description_lower)
            if match_count > 0:
                domain_matches[domain] = match_count / len(keywords)
        
        # Get top matching domains
        for domain, match_score in sorted(domain_matches.items(), key=lambda x: x[1], reverse=True)[:3]:
            if match_score >= 0.3:
                patterns.append(DetectedPattern(
                    pattern_id=str(uuid4()),
                    pattern_type=PatternType.TEMPLATE_TRIGGER,
                    name=f"{domain.replace('-', ' ').title()} Project",
                    description=f"Project matches {domain} domain patterns ({match_score:.1%} confidence)",
                    confidence=match_score,
                    confidence_level=self._get_confidence_level(match_score),
                    frequency=0.5,
                    success_correlation=0.6,
                    risk_impact=-0.2,  # Domain expertise reduces risk
                    indicators=self._domain_keywords[domain],
                    template_suggestions=[f"{domain}-template", f"{domain}-best-practices"],
                    mitigation_strategies=[f"Use {domain} expertise", f"Follow {domain} standards"],
                    historical_projects=80
                ))
        
        return patterns
    
    async def _detect_risk_patterns(self, description: str) -> List[DetectedPattern]:
        """Detect patterns that correlate with specific risks."""
        patterns = []
        description_lower = description.lower()
        
        # Integration risk pattern
        integration_keywords = ["integrate", "connect", "sync", "merge", "api", "third-party"]
        integration_matches = sum(1 for keyword in integration_keywords if keyword in description_lower)
        
        if integration_matches >= 2:
            patterns.append(DetectedPattern(
                pattern_id=str(uuid4()),
                pattern_type=PatternType.RISK_CORRELATE,
                name="Integration Complexity Risk",
                description="Multiple integration points detected",
                confidence=min(integration_matches / 4.0, 1.0),
                confidence_level=self._get_confidence_level(min(integration_matches / 4.0, 1.0)),
                frequency=0.4,
                success_correlation=-0.3,
                risk_impact=0.4,
                indicators=integration_keywords[:integration_matches],
                template_suggestions=["integration-testing", "api-design"],
                mitigation_strategies=[
                    "Design robust API contracts",
                    "Implement comprehensive integration testing",
                    "Plan for third-party service failures"
                ],
                historical_projects=60
            ))
        
        # Data migration risk pattern
        data_keywords = ["migrate", "import", "export", "transform", "legacy"]
        data_matches = sum(1 for keyword in data_keywords if keyword in description_lower)
        
        if data_matches >= 2:
            patterns.append(DetectedPattern(
                pattern_id=str(uuid4()),
                pattern_type=PatternType.RISK_CORRELATE,
                name="Data Migration Risk",
                description="Data migration complexity detected",
                confidence=min(data_matches / 3.0, 1.0),
                confidence_level=self._get_confidence_level(min(data_matches / 3.0, 1.0)),
                frequency=0.3,
                success_correlation=-0.5,
                risk_impact=0.6,
                indicators=data_keywords[:data_matches],
                template_suggestions=["data-migration", "backup-recovery"],
                mitigation_strategies=[
                    "Implement comprehensive data validation",
                    "Plan rollback procedures",
                    "Use phased migration approach"
                ],
                historical_projects=40
            ))
        
        return patterns
    
    async def _calculate_success_probability(self, description: str) -> float:
        """Calculate probability of project success based on patterns."""
        # This is a simplified version - in production would use ML models
        description_lower = description.lower()
        
        # Positive indicators
        positive_score = sum(
            weight for keyword, weight in self._success_keywords.items()
            if keyword in description_lower
        )
        
        # Negative indicators
        negative_score = sum(
            weight for keyword, weight in self._failure_keywords.items()
            if keyword in description_lower
        )
        
        # Normalize and calculate probability
        net_score = positive_score - negative_score
        probability = 0.5 + (net_score / 20.0)  # Base 50% + adjustment
        
        return max(0.1, min(0.9, probability))  # Clamp between 10% and 90%
    
    async def _recommend_templates(
        self, 
        description: str, 
        context: Dict[str, Any]
    ) -> List[TemplateRecommendation]:
        """Recommend templates based on detected patterns."""
        try:
            # Query database for relevant templates
            template_query = """
            MATCH (t:Template)
            WHERE t.active = true
            AND any(keyword IN split(toLower($description), ' ') WHERE 
                any(trigger IN t.trigger_keywords WHERE 
                    toLower(trigger) CONTAINS keyword OR keyword CONTAINS toLower(trigger)))
            RETURN t.id as template_id, t.name as name, t.category as category,
                   t.success_rate as success_rate, t.risk_reduction as risk_reduction,
                   t.applicable_phases as phases, t.prerequisites as prerequisites,
                   t.trigger_keywords as triggers, t.customization_notes as notes
            ORDER BY t.success_rate DESC, t.risk_reduction DESC
            LIMIT 5
            """
            
            templates_data = await self.neo4j_conn.execute_query(
                template_query, {"description": description}
            )
            
            recommendations = []
            for template_data in templates_data:
                # Calculate relevance score based on keyword matching
                relevance = self._calculate_template_relevance(
                    description, template_data.get("triggers", [])
                )
                
                if relevance >= 0.3:
                    recommendations.append(TemplateRecommendation(
                        template_id=template_data["template_id"],
                        name=template_data["name"],
                        category=template_data["category"],
                        relevance_score=relevance,
                        success_rate=template_data["success_rate"],
                        risk_reduction=template_data["risk_reduction"],
                        matching_patterns=[],  # Would be populated with pattern IDs
                        applicable_phases=template_data.get("phases", []),
                        prerequisites=template_data.get("prerequisites", []),
                        customization_notes=template_data.get("notes", [])
                    ))
            
            return recommendations
            
        except Exception as e:
            logger.warning("Template recommendation failed", error=str(e))
            return []
    
    def _calculate_template_relevance(self, description: str, triggers: List[str]) -> float:
        """Calculate template relevance score."""
        if not triggers:
            return 0.0
        
        description_lower = description.lower()
        matches = sum(1 for trigger in triggers if trigger.lower() in description_lower)
        
        return matches / len(triggers)
    
    def _calculate_complexity_score(self, patterns: List[DetectedPattern], description: str) -> float:
        """Calculate overall complexity score."""
        complexity_patterns = [p for p in patterns if p.pattern_type == PatternType.COMPLEXITY_DRIVER]
        
        if not complexity_patterns:
            # Fallback to simple heuristics
            word_count = len(description.split())
            return min(word_count / 1000.0, 1.0)
        
        # Average complexity from patterns
        avg_complexity = np.mean([p.confidence for p in complexity_patterns])
        return avg_complexity
    
    def _generate_insights(
        self, 
        patterns: List[DetectedPattern], 
        templates: List[TemplateRecommendation],
        complexity_score: float
    ) -> List[str]:
        """Generate actionable insights from pattern analysis."""
        insights = []
        
        # Pattern-based insights
        high_confidence_patterns = [p for p in patterns if p.confidence > 0.7]
        if high_confidence_patterns:
            insights.append(f"Strong patterns detected: {', '.join([p.name for p in high_confidence_patterns[:3]])}")
        
        # Risk insights
        risk_patterns = [p for p in patterns if p.pattern_type == PatternType.RISK_CORRELATE]
        if risk_patterns:
            top_risk = max(risk_patterns, key=lambda p: p.risk_impact)
            insights.append(f"Primary risk area: {top_risk.name}")
        
        # Template insights
        if templates:
            best_template = max(templates, key=lambda t: t.relevance_score)
            insights.append(f"Recommended approach: {best_template.name} (success rate: {best_template.success_rate:.0%})")
        
        # Complexity insights
        if complexity_score > 0.7:
            insights.append("High complexity project - consider phased approach")
        elif complexity_score < 0.3:
            insights.append("Low complexity project - opportunity for fast delivery")
        
        # Success probability insights
        success_patterns = [p for p in patterns if p.pattern_type == PatternType.SUCCESS_FACTOR]
        if len(success_patterns) > len(risk_patterns):
            insights.append("Positive indicators outweigh risks - good success potential")
        
        return insights[:5]
    
    def _calculate_analysis_confidence(self, patterns: List[DetectedPattern], description_length: int) -> float:
        """Calculate confidence in the pattern analysis."""
        # Factors affecting confidence
        pattern_confidence = np.mean([p.confidence for p in patterns]) if patterns else 0.3
        description_factor = min(description_length / 300.0, 1.0)
        pattern_count_factor = min(len(patterns) / 10.0, 1.0)
        
        # Weighted confidence
        confidence = (
            pattern_confidence * 0.5 +
            description_factor * 0.3 +
            pattern_count_factor * 0.2
        )
        
        return min(confidence, 0.9)
    
    def _get_confidence_level(self, confidence: float) -> PatternConfidence:
        """Convert confidence score to confidence level."""
        if confidence >= 0.8:
            return PatternConfidence.HIGH
        elif confidence >= 0.5:
            return PatternConfidence.MEDIUM
        else:
            return PatternConfidence.LOW
    
    async def _store_pattern_analysis(
        self, 
        result: PatternAnalysisResult, 
        description: str
    ) -> None:
        """Store pattern analysis results for machine learning."""
        try:
            store_query = """
            CREATE (a:PatternAnalysis {
                id: $analysis_id,
                description_hash: $description_hash,
                patterns_count: $patterns_count,
                templates_count: $templates_count,
                complexity_score: $complexity_score,
                success_probability: $success_probability,
                confidence: $confidence,
                created_at: datetime()
            })
            """
            
            await self.neo4j_conn.execute_write(
                store_query,
                {
                    "analysis_id": str(uuid4()),
                    "description_hash": result.project_description_hash,
                    "patterns_count": len(result.detected_patterns),
                    "templates_count": len(result.template_recommendations),
                    "complexity_score": result.overall_complexity_score,
                    "success_probability": result.success_probability,
                    "confidence": result.confidence
                }
            )
            
        except Exception as e:
            logger.warning("Failed to store pattern analysis", error=str(e))
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for pattern recognition service."""
        try:
            if not self.is_initialized:
                return {"status": "unhealthy", "error": "Service not initialized"}
            
            # Test basic functionality
            test_description = "Build a web application with user authentication"
            test_result = await self.analyze_patterns(test_description)
            
            return {
                "status": "healthy",
                "initialized": self.is_initialized,
                "cached_patterns": len(getattr(self, '_cached_patterns', [])),
                "test_patterns_detected": len(test_result.detected_patterns)
            }
            
        except Exception as e:
            logger.error("Pattern recognition health check failed", error=str(e))
            return {"status": "unhealthy", "error": str(e)}


# Global service instance
_pattern_service = None

async def get_pattern_recognition_service() -> PatternRecognitionService:
    """Get or create pattern recognition service instance."""
    global _pattern_service
    if _pattern_service is None:
        _pattern_service = PatternRecognitionService()
        await _pattern_service.initialize()
    return _pattern_service