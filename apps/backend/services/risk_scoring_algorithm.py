"""
Risk Scoring Algorithm Module

Advanced risk scoring algorithms based on historical project data,
machine learning models, and expert knowledge systems.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
import numpy as np
import json
from dataclasses import dataclass
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

from core.config import get_settings
from core.database import get_neo4j

logger = structlog.get_logger(__name__)
settings = get_settings()


class ScoreComponent(str, Enum):
    """Risk score component types."""
    TECHNICAL_COMPLEXITY = "technical_complexity"
    SCHEDULE_PRESSURE = "schedule_pressure"
    SCOPE_CLARITY = "scope_clarity"
    TEAM_EXPERIENCE = "team_experience"
    EXTERNAL_DEPENDENCIES = "external_dependencies"
    BUDGET_CONSTRAINTS = "budget_constraints"
    STAKEHOLDER_ALIGNMENT = "stakeholder_alignment"
    TECHNOLOGY_MATURITY = "technology_maturity"
    INTEGRATION_COMPLEXITY = "integration_complexity"
    DATA_COMPLEXITY = "data_complexity"


class RiskWeight(str, Enum):
    """Risk weight categories."""
    CRITICAL = "critical"    # 1.0
    HIGH = "high"           # 0.8
    MEDIUM = "medium"       # 0.6
    LOW = "low"             # 0.4
    MINIMAL = "minimal"     # 0.2


@dataclass
class ScoringConfig:
    """Configuration for risk scoring algorithm."""
    weights: Dict[ScoreComponent, float]
    thresholds: Dict[str, float]
    model_version: str
    calibration_date: datetime
    historical_accuracy: float


class RiskScoreBreakdown(BaseModel):
    """Detailed risk score breakdown."""
    component: ScoreComponent = Field(..., description="Score component")
    raw_score: float = Field(..., ge=0.0, le=1.0, description="Raw component score")
    weighted_score: float = Field(..., ge=0.0, le=1.0, description="Weighted component score")
    weight: float = Field(..., ge=0.0, le=1.0, description="Component weight")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Score confidence")
    contributing_factors: List[str] = Field(default_factory=list, description="Contributing factors")
    historical_basis: int = Field(default=0, description="Number of historical data points")


class RiskScoreResult(BaseModel):
    """Complete risk score result."""
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall risk score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    components: List[RiskScoreBreakdown] = Field(..., description="Component breakdowns")
    calibration_info: Dict[str, Any] = Field(default_factory=dict, description="Calibration information")
    historical_comparison: Dict[str, Any] = Field(default_factory=dict, description="Historical comparison")
    prediction_intervals: Dict[str, float] = Field(default_factory=dict, description="Prediction intervals")
    model_version: str = Field(default="1.0", description="Algorithm version")
    computed_at: datetime = Field(default_factory=datetime.utcnow, description="Computation timestamp")


class HistoricalCalibration(BaseModel):
    """Historical calibration data."""
    score_bucket: str = Field(..., description="Score bucket (e.g., '0.6-0.8')")
    actual_success_rate: float = Field(..., description="Actual success rate in this bucket")
    predicted_success_rate: float = Field(..., description="Predicted success rate")
    enterprise_size: int = Field(..., description="Number of projects in bucket")
    confidence_interval: Tuple[float, float] = Field(..., description="95% confidence interval")


class RiskScoringAlgorithm:
    """Advanced risk scoring algorithm with machine learning capabilities."""
    
    def __init__(self):
        self.neo4j_conn = None
        self.is_initialized = False
        self.config = None
        self.calibration_data = {}
        self.model_performance = {}
        
    async def initialize(self) -> None:
        """Initialize the risk scoring algorithm."""
        try:
            # Get Neo4j connection
            self.neo4j_conn = await get_neo4j()
            
            # Load configuration
            await self._load_scoring_config()
            
            # Load calibration data
            await self._load_calibration_data()
            
            # Initialize scoring models
            await self._initialize_scoring_models()
            
            self.is_initialized = True
            logger.info("Risk scoring algorithm initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize risk scoring algorithm", error=str(e))
            raise
    
    async def _load_scoring_config(self) -> None:
        """Load scoring configuration from database or defaults."""
        try:
            config_query = """
            MATCH (c:ScoringConfig)
            WHERE c.active = true
            RETURN c.weights as weights, c.thresholds as thresholds,
                   c.model_version as version, c.calibration_date as calibration_date,
                   c.historical_accuracy as accuracy
            ORDER BY c.created_at DESC
            LIMIT 1
            """
            
            config_data = await self.neo4j_conn.execute_query(config_query)
            
            if config_data:
                config_record = config_data[0]
                self.config = ScoringConfig(
                    weights=json.loads(config_record["weights"]),
                    thresholds=json.loads(config_record["thresholds"]),
                    model_version=config_record["version"],
                    calibration_date=config_record["calibration_date"],
                    historical_accuracy=config_record["accuracy"]
                )
            else:
                # Use default configuration
                self.config = self._get_default_config()
                
            logger.info(f"Loaded scoring config version {self.config.model_version}")
            
        except Exception as e:
            logger.warning("Failed to load scoring config, using defaults", error=str(e))
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> ScoringConfig:
        """Get default scoring configuration."""
        return ScoringConfig(
            weights={
                ScoreComponent.TECHNICAL_COMPLEXITY: 0.20,
                ScoreComponent.SCOPE_CLARITY: 0.18,
                ScoreComponent.SCHEDULE_PRESSURE: 0.15,
                ScoreComponent.TEAM_EXPERIENCE: 0.12,
                ScoreComponent.EXTERNAL_DEPENDENCIES: 0.10,
                ScoreComponent.INTEGRATION_COMPLEXITY: 0.08,
                ScoreComponent.STAKEHOLDER_ALIGNMENT: 0.07,
                ScoreComponent.DATA_COMPLEXITY: 0.05,
                ScoreComponent.TECHNOLOGY_MATURITY: 0.03,
                ScoreComponent.BUDGET_CONSTRAINTS: 0.02
            },
            thresholds={
                "low_risk": 0.3,
                "medium_risk": 0.6,
                "high_risk": 0.8,
                "critical_risk": 0.9
            },
            model_version="1.0",
            calibration_date=datetime.utcnow(),
            historical_accuracy=0.75
        )
    
    async def _load_calibration_data(self) -> None:
        """Load historical calibration data for score validation."""
        try:
            calibration_query = """
            MATCH (p:Project)
            WHERE p.risk_score IS NOT NULL AND p.success_score IS NOT NULL
            WITH 
                CASE 
                    WHEN p.risk_score < 0.2 THEN '0.0-0.2'
                    WHEN p.risk_score < 0.4 THEN '0.2-0.4'
                    WHEN p.risk_score < 0.6 THEN '0.4-0.6'
                    WHEN p.risk_score < 0.8 THEN '0.6-0.8'
                    ELSE '0.8-1.0'
                END as score_bucket,
                AVG(p.success_score) as avg_success,
                COUNT(p) as enterprise_size,
                STDDEV(p.success_score) as success_stddev
            RETURN score_bucket, avg_success, enterprise_size, success_stddev
            ORDER BY score_bucket
            """
            
            calibration_results = await self.neo4j_conn.execute_query(calibration_query)
            
            for result in calibration_results:
                bucket = result["score_bucket"]
                success_rate = result["avg_success"]
                enterprise_size = result["enterprise_size"]
                stddev = result.get("success_stddev", 0.1)
                
                # Calculate confidence interval
                margin_error = 1.96 * (stddev / np.sqrt(enterprise_size)) if enterprise_size > 0 else 0.5
                conf_interval = (
                    max(0.0, success_rate - margin_error),
                    min(1.0, success_rate + margin_error)
                )
                
                self.calibration_data[bucket] = HistoricalCalibration(
                    score_bucket=bucket,
                    actual_success_rate=success_rate,
                    predicted_success_rate=1.0 - float(bucket.split('-')[0]),  # Inverse of risk
                    enterprise_size=enterprise_size,
                    confidence_interval=conf_interval
                )
            
            logger.info(f"Loaded calibration data for {len(self.calibration_data)} score buckets")
            
        except Exception as e:
            logger.warning("Failed to load calibration data", error=str(e))
            self.calibration_data = {}
    
    async def _initialize_scoring_models(self) -> None:
        """Initialize component scoring models."""
        # Initialize scoring functions for each component
        self.scoring_functions = {
            ScoreComponent.TECHNICAL_COMPLEXITY: self._score_technical_complexity,
            ScoreComponent.SCOPE_CLARITY: self._score_scope_clarity,
            ScoreComponent.SCHEDULE_PRESSURE: self._score_schedule_pressure,
            ScoreComponent.TEAM_EXPERIENCE: self._score_team_experience,
            ScoreComponent.EXTERNAL_DEPENDENCIES: self._score_external_dependencies,
            ScoreComponent.INTEGRATION_COMPLEXITY: self._score_integration_complexity,
            ScoreComponent.STAKEHOLDER_ALIGNMENT: self._score_stakeholder_alignment,
            ScoreComponent.DATA_COMPLEXITY: self._score_data_complexity,
            ScoreComponent.TECHNOLOGY_MATURITY: self._score_technology_maturity,
            ScoreComponent.BUDGET_CONSTRAINTS: self._score_budget_constraints
        }
    
    async def calculate_risk_score(
        self, 
        project_description: str, 
        context: Dict[str, Any] = None
    ) -> RiskScoreResult:
        """
        Calculate comprehensive risk score for a project.
        
        Args:
            project_description: Project description text
            context: Additional context information
            
        Returns:
            RiskScoreResult with detailed score breakdown
        """
        if not self.is_initialized:
            raise RuntimeError("Risk scoring algorithm not initialized")
        
        if not context:
            context = {}
        
        try:
            logger.info("Calculating risk score", description_length=len(project_description))
            
            # Calculate component scores in parallel
            component_tasks = []
            for component, scoring_func in self.scoring_functions.items():
                task = self._calculate_component_score(
                    component, scoring_func, project_description, context
                )
                component_tasks.append(task)
            
            component_results = await asyncio.gather(*component_tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            valid_components = []
            for i, result in enumerate(component_results):
                if isinstance(result, Exception):
                    component = list(self.scoring_functions.keys())[i]
                    logger.warning(f"Failed to score {component}", error=str(result))
                    # Create fallback component
                    fallback_component = RiskScoreBreakdown(
                        component=component,
                        raw_score=0.5,  # Neutral score
                        weighted_score=0.5 * self.config.weights.get(component, 0.1),
                        weight=self.config.weights.get(component, 0.1),
                        confidence=0.3,  # Low confidence
                        contributing_factors=["Score calculation failed"],
                        historical_basis=0
                    )
                    valid_components.append(fallback_component)
                else:
                    valid_components.append(result)
            
            # Calculate overall score
            overall_score = sum(comp.weighted_score for comp in valid_components)
            overall_confidence = np.mean([comp.confidence for comp in valid_components])
            
            # Get historical comparison
            historical_comparison = await self._get_historical_comparison(
                overall_score, project_description, context
            )
            
            # Calculate prediction intervals
            prediction_intervals = self._calculate_prediction_intervals(
                overall_score, overall_confidence
            )
            
            # Get calibration info
            calibration_info = self._get_calibration_info(overall_score)
            
            result = RiskScoreResult(
                overall_score=overall_score,
                confidence=overall_confidence,
                components=valid_components,
                calibration_info=calibration_info,
                historical_comparison=historical_comparison,
                prediction_intervals=prediction_intervals,
                model_version=self.config.model_version
            )
            
            # Store result for future calibration
            await self._store_scoring_result(result, project_description, context)
            
            logger.info(
                "Risk score calculated",
                overall_score=overall_score,
                confidence=overall_confidence,
                components_count=len(valid_components)
            )
            
            return result
            
        except Exception as e:
            logger.error("Risk score calculation failed", error=str(e))
            raise
    
    async def _calculate_component_score(
        self, 
        component: ScoreComponent, 
        scoring_func: Callable,
        description: str, 
        context: Dict[str, Any]
    ) -> RiskScoreBreakdown:
        """Calculate score for a specific component."""
        try:
            # Call component-specific scoring function
            raw_score, confidence, factors, historical_basis = await scoring_func(description, context)
            
            # Apply component weight
            weight = self.config.weights.get(component, 0.1)
            weighted_score = raw_score * weight
            
            return RiskScoreBreakdown(
                component=component,
                raw_score=raw_score,
                weighted_score=weighted_score,
                weight=weight,
                confidence=confidence,
                contributing_factors=factors,
                historical_basis=historical_basis
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate {component} score", error=str(e))
            raise
    
    async def _score_technical_complexity(
        self, 
        description: str, 
        context: Dict[str, Any]
    ) -> Tuple[float, float, List[str], int]:
        """Score technical complexity component."""
        factors = []
        score_components = []
        
        description_lower = description.lower()
        
        # Technology complexity indicators
        tech_keywords = {
            "microservices": 0.8, "distributed": 0.7, "blockchain": 0.9, "machine-learning": 0.8,
            "ai": 0.7, "real-time": 0.7, "scalable": 0.6, "high-performance": 0.7,
            "cloud-native": 0.5, "containerized": 0.4, "serverless": 0.5
        }
        
        for keyword, complexity in tech_keywords.items():
            if keyword in description_lower:
                score_components.append(complexity)
                factors.append(f"Technology: {keyword}")
        
        # Integration complexity
        integration_keywords = ["integrate", "api", "third-party", "legacy", "sync", "webhook"]
        integration_count = sum(1 for keyword in integration_keywords if keyword in description_lower)
        if integration_count > 0:
            integration_score = min(integration_count * 0.2, 1.0)
            score_components.append(integration_score)
            factors.append(f"Integrations: {integration_count} systems")
        
        # Calculate base score
        if score_components:
            base_score = np.mean(score_components)
        else:
            # Fallback to simple heuristics
            word_count = len(description.split())
            base_score = min(word_count / 500.0, 1.0) * 0.5  # Conservative baseline
        
        # Query historical data for calibration
        try:
            historical_query = """
            MATCH (p:Project)
            WHERE p.technical_complexity_score IS NOT NULL
            AND any(keyword IN $keywords WHERE toLower(p.description) CONTAINS keyword)
            RETURN AVG(p.technical_complexity_score) as avg_score, COUNT(p) as count
            """
            
            historical_data = await self.neo4j_conn.execute_query(
                historical_query, {"keywords": list(tech_keywords.keys())}
            )
            
            if historical_data and historical_data[0]["count"] > 5:
                historical_avg = historical_data[0]["avg_score"]
                historical_count = historical_data[0]["count"]
                
                # Blend with historical average
                calibrated_score = base_score * 0.7 + historical_avg * 0.3
                confidence = min(0.9, 0.5 + (historical_count / 100.0))
            else:
                calibrated_score = base_score
                confidence = 0.6
                historical_count = 0
                
        except Exception:
            calibrated_score = base_score
            confidence = 0.5
            historical_count = 0
        
        return calibrated_score, confidence, factors, historical_count
    
    async def _score_scope_clarity(
        self, 
        description: str, 
        context: Dict[str, Any]
    ) -> Tuple[float, float, List[str], int]:
        """Score scope clarity component."""
        factors = []
        
        description_lower = description.lower()
        
        # Clarity indicators (lower score = higher risk)
        clear_indicators = ["must", "shall", "will", "specific", "defined", "measurable"]
        vague_indicators = ["might", "could", "possibly", "various", "several", "many"]
        
        clear_count = sum(1 for indicator in clear_indicators if indicator in description_lower)
        vague_count = sum(1 for indicator in vague_indicators if indicator in description_lower)
        
        # Calculate clarity ratio (inverse for risk score)
        total_indicators = clear_count + vague_count
        if total_indicators > 0:
            vague_ratio = vague_count / total_indicators
            scope_risk_score = vague_ratio  # Higher vagueness = higher risk
        else:
            scope_risk_score = 0.5  # Neutral if no indicators
        
        # Length and structure analysis
        sentences = len([s for s in description.split('.') if len(s.strip()) > 10])
        if sentences < 3:
            scope_risk_score += 0.2
            factors.append("Insufficient detail")
        elif sentences > 20:
            scope_risk_score += 0.1
            factors.append("Overly complex description")
        
        # Requirements structure
        if "requirements" in description_lower or "acceptance criteria" in description_lower:
            scope_risk_score -= 0.2
            factors.append("Formal requirements structure")
        
        if clear_count > 0:
            factors.append(f"Clear indicators: {clear_count}")
        if vague_count > 0:
            factors.append(f"Vague indicators: {vague_count}")
        
        # Clamp score
        final_score = max(0.0, min(1.0, scope_risk_score))
        
        # Confidence based on text length and structure
        confidence = min(0.9, 0.3 + (len(description) / 1000.0) + (sentences / 50.0))
        
        return final_score, confidence, factors, 0  # No historical lookup for simplicity
    
    async def _score_schedule_pressure(
        self, 
        description: str, 
        context: Dict[str, Any]
    ) -> Tuple[float, float, List[str], int]:
        """Score schedule pressure component."""
        factors = []
        score = 0.0
        
        description_lower = description.lower()
        
        # Urgency keywords
        urgency_keywords = {
            "urgent": 0.8, "asap": 0.9, "immediately": 0.8, "rush": 0.9,
            "deadline": 0.6, "tight": 0.7, "quick": 0.5, "fast": 0.5
        }
        
        for keyword, pressure in urgency_keywords.items():
            if keyword in description_lower:
                score = max(score, pressure)
                factors.append(f"Schedule pressure: {keyword}")
        
        # Context-based schedule pressure
        if context.get("deadline"):
            try:
                deadline = datetime.fromisoformat(context["deadline"])
                days_to_deadline = (deadline - datetime.utcnow()).days
                
                if days_to_deadline < 30:
                    score = max(score, 0.8)
                    factors.append(f"Tight deadline: {days_to_deadline} days")
                elif days_to_deadline < 90:
                    score = max(score, 0.5)
                    factors.append(f"Moderate timeline: {days_to_deadline} days")
                    
            except (ValueError, TypeError):
                pass
        
        # Estimate vs scope mismatch indicators
        if any(word in description_lower for word in ["estimate", "rough", "ballpark"]):
            score += 0.3
            factors.append("Estimation uncertainty")
        
        confidence = 0.7 if factors else 0.4
        
        return min(score, 1.0), confidence, factors, 0
    
    async def _score_team_experience(
        self, 
        description: str, 
        context: Dict[str, Any]
    ) -> Tuple[float, float, List[str], int]:
        """Score team experience component."""
        factors = []
        risk_score = 0.5  # Default neutral score
        
        # Check context for team information
        team_info = context.get("team", {})
        
        if team_info:
            # Experience level indicators
            experience_level = team_info.get("experience_level", "medium")
            if experience_level == "senior":
                risk_score = 0.2
                factors.append("Senior team experience")
            elif experience_level == "junior":
                risk_score = 0.8
                factors.append("Junior team experience")
            elif experience_level == "mixed":
                risk_score = 0.4
                factors.append("Mixed team experience")
            
            # Team size considerations
            team_size = team_info.get("size", 5)
            if team_size < 3:
                risk_score += 0.2
                factors.append("Small team size")
            elif team_size > 12:
                risk_score += 0.3
                factors.append("Large team coordination risk")
            
            # Technology familiarity
            tech_familiarity = team_info.get("technology_familiarity", 0.5)
            if tech_familiarity < 0.3:
                risk_score += 0.4
                factors.append("Low technology familiarity")
            elif tech_familiarity > 0.8:
                risk_score -= 0.2
                factors.append("High technology familiarity")
        
        # Look for experience indicators in description
        description_lower = description.lower()
        inexperience_indicators = ["new", "learning", "first-time", "unfamiliar", "explore"]
        experience_indicators = ["proven", "experienced", "familiar", "established", "mature"]
        
        inexperience_count = sum(1 for indicator in inexperience_indicators 
                                if indicator in description_lower)
        experience_count = sum(1 for indicator in experience_indicators 
                             if indicator in description_lower)
        
        if inexperience_count > experience_count:
            risk_score += 0.2
            factors.append("Inexperience indicators in description")
        elif experience_count > inexperience_count:
            risk_score -= 0.2
            factors.append("Experience indicators in description")
        
        confidence = 0.8 if team_info else 0.4
        
        return min(max(risk_score, 0.0), 1.0), confidence, factors, 0
    
    async def _score_external_dependencies(
        self, 
        description: str, 
        context: Dict[str, Any]
    ) -> Tuple[float, float, List[str], int]:
        """Score external dependencies component."""
        factors = []
        score = 0.0
        
        description_lower = description.lower()
        
        # External dependency keywords
        dependency_keywords = {
            "third-party": 0.6, "vendor": 0.5, "external": 0.5, "partner": 0.4,
            "integration": 0.6, "api": 0.4, "service": 0.3, "provider": 0.5
        }
        
        dependency_count = 0
        for keyword, risk_factor in dependency_keywords.items():
            if keyword in description_lower:
                score += risk_factor
                dependency_count += 1
                factors.append(f"External dependency: {keyword}")
        
        # Normalize score based on number of dependencies
        if dependency_count > 0:
            score = min(score / dependency_count * (1 + dependency_count * 0.1), 1.0)
        
        # Specific high-risk dependencies
        high_risk_deps = ["government", "regulatory", "compliance", "approval", "certification"]
        for dep in high_risk_deps:
            if dep in description_lower:
                score = max(score, 0.7)
                factors.append(f"High-risk dependency: {dep}")
        
        confidence = 0.7 if factors else 0.3
        
        return score, confidence, factors, 0
    
    # Simplified implementations for remaining components
    async def _score_integration_complexity(self, description: str, context: Dict[str, Any]) -> Tuple[float, float, List[str], int]:
        """Score integration complexity component."""
        factors = []
        integration_keywords = ["integrate", "connect", "sync", "merge", "api", "webhook", "event"]
        count = sum(1 for keyword in integration_keywords if keyword in description.lower())
        score = min(count * 0.2, 1.0)
        if count > 0:
            factors.append(f"Integration points: {count}")
        return score, 0.6 if count > 0 else 0.3, factors, 0
    
    async def _score_stakeholder_alignment(self, description: str, context: Dict[str, Any]) -> Tuple[float, float, List[str], int]:
        """Score stakeholder alignment component."""
        factors = []
        alignment_indicators = ["stakeholder", "consensus", "agreement", "alignment", "approval"]
        conflict_indicators = ["conflict", "disagreement", "unclear", "disputed"]
        
        alignment_count = sum(1 for indicator in alignment_indicators if indicator in description.lower())
        conflict_count = sum(1 for indicator in conflict_indicators if indicator in description.lower())
        
        score = max(0.0, conflict_count * 0.3 - alignment_count * 0.2 + 0.3)
        
        if alignment_count > 0:
            factors.append(f"Alignment indicators: {alignment_count}")
        if conflict_count > 0:
            factors.append(f"Conflict indicators: {conflict_count}")
        
        return min(score, 1.0), 0.5, factors, 0
    
    async def _score_data_complexity(self, description: str, context: Dict[str, Any]) -> Tuple[float, float, List[str], int]:
        """Score data complexity component."""
        factors = []
        data_keywords = ["data", "database", "migration", "etl", "analytics", "reporting"]
        complex_data_keywords = ["big-data", "real-time", "streaming", "warehousing"]
        
        basic_count = sum(1 for keyword in data_keywords if keyword in description.lower())
        complex_count = sum(1 for keyword in complex_data_keywords if keyword in description.lower())
        
        score = basic_count * 0.1 + complex_count * 0.3
        
        if basic_count > 0:
            factors.append(f"Data operations: {basic_count}")
        if complex_count > 0:
            factors.append(f"Complex data operations: {complex_count}")
        
        return min(score, 1.0), 0.6 if (basic_count + complex_count) > 0 else 0.3, factors, 0
    
    async def _score_technology_maturity(self, description: str, context: Dict[str, Any]) -> Tuple[float, float, List[str], int]:
        """Score technology maturity component."""
        factors = []
        bleeding_edge = ["cutting-edge", "latest", "beta", "experimental", "prototype"]
        mature_tech = ["established", "proven", "stable", "production"]
        
        bleeding_count = sum(1 for term in bleeding_edge if term in description.lower())
        mature_count = sum(1 for term in mature_tech if term in description.lower())
        
        score = bleeding_count * 0.4 - mature_count * 0.2 + 0.3
        
        if bleeding_count > 0:
            factors.append(f"Bleeding-edge technology: {bleeding_count}")
        if mature_count > 0:
            factors.append(f"Mature technology: {mature_count}")
        
        return max(0.0, min(score, 1.0)), 0.5, factors, 0
    
    async def _score_budget_constraints(self, description: str, context: Dict[str, Any]) -> Tuple[float, float, List[str], int]:
        """Score budget constraints component."""
        factors = []
        budget_keywords = ["budget", "cost", "expensive", "cheap", "funding", "financial"]
        constraint_keywords = ["limited", "tight", "constrained", "minimal"]
        
        budget_mentions = sum(1 for keyword in budget_keywords if keyword in description.lower())
        constraint_mentions = sum(1 for keyword in constraint_keywords if keyword in description.lower())
        
        score = constraint_mentions * 0.3
        
        if budget_mentions > 0:
            factors.append(f"Budget considerations: {budget_mentions}")
        if constraint_mentions > 0:
            factors.append(f"Budget constraints: {constraint_mentions}")
        
        return min(score, 1.0), 0.4 if budget_mentions > 0 else 0.2, factors, 0
    
    async def _get_historical_comparison(
        self, 
        score: float, 
        description: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get historical comparison data."""
        try:
            # Find similar projects by score range
            score_range_query = """
            MATCH (p:Project)
            WHERE p.risk_score >= $min_score AND p.risk_score <= $max_score
            AND p.success_score IS NOT NULL
            RETURN AVG(p.success_score) as avg_success, COUNT(p) as count,
                   STDDEV(p.success_score) as success_stddev
            """
            
            min_score = max(0.0, score - 0.1)
            max_score = min(1.0, score + 0.1)
            
            historical_data = await self.neo4j_conn.execute_query(
                score_range_query, {"min_score": min_score, "max_score": max_score}
            )
            
            if historical_data and historical_data[0]["count"] > 0:
                return {
                    "similar_projects_count": historical_data[0]["count"],
                    "average_success_rate": historical_data[0]["avg_success"],
                    "success_rate_stddev": historical_data[0].get("success_stddev", 0.2),
                    "score_range": f"{min_score:.2f}-{max_score:.2f}"
                }
            else:
                return {
                    "similar_projects_count": 0,
                    "average_success_rate": None,
                    "message": "Limited historical data for this score range"
                }
                
        except Exception as e:
            logger.warning("Failed to get historical comparison", error=str(e))
            return {"error": "Historical comparison unavailable"}
    
    def _calculate_prediction_intervals(
        self, 
        score: float, 
        confidence: float
    ) -> Dict[str, float]:
        """Calculate prediction intervals for the risk score."""
        # Simple prediction intervals based on confidence
        margin = (1.0 - confidence) * 0.2  # Larger margin for lower confidence
        
        return {
            "lower_bound": max(0.0, score - margin),
            "upper_bound": min(1.0, score + margin),
            "confidence_level": 0.95
        }
    
    def _get_calibration_info(self, score: float) -> Dict[str, Any]:
        """Get calibration information for the score."""
        # Find appropriate score bucket
        bucket = None
        if score < 0.2:
            bucket = "0.0-0.2"
        elif score < 0.4:
            bucket = "0.2-0.4"
        elif score < 0.6:
            bucket = "0.4-0.6"
        elif score < 0.8:
            bucket = "0.6-0.8"
        else:
            bucket = "0.8-1.0"
        
        if bucket in self.calibration_data:
            calibration = self.calibration_data[bucket]
            return {
                "score_bucket": bucket,
                "historical_success_rate": calibration.actual_success_rate,
                "enterprise_size": calibration.enterprise_size,
                "confidence_interval": calibration.confidence_interval,
                "calibration_quality": "good" if calibration.enterprise_size > 20 else "limited"
            }
        else:
            return {
                "score_bucket": bucket,
                "calibration_quality": "no_data",
                "message": "No historical calibration data available"
            }
    
    async def _store_scoring_result(
        self, 
        result: RiskScoreResult, 
        description: str, 
        context: Dict[str, Any]
    ) -> None:
        """Store scoring result for future model training."""
        try:
            store_query = """
            CREATE (r:RiskScoringResult {
                id: $result_id,
                overall_score: $overall_score,
                confidence: $confidence,
                model_version: $model_version,
                description_length: $description_length,
                components_count: $components_count,
                created_at: datetime()
            })
            """
            
            await self.neo4j_conn.execute_write(
                store_query,
                {
                    "result_id": str(uuid4()),
                    "overall_score": result.overall_score,
                    "confidence": result.confidence,
                    "model_version": result.model_version,
                    "description_length": len(description),
                    "components_count": len(result.components)
                }
            )
            
        except Exception as e:
            logger.warning("Failed to store scoring result", error=str(e))
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for risk scoring algorithm."""
        try:
            if not self.is_initialized:
                return {"status": "unhealthy", "error": "Algorithm not initialized"}
            
            # Test scoring with enterprise data
            test_description = "Build a web application with user authentication and payment processing"
            test_result = await self.calculate_risk_score(test_description)
            
            return {
                "status": "healthy",
                "initialized": self.is_initialized,
                "model_version": self.config.model_version,
                "calibration_buckets": len(self.calibration_data),
                "test_score": test_result.overall_score,
                "test_confidence": test_result.confidence
            }
            
        except Exception as e:
            logger.error("Risk scoring health check failed", error=str(e))
            return {"status": "unhealthy", "error": str(e)}


# Global algorithm instance
_scoring_algorithm = None

async def get_risk_scoring_algorithm() -> RiskScoringAlgorithm:
    """Get or create risk scoring algorithm instance."""
    global _scoring_algorithm
    if _scoring_algorithm is None:
        _scoring_algorithm = RiskScoringAlgorithm()
        await _scoring_algorithm.initialize()
    return _scoring_algorithm