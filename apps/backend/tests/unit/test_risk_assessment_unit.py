"""
Unit tests for Risk Assessment system components.

Tests individual components in isolation without external dependencies.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import Dict, Any, List

from services.risk_assessment_service import (
    RiskAssessmentService,
    RiskFactor,
    RiskLevel,
    RiskCategory,
    HistoricalPattern,
    ProjectTemplate,
    RiskAssessmentResult
)
from services.pattern_recognition_service import (
    PatternRecognitionService,
    PatternType,
    DetectedPattern,
    TemplateRecommendation,
    PatternAnalysisResult
)
from services.risk_scoring_algorithm import (
    RiskScoringAlgorithm,
    ScoreComponent,
    RiskScoreBreakdown,
    RiskScoreResult
)


class TestRiskAssessmentService:
    """Unit tests for RiskAssessmentService."""
    
    @pytest.fixture
    def mock_neo4j(self):
        """Mock Neo4j connection."""
        mock_conn = AsyncMock()
        mock_conn.execute_query = AsyncMock()
        mock_conn.execute_write = AsyncMock()
        mock_conn.is_connected = True
        return mock_conn
    
    @pytest.fixture
    def risk_service(self, mock_neo4j):
        """Create RiskAssessmentService with mocked dependencies."""
        service = RiskAssessmentService()
        service.neo4j_conn = mock_neo4j
        service.is_initialized = True
        return service
    
    @pytest.mark.asyncio
    async def test_technical_risk_analysis(self, risk_service, mock_neo4j):
        """Test technical risk analysis logic."""
        # Mock Neo4j response for technical risks
        mock_neo4j.execute_query.return_value = [
            {
                "name": "Integration Complexity",
                "description": "Multiple third-party integrations",
                "avg_probability": 0.7,
                "avg_impact": 0.8,
                "frequency": 15,
                "mitigations": [["Use API testing", "Implement fallbacks"]]
            },
            {
                "name": "Technology Stack Risk",
                "description": "New technology adoption",
                "avg_probability": 0.5,
                "avg_impact": 0.6,
                "frequency": 8,
                "mitigations": [["Proof of concept", "Training plan"]]
            }
        ]
        
        # Test project description with technical complexity
        description = "Build a microservices platform with machine learning and real-time processing"
        context = {"team_size": 5}
        
        # Call the private method directly for unit testing
        risks = await risk_service._analyze_technical_risks(description, context)
        
        # Validate results
        assert len(risks) >= 2
        
        # Check first risk
        first_risk = risks[0]
        assert first_risk.category == RiskCategory.TECHNICAL
        assert first_risk.name == "Integration Complexity"
        assert 0.0 <= first_risk.risk_score <= 1.0
        assert first_risk.level in [level.value for level in RiskLevel]
        assert len(first_risk.mitigation_strategies) > 0
    
    @pytest.mark.asyncio
    async def test_scope_clarity_analysis(self, risk_service):
        """Test scope clarity analysis."""
        # Test clear scope description
        clear_description = """
        The system must provide user authentication with the following specific requirements:
        - Users shall be able to register with email and password
        - Password must be at least 8 characters with special characters
        - Users will receive email confirmation within 5 minutes
        - Failed login attempts will be logged and blocked after 3 attempts
        """
        
        risks = await risk_service._analyze_scope_risks(clear_description, {})
        
        # Should have low scope risk for clear requirements
        scope_risks = [r for r in risks if "scope" in r.name.lower()]
        if scope_risks:
            assert all(r.risk_score < 0.7 for r in scope_risks)
        
        # Test vague scope description
        vague_description = """
        Build something that might help users do various tasks.
        It should be fast and possibly integrate with several systems.
        The interface could be modern and approximately user-friendly.
        """
        
        risks = await risk_service._analyze_scope_risks(vague_description, {})
        
        # Should have higher scope risk for vague requirements
        assert len(risks) > 0
        assert any(r.risk_score > 0.5 for r in risks)
    
    @pytest.mark.asyncio
    async def test_schedule_pressure_analysis(self, risk_service):
        """Test schedule pressure analysis."""
        # Test urgent project description
        urgent_description = "Need to build ASAP for urgent deadline"
        urgent_context = {
            "deadline": (datetime.utcnow()).isoformat()  # Today's date = urgent
        }
        
        risks = await risk_service._analyze_schedule_risks(urgent_description, urgent_context)
        
        # Should identify high schedule pressure
        assert len(risks) > 0
        schedule_risk = risks[0] if risks else None
        if schedule_risk:
            assert schedule_risk.category == RiskCategory.SCHEDULE
            assert schedule_risk.risk_score > 0.6  # High risk for urgent timeline
    
    @pytest.mark.asyncio
    async def test_risk_level_determination(self, risk_service):
        """Test risk level calculation."""
        # Test different risk scores
        test_cases = [
            (0.1, RiskLevel.LOW),
            (0.4, RiskLevel.MEDIUM),
            (0.7, RiskLevel.HIGH),
            (0.9, RiskLevel.CRITICAL)
        ]
        
        for score, expected_level in test_cases:
            level = risk_service._determine_risk_level(score)
            assert level == expected_level
    
    @pytest.mark.asyncio
    async def test_overall_risk_calculation(self, risk_service):
        """Test overall risk score calculation."""
        # Create test risk factors
        risks = [
            RiskFactor(
                id="1",
                category=RiskCategory.TECHNICAL,
                name="Test Risk 1",
                description="Test",
                probability=0.8,
                impact=0.7,
                risk_score=0.56,
                level=RiskLevel.HIGH,
                mitigation_strategies=[]
            ),
            RiskFactor(
                id="2",
                category=RiskCategory.SCHEDULE,
                name="Test Risk 2", 
                description="Test",
                probability=0.5,
                impact=0.6,
                risk_score=0.30,
                level=RiskLevel.MEDIUM,
                mitigation_strategies=[]
            )
        ]
        
        overall_score = risk_service._calculate_overall_risk_score(risks)
        
        # Should be weighted average considering category weights
        assert 0.0 <= overall_score <= 1.0
        assert overall_score > 0.3  # Should be influenced by the high-risk factor
    
    @pytest.mark.asyncio
    async def test_empty_input_handling(self, risk_service):
        """Test handling of edge cases."""
        # Empty risk factors
        overall_score = risk_service._calculate_overall_risk_score([])
        assert overall_score == 0.0
        
        # Test with minimal description
        minimal_risks = await risk_service._analyze_scope_risks("Test project", {})
        assert isinstance(minimal_risks, list)  # Should not crash


class TestPatternRecognitionService:
    """Unit tests for PatternRecognitionService."""
    
    @pytest.fixture
    def pattern_service(self):
        """Create PatternRecognitionService with mocked dependencies."""
        service = PatternRecognitionService()
        service.neo4j_conn = AsyncMock()
        service.is_initialized = True
        service._cached_patterns = []
        return service
    
    @pytest.mark.asyncio
    async def test_success_failure_pattern_detection(self, pattern_service):
        """Test success and failure pattern detection."""
        # Test description with success indicators
        success_description = "Agile iterative development with comprehensive testing and stakeholder feedback"
        
        patterns = await pattern_service._detect_success_failure_patterns(success_description)
        
        # Should detect success patterns
        success_patterns = [p for p in patterns if p.pattern_type == PatternType.SUCCESS_FACTOR]
        assert len(success_patterns) > 0
        
        success_pattern = success_patterns[0]
        assert success_pattern.success_correlation > 0
        assert success_pattern.risk_impact < 0  # Should reduce risk
        assert len(success_pattern.indicators) > 0
        
        # Test description with failure indicators
        failure_description = "Rush to complete everything immediately with unclear requirements"
        
        patterns = await pattern_service._detect_success_failure_patterns(failure_description)
        
        # Should detect failure patterns
        failure_patterns = [p for p in patterns if p.pattern_type == PatternType.FAILURE_INDICATOR]
        if failure_patterns:
            failure_pattern = failure_patterns[0]
            assert failure_pattern.success_correlation < 0
            assert failure_pattern.risk_impact > 0  # Should increase risk
    
    @pytest.mark.asyncio
    async def test_complexity_pattern_detection(self, pattern_service):
        """Test complexity pattern detection."""
        # High complexity description
        complex_description = """
        Integrate multiple legacy systems with modern microservices architecture
        while migrating terabytes of data and ensuring zero downtime.
        System must handle real-time processing of millions of events
        across distributed cloud infrastructure with complex business rules.
        """
        
        patterns = await pattern_service._detect_complexity_patterns(complex_description)
        
        # Should detect high complexity
        complexity_patterns = [p for p in patterns if p.pattern_type == PatternType.COMPLEXITY_DRIVER]
        assert len(complexity_patterns) > 0
        
        complexity_pattern = complexity_patterns[0]
        assert complexity_pattern.confidence > 0.5
        assert "complexity" in complexity_pattern.name.lower()
        assert len(complexity_pattern.mitigation_strategies) > 0
    
    @pytest.mark.asyncio
    async def test_domain_pattern_matching(self, pattern_service):
        """Test domain-specific pattern matching."""
        # E-commerce description
        ecommerce_description = "Build online store with shopping cart, payment processing, and inventory management"
        
        patterns = await pattern_service._detect_domain_patterns(ecommerce_description)
        
        # Should match e-commerce domain
        domain_patterns = [p for p in patterns if p.pattern_type == PatternType.TEMPLATE_TRIGGER]
        
        if domain_patterns:
            ecommerce_pattern = next((p for p in domain_patterns if "commerce" in p.name.lower()), None)
            if ecommerce_pattern:
                assert ecommerce_pattern.confidence > 0.3
                assert "commerce" in ecommerce_pattern.template_suggestions[0] if ecommerce_pattern.template_suggestions else True
    
    @pytest.mark.asyncio
    async def test_success_probability_calculation(self, pattern_service):
        """Test success probability calculation."""
        # High success probability description
        high_success_desc = "Proven agile methodology with experienced team and clear requirements"
        probability = await pattern_service._calculate_success_probability(high_success_desc)
        assert 0.5 <= probability <= 0.9
        
        # Low success probability description  
        low_success_desc = "Unclear requirements with tight deadlines and new technology"
        probability = await pattern_service._calculate_success_probability(low_success_desc)
        assert 0.1 <= probability <= 0.6
    
    def test_keyword_loading(self, pattern_service):
        """Test keyword loading functionality."""
        # Test success keywords are loaded
        success_keywords = pattern_service._success_keywords
        assert len(success_keywords) > 0
        assert "agile" in success_keywords
        assert success_keywords["agile"] > 0
        
        # Test failure keywords are loaded
        failure_keywords = pattern_service._failure_keywords
        assert len(failure_keywords) > 0
        assert "unclear" in failure_keywords
        assert failure_keywords["unclear"] > 0
        
        # Test complexity keywords are loaded
        complexity_keywords = pattern_service._complexity_keywords
        assert len(complexity_keywords) > 0
        assert "integrate" in complexity_keywords
        assert complexity_keywords["integrate"] > 0


class TestRiskScoringAlgorithm:
    """Unit tests for RiskScoringAlgorithm."""
    
    @pytest.fixture
    def scoring_algorithm(self):
        """Create RiskScoringAlgorithm with mocked dependencies."""
        algorithm = RiskScoringAlgorithm()
        algorithm.neo4j_conn = AsyncMock()
        algorithm.is_initialized = True
        algorithm.config = algorithm._get_default_config()
        algorithm.calibration_data = {}
        algorithm.scoring_functions = {}
        return algorithm
    
    @pytest.mark.asyncio
    async def test_technical_complexity_scoring(self, scoring_algorithm):
        """Test technical complexity component scoring."""
        # Mock Neo4j response
        scoring_algorithm.neo4j_conn.execute_query.return_value = [
            {"avg_score": 0.6, "count": 10}
        ]
        
        # Test high-tech description
        high_tech_desc = "Machine learning microservices with blockchain and real-time processing"
        context = {"team_experience": 0.7}
        
        score, confidence, factors, historical = await scoring_algorithm._score_technical_complexity(
            high_tech_desc, context
        )
        
        # Should have high technical complexity score
        assert 0.0 <= score <= 1.0
        assert 0.0 <= confidence <= 1.0
        assert len(factors) > 0
        
        # Should identify specific technology factors
        tech_factors = [f for f in factors if "machine-learning" in f.lower() or "blockchain" in f.lower()]
        assert len(tech_factors) > 0
    
    @pytest.mark.asyncio  
    async def test_scope_clarity_scoring(self, scoring_algorithm):
        """Test scope clarity component scoring."""
        # Clear requirements
        clear_desc = "The system must implement user authentication with specific requirements: users shall register with email, password must be 8+ characters, system will send confirmation emails."
        
        score, confidence, factors, _ = await scoring_algorithm._score_scope_clarity(clear_desc, {})
        
        # Should have low risk score for clear requirements
        assert score <= 0.6
        assert confidence > 0.3
        
        # Vague requirements
        vague_desc = "Build something that might work with various features that could be useful."
        
        score, confidence, factors, _ = await scoring_algorithm._score_scope_clarity(vague_desc, {})
        
        # Should have higher risk score for vague requirements
        assert score >= 0.3
        assert "vague" in str(factors).lower() or "insufficient" in str(factors).lower()
    
    @pytest.mark.asyncio
    async def test_schedule_pressure_scoring(self, scoring_algorithm):
        """Test schedule pressure component scoring."""
        # Urgent timeline
        urgent_desc = "Need this ASAP for urgent deadline"
        urgent_context = {"deadline": datetime.utcnow().isoformat()}
        
        score, confidence, factors, _ = await scoring_algorithm._score_schedule_pressure(
            urgent_desc, urgent_context
        )
        
        # Should have high schedule pressure score
        assert score >= 0.5
        assert len(factors) > 0
        
        # Normal timeline
        normal_desc = "Standard development timeline with reasonable expectations"
        normal_context = {}
        
        score, confidence, factors, _ = await scoring_algorithm._score_schedule_pressure(
            normal_desc, normal_context
        )
        
        # Should have lower schedule pressure score
        assert score <= 0.7
    
    @pytest.mark.asyncio
    async def test_team_experience_scoring(self, scoring_algorithm):
        """Test team experience component scoring."""
        # Senior team
        desc = "Project with experienced team"
        senior_context = {
            "team": {
                "experience_level": "senior",
                "size": 5,
                "technology_familiarity": 0.9
            }
        }
        
        score, confidence, factors, _ = await scoring_algorithm._score_team_experience(desc, senior_context)
        
        # Should have low risk for senior team
        assert score <= 0.4
        assert confidence > 0.6
        assert "senior" in str(factors).lower()
        
        # Junior team
        junior_context = {
            "team": {
                "experience_level": "junior",
                "size": 3,
                "technology_familiarity": 0.2
            }
        }
        
        score, confidence, factors, _ = await scoring_algorithm._score_team_experience(desc, junior_context)
        
        # Should have higher risk for junior team
        assert score >= 0.6
        assert "junior" in str(factors).lower()
    
    @pytest.mark.asyncio
    async def test_component_score_aggregation(self, scoring_algorithm):
        """Test component score aggregation logic."""
        # Create mock scoring functions
        async def mock_tech_scoring(desc, ctx):
            return 0.7, 0.8, ["high complexity"], 10
        
        async def mock_scope_scoring(desc, ctx):
            return 0.3, 0.9, ["clear requirements"], 0
        
        scoring_algorithm.scoring_functions = {
            ScoreComponent.TECHNICAL_COMPLEXITY: mock_tech_scoring,
            ScoreComponent.SCOPE_CLARITY: mock_scope_scoring
        }
        
        # Calculate component score
        tech_component = await scoring_algorithm._calculate_component_score(
            ScoreComponent.TECHNICAL_COMPLEXITY,
            mock_tech_scoring,
            "test description",
            {}
        )
        
        # Validate component breakdown
        assert tech_component.component == ScoreComponent.TECHNICAL_COMPLEXITY
        assert tech_component.raw_score == 0.7
        assert tech_component.confidence == 0.8
        assert len(tech_component.contributing_factors) == 1
        assert tech_component.historical_basis == 10
        
        # Weight should be applied
        expected_weight = scoring_algorithm.config.weights.get(ScoreComponent.TECHNICAL_COMPLEXITY, 0.1)
        assert tech_component.weight == expected_weight
        assert tech_component.weighted_score == 0.7 * expected_weight
    
    def test_confidence_level_classification(self, scoring_algorithm):
        """Test confidence level classification."""
        from services.risk_scoring_algorithm import PatternConfidence
        
        # Test confidence level mapping
        assert scoring_algorithm._get_confidence_level(0.9) == PatternConfidence.HIGH
        assert scoring_algorithm._get_confidence_level(0.6) == PatternConfidence.MEDIUM  
        assert scoring_algorithm._get_confidence_level(0.3) == PatternConfidence.LOW
    
    def test_prediction_intervals(self, scoring_algorithm):
        """Test prediction interval calculations."""
        # High confidence
        intervals = scoring_algorithm._calculate_prediction_intervals(0.6, 0.9)
        assert intervals["lower_bound"] <= 0.6 <= intervals["upper_bound"]
        assert intervals["upper_bound"] - intervals["lower_bound"] <= 0.3  # Narrow interval
        
        # Low confidence  
        intervals = scoring_algorithm._calculate_prediction_intervals(0.6, 0.3)
        assert intervals["lower_bound"] <= 0.6 <= intervals["upper_bound"]
        assert intervals["upper_bound"] - intervals["lower_bound"] >= 0.2  # Wider interval


class TestIntegrationHelpers:
    """Test helper functions and utilities."""
    
    def test_risk_level_mapping(self):
        """Test risk level enumeration mapping."""
        # Test all risk levels are defined
        levels = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert len(levels) == 4
        
        # Test string values
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.CRITICAL.value == "critical"
    
    def test_risk_category_mapping(self):
        """Test risk category enumeration mapping."""
        categories = [
            RiskCategory.TECHNICAL, RiskCategory.SCHEDULE, RiskCategory.SCOPE,
            RiskCategory.TEAM, RiskCategory.EXTERNAL, RiskCategory.BUDGET
        ]
        
        # All categories should be defined
        assert len(categories) >= 6
        
        # Test string values
        assert RiskCategory.TECHNICAL.value == "technical"
        assert RiskCategory.SCHEDULE.value == "schedule"
    
    def test_score_component_mapping(self):
        """Test score component enumeration mapping."""
        components = [
            ScoreComponent.TECHNICAL_COMPLEXITY,
            ScoreComponent.SCHEDULE_PRESSURE,
            ScoreComponent.SCOPE_CLARITY
        ]
        
        # Components should be defined
        assert len(components) >= 3
        
        # Test string values
        assert ScoreComponent.TECHNICAL_COMPLEXITY.value == "technical_complexity"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])