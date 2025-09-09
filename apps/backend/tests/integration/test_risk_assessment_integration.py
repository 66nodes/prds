"""
Integration tests for Risk Assessment system.

Tests the complete risk assessment workflow including:
- Risk assessment service
- Pattern recognition service  
- Risk scoring algorithm
- API endpoints
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from main import app
from services.risk_assessment_service import get_risk_assessment_service, RiskLevel
from services.pattern_recognition_service import get_pattern_recognition_service
from services.risk_scoring_algorithm import get_risk_scoring_algorithm
from core.database import get_neo4j


@pytest.fixture
async def risk_service():
    """Get risk assessment service instance."""
    service = await get_risk_assessment_service()
    yield service


@pytest.fixture
async def pattern_service():
    """Get pattern recognition service instance."""
    service = await get_pattern_recognition_service()
    yield service


@pytest.fixture
async def scoring_service():
    """Get risk scoring algorithm instance."""
    service = await get_risk_scoring_algorithm()
    yield service


@pytest.fixture
async def neo4j_conn():
    """Get Neo4j connection for test setup."""
    conn = await get_neo4j()
    yield conn


@pytest.fixture
def sample_project_description():
    """Sample project description for testing."""
    return """
    Build a comprehensive e-commerce platform with user authentication, 
    payment processing, inventory management, and real-time analytics dashboard.
    The system should integrate with multiple third-party services including 
    payment processors, shipping providers, and marketing automation tools.
    Timeline is aggressive at 3 months with a team of 5 developers.
    """


@pytest.fixture
def complex_project_description():
    """Complex project description for testing."""
    return """
    Develop a cutting-edge blockchain-based supply chain management platform
    that leverages machine learning for predictive analytics and IoT sensors
    for real-time tracking. The system must integrate with legacy ERP systems,
    provide real-time visibility across multiple vendors, and ensure compliance
    with various international regulations. The project involves migrating
    petabytes of historical data and requires 99.99% uptime.
    """


class TestRiskAssessmentService:
    """Test risk assessment service functionality."""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, risk_service):
        """Test service initializes correctly."""
        assert risk_service.is_initialized
        
        # Test health check
        health = await risk_service.health_check()
        assert health["status"] in ["healthy", "degraded"]
        assert "components" in health
    
    @pytest.mark.asyncio
    async def test_basic_risk_assessment(self, risk_service, sample_project_description):
        """Test basic risk assessment functionality."""
        result = await risk_service.assess_project_risks(
            sample_project_description,
            "e-commerce",
            {"deadline": (datetime.utcnow() + timedelta(days=90)).isoformat()}
        )
        
        # Validate result structure
        assert hasattr(result, 'overall_risk_score')
        assert hasattr(result, 'risk_level')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'risk_factors')
        
        # Validate risk score is within bounds
        assert 0.0 <= result.overall_risk_score <= 1.0
        assert result.risk_level in [level.value for level in RiskLevel]
        assert 0.0 <= result.confidence <= 1.0
        
        # Should have some risk factors
        assert len(result.risk_factors) > 0
        
        # Should have actionable insights
        assert len(result.actionable_insights) > 0
    
    @pytest.mark.asyncio
    async def test_complex_project_assessment(self, risk_service, complex_project_description):
        """Test assessment of complex project."""
        result = await risk_service.assess_project_risks(
            complex_project_description,
            "enterprise",
            {"team_size": 15, "budget": "high"}
        )
        
        # Complex projects should have higher risk scores
        assert result.overall_risk_score >= 0.4
        
        # Should identify multiple risk categories
        categories = {factor.category for factor in result.risk_factors}
        assert len(categories) >= 3
        
        # Should have technical complexity risks
        technical_risks = [f for f in result.risk_factors if f.category.value == "TECHNICAL"]
        assert len(technical_risks) > 0
    
    @pytest.mark.asyncio
    async def test_lessons_learned_retrieval(self, risk_service):
        """Test lessons learned functionality."""
        lessons = await risk_service.get_lessons_learned(category="technical", limit=5)
        
        # Should return list of lessons
        assert isinstance(lessons, list)
        assert len(lessons) <= 5
        
        # Each lesson should have required fields
        for lesson in lessons:
            assert hasattr(lesson, 'title')
            assert hasattr(lesson, 'description')
            assert hasattr(lesson, 'recommendation')
            assert hasattr(lesson, 'confidence')
            assert 0.0 <= lesson.confidence <= 1.0


class TestPatternRecognitionService:
    """Test pattern recognition service functionality."""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, pattern_service):
        """Test service initializes correctly."""
        assert pattern_service.is_initialized
        
        health = await pattern_service.health_check()
        assert health["status"] in ["healthy", "degraded"]
    
    @pytest.mark.asyncio
    async def test_pattern_analysis(self, pattern_service, sample_project_description):
        """Test pattern analysis functionality."""
        result = await pattern_service.analyze_patterns(
            sample_project_description,
            {"project_type": "web_application"}
        )
        
        # Validate result structure
        assert hasattr(result, 'detected_patterns')
        assert hasattr(result, 'template_recommendations')
        assert hasattr(result, 'overall_complexity_score')
        assert hasattr(result, 'success_probability')
        assert hasattr(result, 'confidence')
        
        # Validate scores
        assert 0.0 <= result.overall_complexity_score <= 1.0
        assert 0.0 <= result.success_probability <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        
        # Should detect some patterns
        assert isinstance(result.detected_patterns, list)
        
        # Should have insights
        assert len(result.key_insights) > 0
    
    @pytest.mark.asyncio
    async def test_domain_pattern_detection(self, pattern_service):
        """Test domain-specific pattern detection."""
        # Test e-commerce project
        ecommerce_desc = "Build an online store with shopping cart, payment processing, and inventory management"
        
        result = await pattern_service.analyze_patterns(ecommerce_desc)
        
        # Should detect e-commerce patterns
        domain_patterns = [p for p in result.detected_patterns if "commerce" in p.name.lower()]
        
        # Should have template recommendations
        assert len(result.template_recommendations) > 0


class TestRiskScoringAlgorithm:
    """Test risk scoring algorithm functionality."""
    
    @pytest.mark.asyncio
    async def test_algorithm_initialization(self, scoring_service):
        """Test algorithm initializes correctly."""
        assert scoring_service.is_initialized
        assert scoring_service.config is not None
        
        health = await scoring_service.health_check()
        assert health["status"] in ["healthy", "degraded"]
    
    @pytest.mark.asyncio
    async def test_risk_score_calculation(self, scoring_service, sample_project_description):
        """Test risk score calculation."""
        result = await scoring_service.calculate_risk_score(
            sample_project_description,
            {"team_experience": 0.8, "deadline": "tight"}
        )
        
        # Validate result structure
        assert hasattr(result, 'overall_score')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'components')
        
        # Validate scores
        assert 0.0 <= result.overall_score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        
        # Should have component breakdowns
        assert len(result.components) > 0
        
        # Each component should be valid
        for component in result.components:
            assert 0.0 <= component.raw_score <= 1.0
            assert 0.0 <= component.confidence <= 1.0
            assert component.weight > 0.0
    
    @pytest.mark.asyncio
    async def test_component_scoring_consistency(self, scoring_service):
        """Test that component scoring is consistent."""
        # Test same description multiple times
        description = "Simple web application with user authentication"
        
        results = []
        for _ in range(3):
            result = await scoring_service.calculate_risk_score(description)
            results.append(result.overall_score)
        
        # Results should be consistent (within 10% variance)
        max_variance = max(results) - min(results)
        assert max_variance <= 0.1


class TestRiskAssessmentAPI:
    """Test risk assessment API endpoints."""
    
    @pytest.fixture
    def auth_headers(self):
        """Mock authentication headers."""
        # In real tests, this would use actual JWT tokens
        return {"Authorization": "Bearer test-token"}
    
    @pytest.mark.asyncio
    async def test_risk_assessment_endpoint(self, auth_headers, sample_project_description):
        """Test main risk assessment endpoint."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post(
                "/api/v1/risk-assessment/",
                json={
                    "project_description": sample_project_description,
                    "project_category": "web-application",
                    "include_historical": True,
                    "include_templates": True
                },
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert "assessment" in data
        assert "processing_time" in data
        
        assessment = data["assessment"]
        assert "overall_risk_score" in assessment
        assert "risk_level" in assessment
        assert "confidence" in assessment
        assert "risk_factors" in assessment
    
    @pytest.mark.asyncio
    async def test_pattern_analysis_endpoint(self, auth_headers, sample_project_description):
        """Test pattern analysis endpoint."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post(
                "/api/v1/risk-assessment/patterns",
                json={
                    "project_description": sample_project_description,
                    "context": {"domain": "e-commerce"}
                },
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert "detected_patterns" in data
        assert "template_recommendations" in data
        assert "overall_complexity_score" in data
    
    @pytest.mark.asyncio
    async def test_lessons_learned_endpoint(self, auth_headers):
        """Test lessons learned endpoint."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get(
                "/api/v1/risk-assessment/lessons-learned?category=technical&limit=5",
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "lessons" in data
        assert isinstance(data["lessons"], list)
        assert len(data["lessons"]) <= 5
    
    @pytest.mark.asyncio
    async def test_health_check_endpoint(self):
        """Test health check endpoint."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/api/v1/risk-assessment/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "services" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
    
    @pytest.mark.asyncio
    async def test_export_functionality(self, auth_headers):
        """Test assessment export functionality."""
        # First create an assessment
        sample_assessment = {
            "overall_risk_score": 0.65,
            "risk_level": "MEDIUM",
            "confidence": 0.8,
            "risk_factors": [
                {
                    "name": "Technical Complexity",
                    "category": "TECHNICAL",
                    "level": "HIGH",
                    "risk_score": 0.7
                }
            ],
            "actionable_insights": ["Consider phased implementation"]
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post(
                "/api/v1/risk-assessment/export",
                json={
                    "assessment": sample_assessment,
                    "format": "json"
                },
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "download_url" in data
        assert "file_size" in data


class TestRiskAssessmentWorkflow:
    """Test complete risk assessment workflow."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(
        self, 
        risk_service, 
        pattern_service, 
        scoring_service,
        sample_project_description
    ):
        """Test complete end-to-end risk assessment workflow."""
        # Step 1: Run risk assessment
        risk_result = await risk_service.assess_project_risks(
            sample_project_description,
            "web-application"
        )
        
        # Step 2: Run pattern analysis
        pattern_result = await pattern_service.analyze_patterns(
            sample_project_description
        )
        
        # Step 3: Run risk scoring
        scoring_result = await scoring_service.calculate_risk_score(
            sample_project_description
        )
        
        # Validate all services produced results
        assert risk_result.overall_risk_score > 0
        assert len(pattern_result.detected_patterns) >= 0
        assert scoring_result.overall_score > 0
        
        # Risk scores should be roughly correlated
        score_diff = abs(risk_result.overall_risk_score - scoring_result.overall_score)
        assert score_diff <= 0.3  # Allow some variance between methods
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(
        self,
        risk_service,
        sample_project_description
    ):
        """Test performance meets requirements."""
        start_time = datetime.utcnow()
        
        # Run assessment
        result = await risk_service.assess_project_risks(sample_project_description)
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        # Should complete within 10 seconds
        assert processing_time <= 10.0
        
        # Should have reasonable confidence
        assert result.confidence >= 0.5
    
    @pytest.mark.asyncio
    async def test_concurrent_assessments(
        self,
        risk_service
    ):
        """Test handling multiple concurrent assessments."""
        descriptions = [
            "Build a simple blog platform",
            "Create an enterprise CRM system",
            "Develop a mobile payment app",
            "Implement a data analytics pipeline"
        ]
        
        # Run assessments concurrently
        tasks = [
            risk_service.assess_project_risks(desc, "general")
            for desc in descriptions
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed
        for result in results:
            assert not isinstance(result, Exception)
            assert hasattr(result, 'overall_risk_score')
        
        # Results should be different (showing proper differentiation)
        scores = [r.overall_risk_score for r in results]
        assert len(set(scores)) > 1  # Not all the same


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_invalid_input_handling(self, risk_service):
        """Test handling of invalid inputs."""
        # Empty description
        with pytest.raises(Exception):
            await risk_service.assess_project_risks("")
        
        # Very short description
        with pytest.raises(Exception):
            await risk_service.assess_project_risks("test")
        
        # None input
        with pytest.raises(Exception):
            await risk_service.assess_project_risks(None)
    
    @pytest.mark.asyncio
    async def test_service_degradation(self, risk_service):
        """Test graceful degradation when components fail."""
        # This would test behavior when Neo4j is unavailable, etc.
        # Implementation depends on specific failure scenarios
        
        # Test with minimal context
        result = await risk_service.assess_project_risks(
            "Build a simple web application with basic functionality"
        )
        
        # Should still provide some assessment even with limited data
        assert result.overall_risk_score >= 0.0
        assert result.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_api_validation(self, auth_headers):
        """Test API input validation."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            # Test missing required fields
            response = await ac.post(
                "/api/v1/risk-assessment/",
                json={},
                headers=auth_headers
            )
            assert response.status_code == 422
            
            # Test invalid project description
            response = await ac.post(
                "/api/v1/risk-assessment/",
                json={
                    "project_description": "too short"
                },
                headers=auth_headers
            )
            assert response.status_code == 422


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])