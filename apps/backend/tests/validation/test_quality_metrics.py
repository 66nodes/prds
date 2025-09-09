"""
Tests for content quality metrics and validation scoring.

These tests verify the quality assessment algorithms and ensure
consistent scoring across different content types and domains.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any

from backend.services.quality_metrics import (
    QualityMetricsCalculator,
    ContentQualityAnalyzer,
    ValidationScoreCalculator
)
from backend.models.quality import (
    QualityScore,
    ContentMetrics,
    ValidationMetrics
)


class TestQualityMetricsCalculator:
    """Test suite for quality metrics calculation."""

    @pytest.fixture
    def calculator(self):
        """Create quality metrics calculator."""
        return QualityMetricsCalculator()

    @pytest.fixture
    def high_quality_prd(self):
        """High quality PRD content."""
        return """
        # Product Requirements Document: E-Commerce Platform

        ## Executive Summary
        A comprehensive e-commerce platform designed for medium-sized retailers 
        to manage online sales, inventory, and customer relationships.

        ## Technical Requirements
        
        ### Architecture
        - **Frontend**: React 18 with TypeScript for type safety
        - **Backend**: Node.js 18 with Express framework
        - **Database**: PostgreSQL 15 with proper indexing strategy
        - **Caching**: Redis 7.0 for session and catalog caching
        - **Authentication**: OAuth 2.0 with JWT tokens
        
        ### Performance Requirements
        - Page load time: < 2 seconds on 3G connection
        - API response time: < 200ms for 95% of requests
        - Support for 10,000 concurrent users
        - 99.9% uptime requirement
        
        ### Security Requirements
        - PCI DSS compliance for payment processing
        - Data encryption in transit (TLS 1.3) and at rest (AES-256)
        - Regular security audits and penetration testing
        - Rate limiting: 100 requests/minute per IP
        
        ## Business Requirements
        
        ### Core Features
        1. **Product Catalog Management**
           - Bulk product import/export
           - Category management with hierarchical structure
           - Inventory tracking with low-stock alerts
           
        2. **Order Management**
           - Order processing workflow
           - Payment integration with Stripe and PayPal
           - Shipping integration with FedEx and UPS
           
        3. **Customer Management**
           - User registration and profile management
           - Order history and tracking
           - Customer support ticketing system
        
        ### Success Metrics
        - Conversion rate improvement: 15% increase
        - Average order value: $75 target
        - Customer satisfaction: 4.5/5.0 rating
        - System response time: 99% under 200ms
        
        ## Implementation Timeline
        - Phase 1 (MVP): 3 months
        - Phase 2 (Advanced features): 2 months  
        - Phase 3 (Analytics & reporting): 1 month
        
        ## Risk Assessment
        - **Technical Risk**: Medium - Complex integration requirements
        - **Timeline Risk**: Low - Realistic milestones with buffer
        - **Resource Risk**: Medium - Requires experienced developers
        """

    @pytest.fixture
    def low_quality_prd(self):
        """Low quality PRD content."""
        return """
        make website sell things online fast good looking modern tech stack
        
        use react maybe vue whatever works
        database probably mysql or postgres
        need users login and buy products
        
        should be fast and secure obvs
        mobile friendly responsive design
        payment stuff stripe paypal amazon pay google pay
        
        want lots of features:
        - shopping cart obviously 
        - product pages with pics
        - search that actually works unlike our current site
        - admin panel for managing everything
        - reports and analytics would be nice
        - social media integration why not
        
        budget around $50k timeline 2 months launch before holiday season
        
        make it look professional but not boring
        needs to handle black friday traffic without crashing
        99.99% uptime guarantee absolutely critical
        zero bugs in production ever
        """

    async def test_calculate_content_quality_high(self, calculator, high_quality_prd):
        """Test quality calculation for high-quality content."""
        quality_score = await calculator.calculate_content_quality(high_quality_prd)
        
        assert quality_score.overall_score > 0.8
        assert quality_score.structure_score > 0.8
        assert quality_score.completeness_score > 0.8
        assert quality_score.clarity_score > 0.8
        assert quality_score.technical_accuracy_score > 0.7
        
        # Should identify strong aspects
        assert "comprehensive structure" in quality_score.strengths
        assert "technical specifications" in " ".join(quality_score.strengths).lower()

    async def test_calculate_content_quality_low(self, calculator, low_quality_prd):
        """Test quality calculation for low-quality content."""
        quality_score = await calculator.calculate_content_quality(low_quality_prd)
        
        assert quality_score.overall_score < 0.5
        assert quality_score.structure_score < 0.6
        assert quality_score.completeness_score < 0.6
        assert quality_score.clarity_score < 0.5
        
        # Should identify weaknesses
        assert len(quality_score.areas_for_improvement) > 0
        issues = " ".join(quality_score.areas_for_improvement).lower()
        assert any(word in issues for word in ["structure", "clarity", "specification"])

    async def test_structure_analysis(self, calculator):
        """Test document structure analysis."""
        structured_content = """
        # Title
        ## Section 1
        ### Subsection 1.1
        Content here
        ### Subsection 1.2
        More content
        ## Section 2
        Final content
        """
        
        unstructured_content = """
        some random text without any clear structure or headings
        just a bunch of paragraphs mixed together without organization
        requirements mixed with implementation details mixed with business goals
        """
        
        structured_score = await calculator.analyze_structure(structured_content)
        unstructured_score = await calculator.analyze_structure(unstructured_content)
        
        assert structured_score.hierarchy_score > 0.8
        assert unstructured_score.hierarchy_score < 0.3
        assert structured_score.section_balance > unstructured_score.section_balance

    async def test_completeness_assessment(self, calculator):
        """Test completeness assessment."""
        complete_content = """
        # Product Requirements
        ## Executive Summary
        Clear overview of the project
        ## Technical Requirements
        Detailed technical specifications
        ## Business Requirements  
        User stories and acceptance criteria
        ## Success Metrics
        Measurable success criteria
        ## Timeline
        Implementation schedule
        ## Risk Assessment
        Identified risks and mitigation
        """
        
        incomplete_content = """
        # Some Product
        Need to build something
        Should be fast and good
        """
        
        complete_score = await calculator.assess_completeness(complete_content)
        incomplete_score = await calculator.assess_completeness(incomplete_content)
        
        assert complete_score.coverage_score > 0.8
        assert incomplete_score.coverage_score < 0.3
        
        assert len(complete_score.covered_sections) > len(incomplete_score.covered_sections)
        assert len(incomplete_score.missing_sections) > 0

    async def test_clarity_evaluation(self, calculator):
        """Test content clarity evaluation."""
        clear_content = """
        The system shall provide user authentication using OAuth 2.0 protocol.
        Authentication tokens will expire after 24 hours for security.
        Users can refresh tokens using the /auth/refresh endpoint.
        """
        
        unclear_content = """
        auth stuff should work good maybe oauth or whatever
        tokens and sessions and things need to be secure obvs
        users login somehow and stay logged in for reasonable time
        """
        
        clear_score = await calculator.evaluate_clarity(clear_content)
        unclear_score = await calculator.evaluate_clarity(unclear_content)
        
        assert clear_score.precision_score > 0.8
        assert unclear_score.precision_score < 0.4
        assert clear_score.readability_score > unclear_score.readability_score

    async def test_technical_accuracy_validation(self, calculator):
        """Test technical accuracy validation."""
        accurate_content = """
        Frontend: React 18 with TypeScript
        Backend: Node.js 18 with Express 4.x
        Database: PostgreSQL 15 with JSONB support
        Caching: Redis 7.0 for session storage
        Authentication: OAuth 2.0 with JWT tokens
        API: RESTful endpoints following OpenAPI 3.0
        """
        
        inaccurate_content = """
        Frontend: React 25 with SuperScript
        Backend: Node.js 99 with Lightning framework  
        Database: PostgreSQL 50 with quantum storage
        Caching: Redis 20 with telepathic memory
        Authentication: OAuth 5.0 with blockchain tokens
        """
        
        with patch.object(calculator.tech_validator, 'validate_technologies') as mock_validate:
            mock_validate.side_effect = [
                {"accuracy_score": 0.95, "verified_count": 6, "total_count": 6},
                {"accuracy_score": 0.1, "verified_count": 0, "total_count": 6}
            ]
            
            accurate_score = await calculator.validate_technical_accuracy(accurate_content)
            inaccurate_score = await calculator.validate_technical_accuracy(inaccurate_content)
            
            assert accurate_score.accuracy_score > 0.9
            assert inaccurate_score.accuracy_score < 0.2

    async def test_consistency_checking(self, calculator):
        """Test internal consistency checking."""
        consistent_content = """
        The system will use React for frontend development.
        React components will be written in TypeScript for type safety.
        The frontend will communicate with a Node.js backend via REST API.
        Both frontend and backend will use JavaScript/TypeScript ecosystem.
        """
        
        inconsistent_content = """
        The system will use React for frontend development.
        Vue.js components will handle the user interface.
        Angular routing will manage navigation between pages.
        The backend Python Flask API will serve data.
        PHP will handle database operations.
        """
        
        consistent_score = await calculator.check_consistency(consistent_content)
        inconsistent_score = await calculator.check_consistency(inconsistent_content)
        
        assert consistent_score.internal_consistency > 0.8
        assert inconsistent_score.internal_consistency < 0.5
        assert len(inconsistent_score.contradictions) > 0

    async def test_business_value_assessment(self, calculator):
        """Test business value assessment."""
        valuable_content = """
        ## Business Objectives
        - Increase online sales by 25% within 6 months
        - Reduce customer acquisition cost by 15%
        - Improve customer lifetime value by 30%
        
        ## Target Market
        Small to medium businesses with 10-500 employees
        Annual revenue between $1M-$50M
        Currently using spreadsheets or basic tools
        
        ## Success Metrics
        - User adoption: 80% of target users active monthly
        - Revenue impact: $2M additional ARR
        - Customer satisfaction: NPS score > 50
        - ROI: 300% within 24 months
        """
        
        vague_content = """
        make business better
        more customers and money
        users should like it
        success means good numbers
        """
        
        valuable_score = await calculator.assess_business_value(valuable_content)
        vague_score = await calculator.assess_business_value(vague_content)
        
        assert valuable_score.business_clarity > 0.8
        assert valuable_score.measurability > 0.8
        assert vague_score.business_clarity < 0.3
        assert vague_score.measurability < 0.2

    async def test_quality_trends_analysis(self, calculator):
        """Test analysis of quality trends over time."""
        historical_scores = [
            {"date": "2024-01-01", "overall_score": 0.6},
            {"date": "2024-01-02", "overall_score": 0.65},
            {"date": "2024-01-03", "overall_score": 0.7},
            {"date": "2024-01-04", "overall_score": 0.75},
            {"date": "2024-01-05", "overall_score": 0.8}
        ]
        
        trend_analysis = await calculator.analyze_quality_trends(historical_scores)
        
        assert trend_analysis.trend_direction == "improving"
        assert trend_analysis.improvement_rate > 0
        assert trend_analysis.current_quality > trend_analysis.baseline_quality


class TestContentQualityAnalyzer:
    """Test suite for content quality analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create content quality analyzer."""
        return ContentQualityAnalyzer()

    async def test_semantic_coherence_analysis(self, analyzer):
        """Test semantic coherence analysis."""
        coherent_content = """
        This document describes the requirements for an e-commerce platform.
        The platform will enable online sales and inventory management.
        Customers will be able to browse products and make purchases securely.
        The system will track orders and provide shipping updates.
        """
        
        incoherent_content = """
        This document describes a weather monitoring system.
        Users will purchase products through the shopping cart.
        The database will store temperature readings from sensors.
        Payment processing requires credit card validation.
        """
        
        coherent_score = await analyzer.analyze_semantic_coherence(coherent_content)
        incoherent_score = await analyzer.analyze_semantic_coherence(incoherent_content)
        
        assert coherent_score.coherence_score > 0.8
        assert incoherent_score.coherence_score < 0.5
        assert coherent_score.topic_consistency > incoherent_score.topic_consistency

    async def test_requirement_traceability(self, analyzer):
        """Test requirement traceability analysis."""
        traceable_content = """
        ## Business Requirements
        BR-001: System must support user registration
        BR-002: System must process payments securely
        
        ## Technical Requirements  
        TR-001: Implement user registration API (implements BR-001)
        TR-002: Integrate payment gateway (implements BR-002)
        
        ## Acceptance Criteria
        AC-001: Registration form validates email format (validates TR-001)
        AC-002: Payment processing uses HTTPS encryption (validates TR-002)
        """
        
        traceability = await analyzer.analyze_requirement_traceability(traceable_content)
        
        assert traceability.traceability_score > 0.8
        assert len(traceability.requirement_links) > 0
        assert len(traceability.orphaned_requirements) == 0

    async def test_domain_expertise_assessment(self, analyzer):
        """Test domain expertise assessment."""
        expert_content = """
        The system will implement OAuth 2.0 authorization code flow
        with PKCE extension for enhanced security. JWT tokens will
        use RS256 algorithm with proper key rotation. Database queries
        will use prepared statements to prevent SQL injection attacks.
        API endpoints will follow RESTful principles with HATEOAS
        for discoverability.
        """
        
        novice_content = """
        Users can login with username and password somehow.
        Data goes in database and comes back out when needed.
        Website should look nice and work on phones.
        Security is important so make it secure.
        """
        
        expert_score = await analyzer.assess_domain_expertise(expert_content, "technology")
        novice_score = await analyzer.assess_domain_expertise(novice_content, "technology")
        
        assert expert_score.expertise_level > 0.8
        assert novice_score.expertise_level < 0.4
        assert expert_score.technical_depth > novice_score.technical_depth

    async def test_stakeholder_alignment_analysis(self, analyzer):
        """Test stakeholder alignment analysis."""
        aligned_content = """
        ## Stakeholder Requirements
        
        ### Business Stakeholders
        - Increase revenue by 20% (CMO requirement)
        - Reduce operational costs by 15% (CFO requirement)
        - Improve customer satisfaction (Customer Success requirement)
        
        ### Technical Stakeholders  
        - System must scale to 100k users (CTO requirement)
        - 99.9% uptime requirement (DevOps requirement)
        - Maintainable codebase (Engineering requirement)
        
        ### End Users
        - Intuitive user interface (UX Research findings)
        - Fast page load times (User feedback)
        - Mobile-first experience (Analytics data)
        """
        
        alignment_score = await analyzer.analyze_stakeholder_alignment(aligned_content)
        
        assert alignment_score.overall_alignment > 0.7
        assert len(alignment_score.stakeholder_coverage) >= 3
        assert alignment_score.conflict_resolution_score > 0.6


class TestValidationScoreCalculator:
    """Test suite for validation score calculation."""

    @pytest.fixture
    def score_calculator(self):
        """Create validation score calculator."""
        return ValidationScoreCalculator()

    async def test_composite_score_calculation(self, score_calculator):
        """Test composite validation score calculation."""
        metrics = {
            "hallucination_rate": 0.02,
            "graph_coverage": 0.85,
            "confidence_score": 0.9,
            "quality_score": 0.8,
            "completeness_score": 0.75,
            "technical_accuracy": 0.95
        }
        
        weights = {
            "hallucination_rate": 0.25,  # Higher weight for critical metric
            "graph_coverage": 0.15,
            "confidence_score": 0.15,
            "quality_score": 0.15,
            "completeness_score": 0.15,
            "technical_accuracy": 0.15
        }
        
        composite_score = await score_calculator.calculate_composite_score(
            metrics, 
            weights
        )
        
        assert 0.7 < composite_score.final_score < 0.9
        assert composite_score.weighted_components is not None
        assert len(composite_score.contributing_factors) > 0

    async def test_score_normalization(self, score_calculator):
        """Test score normalization across different scales."""
        raw_scores = {
            "metric_a": 85,  # 0-100 scale
            "metric_b": 0.75,  # 0-1 scale  
            "metric_c": 4.2,  # 1-5 scale
            "metric_d": 0.03  # 0-1 scale (lower is better)
        }
        
        scale_configs = {
            "metric_a": {"min": 0, "max": 100, "higher_better": True},
            "metric_b": {"min": 0, "max": 1, "higher_better": True},
            "metric_c": {"min": 1, "max": 5, "higher_better": True},
            "metric_d": {"min": 0, "max": 1, "higher_better": False}
        }
        
        normalized_scores = await score_calculator.normalize_scores(
            raw_scores,
            scale_configs
        )
        
        # All normalized scores should be 0-1
        for score in normalized_scores.values():
            assert 0 <= score <= 1
        
        # Check specific normalizations
        assert normalized_scores["metric_a"] == 0.85  # 85/100
        assert normalized_scores["metric_b"] == 0.75  # Already normalized
        assert abs(normalized_scores["metric_c"] - 0.8) < 0.01  # (4.2-1)/(5-1)
        assert normalized_scores["metric_d"] == 0.97  # 1 - 0.03 (inverted)

    async def test_confidence_interval_calculation(self, score_calculator):
        """Test confidence interval calculation for scores."""
        validation_scores = [0.85, 0.82, 0.88, 0.84, 0.86, 0.83, 0.87, 0.85, 0.86, 0.84]
        
        confidence_interval = await score_calculator.calculate_confidence_interval(
            validation_scores,
            confidence_level=0.95
        )
        
        assert confidence_interval.mean > 0.8
        assert confidence_interval.lower_bound < confidence_interval.mean
        assert confidence_interval.upper_bound > confidence_interval.mean
        assert confidence_interval.margin_of_error > 0
        assert confidence_interval.confidence_level == 0.95

    async def test_score_degradation_detection(self, score_calculator):
        """Test detection of score degradation over time."""
        time_series_scores = [
            {"timestamp": "2024-01-01T10:00:00", "score": 0.85},
            {"timestamp": "2024-01-01T11:00:00", "score": 0.83},
            {"timestamp": "2024-01-01T12:00:00", "score": 0.80},
            {"timestamp": "2024-01-01T13:00:00", "score": 0.78},
            {"timestamp": "2024-01-01T14:00:00", "score": 0.75}
        ]
        
        degradation_analysis = await score_calculator.detect_score_degradation(
            time_series_scores,
            threshold_decline=0.05
        )
        
        assert degradation_analysis.is_degrading is True
        assert degradation_analysis.decline_rate > 0
        assert degradation_analysis.total_decline >= 0.05

    async def test_benchmark_comparison(self, score_calculator):
        """Test comparison against benchmark scores."""
        current_scores = {
            "hallucination_rate": 0.02,
            "quality_score": 0.85,
            "completeness": 0.78
        }
        
        benchmark_scores = {
            "hallucination_rate": 0.03,
            "quality_score": 0.80,
            "completeness": 0.75
        }
        
        comparison = await score_calculator.compare_to_benchmark(
            current_scores,
            benchmark_scores
        )
        
        assert comparison.overall_performance == "above_benchmark"
        assert comparison.improvements["hallucination_rate"] > 0  # Lower is better
        assert comparison.improvements["quality_score"] > 0
        assert comparison.improvements["completeness"] > 0

    async def test_score_aggregation_strategies(self, score_calculator):
        """Test different score aggregation strategies."""
        component_scores = [0.85, 0.78, 0.92, 0.81, 0.88]
        
        # Test different aggregation methods
        arithmetic_mean = await score_calculator.aggregate_scores(
            component_scores, 
            method="arithmetic_mean"
        )
        
        geometric_mean = await score_calculator.aggregate_scores(
            component_scores,
            method="geometric_mean"
        )
        
        harmonic_mean = await score_calculator.aggregate_scores(
            component_scores,
            method="harmonic_mean"
        )
        
        # Geometric mean should be less than arithmetic mean
        # Harmonic mean should be less than geometric mean
        assert harmonic_mean <= geometric_mean <= arithmetic_mean
        assert 0.7 < arithmetic_mean < 0.9


class TestQualityMetricsIntegration:
    """Integration tests for quality metrics system."""

    @pytest.fixture
    async def metrics_system(self):
        """Create complete quality metrics system."""
        from backend.services.quality_metrics_system import QualityMetricsSystem
        system = QualityMetricsSystem()
        await system.initialize()
        return system

    async def test_complete_quality_assessment(self, metrics_system):
        """Test complete quality assessment pipeline."""
        prd_content = """
        # Product Requirements: Customer Support Platform
        
        ## Executive Summary
        A comprehensive customer support platform for SaaS companies
        to manage tickets, knowledge base, and customer communications.
        
        ## Technical Architecture
        - **Frontend**: React 18 with Next.js for SSR
        - **Backend**: Node.js with Fastify framework
        - **Database**: PostgreSQL 15 with full-text search
        - **Cache**: Redis for session and query caching
        - **Search**: Elasticsearch for ticket and knowledge base search
        
        ## Core Features
        1. **Ticket Management**
           - Multi-channel support (email, chat, API)
           - SLA tracking and escalation
           - Custom fields and workflows
           
        2. **Knowledge Base**
           - Article creation and management
           - Search with auto-suggestions
           - Analytics and usage tracking
        
        ## Performance Requirements
        - Response time: < 100ms for dashboard
        - Search latency: < 200ms for knowledge base
        - Concurrent users: 5,000 support agents
        - Uptime: 99.95% availability
        
        ## Success Metrics
        - Ticket resolution time: 25% improvement
        - Customer satisfaction: CSAT > 4.5/5
        - Agent productivity: 30% increase in tickets/hour
        - Knowledge base usage: 60% self-service rate
        """
        
        assessment = await metrics_system.assess_complete_quality(
            prd_content,
            project_id="support_platform",
            domain="saas_product"
        )
        
        # Should have high overall quality
        assert assessment.overall_score > 0.75
        
        # Should pass validation thresholds
        assert assessment.validation_result.is_valid
        assert assessment.validation_result.hallucination_rate < 0.05
        
        # Should have detailed breakdowns
        assert assessment.structure_quality > 0.7
        assert assessment.technical_quality > 0.7
        assert assessment.business_quality > 0.7
        
        # Should include actionable recommendations
        assert len(assessment.recommendations) > 0
        assert assessment.improvement_priority_order is not None

    async def test_quality_monitoring_dashboard(self, metrics_system):
        """Test quality monitoring dashboard data."""
        project_id = "monitoring_test"
        
        # Simulate multiple assessments over time
        assessment_data = []
        for i in range(10):
            mock_assessment = {
                "timestamp": datetime.now() - timedelta(days=i),
                "overall_score": 0.8 + (i * 0.02),  # Improving trend
                "hallucination_rate": 0.05 - (i * 0.003),  # Decreasing trend
                "technical_accuracy": 0.85 + (i * 0.01)
            }
            assessment_data.append(mock_assessment)
        
        dashboard_data = await metrics_system.generate_dashboard_data(
            project_id,
            assessment_data
        )
        
        assert "quality_trends" in dashboard_data
        assert "performance_metrics" in dashboard_data
        assert "alert_summary" in dashboard_data
        
        # Should detect improving trends
        trends = dashboard_data["quality_trends"]
        assert trends["overall_quality"]["direction"] == "improving"
        assert trends["hallucination_rate"]["direction"] == "improving"

    async def test_automated_quality_alerts(self, metrics_system):
        """Test automated quality alert system."""
        # Simulate quality degradation
        declining_scores = [
            {"timestamp": "2024-01-01", "overall_score": 0.85},
            {"timestamp": "2024-01-02", "overall_score": 0.80},
            {"timestamp": "2024-01-03", "overall_score": 0.75},
            {"timestamp": "2024-01-04", "overall_score": 0.65}  # Significant drop
        ]
        
        alerts = await metrics_system.check_quality_alerts(
            "alert_test_project",
            declining_scores
        )
        
        assert len(alerts) > 0
        
        # Should have quality degradation alert
        alert_types = [alert["type"] for alert in alerts]
        assert "quality_degradation" in alert_types
        
        # Should have specific threshold alerts
        critical_alert = next(
            (a for a in alerts if a["severity"] == "critical"), 
            None
        )
        assert critical_alert is not None