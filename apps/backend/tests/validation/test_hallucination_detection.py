"""
Comprehensive tests for hallucination detection and validation.

These tests verify that the GraphRAG system correctly identifies
hallucinated content and maintains quality thresholds.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import Dict, List, Any

# Import services (adjust paths based on actual structure)
from backend.services.graphrag_services import (
    HallucinationDetector,
    EntityExtractor,
    FactChecker
)
from backend.models.validation import ValidationResult
from backend.core.exceptions import HallucinationThresholdExceeded


class TestHallucinationDetector:
    """Test suite for hallucination detection service."""

    @pytest.fixture
    async def detector(self):
        """Create hallucination detector instance."""
        detector = HallucinationDetector()
        await detector.initialize()
        return detector

    @pytest.fixture
    def valid_content(self):
        """Sample content that should pass validation."""
        return """
        This PRD describes a task management system with the following features:
        - User authentication using OAuth 2.0
        - Task creation and assignment
        - Real-time notifications via WebSocket
        - Data storage using PostgreSQL
        - RESTful API following OpenAPI 3.0 specification
        """

    @pytest.fixture
    def hallucinated_content(self):
        """Sample content with hallucinated information."""
        return """
        This PRD describes a revolutionary quantum-computing task management system:
        - Uses quantum encryption with 2048-qubit processors
        - Processes 1 billion tasks per nanosecond
        - Telepathic user interface via brain-computer interface
        - Runs on proprietary FasterThanLight™ processors
        - Guaranteed 100% uptime using time-travel backup systems
        """

    @pytest.fixture
    def mixed_content(self):
        """Content with both valid and questionable claims."""
        return """
        This PRD describes a modern web application:
        - Frontend built with React 18
        - Backend API using Node.js and Express
        - Database: MongoDB with automatic sharding
        - Authentication via JWT tokens
        - Deployment on AWS with 99.999% uptime guarantee
        - AI-powered sentiment analysis with 100% accuracy
        - Real-time collaboration for unlimited users
        """

    @pytest.fixture
    def graph_data_valid(self):
        """Mock graph data supporting valid claims."""
        return {
            "entities": [
                {"name": "OAuth 2.0", "type": "authentication_protocol", "verified": True},
                {"name": "WebSocket", "type": "communication_protocol", "verified": True},
                {"name": "PostgreSQL", "type": "database", "verified": True},
                {"name": "OpenAPI 3.0", "type": "api_specification", "verified": True}
            ],
            "relationships": [
                {"from": "OAuth 2.0", "to": "authentication", "relationship": "implements"},
                {"from": "PostgreSQL", "to": "data_storage", "relationship": "provides"}
            ]
        }

    @pytest.fixture
    def graph_data_invalid(self):
        """Mock graph data showing no support for claims."""
        return {
            "entities": [
                {"name": "quantum-computing", "type": "technology", "verified": False},
                {"name": "2048-qubit processors", "type": "hardware", "verified": False},
                {"name": "telepathic interface", "type": "interface", "verified": False}
            ],
            "relationships": []
        }

    async def test_validate_content_success(self, detector, valid_content, graph_data_valid):
        """Test successful validation of valid content."""
        project_id = "test_project_123"
        
        # Mock graph query
        with patch.object(detector.graph_client, 'query_entities') as mock_query:
            mock_query.return_value = graph_data_valid
            
            result = await detector.validate_content(valid_content, project_id)
            
            assert result.is_valid
            assert result.hallucination_rate < 0.02  # Under 2% threshold
            assert result.confidence_score > 0.8
            assert len(result.supported_claims) > 0
            assert len(result.unsupported_claims) == 0

    async def test_validate_content_hallucination_detected(
        self, 
        detector, 
        hallucinated_content, 
        graph_data_invalid
    ):
        """Test detection of hallucinated content."""
        project_id = "test_project_123"
        
        with patch.object(detector.graph_client, 'query_entities') as mock_query:
            mock_query.return_value = graph_data_invalid
            
            result = await detector.validate_content(hallucinated_content, project_id)
            
            assert not result.is_valid
            assert result.hallucination_rate > 0.5  # High hallucination rate
            assert result.confidence_score < 0.3
            assert len(result.unsupported_claims) > 0
            
            # Check that specific hallucinated terms are flagged
            unsupported_text = " ".join(result.unsupported_claims)
            assert "quantum-computing" in unsupported_text or "telepathic" in unsupported_text

    async def test_validate_content_mixed_quality(self, detector, mixed_content):
        """Test validation of content with mixed quality."""
        project_id = "test_project_123"
        
        # Mock partial graph support
        mixed_graph_data = {
            "entities": [
                {"name": "React 18", "type": "framework", "verified": True},
                {"name": "Node.js", "type": "runtime", "verified": True},
                {"name": "MongoDB", "type": "database", "verified": True},
                {"name": "JWT", "type": "token_standard", "verified": True},
                {"name": "99.999% uptime", "type": "claim", "verified": False},
                {"name": "100% accuracy", "type": "claim", "verified": False}
            ]
        }
        
        with patch.object(detector.graph_client, 'query_entities') as mock_query:
            mock_query.return_value = mixed_graph_data
            
            result = await detector.validate_content(mixed_content, project_id)
            
            # Should have moderate hallucination rate
            assert 0.1 < result.hallucination_rate < 0.4
            assert len(result.supported_claims) > 0
            assert len(result.unsupported_claims) > 0

    async def test_threshold_enforcement(self, detector, hallucinated_content):
        """Test that validation enforces hallucination thresholds."""
        project_id = "test_project_123"
        
        with patch.object(detector.graph_client, 'query_entities') as mock_query:
            mock_query.return_value = {"entities": [], "relationships": []}
            
            with pytest.raises(HallucinationThresholdExceeded) as exc_info:
                await detector.validate_with_threshold(
                    hallucinated_content, 
                    project_id, 
                    max_hallucination_rate=0.02
                )
            
            assert "exceeds threshold" in str(exc_info.value)
            assert "2%" in str(exc_info.value)

    async def test_entity_extraction(self, detector, valid_content):
        """Test entity extraction from content."""
        entities = await detector.extract_entities(valid_content)
        
        assert len(entities) > 0
        
        # Should extract technical terms
        entity_names = [e['name'].lower() for e in entities]
        assert any('oauth' in name for name in entity_names)
        assert any('postgresql' in name for name in entity_names)
        assert any('websocket' in name for name in entity_names)

    async def test_fact_checking_integration(self, detector, valid_content):
        """Test integration with fact checking service."""
        project_id = "test_project_123"
        
        with patch.object(detector.fact_checker, 'verify_claims') as mock_verify:
            mock_verify.return_value = {
                "verified_claims": 4,
                "total_claims": 5,
                "confidence": 0.85
            }
            
            result = await detector.validate_content(valid_content, project_id)
            
            mock_verify.assert_called_once()
            assert result.confidence_score > 0.8

    async def test_performance_within_limits(self, detector, valid_content):
        """Test that validation completes within performance limits."""
        import time
        
        project_id = "test_project_123"
        
        start_time = time.time()
        result = await detector.validate_content(valid_content, project_id)
        end_time = time.time()
        
        validation_time = end_time - start_time
        
        # Should complete within 5 seconds
        assert validation_time < 5.0
        assert result is not None

    async def test_batch_validation(self, detector):
        """Test batch validation of multiple contents."""
        contents = [
            "Valid content with React and Node.js",
            "Invalid content with quantum telepathy",
            "Mixed content with React and 100% accuracy claims"
        ]
        project_id = "test_project_123"
        
        results = await detector.validate_batch(contents, project_id)
        
        assert len(results) == 3
        assert results[0].hallucination_rate < results[1].hallucination_rate
        assert all(r.project_id == project_id for r in results)

    async def test_validation_caching(self, detector, valid_content):
        """Test that validation results are cached for performance."""
        project_id = "test_project_123"
        
        # First validation
        result1 = await detector.validate_content(valid_content, project_id)
        
        # Second validation of same content
        with patch.object(detector.graph_client, 'query_entities') as mock_query:
            result2 = await detector.validate_content(valid_content, project_id)
            
            # Should use cache, not call graph again
            mock_query.assert_not_called()
            
            assert result1.hallucination_rate == result2.hallucination_rate

    async def test_error_handling_graph_failure(self, detector, valid_content):
        """Test handling of graph database failures."""
        project_id = "test_project_123"
        
        with patch.object(detector.graph_client, 'query_entities') as mock_query:
            mock_query.side_effect = Exception("Graph database connection failed")
            
            # Should handle gracefully with fallback validation
            result = await detector.validate_content(valid_content, project_id)
            
            assert result is not None
            assert hasattr(result, 'fallback_used')
            assert result.fallback_used is True

    async def test_language_specific_validation(self, detector):
        """Test validation works with different languages."""
        spanish_content = """
        Esta PRD describe un sistema de gestión de tareas:
        - Autenticación de usuarios con OAuth 2.0
        - Base de datos PostgreSQL
        - Interfaz web responsive
        """
        
        project_id = "test_project_123"
        result = await detector.validate_content(spanish_content, project_id)
        
        # Should still extract entities and validate
        assert result is not None
        assert isinstance(result.hallucination_rate, float)


class TestEntityExtractor:
    """Test suite for entity extraction service."""

    @pytest.fixture
    def extractor(self):
        """Create entity extractor instance."""
        return EntityExtractor()

    async def test_extract_technical_entities(self, extractor):
        """Test extraction of technical entities."""
        content = """
        The system uses React 18, Node.js 16, PostgreSQL 14, and Redis 6.
        API follows RESTful principles with OpenAPI 3.0 specification.
        Authentication via OAuth 2.0 with JWT tokens.
        """
        
        entities = await extractor.extract(content)
        
        assert len(entities) > 5
        
        # Should extract versions and technologies
        entity_names = [e['name'] for e in entities]
        assert "React 18" in entity_names
        assert "Node.js 16" in entity_names
        assert "PostgreSQL 14" in entity_names
        assert "OAuth 2.0" in entity_names

    async def test_extract_business_entities(self, extractor):
        """Test extraction of business-related entities."""
        content = """
        The target market includes small businesses with 10-50 employees.
        Expected revenue of $2M ARR within 18 months.
        Pricing starts at $29/month per user.
        Competitors include Asana, Trello, and Monday.com.
        """
        
        entities = await extractor.extract(content)
        
        # Should extract business metrics and competitors
        entity_names = [e['name'] for e in entities]
        assert any('$2M ARR' in name for name in entity_names)
        assert any('Asana' in name for name in entity_names)
        assert any('$29/month' in name for name in entity_names)

    async def test_entity_categorization(self, extractor):
        """Test that entities are properly categorized."""
        content = "Use PostgreSQL database with OAuth 2.0 authentication."
        
        entities = await extractor.extract(content)
        
        # Find PostgreSQL entity
        pg_entity = next((e for e in entities if 'PostgreSQL' in e['name']), None)
        assert pg_entity is not None
        assert pg_entity['category'] in ['database', 'technology']
        
        # Find OAuth entity
        oauth_entity = next((e for e in entities if 'OAuth' in e['name']), None)
        assert oauth_entity is not None
        assert oauth_entity['category'] in ['authentication', 'protocol']


class TestFactChecker:
    """Test suite for fact checking service."""

    @pytest.fixture
    def fact_checker(self):
        """Create fact checker instance."""
        return FactChecker()

    async def test_verify_technical_claims(self, fact_checker):
        """Test verification of technical claims."""
        claims = [
            "React is a JavaScript library",
            "PostgreSQL supports ACID transactions", 
            "OAuth 2.0 is an authorization framework",
            "WebSocket enables real-time communication"
        ]
        
        results = await fact_checker.verify_claims(claims, "technology")
        
        assert results['verified_claims'] >= 3
        assert results['confidence'] > 0.8
        assert len(results['details']) == len(claims)

    async def test_detect_impossible_claims(self, fact_checker):
        """Test detection of impossible or exaggerated claims."""
        claims = [
            "System processes 1 billion requests per nanosecond",
            "Achieves 100% accuracy on all predictions",
            "Guarantees zero bugs in all code",
            "Runs on quantum computers with telepathic interface"
        ]
        
        results = await fact_checker.verify_claims(claims, "technology")
        
        assert results['verified_claims'] == 0
        assert results['confidence'] < 0.2
        assert all(not detail['verified'] for detail in results['details'])

    async def test_verify_performance_claims(self, fact_checker):
        """Test verification of performance-related claims."""
        claims = [
            "API response time under 200ms",
            "99.9% uptime guarantee",
            "Supports 10,000 concurrent users",
            "99.999% uptime guarantee"  # Questionable claim
        ]
        
        results = await fact_checker.verify_claims(claims, "performance")
        
        # First three should be reasonable, last one questionable
        assert 2 <= results['verified_claims'] <= 3
        
        # Check specific claim verification
        details = {d['claim']: d['verified'] for d in results['details']}
        assert details["API response time under 200ms"] is True
        assert details["99.9% uptime guarantee"] is True


class TestValidationIntegration:
    """Integration tests for the complete validation pipeline."""

    @pytest.fixture
    async def validation_service(self):
        """Create complete validation service."""
        from backend.services.validation_service import ValidationService
        service = ValidationService()
        await service.initialize()
        return service

    async def test_end_to_end_validation(self, validation_service):
        """Test complete validation pipeline."""
        prd_content = """
        Product Requirements Document: Task Management System
        
        Overview:
        A modern web-based task management application for small teams.
        
        Technical Requirements:
        - Frontend: React 18 with TypeScript
        - Backend: Node.js with Express framework
        - Database: PostgreSQL with proper indexing
        - Authentication: OAuth 2.0 with JWT tokens
        - Real-time updates: WebSocket connections
        
        Performance Requirements:
        - API response time: < 200ms for 95% of requests
        - Support for 1,000 concurrent users
        - 99.9% uptime target
        
        Security:
        - All data encrypted in transit and at rest
        - Regular security audits
        - GDPR compliant data handling
        """
        
        project_id = "integration_test_project"
        
        result = await validation_service.validate_prd(prd_content, project_id)
        
        assert result.is_valid
        assert result.hallucination_rate < 0.05
        assert result.confidence_score > 0.85
        assert len(result.supported_claims) > 5
        
        # Should have detailed validation report
        assert result.validation_report is not None
        assert 'technical_accuracy' in result.validation_report
        assert 'performance_feasibility' in result.validation_report

    async def test_validation_with_citations(self, validation_service):
        """Test that validation includes proper citations."""
        content = "Use PostgreSQL for data storage and Redis for caching."
        project_id = "test_project"
        
        result = await validation_service.validate_prd(content, project_id)
        
        assert result.citations is not None
        assert len(result.citations) > 0
        
        # Should have citations for technical claims
        citation_texts = [c['text'] for c in result.citations]
        assert any('PostgreSQL' in text for text in citation_texts)

    async def test_validation_performance_benchmark(self, validation_service):
        """Test validation performance meets benchmarks."""
        import time
        
        # Large PRD content
        large_content = """
        This is a comprehensive PRD for a large-scale application.
        """ + " ".join([f"Requirement {i}: {' '.join(['feature'] * 20)}" for i in range(100)])
        
        project_id = "performance_test"
        
        start_time = time.time()
        result = await validation_service.validate_prd(large_content, project_id)
        end_time = time.time()
        
        validation_time = end_time - start_time
        
        # Should complete within reasonable time even for large content
        assert validation_time < 30.0  # 30 seconds max
        assert result is not None

    async def test_concurrent_validations(self, validation_service):
        """Test handling of concurrent validation requests."""
        contents = [
            f"PRD content number {i} with React and PostgreSQL"
            for i in range(10)
        ]
        project_id = "concurrent_test"
        
        # Run validations concurrently
        import asyncio
        tasks = [
            validation_service.validate_prd(content, project_id)
            for content in contents
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        assert all(r is not None for r in results)
        assert all(r.hallucination_rate < 1.0 for r in results)


class TestValidationMetrics:
    """Test validation metrics and reporting."""

    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector."""
        from backend.services.validation_metrics import ValidationMetricsCollector
        return ValidationMetricsCollector()

    async def test_collect_validation_metrics(self, metrics_collector):
        """Test collection of validation metrics."""
        # Sample validation results
        results = [
            ValidationResult(
                is_valid=True,
                hallucination_rate=0.01,
                confidence_score=0.9,
                validation_time=2.5
            ),
            ValidationResult(
                is_valid=False,
                hallucination_rate=0.15,
                confidence_score=0.3,
                validation_time=3.2
            )
        ]
        
        metrics = await metrics_collector.collect_metrics(results)
        
        assert metrics['total_validations'] == 2
        assert metrics['success_rate'] == 0.5
        assert metrics['avg_hallucination_rate'] == 0.08
        assert metrics['avg_confidence_score'] == 0.6
        assert metrics['avg_validation_time'] == 2.85

    async def test_quality_trend_analysis(self, metrics_collector):
        """Test quality trend analysis over time."""
        # Simulate improving quality over time
        historical_data = [
            {'date': '2024-01-01', 'avg_hallucination_rate': 0.08},
            {'date': '2024-01-02', 'avg_hallucination_rate': 0.06},
            {'date': '2024-01-03', 'avg_hallucination_rate': 0.04},
            {'date': '2024-01-04', 'avg_hallucination_rate': 0.03}
        ]
        
        trend = await metrics_collector.analyze_quality_trend(historical_data)
        
        assert trend['direction'] == 'improving'
        assert trend['improvement_rate'] > 0
        assert trend['current_quality_score'] > trend['initial_quality_score']