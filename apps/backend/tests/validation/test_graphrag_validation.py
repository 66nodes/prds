"""
Tests for GraphRAG validation system integration.

Validates the complete GraphRAG pipeline including knowledge graph
operations, embedding generation, and semantic validation.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import Dict, List, Any

from backend.services.graphrag_validator import GraphRAGValidator
from backend.models.graph import GraphNode, GraphRelationship
from backend.core.exceptions import ValidationError, GraphConnectionError


class TestGraphRAGValidator:
    """Test suite for GraphRAG validation system."""

    @pytest.fixture
    async def validator(self):
        """Create GraphRAG validator instance."""
        validator = GraphRAGValidator()
        await validator.initialize()
        return validator

    @pytest.fixture
    def sample_knowledge_graph(self):
        """Sample knowledge graph data."""
        return {
            "nodes": [
                {
                    "id": "react-18",
                    "type": "technology",
                    "properties": {
                        "name": "React 18",
                        "category": "frontend_framework",
                        "release_date": "2022-03-29",
                        "verified": True,
                        "popularity_score": 0.95
                    }
                },
                {
                    "id": "postgresql-14",
                    "type": "database",
                    "properties": {
                        "name": "PostgreSQL 14",
                        "category": "relational_database",
                        "release_date": "2021-10-14",
                        "verified": True,
                        "popularity_score": 0.88
                    }
                },
                {
                    "id": "oauth-2",
                    "type": "protocol",
                    "properties": {
                        "name": "OAuth 2.0",
                        "category": "authentication_protocol",
                        "standardized": True,
                        "verified": True,
                        "security_rating": "high"
                    }
                }
            ],
            "relationships": [
                {
                    "from": "react-18",
                    "to": "frontend_development",
                    "type": "USED_FOR",
                    "weight": 0.9
                },
                {
                    "from": "postgresql-14",
                    "to": "data_storage",
                    "type": "PROVIDES",
                    "weight": 0.95
                },
                {
                    "from": "oauth-2",
                    "to": "authentication",
                    "type": "IMPLEMENTS",
                    "weight": 0.92
                }
            ]
        }

    @pytest.fixture
    def content_with_entities(self):
        """Content with extractable entities."""
        return """
        This system will use React 18 for the frontend interface,
        PostgreSQL 14 for data persistence, and OAuth 2.0 for
        user authentication. The API will follow RESTful principles
        with OpenAPI 3.0 specification for documentation.
        """

    @pytest.fixture
    def content_with_hallucinations(self):
        """Content containing hallucinated information."""
        return """
        Our revolutionary system uses QuantumReact 25.0, a framework
        that processes UI updates at the speed of light using quantum
        entanglement. Data is stored in TelepathicDB, which can read
        your thoughts to predict queries. Authentication is handled
        by MindMeld 5.0, eliminating the need for passwords entirely.
        """

    async def test_validate_content_against_graph(
        self,
        validator,
        content_with_entities,
        sample_knowledge_graph
    ):
        """Test validation of content against knowledge graph."""
        project_id = "test_project_123"
        
        # Mock graph query response
        with patch.object(validator.graph_client, 'query_subgraph') as mock_query:
            mock_query.return_value = sample_knowledge_graph
            
            result = await validator.validate_against_graph(
                content_with_entities,
                project_id
            )
            
            assert result.is_valid
            assert result.graph_coverage > 0.7  # Good coverage
            assert len(result.verified_entities) > 0
            assert result.confidence_score > 0.8
            
            # Should identify supported technologies
            verified_names = [e['name'] for e in result.verified_entities]
            assert "React 18" in verified_names
            assert "PostgreSQL 14" in verified_names
            assert "OAuth 2.0" in verified_names

    async def test_detect_hallucinated_entities(
        self,
        validator,
        content_with_hallucinations
    ):
        """Test detection of hallucinated entities not in graph."""
        project_id = "test_project_123"
        
        # Mock empty graph response (no matching entities)
        with patch.object(validator.graph_client, 'query_subgraph') as mock_query:
            mock_query.return_value = {"nodes": [], "relationships": []}
            
            result = await validator.validate_against_graph(
                content_with_hallucinations,
                project_id
            )
            
            assert not result.is_valid
            assert result.graph_coverage < 0.1  # Poor coverage
            assert len(result.unverified_entities) > 0
            assert result.confidence_score < 0.3
            
            # Should flag hallucinated technologies
            unverified_names = [e['name'] for e in result.unverified_entities]
            assert any("QuantumReact" in name for name in unverified_names)
            assert any("TelepathicDB" in name for name in unverified_names)

    async def test_semantic_similarity_validation(self, validator):
        """Test semantic similarity validation using embeddings."""
        reference_content = "React is a JavaScript library for building user interfaces."
        test_content = "React helps developers create interactive web applications."
        
        similarity_score = await validator.calculate_semantic_similarity(
            reference_content,
            test_content
        )
        
        assert 0.7 < similarity_score < 1.0  # Should be semantically similar
        
        # Test with completely different content
        different_content = "PostgreSQL is a relational database management system."
        
        diff_similarity = await validator.calculate_semantic_similarity(
            reference_content,
            different_content
        )
        
        assert diff_similarity < similarity_score  # Should be less similar

    async def test_knowledge_graph_expansion(self, validator, sample_knowledge_graph):
        """Test dynamic expansion of knowledge graph."""
        new_entity = {
            "name": "Next.js 13",
            "category": "fullstack_framework",
            "based_on": "React 18",
            "verified": True
        }
        
        project_id = "test_project_123"
        
        with patch.object(validator.graph_client, 'add_entity') as mock_add:
            await validator.expand_knowledge_graph(new_entity, project_id)
            
            mock_add.assert_called_once()
            
            # Should create relationship to React
            call_args = mock_add.call_args[0]
            entity_data = call_args[0]
            
            assert entity_data["name"] == "Next.js 13"
            assert entity_data["category"] == "fullstack_framework"

    async def test_fact_verification_with_sources(self, validator):
        """Test fact verification with source attribution."""
        claim = "React 18 introduced automatic batching for better performance"
        
        with patch.object(validator.source_verifier, 'verify_claim') as mock_verify:
            mock_verify.return_value = {
                "verified": True,
                "confidence": 0.92,
                "sources": [
                    {
                        "url": "https://reactjs.org/blog/2022/03/29/react-v18.html",
                        "title": "React v18.0",
                        "credibility": 0.98
                    }
                ]
            }
            
            result = await validator.verify_fact(claim)
            
            assert result["verified"] is True
            assert result["confidence"] > 0.9
            assert len(result["sources"]) > 0
            assert result["sources"][0]["credibility"] > 0.9

    async def test_temporal_consistency_validation(self, validator):
        """Test validation of temporal consistency in claims."""
        content_with_dates = """
        This system was built using React 19 (released in 2021)
        and will integrate with the upcoming PostgreSQL 16 features
        that were announced in 2020.
        """
        
        result = await validator.validate_temporal_consistency(content_with_dates)
        
        assert not result.is_consistent
        assert len(result.temporal_conflicts) > 0
        
        # Should identify the date inconsistencies
        conflicts = result.temporal_conflicts
        assert any("React 19" in str(conflict) for conflict in conflicts)

    async def test_cross_domain_validation(self, validator):
        """Test validation across different knowledge domains."""
        content = """
        The medical device will use React for the user interface,
        PostgreSQL for patient data storage, and comply with HIPAA
        regulations for healthcare data protection.
        """
        
        project_id = "medical_project"
        
        result = await validator.validate_cross_domain(content, project_id)
        
        # Should validate both technical and regulatory domains
        assert "technology" in result.validated_domains
        assert "healthcare_compliance" in result.validated_domains
        assert result.cross_domain_consistency_score > 0.7

    async def test_performance_metrics_tracking(self, validator):
        """Test that validation performance is tracked."""
        content = "Simple content for performance testing."
        project_id = "perf_test"
        
        result = await validator.validate_against_graph(content, project_id)
        
        assert hasattr(result, 'performance_metrics')
        metrics = result.performance_metrics
        
        assert 'validation_time_ms' in metrics
        assert 'graph_query_time_ms' in metrics
        assert 'embedding_time_ms' in metrics
        assert metrics['validation_time_ms'] > 0

    async def test_batch_validation_optimization(self, validator):
        """Test optimized batch validation of multiple contents."""
        contents = [
            "React application with PostgreSQL database",
            "Vue.js frontend with MongoDB storage", 
            "Angular app using MySQL database"
        ]
        project_id = "batch_test"
        
        # Mock batch processing
        with patch.object(validator.graph_client, 'batch_query') as mock_batch:
            mock_batch.return_value = [
                {"coverage": 0.8, "verified": True},
                {"coverage": 0.6, "verified": True},
                {"coverage": 0.7, "verified": True}
            ]
            
            results = await validator.validate_batch(contents, project_id)
            
            assert len(results) == 3
            assert all(r.is_valid for r in results)
            mock_batch.assert_called_once()  # Single batch call

    async def test_confidence_calibration(self, validator):
        """Test that confidence scores are properly calibrated."""
        # High confidence case
        high_conf_content = "Use React 18 and PostgreSQL 14 for development."
        
        with patch.object(validator.graph_client, 'query_subgraph') as mock_query:
            mock_query.return_value = {
                "nodes": [
                    {"properties": {"name": "React 18", "verified": True, "popularity_score": 0.95}},
                    {"properties": {"name": "PostgreSQL 14", "verified": True, "popularity_score": 0.88}}
                ]
            }
            
            high_result = await validator.validate_against_graph(high_conf_content, "test")
            
            # Low confidence case - unknown technologies
            mock_query.return_value = {"nodes": []}
            low_conf_content = "Use UnknownTech 99 and MysteryDB 2000."
            
            low_result = await validator.validate_against_graph(low_conf_content, "test")
            
            assert high_result.confidence_score > low_result.confidence_score
            assert high_result.confidence_score > 0.8
            assert low_result.confidence_score < 0.4

    async def test_graph_relationship_validation(self, validator):
        """Test validation of relationships between entities."""
        content = """
        Use React for frontend and Express for backend API.
        Both run on Node.js runtime environment.
        """
        
        mock_graph = {
            "nodes": [
                {"id": "react", "properties": {"name": "React", "type": "frontend"}},
                {"id": "express", "properties": {"name": "Express", "type": "backend"}},
                {"id": "nodejs", "properties": {"name": "Node.js", "type": "runtime"}}
            ],
            "relationships": [
                {"from": "react", "to": "nodejs", "type": "RUNS_ON"},
                {"from": "express", "to": "nodejs", "type": "RUNS_ON"}
            ]
        }
        
        with patch.object(validator.graph_client, 'query_subgraph') as mock_query:
            mock_query.return_value = mock_graph
            
            result = await validator.validate_against_graph(content, "test")
            
            assert result.relationship_consistency_score > 0.8
            assert len(result.validated_relationships) > 0

    async def test_error_handling_graph_unavailable(self, validator):
        """Test graceful handling when graph is unavailable."""
        content = "React application with database storage"
        
        with patch.object(validator.graph_client, 'query_subgraph') as mock_query:
            mock_query.side_effect = GraphConnectionError("Neo4j unavailable")
            
            # Should not raise exception, use fallback validation
            result = await validator.validate_against_graph(content, "test")
            
            assert result.is_fallback_validation
            assert result.validation_method == "fallback"
            assert result.confidence_score < 0.5  # Lower confidence for fallback

    async def test_custom_domain_knowledge_injection(self, validator):
        """Test injection of custom domain-specific knowledge."""
        domain_knowledge = {
            "domain": "fintech",
            "entities": [
                {"name": "PCI DSS", "type": "compliance_standard"},
                {"name": "PSD2", "type": "regulation"},
                {"name": "Stripe API", "type": "payment_service"}
            ],
            "relationships": [
                {"from": "Stripe API", "to": "PCI DSS", "type": "COMPLIES_WITH"}
            ]
        }
        
        await validator.inject_domain_knowledge("fintech_project", domain_knowledge)
        
        # Test that domain knowledge is used in validation
        content = "Payment processing using Stripe API with PCI DSS compliance"
        
        result = await validator.validate_against_graph(content, "fintech_project")
        
        # Should recognize domain-specific entities
        verified_names = [e['name'] for e in result.verified_entities]
        assert "Stripe API" in verified_names
        assert "PCI DSS" in verified_names


class TestGraphRAGMetrics:
    """Test GraphRAG validation metrics and analytics."""

    @pytest.fixture
    def metrics_service(self):
        """Create metrics service."""
        from backend.services.graphrag_metrics import GraphRAGMetrics
        return GraphRAGMetrics()

    async def test_validation_quality_metrics(self, metrics_service):
        """Test calculation of validation quality metrics."""
        validation_results = [
            {
                "project_id": "proj1",
                "hallucination_rate": 0.02,
                "graph_coverage": 0.85,
                "confidence_score": 0.9,
                "validation_time": 2.5
            },
            {
                "project_id": "proj1", 
                "hallucination_rate": 0.08,
                "graph_coverage": 0.70,
                "confidence_score": 0.7,
                "validation_time": 3.2
            }
        ]
        
        metrics = await metrics_service.calculate_quality_metrics(validation_results)
        
        assert metrics["avg_hallucination_rate"] == 0.05
        assert metrics["avg_graph_coverage"] == 0.775
        assert metrics["avg_confidence_score"] == 0.8
        assert metrics["quality_trend"] in ["stable", "improving", "declining"]

    async def test_graph_health_monitoring(self, metrics_service):
        """Test monitoring of graph database health."""
        health_status = await metrics_service.check_graph_health()
        
        assert "connection_status" in health_status
        assert "node_count" in health_status
        assert "relationship_count" in health_status
        assert "query_performance" in health_status
        
        # Health should be boolean or score
        assert isinstance(health_status["connection_status"], (bool, float))

    async def test_knowledge_gap_analysis(self, metrics_service):
        """Test analysis of knowledge gaps in the graph."""
        unverified_entities = [
            "UnknownFramework 1.0",
            "MysteryLibrary 2.5", 
            "SecretAPI v3"
        ]
        
        gap_analysis = await metrics_service.analyze_knowledge_gaps(
            unverified_entities,
            domain="technology"
        )
        
        assert "gap_count" in gap_analysis
        assert "critical_gaps" in gap_analysis
        assert "recommendations" in gap_analysis
        assert gap_analysis["gap_count"] == 3


class TestGraphRAGIntegration:
    """Integration tests for complete GraphRAG system."""

    @pytest.fixture
    async def complete_system(self):
        """Set up complete GraphRAG validation system."""
        from backend.services.graphrag_system import GraphRAGSystem
        system = GraphRAGSystem()
        await system.initialize()
        return system

    async def test_end_to_end_prd_validation(self, complete_system):
        """Test complete PRD validation through GraphRAG system."""
        prd_content = """
        Product Requirements Document: Task Management Platform
        
        Technology Stack:
        - Frontend: React 18 with TypeScript
        - Backend: Node.js 18 with Express framework  
        - Database: PostgreSQL 15 with proper indexing
        - Authentication: OAuth 2.0 with JWT tokens
        - Real-time: WebSocket connections
        - Caching: Redis 7.0 for session management
        
        Architecture:
        - Microservices architecture using Docker containers
        - API Gateway for request routing
        - Message queue using RabbitMQ for async processing
        - CDN for static asset delivery
        
        Performance:
        - API response time < 200ms for 95% of requests
        - Support 10,000 concurrent users
        - 99.9% uptime SLA
        
        Security:
        - Data encryption in transit and at rest
        - Rate limiting and DDoS protection
        - Regular security audits and penetration testing
        """
        
        project_id = "integration_test"
        
        validation_result = await complete_system.validate_prd(prd_content, project_id)
        
        # Should pass validation with high confidence
        assert validation_result.is_valid
        assert validation_result.hallucination_rate < 0.05
        assert validation_result.confidence_score > 0.85
        
        # Should have comprehensive validation details
        assert len(validation_result.verified_entities) >= 8
        assert validation_result.graph_coverage > 0.8
        assert len(validation_result.validated_relationships) > 0
        
        # Should include proper citations and sources
        assert validation_result.citations is not None
        assert len(validation_result.citations) > 0

    async def test_system_performance_under_load(self, complete_system):
        """Test system performance with concurrent validations."""
        import asyncio
        import time
        
        # Create multiple PRD contents
        prd_templates = [
            "React application with PostgreSQL database and Redis caching",
            "Vue.js frontend with MongoDB storage and Express backend",
            "Angular app using MySQL database and Node.js API",
            "Python Django application with PostgreSQL and Celery",
            "Ruby on Rails app with PostgreSQL and Sidekiq"
        ]
        
        start_time = time.time()
        
        # Run validations concurrently
        tasks = [
            complete_system.validate_prd(content, f"project_{i}")
            for i, content in enumerate(prd_templates)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete all validations within reasonable time
        assert total_time < 30.0  # 30 seconds for 5 validations
        
        # All validations should succeed
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 5
        
        # All should have reasonable quality scores
        assert all(r.confidence_score > 0.6 for r in successful_results)