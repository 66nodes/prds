"""
Unit tests for GraphRAG services.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import numpy as np

from services.graphrag.hallucination_detector import HallucinationDetector, ValidationResult
from services.graphrag.entity_extractor import EntityExtractor
from services.graphrag.relationship_extractor import RelationshipExtractor
from services.graphrag.graph_service import GraphService
from services.graphrag.fact_checker import FactChecker


class TestHallucinationDetector:
    """Test suite for HallucinationDetector."""

    @pytest.fixture
    def detector(self):
        """Create a HallucinationDetector instance for testing."""
        with patch('services.graphrag.graph_service.GraphService'):
            return HallucinationDetector()

    @pytest.fixture
    def sample_content(self):
        """Sample content for testing."""
        return """
        The Strategic Planning Platform is an AI-powered system that helps organizations 
        create comprehensive Product Requirements Documents (PRDs) using advanced 
        GraphRAG technology. The system maintains a hallucination rate below 2% 
        through rigorous validation processes.
        """

    @pytest.fixture
    def mock_graph_evidence(self):
        """Mock graph evidence for testing."""
        return [
            {
                "node_id": "concept_1",
                "node_type": "concept",
                "content": "Strategic Planning Platform",
                "confidence": 0.95,
                "relationships": [
                    {"type": "USES", "target": "concept_2", "strength": 0.8}
                ]
            },
            {
                "node_id": "concept_2", 
                "node_type": "technology",
                "content": "GraphRAG technology",
                "confidence": 0.92,
                "relationships": []
            }
        ]

    @pytest.mark.asyncio
    async def test_validate_content_success(self, detector, sample_content, mock_graph_evidence):
        """Test successful content validation."""
        project_id = "test-project-123"
        
        with patch.object(detector, '_extract_entities', new_callable=AsyncMock) as mock_extract, \
             patch.object(detector, '_query_graph_evidence', new_callable=AsyncMock) as mock_query, \
             patch.object(detector, '_calculate_hallucination_score') as mock_calculate:
            
            # Mock entity extraction
            mock_extract.return_value = ["Strategic Planning Platform", "GraphRAG technology"]
            
            # Mock graph query
            mock_query.return_value = mock_graph_evidence
            
            # Mock score calculation
            mock_calculate.return_value = 0.015  # 1.5% hallucination rate
            
            result = await detector.validate_content(sample_content, project_id)
            
            assert isinstance(result, ValidationResult)
            assert result.content == sample_content
            assert result.hallucination_rate == 0.015
            assert result.validation_score > 0.98
            assert len(result.graph_evidence) == 2
            assert result.is_valid
            
            mock_extract.assert_called_once_with(sample_content)
            mock_query.assert_called_once()
            mock_calculate.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_content_high_hallucination(self, detector, sample_content):
        """Test content validation with high hallucination rate."""
        project_id = "test-project-123"
        
        with patch.object(detector, '_extract_entities', new_callable=AsyncMock) as mock_extract, \
             patch.object(detector, '_query_graph_evidence', new_callable=AsyncMock) as mock_query, \
             patch.object(detector, '_calculate_hallucination_score') as mock_calculate:
            
            mock_extract.return_value = ["Fictional concept", "Made up technology"]
            mock_query.return_value = []  # No graph evidence found
            mock_calculate.return_value = 0.045  # 4.5% hallucination rate
            
            result = await detector.validate_content(sample_content, project_id)
            
            assert result.hallucination_rate == 0.045
            assert not result.is_valid  # Should fail validation
            assert len(result.issues) > 0

    def test_calculate_hallucination_score_no_evidence(self, detector):
        """Test hallucination score calculation with no evidence."""
        entities = ["Entity1", "Entity2", "Entity3"]
        evidence = []
        
        score = detector._calculate_hallucination_score(entities, evidence)
        
        assert score == 1.0  # 100% hallucination when no evidence

    def test_calculate_hallucination_score_partial_evidence(self, detector, mock_graph_evidence):
        """Test hallucination score calculation with partial evidence."""
        entities = ["Strategic Planning Platform", "GraphRAG technology", "Unsupported concept"]
        evidence = mock_graph_evidence
        
        score = detector._calculate_hallucination_score(entities, evidence)
        
        assert 0.0 < score < 1.0  # Partial hallucination
        assert score == pytest.approx(0.33, abs=0.1)  # ~33% unsupported

    def test_calculate_hallucination_score_full_evidence(self, detector, mock_graph_evidence):
        """Test hallucination score calculation with full evidence."""
        entities = ["Strategic Planning Platform", "GraphRAG technology"]
        evidence = mock_graph_evidence
        
        score = detector._calculate_hallucination_score(entities, evidence)
        
        assert score < 0.1  # Very low hallucination rate

    def test_calculate_confidence_score(self, detector, mock_graph_evidence):
        """Test confidence score calculation."""
        score = detector._calculate_confidence_score(mock_graph_evidence)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.9  # High confidence with good evidence


class TestEntityExtractor:
    """Test suite for EntityExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create an EntityExtractor instance for testing."""
        return EntityExtractor()

    @pytest.fixture
    def sample_text(self):
        """Sample text for entity extraction testing."""
        return """
        Apple Inc. is a technology company founded by Steve Jobs in Cupertino, California.
        The iPhone was released in 2007 and revolutionized the smartphone industry.
        Tim Cook became CEO after Steve Jobs in 2011.
        """

    @pytest.mark.asyncio
    async def test_extract_entities_success(self, extractor, sample_text):
        """Test successful entity extraction."""
        entities = await extractor.extract_entities(sample_text)
        
        assert isinstance(entities, list)
        assert len(entities) > 0
        
        # Check for expected entity types
        entity_texts = [e.text for e in entities]
        assert any("Apple" in text for text in entity_texts)
        assert any("Steve Jobs" in text for text in entity_texts)
        assert any("Cupertino" in text for text in entity_texts)

    @pytest.mark.asyncio
    async def test_extract_entities_empty_text(self, extractor):
        """Test entity extraction with empty text."""
        entities = await extractor.extract_entities("")
        
        assert isinstance(entities, list)
        assert len(entities) == 0

    @pytest.mark.asyncio
    async def test_extract_named_entities(self, extractor, sample_text):
        """Test named entity extraction specifically."""
        entities = await extractor.extract_named_entities(sample_text)
        
        assert isinstance(entities, list)
        
        for entity in entities:
            assert hasattr(entity, 'text')
            assert hasattr(entity, 'label')
            assert hasattr(entity, 'start')
            assert hasattr(entity, 'end')
            assert hasattr(entity, 'confidence')

    @pytest.mark.asyncio
    async def test_extract_concepts(self, extractor, sample_text):
        """Test concept extraction."""
        concepts = await extractor.extract_concepts(sample_text)
        
        assert isinstance(concepts, list)
        assert len(concepts) > 0
        
        # Should identify technology-related concepts
        concept_texts = [c.text for c in concepts]
        assert any("technology" in text.lower() for text in concept_texts)
        assert any("smartphone" in text.lower() for text in concept_texts)

    def test_filter_entities_by_confidence(self, extractor):
        """Test entity filtering by confidence threshold."""
        # Mock entities with different confidence scores
        entities = [
            type('Entity', (), {'text': 'High Confidence', 'confidence': 0.95})(),
            type('Entity', (), {'text': 'Medium Confidence', 'confidence': 0.75})(),
            type('Entity', (), {'text': 'Low Confidence', 'confidence': 0.45})()
        ]
        
        filtered = extractor.filter_entities_by_confidence(entities, threshold=0.7)
        
        assert len(filtered) == 2
        assert all(e.confidence >= 0.7 for e in filtered)

    def test_deduplicate_entities(self, extractor):
        """Test entity deduplication."""
        entities = [
            type('Entity', (), {'text': 'Apple Inc.', 'confidence': 0.95})(),
            type('Entity', (), {'text': 'Apple', 'confidence': 0.85})(),
            type('Entity', (), {'text': 'Microsoft', 'confidence': 0.90})(),
            type('Entity', (), {'text': 'Apple Inc.', 'confidence': 0.88})()  # Duplicate
        ]
        
        deduplicated = extractor.deduplicate_entities(entities)
        
        assert len(deduplicated) == 3  # Should remove one duplicate
        entity_texts = [e.text for e in deduplicated]
        assert entity_texts.count('Apple Inc.') == 1


class TestRelationshipExtractor:
    """Test suite for RelationshipExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create a RelationshipExtractor instance for testing."""
        return RelationshipExtractor()

    @pytest.fixture
    def sample_entities(self):
        """Sample entities for relationship extraction."""
        return [
            {"text": "Apple Inc.", "type": "ORGANIZATION", "start": 0, "end": 10},
            {"text": "Steve Jobs", "type": "PERSON", "start": 50, "end": 60},
            {"text": "iPhone", "type": "PRODUCT", "start": 100, "end": 106},
            {"text": "Cupertino", "type": "LOCATION", "start": 150, "end": 159}
        ]

    @pytest.fixture
    def sample_text_with_relationships(self):
        """Sample text with clear relationships."""
        return """
        Steve Jobs founded Apple Inc. in Cupertino, California.
        Apple Inc. developed the iPhone, which was released in 2007.
        Tim Cook succeeded Steve Jobs as CEO of Apple Inc.
        """

    @pytest.mark.asyncio
    async def test_extract_relationships_success(self, extractor, sample_text_with_relationships, sample_entities):
        """Test successful relationship extraction."""
        relationships = await extractor.extract_relationships(
            sample_text_with_relationships, sample_entities
        )
        
        assert isinstance(relationships, list)
        assert len(relationships) > 0
        
        # Check relationship structure
        for rel in relationships:
            assert 'source' in rel
            assert 'target' in rel
            assert 'relation_type' in rel
            assert 'confidence' in rel

    @pytest.mark.asyncio
    async def test_extract_relationships_empty_entities(self, extractor, sample_text_with_relationships):
        """Test relationship extraction with empty entities list."""
        relationships = await extractor.extract_relationships(sample_text_with_relationships, [])
        
        assert isinstance(relationships, list)
        assert len(relationships) == 0

    def test_identify_relationship_patterns(self, extractor):
        """Test relationship pattern identification."""
        text = "John works for Microsoft as a software engineer."
        
        patterns = extractor.identify_relationship_patterns(text)
        
        assert isinstance(patterns, list)
        # Should identify employment relationship pattern
        assert any("works for" in str(pattern).lower() for pattern in patterns)

    def test_calculate_relationship_strength(self, extractor):
        """Test relationship strength calculation."""
        entity1 = {"text": "Apple", "type": "ORGANIZATION"}
        entity2 = {"text": "iPhone", "type": "PRODUCT"}
        context = "Apple developed the iPhone"
        
        strength = extractor.calculate_relationship_strength(entity1, entity2, context)
        
        assert 0.0 <= strength <= 1.0
        assert strength > 0.5  # Should be strong relationship

    def test_filter_relationships_by_confidence(self, extractor):
        """Test filtering relationships by confidence threshold."""
        relationships = [
            {"source": "A", "target": "B", "relation_type": "RELATED", "confidence": 0.95},
            {"source": "C", "target": "D", "relation_type": "RELATED", "confidence": 0.65},
            {"source": "E", "target": "F", "relation_type": "RELATED", "confidence": 0.45}
        ]
        
        filtered = extractor.filter_relationships_by_confidence(relationships, threshold=0.7)
        
        assert len(filtered) == 1
        assert filtered[0]["confidence"] == 0.95


class TestGraphService:
    """Test suite for GraphService."""

    @pytest.fixture
    def graph_service(self):
        """Create a GraphService instance for testing."""
        with patch('services.graphrag.graph_service.GraphDatabase'):
            return GraphService()

    @pytest.fixture
    def sample_node(self):
        """Sample node data for testing."""
        return {
            "id": "node_123",
            "type": "concept",
            "content": "Strategic Planning",
            "properties": {"domain": "business", "confidence": 0.95},
            "created_at": datetime.utcnow()
        }

    @pytest.fixture
    def sample_relationship(self):
        """Sample relationship data for testing."""
        return {
            "source_id": "node_123",
            "target_id": "node_456", 
            "relation_type": "RELATES_TO",
            "properties": {"strength": 0.85, "context": "business planning"},
            "created_at": datetime.utcnow()
        }

    @pytest.mark.asyncio
    async def test_create_node_success(self, graph_service, sample_node):
        """Test successful node creation."""
        with patch.object(graph_service, '_execute_query', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = [{"node_id": sample_node["id"]}]
            
            result = await graph_service.create_node(
                sample_node["type"], 
                sample_node["content"], 
                sample_node["properties"]
            )
            
            assert result is not None
            assert "node_id" in result
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_relationship_success(self, graph_service, sample_relationship):
        """Test successful relationship creation."""
        with patch.object(graph_service, '_execute_query', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = [{"relationship_id": "rel_789"}]
            
            result = await graph_service.create_relationship(
                sample_relationship["source_id"],
                sample_relationship["target_id"],
                sample_relationship["relation_type"],
                sample_relationship["properties"]
            )
            
            assert result is not None
            assert "relationship_id" in result
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_find_nodes_by_content(self, graph_service):
        """Test finding nodes by content."""
        search_content = "Strategic Planning"
        
        with patch.object(graph_service, '_execute_query', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = [
                {"node_id": "node_123", "content": "Strategic Planning", "confidence": 0.95},
                {"node_id": "node_124", "content": "Strategic Planning Framework", "confidence": 0.87}
            ]
            
            results = await graph_service.find_nodes_by_content(search_content)
            
            assert isinstance(results, list)
            assert len(results) == 2
            assert all("node_id" in result for result in results)
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_node_relationships(self, graph_service):
        """Test getting node relationships."""
        node_id = "node_123"
        
        with patch.object(graph_service, '_execute_query', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = [
                {"target_id": "node_456", "relation_type": "RELATES_TO", "strength": 0.85},
                {"target_id": "node_789", "relation_type": "PART_OF", "strength": 0.92}
            ]
            
            relationships = await graph_service.get_node_relationships(node_id)
            
            assert isinstance(relationships, list)
            assert len(relationships) == 2
            assert all("target_id" in rel for rel in relationships)
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_subgraph(self, graph_service):
        """Test querying a subgraph."""
        start_node = "node_123"
        max_depth = 2
        
        with patch.object(graph_service, '_execute_query', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = [
                {"path": [{"id": "node_123"}, {"id": "node_456"}, {"id": "node_789"}]},
                {"path": [{"id": "node_123"}, {"id": "node_456"}]}
            ]
            
            subgraph = await graph_service.query_subgraph(start_node, max_depth)
            
            assert isinstance(subgraph, list)
            assert len(subgraph) == 2
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_calculate_node_importance(self, graph_service):
        """Test node importance calculation."""
        node_id = "node_123"
        
        with patch.object(graph_service, '_execute_query', new_callable=AsyncMock) as mock_execute:
            # Mock centrality calculation results
            mock_execute.return_value = [
                {"node_id": node_id, "degree_centrality": 0.45, "betweenness": 0.23, "pagerank": 0.15}
            ]
            
            importance = await graph_service.calculate_node_importance(node_id)
            
            assert isinstance(importance, dict)
            assert "importance_score" in importance
            assert 0.0 <= importance["importance_score"] <= 1.0
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_node_properties(self, graph_service):
        """Test updating node properties."""
        node_id = "node_123"
        new_properties = {"confidence": 0.98, "updated": True}
        
        with patch.object(graph_service, '_execute_query', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = [{"node_id": node_id, "updated": True}]
            
            result = await graph_service.update_node_properties(node_id, new_properties)
            
            assert result is not None
            assert result.get("updated") is True
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_node(self, graph_service):
        """Test node deletion."""
        node_id = "node_123"
        
        with patch.object(graph_service, '_execute_query', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = [{"deleted_count": 1}]
            
            result = await graph_service.delete_node(node_id)
            
            assert result is True
            mock_execute.assert_called_once()

    def test_build_cypher_query(self, graph_service):
        """Test Cypher query building."""
        query_params = {
            "node_type": "concept",
            "properties": {"domain": "business"}
        }
        
        query = graph_service.build_cypher_query("MATCH", query_params)
        
        assert isinstance(query, str)
        assert "MATCH" in query
        assert "concept" in query
        assert "$domain" in query  # Parameter placeholder


class TestFactChecker:
    """Test suite for FactChecker."""

    @pytest.fixture
    def fact_checker(self):
        """Create a FactChecker instance for testing."""
        with patch('services.graphrag.graph_service.GraphService'):
            return FactChecker()

    @pytest.fixture
    def sample_claims(self):
        """Sample claims for fact checking."""
        return [
            "The Strategic Planning Platform uses GraphRAG technology",
            "Hallucination rates are maintained below 2%",
            "The system supports real-time validation",
            "Users can generate PRDs in multiple formats"
        ]

    @pytest.mark.asyncio
    async def test_check_facts_success(self, fact_checker, sample_claims):
        """Test successful fact checking."""
        project_id = "test-project-123"
        
        with patch.object(fact_checker, '_verify_claim', new_callable=AsyncMock) as mock_verify:
            # Mock fact verification results
            mock_verify.side_effect = [
                {"claim": sample_claims[0], "verified": True, "confidence": 0.95, "evidence": ["source1"]},
                {"claim": sample_claims[1], "verified": True, "confidence": 0.88, "evidence": ["source2"]},
                {"claim": sample_claims[2], "verified": True, "confidence": 0.92, "evidence": ["source3"]},
                {"claim": sample_claims[3], "verified": False, "confidence": 0.45, "evidence": []}
            ]
            
            results = await fact_checker.check_facts(sample_claims, project_id)
            
            assert isinstance(results, list)
            assert len(results) == 4
            assert sum(1 for r in results if r["verified"]) == 3  # 3 verified claims
            assert mock_verify.call_count == 4

    @pytest.mark.asyncio
    async def test_verify_single_claim(self, fact_checker):
        """Test verifying a single claim."""
        claim = "The Strategic Planning Platform uses AI technology"
        project_id = "test-project-123"
        
        with patch.object(fact_checker.graph_service, 'find_nodes_by_content', new_callable=AsyncMock) as mock_find:
            mock_find.return_value = [
                {"content": "Strategic Planning Platform", "confidence": 0.95},
                {"content": "AI technology", "confidence": 0.92}
            ]
            
            result = await fact_checker._verify_claim(claim, project_id)
            
            assert isinstance(result, dict)
            assert "claim" in result
            assert "verified" in result
            assert "confidence" in result
            assert "evidence" in result

    def test_extract_claims_from_text(self, fact_checker):
        """Test claim extraction from text."""
        text = """
        The Strategic Planning Platform is an advanced system. 
        It uses GraphRAG technology for validation.
        The hallucination rate is below 2%.
        Users can export documents in PDF format.
        """
        
        claims = fact_checker.extract_claims_from_text(text)
        
        assert isinstance(claims, list)
        assert len(claims) > 0
        assert all(isinstance(claim, str) for claim in claims)

    def test_calculate_fact_score(self, fact_checker):
        """Test fact score calculation."""
        fact_results = [
            {"verified": True, "confidence": 0.95},
            {"verified": True, "confidence": 0.88},
            {"verified": False, "confidence": 0.45},
            {"verified": True, "confidence": 0.92}
        ]
        
        score = fact_checker.calculate_fact_score(fact_results)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.7  # Should be high with mostly verified facts

    @pytest.mark.asyncio
    async def test_generate_fact_report(self, fact_checker, sample_claims):
        """Test fact checking report generation."""
        project_id = "test-project-123"
        
        with patch.object(fact_checker, 'check_facts', new_callable=AsyncMock) as mock_check:
            mock_check.return_value = [
                {"claim": sample_claims[0], "verified": True, "confidence": 0.95},
                {"claim": sample_claims[1], "verified": False, "confidence": 0.45}
            ]
            
            report = await fact_checker.generate_fact_report(sample_claims, project_id)
            
            assert isinstance(report, dict)
            assert "total_claims" in report
            assert "verified_claims" in report
            assert "fact_score" in report
            assert "detailed_results" in report
            assert report["total_claims"] == 2
            assert report["verified_claims"] == 1