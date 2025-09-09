"""
Relationship extraction for knowledge graph construction.
Extracts semantic relationships between entities using dependency parsing and pattern matching.
"""

import asyncio
from typing import Any, Dict, List, Set, Tuple, Optional
import uuid
from datetime import datetime
import hashlib
import re

import spacy
from spacy.tokens import Doc, Span, Token
from spacy.matcher import Matcher, DependencyMatcher
import networkx as nx
import numpy as np
import structlog

from core.config import get_settings
from .entity_extractor import EntityExtractionPipeline

logger = structlog.get_logger(__name__)
settings = get_settings()


class RelationshipExtractor:
    """
    Extract semantic relationships between entities for knowledge graph construction.
    Optimized for accuracy and graph connectivity with performance <50ms per document.
    """
    
    def __init__(self):
        self.nlp = None  # spaCy model with dependency parsing
        self.entity_extractor = EntityExtractionPipeline()
        self.matcher = None  # Pattern matcher
        self.dep_matcher = None  # Dependency matcher
        self.is_initialized = False
        
        # Relationship type taxonomy
        self.relationship_types = {
            'IS_A': {'weight': 1.0, 'bidirectional': False, 'semantic': 'taxonomy'},
            'PART_OF': {'weight': 0.9, 'bidirectional': False, 'semantic': 'meronymy'},
            'LOCATED_IN': {'weight': 0.8, 'bidirectional': False, 'semantic': 'spatial'},
            'WORKS_FOR': {'weight': 0.8, 'bidirectional': False, 'semantic': 'employment'},
            'FOUNDED_BY': {'weight': 0.9, 'bidirectional': False, 'semantic': 'creation'},
            'ACQUIRED_BY': {'weight': 0.9, 'bidirectional': False, 'semantic': 'ownership'},
            'COLLABORATES_WITH': {'weight': 0.7, 'bidirectional': True, 'semantic': 'cooperation'},
            'COMPETES_WITH': {'weight': 0.7, 'bidirectional': True, 'semantic': 'competition'},
            'USES': {'weight': 0.6, 'bidirectional': False, 'semantic': 'utilization'},
            'CREATES': {'weight': 0.8, 'bidirectional': False, 'semantic': 'creation'},
            'INFLUENCES': {'weight': 0.7, 'bidirectional': False, 'semantic': 'causation'},
            'SIMILAR_TO': {'weight': 0.6, 'bidirectional': True, 'semantic': 'similarity'},
            'RELATED_TO': {'weight': 0.5, 'bidirectional': True, 'semantic': 'general'}
        }
        
        # Performance tracking
        self.extraction_stats = {
            'total_extractions': 0,
            'avg_processing_time_ms': 0,
            'relationship_type_counts': {},
            'avg_relationships_per_doc': 0
        }
    
    async def initialize(self) -> None:
        """Initialize the relationship extraction pipeline."""
        try:
            logger.info("Initializing relationship extraction pipeline...")
            start_time = datetime.now()
            
            # Load spaCy model with dependency parsing
            self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize entity extractor
            await self.entity_extractor.initialize()
            
            # Setup pattern matchers
            self._setup_pattern_matchers()
            self._setup_dependency_patterns()
            
            init_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"Relationship extraction pipeline initialized in {init_time:.2f}ms")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize relationship extraction: {str(e)}")
            raise
    
    def _setup_pattern_matchers(self) -> None:
        """Setup pattern-based relationship matchers."""
        self.matcher = Matcher(self.nlp.vocab)
        
        # IS_A patterns
        is_a_patterns = [
            [{"ENT_TYPE": {"IN": ["PERSON", "ORG", "PRODUCT"]}}, 
             {"LEMMA": "be"}, 
             {"ENT_TYPE": {"IN": ["PERSON", "ORG", "PRODUCT"]}}],
            [{"ENT_TYPE": {"IN": ["PERSON", "ORG"]}}, 
             {"LEMMA": {"IN": ["call", "know"]}}, 
             {"LEMMA": "as"}, 
             {"ENT_TYPE": {"IN": ["PERSON", "ORG"]}}]
        ]
        self.matcher.add("IS_A_PATTERN", is_a_patterns)
        
        # WORKS_FOR patterns
        works_for_patterns = [
            [{"ENT_TYPE": "PERSON"}, 
             {"LEMMA": {"IN": ["work", "employ"]}}, 
             {"LEMMA": {"IN": ["for", "at", "with"]}}, 
             {"ENT_TYPE": "ORG"}],
            [{"ENT_TYPE": "PERSON"}, 
             {"POS": "VERB"}, 
             {"LEMMA": "at"}, 
             {"ENT_TYPE": "ORG"}]
        ]
        self.matcher.add("WORKS_FOR_PATTERN", works_for_patterns)
        
        # LOCATED_IN patterns
        location_patterns = [
            [{"ENT_TYPE": {"IN": ["ORG", "PERSON"]}}, 
             {"LEMMA": {"IN": ["locate", "base", "situate"]}}, 
             {"LEMMA": "in"}, 
             {"ENT_TYPE": "GPE"}],
            [{"ENT_TYPE": {"IN": ["ORG", "PERSON"]}}, 
             {"LEMMA": "in"}, 
             {"ENT_TYPE": "GPE"}]
        ]
        self.matcher.add("LOCATED_IN_PATTERN", location_patterns)
        
        # FOUNDED_BY patterns
        founded_patterns = [
            [{"ENT_TYPE": "ORG"}, 
             {"LEMMA": {"IN": ["found", "establish", "create"]}}, 
             {"LEMMA": "by"}, 
             {"ENT_TYPE": "PERSON"}],
            [{"ENT_TYPE": "PERSON"}, 
             {"LEMMA": {"IN": ["found", "establish", "create"]}}, 
             {"ENT_TYPE": "ORG"}]
        ]
        self.matcher.add("FOUNDED_BY_PATTERN", founded_patterns)
        
        # USES patterns
        uses_patterns = [
            [{"ENT_TYPE": {"IN": ["PERSON", "ORG"]}}, 
             {"LEMMA": {"IN": ["use", "utilize", "employ", "implement"]}}, 
             {"ENT_TYPE": {"IN": ["PRODUCT", "TECHNOLOGY"]}}],
            [{"ENT_TYPE": {"IN": ["PERSON", "ORG"]}}, 
             {"LEMMA": {"IN": ["adopt", "deploy", "integrate"]}}, 
             {"ENT_TYPE": {"IN": ["PRODUCT", "TECHNOLOGY"]}}]
        ]
        self.matcher.add("USES_PATTERN", uses_patterns)
    
    def _setup_dependency_patterns(self) -> None:
        """Setup dependency-based relationship patterns."""
        self.dep_matcher = DependencyMatcher(self.nlp.vocab)
        
        # Subject-Object-Verb patterns for general relationships
        svo_pattern = [
            {
                "RIGHT_ID": "verb",
                "RIGHT_ATTRS": {"POS": "VERB"}
            },
            {
                "LEFT_ID": "verb",
                "REL_OP": ">",
                "RIGHT_ID": "subject",
                "RIGHT_ATTRS": {"DEP": "nsubj", "ENT_TYPE": {"NOT_IN": ["", "CARDINAL", "ORDINAL"]}}
            },
            {
                "LEFT_ID": "verb",
                "REL_OP": ">",
                "RIGHT_ID": "object",
                "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "pobj"]}, "ENT_TYPE": {"NOT_IN": ["", "CARDINAL", "ORDINAL"]}}
            }
        ]
        self.dep_matcher.add("SVO_RELATIONSHIP", [svo_pattern])
        
        # Compound relationships (e.g., "CEO Steve Jobs")
        compound_pattern = [
            {
                "RIGHT_ID": "head",
                "RIGHT_ATTRS": {"ENT_TYPE": {"IN": ["PERSON", "ORG"]}}
            },
            {
                "LEFT_ID": "head", 
                "REL_OP": ">",
                "RIGHT_ID": "modifier",
                "RIGHT_ATTRS": {"DEP": "compound", "ENT_TYPE": {"NOT_IN": [""]}}
            }
        ]
        self.dep_matcher.add("COMPOUND_RELATIONSHIP", [compound_pattern])
    
    async def extract_relationships(
        self,
        text: str,
        entities: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
        min_confidence: float = 0.6
    ) -> Dict[str, Any]:
        """
        Extract relationships between entities in text.
        
        Args:
            text: Input text for relationship extraction
            entities: Pre-extracted entities (if available)
            context: Optional context for domain-specific relationships
            min_confidence: Minimum confidence threshold for relationships
            
        Returns:
            Dictionary with extracted relationships and metadata
        """
        if not self.is_initialized:
            raise RuntimeError("Relationship extraction pipeline not initialized")
        
        start_time = datetime.now()
        extraction_id = str(uuid.uuid4())
        
        try:
            # Extract entities if not provided
            if entities is None:
                entity_result = await self.entity_extractor.extract_entities(
                    text, context, use_transformer=False, min_confidence=min_confidence
                )
                entities = entity_result['entities']
            
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract relationships using multiple methods
            pattern_relationships = self._extract_pattern_relationships(doc, entities)
            dependency_relationships = self._extract_dependency_relationships(doc, entities)
            proximity_relationships = self._extract_proximity_relationships(doc, entities)
            
            # Merge and deduplicate relationships
            all_relationships = (
                pattern_relationships + 
                dependency_relationships + 
                proximity_relationships
            )
            
            merged_relationships = self._merge_and_rank_relationships(
                all_relationships, min_confidence
            )
            
            # Build relationship graph for analysis
            relationship_graph = self._build_relationship_graph(merged_relationships)
            graph_metrics = self._calculate_graph_metrics(relationship_graph)
            
            # Calculate processing time
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update statistics
            self._update_extraction_stats(merged_relationships, processing_time_ms)
            
            result = {
                'extraction_id': extraction_id,
                'relationships': merged_relationships,
                'relationship_count': len(merged_relationships),
                'entity_count': len(entities),
                'graph_metrics': graph_metrics,
                'processing_time_ms': processing_time_ms,
                'methods_used': ['patterns', 'dependencies', 'proximity'],
                'confidence_threshold': min_confidence,
                'timestamp': start_time.isoformat()
            }
            
            logger.info(
                "Relationship extraction completed",
                extraction_id=extraction_id,
                relationship_count=len(merged_relationships),
                entity_count=len(entities),
                processing_time_ms=processing_time_ms
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Relationship extraction failed: {str(e)}", extraction_id=extraction_id)
            raise
    
    def _extract_pattern_relationships(
        self, 
        doc: Doc, 
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract relationships using pattern matching."""
        relationships = []
        matches = self.matcher(doc)
        
        for match_id, start, end in matches:
            pattern_name = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            
            # Find entities in the matched span
            span_entities = self._find_entities_in_span(span, entities)
            
            if len(span_entities) >= 2:
                relationship_type = self._pattern_to_relationship_type(pattern_name)
                
                # Create relationships between entity pairs
                for i, entity1 in enumerate(span_entities[:-1]):
                    for entity2 in span_entities[i+1:]:
                        confidence = self._calculate_pattern_confidence(
                            pattern_name, span, entity1, entity2
                        )
                        
                        if confidence >= 0.5:  # Pattern-specific threshold
                            relationship = {
                                'source_entity': entity1['text'],
                                'source_entity_id': entity1.get('entity_id'),
                                'target_entity': entity2['text'],
                                'target_entity_id': entity2.get('entity_id'),
                                'relationship_type': relationship_type,
                                'confidence': confidence,
                                'evidence_text': span.text,
                                'evidence_start': span.start_char,
                                'evidence_end': span.end_char,
                                'extraction_method': 'pattern',
                                'pattern_name': pattern_name
                            }
                            
                            relationships.append(relationship)
        
        return relationships
    
    def _extract_dependency_relationships(
        self, 
        doc: Doc, 
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract relationships using dependency parsing."""
        relationships = []
        matches = self.dep_matcher(doc)
        
        for match_id, token_ids in matches:
            pattern_name = self.nlp.vocab.strings[match_id]
            
            if pattern_name == "SVO_RELATIONSHIP":
                relationships.extend(
                    self._process_svo_relationship(doc, token_ids, entities)
                )
            elif pattern_name == "COMPOUND_RELATIONSHIP":
                relationships.extend(
                    self._process_compound_relationship(doc, token_ids, entities)
                )
        
        return relationships
    
    def _extract_proximity_relationships(
        self, 
        doc: Doc, 
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract relationships based on entity proximity and co-occurrence."""
        relationships = []
        
        # Group entities by sentence
        sentence_entities = {}
        for sent_idx, sent in enumerate(doc.sents):
            sentence_entities[sent_idx] = []
            
            for entity in entities:
                if sent.start_char <= entity['start_char'] < sent.end_char:
                    sentence_entities[sent_idx].append(entity)
        
        # Create proximity relationships within sentences
        for sent_idx, sent_entities in sentence_entities.items():
            if len(sent_entities) < 2:
                continue
            
            sent = list(doc.sents)[sent_idx]
            
            for i, entity1 in enumerate(sent_entities[:-1]):
                for entity2 in sent_entities[i+1:]:
                    # Calculate proximity score
                    distance = abs(entity1['start_char'] - entity2['start_char'])
                    proximity_score = max(0, 1 - (distance / len(sent.text)))
                    
                    if proximity_score > 0.3:  # Proximity threshold
                        relationship_type = self._infer_relationship_type(
                            entity1, entity2, sent.text
                        )
                        
                        confidence = proximity_score * 0.6  # Lower confidence for proximity
                        
                        relationship = {
                            'source_entity': entity1['text'],
                            'source_entity_id': entity1.get('entity_id'),
                            'target_entity': entity2['text'], 
                            'target_entity_id': entity2.get('entity_id'),
                            'relationship_type': relationship_type,
                            'confidence': confidence,
                            'evidence_text': sent.text,
                            'evidence_start': sent.start_char,
                            'evidence_end': sent.end_char,
                            'extraction_method': 'proximity',
                            'proximity_score': proximity_score
                        }
                        
                        relationships.append(relationship)
        
        return relationships
    
    def _find_entities_in_span(
        self, 
        span: Span, 
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find entities that overlap with the given span."""
        span_entities = []
        
        for entity in entities:
            # Check if entity overlaps with span
            if (entity['start_char'] >= span.start_char and 
                entity['end_char'] <= span.end_char):
                span_entities.append(entity)
        
        return span_entities
    
    def _pattern_to_relationship_type(self, pattern_name: str) -> str:
        """Map pattern names to relationship types."""
        mapping = {
            'IS_A_PATTERN': 'IS_A',
            'WORKS_FOR_PATTERN': 'WORKS_FOR',
            'LOCATED_IN_PATTERN': 'LOCATED_IN',
            'FOUNDED_BY_PATTERN': 'FOUNDED_BY',
            'USES_PATTERN': 'USES'
        }
        return mapping.get(pattern_name, 'RELATED_TO')
    
    def _calculate_pattern_confidence(
        self, 
        pattern_name: str, 
        span: Span, 
        entity1: Dict[str, Any], 
        entity2: Dict[str, Any]
    ) -> float:
        """Calculate confidence for pattern-based relationships."""
        base_confidence = 0.8  # High confidence for patterns
        
        # Boost based on pattern specificity
        pattern_boosts = {
            'FOUNDED_BY_PATTERN': 0.1,
            'WORKS_FOR_PATTERN': 0.05,
            'IS_A_PATTERN': 0.1,
            'LOCATED_IN_PATTERN': 0.05,
            'USES_PATTERN': 0.0
        }
        
        confidence = base_confidence + pattern_boosts.get(pattern_name, 0)
        
        # Entity type compatibility boost
        if self._are_entities_compatible(entity1, entity2, pattern_name):
            confidence += 0.05
        
        # Evidence quality boost
        if len(span.text.split()) >= 3:  # Longer evidence
            confidence += 0.05
        
        return min(confidence, 0.95)
    
    def _process_svo_relationship(
        self, 
        doc: Doc, 
        token_ids: List[int], 
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process subject-verb-object relationships."""
        relationships = []
        
        verb_token = doc[token_ids[0]]
        subject_token = doc[token_ids[1]]
        object_token = doc[token_ids[2]]
        
        # Find corresponding entities
        subject_entity = self._find_entity_for_token(subject_token, entities)
        object_entity = self._find_entity_for_token(object_token, entities)
        
        if subject_entity and object_entity:
            relationship_type = self._verb_to_relationship_type(verb_token.lemma_)
            confidence = self._calculate_dependency_confidence(verb_token, subject_token, object_token)
            
            relationship = {
                'source_entity': subject_entity['text'],
                'source_entity_id': subject_entity.get('entity_id'),
                'target_entity': object_entity['text'],
                'target_entity_id': object_entity.get('entity_id'),
                'relationship_type': relationship_type,
                'confidence': confidence,
                'evidence_text': f"{subject_token.text} {verb_token.text} {object_token.text}",
                'evidence_start': subject_token.idx,
                'evidence_end': object_token.idx + len(object_token.text),
                'extraction_method': 'dependency',
                'verb': verb_token.lemma_
            }
            
            relationships.append(relationship)
        
        return relationships
    
    def _process_compound_relationship(
        self, 
        doc: Doc, 
        token_ids: List[int], 
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process compound relationships (e.g., 'CEO Steve Jobs')."""
        relationships = []
        
        head_token = doc[token_ids[0]]
        modifier_token = doc[token_ids[1]]
        
        # Find corresponding entities
        head_entity = self._find_entity_for_token(head_token, entities)
        modifier_entity = self._find_entity_for_token(modifier_token, entities)
        
        if head_entity and modifier_entity:
            confidence = 0.7  # Medium confidence for compound relationships
            
            relationship = {
                'source_entity': modifier_entity['text'],
                'source_entity_id': modifier_entity.get('entity_id'),
                'target_entity': head_entity['text'],
                'target_entity_id': head_entity.get('entity_id'),
                'relationship_type': 'IS_A',  # Compound usually indicates type relationship
                'confidence': confidence,
                'evidence_text': f"{modifier_token.text} {head_token.text}",
                'evidence_start': modifier_token.idx,
                'evidence_end': head_token.idx + len(head_token.text),
                'extraction_method': 'dependency',
                'dependency_relation': 'compound'
            }
            
            relationships.append(relationship)
        
        return relationships
    
    def _find_entity_for_token(
        self, 
        token: Token, 
        entities: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find entity that contains the given token."""
        for entity in entities:
            if (entity['start_char'] <= token.idx < entity['end_char']):
                return entity
        return None
    
    def _verb_to_relationship_type(self, verb_lemma: str) -> str:
        """Map verbs to relationship types."""
        verb_mapping = {
            'create': 'CREATES',
            'found': 'FOUNDED_BY',
            'establish': 'FOUNDED_BY',
            'use': 'USES',
            'employ': 'USES',
            'work': 'WORKS_FOR',
            'acquire': 'ACQUIRED_BY',
            'buy': 'ACQUIRED_BY',
            'influence': 'INFLUENCES',
            'affect': 'INFLUENCES',
            'compete': 'COMPETES_WITH',
            'collaborate': 'COLLABORATES_WITH'
        }
        return verb_mapping.get(verb_lemma, 'RELATED_TO')
    
    def _calculate_dependency_confidence(
        self, 
        verb: Token, 
        subject: Token, 
        object: Token
    ) -> float:
        """Calculate confidence for dependency-based relationships."""
        base_confidence = 0.6
        
        # Verb-specific boosts
        strong_verbs = ['create', 'found', 'establish', 'acquire', 'buy']
        if verb.lemma_ in strong_verbs:
            base_confidence += 0.2
        
        # Entity type boosts
        if subject.ent_type_ in ['PERSON', 'ORG'] and object.ent_type_ in ['ORG', 'PRODUCT']:
            base_confidence += 0.1
        
        return min(base_confidence, 0.9)
    
    def _infer_relationship_type(
        self, 
        entity1: Dict[str, Any], 
        entity2: Dict[str, Any], 
        context: str
    ) -> str:
        """Infer relationship type based on entity types and context."""
        label1, label2 = entity1['label'], entity2['label']
        
        # Rule-based inference
        if label1 == 'PERSON' and label2 == 'ORG':
            if any(word in context.lower() for word in ['work', 'employ', 'job', 'career']):
                return 'WORKS_FOR'
            elif any(word in context.lower() for word in ['found', 'establish', 'start']):
                return 'FOUNDED_BY'
        
        if label1 == 'ORG' and label2 == 'GPE':
            return 'LOCATED_IN'
        
        if label1 == 'ORG' and label2 == 'PRODUCT':
            return 'CREATES'
        
        if label1 in ['PERSON', 'ORG'] and label2 in ['PRODUCT', 'TECHNOLOGY']:
            return 'USES'
        
        # Default to general relationship
        return 'RELATED_TO'
    
    def _are_entities_compatible(
        self, 
        entity1: Dict[str, Any], 
        entity2: Dict[str, Any], 
        pattern_name: str
    ) -> bool:
        """Check if entity types are compatible with the relationship pattern."""
        label1, label2 = entity1['label'], entity2['label']
        
        compatibility_rules = {
            'WORKS_FOR_PATTERN': (label1 == 'PERSON' and label2 == 'ORG'),
            'FOUNDED_BY_PATTERN': (
                (label1 == 'ORG' and label2 == 'PERSON') or
                (label1 == 'PERSON' and label2 == 'ORG')
            ),
            'LOCATED_IN_PATTERN': (
                label1 in ['PERSON', 'ORG'] and label2 == 'GPE'
            ),
            'USES_PATTERN': (
                label1 in ['PERSON', 'ORG'] and 
                label2 in ['PRODUCT', 'TECHNOLOGY', 'ORG']
            )
        }
        
        return compatibility_rules.get(pattern_name, True)
    
    def _merge_and_rank_relationships(
        self, 
        relationships: List[Dict[str, Any]], 
        min_confidence: float
    ) -> List[Dict[str, Any]]:
        """Merge duplicate relationships and rank by confidence."""
        # Create relationship signatures for deduplication
        relationship_map = {}
        
        for rel in relationships:
            # Create signature based on entities and relationship type
            signature = (
                rel['source_entity'].lower(),
                rel['target_entity'].lower(),
                rel['relationship_type']
            )
            
            if signature in relationship_map:
                # Keep relationship with higher confidence
                if rel['confidence'] > relationship_map[signature]['confidence']:
                    # Merge evidence from both
                    existing = relationship_map[signature]
                    rel['evidence_sources'] = [
                        existing.get('evidence_text', ''),
                        rel['evidence_text']
                    ]
                    relationship_map[signature] = rel
                else:
                    # Add evidence to existing relationship
                    existing = relationship_map[signature]
                    if 'evidence_sources' not in existing:
                        existing['evidence_sources'] = [existing['evidence_text']]
                    existing['evidence_sources'].append(rel['evidence_text'])
            else:
                relationship_map[signature] = rel
        
        # Filter by confidence and add relationship metadata
        filtered_relationships = []
        for rel in relationship_map.values():
            if rel['confidence'] >= min_confidence:
                # Add relationship metadata
                rel['relationship_id'] = self._generate_relationship_id(
                    rel['source_entity'], rel['target_entity'], rel['relationship_type']
                )
                rel['weight'] = self.relationship_types.get(
                    rel['relationship_type'], {}
                ).get('weight', 0.5)
                rel['is_bidirectional'] = self.relationship_types.get(
                    rel['relationship_type'], {}
                ).get('bidirectional', False)
                
                filtered_relationships.append(rel)
        
        # Sort by confidence and importance
        filtered_relationships.sort(
            key=lambda x: (-x['confidence'], -x['weight']), 
            reverse=False
        )
        
        return filtered_relationships
    
    def _build_relationship_graph(
        self, 
        relationships: List[Dict[str, Any]]
    ) -> nx.DiGraph:
        """Build NetworkX graph from relationships for analysis."""
        G = nx.DiGraph()
        
        # Add nodes (entities)
        entities = set()
        for rel in relationships:
            entities.add(rel['source_entity'])
            entities.add(rel['target_entity'])
        
        for entity in entities:
            G.add_node(entity)
        
        # Add edges (relationships)
        for rel in relationships:
            G.add_edge(
                rel['source_entity'],
                rel['target_entity'],
                relationship_type=rel['relationship_type'],
                confidence=rel['confidence'],
                weight=rel['weight']
            )
            
            # Add reverse edge if bidirectional
            if rel['is_bidirectional']:
                G.add_edge(
                    rel['target_entity'],
                    rel['source_entity'],
                    relationship_type=rel['relationship_type'],
                    confidence=rel['confidence'],
                    weight=rel['weight']
                )
        
        return G
    
    def _calculate_graph_metrics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Calculate graph connectivity and structure metrics."""
        if len(graph.nodes()) == 0:
            return {
                'node_count': 0,
                'edge_count': 0,
                'density': 0,
                'avg_clustering': 0,
                'connected_components': 0
            }
        
        try:
            # Convert to undirected for some metrics
            undirected = graph.to_undirected()
            
            metrics = {
                'node_count': len(graph.nodes()),
                'edge_count': len(graph.edges()),
                'density': nx.density(graph),
                'avg_clustering': nx.average_clustering(undirected),
                'connected_components': nx.number_connected_components(undirected),
                'avg_degree': sum(dict(graph.degree()).values()) / len(graph.nodes()) if graph.nodes() else 0
            }
            
            # Add centrality measures for key nodes
            if len(graph.nodes()) > 1:
                centrality = nx.degree_centrality(graph)
                top_central_nodes = sorted(
                    centrality.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3]
                metrics['top_central_entities'] = top_central_nodes
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Graph metrics calculation failed: {str(e)}")
            return {
                'node_count': len(graph.nodes()),
                'edge_count': len(graph.edges()),
                'error': str(e)
            }
    
    def _generate_relationship_id(
        self, 
        source: str, 
        target: str, 
        rel_type: str
    ) -> str:
        """Generate unique relationship ID."""
        content = f"{source.lower()}:{target.lower()}:{rel_type}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _update_extraction_stats(
        self, 
        relationships: List[Dict[str, Any]], 
        processing_time_ms: float
    ) -> None:
        """Update relationship extraction statistics."""
        self.extraction_stats['total_extractions'] += 1
        
        # Update average processing time
        total_time = (
            self.extraction_stats['avg_processing_time_ms'] * 
            (self.extraction_stats['total_extractions'] - 1) + 
            processing_time_ms
        )
        self.extraction_stats['avg_processing_time_ms'] = (
            total_time / self.extraction_stats['total_extractions']
        )
        
        # Update relationship type counts
        for rel in relationships:
            rel_type = rel['relationship_type']
            self.extraction_stats['relationship_type_counts'][rel_type] = (
                self.extraction_stats['relationship_type_counts'].get(rel_type, 0) + 1
            )
        
        # Update average relationships per document
        total_rels = (
            self.extraction_stats['avg_relationships_per_doc'] * 
            (self.extraction_stats['total_extractions'] - 1) + 
            len(relationships)
        )
        self.extraction_stats['avg_relationships_per_doc'] = (
            total_rels / self.extraction_stats['total_extractions']
        )
    
    async def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get current relationship extraction statistics."""
        return {
            'total_extractions': self.extraction_stats['total_extractions'],
            'avg_processing_time_ms': round(self.extraction_stats['avg_processing_time_ms'], 2),
            'avg_relationships_per_doc': round(self.extraction_stats['avg_relationships_per_doc'], 2),
            'relationship_type_distribution': self.extraction_stats['relationship_type_counts'],
            'performance_target_met': self.extraction_stats['avg_processing_time_ms'] < 50,
            'supported_relationship_types': list(self.relationship_types.keys()),
            'extraction_methods': ['patterns', 'dependencies', 'proximity']
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check relationship extraction pipeline health."""
        try:
            if not self.is_initialized:
                return {
                    'status': 'unhealthy',
                    'error': 'Pipeline not initialized'
                }
            
            # Test extraction with simple text
            test_text = "Steve Jobs founded Apple Inc. in Cupertino, California. The company creates innovative products."
            start_time = datetime.now()
            
            test_result = await self.extract_relationships(
                test_text,
                min_confidence=0.5
            )
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'status': 'healthy' if response_time < 100 else 'degraded',
                'response_time_ms': round(response_time, 2),
                'test_relationships_found': test_result['relationship_count'],
                'performance_target_met': response_time < 50,
                'components_loaded': {
                    'spacy': self.nlp is not None,
                    'entity_extractor': self.entity_extractor.is_initialized,
                    'pattern_matcher': self.matcher is not None,
                    'dependency_matcher': self.dep_matcher is not None
                }
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }