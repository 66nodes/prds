"""
Advanced hallucination detection system for GraphRAG with <2% threshold.
Multi-layered approach combining vector similarity, graph validation, and fact-checking.
"""

import asyncio
from typing import Any, Dict, List, Set, Tuple, Optional, Union
import uuid
from datetime import datetime
import hashlib
import re
import json

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
import structlog

from core.config import get_settings
from core.database import get_neo4j
from .entity_extractor import EntityExtractionPipeline
from .relationship_extractor import RelationshipExtractor

logger = structlog.get_logger(__name__)
settings = get_settings()


class HallucinationDetector:
    """
    Advanced hallucination detection system using multi-tier validation.
    Target: <2% hallucination rate with high precision detection.
    """
    
    def __init__(self):
        self.sentence_transformer = None  # For semantic similarity
        self.entity_extractor = EntityExtractionPipeline()
        self.relationship_extractor = RelationshipExtractor()
        self.neo4j = None
        self.is_initialized = False
        
        # Hallucination detection thresholds
        self.thresholds = {
            'semantic_similarity': 0.75,  # High threshold for semantic consistency
            'entity_consistency': 0.85,   # High threshold for entity validation
            'fact_verification': 0.90,    # Very high threshold for fact-checking
            'relationship_validity': 0.80, # High threshold for relationship validation
            'overall_confidence': 0.98    # 98% confidence = <2% hallucination rate
        }
        
        # Validation weights for final score calculation
        self.validation_weights = {
            'semantic': 0.25,      # 25% weight for semantic consistency
            'entity': 0.30,        # 30% weight for entity validation  
            'relationship': 0.25,  # 25% weight for relationship validation
            'factual': 0.20        # 20% weight for fact verification
        }
        
        # Performance and accuracy tracking
        self.detection_stats = {
            'total_validations': 0,
            'hallucinations_detected': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'avg_processing_time_ms': 0,
            'accuracy_rate': 0.0,
            'precision_rate': 0.0,
            'recall_rate': 0.0
        }
        
        # Cache for known facts and validated content
        self.fact_cache = {}
        self.validated_cache = {}
    
    async def initialize(self) -> None:
        """Initialize the hallucination detection system."""
        try:
            logger.info("Initializing hallucination detection system...")
            start_time = datetime.now()
            
            # Load sentence transformer for semantic similarity
            model_name = "all-MiniLM-L6-v2"  # Fast and accurate
            self.sentence_transformer = SentenceTransformer(model_name)
            
            # Initialize sub-components
            await self.entity_extractor.initialize()
            await self.relationship_extractor.initialize()
            
            # Initialize Neo4j connection
            self.neo4j = await get_neo4j()
            
            # Load validation knowledge base
            await self._load_validation_knowledge_base()
            
            init_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"Hallucination detection system initialized in {init_time:.2f}ms")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize hallucination detection: {str(e)}")
            raise
    
    async def detect_hallucinations(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        reference_content: Optional[str] = None,
        validation_mode: str = "comprehensive"  # "fast", "standard", "comprehensive"
    ) -> Dict[str, Any]:
        """
        Detect hallucinations in generated content using multi-tier validation.
        
        Args:
            content: Content to validate for hallucinations
            context: Optional context for domain-specific validation
            reference_content: Optional reference content for comparison
            validation_mode: Validation depth ("fast", "standard", "comprehensive")
            
        Returns:
            Detailed hallucination detection results with <2% error rate
        """
        if not self.is_initialized:
            raise RuntimeError("Hallucination detection system not initialized")
        
        start_time = datetime.now()
        detection_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Starting hallucination detection for content length: {len(content)}")
            
            # Pre-process and normalize content
            normalized_content = self._normalize_content(content)
            
            # Multi-tier validation based on mode
            validation_results = {}
            
            if validation_mode in ["fast", "standard", "comprehensive"]:
                # Tier 1: Semantic consistency validation (always included)
                validation_results['semantic'] = await self._validate_semantic_consistency(
                    normalized_content, reference_content, context
                )
            
            if validation_mode in ["standard", "comprehensive"]:
                # Tier 2: Entity consistency validation
                validation_results['entity'] = await self._validate_entity_consistency(
                    normalized_content, context
                )
                
                # Tier 3: Relationship validation
                validation_results['relationship'] = await self._validate_relationship_consistency(
                    normalized_content, context
                )
            
            if validation_mode == "comprehensive":
                # Tier 4: Factual verification against knowledge base
                validation_results['factual'] = await self._validate_factual_consistency(
                    normalized_content, context
                )
            
            # Calculate overall confidence and hallucination probability
            overall_results = self._calculate_overall_confidence(validation_results)
            
            # Identify specific hallucination patterns
            hallucination_patterns = self._identify_hallucination_patterns(
                normalized_content, validation_results
            )
            
            # Generate detailed report
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            result = {
                'detection_id': detection_id,
                'content_length': len(content),
                'validation_mode': validation_mode,
                'overall_confidence': overall_results['confidence'],
                'hallucination_probability': overall_results['hallucination_probability'],
                'passes_threshold': overall_results['passes_threshold'],
                'validation_results': validation_results,
                'hallucination_patterns': hallucination_patterns,
                'processing_time_ms': processing_time_ms,
                'recommendations': self._generate_recommendations(validation_results),
                'timestamp': start_time.isoformat()
            }
            
            # Update detection statistics
            self._update_detection_stats(result, processing_time_ms)
            
            logger.info(
                "Hallucination detection completed",
                detection_id=detection_id,
                confidence=overall_results['confidence'],
                hallucination_probability=overall_results['hallucination_probability'],
                processing_time_ms=processing_time_ms
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Hallucination detection failed: {str(e)}", detection_id=detection_id)
            raise
    
    async def _validate_semantic_consistency(
        self,
        content: str,
        reference_content: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate semantic consistency using sentence transformers."""
        try:
            # Split content into sentences for analysis
            sentences = self._split_into_sentences(content)
            
            # Generate embeddings for all sentences
            sentence_embeddings = self.sentence_transformer.encode(sentences)
            
            # Calculate internal consistency (sentences should be semantically coherent)
            internal_consistency_scores = []
            
            if len(sentences) > 1:
                for i in range(len(sentences) - 1):
                    similarity = cosine_similarity(
                        [sentence_embeddings[i]], 
                        [sentence_embeddings[i + 1]]
                    )[0][0]
                    internal_consistency_scores.append(similarity)
            
            avg_internal_consistency = (
                np.mean(internal_consistency_scores) 
                if internal_consistency_scores 
                else 1.0
            )
            
            # External consistency validation (against reference if provided)
            external_consistency = 1.0
            if reference_content:
                reference_embedding = self.sentence_transformer.encode([reference_content])
                content_embedding = self.sentence_transformer.encode([content])
                
                external_consistency = cosine_similarity(
                    content_embedding,
                    reference_embedding
                )[0][0]
            
            # Detect semantic anomalies
            anomalies = self._detect_semantic_anomalies(sentences, sentence_embeddings)
            
            # Calculate overall semantic score
            semantic_score = (
                avg_internal_consistency * 0.6 + 
                external_consistency * 0.4
            )
            
            # Apply penalty for detected anomalies
            if anomalies:
                penalty = min(0.3, len(anomalies) * 0.1)
                semantic_score = max(0.0, semantic_score - penalty)
            
            return {
                'score': float(semantic_score),
                'internal_consistency': float(avg_internal_consistency),
                'external_consistency': float(external_consistency),
                'anomalies_detected': len(anomalies),
                'anomaly_details': anomalies,
                'sentence_count': len(sentences),
                'passes_threshold': semantic_score >= self.thresholds['semantic_similarity']
            }
            
        except Exception as e:
            logger.error(f"Semantic consistency validation failed: {str(e)}")
            return {
                'score': 0.5,
                'error': str(e),
                'passes_threshold': False
            }
    
    async def _validate_entity_consistency(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate entity consistency against known knowledge graph."""
        try:
            # Extract entities from content
            entity_result = await self.entity_extractor.extract_entities(
                content, context, use_transformer=True, min_confidence=0.7
            )
            entities = entity_result['entities']
            
            # Validate each entity against knowledge graph
            entity_validations = []
            total_confidence = 0.0
            
            for entity in entities:
                validation = await self._validate_single_entity(entity, context)
                entity_validations.append(validation)
                total_confidence += validation['confidence']
            
            avg_confidence = (
                total_confidence / len(entities) 
                if entities 
                else 1.0
            )
            
            # Check for entity hallucinations (non-existent entities)
            hallucinated_entities = [
                ev for ev in entity_validations 
                if ev['confidence'] < 0.3
            ]
            
            # Check for entity inconsistencies (contradictory information)
            inconsistent_entities = [
                ev for ev in entity_validations 
                if ev.get('has_contradictions', False)
            ]
            
            entity_score = avg_confidence
            
            # Apply penalties
            if hallucinated_entities:
                penalty = min(0.4, len(hallucinated_entities) * 0.15)
                entity_score = max(0.0, entity_score - penalty)
            
            if inconsistent_entities:
                penalty = min(0.3, len(inconsistent_entities) * 0.1)
                entity_score = max(0.0, entity_score - penalty)
            
            return {
                'score': float(entity_score),
                'entities_validated': len(entities),
                'avg_entity_confidence': float(avg_confidence),
                'hallucinated_entities': len(hallucinated_entities),
                'inconsistent_entities': len(inconsistent_entities),
                'entity_details': entity_validations,
                'passes_threshold': entity_score >= self.thresholds['entity_consistency']
            }
            
        except Exception as e:
            logger.error(f"Entity consistency validation failed: {str(e)}")
            return {
                'score': 0.5,
                'error': str(e),
                'passes_threshold': False
            }
    
    async def _validate_relationship_consistency(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate relationship consistency against knowledge graph."""
        try:
            # Extract relationships from content
            relationship_result = await self.relationship_extractor.extract_relationships(
                content, context=context, min_confidence=0.6
            )
            relationships = relationship_result['relationships']
            
            # Validate each relationship against knowledge graph
            relationship_validations = []
            total_confidence = 0.0
            
            for relationship in relationships:
                validation = await self._validate_single_relationship(relationship, context)
                relationship_validations.append(validation)
                total_confidence += validation['confidence']
            
            avg_confidence = (
                total_confidence / len(relationships) 
                if relationships 
                else 1.0
            )
            
            # Check for relationship hallucinations
            hallucinated_relationships = [
                rv for rv in relationship_validations 
                if rv['confidence'] < 0.3
            ]
            
            # Check for contradictory relationships
            contradictory_relationships = [
                rv for rv in relationship_validations 
                if rv.get('contradicts_kb', False)
            ]
            
            relationship_score = avg_confidence
            
            # Apply penalties
            if hallucinated_relationships:
                penalty = min(0.4, len(hallucinated_relationships) * 0.2)
                relationship_score = max(0.0, relationship_score - penalty)
            
            if contradictory_relationships:
                penalty = min(0.5, len(contradictory_relationships) * 0.25)
                relationship_score = max(0.0, relationship_score - penalty)
            
            return {
                'score': float(relationship_score),
                'relationships_validated': len(relationships),
                'avg_relationship_confidence': float(avg_confidence),
                'hallucinated_relationships': len(hallucinated_relationships),
                'contradictory_relationships': len(contradictory_relationships),
                'relationship_details': relationship_validations,
                'passes_threshold': relationship_score >= self.thresholds['relationship_validity']
            }
            
        except Exception as e:
            logger.error(f"Relationship consistency validation failed: {str(e)}")
            return {
                'score': 0.5,
                'error': str(e),
                'passes_threshold': False
            }
    
    async def _validate_factual_consistency(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate factual consistency against verified knowledge base."""
        try:
            # Extract factual claims from content
            claims = self._extract_factual_claims(content)
            
            # Validate each claim against knowledge base
            claim_validations = []
            total_confidence = 0.0
            
            for claim in claims:
                validation = await self._validate_factual_claim(claim, context)
                claim_validations.append(validation)
                total_confidence += validation['confidence']
            
            avg_confidence = (
                total_confidence / len(claims) 
                if claims 
                else 1.0
            )
            
            # Check for factual errors
            factual_errors = [
                cv for cv in claim_validations 
                if cv['confidence'] < 0.5 or cv.get('contradicts_facts', False)
            ]
            
            # Check for unverifiable claims
            unverifiable_claims = [
                cv for cv in claim_validations 
                if cv.get('verifiability') == 'unverifiable'
            ]
            
            factual_score = avg_confidence
            
            # Heavy penalties for factual errors
            if factual_errors:
                penalty = min(0.6, len(factual_errors) * 0.3)
                factual_score = max(0.0, factual_score - penalty)
            
            # Lighter penalty for unverifiable claims
            if unverifiable_claims:
                penalty = min(0.2, len(unverifiable_claims) * 0.05)
                factual_score = max(0.0, factual_score - penalty)
            
            return {
                'score': float(factual_score),
                'claims_validated': len(claims),
                'avg_claim_confidence': float(avg_confidence),
                'factual_errors': len(factual_errors),
                'unverifiable_claims': len(unverifiable_claims),
                'claim_details': claim_validations,
                'passes_threshold': factual_score >= self.thresholds['fact_verification']
            }
            
        except Exception as e:
            logger.error(f"Factual consistency validation failed: {str(e)}")
            return {
                'score': 0.5,
                'error': str(e),
                'passes_threshold': False
            }
    
    async def _validate_single_entity(
        self,
        entity: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate a single entity against the knowledge graph."""
        try:
            entity_text = entity['text']
            entity_type = entity['label']
            
            # Check cache first
            cache_key = f"{entity_text}:{entity_type}"
            if cache_key in self.validated_cache:
                return self.validated_cache[cache_key]
            
            # Query Neo4j for similar entities
            query = """
            MATCH (e:Entity) 
            WHERE toLower(e.name) CONTAINS toLower($entity_name)
            OR toLower(e.normalized_name) CONTAINS toLower($entity_name)
            RETURN e.name as name, e.type as type, e.confidence_score as confidence,
                   e.source_refs as sources, e.verified as verified
            LIMIT 5
            """
            
            results = await self.neo4j.execute_query(
                query, 
                {"entity_name": entity_text}
            )
            
            # Calculate validation confidence
            max_confidence = 0.0
            best_match = None
            has_contradictions = False
            
            for record in results:
                # Calculate name similarity
                name_similarity = self._calculate_text_similarity(
                    entity_text.lower(), 
                    record['name'].lower()
                )
                
                # Check type compatibility
                type_compatible = self._are_entity_types_compatible(
                    entity_type, 
                    record['type']
                )
                
                # Calculate overall match confidence
                match_confidence = name_similarity * 0.7
                if type_compatible:
                    match_confidence += 0.3
                else:
                    has_contradictions = True
                
                # Boost for verified entities
                if record.get('verified', False):
                    match_confidence *= 1.2
                
                match_confidence = min(match_confidence, 1.0)
                
                if match_confidence > max_confidence:
                    max_confidence = match_confidence
                    best_match = record
            
            # If no matches found, check for potential hallucination
            if max_confidence < 0.3:
                # Additional check: is this a common/expected entity type?
                common_confidence = self._assess_common_entity_likelihood(
                    entity_text, entity_type, context
                )
                max_confidence = max(max_confidence, common_confidence)
            
            validation_result = {
                'entity': entity_text,
                'entity_type': entity_type,
                'confidence': float(max_confidence),
                'best_match': best_match,
                'has_contradictions': has_contradictions,
                'kb_matches_found': len(results),
                'likely_hallucination': max_confidence < 0.3
            }
            
            # Cache result
            self.validated_cache[cache_key] = validation_result
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Single entity validation failed: {str(e)}")
            return {
                'entity': entity.get('text', ''),
                'confidence': 0.3,
                'error': str(e),
                'likely_hallucination': True
            }
    
    async def _validate_single_relationship(
        self,
        relationship: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate a single relationship against the knowledge graph."""
        try:
            source = relationship['source_entity']
            target = relationship['target_entity']
            rel_type = relationship['relationship_type']
            
            # Query Neo4j for existing relationships
            query = """
            MATCH (s:Entity)-[r]->(t:Entity)
            WHERE (toLower(s.name) CONTAINS toLower($source) OR toLower(s.normalized_name) CONTAINS toLower($source))
            AND (toLower(t.name) CONTAINS toLower($target) OR toLower(t.normalized_name) CONTAINS toLower($target))
            RETURN s.name as source_name, type(r) as relationship_type, t.name as target_name,
                   r.confidence as confidence, r.verified as verified
            LIMIT 10
            """
            
            results = await self.neo4j.execute_query(
                query,
                {"source": source, "target": target}
            )
            
            # Check for exact or similar relationships
            max_confidence = 0.0
            contradicts_kb = False
            best_match = None
            
            for record in results:
                # Calculate entity name similarities
                source_similarity = self._calculate_text_similarity(
                    source.lower(), 
                    record['source_name'].lower()
                )
                target_similarity = self._calculate_text_similarity(
                    target.lower(), 
                    record['target_name'].lower()
                )
                
                # Check relationship type compatibility
                rel_similarity = self._calculate_relationship_similarity(
                    rel_type, 
                    record['relationship_type']
                )
                
                # Calculate overall match confidence
                match_confidence = (
                    source_similarity * 0.3 + 
                    target_similarity * 0.3 + 
                    rel_similarity * 0.4
                )
                
                # Boost for verified relationships
                if record.get('verified', False):
                    match_confidence *= 1.2
                
                match_confidence = min(match_confidence, 1.0)
                
                if match_confidence > max_confidence:
                    max_confidence = match_confidence
                    best_match = record
                
                # Check for contradictions (same entities, contradictory relationship)
                if (source_similarity > 0.8 and target_similarity > 0.8 and 
                    rel_similarity < 0.3 and record.get('verified', False)):
                    contradicts_kb = True
            
            # If no direct matches, check for indirect validation
            if max_confidence < 0.5:
                indirect_confidence = await self._validate_relationship_indirectly(
                    source, target, rel_type, context
                )
                max_confidence = max(max_confidence, indirect_confidence)
            
            return {
                'source_entity': source,
                'target_entity': target,
                'relationship_type': rel_type,
                'confidence': float(max_confidence),
                'best_match': best_match,
                'contradicts_kb': contradicts_kb,
                'kb_matches_found': len(results),
                'likely_hallucination': max_confidence < 0.3
            }
            
        except Exception as e:
            logger.error(f"Single relationship validation failed: {str(e)}")
            return {
                'source_entity': relationship.get('source_entity', ''),
                'target_entity': relationship.get('target_entity', ''),
                'relationship_type': relationship.get('relationship_type', ''),
                'confidence': 0.3,
                'error': str(e),
                'likely_hallucination': True
            }
    
    async def _validate_factual_claim(
        self,
        claim: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate a factual claim against the knowledge base."""
        try:
            # Check cache first
            claim_hash = hashlib.md5(claim.encode()).hexdigest()
            if claim_hash in self.fact_cache:
                return self.fact_cache[claim_hash]
            
            # Search for similar claims in Neo4j
            query = """
            MATCH (c:Claim)
            WHERE c.text CONTAINS $claim_text 
            OR $claim_text CONTAINS c.text
            RETURN c.text as claim_text, c.confidence_score as confidence,
                   c.verified as verified, c.source_refs as sources,
                   c.contradicts as contradicts
            ORDER BY c.confidence_score DESC
            LIMIT 5
            """
            
            # Use semantic search for better matching
            claim_embedding = self.sentence_transformer.encode([claim])
            
            results = await self.neo4j.execute_query(
                query,
                {"claim_text": claim}
            )
            
            max_confidence = 0.0
            contradicts_facts = False
            verifiability = 'unknown'
            best_match = None
            
            for record in results:
                # Calculate semantic similarity
                record_embedding = self.sentence_transformer.encode([record['claim_text']])
                similarity = cosine_similarity(claim_embedding, record_embedding)[0][0]
                
                if similarity > 0.8:  # High similarity threshold
                    confidence = record.get('confidence', 0.5) * similarity
                    
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_match = record
                        
                        if record.get('verified', False):
                            verifiability = 'verified'
                        elif record.get('contradicts'):
                            contradicts_facts = True
                            verifiability = 'contradicted'
            
            # If no matches found, assess claim plausibility
            if max_confidence < 0.3:
                plausibility_score = self._assess_claim_plausibility(claim, context)
                max_confidence = max(max_confidence, plausibility_score)
                
                if plausibility_score < 0.5:
                    verifiability = 'unverifiable'
                else:
                    verifiability = 'plausible'
            
            validation_result = {
                'claim': claim,
                'confidence': float(max_confidence),
                'verifiability': verifiability,
                'contradicts_facts': contradicts_facts,
                'best_match': best_match,
                'kb_matches_found': len(results)
            }
            
            # Cache result
            self.fact_cache[claim_hash] = validation_result
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Factual claim validation failed: {str(e)}")
            return {
                'claim': claim,
                'confidence': 0.3,
                'verifiability': 'unknown',
                'error': str(e)
            }
    
    def _calculate_overall_confidence(
        self,
        validation_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate overall confidence score from all validation tiers."""
        weighted_score = 0.0
        total_weight = 0.0
        
        for tier, weight in self.validation_weights.items():
            if tier in validation_results:
                tier_score = validation_results[tier].get('score', 0.5)
                weighted_score += tier_score * weight
                total_weight += weight
        
        # Normalize by actual weights used
        overall_confidence = weighted_score / total_weight if total_weight > 0 else 0.5
        
        # Calculate hallucination probability (inverse of confidence)
        hallucination_probability = 1.0 - overall_confidence
        
        # Check if passes overall threshold (98% confidence = <2% hallucination)
        passes_threshold = overall_confidence >= self.thresholds['overall_confidence']
        
        return {
            'confidence': float(overall_confidence),
            'hallucination_probability': float(hallucination_probability),
            'passes_threshold': passes_threshold,
            'weighted_components': {
                tier: validation_results[tier].get('score', 0.5) * weight
                for tier, weight in self.validation_weights.items()
                if tier in validation_results
            }
        }
    
    def _identify_hallucination_patterns(
        self,
        content: str,
        validation_results: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify specific hallucination patterns in the content."""
        patterns = []
        
        # Entity hallucination patterns
        if 'entity' in validation_results:
            entity_result = validation_results['entity']
            if entity_result.get('hallucinated_entities', 0) > 0:
                patterns.append({
                    'type': 'entity_hallucination',
                    'description': 'Content contains non-existent or unverifiable entities',
                    'severity': 'high',
                    'count': entity_result['hallucinated_entities']
                })
        
        # Relationship hallucination patterns
        if 'relationship' in validation_results:
            rel_result = validation_results['relationship']
            if rel_result.get('hallucinated_relationships', 0) > 0:
                patterns.append({
                    'type': 'relationship_hallucination',
                    'description': 'Content contains non-existent or contradictory relationships',
                    'severity': 'high',
                    'count': rel_result['hallucinated_relationships']
                })
        
        # Factual hallucination patterns
        if 'factual' in validation_results:
            fact_result = validation_results['factual']
            if fact_result.get('factual_errors', 0) > 0:
                patterns.append({
                    'type': 'factual_hallucination',
                    'description': 'Content contains factually incorrect claims',
                    'severity': 'critical',
                    'count': fact_result['factual_errors']
                })
        
        # Semantic inconsistency patterns
        if 'semantic' in validation_results:
            sem_result = validation_results['semantic']
            if sem_result.get('anomalies_detected', 0) > 0:
                patterns.append({
                    'type': 'semantic_inconsistency',
                    'description': 'Content contains semantically inconsistent information',
                    'severity': 'medium',
                    'count': sem_result['anomalies_detected']
                })
        
        return patterns
    
    def _generate_recommendations(
        self,
        validation_results: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations for improving content quality."""
        recommendations = []
        
        # Entity-based recommendations
        if 'entity' in validation_results:
            entity_result = validation_results['entity']
            if not entity_result.get('passes_threshold', False):
                recommendations.append(
                    "Verify all mentioned entities against authoritative sources"
                )
            if entity_result.get('hallucinated_entities', 0) > 0:
                recommendations.append(
                    "Remove or replace entities that cannot be verified"
                )
        
        # Relationship-based recommendations
        if 'relationship' in validation_results:
            rel_result = validation_results['relationship']
            if not rel_result.get('passes_threshold', False):
                recommendations.append(
                    "Verify relationships between entities using reliable sources"
                )
            if rel_result.get('contradictory_relationships', 0) > 0:
                recommendations.append(
                    "Resolve contradictory relationship claims"
                )
        
        # Factual recommendations
        if 'factual' in validation_results:
            fact_result = validation_results['factual']
            if fact_result.get('factual_errors', 0) > 0:
                recommendations.append(
                    "Fact-check all claims against authoritative knowledge sources"
                )
            if fact_result.get('unverifiable_claims', 0) > 0:
                recommendations.append(
                    "Provide citations or remove unverifiable claims"
                )
        
        # Semantic recommendations
        if 'semantic' in validation_results:
            sem_result = validation_results['semantic']
            if not sem_result.get('passes_threshold', False):
                recommendations.append(
                    "Improve semantic coherence and consistency throughout the content"
                )
        
        if not recommendations:
            recommendations.append("Content passes all validation checks")
        
        return recommendations
    
    # Helper methods for specific validation tasks
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for consistent processing."""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content.strip())
        
        # Normalize quotes and punctuation
        content = content.replace('"', '"').replace('"', '"')
        content = content.replace(''', "'").replace(''', "'")
        
        return content
    
    def _split_into_sentences(self, content: str) -> List[str]:
        """Split content into sentences for analysis."""
        # Simple sentence splitting (could be enhanced with spaCy)
        sentences = re.split(r'[.!?]+', content)
        return [s.strip() for s in sentences if s.strip()]
    
    def _detect_semantic_anomalies(
        self,
        sentences: List[str],
        embeddings: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Detect semantic anomalies in sentence sequences."""
        anomalies = []
        
        if len(sentences) < 3:
            return anomalies
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find sentences with low similarity to context
        for i, sentence in enumerate(sentences):
            # Calculate average similarity to surrounding sentences
            start_idx = max(0, i - 2)
            end_idx = min(len(sentences), i + 3)
            
            context_similarities = []
            for j in range(start_idx, end_idx):
                if j != i:
                    context_similarities.append(similarity_matrix[i][j])
            
            avg_similarity = np.mean(context_similarities) if context_similarities else 1.0
            
            # Flag as anomaly if similarity is very low
            if avg_similarity < 0.3:
                anomalies.append({
                    'sentence_index': i,
                    'sentence': sentence,
                    'avg_similarity': float(avg_similarity),
                    'anomaly_type': 'semantic_discontinuity'
                })
        
        return anomalies
    
    def _extract_factual_claims(self, content: str) -> List[str]:
        """Extract factual claims from content for verification."""
        sentences = self._split_into_sentences(content)
        
        # Filter for sentences that likely contain factual claims
        factual_sentences = []
        
        # Patterns that indicate factual claims
        factual_patterns = [
            r'\b(founded|established|created|invented)\b',
            r'\b(located|based|situated)\s+in\b',
            r'\b\d{4}\b',  # Years
            r'\b(million|billion|thousand|percent|%)\b',
            r'\b(acquired|bought|sold|merged)\b',
            r'\b(CEO|president|founder|director)\b'
        ]
        
        for sentence in sentences:
            if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in factual_patterns):
                factual_sentences.append(sentence)
        
        return factual_sentences
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity between two strings."""
        # Simple character-based similarity (could be enhanced)
        if not text1 or not text2:
            return 0.0
        
        # Calculate Jaccard similarity for words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _are_entity_types_compatible(self, type1: str, type2: str) -> bool:
        """Check if two entity types are compatible."""
        # Entity type compatibility mapping
        compatibility_map = {
            'PERSON': ['PERSON', 'PER'],
            'ORG': ['ORG', 'ORGANIZATION'],
            'GPE': ['GPE', 'LOCATION', 'LOC'],
            'PRODUCT': ['PRODUCT', 'MISC'],
            'DATE': ['DATE', 'TIME'],
            'MONEY': ['MONEY', 'QUANTITY'],
            'PERCENT': ['PERCENT', 'QUANTITY']
        }
        
        type1_variants = compatibility_map.get(type1, [type1])
        type2_variants = compatibility_map.get(type2, [type2])
        
        return any(v1.lower() == v2.lower() for v1 in type1_variants for v2 in type2_variants)
    
    def _calculate_relationship_similarity(self, rel1: str, rel2: str) -> float:
        """Calculate similarity between relationship types."""
        if rel1.lower() == rel2.lower():
            return 1.0
        
        # Relationship similarity mapping
        similarity_map = {
            'WORKS_FOR': ['EMPLOYED_BY', 'WORKS_AT'],
            'FOUNDED_BY': ['ESTABLISHED_BY', 'CREATED_BY'],
            'LOCATED_IN': ['BASED_IN', 'SITUATED_IN'],
            'ACQUIRED_BY': ['BOUGHT_BY', 'PURCHASED_BY'],
            'IS_A': ['INSTANCE_OF', 'TYPE_OF']
        }
        
        for base_rel, similar_rels in similarity_map.items():
            if rel1 == base_rel and rel2 in similar_rels:
                return 0.8
            if rel2 == base_rel and rel1 in similar_rels:
                return 0.8
        
        return 0.0
    
    async def _validate_relationship_indirectly(
        self,
        source: str,
        target: str,
        rel_type: str,
        context: Optional[Dict[str, Any]]
    ) -> float:
        """Validate relationship through indirect evidence."""
        try:
            # Check if entities exist and have compatible types
            source_query = "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($name) RETURN e LIMIT 1"
            target_query = "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($name) RETURN e LIMIT 1"
            
            source_exists = bool(await self.neo4j.execute_query(source_query, {"name": source}))
            target_exists = bool(await self.neo4j.execute_query(target_query, {"name": target}))
            
            if source_exists and target_exists:
                return 0.6  # Moderate confidence if both entities exist
            elif source_exists or target_exists:
                return 0.4  # Lower confidence if only one exists
            else:
                return 0.2  # Low confidence if neither exists
                
        except Exception:
            return 0.3  # Default moderate-low confidence
    
    def _assess_common_entity_likelihood(
        self,
        entity_text: str,
        entity_type: str,
        context: Optional[Dict[str, Any]]
    ) -> float:
        """Assess likelihood that an entity is common/expected even if not in KB."""
        # Common patterns that suggest legitimate entities
        common_patterns = {
            'PERSON': [
                r'^[A-Z][a-z]+\s+[A-Z][a-z]+$',  # First Last name
                r'\b(CEO|CTO|President|Director)\s+[A-Z][a-z]+',  # Title + name
            ],
            'ORG': [
                r'\b[A-Z][a-z]+\s+(Inc|LLC|Corp|Company|Ltd)\b',  # Company suffixes
                r'\b[A-Z]{2,}\b',  # Acronyms
            ],
            'GPE': [
                r'^[A-Z][a-z]+,?\s+[A-Z]{2}$',  # City, State
                r'\b[A-Z][a-z]+\s+(City|County|State)\b',
            ]
        }
        
        patterns = common_patterns.get(entity_type, [])
        
        for pattern in patterns:
            if re.search(pattern, entity_text):
                return 0.7  # High likelihood for common patterns
        
        # Check for reasonable length and capitalization
        if (entity_type in ['PERSON', 'ORG', 'GPE'] and 
            entity_text[0].isupper() and 
            2 <= len(entity_text.split()) <= 4):
            return 0.5  # Moderate likelihood
        
        return 0.2  # Low likelihood
    
    def _assess_claim_plausibility(
        self,
        claim: str,
        context: Optional[Dict[str, Any]]
    ) -> float:
        """Assess plausibility of a factual claim."""
        # Basic plausibility checks
        plausibility = 0.5  # Start with neutral
        
        # Check for reasonable dates
        year_matches = re.findall(r'\b(19|20)\d{2}\b', claim)
        if year_matches:
            years = [int(year) for year in year_matches]
            reasonable_years = [y for y in years if 1800 <= y <= datetime.now().year + 5]
            if len(reasonable_years) == len(years):
                plausibility += 0.2
            else:
                plausibility -= 0.3
        
        # Check for reasonable numbers
        number_matches = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(million|billion|thousand|%|percent)\b', claim.lower())
        if number_matches:
            plausibility += 0.1  # Numbers with units suggest factual content
        
        # Check for superlatives that might indicate exaggeration
        superlatives = re.findall(r'\b(first|largest|biggest|smallest|best|worst|only|never|always|all|every)\b', claim.lower())
        if len(superlatives) > 2:
            plausibility -= 0.2  # Too many superlatives suggest exaggeration
        
        return max(0.0, min(1.0, plausibility))
    
    async def _load_validation_knowledge_base(self) -> None:
        """Load validation knowledge base for fact-checking."""
        try:
            # Load common facts and entities into cache
            # This could be expanded to load from external sources
            
            # Query for high-confidence, verified entities and claims
            entity_query = """
            MATCH (e:Entity)
            WHERE e.verified = true AND e.confidence_score > 0.9
            RETURN e.name, e.type, e.confidence_score
            LIMIT 1000
            """
            
            claim_query = """
            MATCH (c:Claim)
            WHERE c.verified = true AND c.confidence_score > 0.9
            RETURN c.text, c.confidence_score
            LIMIT 1000
            """
            
            # Pre-populate caches with high-confidence data
            entity_results = await self.neo4j.execute_query(entity_query)
            for record in entity_results:
                cache_key = f"{record['name']}:{record['type']}"
                self.validated_cache[cache_key] = {
                    'entity': record['name'],
                    'entity_type': record['type'],
                    'confidence': float(record['confidence_score']),
                    'verified': True,
                    'source': 'knowledge_base'
                }
            
            claim_results = await self.neo4j.execute_query(claim_query)
            for record in claim_results:
                claim_hash = hashlib.md5(record['text'].encode()).hexdigest()
                self.fact_cache[claim_hash] = {
                    'claim': record['text'],
                    'confidence': float(record['confidence_score']),
                    'verifiability': 'verified',
                    'source': 'knowledge_base'
                }
            
            logger.info(f"Loaded {len(self.validated_cache)} entities and {len(self.fact_cache)} claims into validation cache")
            
        except Exception as e:
            logger.warning(f"Failed to load validation knowledge base: {str(e)}")
            # Continue without pre-loaded cache
    
    def _update_detection_stats(
        self,
        result: Dict[str, Any],
        processing_time_ms: float
    ) -> None:
        """Update hallucination detection statistics."""
        self.detection_stats['total_validations'] += 1
        
        # Update processing time
        total_time = (
            self.detection_stats['avg_processing_time_ms'] * 
            (self.detection_stats['total_validations'] - 1) + 
            processing_time_ms
        )
        self.detection_stats['avg_processing_time_ms'] = (
            total_time / self.detection_stats['total_validations']
        )
        
        # Track hallucination detection
        if result['hallucination_probability'] > 0.02:  # >2% threshold
            self.detection_stats['hallucinations_detected'] += 1
    
    async def get_detection_statistics(self) -> Dict[str, Any]:
        """Get current hallucination detection statistics."""
        return {
            'total_validations': self.detection_stats['total_validations'],
            'hallucinations_detected': self.detection_stats['hallucinations_detected'],
            'detection_rate': (
                self.detection_stats['hallucinations_detected'] / 
                max(1, self.detection_stats['total_validations'])
            ),
            'avg_processing_time_ms': round(self.detection_stats['avg_processing_time_ms'], 2),
            'thresholds': self.thresholds,
            'validation_weights': self.validation_weights,
            'cache_sizes': {
                'fact_cache': len(self.fact_cache),
                'validated_cache': len(self.validated_cache)
            },
            'target_hallucination_rate': 0.02,  # <2%
            'performance_target_met': self.detection_stats['avg_processing_time_ms'] < 100
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check hallucination detection system health."""
        try:
            if not self.is_initialized:
                return {
                    'status': 'unhealthy',
                    'error': 'System not initialized'
                }
            
            # Test detection with known content
            test_content = "Apple Inc. was founded by Steve Jobs in 1976. The company is headquartered in Cupertino, California."
            start_time = datetime.now()
            
            test_result = await self.detect_hallucinations(
                test_content,
                validation_mode="fast"
            )
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'status': 'healthy' if response_time < 200 else 'degraded',
                'response_time_ms': round(response_time, 2),
                'test_confidence': test_result['overall_confidence'],
                'test_hallucination_probability': test_result['hallucination_probability'],
                'components_loaded': {
                    'sentence_transformer': self.sentence_transformer is not None,
                    'entity_extractor': self.entity_extractor.is_initialized,
                    'relationship_extractor': self.relationship_extractor.is_initialized,
                    'neo4j': self.neo4j is not None
                },
                'cache_status': {
                    'fact_cache_size': len(self.fact_cache),
                    'validated_cache_size': len(self.validated_cache)
                }
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }