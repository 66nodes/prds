"""
Fact-checking service against knowledge base for GraphRAG system.
Validates claims against structured knowledge graph and external sources.
"""

import asyncio
from typing import Any, Dict, List, Set, Tuple, Optional, Union
import uuid
import re
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import structlog
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from core.config import get_settings
from .neo4j_optimizer import Neo4jQueryOptimizer, verify_factual_claim
from .entity_extractor import EntityExtractionPipeline
from .relationship_extractor import RelationshipExtractor
from .hallucination_detector import HallucinationDetector

logger = structlog.get_logger(__name__)
settings = get_settings()


class ClaimType(Enum):
    """Types of claims that can be fact-checked."""
    FACTUAL = "factual"                    # Verifiable facts
    DEFINITIONAL = "definitional"          # Definitions and descriptions  
    NUMERICAL = "numerical"                # Numbers, dates, quantities
    RELATIONAL = "relational"              # Relationships between entities
    TEMPORAL = "temporal"                  # Time-based claims
    GEOGRAPHICAL = "geographical"          # Location-based claims
    ORGANIZATIONAL = "organizational"      # Company/org structure claims
    BIOGRAPHICAL = "biographical"          # People-related facts


class FactCheckResult(Enum):
    """Fact-checking result status."""
    VERIFIED = "verified"                  # Claim is verified as true
    CONTRADICTED = "contradicted"          # Claim contradicts known facts
    UNSUPPORTED = "unsupported"           # No evidence found for claim
    PARTIALLY_SUPPORTED = "partially_supported"  # Some evidence supports claim
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"  # Not enough evidence
    AMBIGUOUS = "ambiguous"               # Evidence is conflicting


@dataclass
class Claim:
    """Represents a factual claim extracted from content."""
    claim_id: str
    text: str
    claim_type: ClaimType
    entities: List[str]
    confidence: float
    start_pos: int
    end_pos: int
    extracted_facts: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extracted_facts is None:
            self.extracted_facts = {}


@dataclass
class FactCheckEvidence:
    """Evidence supporting or contradicting a claim."""
    evidence_id: str
    source_type: str  # 'knowledge_graph', 'external_source', 'document'
    evidence_text: str
    confidence: float
    supports_claim: bool
    source_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.source_metadata is None:
            self.source_metadata = {}


@dataclass
class FactCheckReport:
    """Comprehensive fact-checking report for a claim."""
    claim: Claim
    result: FactCheckResult
    overall_confidence: float
    evidence: List[FactCheckEvidence]
    contradictions: List[str]
    supporting_facts: List[str]
    confidence_breakdown: Dict[str, float]
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


class KnowledgeBaseFactChecker:
    """
    Advanced fact-checking service against knowledge base and external sources.
    Uses GraphRAG components for comprehensive claim verification.
    """
    
    def __init__(self):
        self.neo4j_optimizer = Neo4jQueryOptimizer()
        self.entity_extractor = EntityExtractionPipeline()
        self.relationship_extractor = RelationshipExtractor()
        self.hallucination_detector = HallucinationDetector()
        
        # Sentence transformer for semantic similarity
        self.sentence_model = None
        self.is_initialized = False
        
        # Claim extraction patterns
        self.claim_patterns = {
            ClaimType.FACTUAL: [
                r'(.+?)\s+(?:is|was|are|were)\s+(.+?)(?:\.|$)',
                r'(.+?)\s+(?:has|have|had)\s+(.+?)(?:\.|$)',
                r'According to (.+?),\s*(.+?)(?:\.|$)'
            ],
            ClaimType.NUMERICAL: [
                r'(.+?)\s+(?:is|was|are|were)\s+(\d+(?:,\d{3})*(?:\.\d+)?)\s*(.+?)(?:\.|$)',
                r'(.+?)\s+(?:costs?|worth|valued at)\s+\$?(\d+(?:,\d{3})*(?:\.\d+)?)\s*(.+?)(?:\.|$)',
                r'(.+?)\s+(?:in|since|during)\s+(\d{4})\s*(.+?)(?:\.|$)'
            ],
            ClaimType.RELATIONAL: [
                r'(.+?)\s+(?:founded|established|created)\s+(.+?)(?:\.|$)',
                r'(.+?)\s+(?:works for|employed by|CEO of)\s+(.+?)(?:\.|$)',
                r'(.+?)\s+(?:located in|based in|headquartered in)\s+(.+?)(?:\.|$)'
            ],
            ClaimType.TEMPORAL: [
                r'(.+?)\s+(?:in|on|during)\s+(\d{4}|\d{1,2}/\d{1,2}/\d{4}|January|February|March|April|May|June|July|August|September|October|November|December)\s*(.+?)(?:\.|$)',
                r'(.+?)\s+(?:since|from|until|before|after)\s+(\d{4})\s*(.+?)(?:\.|$)'
            ]
        }
        
        # Confidence thresholds for different evidence types
        self.evidence_thresholds = {
            'knowledge_graph': 0.8,     # High confidence for graph facts
            'verified_document': 0.7,   # Medium-high for verified docs
            'external_source': 0.6,     # Medium for external sources
            'computed_fact': 0.5        # Lower for computed/inferred facts
        }
        
        # Performance tracking
        self.fact_check_stats = {
            'total_claims_checked': 0,
            'avg_processing_time_ms': 0,
            'result_distribution': {result.value: 0 for result in FactCheckResult},
            'claim_type_distribution': {claim_type.value: 0 for claim_type in ClaimType},
            'evidence_source_counts': {},
            'avg_evidence_per_claim': 0
        }
    
    async def initialize(self) -> None:
        """Initialize the fact-checking service and all components."""
        try:
            logger.info("Initializing knowledge base fact-checker...")
            start_time = datetime.now()
            
            # Initialize components in parallel
            await asyncio.gather(
                self.neo4j_optimizer.initialize(),
                self.entity_extractor.initialize(),
                self.relationship_extractor.initialize(),
                self.hallucination_detector.initialize()
            )
            
            # Initialize sentence transformer for semantic similarity
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence transformer model loaded")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {str(e)}")
                self.sentence_model = None
            
            init_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"Knowledge base fact-checker initialized in {init_time:.2f}ms")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base fact-checker: {str(e)}")
            raise
    
    async def fact_check_content(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        claim_types: Optional[List[ClaimType]] = None,
        min_claim_confidence: float = 0.6,
        max_claims: int = 50
    ) -> Dict[str, Any]:
        """
        Perform comprehensive fact-checking on content.
        
        Args:
            content: Content to fact-check
            context: Optional context for domain-specific checking
            claim_types: Types of claims to extract and check
            min_claim_confidence: Minimum confidence for extracted claims
            max_claims: Maximum number of claims to check
            
        Returns:
            Dictionary with fact-checking results and analysis
        """
        if not self.is_initialized:
            raise RuntimeError("Knowledge base fact-checker not initialized")
        
        fact_check_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            logger.info(
                "Starting content fact-checking",
                fact_check_id=fact_check_id,
                content_length=len(content),
                max_claims=max_claims
            )
            
            # Extract claims from content
            claims = await self._extract_claims(
                content, context, claim_types, min_claim_confidence
            )
            
            # Limit number of claims to check
            claims = claims[:max_claims]
            
            # Fact-check each claim
            fact_check_reports = []
            for claim in claims:
                report = await self._fact_check_claim(claim, context)
                fact_check_reports.append(report)
            
            # Analyze overall fact-checking results
            overall_analysis = self._analyze_fact_check_results(fact_check_reports)
            
            # Generate content-level recommendations
            recommendations = self._generate_content_recommendations(fact_check_reports, overall_analysis)
            
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            result = {
                'fact_check_id': fact_check_id,
                'claims_extracted': len(claims),
                'claims_checked': len(fact_check_reports),
                'fact_check_reports': [self._report_to_dict(report) for report in fact_check_reports],
                'overall_analysis': overall_analysis,
                'recommendations': recommendations,
                'processing_time_ms': processing_time_ms,
                'timestamp': start_time.isoformat()
            }
            
            # Update statistics
            self._update_fact_check_stats(fact_check_reports, processing_time_ms)
            
            logger.info(
                "Content fact-checking completed",
                fact_check_id=fact_check_id,
                claims_checked=len(fact_check_reports),
                processing_time_ms=processing_time_ms,
                overall_accuracy=overall_analysis.get('accuracy_score', 0)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Content fact-checking failed: {str(e)}", fact_check_id=fact_check_id)
            raise
    
    async def fact_check_claim(
        self,
        claim_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> FactCheckReport:
        """
        Fact-check a single claim against the knowledge base.
        
        Args:
            claim_text: Text of the claim to fact-check
            context: Optional context for fact-checking
            
        Returns:
            Detailed fact-checking report
        """
        # Create claim object
        claim = Claim(
            claim_id=str(uuid.uuid4()),
            text=claim_text,
            claim_type=ClaimType.FACTUAL,  # Default type
            entities=[],
            confidence=1.0,
            start_pos=0,
            end_pos=len(claim_text)
        )
        
        # Classify claim type
        claim.claim_type = self._classify_claim_type(claim_text)
        
        # Extract entities from claim
        entity_result = await self.entity_extractor.extract_entities(
            claim_text, context, use_transformer=True, min_confidence=0.6
        )
        claim.entities = [entity['text'] for entity in entity_result['entities']]
        
        # Perform fact-checking
        return await self._fact_check_claim(claim, context)
    
    async def _extract_claims(
        self,
        content: str,
        context: Optional[Dict[str, Any]],
        claim_types: Optional[List[ClaimType]],
        min_confidence: float
    ) -> List[Claim]:
        """Extract factual claims from content."""
        claims = []
        claim_types = claim_types or list(ClaimType)
        
        try:
            # Use pattern matching to extract claims
            for claim_type in claim_types:
                patterns = self.claim_patterns.get(claim_type, [])
                
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    
                    for match in matches:
                        claim_text = match.group(0).strip()
                        
                        # Skip very short claims
                        if len(claim_text.split()) < 3:
                            continue
                        
                        # Extract entities from claim
                        entity_result = await self.entity_extractor.extract_entities(
                            claim_text, context, use_transformer=False, min_confidence=0.5
                        )
                        
                        entities = [entity['text'] for entity in entity_result['entities']]
                        
                        # Calculate claim confidence based on entity presence and pattern quality
                        claim_confidence = self._calculate_claim_confidence(
                            claim_text, entities, claim_type, match
                        )
                        
                        if claim_confidence >= min_confidence:
                            claim = Claim(
                                claim_id=str(uuid.uuid4()),
                                text=claim_text,
                                claim_type=claim_type,
                                entities=entities,
                                confidence=claim_confidence,
                                start_pos=match.start(),
                                end_pos=match.end()
                            )
                            
                            claims.append(claim)
            
            # Use hallucination detector to identify additional claims
            hall_result = await self.hallucination_detector.detect_hallucinations(
                content, context, validation_mode="comprehensive"
            )
            
            # Extract claims from validation tiers
            validation_tiers = hall_result.get('validation_tiers', {})
            for tier_name, tier_result in validation_tiers.items():
                tier_claims = tier_result.get('claims', [])
                for claim_data in tier_claims:
                    if claim_data.get('confidence', 0) >= min_confidence:
                        claim = Claim(
                            claim_id=str(uuid.uuid4()),
                            text=claim_data.get('text', ''),
                            claim_type=ClaimType.FACTUAL,
                            entities=claim_data.get('entities', []),
                            confidence=claim_data.get('confidence', 0),
                            start_pos=0,
                            end_pos=len(claim_data.get('text', ''))
                        )
                        claims.append(claim)
            
            # Remove duplicate claims
            unique_claims = self._deduplicate_claims(claims)
            
            # Sort by confidence
            unique_claims.sort(key=lambda x: x.confidence, reverse=True)
            
            return unique_claims
            
        except Exception as e:
            logger.error(f"Claim extraction failed: {str(e)}")
            return []
    
    async def _fact_check_claim(self, claim: Claim, context: Optional[Dict[str, Any]]) -> FactCheckReport:
        """Fact-check a single claim against multiple evidence sources."""
        try:
            evidence = []
            
            # Check against knowledge graph
            kg_evidence = await self._check_knowledge_graph(claim, context)
            evidence.extend(kg_evidence)
            
            # Check against verified claims database
            verified_evidence = await self._check_verified_claims(claim, context)
            evidence.extend(verified_evidence)
            
            # Check entity relationships
            if len(claim.entities) >= 2:
                relationship_evidence = await self._check_entity_relationships(claim, context)
                evidence.extend(relationship_evidence)
            
            # Check numerical facts if applicable
            if claim.claim_type == ClaimType.NUMERICAL:
                numerical_evidence = await self._check_numerical_facts(claim, context)
                evidence.extend(numerical_evidence)
            
            # Analyze evidence and determine result
            result, confidence, contradictions, supporting_facts = self._analyze_evidence(claim, evidence)
            
            # Calculate confidence breakdown
            confidence_breakdown = self._calculate_confidence_breakdown(evidence)
            
            # Generate recommendations
            recommendations = self._generate_claim_recommendations(claim, result, evidence)
            
            report = FactCheckReport(
                claim=claim,
                result=result,
                overall_confidence=confidence,
                evidence=evidence,
                contradictions=contradictions,
                supporting_facts=supporting_facts,
                confidence_breakdown=confidence_breakdown,
                recommendations=recommendations
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Fact-checking claim failed: {str(e)}")
            # Return error report
            return FactCheckReport(
                claim=claim,
                result=FactCheckResult.INSUFFICIENT_EVIDENCE,
                overall_confidence=0.0,
                evidence=[],
                contradictions=[f"Error during fact-checking: {str(e)}"],
                supporting_facts=[],
                confidence_breakdown={}
            )
    
    async def _check_knowledge_graph(self, claim: Claim, context: Optional[Dict[str, Any]]) -> List[FactCheckEvidence]:
        """Check claim against knowledge graph data."""
        evidence = []
        
        try:
            # Use Neo4j optimizer to verify claim
            verification_results = await verify_factual_claim(
                self.neo4j_optimizer,
                claim.text,
                min_score=0.5,
                limit=5
            )
            
            for result in verification_results:
                confidence = result.get('confidence', 0)
                supports_claim = confidence > 0.7
                
                evidence_item = FactCheckEvidence(
                    evidence_id=str(uuid.uuid4()),
                    source_type='knowledge_graph',
                    evidence_text=result.get('claim', ''),
                    confidence=confidence,
                    supports_claim=supports_claim,
                    source_metadata={
                        'sources': result.get('sources', []),
                        'score': result.get('score', 0)
                    }
                )
                evidence.append(evidence_item)
            
            # Check entity facts
            for entity in claim.entities:
                entity_query = """
                MATCH (e:Entity {name: $entity_name})
                RETURN e.description as description, e.confidence_score as confidence,
                       e.verified as verified, e.source_refs as sources
                """
                
                entity_results, metrics = await self.neo4j_optimizer.execute_custom_query(
                    entity_query, {'entity_name': entity}
                )
                
                for record in entity_results:
                    description = record.get('description', '')
                    if description and len(description) > 10:
                        evidence_item = FactCheckEvidence(
                            evidence_id=str(uuid.uuid4()),
                            source_type='knowledge_graph',
                            evidence_text=f"Entity {entity}: {description}",
                            confidence=record.get('confidence', 0.5),
                            supports_claim=True,  # Assume supportive for now
                            source_metadata={
                                'entity': entity,
                                'verified': record.get('verified', False),
                                'sources': record.get('sources', [])
                            }
                        )
                        evidence.append(evidence_item)
            
            return evidence
            
        except Exception as e:
            logger.error(f"Knowledge graph fact-checking failed: {str(e)}")
            return []
    
    async def _check_verified_claims(self, claim: Claim, context: Optional[Dict[str, Any]]) -> List[FactCheckEvidence]:
        """Check claim against verified claims in the database."""
        evidence = []
        
        try:
            # Query for similar verified claims
            if self.sentence_model:
                # Use semantic similarity
                claims_query = """
                MATCH (c:Claim)
                WHERE c.verified = true AND c.confidence_score > 0.7
                RETURN c.text as text, c.confidence_score as confidence,
                       c.source_refs as sources, c.verified as verified
                LIMIT 20
                """
                
                claims_results, metrics = await self.neo4j_optimizer.execute_custom_query(
                    claims_query, {}
                )
                
                if claims_results:
                    # Calculate semantic similarity
                    claim_texts = [record['text'] for record in claims_results]
                    if claim_texts:
                        claim_embeddings = self.sentence_model.encode([claim.text] + claim_texts)
                        similarities = cosine_similarity([claim_embeddings[0]], claim_embeddings[1:])[0]
                        
                        for i, (record, similarity) in enumerate(zip(claims_results, similarities)):
                            if similarity > 0.7:  # High similarity threshold
                                supports_claim = similarity > 0.8
                                
                                evidence_item = FactCheckEvidence(
                                    evidence_id=str(uuid.uuid4()),
                                    source_type='verified_document',
                                    evidence_text=record['text'],
                                    confidence=min(record.get('confidence', 0.5), similarity),
                                    supports_claim=supports_claim,
                                    source_metadata={
                                        'semantic_similarity': float(similarity),
                                        'sources': record.get('sources', []),
                                        'verified': record.get('verified', False)
                                    }
                                )
                                evidence.append(evidence_item)
            
            return evidence
            
        except Exception as e:
            logger.error(f"Verified claims checking failed: {str(e)}")
            return []
    
    async def _check_entity_relationships(self, claim: Claim, context: Optional[Dict[str, Any]]) -> List[FactCheckEvidence]:
        """Check claim against known entity relationships."""
        evidence = []
        
        try:
            # Extract relationships from claim
            rel_result = await self.relationship_extractor.extract_relationships(
                claim.text, context=context, min_confidence=0.5
            )
            
            relationships = rel_result['relationships']
            
            for rel in relationships:
                # Query knowledge graph for this relationship
                rel_query = """
                MATCH (s:Entity {name: $source})-[r]->(t:Entity {name: $target})
                WHERE type(r) = $rel_type OR r.relationship_type = $rel_type
                RETURN s.name as source, type(r) as relationship, t.name as target,
                       r.confidence as confidence, r.verified as verified
                """
                
                rel_results, metrics = await self.neo4j_optimizer.execute_custom_query(
                    rel_query, {
                        'source': rel['source_entity'],
                        'target': rel['target_entity'],
                        'rel_type': rel['relationship_type']
                    }
                )
                
                for record in rel_results:
                    evidence_text = f"{record['source']} {record['relationship']} {record['target']}"
                    confidence = record.get('confidence', 0.5)
                    
                    evidence_item = FactCheckEvidence(
                        evidence_id=str(uuid.uuid4()),
                        source_type='knowledge_graph',
                        evidence_text=evidence_text,
                        confidence=confidence,
                        supports_claim=confidence > 0.6,
                        source_metadata={
                            'relationship_type': record['relationship'],
                            'verified': record.get('verified', False),
                            'extracted_relationship': rel
                        }
                    )
                    evidence.append(evidence_item)
            
            return evidence
            
        except Exception as e:
            logger.error(f"Entity relationship checking failed: {str(e)}")
            return []
    
    async def _check_numerical_facts(self, claim: Claim, context: Optional[Dict[str, Any]]) -> List[FactCheckEvidence]:
        """Check numerical claims against known facts."""
        evidence = []
        
        try:
            # Extract numbers from claim
            numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', claim.text)
            dates = re.findall(r'\b\d{4}\b', claim.text)
            
            # Check against numerical facts in knowledge graph
            for number in numbers:
                num_query = """
                MATCH (f:Fact)
                WHERE f.numerical_value = $number OR f.text CONTAINS $number
                RETURN f.text as fact, f.confidence_score as confidence,
                       f.source_refs as sources, f.verified as verified
                LIMIT 5
                """
                
                num_results, metrics = await self.neo4j_optimizer.execute_custom_query(
                    num_query, {'number': number}
                )
                
                for record in num_results:
                    evidence_item = FactCheckEvidence(
                        evidence_id=str(uuid.uuid4()),
                        source_type='computed_fact',
                        evidence_text=record['fact'],
                        confidence=record.get('confidence', 0.5),
                        supports_claim=True,
                        source_metadata={
                            'numerical_value': number,
                            'sources': record.get('sources', []),
                            'verified': record.get('verified', False)
                        }
                    )
                    evidence.append(evidence_item)
            
            # Check dates
            for date in dates:
                date_query = """
                MATCH (e:Event)
                WHERE e.year = $year OR e.date CONTAINS $year
                RETURN e.description as description, e.confidence_score as confidence,
                       e.verified as verified
                LIMIT 3
                """
                
                date_results, metrics = await self.neo4j_optimizer.execute_custom_query(
                    date_query, {'year': date}
                )
                
                for record in date_results:
                    evidence_item = FactCheckEvidence(
                        evidence_id=str(uuid.uuid4()),
                        source_type='knowledge_graph',
                        evidence_text=f"Event in {date}: {record['description']}",
                        confidence=record.get('confidence', 0.5),
                        supports_claim=True,
                        source_metadata={
                            'date': date,
                            'verified': record.get('verified', False)
                        }
                    )
                    evidence.append(evidence_item)
            
            return evidence
            
        except Exception as e:
            logger.error(f"Numerical fact checking failed: {str(e)}")
            return []
    
    def _analyze_evidence(
        self, 
        claim: Claim, 
        evidence: List[FactCheckEvidence]
    ) -> Tuple[FactCheckResult, float, List[str], List[str]]:
        """Analyze evidence to determine fact-check result."""
        if not evidence:
            return FactCheckResult.INSUFFICIENT_EVIDENCE, 0.0, [], []
        
        # Separate supporting and contradicting evidence
        supporting = [e for e in evidence if e.supports_claim]
        contradicting = [e for e in evidence if not e.supports_claim]
        
        # Calculate confidence scores
        support_confidence = np.mean([e.confidence for e in supporting]) if supporting else 0.0
        contradict_confidence = np.mean([e.confidence for e in contradicting]) if contradicting else 0.0
        
        # Determine result based on evidence balance
        if len(supporting) > 0 and len(contradicting) == 0:
            if support_confidence > 0.8:
                result = FactCheckResult.VERIFIED
            elif support_confidence > 0.6:
                result = FactCheckResult.PARTIALLY_SUPPORTED
            else:
                result = FactCheckResult.UNSUPPORTED
            overall_confidence = support_confidence
            
        elif len(contradicting) > 0 and len(supporting) == 0:
            result = FactCheckResult.CONTRADICTED
            overall_confidence = contradict_confidence
            
        elif len(supporting) > 0 and len(contradicting) > 0:
            if support_confidence > contradict_confidence * 1.2:
                result = FactCheckResult.PARTIALLY_SUPPORTED
                overall_confidence = support_confidence * 0.8  # Reduce confidence due to contradiction
            elif contradict_confidence > support_confidence * 1.2:
                result = FactCheckResult.CONTRADICTED
                overall_confidence = contradict_confidence * 0.8
            else:
                result = FactCheckResult.AMBIGUOUS
                overall_confidence = (support_confidence + contradict_confidence) / 2 * 0.6
        
        else:
            result = FactCheckResult.INSUFFICIENT_EVIDENCE
            overall_confidence = 0.0
        
        # Extract contradictions and supporting facts
        contradictions = [e.evidence_text for e in contradicting[:3]]  # Top 3
        supporting_facts = [e.evidence_text for e in supporting[:3]]   # Top 3
        
        return result, overall_confidence, contradictions, supporting_facts
    
    def _calculate_confidence_breakdown(self, evidence: List[FactCheckEvidence]) -> Dict[str, float]:
        """Calculate confidence breakdown by evidence source type."""
        breakdown = {}
        
        source_groups = {}
        for e in evidence:
            source_type = e.source_type
            if source_type not in source_groups:
                source_groups[source_type] = []
            source_groups[source_type].append(e.confidence)
        
        for source_type, confidences in source_groups.items():
            breakdown[source_type] = {
                'avg_confidence': np.mean(confidences),
                'evidence_count': len(confidences),
                'weight': len(confidences) / len(evidence)
            }
        
        return breakdown
    
    def _generate_claim_recommendations(
        self, 
        claim: Claim, 
        result: FactCheckResult, 
        evidence: List[FactCheckEvidence]
    ) -> List[str]:
        """Generate recommendations based on fact-check result."""
        recommendations = []
        
        if result == FactCheckResult.CONTRADICTED:
            recommendations.append(f"Claim contradicts known facts. Consider revising or removing.")
            if evidence:
                high_conf_contradictions = [e for e in evidence if not e.supports_claim and e.confidence > 0.8]
                if high_conf_contradictions:
                    recommendations.append(f"Strong contradictory evidence found from {high_conf_contradictions[0].source_type}")
        
        elif result == FactCheckResult.UNSUPPORTED:
            recommendations.append("No supporting evidence found. Add citations or verify claim accuracy.")
        
        elif result == FactCheckResult.INSUFFICIENT_EVIDENCE:
            recommendations.append("Insufficient evidence to verify claim. Consider adding more context or sources.")
        
        elif result == FactCheckResult.AMBIGUOUS:
            recommendations.append("Conflicting evidence found. Review claim for accuracy and provide clarification.")
        
        elif result == FactCheckResult.PARTIALLY_SUPPORTED:
            recommendations.append("Claim is partially supported. Consider strengthening with additional evidence.")
        
        elif result == FactCheckResult.VERIFIED:
            if len(evidence) < 2:
                recommendations.append("Claim is verified but consider adding additional supporting sources.")
        
        # Entity-specific recommendations
        if not claim.entities:
            recommendations.append("Claim lacks specific entities. Consider making claim more specific.")
        
        return recommendations
    
    def _classify_claim_type(self, claim_text: str) -> ClaimType:
        """Classify the type of claim based on content patterns."""
        claim_lower = claim_text.lower()
        
        # Check for numerical patterns
        if re.search(r'\d+', claim_text):
            if re.search(r'\b\d{4}\b', claim_text):  # Year pattern
                return ClaimType.TEMPORAL
            elif re.search(r'\$|\d+(?:,\d{3})*(?:\.\d+)?', claim_text):  # Money/number
                return ClaimType.NUMERICAL
        
        # Check for relational patterns
        relational_keywords = ['founded', 'established', 'created', 'works for', 'CEO', 'located', 'based in']
        if any(keyword in claim_lower for keyword in relational_keywords):
            return ClaimType.RELATIONAL
        
        # Check for geographical patterns
        geo_keywords = ['located', 'based', 'headquarters', 'country', 'city', 'state']
        if any(keyword in claim_lower for keyword in geo_keywords):
            return ClaimType.GEOGRAPHICAL
        
        # Check for temporal patterns
        temporal_keywords = ['in', 'since', 'during', 'before', 'after', 'when']
        if any(keyword in claim_lower for keyword in temporal_keywords):
            return ClaimType.TEMPORAL
        
        # Check for definitional patterns
        definitional_keywords = ['is', 'are', 'was', 'were', 'definition', 'means']
        if any(keyword in claim_lower for keyword in definitional_keywords):
            return ClaimType.DEFINITIONAL
        
        # Default to factual
        return ClaimType.FACTUAL
    
    def _calculate_claim_confidence(
        self, 
        claim_text: str, 
        entities: List[str], 
        claim_type: ClaimType, 
        match: re.Match
    ) -> float:
        """Calculate confidence score for extracted claim."""
        base_confidence = 0.7
        
        # Entity presence boost
        entity_boost = min(len(entities) * 0.1, 0.3)
        
        # Claim length consideration
        words = claim_text.split()
        if len(words) < 5:
            length_penalty = 0.2
        elif len(words) > 20:
            length_penalty = 0.1
        else:
            length_penalty = 0.0
        
        # Claim type specific adjustments
        type_boosts = {
            ClaimType.NUMERICAL: 0.1,      # Numbers are usually verifiable
            ClaimType.RELATIONAL: 0.15,    # Relationships are important
            ClaimType.DEFINITIONAL: 0.05,  # Definitions vary
            ClaimType.TEMPORAL: 0.1        # Dates are verifiable
        }
        
        type_boost = type_boosts.get(claim_type, 0.0)
        
        # Pattern match quality (based on regex match span)
        match_quality = min(len(match.group(0)) / len(claim_text), 1.0) * 0.1
        
        final_confidence = base_confidence + entity_boost + type_boost + match_quality - length_penalty
        
        return min(max(final_confidence, 0.1), 1.0)  # Clamp between 0.1 and 1.0
    
    def _deduplicate_claims(self, claims: List[Claim]) -> List[Claim]:
        """Remove duplicate claims based on similarity."""
        if not self.sentence_model or len(claims) <= 1:
            return claims
        
        try:
            # Calculate embeddings for all claims
            claim_texts = [claim.text for claim in claims]
            embeddings = self.sentence_model.encode(claim_texts)
            
            # Calculate similarity matrix
            similarities = cosine_similarity(embeddings)
            
            # Track claims to keep
            keep_indices = set(range(len(claims)))
            
            # Remove duplicates (similarity > 0.85)
            for i in range(len(claims)):
                if i not in keep_indices:
                    continue
                    
                for j in range(i + 1, len(claims)):
                    if j not in keep_indices:
                        continue
                        
                    if similarities[i][j] > 0.85:
                        # Keep the claim with higher confidence
                        if claims[i].confidence >= claims[j].confidence:
                            keep_indices.discard(j)
                        else:
                            keep_indices.discard(i)
                            break
            
            return [claims[i] for i in sorted(keep_indices)]
            
        except Exception as e:
            logger.warning(f"Claim deduplication failed: {str(e)}")
            return claims
    
    def _analyze_fact_check_results(self, reports: List[FactCheckReport]) -> Dict[str, Any]:
        """Analyze overall fact-checking results."""
        if not reports:
            return {
                'total_claims': 0,
                'accuracy_score': 0.0,
                'result_distribution': {},
                'avg_confidence': 0.0
            }
        
        # Calculate result distribution
        result_counts = {}
        for result in FactCheckResult:
            result_counts[result.value] = len([r for r in reports if r.result == result])
        
        # Calculate accuracy score
        verified_count = result_counts.get(FactCheckResult.VERIFIED.value, 0)
        partially_count = result_counts.get(FactCheckResult.PARTIALLY_SUPPORTED.value, 0)
        accuracy_score = (verified_count + partially_count * 0.5) / len(reports)
        
        # Calculate average confidence
        avg_confidence = np.mean([r.overall_confidence for r in reports])
        
        # Identify problematic claims
        problematic_claims = [
            r.claim.text for r in reports 
            if r.result in [FactCheckResult.CONTRADICTED, FactCheckResult.UNSUPPORTED]
        ]
        
        return {
            'total_claims': len(reports),
            'accuracy_score': round(accuracy_score, 3),
            'result_distribution': result_counts,
            'avg_confidence': round(avg_confidence, 3),
            'problematic_claims': problematic_claims[:5],  # Top 5
            'claim_type_distribution': {
                claim_type.value: len([r for r in reports if r.claim.claim_type == claim_type])
                for claim_type in ClaimType
            }
        }
    
    def _generate_content_recommendations(
        self, 
        reports: List[FactCheckReport], 
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate content-level recommendations based on fact-checking results."""
        recommendations = []
        
        accuracy = analysis.get('accuracy_score', 0)
        
        if accuracy < 0.5:
            recommendations.append("Content has low fact-check accuracy (< 50%). Major revision recommended.")
        elif accuracy < 0.7:
            recommendations.append("Content has moderate fact-check accuracy. Review and improve claims.")
        elif accuracy < 0.9:
            recommendations.append("Content has good fact-check accuracy. Minor improvements possible.")
        else:
            recommendations.append("Content has excellent fact-check accuracy.")
        
        # Check for contradicted claims
        contradicted = analysis.get('result_distribution', {}).get(FactCheckResult.CONTRADICTED.value, 0)
        if contradicted > 0:
            recommendations.append(f"Remove or correct {contradicted} contradicted claims.")
        
        # Check for unsupported claims
        unsupported = analysis.get('result_distribution', {}).get(FactCheckResult.UNSUPPORTED.value, 0)
        if unsupported > 0:
            recommendations.append(f"Add evidence for {unsupported} unsupported claims.")
        
        # Check evidence quality
        avg_evidence_per_claim = np.mean([len(r.evidence) for r in reports]) if reports else 0
        if avg_evidence_per_claim < 1.5:
            recommendations.append("Increase evidence sources for claims (target: 2+ sources per claim).")
        
        return recommendations
    
    def _report_to_dict(self, report: FactCheckReport) -> Dict[str, Any]:
        """Convert FactCheckReport to dictionary for serialization."""
        return {
            'claim': {
                'claim_id': report.claim.claim_id,
                'text': report.claim.text,
                'claim_type': report.claim.claim_type.value,
                'entities': report.claim.entities,
                'confidence': report.claim.confidence
            },
            'result': report.result.value,
            'overall_confidence': round(report.overall_confidence, 3),
            'evidence_count': len(report.evidence),
            'evidence': [
                {
                    'evidence_id': e.evidence_id,
                    'source_type': e.source_type,
                    'evidence_text': e.evidence_text[:200] + '...' if len(e.evidence_text) > 200 else e.evidence_text,
                    'confidence': round(e.confidence, 3),
                    'supports_claim': e.supports_claim
                }
                for e in report.evidence[:3]  # Top 3 evidence items
            ],
            'contradictions': report.contradictions,
            'supporting_facts': report.supporting_facts,
            'confidence_breakdown': {
                source_type: {
                    'avg_confidence': round(breakdown['avg_confidence'], 3),
                    'evidence_count': breakdown['evidence_count']
                }
                for source_type, breakdown in report.confidence_breakdown.items()
            },
            'recommendations': report.recommendations
        }
    
    def _update_fact_check_stats(self, reports: List[FactCheckReport], processing_time_ms: float) -> None:
        """Update fact-checking statistics."""
        self.fact_check_stats['total_claims_checked'] += len(reports)
        
        # Update average processing time
        if self.fact_check_stats['total_claims_checked'] > 0:
            total_time = (
                self.fact_check_stats['avg_processing_time_ms'] * 
                (self.fact_check_stats['total_claims_checked'] - len(reports)) + 
                processing_time_ms
            )
            self.fact_check_stats['avg_processing_time_ms'] = (
                total_time / self.fact_check_stats['total_claims_checked']
            )
        
        # Update result distribution
        for report in reports:
            result_key = report.result.value
            self.fact_check_stats['result_distribution'][result_key] += 1
        
        # Update claim type distribution
        for report in reports:
            claim_type_key = report.claim.claim_type.value
            self.fact_check_stats['claim_type_distribution'][claim_type_key] += 1
        
        # Update evidence source counts
        for report in reports:
            for evidence in report.evidence:
                source_type = evidence.source_type
                self.fact_check_stats['evidence_source_counts'][source_type] = (
                    self.fact_check_stats['evidence_source_counts'].get(source_type, 0) + 1
                )
        
        # Update average evidence per claim
        if reports:
            total_evidence = sum(len(report.evidence) for report in reports)
            self.fact_check_stats['avg_evidence_per_claim'] = total_evidence / len(reports)
    
    async def get_fact_check_statistics(self) -> Dict[str, Any]:
        """Get current fact-checking statistics."""
        stats = dict(self.fact_check_stats)
        
        # Calculate additional metrics
        if stats['total_claims_checked'] > 0:
            verified_count = stats['result_distribution'].get(FactCheckResult.VERIFIED.value, 0)
            partially_count = stats['result_distribution'].get(FactCheckResult.PARTIALLY_SUPPORTED.value, 0)
            stats['accuracy_rate'] = (verified_count + partially_count * 0.5) / stats['total_claims_checked']
            
            contradicted_count = stats['result_distribution'].get(FactCheckResult.CONTRADICTED.value, 0)
            stats['error_rate'] = contradicted_count / stats['total_claims_checked']
        else:
            stats['accuracy_rate'] = 0
            stats['error_rate'] = 0
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Check fact-checker health."""
        try:
            if not self.is_initialized:
                return {
                    'status': 'unhealthy',
                    'error': 'Knowledge base fact-checker not initialized'
                }
            
            # Test fact-checking
            start_time = datetime.now()
            
            test_claim = "Apple Inc. was founded by Steve Jobs."
            test_result = await self.fact_check_claim(test_claim)
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Check component health
            component_health = await asyncio.gather(
                self.neo4j_optimizer.health_check(),
                self.entity_extractor.health_check(),
                self.relationship_extractor.health_check(),
                self.hallucination_detector.health_check(),
                return_exceptions=True
            )
            
            component_status = {}
            for i, (component_name, health) in enumerate([
                ('neo4j_optimizer', component_health[0]),
                ('entity_extractor', component_health[1]),
                ('relationship_extractor', component_health[2]),
                ('hallucination_detector', component_health[3])
            ]):
                if isinstance(health, Exception):
                    component_status[component_name] = 'error'
                else:
                    component_status[component_name] = health.get('status', 'unknown')
            
            all_healthy = all(status in ['healthy', 'degraded'] for status in component_status.values())
            
            return {
                'status': 'healthy' if all_healthy and response_time < 3000 else 'degraded',
                'response_time_ms': round(response_time, 2),
                'test_fact_check_result': test_result.result.value,
                'test_confidence': test_result.overall_confidence,
                'component_status': component_status,
                'sentence_transformer_loaded': self.sentence_model is not None,
                'total_claims_checked': self.fact_check_stats['total_claims_checked'],
                'avg_processing_time_ms': round(self.fact_check_stats['avg_processing_time_ms'], 2),
                'supported_claim_types': [claim_type.value for claim_type in ClaimType]
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def close(self) -> None:
        """Close fact-checker and cleanup resources."""
        await asyncio.gather(
            self.neo4j_optimizer.close(),
            self.entity_extractor.close() if hasattr(self.entity_extractor, 'close') else asyncio.sleep(0),
            self.relationship_extractor.close() if hasattr(self.relationship_extractor, 'close') else asyncio.sleep(0),
            self.hallucination_detector.close() if hasattr(self.hallucination_detector, 'close') else asyncio.sleep(0),
            return_exceptions=True
        )
        
        # Clear sentence model
        self.sentence_model = None
        self.is_initialized = False
        logger.info("Knowledge base fact-checker closed")


# Convenience functions for common fact-checking operations

async def quick_fact_check(
    fact_checker: KnowledgeBaseFactChecker,
    claim_text: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Quick fact-check for a single claim."""
    report = await fact_checker.fact_check_claim(claim_text, context)
    return {
        'claim': claim_text,
        'result': report.result.value,
        'confidence': report.overall_confidence,
        'evidence_count': len(report.evidence),
        'recommendations': report.recommendations
    }


async def batch_fact_check_claims(
    fact_checker: KnowledgeBaseFactChecker,
    claims: List[str],
    context: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Fact-check multiple claims in batch."""
    tasks = [fact_checker.fact_check_claim(claim, context) for claim in claims]
    reports = await asyncio.gather(*tasks)
    
    return [
        {
            'claim': report.claim.text,
            'result': report.result.value,
            'confidence': report.overall_confidence,
            'evidence_count': len(report.evidence)
        }
        for report in reports
    ]


async def verify_document_accuracy(
    fact_checker: KnowledgeBaseFactChecker,
    document_content: str,
    context: Optional[Dict[str, Any]] = None,
    accuracy_threshold: float = 0.8
) -> Dict[str, Any]:
    """Verify overall document accuracy against knowledge base."""
    result = await fact_checker.fact_check_content(
        document_content, context, max_claims=20
    )
    
    accuracy = result['overall_analysis']['accuracy_score']
    
    return {
        'document_verified': accuracy >= accuracy_threshold,
        'accuracy_score': accuracy,
        'claims_checked': result['claims_checked'],
        'problematic_claims': result['overall_analysis'].get('problematic_claims', []),
        'recommendations': result['recommendations']
    }