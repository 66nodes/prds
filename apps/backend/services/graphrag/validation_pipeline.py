"""
Comprehensive validation pipeline for GraphRAG content validation.
Orchestrates entity extraction, relationship extraction, hallucination detection, and graph traversal.
"""

import asyncio
from typing import Any, Dict, List, Set, Tuple, Optional, Union
import uuid
from datetime import datetime
from enum import Enum
import json
from dataclasses import dataclass, asdict

import structlog
import numpy as np

from core.config import get_settings
from .entity_extractor import EntityExtractionPipeline
from .relationship_extractor import RelationshipExtractor
from .hallucination_detector import HallucinationDetector
from .neo4j_optimizer import Neo4jQueryOptimizer
from .graph_traversal import GraphTraversalStrategies
from ..cache_service import get_cache_service, CacheNamespace

logger = structlog.get_logger(__name__)
settings = get_settings()


class ValidationLevel(Enum):
    """Validation levels for different content types and requirements."""
    BASIC = "basic"          # Fast validation for drafts
    STANDARD = "standard"    # Comprehensive validation for production
    STRICT = "strict"        # Maximum validation for critical content
    CUSTOM = "custom"        # Custom validation configuration


class ValidationStatus(Enum):
    """Validation result status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    REQUIRES_REVIEW = "requires_review"
    ERROR = "error"


@dataclass
class ValidationConfig:
    """Configuration for validation pipeline."""
    level: ValidationLevel = ValidationLevel.STANDARD
    
    # Component enablement
    enable_entity_extraction: bool = True
    enable_relationship_extraction: bool = True
    enable_hallucination_detection: bool = True
    enable_graph_traversal: bool = True
    
    # Thresholds
    min_entity_confidence: float = 0.7
    min_relationship_confidence: float = 0.6
    hallucination_threshold: float = 0.02  # 2% max hallucination rate
    overall_confidence_threshold: float = 0.8
    
    # Processing limits
    max_processing_time_ms: int = 10000  # 10 seconds max
    max_entities: int = 100
    max_relationships: int = 200
    max_graph_depth: int = 3
    
    # Content-specific settings
    context_domain: Optional[str] = None
    focus_entity_types: Optional[List[str]] = None
    required_evidence_sources: int = 2
    
    # Output control
    include_detailed_analysis: bool = True
    include_recommendations: bool = True
    include_graph_insights: bool = False


@dataclass
class ValidationResult:
    """Comprehensive validation result."""
    validation_id: str
    status: ValidationStatus
    overall_confidence: float
    
    # Component results
    entity_extraction_result: Optional[Dict[str, Any]] = None
    relationship_extraction_result: Optional[Dict[str, Any]] = None
    hallucination_detection_result: Optional[Dict[str, Any]] = None
    graph_analysis_result: Optional[Dict[str, Any]] = None
    
    # Analysis
    issues_found: List[Dict[str, Any]] = None
    recommendations: List[str] = None
    corrections: List[Dict[str, Any]] = None
    
    # Metadata
    processing_time_ms: float = 0
    config_used: Optional[ValidationConfig] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.issues_found is None:
            self.issues_found = []
        if self.recommendations is None:
            self.recommendations = []
        if self.corrections is None:
            self.corrections = []


class ValidationPipeline:
    """
    Comprehensive validation pipeline orchestrating all GraphRAG components.
    Provides configurable validation levels and detailed analysis.
    """
    
    def __init__(self):
        # Initialize components
        self.entity_extractor = EntityExtractionPipeline()
        self.relationship_extractor = RelationshipExtractor()
        self.hallucination_detector = HallucinationDetector()
        self.neo4j_optimizer = Neo4jQueryOptimizer()
        self.graph_traversal = GraphTraversalStrategies()
        self._cache_service = get_cache_service()
        
        self.is_initialized = False
        
        # Predefined validation configurations
        self.validation_configs = {
            ValidationLevel.BASIC: ValidationConfig(
                level=ValidationLevel.BASIC,
                enable_graph_traversal=False,
                min_entity_confidence=0.6,
                min_relationship_confidence=0.5,
                hallucination_threshold=0.05,  # 5% for drafts
                overall_confidence_threshold=0.7,
                max_processing_time_ms=3000,
                include_graph_insights=False
            ),
            
            ValidationLevel.STANDARD: ValidationConfig(
                level=ValidationLevel.STANDARD,
                min_entity_confidence=0.7,
                min_relationship_confidence=0.6,
                hallucination_threshold=0.02,  # 2% standard
                overall_confidence_threshold=0.8,
                max_processing_time_ms=10000,
                include_graph_insights=True
            ),
            
            ValidationLevel.STRICT: ValidationConfig(
                level=ValidationLevel.STRICT,
                min_entity_confidence=0.8,
                min_relationship_confidence=0.7,
                hallucination_threshold=0.01,  # 1% strict
                overall_confidence_threshold=0.9,
                max_processing_time_ms=20000,
                required_evidence_sources=3,
                include_detailed_analysis=True,
                include_graph_insights=True
            )
        }
        
        # Performance tracking
        self.validation_stats = {
            'total_validations': 0,
            'avg_processing_time_ms': 0,
            'validation_level_counts': {},
            'avg_confidence_scores': {},
            'status_distribution': {status.value: 0 for status in ValidationStatus},
            'component_performance': {}
        }
    
    async def initialize(self) -> None:
        """Initialize the validation pipeline and all components."""
        try:
            logger.info("Initializing validation pipeline...")
            start_time = datetime.now()
            
            # Initialize components in parallel
            await asyncio.gather(
                self.entity_extractor.initialize(),
                self.relationship_extractor.initialize(),
                self.hallucination_detector.initialize(),
                self.neo4j_optimizer.initialize(),
                self.graph_traversal.initialize()
            )
            
            init_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"Validation pipeline initialized in {init_time:.2f}ms")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize validation pipeline: {str(e)}")
            raise
    
    async def validate_content(
        self,
        content: str,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        custom_config: Optional[ValidationConfig] = None,
        context: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate content using the specified validation level or custom configuration.
        
        Args:
            content: Content to validate
            validation_level: Predefined validation level
            custom_config: Custom validation configuration (overrides level)
            context: Optional context for domain-specific validation
            project_id: Optional project ID for cache scoping
            
        Returns:
            Comprehensive validation result
        """
        if not self.is_initialized:
            raise RuntimeError("Validation pipeline not initialized")
        
        validation_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Try to get cached validation result first
        if project_id and self._cache_service.is_available:
            cached_result = await self._cache_service.get_graphrag_validation(
                content=content,
                project_id=project_id
            )
            
            if cached_result:
                logger.info(
                    "Cache hit for GraphRAG validation",
                    project_id=project_id,
                    cached_at=cached_result.get("cached_at")
                )
                
                # Convert cached result back to ValidationResult
                result = ValidationResult(
                    validation_id=cached_result.get("validation_id", validation_id),
                    status=ValidationStatus(cached_result["status"]),
                    overall_confidence=cached_result["overall_confidence"],
                    entity_extraction_result=cached_result.get("entity_extraction_result"),
                    relationship_extraction_result=cached_result.get("relationship_extraction_result"),
                    hallucination_detection_result=cached_result.get("hallucination_detection_result"),
                    graph_analysis_result=cached_result.get("graph_analysis_result"),
                    issues_found=cached_result.get("issues_found", []),
                    recommendations=cached_result.get("recommendations", []),
                    corrections=cached_result.get("corrections", []),
                    processing_time_ms=cached_result.get("processing_time_ms", 0),
                    timestamp=cached_result.get("timestamp")
                )
                return result
        
        try:
            # Get validation configuration
            config = custom_config or self.validation_configs.get(
                validation_level, self.validation_configs[ValidationLevel.STANDARD]
            )
            
            logger.info(
                "Starting content validation",
                validation_id=validation_id,
                level=config.level.value,
                content_length=len(content)
            )
            
            # Initialize result
            result = ValidationResult(
                validation_id=validation_id,
                status=ValidationStatus.PASSED,
                overall_confidence=0.0,
                config_used=config,
                timestamp=start_time.isoformat()
            )
            
            # Execute validation components based on configuration
            component_results = {}
            
            if config.enable_entity_extraction:
                entity_start = datetime.now()
                result.entity_extraction_result = await self._extract_and_validate_entities(
                    content, config, context
                )
                entity_time = (datetime.now() - entity_start).total_seconds() * 1000
                component_results['entity_extraction_ms'] = entity_time
            
            if config.enable_relationship_extraction:
                rel_start = datetime.now()
                result.relationship_extraction_result = await self._extract_and_validate_relationships(
                    content, config, context, result.entity_extraction_result
                )
                rel_time = (datetime.now() - rel_start).total_seconds() * 1000
                component_results['relationship_extraction_ms'] = rel_time
            
            if config.enable_hallucination_detection:
                hall_start = datetime.now()
                result.hallucination_detection_result = await self._detect_and_validate_hallucinations(
                    content, config, context
                )
                hall_time = (datetime.now() - hall_start).total_seconds() * 1000
                component_results['hallucination_detection_ms'] = hall_time
            
            if config.enable_graph_traversal and result.entity_extraction_result:
                graph_start = datetime.now()
                result.graph_analysis_result = await self._analyze_graph_context(
                    content, config, context, result.entity_extraction_result
                )
                graph_time = (datetime.now() - graph_start).total_seconds() * 1000
                component_results['graph_analysis_ms'] = graph_time
            
            # Calculate overall confidence and status
            result.overall_confidence = self._calculate_overall_confidence(result, config)
            result.status = self._determine_validation_status(result, config)
            
            # Generate analysis and recommendations
            if config.include_detailed_analysis:
                result.issues_found = self._analyze_issues(result, config)
            
            if config.include_recommendations:
                result.recommendations = self._generate_recommendations(result, config)
                result.corrections = self._generate_corrections(result, config)
            
            # Calculate processing time
            result.processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Check processing time limit
            if result.processing_time_ms > config.max_processing_time_ms:
                result.status = ValidationStatus.WARNING
                result.issues_found.append({
                    'type': 'performance',
                    'severity': 'medium',
                    'message': f"Processing time ({result.processing_time_ms:.0f}ms) exceeded limit ({config.max_processing_time_ms}ms)",
                    'component': 'pipeline'
                })
            
            # Update statistics
            self._update_validation_stats(config.level, result, component_results)
            
            logger.info(
                "Content validation completed",
                validation_id=validation_id,
                status=result.status.value,
                confidence=result.overall_confidence,
                processing_time_ms=result.processing_time_ms
            )
            
            # Cache the validation result if project_id is provided
            if project_id and self._cache_service.is_available:
                # Convert result to cacheable format
                cache_data = {
                    "validation_id": result.validation_id,
                    "status": result.status.value,
                    "overall_confidence": result.overall_confidence,
                    "entity_extraction_result": result.entity_extraction_result,
                    "relationship_extraction_result": result.relationship_extraction_result,
                    "hallucination_detection_result": result.hallucination_detection_result,
                    "graph_analysis_result": result.graph_analysis_result,
                    "issues_found": result.issues_found,
                    "recommendations": result.recommendations,
                    "corrections": result.corrections,
                    "processing_time_ms": result.processing_time_ms,
                    "timestamp": result.timestamp
                }
                
                await self._cache_service.cache_graphrag_validation(
                    content=content,
                    project_id=project_id,
                    validation_result=cache_data
                )
                logger.debug(f"Cached GraphRAG validation result for project {project_id}")
            
            return result
            
        except Exception as e:
            error_result = ValidationResult(
                validation_id=validation_id,
                status=ValidationStatus.ERROR,
                overall_confidence=0.0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                timestamp=start_time.isoformat()
            )
            error_result.issues_found.append({
                'type': 'system_error',
                'severity': 'high',
                'message': f"Validation pipeline error: {str(e)}",
                'component': 'pipeline'
            })
            
            logger.error(f"Content validation failed: {str(e)}", validation_id=validation_id)
            return error_result
    
    async def _extract_and_validate_entities(
        self,
        content: str,
        config: ValidationConfig,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract entities and validate them against confidence thresholds."""
        try:
            # Extract entities
            entity_result = await self.entity_extractor.extract_entities(
                content,
                context,
                use_transformer=config.level != ValidationLevel.BASIC,
                min_confidence=config.min_entity_confidence
            )
            
            # Validate entity results
            entities = entity_result['entities']
            validation_issues = []
            
            # Check entity count limits
            if len(entities) > config.max_entities:
                validation_issues.append({
                    'type': 'entity_limit_exceeded',
                    'severity': 'medium',
                    'count': len(entities),
                    'limit': config.max_entities
                })
            
            # Check for required entity types
            if config.focus_entity_types:
                found_types = set(entity['label'] for entity in entities)
                missing_types = set(config.focus_entity_types) - found_types
                if missing_types:
                    validation_issues.append({
                        'type': 'missing_required_entity_types',
                        'severity': 'high',
                        'missing_types': list(missing_types)
                    })
            
            # Check confidence distribution
            confidences = [entity['confidence'] for entity in entities]
            low_confidence_entities = [
                entity for entity in entities 
                if entity['confidence'] < config.min_entity_confidence * 1.2
            ]
            
            if len(low_confidence_entities) > len(entities) * 0.3:  # More than 30% low confidence
                validation_issues.append({
                    'type': 'low_confidence_entities',
                    'severity': 'medium',
                    'count': len(low_confidence_entities),
                    'percentage': len(low_confidence_entities) / len(entities) * 100
                })
            
            # Enhanced result
            enhanced_result = {
                **entity_result,
                'validation_issues': validation_issues,
                'confidence_stats': {
                    'mean': np.mean(confidences) if confidences else 0,
                    'min': min(confidences) if confidences else 0,
                    'max': max(confidences) if confidences else 0,
                    'std': np.std(confidences) if confidences else 0
                },
                'entity_type_distribution': {
                    entity_type: len([e for e in entities if e['label'] == entity_type])
                    for entity_type in set(entity['label'] for entity in entities)
                },
                'passes_validation': len(validation_issues) == 0
            }
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Entity extraction validation failed: {str(e)}")
            return {
                'entities': [],
                'entity_count': 0,
                'validation_issues': [{
                    'type': 'entity_extraction_error',
                    'severity': 'high',
                    'message': str(e)
                }],
                'passes_validation': False
            }
    
    async def _extract_and_validate_relationships(
        self,
        content: str,
        config: ValidationConfig,
        context: Optional[Dict[str, Any]],
        entity_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract relationships and validate them."""
        try:
            # Extract relationships
            entities = entity_result['entities'] if entity_result else None
            relationship_result = await self.relationship_extractor.extract_relationships(
                content,
                entities,
                context,
                min_confidence=config.min_relationship_confidence
            )
            
            # Validate relationship results
            relationships = relationship_result['relationships']
            validation_issues = []
            
            # Check relationship count limits
            if len(relationships) > config.max_relationships:
                validation_issues.append({
                    'type': 'relationship_limit_exceeded',
                    'severity': 'medium',
                    'count': len(relationships),
                    'limit': config.max_relationships
                })
            
            # Check for isolated entities
            if entities:
                entities_in_relationships = set()
                for rel in relationships:
                    entities_in_relationships.add(rel['source_entity'])
                    entities_in_relationships.add(rel['target_entity'])
                
                isolated_entities = [
                    entity['text'] for entity in entities
                    if entity['text'] not in entities_in_relationships
                ]
                
                if len(isolated_entities) > len(entities) * 0.4:  # More than 40% isolated
                    validation_issues.append({
                        'type': 'high_isolated_entities',
                        'severity': 'medium',
                        'isolated_count': len(isolated_entities),
                        'percentage': len(isolated_entities) / len(entities) * 100
                    })
            
            # Check relationship confidence distribution
            confidences = [rel['confidence'] for rel in relationships]
            if confidences:
                low_confidence_rels = [
                    rel for rel in relationships
                    if rel['confidence'] < config.min_relationship_confidence * 1.2
                ]
                
                if len(low_confidence_rels) > len(relationships) * 0.3:
                    validation_issues.append({
                        'type': 'low_confidence_relationships',
                        'severity': 'medium',
                        'count': len(low_confidence_rels),
                        'percentage': len(low_confidence_rels) / len(relationships) * 100
                    })
            
            # Enhanced result
            enhanced_result = {
                **relationship_result,
                'validation_issues': validation_issues,
                'confidence_stats': {
                    'mean': np.mean(confidences) if confidences else 0,
                    'min': min(confidences) if confidences else 0,
                    'max': max(confidences) if confidences else 0,
                    'std': np.std(confidences) if confidences else 0
                },
                'relationship_type_distribution': {
                    rel_type: len([r for r in relationships if r['relationship_type'] == rel_type])
                    for rel_type in set(rel['relationship_type'] for rel in relationships)
                },
                'graph_connectivity': {
                    'total_entities_connected': len(entities_in_relationships) if 'entities_in_relationships' in locals() else 0,
                    'connectivity_ratio': len(entities_in_relationships) / len(entities) if entities and 'entities_in_relationships' in locals() else 0
                },
                'passes_validation': len(validation_issues) == 0
            }
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Relationship extraction validation failed: {str(e)}")
            return {
                'relationships': [],
                'relationship_count': 0,
                'validation_issues': [{
                    'type': 'relationship_extraction_error',
                    'severity': 'high',
                    'message': str(e)
                }],
                'passes_validation': False
            }
    
    async def _detect_and_validate_hallucinations(
        self,
        content: str,
        config: ValidationConfig,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Detect hallucinations and validate against threshold."""
        try:
            # Detect hallucinations
            hallucination_result = await self.hallucination_detector.detect_hallucinations(
                content,
                context,
                validation_mode="comprehensive" if config.level == ValidationLevel.STRICT else "standard"
            )
            
            # Validate hallucination results
            validation_issues = []
            hallucination_rate = hallucination_result['hallucination_rate']
            
            # Check hallucination threshold
            if hallucination_rate > config.hallucination_threshold:
                validation_issues.append({
                    'type': 'hallucination_threshold_exceeded',
                    'severity': 'high',
                    'hallucination_rate': hallucination_rate,
                    'threshold': config.hallucination_threshold,
                    'difference': hallucination_rate - config.hallucination_threshold
                })
            
            # Check validation tier performance
            validation_tiers = hallucination_result.get('validation_tiers', {})
            for tier_name, tier_result in validation_tiers.items():
                tier_confidence = tier_result.get('confidence', 0)
                if tier_confidence < 0.7:  # Tier-specific low confidence
                    validation_issues.append({
                        'type': f'low_{tier_name}_validation_confidence',
                        'severity': 'medium',
                        'tier': tier_name,
                        'confidence': tier_confidence
                    })
            
            # Check evidence sources
            evidence_sources = hallucination_result.get('evidence_sources', [])
            if len(evidence_sources) < config.required_evidence_sources:
                validation_issues.append({
                    'type': 'insufficient_evidence_sources',
                    'severity': 'high',
                    'found_sources': len(evidence_sources),
                    'required_sources': config.required_evidence_sources
                })
            
            # Enhanced result
            enhanced_result = {
                **hallucination_result,
                'validation_issues': validation_issues,
                'passes_threshold': hallucination_rate <= config.hallucination_threshold,
                'threshold_margin': config.hallucination_threshold - hallucination_rate,
                'evidence_adequacy': len(evidence_sources) >= config.required_evidence_sources,
                'passes_validation': len(validation_issues) == 0
            }
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Hallucination detection validation failed: {str(e)}")
            return {
                'hallucination_rate': 1.0,  # Assume worst case on error
                'validation_issues': [{
                    'type': 'hallucination_detection_error',
                    'severity': 'high',
                    'message': str(e)
                }],
                'passes_validation': False
            }
    
    async def _analyze_graph_context(
        self,
        content: str,
        config: ValidationConfig,
        context: Optional[Dict[str, Any]],
        entity_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze graph context for entities mentioned in content."""
        try:
            entities = entity_result['entities']
            if not entities:
                return {'graph_insights': [], 'passes_validation': True}
            
            # Select key entities for graph analysis
            key_entities = sorted(entities, key=lambda x: x['importance_score'], reverse=True)[:5]
            entity_names = [entity['text'] for entity in key_entities]
            
            # Perform graph analysis
            graph_results = {}
            
            # Entity neighborhood exploration
            if len(entity_names) > 0:
                neighborhood_result = await self.graph_traversal.traverse_breadth_first(
                    entity_names[0],
                    max_depth=config.max_graph_depth,
                    max_nodes=20,
                    min_confidence=0.5
                )
                graph_results['neighborhood_analysis'] = neighborhood_result
            
            # Find connections between entities
            if len(entity_names) > 1:
                connections_result = await self.graph_traversal.find_shortest_paths(
                    entity_names[0],
                    entity_names[1:3],  # Limit to avoid long processing
                    max_path_length=config.max_graph_depth
                )
                graph_results['entity_connections'] = connections_result
            
            # Centrality analysis for key entities
            centrality_result = await self.graph_traversal.analyze_centrality(
                entity_subset=entity_names,
                top_n=10
            )
            graph_results['centrality_analysis'] = centrality_result
            
            # Extract graph insights
            graph_insights = self._extract_graph_insights(graph_results, entities)
            
            # Validate graph consistency
            validation_issues = []
            
            # Check for disconnected entities
            neighborhood = graph_results.get('neighborhood_analysis', {})
            if neighborhood.get('total_nodes_explored', 0) < len(key_entities) * 0.8:
                validation_issues.append({
                    'type': 'low_graph_connectivity',
                    'severity': 'medium',
                    'message': 'Some key entities appear disconnected from the main graph'
                })
            
            # Check centrality consistency
            centrality = graph_results.get('centrality_analysis', {})
            influential_entities = centrality.get('influential_entities', [])
            mentioned_influential = [
                entity for entity in influential_entities
                if entity['entity'] in entity_names
            ]
            
            if len(mentioned_influential) == 0 and len(influential_entities) > 0:
                validation_issues.append({
                    'type': 'mentions_non_influential_entities',
                    'severity': 'low',
                    'message': 'Content mentions entities with low graph centrality'
                })
            
            return {
                'graph_results': graph_results,
                'graph_insights': graph_insights,
                'validation_issues': validation_issues,
                'entities_analyzed': entity_names,
                'graph_connectivity_score': neighborhood.get('total_nodes_explored', 0) / max(len(key_entities), 1),
                'passes_validation': len(validation_issues) == 0
            }
            
        except Exception as e:
            logger.error(f"Graph context analysis failed: {str(e)}")
            return {
                'graph_insights': [],
                'validation_issues': [{
                    'type': 'graph_analysis_error',
                    'severity': 'medium',
                    'message': str(e)
                }],
                'passes_validation': False
            }
    
    def _calculate_overall_confidence(self, result: ValidationResult, config: ValidationConfig) -> float:
        """Calculate overall confidence score from component results."""
        confidence_scores = []
        weights = []
        
        # Entity extraction confidence
        if result.entity_extraction_result:
            entity_confidence = result.entity_extraction_result.get('confidence_stats', {}).get('mean', 0)
            confidence_scores.append(entity_confidence)
            weights.append(0.25)
        
        # Relationship extraction confidence
        if result.relationship_extraction_result:
            rel_confidence = result.relationship_extraction_result.get('confidence_stats', {}).get('mean', 0)
            confidence_scores.append(rel_confidence)
            weights.append(0.25)
        
        # Hallucination detection confidence (inverse of hallucination rate)
        if result.hallucination_detection_result:
            hall_confidence = 1.0 - result.hallucination_detection_result.get('hallucination_rate', 1.0)
            confidence_scores.append(hall_confidence)
            weights.append(0.4)
        
        # Graph analysis confidence
        if result.graph_analysis_result:
            graph_confidence = result.graph_analysis_result.get('graph_connectivity_score', 0)
            confidence_scores.append(graph_confidence)
            weights.append(0.1)
        
        if not confidence_scores:
            return 0.0
        
        # Calculate weighted average
        total_weight = sum(weights[:len(confidence_scores)])
        weighted_scores = [score * weight for score, weight in zip(confidence_scores, weights)]
        
        return sum(weighted_scores) / total_weight if total_weight > 0 else 0.0
    
    def _determine_validation_status(self, result: ValidationResult, config: ValidationConfig) -> ValidationStatus:
        """Determine validation status based on confidence and issues."""
        # Check for system errors first
        all_issues = result.issues_found
        for component_result in [result.entity_extraction_result, result.relationship_extraction_result, 
                               result.hallucination_detection_result, result.graph_analysis_result]:
            if component_result:
                all_issues.extend(component_result.get('validation_issues', []))
        
        # Check for high severity issues
        high_severity_issues = [issue for issue in all_issues if issue.get('severity') == 'high']
        if high_severity_issues:
            return ValidationStatus.FAILED
        
        # Check overall confidence threshold
        if result.overall_confidence < config.overall_confidence_threshold:
            return ValidationStatus.WARNING if result.overall_confidence > 0.6 else ValidationStatus.FAILED
        
        # Check hallucination threshold specifically
        if result.hallucination_detection_result:
            hallucination_rate = result.hallucination_detection_result.get('hallucination_rate', 1.0)
            if hallucination_rate > config.hallucination_threshold:
                return ValidationStatus.FAILED
        
        # Check for medium severity issues requiring review
        medium_severity_issues = [issue for issue in all_issues if issue.get('severity') == 'medium']
        if len(medium_severity_issues) > 2:  # More than 2 medium issues
            return ValidationStatus.REQUIRES_REVIEW
        
        # Check if content passes all component validations
        component_validations = []
        for component_result in [result.entity_extraction_result, result.relationship_extraction_result,
                               result.hallucination_detection_result, result.graph_analysis_result]:
            if component_result:
                component_validations.append(component_result.get('passes_validation', True))
        
        if all(component_validations) and len(medium_severity_issues) == 0:
            return ValidationStatus.PASSED
        else:
            return ValidationStatus.WARNING
    
    def _analyze_issues(self, result: ValidationResult, config: ValidationConfig) -> List[Dict[str, Any]]:
        """Analyze and categorize all validation issues."""
        all_issues = []
        
        # Collect issues from all components
        for component_name, component_result in [
            ('entity_extraction', result.entity_extraction_result),
            ('relationship_extraction', result.relationship_extraction_result),
            ('hallucination_detection', result.hallucination_detection_result),
            ('graph_analysis', result.graph_analysis_result)
        ]:
            if component_result:
                component_issues = component_result.get('validation_issues', [])
                for issue in component_issues:
                    issue['component'] = component_name
                all_issues.extend(component_issues)
        
        # Add overall confidence issue if applicable
        if result.overall_confidence < config.overall_confidence_threshold:
            all_issues.append({
                'type': 'low_overall_confidence',
                'severity': 'high' if result.overall_confidence < 0.6 else 'medium',
                'confidence': result.overall_confidence,
                'threshold': config.overall_confidence_threshold,
                'component': 'overall'
            })
        
        # Sort by severity
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        all_issues.sort(key=lambda x: severity_order.get(x.get('severity', 'low'), 2))
        
        return all_issues
    
    def _generate_recommendations(self, result: ValidationResult, config: ValidationConfig) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Entity-based recommendations
        if result.entity_extraction_result:
            entity_issues = result.entity_extraction_result.get('validation_issues', [])
            for issue in entity_issues:
                if issue['type'] == 'low_confidence_entities':
                    recommendations.append(
                        f"Review and clarify entities with low confidence scores. "
                        f"Consider providing more context or using more specific names."
                    )
                elif issue['type'] == 'missing_required_entity_types':
                    recommendations.append(
                        f"Include mentions of required entity types: {', '.join(issue['missing_types'])}"
                    )
        
        # Relationship-based recommendations
        if result.relationship_extraction_result:
            rel_issues = result.relationship_extraction_result.get('validation_issues', [])
            for issue in rel_issues:
                if issue['type'] == 'high_isolated_entities':
                    recommendations.append(
                        "Establish clearer relationships between entities. "
                        "Consider adding connecting information or removing unrelated entities."
                    )
                elif issue['type'] == 'low_confidence_relationships':
                    recommendations.append(
                        "Strengthen relationship descriptions with more explicit connecting language."
                    )
        
        # Hallucination-based recommendations
        if result.hallucination_detection_result:
            hall_issues = result.hallucination_detection_result.get('validation_issues', [])
            for issue in hall_issues:
                if issue['type'] == 'hallucination_threshold_exceeded':
                    recommendations.append(
                        f"Reduce hallucination rate from {issue['hallucination_rate']:.1%} to below "
                        f"{issue['threshold']:.1%} by verifying facts against authoritative sources."
                    )
                elif issue['type'] == 'insufficient_evidence_sources':
                    recommendations.append(
                        f"Add {issue['required_sources'] - issue['found_sources']} more credible sources "
                        f"to support the claims made in the content."
                    )
        
        # Overall recommendations
        if result.overall_confidence < config.overall_confidence_threshold:
            recommendations.append(
                f"Improve overall content confidence from {result.overall_confidence:.1%} to above "
                f"{config.overall_confidence_threshold:.1%} through fact verification and source citation."
            )
        
        return list(set(recommendations))  # Remove duplicates
    
    def _generate_corrections(self, result: ValidationResult, config: ValidationConfig) -> List[Dict[str, Any]]:
        """Generate specific corrections for validation issues."""
        corrections = []
        
        # Extract corrections from hallucination detection
        if result.hallucination_detection_result:
            hall_corrections = result.hallucination_detection_result.get('corrections', [])
            for correction in hall_corrections:
                corrections.append({
                    'type': 'factual_correction',
                    'component': 'hallucination_detection',
                    'correction': correction
                })
        
        # Extract corrections from other components
        for component_name, component_result in [
            ('entity_extraction', result.entity_extraction_result),
            ('relationship_extraction', result.relationship_extraction_result),
            ('graph_analysis', result.graph_analysis_result)
        ]:
            if component_result and 'corrections' in component_result:
                component_corrections = component_result['corrections']
                for correction in component_corrections:
                    corrections.append({
                        'type': 'content_correction',
                        'component': component_name,
                        'correction': correction
                    })
        
        return corrections
    
    def _extract_graph_insights(self, graph_results: Dict[str, Any], entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract insights from graph analysis results."""
        insights = []
        
        # Neighborhood analysis insights
        neighborhood = graph_results.get('neighborhood_analysis', {})
        if neighborhood:
            insights.append({
                'type': 'connectivity',
                'message': f"Entity neighborhood contains {neighborhood.get('total_nodes_explored', 0)} connected entities",
                'detail': neighborhood.get('insights', {})
            })
        
        # Connection insights
        connections = graph_results.get('entity_connections', {})
        if connections:
            paths_found = connections.get('total_paths_found', 0)
            if paths_found > 0:
                avg_path_length = connections.get('avg_path_length', 0)
                insights.append({
                    'type': 'relationships',
                    'message': f"Found {paths_found} connection paths with average length {avg_path_length:.1f}",
                    'detail': connections.get('path_analysis', {})
                })
        
        # Centrality insights
        centrality = graph_results.get('centrality_analysis', {})
        if centrality:
            influential = centrality.get('influential_entities', [])
            if influential:
                insights.append({
                    'type': 'influence',
                    'message': f"Content mentions {len(influential)} highly influential entities",
                    'detail': {'top_influential': influential[:3]}
                })
        
        return insights
    
    def _update_validation_stats(
        self,
        validation_level: ValidationLevel,
        result: ValidationResult,
        component_results: Dict[str, float]
    ) -> None:
        """Update validation pipeline statistics."""
        self.validation_stats['total_validations'] += 1
        
        # Update level counts
        level_key = validation_level.value
        self.validation_stats['validation_level_counts'][level_key] = (
            self.validation_stats['validation_level_counts'].get(level_key, 0) + 1
        )
        
        # Update status distribution
        status_key = result.status.value
        self.validation_stats['status_distribution'][status_key] += 1
        
        # Update average processing time
        total_time = (
            self.validation_stats['avg_processing_time_ms'] * 
            (self.validation_stats['total_validations'] - 1) + 
            result.processing_time_ms
        )
        self.validation_stats['avg_processing_time_ms'] = (
            total_time / self.validation_stats['total_validations']
        )
        
        # Update confidence scores by level
        if level_key not in self.validation_stats['avg_confidence_scores']:
            self.validation_stats['avg_confidence_scores'][level_key] = []
        
        self.validation_stats['avg_confidence_scores'][level_key].append(result.overall_confidence)
        
        # Keep only last 100 confidence scores per level for memory efficiency
        if len(self.validation_stats['avg_confidence_scores'][level_key]) > 100:
            self.validation_stats['avg_confidence_scores'][level_key] = (
                self.validation_stats['avg_confidence_scores'][level_key][-100:]
            )
        
        # Update component performance
        for component, time_ms in component_results.items():
            if component not in self.validation_stats['component_performance']:
                self.validation_stats['component_performance'][component] = []
            
            self.validation_stats['component_performance'][component].append(time_ms)
            
            # Keep only last 50 measurements per component
            if len(self.validation_stats['component_performance'][component]) > 50:
                self.validation_stats['component_performance'][component] = (
                    self.validation_stats['component_performance'][component][-50:]
                )
    
    async def get_validation_statistics(self) -> Dict[str, Any]:
        """Get current validation pipeline statistics."""
        stats = dict(self.validation_stats)
        
        # Calculate average confidence scores by level
        for level, confidences in stats['avg_confidence_scores'].items():
            if confidences:
                stats['avg_confidence_scores'][level] = {
                    'mean': round(np.mean(confidences), 3),
                    'std': round(np.std(confidences), 3),
                    'min': round(min(confidences), 3),
                    'max': round(max(confidences), 3),
                    'count': len(confidences)
                }
        
        # Calculate component performance averages
        for component, times in stats['component_performance'].items():
            if times:
                stats['component_performance'][component] = {
                    'avg_time_ms': round(np.mean(times), 2),
                    'min_time_ms': round(min(times), 2),
                    'max_time_ms': round(max(times), 2),
                    'std_time_ms': round(np.std(times), 2)
                }
        
        # Add success rate
        total = stats['total_validations']
        if total > 0:
            passed = stats['status_distribution']['passed']
            stats['success_rate'] = round(passed / total * 100, 1)
        else:
            stats['success_rate'] = 0
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Check validation pipeline health."""
        try:
            if not self.is_initialized:
                return {
                    'status': 'unhealthy',
                    'error': 'Validation pipeline not initialized'
                }
            
            # Test basic validation
            start_time = datetime.now()
            
            test_content = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
            test_result = await self.validate_content(
                test_content,
                ValidationLevel.BASIC
            )
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Check component health
            component_health = await asyncio.gather(
                self.entity_extractor.health_check(),
                self.relationship_extractor.health_check(),
                self.hallucination_detector.health_check(),
                self.neo4j_optimizer.health_check(),
                self.graph_traversal.health_check(),
                return_exceptions=True
            )
            
            component_status = {}
            for i, (component_name, health) in enumerate([
                ('entity_extractor', component_health[0]),
                ('relationship_extractor', component_health[1]),
                ('hallucination_detector', component_health[2]),
                ('neo4j_optimizer', component_health[3]),
                ('graph_traversal', component_health[4])
            ]):
                if isinstance(health, Exception):
                    component_status[component_name] = 'error'
                else:
                    component_status[component_name] = health.get('status', 'unknown')
            
            all_healthy = all(status in ['healthy', 'degraded'] for status in component_status.values())
            
            return {
                'status': 'healthy' if all_healthy and response_time < 5000 else 'degraded',
                'response_time_ms': round(response_time, 2),
                'test_validation_status': test_result.status.value,
                'test_confidence': test_result.overall_confidence,
                'component_status': component_status,
                'total_validations_performed': self.validation_stats['total_validations'],
                'avg_processing_time_ms': round(self.validation_stats['avg_processing_time_ms'], 2),
                'available_validation_levels': [level.value for level in ValidationLevel]
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def record_human_approval(
        self,
        validation_context: Dict[str, Any],
        confidence_boost: float = 0.2
    ) -> None:
        """
        Record human approval for content and boost confidence in GraphRAG system.
        
        Args:
            validation_context: Context from human validation system
            confidence_boost: Amount to boost confidence scores (0.0-0.5)
        """
        try:
            # Extract relevant information
            validation_id = validation_context.get('validation_id')
            conversation_id = validation_context.get('conversation_id')
            user_response = validation_context.get('user_response', {})
            
            # Create knowledge graph update
            if hasattr(self, 'neo4j_optimizer') and self.neo4j_optimizer:
                # Record human validation in Neo4j
                query = """
                MERGE (hv:HumanValidation {id: $validation_id})
                SET hv.conversation_id = $conversation_id,
                    hv.validation_type = $validation_type,
                    hv.approved = true,
                    hv.confidence_boost = $confidence_boost,
                    hv.timestamp = datetime(),
                    hv.user_feedback = $user_feedback
                
                // Link to conversation if it exists
                WITH hv
                OPTIONAL MATCH (c:Conversation {id: $conversation_id})
                FOREACH (conversation IN CASE WHEN c IS NOT NULL THEN [c] ELSE [] END |
                    CREATE (conversation)-[:HAS_VALIDATION]->(hv)
                )
                
                // Boost confidence of related entities and relationships
                WITH hv
                MATCH (e:Entity), (r:Relationship)
                WHERE e.conversation_id = $conversation_id OR r.conversation_id = $conversation_id
                SET e.confidence = CASE 
                    WHEN e.confidence IS NOT NULL 
                    THEN LEAST(1.0, e.confidence + $confidence_boost)
                    ELSE e.confidence 
                    END,
                    r.confidence = CASE 
                    WHEN r.confidence IS NOT NULL 
                    THEN LEAST(1.0, r.confidence + $confidence_boost)
                    ELSE r.confidence 
                    END
                """
                
                await self.neo4j_optimizer.execute_query(query, {
                    'validation_id': validation_id,
                    'conversation_id': conversation_id,
                    'validation_type': validation_context.get('validation_type', 'approval'),
                    'confidence_boost': min(confidence_boost, 0.5),  # Cap at 0.5
                    'user_feedback': str(validation_context.get('feedback', ''))
                })
            
            logger.info(
                "Human approval recorded in GraphRAG",
                validation_id=validation_id,
                conversation_id=conversation_id,
                confidence_boost=confidence_boost
            )
            
        except Exception as e:
            logger.error(f"Failed to record human approval: {str(e)}")
    
    async def record_human_rejection(
        self,
        validation_context: Dict[str, Any],
        confidence_penalty: float = 0.3
    ) -> None:
        """
        Record human rejection for content and reduce confidence in GraphRAG system.
        
        Args:
            validation_context: Context from human validation system
            confidence_penalty: Amount to reduce confidence scores (0.0-0.5)
        """
        try:
            # Extract relevant information
            validation_id = validation_context.get('validation_id')
            conversation_id = validation_context.get('conversation_id')
            user_response = validation_context.get('user_response', {})
            
            # Create knowledge graph update
            if hasattr(self, 'neo4j_optimizer') and self.neo4j_optimizer:
                # Record human validation in Neo4j
                query = """
                MERGE (hv:HumanValidation {id: $validation_id})
                SET hv.conversation_id = $conversation_id,
                    hv.validation_type = $validation_type,
                    hv.approved = false,
                    hv.confidence_penalty = $confidence_penalty,
                    hv.timestamp = datetime(),
                    hv.user_feedback = $user_feedback,
                    hv.rejection_reason = $rejection_reason
                
                // Link to conversation if it exists
                WITH hv
                OPTIONAL MATCH (c:Conversation {id: $conversation_id})
                FOREACH (conversation IN CASE WHEN c IS NOT NULL THEN [c] ELSE [] END |
                    CREATE (conversation)-[:HAS_VALIDATION]->(hv)
                )
                
                // Apply confidence penalty to related entities and relationships
                WITH hv
                MATCH (e:Entity), (r:Relationship)
                WHERE e.conversation_id = $conversation_id OR r.conversation_id = $conversation_id
                SET e.confidence = CASE 
                    WHEN e.confidence IS NOT NULL 
                    THEN GREATEST(0.0, e.confidence - $confidence_penalty)
                    ELSE e.confidence 
                    END,
                    r.confidence = CASE 
                    WHEN r.confidence IS NOT NULL 
                    THEN GREATEST(0.0, r.confidence - $confidence_penalty)
                    ELSE r.confidence 
                    END,
                    e.requires_review = true,
                    r.requires_review = true
                """
                
                await self.neo4j_optimizer.execute_query(query, {
                    'validation_id': validation_id,
                    'conversation_id': conversation_id,
                    'validation_type': validation_context.get('validation_type', 'approval'),
                    'confidence_penalty': min(confidence_penalty, 0.5),  # Cap at 0.5
                    'user_feedback': str(validation_context.get('feedback', '')),
                    'rejection_reason': str(user_response.get('reason', 'Content rejected by human reviewer'))
                })
            
            logger.info(
                "Human rejection recorded in GraphRAG",
                validation_id=validation_id,
                conversation_id=conversation_id,
                confidence_penalty=confidence_penalty
            )
            
        except Exception as e:
            logger.error(f"Failed to record human rejection: {str(e)}")
    
    async def get_human_validation_history(
        self,
        conversation_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get history of human validations for analysis and learning.
        
        Args:
            conversation_id: Optional conversation filter
            limit: Maximum number of records to return
            
        Returns:
            List of human validation records
        """
        try:
            if not hasattr(self, 'neo4j_optimizer') or not self.neo4j_optimizer:
                return []
            
            # Query human validations from Neo4j
            if conversation_id:
                query = """
                MATCH (hv:HumanValidation {conversation_id: $conversation_id})
                OPTIONAL MATCH (c:Conversation)-[:HAS_VALIDATION]->(hv)
                RETURN hv, c
                ORDER BY hv.timestamp DESC
                LIMIT $limit
                """
                params = {'conversation_id': conversation_id, 'limit': limit}
            else:
                query = """
                MATCH (hv:HumanValidation)
                OPTIONAL MATCH (c:Conversation)-[:HAS_VALIDATION]->(hv)
                RETURN hv, c
                ORDER BY hv.timestamp DESC
                LIMIT $limit
                """
                params = {'limit': limit}
            
            result = await self.neo4j_optimizer.execute_query(query, params)
            
            validations = []
            for record in result:
                validation_data = record['hv']
                conversation_data = record.get('c')
                
                validations.append({
                    'validation_id': validation_data.get('id'),
                    'conversation_id': validation_data.get('conversation_id'),
                    'validation_type': validation_data.get('validation_type'),
                    'approved': validation_data.get('approved', False),
                    'confidence_impact': validation_data.get('confidence_boost', 0) if validation_data.get('approved') else -validation_data.get('confidence_penalty', 0),
                    'user_feedback': validation_data.get('user_feedback'),
                    'rejection_reason': validation_data.get('rejection_reason'),
                    'timestamp': validation_data.get('timestamp'),
                    'conversation_title': conversation_data.get('title') if conversation_data else None
                })
            
            return validations
            
        except Exception as e:
            logger.error(f"Failed to get human validation history: {str(e)}")
            return []
    
    async def analyze_human_feedback_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in human feedback to improve the validation pipeline.
        
        Returns:
            Analysis of human feedback patterns and suggestions for improvement
        """
        try:
            if not hasattr(self, 'neo4j_optimizer') or not self.neo4j_optimizer:
                return {'error': 'GraphRAG not available for analysis'}
            
            # Query validation statistics
            stats_query = """
            MATCH (hv:HumanValidation)
            WITH COUNT(hv) as total_validations,
                 COUNT(CASE WHEN hv.approved = true THEN 1 END) as approved_count,
                 COUNT(CASE WHEN hv.approved = false THEN 1 END) as rejected_count,
                 COLLECT(hv.validation_type) as validation_types,
                 COLLECT(CASE WHEN hv.approved = false THEN hv.rejection_reason END) as rejection_reasons
            
            RETURN total_validations, approved_count, rejected_count, validation_types, rejection_reasons
            """
            
            stats_result = await self.neo4j_optimizer.execute_query(stats_query)
            
            if not stats_result:
                return {'message': 'No human validation data available'}
            
            stats = stats_result[0]
            
            # Analyze validation type distribution
            validation_types = stats['validation_types']
            type_distribution = {}
            for vtype in validation_types:
                type_distribution[vtype] = type_distribution.get(vtype, 0) + 1
            
            # Analyze rejection reasons
            rejection_reasons = [r for r in stats['rejection_reasons'] if r]
            reason_patterns = {}
            for reason in rejection_reasons:
                # Simple keyword extraction for common rejection patterns
                if 'inaccurate' in reason.lower() or 'incorrect' in reason.lower():
                    reason_patterns['accuracy'] = reason_patterns.get('accuracy', 0) + 1
                elif 'incomplete' in reason.lower() or 'missing' in reason.lower():
                    reason_patterns['completeness'] = reason_patterns.get('completeness', 0) + 1
                elif 'unclear' in reason.lower() or 'confusing' in reason.lower():
                    reason_patterns['clarity'] = reason_patterns.get('clarity', 0) + 1
                else:
                    reason_patterns['other'] = reason_patterns.get('other', 0) + 1
            
            # Calculate metrics
            total_validations = stats['total_validations']
            approval_rate = (stats['approved_count'] / total_validations * 100) if total_validations > 0 else 0
            
            # Generate improvement suggestions
            suggestions = []
            
            if approval_rate < 70:
                suggestions.append("Low approval rate indicates need for better content generation quality")
            
            if reason_patterns.get('accuracy', 0) > reason_patterns.get('other', 0):
                suggestions.append("Focus on improving factual accuracy through better source verification")
            
            if reason_patterns.get('completeness', 0) > reason_patterns.get('other', 0):
                suggestions.append("Enhance content completeness by including more comprehensive information")
            
            if reason_patterns.get('clarity', 0) > reason_patterns.get('other', 0):
                suggestions.append("Improve content clarity and reduce ambiguity")
            
            # Confidence impact analysis
            confidence_query = """
            MATCH (hv:HumanValidation)
            WHERE hv.approved = true AND hv.confidence_boost IS NOT NULL
            WITH AVG(hv.confidence_boost) as avg_boost
            
            MATCH (hv:HumanValidation)
            WHERE hv.approved = false AND hv.confidence_penalty IS NOT NULL
            WITH avg_boost, AVG(hv.confidence_penalty) as avg_penalty
            
            RETURN avg_boost, avg_penalty
            """
            
            confidence_result = await self.neo4j_optimizer.execute_query(confidence_query)
            confidence_impact = confidence_result[0] if confidence_result else {}
            
            return {
                'total_validations': total_validations,
                'approval_rate': round(approval_rate, 1),
                'validation_type_distribution': type_distribution,
                'rejection_reason_patterns': reason_patterns,
                'confidence_impact': {
                    'average_boost': round(confidence_impact.get('avg_boost', 0), 3),
                    'average_penalty': round(confidence_impact.get('avg_penalty', 0), 3)
                },
                'improvement_suggestions': suggestions,
                'pipeline_health': 'good' if approval_rate >= 80 else 'needs_improvement' if approval_rate >= 60 else 'poor'
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze human feedback patterns: {str(e)}")
            return {'error': f'Analysis failed: {str(e)}'}

    async def close(self) -> None:
        """Close validation pipeline and cleanup resources."""
        await asyncio.gather(
            self.entity_extractor.close() if hasattr(self.entity_extractor, 'close') else asyncio.sleep(0),
            self.relationship_extractor.close() if hasattr(self.relationship_extractor, 'close') else asyncio.sleep(0),
            self.hallucination_detector.close() if hasattr(self.hallucination_detector, 'close') else asyncio.sleep(0),
            self.neo4j_optimizer.close(),
            self.graph_traversal.close(),
            return_exceptions=True
        )
        
        self.is_initialized = False
        logger.info("Validation pipeline closed")
    
    def to_dict(self, result: ValidationResult) -> Dict[str, Any]:
        """Convert ValidationResult to dictionary for serialization."""
        result_dict = asdict(result)
        result_dict['status'] = result.status.value
        if result.config_used:
            result_dict['config_used']['level'] = result.config_used.level.value
        return result_dict


# Convenience functions for common validation scenarios

async def validate_prd_content(
    pipeline: ValidationPipeline,
    content: str,
    context: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    """Validate Product Requirements Document content using strict validation."""
    return await pipeline.validate_content(
        content,
        ValidationLevel.STRICT,
        context=context
    )


async def validate_draft_content(
    pipeline: ValidationPipeline,
    content: str,
    context: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    """Validate draft content using basic validation for quick feedback."""
    return await pipeline.validate_content(
        content,
        ValidationLevel.BASIC,
        context=context
    )


async def validate_technical_content(
    pipeline: ValidationPipeline,
    content: str,
    focus_entity_types: List[str] = ['TECHNOLOGY', 'PRODUCT', 'ORG'],
    context: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    """Validate technical content with focus on specific entity types."""
    custom_config = ValidationConfig(
        level=ValidationLevel.STANDARD,
        focus_entity_types=focus_entity_types,
        context_domain='technology',
        include_graph_insights=True
    )
    
    return await pipeline.validate_content(
        content,
        custom_config=custom_config,
        context=context
    )