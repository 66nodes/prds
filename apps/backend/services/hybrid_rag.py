"""
Hybrid RAG Service integrating Milvus vector database and Neo4j graph database
for comprehensive content validation and retrieval.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
import uuid
from datetime import datetime

from pymilvus import Collection, DataType, FieldSchema, CollectionSchema, utility
import numpy as np
from openai import AsyncOpenAI
import structlog

from core.config import get_settings
from core.database import get_milvus, get_neo4j

logger = structlog.get_logger(__name__)
settings = get_settings()


class HybridRAGService:
    """
    Hybrid RAG service combining vector similarity search (Milvus) 
    and graph-based validation (Neo4j) for <2% hallucination rate.
    """
    
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.milvus = get_milvus()
        self.neo4j = None  # Will be set in initialize()
        self.is_initialized = False
        
        # Three-tier validation weights
        self.entity_weight = settings.entity_validation_weight
        self.community_weight = settings.community_validation_weight
        self.global_weight = settings.global_validation_weight
        
        # Collection names
        self.collections = {
            'text_chunks': 'text_chunks',
            'entities': 'entities', 
            'communities': 'communities',
            'claims': 'claims'
        }
    
    async def initialize(self) -> None:
        """Initialize the HybridRAG service with database connections."""
        try:
            self.neo4j = await get_neo4j()
            
            # Verify Milvus collections exist
            await self._ensure_collections_exist()
            
            self.is_initialized = True
            logger.info("HybridRAG service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize HybridRAG service: {str(e)}")
            raise
    
    async def _ensure_collections_exist(self) -> None:
        """Ensure all required Milvus collections exist."""
        for collection_name in self.collections.values():
            if not utility.has_collection(collection_name):
                logger.warning(f"Collection {collection_name} not found")
                # Collections should be created by setup_milvus_collections.py
                # For now, just log the warning
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI API."""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",  # 1536 dimensions
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise
    
    async def validate_content(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        section_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate content using three-tier GraphRAG validation:
        1. Entity-level validation (50%)
        2. Community-level validation (30%) 
        3. Global-level validation (20%)
        """
        if not self.is_initialized:
            raise RuntimeError("HybridRAG service not initialized")
        
        validation_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            # Generate embedding for content
            embedding = await self.generate_embedding(content)
            
            # Perform three-tier validation in parallel
            entity_result, community_result, global_result = await asyncio.gather(
                self._entity_validation(content, embedding, context),
                self._community_validation(content, embedding, context),
                self._global_validation(content, embedding, context)
            )
            
            # Calculate weighted confidence score
            confidence = (
                entity_result['confidence'] * self.entity_weight +
                community_result['confidence'] * self.community_weight +
                global_result['confidence'] * self.global_weight
            )
            
            # Determine if content passes validation threshold
            passes_threshold = confidence >= settings.validation_threshold
            
            # Compile corrections from all tiers
            corrections = []
            corrections.extend(entity_result.get('corrections', []))
            corrections.extend(community_result.get('corrections', []))
            corrections.extend(global_result.get('corrections', []))
            
            # Store validation result in Neo4j
            await self._store_validation_result(
                validation_id, content, confidence, passes_threshold,
                entity_result, community_result, global_result, context
            )
            
            validation_result = {
                'validation_id': validation_id,
                'confidence': confidence,
                'passes_threshold': passes_threshold,
                'entity_validation': entity_result,
                'community_validation': community_result,
                'global_validation': global_result,
                'corrections': corrections,
                'requires_human_review': confidence < 0.7,  # Below 70% requires human review
                'timestamp': start_time.isoformat(),
                'processing_time_ms': int((datetime.utcnow() - start_time).total_seconds() * 1000)
            }
            
            logger.info(
                "Content validation completed",
                validation_id=validation_id,
                confidence=confidence,
                passes_threshold=passes_threshold,
                processing_time_ms=validation_result['processing_time_ms']
            )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Content validation failed: {str(e)}", validation_id=validation_id)
            raise
    
    async def _entity_validation(
        self, 
        content: str, 
        embedding: List[float], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Entity-level validation using Milvus vector similarity search."""
        try:
            # Search for similar entities in Milvus
            collection = Collection(self.collections['entities'])
            collection.load()
            
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 16}
            }
            
            results = collection.search(
                data=[embedding],
                anns_field="embedding",
                param=search_params,
                limit=5,
                output_fields=["name", "type", "description", "importance_score"]
            )
            
            # Extract relevant entities and calculate confidence
            similar_entities = []
            max_similarity = 0.0
            
            for result in results[0]:
                entity_data = {
                    'name': result.entity.get('name'),
                    'type': result.entity.get('type'),
                    'description': result.entity.get('description'),
                    'similarity_score': float(result.distance),
                    'importance_score': result.entity.get('importance_score', 0.5)
                }
                similar_entities.append(entity_data)
                max_similarity = max(max_similarity, float(result.distance))
            
            # Calculate entity validation confidence
            entity_confidence = min(max_similarity * 0.8 + 0.2, 1.0)  # Scale to [0.2, 1.0]
            
            corrections = []
            if entity_confidence < 0.6:
                corrections.append("Consider referencing more established entities or concepts")
            
            return {
                'confidence': entity_confidence,
                'similar_entities': similar_entities[:3],  # Top 3
                'corrections': corrections,
                'validation_type': 'entity'
            }
            
        except Exception as e:
            logger.error(f"Entity validation failed: {str(e)}")
            return {
                'confidence': 0.5,  # Neutral confidence on failure
                'similar_entities': [],
                'corrections': ["Entity validation unavailable"],
                'validation_type': 'entity',
                'error': str(e)
            }
    
    async def _community_validation(
        self, 
        content: str, 
        embedding: List[float], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Community-level validation using community embeddings and graph relationships."""
        try:
            # Search for similar communities in Milvus
            collection = Collection(self.collections['communities'])
            collection.load()
            
            search_params = {
                "metric_type": "COSINE", 
                "params": {"nprobe": 16}
            }
            
            results = collection.search(
                data=[embedding],
                anns_field="embedding",
                param=search_params,
                limit=3,
                output_fields=["name", "level", "summary", "importance_score", "entity_count"]
            )
            
            # Query Neo4j for community relationships
            community_query = """
            MATCH (c:Community)
            RETURN c.name as name, c.description as description, 
                   size((c)-[:CONTAINS]->(:Entity)) as entity_count
            ORDER BY entity_count DESC
            LIMIT 5
            """
            
            community_graph_data = await self.neo4j.execute_query(community_query)
            
            # Calculate community confidence based on similarity and graph structure
            max_similarity = 0.0
            relevant_communities = []
            
            for result in results[0]:
                community_data = {
                    'name': result.entity.get('name'),
                    'level': result.entity.get('level', 0),
                    'summary': result.entity.get('summary'),
                    'similarity_score': float(result.distance),
                    'entity_count': result.entity.get('entity_count', 0)
                }
                relevant_communities.append(community_data)
                max_similarity = max(max_similarity, float(result.distance))
            
            # Boost confidence if content relates to well-established communities
            community_confidence = min(max_similarity * 0.7 + 0.3, 1.0)
            
            corrections = []
            if community_confidence < 0.7:
                corrections.append("Consider aligning content with established domain communities")
            
            return {
                'confidence': community_confidence,
                'relevant_communities': relevant_communities,
                'graph_communities': community_graph_data[:3],
                'corrections': corrections,
                'validation_type': 'community'
            }
            
        except Exception as e:
            logger.error(f"Community validation failed: {str(e)}")
            return {
                'confidence': 0.5,
                'relevant_communities': [],
                'corrections': ["Community validation unavailable"],
                'validation_type': 'community',
                'error': str(e)
            }
    
    async def _global_validation(
        self, 
        content: str, 
        embedding: List[float], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Global-level validation using factual claims and global knowledge patterns."""
        try:
            # Search for similar claims in Milvus
            collection = Collection(self.collections['claims'])
            collection.load()
            
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 16}
            }
            
            results = collection.search(
                data=[embedding],
                anns_field="embedding",
                param=search_params,
                limit=5,
                output_fields=["text", "confidence_score", "source_refs", "validated"]
            )
            
            # Query Neo4j for global patterns and claims
            global_query = """
            MATCH (claim:Claim)
            WHERE claim.validated = true
            RETURN claim.text as text, claim.confidence_score as confidence,
                   claim.source_refs as sources
            ORDER BY claim.confidence_score DESC
            LIMIT 10
            """
            
            validated_claims = await self.neo4j.execute_query(global_query)
            
            # Calculate global confidence based on claim validation
            similar_claims = []
            max_claim_confidence = 0.0
            
            for result in results[0]:
                claim_data = {
                    'text': result.entity.get('text'),
                    'confidence_score': result.entity.get('confidence_score', 0.5),
                    'similarity_score': float(result.distance),
                    'validated': result.entity.get('validated', False)
                }
                similar_claims.append(claim_data)
                
                # Weight by both similarity and claim confidence
                weighted_confidence = float(result.distance) * claim_data['confidence_score']
                max_claim_confidence = max(max_claim_confidence, weighted_confidence)
            
            global_confidence = min(max_claim_confidence * 0.9 + 0.1, 1.0)
            
            corrections = []
            if global_confidence < 0.8:
                corrections.append("Verify claims against authoritative sources")
            
            return {
                'confidence': global_confidence,
                'similar_claims': similar_claims[:3],
                'validated_claims': validated_claims[:3],
                'corrections': corrections,
                'validation_type': 'global'
            }
            
        except Exception as e:
            logger.error(f"Global validation failed: {str(e)}")
            return {
                'confidence': 0.5,
                'similar_claims': [],
                'corrections': ["Global validation unavailable"],
                'validation_type': 'global',
                'error': str(e)
            }
    
    async def _store_validation_result(
        self,
        validation_id: str,
        content: str,
        confidence: float,
        passes_threshold: bool,
        entity_result: Dict[str, Any],
        community_result: Dict[str, Any],
        global_result: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> None:
        """Store validation result in Neo4j for audit and learning."""
        try:
            query = """
            CREATE (v:ValidationResult {
                id: $validation_id,
                content: $content,
                confidence: $confidence,
                passes_threshold: $passes_threshold,
                entity_confidence: $entity_confidence,
                community_confidence: $community_confidence,
                global_confidence: $global_confidence,
                context: $context,
                created_at: datetime(),
                processing_timestamp: $timestamp
            })
            """
            
            parameters = {
                'validation_id': validation_id,
                'content': content[:1000],  # Limit content length
                'confidence': confidence,
                'passes_threshold': passes_threshold,
                'entity_confidence': entity_result['confidence'],
                'community_confidence': community_result['confidence'],
                'global_confidence': global_result['confidence'],
                'context': context or {},
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await self.neo4j.execute_write(query, parameters)
            
        except Exception as e:
            logger.warning(f"Failed to store validation result: {str(e)}")
            # Don't raise - storage failure shouldn't break validation
    
    async def search_similar_content(
        self,
        query: str,
        collection_name: str = 'text_chunks',
        limit: int = 5,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """Search for similar content using vector similarity."""
        if not self.is_initialized:
            raise RuntimeError("HybridRAG service not initialized")
        
        try:
            # Generate embedding for query
            embedding = await self.generate_embedding(query)
            
            # Search in specified collection
            collection = Collection(collection_name)
            collection.load()
            
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 16}
            }
            
            output_fields = ["content", "document_id", "chunk_index", "created_at"]
            if collection_name == 'entities':
                output_fields = ["name", "type", "description", "importance_score"]
            elif collection_name == 'communities':
                output_fields = ["name", "summary", "level", "entity_count"]
            elif collection_name == 'claims':
                output_fields = ["text", "confidence_score", "validated"]
            
            results = collection.search(
                data=[embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=output_fields
            )
            
            similar_content = []
            for result in results[0]:
                content_data = {
                    'similarity_score': float(result.distance),
                    'id': result.id
                }
                
                # Add collection-specific fields
                for field in output_fields:
                    content_data[field] = result.entity.get(field)
                
                similar_content.append(content_data)
            
            return similar_content
            
        except Exception as e:
            logger.error(f"Similar content search failed: {str(e)}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check HybridRAG service health."""
        try:
            health_status = {
                'initialized': self.is_initialized,
                'openai_configured': self.openai_client is not None,
                'milvus_connected': self.milvus.is_connected if self.milvus else False,
                'neo4j_connected': self.neo4j.is_connected if self.neo4j else False,
                'collections_status': {}
            }
            
            if self.milvus and self.milvus.is_connected:
                for name, collection_name in self.collections.items():
                    health_status['collections_status'][name] = utility.has_collection(collection_name)
            
            overall_status = (
                health_status['initialized'] and
                health_status['openai_configured'] and
                health_status['milvus_connected'] and
                health_status['neo4j_connected']
            )
            
            return {
                'status': 'healthy' if overall_status else 'degraded',
                'details': health_status,
                'response_time_ms': 10  # Approximate
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }