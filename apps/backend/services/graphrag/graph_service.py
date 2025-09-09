"""
GraphRAG Service for hallucination prevention and validation
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from uuid import uuid4

import structlog
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from core.config import get_settings
from core.database import get_neo4j
from ..cache_service import get_cache_service, CacheNamespace, cached

logger = structlog.get_logger(__name__)
settings = get_settings()


class ValidationResult:
    """Validation result from GraphRAG analysis."""
    
    def __init__(
        self,
        confidence: float,
        entity_validation: Dict[str, Any],
        community_validation: Dict[str, Any], 
        global_validation: Dict[str, Any],
        corrections: List[str] = None,
        requires_human_review: bool = False
    ):
        self.confidence = confidence
        self.entity_validation = entity_validation
        self.community_validation = community_validation
        self.global_validation = global_validation
        self.corrections = corrections or []
        self.requires_human_review = requires_human_review
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "confidence": self.confidence,
            "entity_validation": self.entity_validation,
            "community_validation": self.community_validation,
            "global_validation": self.global_validation,
            "corrections": self.corrections,
            "requires_human_review": self.requires_human_review,
            "timestamp": self.timestamp.isoformat(),
            "passes_threshold": self.confidence >= settings.validation_threshold
        }


class GraphRAGService:
    """Core GraphRAG service for validation and knowledge management."""
    
    def __init__(self):
        self.neo4j_conn = None
        self.graph_store = None
        self.vector_store = None
        self.chroma_client = None
        self.embedding_model = None
        self.is_initialized = False
        self._cache_service = get_cache_service()
    
    async def initialize(self) -> None:
        """Initialize GraphRAG service components."""
        try:
            # Get Neo4j connection
            self.neo4j_conn = await get_neo4j()
            
            # Initialize graph store
            self.graph_store = Neo4jGraphStore(
                uri=settings.neo4j_uri,
                username=settings.neo4j_user,
                password=settings.neo4j_password,
                database=settings.neo4j_database
            )
            
            # Initialize vector store with ChromaDB
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            chroma_collection = self.chroma_client.get_or_create_collection(
                name="strategic_planning",
                metadata={"hnsw:space": "cosine"}
            )
            
            self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            # Initialize embedding model
            if settings.openai_api_key:
                self.embedding_model = OpenAIEmbedding(
                    api_key=settings.openai_api_key,
                    model="text-embedding-3-large",
                    dimensions=1536
                )
            else:
                logger.warning("No OpenAI API key configured, using default embeddings")
            
            self.is_initialized = True
            logger.info("GraphRAG service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize GraphRAG service", error=str(e))
            raise
    
    async def validate_content(
        self,
        content: str,
        context: Dict[str, Any] = None
    ) -> ValidationResult:
        """
        Three-tier GraphRAG validation pipeline:
        1. Entity validation (50% weight)
        2. Community validation (30% weight)  
        3. Global validation (20% weight)
        """
        if not self.is_initialized:
            raise RuntimeError("GraphRAG service not initialized")
        
        if not context:
            context = {}
        
        try:
            # Run all validation layers in parallel
            entity_task = self._validate_entities(content, context)
            community_task = self._validate_communities(content, context)
            global_task = self._validate_global(content, context)
            
            entity_result, community_result, global_result = await asyncio.gather(
                entity_task, community_task, global_task,
                return_exceptions=True
            )
            
            # Handle any exceptions
            if isinstance(entity_result, Exception):
                logger.error("Entity validation failed", error=str(entity_result))
                entity_result = {"confidence": 0.0, "error": str(entity_result)}
            
            if isinstance(community_result, Exception):
                logger.error("Community validation failed", error=str(community_result))
                community_result = {"confidence": 0.0, "error": str(community_result)}
            
            if isinstance(global_result, Exception):
                logger.error("Global validation failed", error=str(global_result))
                global_result = {"confidence": 0.0, "error": str(global_result)}
            
            # Calculate weighted confidence score
            weighted_confidence = (
                entity_result["confidence"] * settings.entity_validation_weight +
                community_result["confidence"] * settings.community_validation_weight +
                global_result["confidence"] * settings.global_validation_weight
            )
            
            # Generate corrections if confidence is low
            corrections = []
            requires_review = False
            
            if weighted_confidence < settings.validation_threshold:
                corrections = await self._generate_corrections(
                    content, entity_result, community_result, global_result
                )
                requires_review = weighted_confidence < 0.7
            
            # Store validation result
            await self._store_validation_result(
                content, weighted_confidence, entity_result, community_result, global_result
            )
            
            return ValidationResult(
                confidence=weighted_confidence,
                entity_validation=entity_result,
                community_validation=community_result,
                global_validation=global_result,
                corrections=corrections,
                requires_human_review=requires_review
            )
            
        except Exception as e:
            logger.error("Content validation failed", error=str(e))
            # Return low-confidence result on failure
            return ValidationResult(
                confidence=0.0,
                entity_validation={"confidence": 0.0, "error": "Validation failed"},
                community_validation={"confidence": 0.0, "error": "Validation failed"},
                global_validation={"confidence": 0.0, "error": "Validation failed"},
                corrections=["Manual review required - validation system error"],
                requires_human_review=True
            )
    
    async def _validate_entities(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Entity-level validation against existing knowledge graph."""
        try:
            # Generate embedding for content
            if self.embedding_model:
                embedding = await self.embedding_model.aget_text_embedding(content)
            else:
                # Fallback to simple text matching
                embedding = None
            
            # Search for similar entities in graph
            if embedding:
                # Vector similarity search
                query = """
                CALL db.index.vector.queryNodes('req_embedding', $k, $embedding)
                YIELD node, score
                WHERE node:Requirement OR node:Objective
                RETURN node.description as description, score
                ORDER BY score DESC
                LIMIT 10
                """
                
                similar_entities = await self.neo4j_conn.execute_query(
                    query, {"k": 10, "embedding": embedding}
                )
            else:
                # Text-based search as fallback
                query = """
                CALL db.index.fulltext.queryNodes('req_search', $query)
                YIELD node, score
                WHERE score > 0.5
                RETURN node.description as description, score
                ORDER BY score DESC
                LIMIT 5
                """
                
                similar_entities = await self.neo4j_conn.execute_query(
                    query, {"query": content[:200]}  # Limit query length
                )
            
            # Calculate confidence based on similarity scores
            if similar_entities:
                avg_similarity = np.mean([entity["score"] for entity in similar_entities])
                confidence = min(avg_similarity * 1.2, 1.0)  # Boost similarity slightly
            else:
                confidence = 0.3  # Low but not zero for new content
            
            return {
                "confidence": confidence,
                "similar_entities": len(similar_entities),
                "top_matches": similar_entities[:3] if similar_entities else [],
                "method": "vector_search" if embedding else "text_search"
            }
            
        except Exception as e:
            logger.error("Entity validation failed", error=str(e))
            return {"confidence": 0.0, "error": str(e)}
    
    async def _validate_communities(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Community-level validation within requirement clusters."""
        try:
            # Find communities related to the content
            community_query = """
            MATCH (c:Community)
            WHERE any(keyword IN c.entities WHERE 
                toLower($content) CONTAINS toLower(keyword))
            RETURN c.name as name, c.description as description, 
                   c.entities as entities, c.id as id
            ORDER BY size(c.entities) DESC
            LIMIT 5
            """
            
            communities = await self.neo4j_conn.execute_query(
                community_query, {"content": content}
            )
            
            # Calculate community alignment score
            if communities:
                # Check alignment with community patterns
                alignment_scores = []
                
                for community in communities:
                    # Simple pattern matching for now
                    # In production, this would use more sophisticated analysis
                    keywords = community.get("entities", [])
                    matches = sum(1 for keyword in keywords if keyword.lower() in content.lower())
                    alignment_score = matches / len(keywords) if keywords else 0
                    alignment_scores.append(alignment_score)
                
                avg_alignment = np.mean(alignment_scores) if alignment_scores else 0.0
                confidence = min(avg_alignment * 1.5, 1.0)
                
            else:
                confidence = 0.5  # Neutral confidence for no community matches
            
            return {
                "confidence": confidence,
                "communities_found": len(communities),
                "communities": [
                    {"name": c["name"], "id": c["id"]} 
                    for c in communities[:3]
                ],
                "average_alignment": avg_alignment if communities else 0.0
            }
            
        except Exception as e:
            logger.error("Community validation failed", error=str(e))
            return {"confidence": 0.0, "error": str(e)}
    
    async def _validate_global(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Global validation for consistency and strategic alignment."""
        try:
            # Check against organizational objectives and constraints
            global_query = """
            MATCH (o:Objective)-[:PART_OF]->(p:Project)
            WHERE toLower(o.description) CONTAINS any(word IN split(toLower($content), ' ') 
                WHERE length(word) > 3)
            RETURN o.description as objective, p.name as project, 
                   o.priority as priority
            ORDER BY o.priority DESC
            LIMIT 10
            """
            
            related_objectives = await self.neo4j_conn.execute_query(
                global_query, {"content": content}
            )
            
            # Calculate strategic alignment
            if related_objectives:
                # Weight by priority (high=1.0, medium=0.8, low=0.6)
                priority_weights = {"high": 1.0, "medium": 0.8, "low": 0.6}
                
                weighted_alignments = []
                for obj in related_objectives:
                    priority = obj.get("priority", "medium")
                    weight = priority_weights.get(priority.lower(), 0.8)
                    weighted_alignments.append(weight)
                
                avg_alignment = np.mean(weighted_alignments) if weighted_alignments else 0.0
                confidence = min(avg_alignment, 1.0)
                
            else:
                confidence = 0.6  # Neutral-positive for new strategic directions
            
            # Check for contradictions or conflicts
            contradictions = await self._check_contradictions(content, context)
            if contradictions:
                confidence *= 0.7  # Reduce confidence for contradictions
            
            return {
                "confidence": confidence,
                "strategic_alignment": avg_alignment if related_objectives else 0.0,
                "related_objectives": len(related_objectives),
                "contradictions": len(contradictions),
                "objectives": [
                    {"objective": obj["objective"], "project": obj["project"]} 
                    for obj in related_objectives[:3]
                ]
            }
            
        except Exception as e:
            logger.error("Global validation failed", error=str(e))
            return {"confidence": 0.0, "error": str(e)}
    
    async def _check_contradictions(self, content: str, context: Dict[str, Any]) -> List[str]:
        """Check for contradictions with existing requirements."""
        try:
            # Look for conflicting requirements
            contradiction_query = """
            MATCH (r:Requirement)
            WHERE r.status = 'approved' AND 
                  any(word IN ['not', 'never', 'avoid', 'prevent'] WHERE 
                      toLower(r.description) CONTAINS word)
            RETURN r.description as description
            LIMIT 10
            """
            
            constraints = await self.neo4j_conn.execute_query(contradiction_query)
            
            contradictions = []
            for constraint in constraints:
                # Simple contradiction detection
                # In production, use more sophisticated NLP
                constraint_text = constraint["description"].lower()
                if any(word in content.lower() for word in ["must not", "should not", "cannot"]):
                    contradictions.append(constraint["description"])
            
            return contradictions
            
        except Exception as e:
            logger.warning("Contradiction check failed", error=str(e))
            return []
    
    async def _generate_corrections(
        self,
        content: str,
        entity_result: Dict[str, Any],
        community_result: Dict[str, Any], 
        global_result: Dict[str, Any]
    ) -> List[str]:
        """Generate corrections based on validation results."""
        corrections = []
        
        # Entity-level corrections
        if entity_result["confidence"] < 0.6:
            if entity_result.get("similar_entities", 0) > 0:
                corrections.append(
                    "Consider aligning with similar existing requirements for consistency"
                )
            else:
                corrections.append(
                    "Add more specific technical details and acceptance criteria"
                )
        
        # Community-level corrections
        if community_result["confidence"] < 0.6:
            communities = community_result.get("communities", [])
            if communities:
                corrections.append(
                    f"Ensure alignment with {communities[0]['name']} community standards"
                )
            else:
                corrections.append(
                    "Consider establishing clear domain context and patterns"
                )
        
        # Global-level corrections
        if global_result["confidence"] < 0.6:
            if global_result.get("contradictions", 0) > 0:
                corrections.append(
                    "Resolve conflicts with existing organizational constraints"
                )
            if global_result.get("strategic_alignment", 0) < 0.5:
                corrections.append(
                    "Strengthen alignment with strategic organizational objectives"
                )
        
        return corrections
    
    async def _store_validation_result(
        self,
        content: str,
        confidence: float,
        entity_result: Dict[str, Any],
        community_result: Dict[str, Any],
        global_result: Dict[str, Any]
    ) -> None:
        """Store validation results for learning and analysis."""
        try:
            result_id = str(uuid4())
            
            store_query = """
            CREATE (v:ValidationResult {
                id: $result_id,
                content_hash: apoc.util.md5($content),
                confidence: $confidence,
                entity_confidence: $entity_confidence,
                community_confidence: $community_confidence,
                global_confidence: $global_confidence,
                created_at: datetime(),
                passes_threshold: $passes_threshold
            })
            """
            
            await self.neo4j_conn.execute_write(
                store_query,
                {
                    "result_id": result_id,
                    "content": content,
                    "confidence": confidence,
                    "entity_confidence": entity_result["confidence"],
                    "community_confidence": community_result["confidence"],
                    "global_confidence": global_result["confidence"],
                    "passes_threshold": confidence >= settings.validation_threshold
                }
            )
            
        except Exception as e:
            logger.warning("Failed to store validation result", error=str(e))
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for GraphRAG service."""
        try:
            if not self.is_initialized:
                return {"status": "unhealthy", "error": "Service not initialized"}
            
            # Test basic functionality
            test_result = await self._validate_entities("test content", {})
            
            return {
                "status": "healthy",
                "initialized": self.is_initialized,
                "components": {
                    "neo4j": self.neo4j_conn.is_connected if self.neo4j_conn else False,
                    "vector_store": self.vector_store is not None,
                    "embedding_model": self.embedding_model is not None
                }
            }
            
        except Exception as e:
            logger.error("GraphRAG health check failed", error=str(e))
            return {"status": "unhealthy", "error": str(e)}