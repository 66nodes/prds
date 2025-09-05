"""
Dynamic Graph Mutation Service for real-time Neo4j updates.
Handles CRUD operations on graph nodes and relationships with event-driven updates.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from enum import Enum

import structlog
from pydantic import BaseModel, Field, validator
import redis.asyncio as redis

from core.config import get_settings
from core.database import get_neo4j, get_redis
from services.websocket_manager import websocket_manager, MessageType

logger = structlog.get_logger(__name__)
settings = get_settings()


class MutationType(str, Enum):
    """Graph mutation operation types."""
    CREATE_NODE = "create_node"
    UPDATE_NODE = "update_node"
    DELETE_NODE = "delete_node"
    CREATE_RELATIONSHIP = "create_relationship"
    UPDATE_RELATIONSHIP = "update_relationship"
    DELETE_RELATIONSHIP = "delete_relationship"
    BULK_CREATE = "bulk_create"
    BULK_UPDATE = "bulk_update"
    BULK_DELETE = "bulk_delete"


class NodeType(str, Enum):
    """Supported node types in the knowledge graph."""
    REQUIREMENT = "Requirement"
    PRD = "PRD"
    OBJECTIVE = "Objective"
    ENTITY = "Entity"
    COMMUNITY = "Community"
    PROJECT = "Project"
    USER = "User"
    TASK = "Task"
    VALIDATION_RESULT = "ValidationResult"
    AGENT = "Agent"
    WORKFLOW = "Workflow"


class RelationshipType(str, Enum):
    """Supported relationship types."""
    CONTAINS = "CONTAINS"
    RELATES_TO = "RELATES_TO"
    PART_OF = "PART_OF"
    DEPENDS_ON = "DEPENDS_ON"
    VALIDATES = "VALIDATES"
    CREATED_BY = "CREATED_BY"
    ASSIGNED_TO = "ASSIGNED_TO"
    FOLLOWS = "FOLLOWS"
    CONTRADICTS = "CONTRADICTS"
    IMPLEMENTS = "IMPLEMENTS"


class GraphNode(BaseModel):
    """Node representation for graph mutations."""
    id: Optional[str] = Field(default_factory=lambda: str(uuid4()))
    type: NodeType
    properties: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    @validator('properties')
    def add_timestamps(cls, v):
        if 'created_at' not in v:
            v['created_at'] = datetime.utcnow().isoformat()
        v['updated_at'] = datetime.utcnow().isoformat()
        return v


class GraphRelationship(BaseModel):
    """Relationship representation for graph mutations."""
    id: Optional[str] = Field(default_factory=lambda: str(uuid4()))
    type: RelationshipType
    from_node_id: str
    to_node_id: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('properties')
    def add_timestamps(cls, v):
        if 'created_at' not in v:
            v['created_at'] = datetime.utcnow().isoformat()
        return v


class MutationEvent(BaseModel):
    """Event representation for graph mutations."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    mutation_type: MutationType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    source: str = "api"  # api, agent, system
    data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MutationResult(BaseModel):
    """Result of a graph mutation operation."""
    success: bool
    mutation_id: str
    operation: MutationType
    affected_nodes: int = 0
    affected_relationships: int = 0
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: float


class GraphMutationService:
    """
    Service for managing dynamic graph updates with ACID compliance.
    
    Features:
    - CRUD operations for nodes and relationships
    - Event-driven updates via Redis pub/sub
    - Transaction management with rollback
    - Real-time notifications via WebSocket
    - Conflict resolution for concurrent updates
    - Audit trail for all mutations
    """
    
    def __init__(self):
        self.neo4j_conn = None
        self.redis_client = None
        self.is_initialized = False
        self.event_queue = asyncio.Queue()
        self.event_processor_task = None
        
        # Configuration
        self.max_batch_size = 100
        self.transaction_timeout_seconds = 30
        self.event_channel = "graph_mutations"
    
    async def initialize(self) -> None:
        """Initialize the graph mutation service."""
        try:
            self.neo4j_conn = await get_neo4j()
            self.redis_client = await get_redis()
            
            # Start event processor
            self.event_processor_task = asyncio.create_task(self._process_events())
            
            # Subscribe to Redis channel for events
            asyncio.create_task(self._subscribe_to_events())
            
            self.is_initialized = True
            logger.info("Graph mutation service initialized")
            
        except Exception as e:
            logger.error("Failed to initialize graph mutation service", error=str(e))
            raise
    
    # === Node Operations ===
    
    async def create_node(
        self,
        node: GraphNode,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        emit_event: bool = True
    ) -> MutationResult:
        """Create a new node in the graph."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Build Cypher query
            query = f"""
            CREATE (n:{node.type.value} $properties)
            SET n.id = $node_id
            RETURN n
            """
            
            parameters = {
                "node_id": node.id,
                "properties": node.properties
            }
            
            # Add embedding if provided
            if node.embedding:
                query = f"""
                CREATE (n:{node.type.value} $properties)
                SET n.id = $node_id,
                    n.embedding = $embedding
                RETURN n
                """
                parameters["embedding"] = node.embedding
            
            # Execute in transaction
            result = await self.neo4j_conn.execute_write(query, parameters)
            
            # Emit event if enabled
            if emit_event:
                await self._emit_mutation_event(
                    MutationEvent(
                        mutation_type=MutationType.CREATE_NODE,
                        user_id=user_id,
                        agent_id=agent_id,
                        data={
                            "node_id": node.id,
                            "node_type": node.type.value,
                            "properties": node.properties
                        }
                    )
                )
            
            # Send WebSocket notification
            if user_id:
                await self._send_websocket_update(
                    user_id,
                    "graph_node_created",
                    {"node_id": node.id, "type": node.type.value}
                )
            
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return MutationResult(
                success=True,
                mutation_id=node.id,
                operation=MutationType.CREATE_NODE,
                affected_nodes=result.get("nodes_created", 1),
                execution_time_ms=execution_time,
                data={"node_id": node.id}
            )
            
        except Exception as e:
            logger.error("Failed to create node", error=str(e), node_id=node.id)
            return MutationResult(
                success=False,
                mutation_id=node.id,
                operation=MutationType.CREATE_NODE,
                error=str(e),
                execution_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
            )
    
    async def update_node(
        self,
        node_id: str,
        node_type: NodeType,
        updates: Dict[str, Any],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        emit_event: bool = True
    ) -> MutationResult:
        """Update an existing node in the graph."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Add update timestamp
            updates["updated_at"] = datetime.utcnow().isoformat()
            
            # Build SET clause
            set_clauses = [f"n.{key} = ${key}" for key in updates.keys()]
            set_clause = ", ".join(set_clauses)
            
            query = f"""
            MATCH (n:{node_type.value} {{id: $node_id}})
            SET {set_clause}
            RETURN n
            """
            
            parameters = {"node_id": node_id, **updates}
            
            # Execute update
            result = await self.neo4j_conn.execute_write(query, parameters)
            
            # Emit event
            if emit_event:
                await self._emit_mutation_event(
                    MutationEvent(
                        mutation_type=MutationType.UPDATE_NODE,
                        user_id=user_id,
                        agent_id=agent_id,
                        data={
                            "node_id": node_id,
                            "node_type": node_type.value,
                            "updates": updates
                        }
                    )
                )
            
            # WebSocket notification
            if user_id:
                await self._send_websocket_update(
                    user_id,
                    "graph_node_updated",
                    {"node_id": node_id, "type": node_type.value}
                )
            
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return MutationResult(
                success=True,
                mutation_id=node_id,
                operation=MutationType.UPDATE_NODE,
                affected_nodes=result.get("properties_set", 0) > 0,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error("Failed to update node", error=str(e), node_id=node_id)
            return MutationResult(
                success=False,
                mutation_id=node_id,
                operation=MutationType.UPDATE_NODE,
                error=str(e),
                execution_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
            )
    
    async def delete_node(
        self,
        node_id: str,
        node_type: NodeType,
        cascade: bool = False,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        emit_event: bool = True
    ) -> MutationResult:
        """Delete a node from the graph."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if cascade:
                # Delete node and all its relationships
                query = f"""
                MATCH (n:{node_type.value} {{id: $node_id}})
                DETACH DELETE n
                RETURN count(n) as deleted_count
                """
            else:
                # Only delete if no relationships exist
                query = f"""
                MATCH (n:{node_type.value} {{id: $node_id}})
                WHERE NOT (n)--()
                DELETE n
                RETURN count(n) as deleted_count
                """
            
            result = await self.neo4j_conn.execute_write(
                query,
                {"node_id": node_id}
            )
            
            deleted_count = result.get("nodes_deleted", 0)
            
            if deleted_count > 0:
                # Emit event
                if emit_event:
                    await self._emit_mutation_event(
                        MutationEvent(
                            mutation_type=MutationType.DELETE_NODE,
                            user_id=user_id,
                            agent_id=agent_id,
                            data={
                                "node_id": node_id,
                                "node_type": node_type.value,
                                "cascade": cascade
                            }
                        )
                    )
                
                # WebSocket notification
                if user_id:
                    await self._send_websocket_update(
                        user_id,
                        "graph_node_deleted",
                        {"node_id": node_id, "type": node_type.value}
                    )
            
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return MutationResult(
                success=deleted_count > 0,
                mutation_id=node_id,
                operation=MutationType.DELETE_NODE,
                affected_nodes=deleted_count,
                execution_time_ms=execution_time,
                error="Node not found or has relationships" if deleted_count == 0 else None
            )
            
        except Exception as e:
            logger.error("Failed to delete node", error=str(e), node_id=node_id)
            return MutationResult(
                success=False,
                mutation_id=node_id,
                operation=MutationType.DELETE_NODE,
                error=str(e),
                execution_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
            )
    
    # === Relationship Operations ===
    
    async def create_relationship(
        self,
        relationship: GraphRelationship,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        emit_event: bool = True
    ) -> MutationResult:
        """Create a new relationship between nodes."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            query = f"""
            MATCH (a {{id: $from_id}})
            MATCH (b {{id: $to_id}})
            CREATE (a)-[r:{relationship.type.value} $properties]->(b)
            SET r.id = $rel_id
            RETURN r
            """
            
            parameters = {
                "from_id": relationship.from_node_id,
                "to_id": relationship.to_node_id,
                "rel_id": relationship.id,
                "properties": relationship.properties
            }
            
            result = await self.neo4j_conn.execute_write(query, parameters)
            
            # Emit event
            if emit_event:
                await self._emit_mutation_event(
                    MutationEvent(
                        mutation_type=MutationType.CREATE_RELATIONSHIP,
                        user_id=user_id,
                        agent_id=agent_id,
                        data={
                            "relationship_id": relationship.id,
                            "type": relationship.type.value,
                            "from_node": relationship.from_node_id,
                            "to_node": relationship.to_node_id
                        }
                    )
                )
            
            # WebSocket notification
            if user_id:
                await self._send_websocket_update(
                    user_id,
                    "graph_relationship_created",
                    {
                        "relationship_id": relationship.id,
                        "type": relationship.type.value
                    }
                )
            
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return MutationResult(
                success=True,
                mutation_id=relationship.id,
                operation=MutationType.CREATE_RELATIONSHIP,
                affected_relationships=result.get("relationships_created", 1),
                execution_time_ms=execution_time,
                data={"relationship_id": relationship.id}
            )
            
        except Exception as e:
            logger.error("Failed to create relationship", error=str(e))
            return MutationResult(
                success=False,
                mutation_id=relationship.id,
                operation=MutationType.CREATE_RELATIONSHIP,
                error=str(e),
                execution_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
            )
    
    async def update_relationship(
        self,
        relationship_id: str,
        updates: Dict[str, Any],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        emit_event: bool = True
    ) -> MutationResult:
        """Update an existing relationship."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Add update timestamp
            updates["updated_at"] = datetime.utcnow().isoformat()
            
            # Build SET clause
            set_clauses = [f"r.{key} = ${key}" for key in updates.keys()]
            set_clause = ", ".join(set_clauses)
            
            query = f"""
            MATCH ()-[r {{id: $rel_id}}]-()
            SET {set_clause}
            RETURN r
            """
            
            parameters = {"rel_id": relationship_id, **updates}
            
            result = await self.neo4j_conn.execute_write(query, parameters)
            
            # Emit event
            if emit_event:
                await self._emit_mutation_event(
                    MutationEvent(
                        mutation_type=MutationType.UPDATE_RELATIONSHIP,
                        user_id=user_id,
                        agent_id=agent_id,
                        data={
                            "relationship_id": relationship_id,
                            "updates": updates
                        }
                    )
                )
            
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return MutationResult(
                success=True,
                mutation_id=relationship_id,
                operation=MutationType.UPDATE_RELATIONSHIP,
                affected_relationships=1,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error("Failed to update relationship", error=str(e))
            return MutationResult(
                success=False,
                mutation_id=relationship_id,
                operation=MutationType.UPDATE_RELATIONSHIP,
                error=str(e),
                execution_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
            )
    
    async def delete_relationship(
        self,
        relationship_id: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        emit_event: bool = True
    ) -> MutationResult:
        """Delete a relationship from the graph."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            query = """
            MATCH ()-[r {id: $rel_id}]-()
            DELETE r
            RETURN count(r) as deleted_count
            """
            
            result = await self.neo4j_conn.execute_write(
                query,
                {"rel_id": relationship_id}
            )
            
            deleted_count = result.get("relationships_deleted", 0)
            
            if deleted_count > 0:
                # Emit event
                if emit_event:
                    await self._emit_mutation_event(
                        MutationEvent(
                            mutation_type=MutationType.DELETE_RELATIONSHIP,
                            user_id=user_id,
                            agent_id=agent_id,
                            data={"relationship_id": relationship_id}
                        )
                    )
                
                # WebSocket notification
                if user_id:
                    await self._send_websocket_update(
                        user_id,
                        "graph_relationship_deleted",
                        {"relationship_id": relationship_id}
                    )
            
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return MutationResult(
                success=deleted_count > 0,
                mutation_id=relationship_id,
                operation=MutationType.DELETE_RELATIONSHIP,
                affected_relationships=deleted_count,
                execution_time_ms=execution_time,
                error="Relationship not found" if deleted_count == 0 else None
            )
            
        except Exception as e:
            logger.error("Failed to delete relationship", error=str(e))
            return MutationResult(
                success=False,
                mutation_id=relationship_id,
                operation=MutationType.DELETE_RELATIONSHIP,
                error=str(e),
                execution_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
            )
    
    # === Bulk Operations ===
    
    async def bulk_create_nodes(
        self,
        nodes: List[GraphNode],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> MutationResult:
        """Create multiple nodes in a single transaction."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Batch nodes by type for efficiency
            nodes_by_type = {}
            for node in nodes:
                if node.type not in nodes_by_type:
                    nodes_by_type[node.type] = []
                nodes_by_type[node.type].append(node)
            
            total_created = 0
            
            # Create nodes for each type
            for node_type, typed_nodes in nodes_by_type.items():
                batch_data = [
                    {
                        "id": node.id,
                        "properties": node.properties,
                        "embedding": node.embedding
                    }
                    for node in typed_nodes
                ]
                
                query = f"""
                UNWIND $batch as item
                CREATE (n:{node_type.value})
                SET n = item.properties,
                    n.id = item.id
                """
                
                if any(node.embedding for node in typed_nodes):
                    query += ", n.embedding = item.embedding"
                
                query += " RETURN count(n) as created_count"
                
                result = await self.neo4j_conn.execute_write(
                    query,
                    {"batch": batch_data}
                )
                
                total_created += result.get("nodes_created", len(typed_nodes))
            
            # Emit single bulk event
            await self._emit_mutation_event(
                MutationEvent(
                    mutation_type=MutationType.BULK_CREATE,
                    user_id=user_id,
                    agent_id=agent_id,
                    data={
                        "node_count": len(nodes),
                        "node_types": list(nodes_by_type.keys())
                    }
                )
            )
            
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return MutationResult(
                success=True,
                mutation_id=str(uuid4()),
                operation=MutationType.BULK_CREATE,
                affected_nodes=total_created,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error("Failed to bulk create nodes", error=str(e))
            return MutationResult(
                success=False,
                mutation_id=str(uuid4()),
                operation=MutationType.BULK_CREATE,
                error=str(e),
                execution_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
            )
    
    # === Event Processing ===
    
    async def _emit_mutation_event(self, event: MutationEvent) -> None:
        """Emit mutation event to Redis for processing."""
        try:
            if self.redis_client:
                # Publish to Redis channel
                await self.redis_client.publish(
                    self.event_channel,
                    event.model_dump_json()
                )
                
                # Also add to local queue for processing
                await self.event_queue.put(event)
                
                logger.debug(
                    "Mutation event emitted",
                    event_id=event.id,
                    mutation_type=event.mutation_type.value
                )
        except Exception as e:
            logger.error("Failed to emit mutation event", error=str(e))
    
    async def _subscribe_to_events(self) -> None:
        """Subscribe to Redis channel for mutation events."""
        try:
            if not self.redis_client:
                return
            
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe(self.event_channel)
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        event_data = json.loads(message['data'])
                        event = MutationEvent(**event_data)
                        await self.event_queue.put(event)
                    except Exception as e:
                        logger.error("Failed to process Redis message", error=str(e))
                        
        except Exception as e:
            logger.error("Failed to subscribe to events", error=str(e))
    
    async def _process_events(self) -> None:
        """Process mutation events from the queue."""
        while True:
            try:
                event = await self.event_queue.get()
                
                # Log event for audit trail
                await self._log_mutation_event(event)
                
                # Additional processing based on event type
                if event.mutation_type in [MutationType.CREATE_NODE, MutationType.UPDATE_NODE]:
                    # Trigger validation for new/updated content
                    await self._trigger_validation(event)
                
                logger.debug(
                    "Processed mutation event",
                    event_id=event.id,
                    mutation_type=event.mutation_type.value
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error processing event", error=str(e))
                await asyncio.sleep(1)
    
    async def _log_mutation_event(self, event: MutationEvent) -> None:
        """Log mutation event to Neo4j for audit trail."""
        try:
            query = """
            CREATE (e:MutationEvent {
                id: $event_id,
                mutation_type: $mutation_type,
                timestamp: $timestamp,
                user_id: $user_id,
                agent_id: $agent_id,
                source: $source,
                data: $data,
                metadata: $metadata
            })
            """
            
            await self.neo4j_conn.execute_write(
                query,
                {
                    "event_id": event.id,
                    "mutation_type": event.mutation_type.value,
                    "timestamp": event.timestamp.isoformat(),
                    "user_id": event.user_id,
                    "agent_id": event.agent_id,
                    "source": event.source,
                    "data": json.dumps(event.data),
                    "metadata": json.dumps(event.metadata)
                }
            )
        except Exception as e:
            logger.error("Failed to log mutation event", error=str(e))
    
    async def _trigger_validation(self, event: MutationEvent) -> None:
        """Trigger GraphRAG validation for mutation events."""
        # This would integrate with the existing GraphRAG validation service
        # Implementation depends on specific validation requirements
        pass
    
    async def _send_websocket_update(
        self,
        user_id: str,
        update_type: str,
        data: Dict[str, Any]
    ) -> None:
        """Send real-time update via WebSocket."""
        try:
            await websocket_manager.send_to_user(
                user_id,
                {
                    "type": update_type,
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": data
                }
            )
        except Exception as e:
            logger.error("Failed to send WebSocket update", error=str(e))
    
    # === Transaction Management ===
    
    async def execute_transaction(
        self,
        operations: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> MutationResult:
        """Execute multiple operations in a single transaction with rollback capability."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with self.neo4j_conn.session() as session:
                async with session.begin_transaction() as tx:
                    results = []
                    
                    for operation in operations:
                        op_type = operation.get("type")
                        op_data = operation.get("data")
                        
                        if op_type == "create_node":
                            result = await self._tx_create_node(tx, op_data)
                        elif op_type == "update_node":
                            result = await self._tx_update_node(tx, op_data)
                        elif op_type == "create_relationship":
                            result = await self._tx_create_relationship(tx, op_data)
                        else:
                            raise ValueError(f"Unsupported operation type: {op_type}")
                        
                        results.append(result)
                    
                    # Commit transaction
                    await tx.commit()
            
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return MutationResult(
                success=True,
                mutation_id=str(uuid4()),
                operation=MutationType.BULK_UPDATE,
                affected_nodes=sum(r.get("nodes", 0) for r in results),
                affected_relationships=sum(r.get("relationships", 0) for r in results),
                execution_time_ms=execution_time,
                data={"operations_count": len(operations)}
            )
            
        except Exception as e:
            logger.error("Transaction failed and rolled back", error=str(e))
            return MutationResult(
                success=False,
                mutation_id=str(uuid4()),
                operation=MutationType.BULK_UPDATE,
                error=str(e),
                execution_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
            )
    
    async def _tx_create_node(self, tx, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create node within transaction."""
        query = f"""
        CREATE (n:{data['type']} $properties)
        SET n.id = $node_id
        RETURN n
        """
        
        result = await tx.run(
            query,
            {
                "node_id": data['id'],
                "properties": data['properties']
            }
        )
        
        return {"nodes": 1, "relationships": 0}
    
    async def _tx_update_node(self, tx, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update node within transaction."""
        set_clauses = [f"n.{key} = ${key}" for key in data['updates'].keys()]
        set_clause = ", ".join(set_clauses)
        
        query = f"""
        MATCH (n:{data['type']} {{id: $node_id}})
        SET {set_clause}
        RETURN n
        """
        
        result = await tx.run(
            query,
            {"node_id": data['id'], **data['updates']}
        )
        
        return {"nodes": 1, "relationships": 0}
    
    async def _tx_create_relationship(self, tx, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create relationship within transaction."""
        query = f"""
        MATCH (a {{id: $from_id}})
        MATCH (b {{id: $to_id}})
        CREATE (a)-[r:{data['type']} $properties]->(b)
        SET r.id = $rel_id
        RETURN r
        """
        
        result = await tx.run(
            query,
            {
                "from_id": data['from_node_id'],
                "to_id": data['to_node_id'],
                "rel_id": data['id'],
                "properties": data.get('properties', {})
            }
        )
        
        return {"nodes": 0, "relationships": 1}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of graph mutation service."""
        try:
            return {
                "status": "healthy" if self.is_initialized else "unhealthy",
                "initialized": self.is_initialized,
                "neo4j_connected": self.neo4j_conn.is_connected if self.neo4j_conn else False,
                "redis_connected": self.redis_client is not None,
                "event_queue_size": self.event_queue.qsize(),
                "event_processor_active": (
                    self.event_processor_task is not None and 
                    not self.event_processor_task.done()
                )
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Global service instance
graph_mutation_service = GraphMutationService()


async def get_graph_mutation_service() -> GraphMutationService:
    """Get the graph mutation service instance."""
    if not graph_mutation_service.is_initialized:
        await graph_mutation_service.initialize()
    return graph_mutation_service