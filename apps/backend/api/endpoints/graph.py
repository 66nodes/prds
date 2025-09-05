"""
API endpoints for dynamic graph mutations and knowledge graph management.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field

from core.auth import get_current_user
from services.graphrag.graph_mutations import (
    GraphNode,
    GraphRelationship,
    NodeType,
    RelationshipType,
    MutationResult,
    get_graph_mutation_service
)

router = APIRouter(
    prefix="/api/v1/graph",
    tags=["Graph Mutations"],
    dependencies=[Depends(get_current_user)]
)


# === Request/Response Models ===

class CreateNodeRequest(BaseModel):
    """Request model for creating a node."""
    type: NodeType
    properties: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "REQUIREMENT",
                "properties": {
                    "title": "User Authentication",
                    "description": "Implement secure user authentication",
                    "priority": "high",
                    "status": "pending"
                }
            }
        }


class UpdateNodeRequest(BaseModel):
    """Request model for updating a node."""
    updates: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "updates": {
                    "status": "completed",
                    "completion_date": "2025-01-20",
                    "notes": "Successfully implemented"
                }
            }
        }


class CreateRelationshipRequest(BaseModel):
    """Request model for creating a relationship."""
    type: RelationshipType
    from_node_id: str
    to_node_id: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "DEPENDS_ON",
                "from_node_id": "req-001",
                "to_node_id": "req-002",
                "properties": {
                    "dependency_type": "blocking",
                    "priority": "high"
                }
            }
        }


class UpdateRelationshipRequest(BaseModel):
    """Request model for updating a relationship."""
    updates: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "updates": {
                    "priority": "medium",
                    "notes": "Dependency resolved"
                }
            }
        }


class BulkCreateRequest(BaseModel):
    """Request model for bulk node creation."""
    nodes: List[CreateNodeRequest]
    
    class Config:
        json_schema_extra = {
            "example": {
                "nodes": [
                    {
                        "type": "REQUIREMENT",
                        "properties": {"title": "Req 1"}
                    },
                    {
                        "type": "REQUIREMENT",
                        "properties": {"title": "Req 2"}
                    }
                ]
            }
        }


class TransactionRequest(BaseModel):
    """Request model for transactional operations."""
    operations: List[Dict[str, Any]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "operations": [
                    {
                        "type": "create_node",
                        "data": {
                            "type": "REQUIREMENT",
                            "id": "req-001",
                            "properties": {"title": "New Requirement"}
                        }
                    },
                    {
                        "type": "create_relationship",
                        "data": {
                            "type": "DEPENDS_ON",
                            "from_node_id": "req-001",
                            "to_node_id": "req-002",
                            "id": "rel-001",
                            "properties": {}
                        }
                    }
                ]
            }
        }


class GraphQueryRequest(BaseModel):
    """Request model for graph queries."""
    query_type: str = Field(..., description="Type of query: neighbors, path, subgraph")
    node_id: Optional[str] = None
    depth: int = Field(default=1, ge=1, le=5)
    filters: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query_type": "neighbors",
                "node_id": "req-001",
                "depth": 2,
                "filters": {"type": "REQUIREMENT"}
            }
        }


# === Node Endpoints ===

@router.post("/nodes", response_model=MutationResult)
async def create_node(
    request: CreateNodeRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> MutationResult:
    """
    Create a new node in the knowledge graph.
    
    - **type**: Node type (REQUIREMENT, PRD, OBJECTIVE, etc.)
    - **properties**: Key-value pairs for node properties
    - **embedding**: Optional vector embedding for similarity search
    """
    service = await get_graph_mutation_service()
    
    node = GraphNode(
        type=request.type,
        properties=request.properties,
        embedding=request.embedding
    )
    
    result = await service.create_node(
        node=node,
        user_id=current_user.get("user_id"),
        emit_event=True
    )
    
    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error or "Failed to create node"
        )
    
    return result


@router.put("/nodes/{node_id}", response_model=MutationResult)
async def update_node(
    node_id: str,
    node_type: NodeType,
    request: UpdateNodeRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> MutationResult:
    """
    Update an existing node in the knowledge graph.
    
    - **node_id**: Unique identifier of the node
    - **node_type**: Type of the node being updated
    - **updates**: Key-value pairs of properties to update
    """
    service = await get_graph_mutation_service()
    
    result = await service.update_node(
        node_id=node_id,
        node_type=node_type,
        updates=request.updates,
        user_id=current_user.get("user_id"),
        emit_event=True
    )
    
    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error or "Failed to update node"
        )
    
    return result


@router.delete("/nodes/{node_id}", response_model=MutationResult)
async def delete_node(
    node_id: str,
    node_type: NodeType,
    cascade: bool = Query(False, description="Delete relationships if they exist"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> MutationResult:
    """
    Delete a node from the knowledge graph.
    
    - **node_id**: Unique identifier of the node
    - **node_type**: Type of the node being deleted
    - **cascade**: If true, delete all relationships. If false, only delete if no relationships exist
    """
    service = await get_graph_mutation_service()
    
    result = await service.delete_node(
        node_id=node_id,
        node_type=node_type,
        cascade=cascade,
        user_id=current_user.get("user_id"),
        emit_event=True
    )
    
    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND if "not found" in (result.error or "").lower() 
                      else status.HTTP_400_BAD_REQUEST,
            detail=result.error or "Failed to delete node"
        )
    
    return result


# === Relationship Endpoints ===

@router.post("/relationships", response_model=MutationResult)
async def create_relationship(
    request: CreateRelationshipRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> MutationResult:
    """
    Create a new relationship between nodes.
    
    - **type**: Relationship type
    - **from_node_id**: Source node ID
    - **to_node_id**: Target node ID
    - **properties**: Optional relationship properties
    """
    service = await get_graph_mutation_service()
    
    relationship = GraphRelationship(
        type=request.type,
        from_node_id=request.from_node_id,
        to_node_id=request.to_node_id,
        properties=request.properties
    )
    
    result = await service.create_relationship(
        relationship=relationship,
        user_id=current_user.get("user_id"),
        emit_event=True
    )
    
    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error or "Failed to create relationship"
        )
    
    return result


@router.put("/relationships/{relationship_id}", response_model=MutationResult)
async def update_relationship(
    relationship_id: str,
    request: UpdateRelationshipRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> MutationResult:
    """
    Update an existing relationship.
    
    - **relationship_id**: Unique identifier of the relationship
    - **updates**: Properties to update
    """
    service = await get_graph_mutation_service()
    
    result = await service.update_relationship(
        relationship_id=relationship_id,
        updates=request.updates,
        user_id=current_user.get("user_id"),
        emit_event=True
    )
    
    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error or "Failed to update relationship"
        )
    
    return result


@router.delete("/relationships/{relationship_id}", response_model=MutationResult)
async def delete_relationship(
    relationship_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> MutationResult:
    """
    Delete a relationship from the knowledge graph.
    
    - **relationship_id**: Unique identifier of the relationship
    """
    service = await get_graph_mutation_service()
    
    result = await service.delete_relationship(
        relationship_id=relationship_id,
        user_id=current_user.get("user_id"),
        emit_event=True
    )
    
    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND if "not found" in (result.error or "").lower()
                      else status.HTTP_400_BAD_REQUEST,
            detail=result.error or "Failed to delete relationship"
        )
    
    return result


# === Bulk Operations ===

@router.post("/nodes/bulk", response_model=MutationResult)
async def bulk_create_nodes(
    request: BulkCreateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> MutationResult:
    """
    Create multiple nodes in a single operation.
    
    Efficient for batch imports and large-scale data ingestion.
    """
    service = await get_graph_mutation_service()
    
    nodes = [
        GraphNode(
            type=node_req.type,
            properties=node_req.properties,
            embedding=node_req.embedding
        )
        for node_req in request.nodes
    ]
    
    result = await service.bulk_create_nodes(
        nodes=nodes,
        user_id=current_user.get("user_id")
    )
    
    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error or "Failed to create nodes"
        )
    
    return result


# === Transaction Operations ===

@router.post("/transaction", response_model=MutationResult)
async def execute_transaction(
    request: TransactionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> MutationResult:
    """
    Execute multiple graph operations in a single transaction.
    
    All operations succeed or fail together (ACID compliance).
    Supports create_node, update_node, and create_relationship operations.
    """
    service = await get_graph_mutation_service()
    
    result = await service.execute_transaction(
        operations=request.operations,
        user_id=current_user.get("user_id")
    )
    
    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error or "Transaction failed"
        )
    
    return result


# === Query Operations ===

@router.post("/query")
async def query_graph(
    request: GraphQueryRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Query the knowledge graph.
    
    Supported query types:
    - **neighbors**: Get neighboring nodes up to specified depth
    - **path**: Find paths between nodes
    - **subgraph**: Extract a subgraph around a node
    """
    service = await get_graph_mutation_service()
    
    if request.query_type == "neighbors":
        # Get neighboring nodes
        query = """
        MATCH (n {id: $node_id})-[*1..$depth]-(m)
        WHERE ALL(label IN labels(m) WHERE label IN $allowed_types OR $allowed_types = [])
        RETURN DISTINCT m.id as id, labels(m) as types, properties(m) as properties
        LIMIT 100
        """
        
        result = await service.neo4j_conn.execute_query(
            query,
            {
                "node_id": request.node_id,
                "depth": request.depth,
                "allowed_types": request.filters.get("types", [])
            }
        )
        
        return {
            "query_type": "neighbors",
            "node_id": request.node_id,
            "depth": request.depth,
            "results": result
        }
    
    elif request.query_type == "path":
        # Find shortest path between nodes
        from_node = request.filters.get("from_node_id")
        to_node = request.filters.get("to_node_id")
        
        if not from_node or not to_node:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both from_node_id and to_node_id required for path query"
            )
        
        query = """
        MATCH path = shortestPath((a {id: $from_id})-[*..10]-(b {id: $to_id}))
        RETURN [node IN nodes(path) | {id: node.id, type: labels(node)[0]}] as nodes,
               [rel IN relationships(path) | {type: type(rel), properties: properties(rel)}] as relationships
        """
        
        result = await service.neo4j_conn.execute_query(
            query,
            {
                "from_id": from_node,
                "to_id": to_node
            }
        )
        
        return {
            "query_type": "path",
            "from_node_id": from_node,
            "to_node_id": to_node,
            "results": result
        }
    
    elif request.query_type == "subgraph":
        # Extract subgraph around a node
        query = """
        MATCH (n {id: $node_id})-[r*0..$depth]-(m)
        WITH DISTINCT n, m, r
        RETURN 
            collect(DISTINCT {id: n.id, type: labels(n)[0], properties: properties(n)}) +
            collect(DISTINCT {id: m.id, type: labels(m)[0], properties: properties(m)}) as nodes,
            [rel IN r | {type: type(rel), from: startNode(rel).id, to: endNode(rel).id, properties: properties(rel)}] as relationships
        """
        
        result = await service.neo4j_conn.execute_query(
            query,
            {
                "node_id": request.node_id,
                "depth": request.depth
            }
        )
        
        return {
            "query_type": "subgraph",
            "center_node_id": request.node_id,
            "depth": request.depth,
            "results": result
        }
    
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported query type: {request.query_type}"
        )


# === Health Check ===

@router.get("/health")
async def graph_health() -> Dict[str, Any]:
    """Check the health status of the graph mutation service."""
    service = await get_graph_mutation_service()
    return await service.health_check()