"""
Integration tests for dynamic graph mutations with concurrent operations.
"""

import asyncio
import uuid
from typing import Any, Dict, List
import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch

from services.graphrag.graph_mutations import (
    GraphNode,
    GraphRelationship,
    GraphMutationService,
    NodeType,
    RelationshipType,
    MutationType,
    MutationEvent,
    get_graph_mutation_service
)


@pytest.fixture
async def graph_service():
    """Fixture for graph mutation service."""
    service = GraphMutationService()
    
    # Mock the Neo4j and Redis connections
    service.neo4j_conn = AsyncMock()
    service.neo4j_conn.is_connected = True
    service.neo4j_conn.execute_write = AsyncMock()
    service.neo4j_conn.execute_query = AsyncMock()
    service.neo4j_conn.session = AsyncMock()
    
    service.redis_client = AsyncMock()
    service.redis_client.publish = AsyncMock()
    service.redis_client.pubsub = AsyncMock()
    
    service.is_initialized = True
    
    yield service
    
    # Cleanup
    if service.event_processor_task:
        service.event_processor_task.cancel()


@pytest.fixture
async def auth_headers():
    """Fixture for authenticated request headers."""
    return {
        "Authorization": "Bearer jwt_auth_token_001",
        "Content-Type": "application/json"
    }


class TestGraphNodeOperations:
    """Test suite for node CRUD operations."""
    
    @pytest.mark.asyncio
    async def test_create_node_success(self, graph_service):
        """Test successful node creation."""
        # Arrange
        node = GraphNode(
            type=NodeType.REQUIREMENT,
            properties={
                "title": "User Authentication Requirement",
                "description": "Implement secure user authentication with JWT tokens",
                "priority": "high"
            }
        )
        
        graph_service.neo4j_conn.execute_write.return_value = {
            "nodes_created": 1
        }
        
        # Act
        result = await graph_service.create_node(node, user_id="auth_user_001")
        
        # Assert
        assert result.success is True
        assert result.affected_nodes == 1
        assert result.operation == MutationType.CREATE_NODE
        assert result.data["node_id"] == node.id
        
        # Verify Neo4j was called
        graph_service.neo4j_conn.execute_write.assert_called_once()
        
        # Verify event was published
        graph_service.redis_client.publish.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_node_success(self, graph_service):
        """Test successful node update."""
        # Arrange
        node_id = str(uuid.uuid4())
        updates = {
            "status": "completed",
            "completion_date": "2025-01-20"
        }
        
        graph_service.neo4j_conn.execute_write.return_value = {
            "properties_set": len(updates)
        }
        
        # Act
        result = await graph_service.update_node(
            node_id=node_id,
            node_type=NodeType.REQUIREMENT,
            updates=updates,
            user_id="auth_user_001"
        )
        
        # Assert
        assert result.success is True
        assert result.operation == MutationType.UPDATE_NODE
        graph_service.neo4j_conn.execute_write.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_node_cascade(self, graph_service):
        """Test node deletion with cascade option."""
        # Arrange
        node_id = str(uuid.uuid4())
        
        graph_service.neo4j_conn.execute_write.return_value = {
            "nodes_deleted": 1,
            "relationships_deleted": 3
        }
        
        # Act
        result = await graph_service.delete_node(
            node_id=node_id,
            node_type=NodeType.REQUIREMENT,
            cascade=True,
            user_id="auth_user_001"
        )
        
        # Assert
        assert result.success is True
        assert result.affected_nodes == 1
        assert result.operation == MutationType.DELETE_NODE
    
    @pytest.mark.asyncio
    async def test_delete_node_without_cascade_fails_with_relationships(self, graph_service):
        """Test that node deletion fails without cascade when relationships exist."""
        # Arrange
        node_id = str(uuid.uuid4())
        
        graph_service.neo4j_conn.execute_write.return_value = {
            "nodes_deleted": 0
        }
        
        # Act
        result = await graph_service.delete_node(
            node_id=node_id,
            node_type=NodeType.REQUIREMENT,
            cascade=False,
            user_id="auth_user_001"
        )
        
        # Assert
        assert result.success is False
        assert result.error == "Node not found or has relationships"


class TestGraphRelationshipOperations:
    """Test suite for relationship CRUD operations."""
    
    @pytest.mark.asyncio
    async def test_create_relationship_success(self, graph_service):
        """Test successful relationship creation."""
        # Arrange
        relationship = GraphRelationship(
            type=RelationshipType.DEPENDS_ON,
            from_node_id="node1",
            to_node_id="node2",
            properties={"priority": "high"}
        )
        
        graph_service.neo4j_conn.execute_write.return_value = {
            "relationships_created": 1
        }
        
        # Act
        result = await graph_service.create_relationship(
            relationship=relationship,
            user_id="auth_user_001"
        )
        
        # Assert
        assert result.success is True
        assert result.affected_relationships == 1
        assert result.operation == MutationType.CREATE_RELATIONSHIP
    
    @pytest.mark.asyncio
    async def test_update_relationship_success(self, graph_service):
        """Test successful relationship update."""
        # Arrange
        rel_id = str(uuid.uuid4())
        updates = {"priority": "medium", "notes": "Updated"}
        
        graph_service.neo4j_conn.execute_write.return_value = {
            "properties_set": len(updates)
        }
        
        # Act
        result = await graph_service.update_relationship(
            relationship_id=rel_id,
            updates=updates,
            user_id="auth_user_001"
        )
        
        # Assert
        assert result.success is True
        assert result.operation == MutationType.UPDATE_RELATIONSHIP
    
    @pytest.mark.asyncio
    async def test_delete_relationship_success(self, graph_service):
        """Test successful relationship deletion."""
        # Arrange
        rel_id = str(uuid.uuid4())
        
        graph_service.neo4j_conn.execute_write.return_value = {
            "relationships_deleted": 1
        }
        
        # Act
        result = await graph_service.delete_relationship(
            relationship_id=rel_id,
            user_id="auth_user_001"
        )
        
        # Assert
        assert result.success is True
        assert result.affected_relationships == 1


class TestBulkOperations:
    """Test suite for bulk operations."""
    
    @pytest.mark.asyncio
    async def test_bulk_create_nodes(self, graph_service):
        """Test bulk node creation."""
        # Arrange
        nodes = [
            GraphNode(
                type=NodeType.REQUIREMENT,
                properties={"title": f"Requirement {i}"}
            )
            for i in range(10)
        ]
        
        graph_service.neo4j_conn.execute_write.return_value = {
            "nodes_created": len(nodes)
        }
        
        # Act
        result = await graph_service.bulk_create_nodes(
            nodes=nodes,
            user_id="auth_user_001"
        )
        
        # Assert
        assert result.success is True
        assert result.affected_nodes == len(nodes)
        assert result.operation == MutationType.BULK_CREATE


class TestConcurrentOperations:
    """Test suite for concurrent graph operations."""
    
    @pytest.mark.asyncio
    async def test_concurrent_node_creation(self, graph_service):
        """Test multiple nodes being created concurrently."""
        # Arrange
        num_concurrent = 20
        nodes = [
            GraphNode(
                type=NodeType.REQUIREMENT,
                properties={"title": f"Concurrent Req {i}"}
            )
            for i in range(num_concurrent)
        ]
        
        graph_service.neo4j_conn.execute_write.return_value = {
            "nodes_created": 1
        }
        
        # Act - Create all nodes concurrently
        tasks = [
            graph_service.create_node(node, user_id=f"user_{i}")
            for i, node in enumerate(nodes)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Assert
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == num_concurrent
        assert all(r.success for r in successful_results)
        
        # Verify all calls were made
        assert graph_service.neo4j_conn.execute_write.call_count == num_concurrent
    
    @pytest.mark.asyncio
    async def test_concurrent_updates_same_node(self, graph_service):
        """Test concurrent updates to the same node (last-write-wins)."""
        # Arrange
        node_id = str(uuid.uuid4())
        num_concurrent = 10
        
        graph_service.neo4j_conn.execute_write.return_value = {
            "properties_set": 1
        }
        
        # Act - Multiple concurrent updates to same node
        tasks = [
            graph_service.update_node(
                node_id=node_id,
                node_type=NodeType.REQUIREMENT,
                updates={"counter": i},
                user_id=f"user_{i}"
            )
            for i in range(num_concurrent)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Assert
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == num_concurrent
        assert all(r.success for r in successful_results)
    
    @pytest.mark.asyncio
    async def test_concurrent_relationship_creation(self, graph_service):
        """Test concurrent relationship creation between different nodes."""
        # Arrange
        num_concurrent = 15
        relationships = [
            GraphRelationship(
                type=RelationshipType.DEPENDS_ON,
                from_node_id=f"node_{i}",
                to_node_id=f"node_{i+1}",
                properties={"index": i}
            )
            for i in range(num_concurrent)
        ]
        
        graph_service.neo4j_conn.execute_write.return_value = {
            "relationships_created": 1
        }
        
        # Act
        tasks = [
            graph_service.create_relationship(rel, user_id=f"user_{i}")
            for i, rel in enumerate(relationships)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Assert
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == num_concurrent
        assert all(r.success for r in successful_results)


class TestTransactionManagement:
    """Test suite for ACID transaction management."""
    
    @pytest.mark.asyncio
    async def test_transaction_success(self, graph_service):
        """Test successful transaction with multiple operations."""
        # Arrange
        operations = [
            {
                "type": "create_node",
                "data": {
                    "type": NodeType.REQUIREMENT.value,
                    "id": "req-001",
                    "properties": {"title": "New Requirement"}
                }
            },
            {
                "type": "create_relationship",
                "data": {
                    "type": RelationshipType.DEPENDS_ON.value,
                    "from_node_id": "req-001",
                    "to_node_id": "req-002",
                    "id": "rel-001"
                }
            }
        ]
        
        # Mock transaction context
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_tx.run = AsyncMock()
        mock_tx.commit = AsyncMock()
        
        mock_session.begin_transaction = AsyncMock(return_value=mock_tx)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()
        
        graph_service.neo4j_conn.session = AsyncMock(return_value=mock_session)
        
        # Act
        result = await graph_service.execute_transaction(
            operations=operations,
            user_id="auth_user_001"
        )
        
        # Assert
        assert result.success is True
        assert result.operation == MutationType.BULK_UPDATE
        mock_tx.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self, graph_service):
        """Test transaction rollback when an operation fails."""
        # Arrange
        operations = [
            {
                "type": "create_node",
                "data": {
                    "type": NodeType.REQUIREMENT.value,
                    "id": "req-001",
                    "properties": {"title": "New Requirement"}
                }
            },
            {
                "type": "invalid_operation",  # This will cause an error
                "data": {}
            }
        ]
        
        # Mock transaction context
        mock_session = AsyncMock()
        mock_tx = AsyncMock()
        mock_session.begin_transaction = AsyncMock(return_value=mock_tx)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()
        
        graph_service.neo4j_conn.session = AsyncMock(return_value=mock_session)
        
        # Act
        result = await graph_service.execute_transaction(
            operations=operations,
            user_id="auth_user_001"
        )
        
        # Assert
        assert result.success is False
        assert "Unsupported operation type" in result.error
        # Transaction should not be committed on error
        mock_tx.commit.assert_not_called()


class TestEventProcessing:
    """Test suite for event-driven updates."""
    
    @pytest.mark.asyncio
    async def test_event_emission_on_create(self, graph_service):
        """Test that events are emitted when nodes are created."""
        # Arrange
        node = GraphNode(
            type=NodeType.REQUIREMENT,
            properties={"title": "Authentication System"}
        )
        
        graph_service.neo4j_conn.execute_write.return_value = {
            "nodes_created": 1
        }
        
        # Act
        await graph_service.create_node(node, user_id="auth_user_001", emit_event=True)
        
        # Assert
        graph_service.redis_client.publish.assert_called_once()
        call_args = graph_service.redis_client.publish.call_args
        assert call_args[0][0] == "graph_mutations"  # Channel name
        
        # Verify event structure
        import json
        event_data = json.loads(call_args[0][1])
        assert event_data["mutation_type"] == MutationType.CREATE_NODE.value
        assert event_data["user_id"] == "test_user"
    
    @pytest.mark.asyncio
    async def test_event_suppression_when_disabled(self, graph_service):
        """Test that events are not emitted when emit_event=False."""
        # Arrange
        node = GraphNode(
            type=NodeType.REQUIREMENT,
            properties={"title": "Authentication System"}
        )
        
        graph_service.neo4j_conn.execute_write.return_value = {
            "nodes_created": 1
        }
        
        # Act
        await graph_service.create_node(node, user_id="auth_user_001", emit_event=False)
        
        # Assert
        graph_service.redis_client.publish.assert_not_called()


class TestDataConsistency:
    """Test suite for data consistency and integrity."""
    
    @pytest.mark.asyncio
    async def test_timestamp_auto_addition(self):
        """Test that timestamps are automatically added to nodes and relationships."""
        # Arrange
        node = GraphNode(
            type=NodeType.REQUIREMENT,
            properties={"title": "Authentication System"}
        )
        
        relationship = GraphRelationship(
            type=RelationshipType.DEPENDS_ON,
            from_node_id="node1",
            to_node_id="node2"
        )
        
        # Assert - Check timestamps were added
        assert "created_at" in node.properties
        assert "updated_at" in node.properties
        assert "created_at" in relationship.properties
    
    @pytest.mark.asyncio
    async def test_unique_id_generation(self):
        """Test that unique IDs are generated for nodes and relationships."""
        # Arrange
        nodes = [
            GraphNode(type=NodeType.REQUIREMENT, properties={})
            for _ in range(100)
        ]
        
        # Assert - All IDs should be unique
        ids = [node.id for node in nodes]
        assert len(ids) == len(set(ids))  # All unique
    
    @pytest.mark.asyncio
    async def test_concurrent_id_generation(self):
        """Test that concurrent ID generation doesn't produce duplicates."""
        # Arrange
        async def create_node():
            return GraphNode(type=NodeType.REQUIREMENT, properties={})
        
        # Act - Create many nodes concurrently
        tasks = [create_node() for _ in range(100)]
        nodes = await asyncio.gather(*tasks)
        
        # Assert - All IDs should still be unique
        ids = [node.id for node in nodes]
        assert len(ids) == len(set(ids))


class TestAPIIntegration:
    """Test suite for API endpoint integration."""
    
    @pytest.mark.asyncio
    async def test_create_node_api(self, client: AsyncClient, auth_headers):
        """Test node creation via API endpoint."""
        # Arrange
        request_data = {
            "type": "REQUIREMENT",
            "properties": {
                "title": "API Test Requirement",
                "description": "Created via API"
            }
        }
        
        # Act
        response = await client.post(
            "/api/v1/graph/nodes",
            json=request_data,
            headers=auth_headers
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["operation"] == "create_node"
    
    @pytest.mark.asyncio
    async def test_bulk_create_api(self, client: AsyncClient, auth_headers):
        """Test bulk node creation via API endpoint."""
        # Arrange
        request_data = {
            "nodes": [
                {
                    "type": "REQUIREMENT",
                    "properties": {"title": f"Bulk Req {i}"}
                }
                for i in range(5)
            ]
        }
        
        # Act
        response = await client.post(
            "/api/v1/graph/nodes/bulk",
            json=request_data,
            headers=auth_headers
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["operation"] == "bulk_create"
    
    @pytest.mark.asyncio
    async def test_transaction_api(self, client: AsyncClient, auth_headers):
        """Test transaction execution via API endpoint."""
        # Arrange
        request_data = {
            "operations": [
                {
                    "type": "create_node",
                    "data": {
                        "type": "REQUIREMENT",
                        "id": "req-tx-001",
                        "properties": {"title": "Transaction Test"}
                    }
                },
                {
                    "type": "create_relationship",
                    "data": {
                        "type": "DEPENDS_ON",
                        "from_node_id": "req-tx-001",
                        "to_node_id": "req-002",
                        "id": "rel-tx-001"
                    }
                }
            ]
        }
        
        # Act
        response = await client.post(
            "/api/v1/graph/transaction",
            json=request_data,
            headers=auth_headers
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["operation"] == "bulk_update"


# Performance test for high concurrency
@pytest.mark.asyncio
@pytest.mark.performance
async def test_high_concurrency_performance(graph_service):
    """Test system performance under high concurrency load."""
    # Arrange
    num_operations = 100
    start_time = asyncio.get_event_loop().time()
    
    graph_service.neo4j_conn.execute_write.return_value = {
        "nodes_created": 1
    }
    
    # Act - Create many nodes concurrently
    tasks = []
    for i in range(num_operations):
        node = GraphNode(
            type=NodeType.REQUIREMENT,
            properties={"title": f"Perf Test {i}"}
        )
        tasks.append(graph_service.create_node(node, user_id=f"user_{i}"))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Calculate performance metrics
    end_time = asyncio.get_event_loop().time()
    total_time = end_time - start_time
    ops_per_second = num_operations / total_time
    
    # Assert
    successful = [r for r in results if not isinstance(r, Exception) and r.success]
    assert len(successful) == num_operations
    
    # Performance assertions
    assert ops_per_second > 50  # Should handle at least 50 ops/sec
    assert total_time < 5  # Should complete 100 operations in under 5 seconds
    
    print(f"Performance: {ops_per_second:.2f} ops/sec")
    print(f"Total time for {num_operations} operations: {total_time:.2f}s")