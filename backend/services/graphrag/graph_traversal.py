"""
Graph traversal strategies for GraphRAG system.
Implements advanced graph algorithms for knowledge exploration and community detection.
"""

import asyncio
from typing import Any, Dict, List, Set, Tuple, Optional, Union
import uuid
from datetime import datetime
from collections import deque, defaultdict
import heapq
import math

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import structlog

from core.config import get_settings
from .neo4j_optimizer import Neo4jQueryOptimizer, find_shortest_path, find_entity_connections
from .entity_extractor import EntityExtractionPipeline
from .relationship_extractor import RelationshipExtractor

logger = structlog.get_logger(__name__)
settings = get_settings()


class GraphTraversalStrategies:
    """
    Advanced graph traversal strategies for GraphRAG knowledge exploration.
    Implements breadth-first, depth-first, shortest path, community detection, and centrality algorithms.
    """
    
    def __init__(self):
        self.neo4j_optimizer = Neo4jQueryOptimizer()
        self.is_initialized = False
        
        # Traversal algorithms configuration
        self.algorithms = {
            'breadth_first': {'max_depth': 4, 'max_nodes': 100},
            'depth_first': {'max_depth': 6, 'max_nodes': 50}, 
            'shortest_path': {'max_paths': 5, 'max_length': 8},
            'community_detection': {'min_community_size': 3, 'resolution': 1.0},
            'centrality_analysis': {'top_n': 20, 'algorithms': ['degree', 'betweenness', 'pagerank']},
            'similarity_traversal': {'similarity_threshold': 0.7, 'max_similar': 10}
        }
        
        # Performance tracking
        self.traversal_stats = {
            'total_traversals': 0,
            'avg_processing_time_ms': 0,
            'algorithm_usage_counts': {},
            'avg_nodes_explored': 0,
            'avg_relationships_found': 0
        }
    
    async def initialize(self) -> None:
        """Initialize the graph traversal system."""
        try:
            logger.info("Initializing graph traversal strategies...")
            start_time = datetime.now()
            
            # Initialize Neo4j optimizer
            await self.neo4j_optimizer.initialize()
            
            init_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"Graph traversal strategies initialized in {init_time:.2f}ms")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize graph traversal strategies: {str(e)}")
            raise
    
    async def traverse_breadth_first(
        self,
        start_entity: str,
        max_depth: int = 3,
        max_nodes: int = 50,
        relationship_filters: Optional[List[str]] = None,
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Perform breadth-first traversal from a starting entity.
        
        Args:
            start_entity: Starting entity name
            max_depth: Maximum traversal depth
            max_nodes: Maximum number of nodes to explore
            relationship_filters: Optional list of relationship types to include
            min_confidence: Minimum relationship confidence threshold
            
        Returns:
            Dictionary with traversal results and metadata
        """
        if not self.is_initialized:
            raise RuntimeError("Graph traversal strategies not initialized")
        
        traversal_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            # Breadth-first search using queue
            queue = deque([(start_entity, 0)])  # (entity, depth)
            visited = set()
            explored_nodes = []
            relationships = []
            depth_levels = defaultdict(list)
            
            while queue and len(explored_nodes) < max_nodes:
                current_entity, depth = queue.popleft()
                
                if current_entity in visited or depth > max_depth:
                    continue
                
                visited.add(current_entity)
                explored_nodes.append(current_entity)
                depth_levels[depth].append(current_entity)
                
                # Find connected entities
                connections = await find_entity_connections(
                    self.neo4j_optimizer,
                    current_entity,
                    min_confidence,
                    limit=20
                )
                
                for connection in connections:
                    connected_entity = connection['connected_entity']
                    relationship_type = connection['relationship_type']
                    confidence = connection['confidence']
                    
                    # Apply relationship filters
                    if relationship_filters and relationship_type not in relationship_filters:
                        continue
                    
                    # Add relationship
                    relationships.append({
                        'source': current_entity,
                        'target': connected_entity,
                        'relationship_type': relationship_type,
                        'confidence': confidence,
                        'depth': depth
                    })
                    
                    # Add to queue for next level exploration
                    if connected_entity not in visited:
                        queue.append((connected_entity, depth + 1))
            
            # Build traversal graph for analysis
            traversal_graph = self._build_traversal_graph(explored_nodes, relationships)
            graph_metrics = self._calculate_traversal_metrics(traversal_graph)
            
            # Find key insights
            insights = self._extract_traversal_insights(
                explored_nodes, relationships, depth_levels, 'breadth_first'
            )
            
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._update_traversal_stats('breadth_first', len(explored_nodes), len(relationships), processing_time_ms)
            
            result = {
                'traversal_id': traversal_id,
                'algorithm': 'breadth_first',
                'start_entity': start_entity,
                'explored_nodes': explored_nodes,
                'relationships': relationships,
                'depth_levels': dict(depth_levels),
                'graph_metrics': graph_metrics,
                'insights': insights,
                'total_nodes_explored': len(explored_nodes),
                'total_relationships_found': len(relationships),
                'max_depth_reached': max(depth_levels.keys()) if depth_levels else 0,
                'processing_time_ms': processing_time_ms,
                'timestamp': start_time.isoformat()
            }
            
            logger.info(
                "Breadth-first traversal completed",
                traversal_id=traversal_id,
                nodes_explored=len(explored_nodes),
                relationships_found=len(relationships),
                processing_time_ms=processing_time_ms
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Breadth-first traversal failed: {str(e)}", traversal_id=traversal_id)
            raise
    
    async def traverse_depth_first(
        self,
        start_entity: str,
        max_depth: int = 4,
        max_nodes: int = 30,
        relationship_filters: Optional[List[str]] = None,
        min_confidence: float = 0.5,
        exploration_strategy: str = 'highest_confidence'
    ) -> Dict[str, Any]:
        """
        Perform depth-first traversal from a starting entity.
        
        Args:
            start_entity: Starting entity name
            max_depth: Maximum traversal depth
            max_nodes: Maximum number of nodes to explore
            relationship_filters: Optional list of relationship types to include
            min_confidence: Minimum relationship confidence threshold
            exploration_strategy: 'highest_confidence', 'most_connections', or 'random'
            
        Returns:
            Dictionary with traversal results and metadata
        """
        traversal_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            visited = set()
            explored_nodes = []
            relationships = []
            traversal_paths = []
            
            async def dfs_recursive(entity: str, depth: int, current_path: List[str]):
                if (entity in visited or 
                    depth > max_depth or 
                    len(explored_nodes) >= max_nodes):
                    return
                
                visited.add(entity)
                explored_nodes.append(entity)
                current_path.append(entity)
                
                # Find connected entities
                connections = await find_entity_connections(
                    self.neo4j_optimizer,
                    entity,
                    min_confidence,
                    limit=15
                )
                
                # Sort connections based on exploration strategy
                if exploration_strategy == 'highest_confidence':
                    connections.sort(key=lambda x: x['confidence'], reverse=True)
                elif exploration_strategy == 'most_connections':
                    # This would require additional query - simplified for now
                    pass
                
                for connection in connections:
                    connected_entity = connection['connected_entity']
                    relationship_type = connection['relationship_type']
                    confidence = connection['confidence']
                    
                    # Apply relationship filters
                    if relationship_filters and relationship_type not in relationship_filters:
                        continue
                    
                    # Add relationship
                    relationships.append({
                        'source': entity,
                        'target': connected_entity,
                        'relationship_type': relationship_type,
                        'confidence': confidence,
                        'depth': depth
                    })
                    
                    # Continue DFS if not visited
                    if connected_entity not in visited:
                        new_path = current_path.copy()
                        await dfs_recursive(connected_entity, depth + 1, new_path)
                
                # Store complete path
                if len(current_path) > 1:
                    traversal_paths.append(current_path.copy())
            
            # Start DFS
            await dfs_recursive(start_entity, 0, [])
            
            # Build traversal graph for analysis
            traversal_graph = self._build_traversal_graph(explored_nodes, relationships)
            graph_metrics = self._calculate_traversal_metrics(traversal_graph)
            
            # Find key insights
            insights = self._extract_traversal_insights(
                explored_nodes, relationships, {}, 'depth_first'
            )
            
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._update_traversal_stats('depth_first', len(explored_nodes), len(relationships), processing_time_ms)
            
            result = {
                'traversal_id': traversal_id,
                'algorithm': 'depth_first',
                'start_entity': start_entity,
                'explored_nodes': explored_nodes,
                'relationships': relationships,
                'traversal_paths': traversal_paths,
                'graph_metrics': graph_metrics,
                'insights': insights,
                'total_nodes_explored': len(explored_nodes),
                'total_relationships_found': len(relationships),
                'longest_path_length': max(len(path) for path in traversal_paths) if traversal_paths else 0,
                'processing_time_ms': processing_time_ms,
                'timestamp': start_time.isoformat()
            }
            
            logger.info(
                "Depth-first traversal completed",
                traversal_id=traversal_id,
                nodes_explored=len(explored_nodes),
                relationships_found=len(relationships),
                processing_time_ms=processing_time_ms
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Depth-first traversal failed: {str(e)}", traversal_id=traversal_id)
            raise
    
    async def find_shortest_paths(
        self,
        start_entity: str,
        target_entities: List[str],
        max_path_length: int = 5,
        min_confidence: float = 0.3
    ) -> Dict[str, Any]:
        """
        Find shortest paths between start entity and multiple target entities.
        
        Args:
            start_entity: Starting entity name
            target_entities: List of target entity names
            max_path_length: Maximum path length to consider
            min_confidence: Minimum relationship confidence threshold
            
        Returns:
            Dictionary with shortest paths and analysis
        """
        traversal_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            all_paths = []
            path_analysis = {}
            
            # Find paths to each target entity
            for target_entity in target_entities:
                paths = await find_shortest_path(
                    self.neo4j_optimizer,
                    start_entity,
                    target_entity,
                    min_confidence,
                    limit=3
                )
                
                if paths:
                    # Process each path
                    for path in paths:
                        path_data = {
                            'start_entity': start_entity,
                            'target_entity': target_entity,
                            'entity_path': path['entity_path'],
                            'relationship_path': path['relationship_path'],
                            'path_confidence': path['path_confidence'],
                            'path_length': len(path['entity_path']) - 1,
                            'path_id': f"{start_entity}→{target_entity}"
                        }
                        all_paths.append(path_data)
            
            # Analyze path patterns
            path_analysis = self._analyze_path_patterns(all_paths)
            
            # Find common intermediate entities
            intermediate_entities = self._find_common_intermediates(all_paths)
            
            # Calculate path importance scores
            scored_paths = self._score_path_importance(all_paths)
            
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._update_traversal_stats('shortest_path', len(set(target_entities)), len(all_paths), processing_time_ms)
            
            result = {
                'traversal_id': traversal_id,
                'algorithm': 'shortest_paths',
                'start_entity': start_entity,
                'target_entities': target_entities,
                'paths': scored_paths,
                'path_analysis': path_analysis,
                'common_intermediates': intermediate_entities,
                'total_paths_found': len(all_paths),
                'avg_path_length': np.mean([p['path_length'] for p in all_paths]) if all_paths else 0,
                'avg_path_confidence': np.mean([p['path_confidence'] for p in all_paths]) if all_paths else 0,
                'processing_time_ms': processing_time_ms,
                'timestamp': start_time.isoformat()
            }
            
            logger.info(
                "Shortest paths analysis completed",
                traversal_id=traversal_id,
                paths_found=len(all_paths),
                targets=len(target_entities),
                processing_time_ms=processing_time_ms
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Shortest paths analysis failed: {str(e)}", traversal_id=traversal_id)
            raise
    
    async def detect_communities(
        self,
        center_entities: Optional[List[str]] = None,
        min_community_size: int = 3,
        max_communities: int = 10,
        resolution: float = 1.0,
        algorithm: str = 'louvain'
    ) -> Dict[str, Any]:
        """
        Detect communities in the knowledge graph using various algorithms.
        
        Args:
            center_entities: Optional list of entities to focus community detection around
            min_community_size: Minimum size for a community
            max_communities: Maximum number of communities to detect
            resolution: Resolution parameter for community detection
            algorithm: 'louvain', 'leiden', or 'modularity'
            
        Returns:
            Dictionary with detected communities and analysis
        """
        traversal_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            # Query Neo4j for graph structure
            if center_entities:
                # Focus on subgraph around center entities
                graph_query = """
                MATCH (center:Entity)
                WHERE center.name IN $center_entities
                MATCH (center)-[r1]-(n1)-[r2]-(n2)
                WHERE r1.confidence > 0.4 AND r2.confidence > 0.4
                RETURN DISTINCT n1.name as source, type(r2) as relationship, n2.name as target,
                       r2.confidence as confidence
                LIMIT 500
                """
                graph_data, metrics = await self.neo4j_optimizer.execute_custom_query(
                    graph_query, 
                    {'center_entities': center_entities}
                )
            else:
                # Get general graph structure
                graph_query = """
                MATCH (n1:Entity)-[r]->(n2:Entity)
                WHERE r.confidence > 0.5
                RETURN n1.name as source, type(r) as relationship, n2.name as target,
                       r.confidence as confidence
                ORDER BY r.confidence DESC
                LIMIT 1000
                """
                graph_data, metrics = await self.neo4j_optimizer.execute_custom_query(
                    graph_query, 
                    {}
                )
            
            # Build NetworkX graph from Neo4j data
            G = nx.Graph()
            for record in graph_data:
                source = record['source']
                target = record['target']
                confidence = record['confidence']
                G.add_edge(source, target, weight=confidence)
            
            if len(G.nodes()) < min_community_size:
                return {
                    'communities': [],
                    'error': 'Insufficient nodes for community detection'
                }
            
            # Apply community detection algorithm
            communities = []
            if algorithm == 'louvain':
                communities = self._detect_louvain_communities(G, resolution)
            elif algorithm == 'leiden':
                communities = self._detect_leiden_communities(G, resolution)
            else:  # modularity
                communities = self._detect_modularity_communities(G)
            
            # Filter communities by minimum size
            filtered_communities = [
                comm for comm in communities 
                if len(comm['members']) >= min_community_size
            ][:max_communities]
            
            # Analyze community properties
            community_analysis = self._analyze_communities(G, filtered_communities)
            
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._update_traversal_stats('community_detection', len(G.nodes()), len(filtered_communities), processing_time_ms)
            
            result = {
                'traversal_id': traversal_id,
                'algorithm': f'community_detection_{algorithm}',
                'communities': filtered_communities,
                'community_analysis': community_analysis,
                'total_nodes_analyzed': len(G.nodes()),
                'total_edges_analyzed': len(G.edges()),
                'communities_found': len(filtered_communities),
                'avg_community_size': np.mean([len(c['members']) for c in filtered_communities]) if filtered_communities else 0,
                'modularity_score': community_analysis.get('overall_modularity', 0),
                'processing_time_ms': processing_time_ms,
                'timestamp': start_time.isoformat()
            }
            
            logger.info(
                "Community detection completed",
                traversal_id=traversal_id,
                communities_found=len(filtered_communities),
                algorithm=algorithm,
                processing_time_ms=processing_time_ms
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Community detection failed: {str(e)}", traversal_id=traversal_id)
            raise
    
    async def analyze_centrality(
        self,
        entity_subset: Optional[List[str]] = None,
        centrality_measures: List[str] = ['degree', 'betweenness', 'pagerank', 'closeness'],
        top_n: int = 20
    ) -> Dict[str, Any]:
        """
        Analyze entity centrality using various measures.
        
        Args:
            entity_subset: Optional list of entities to focus analysis on
            centrality_measures: List of centrality measures to calculate
            top_n: Number of top entities to return for each measure
            
        Returns:
            Dictionary with centrality analysis results
        """
        traversal_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            # Query Neo4j for graph structure
            if entity_subset:
                graph_query = """
                MATCH (n1:Entity)-[r]->(n2:Entity)
                WHERE n1.name IN $entities OR n2.name IN $entities
                AND r.confidence > 0.4
                RETURN n1.name as source, n2.name as target, r.confidence as weight
                """
                graph_data, metrics = await self.neo4j_optimizer.execute_custom_query(
                    graph_query,
                    {'entities': entity_subset}
                )
            else:
                graph_query = """
                MATCH (n1:Entity)-[r]->(n2:Entity)
                WHERE r.confidence > 0.5
                RETURN n1.name as source, n2.name as target, r.confidence as weight
                ORDER BY r.confidence DESC
                LIMIT 2000
                """
                graph_data, metrics = await self.neo4j_optimizer.execute_custom_query(
                    graph_query,
                    {}
                )
            
            # Build NetworkX graph
            G = nx.DiGraph()  # Use directed graph for more accurate centrality
            for record in graph_data:
                G.add_edge(record['source'], record['target'], weight=record['weight'])
            
            if len(G.nodes()) == 0:
                return {'centrality_results': {}, 'error': 'No graph data available'}
            
            # Calculate centrality measures
            centrality_results = {}
            
            if 'degree' in centrality_measures:
                degree_centrality = nx.degree_centrality(G)
                centrality_results['degree'] = self._top_n_centrality(degree_centrality, top_n)
            
            if 'betweenness' in centrality_measures:
                betweenness_centrality = nx.betweenness_centrality(G, k=min(100, len(G.nodes())))
                centrality_results['betweenness'] = self._top_n_centrality(betweenness_centrality, top_n)
            
            if 'pagerank' in centrality_measures:
                pagerank_centrality = nx.pagerank(G, max_iter=100)
                centrality_results['pagerank'] = self._top_n_centrality(pagerank_centrality, top_n)
            
            if 'closeness' in centrality_measures and nx.is_weakly_connected(G):
                closeness_centrality = nx.closeness_centrality(G)
                centrality_results['closeness'] = self._top_n_centrality(closeness_centrality, top_n)
            
            if 'eigenvector' in centrality_measures:
                try:
                    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=100)
                    centrality_results['eigenvector'] = self._top_n_centrality(eigenvector_centrality, top_n)
                except nx.PowerIterationFailedConvergence:
                    logger.warning("Eigenvector centrality calculation failed to converge")
            
            # Find entities that appear in multiple top rankings
            influential_entities = self._find_influential_entities(centrality_results)
            
            # Analyze centrality patterns
            centrality_analysis = self._analyze_centrality_patterns(centrality_results, G)
            
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._update_traversal_stats('centrality_analysis', len(G.nodes()), len(centrality_measures), processing_time_ms)
            
            result = {
                'traversal_id': traversal_id,
                'algorithm': 'centrality_analysis',
                'centrality_results': centrality_results,
                'influential_entities': influential_entities,
                'centrality_analysis': centrality_analysis,
                'graph_size': len(G.nodes()),
                'total_edges': len(G.edges()),
                'measures_calculated': centrality_measures,
                'processing_time_ms': processing_time_ms,
                'timestamp': start_time.isoformat()
            }
            
            logger.info(
                "Centrality analysis completed",
                traversal_id=traversal_id,
                measures=len(centrality_measures),
                graph_size=len(G.nodes()),
                processing_time_ms=processing_time_ms
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Centrality analysis failed: {str(e)}", traversal_id=traversal_id)
            raise
    
    def _build_traversal_graph(self, nodes: List[str], relationships: List[Dict[str, Any]]) -> nx.Graph:
        """Build NetworkX graph from traversal results."""
        G = nx.Graph()
        
        # Add nodes
        for node in nodes:
            G.add_node(node)
        
        # Add edges
        for rel in relationships:
            G.add_edge(
                rel['source'],
                rel['target'],
                relationship_type=rel['relationship_type'],
                confidence=rel['confidence'],
                depth=rel.get('depth', 0)
            )
        
        return G
    
    def _calculate_traversal_metrics(self, graph: nx.Graph) -> Dict[str, Any]:
        """Calculate metrics for traversal graph."""
        if len(graph.nodes()) == 0:
            return {'nodes': 0, 'edges': 0, 'density': 0}
        
        try:
            metrics = {
                'nodes': len(graph.nodes()),
                'edges': len(graph.edges()),
                'density': nx.density(graph),
                'avg_clustering': nx.average_clustering(graph),
                'connected_components': nx.number_connected_components(graph)
            }
            
            if nx.is_connected(graph):
                metrics['diameter'] = nx.diameter(graph)
                metrics['radius'] = nx.radius(graph)
                metrics['avg_shortest_path_length'] = nx.average_shortest_path_length(graph)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Graph metrics calculation failed: {str(e)}")
            return {
                'nodes': len(graph.nodes()),
                'edges': len(graph.edges()),
                'error': str(e)
            }
    
    def _extract_traversal_insights(
        self,
        nodes: List[str],
        relationships: List[Dict[str, Any]],
        depth_levels: Dict[int, List[str]],
        algorithm: str
    ) -> Dict[str, Any]:
        """Extract insights from traversal results."""
        insights = {
            'node_types': {},
            'relationship_types': {},
            'high_confidence_relationships': 0,
            'key_connectors': []
        }
        
        try:
            # Analyze relationship types and confidences
            rel_type_counts = defaultdict(int)
            high_conf_count = 0
            
            for rel in relationships:
                rel_type_counts[rel['relationship_type']] += 1
                if rel['confidence'] > 0.8:
                    high_conf_count += 1
            
            insights['relationship_types'] = dict(rel_type_counts)
            insights['high_confidence_relationships'] = high_conf_count
            insights['avg_relationship_confidence'] = np.mean([r['confidence'] for r in relationships]) if relationships else 0
            
            # Find nodes with most connections (potential key connectors)
            node_connections = defaultdict(int)
            for rel in relationships:
                node_connections[rel['source']] += 1
                node_connections[rel['target']] += 1
            
            top_connectors = sorted(node_connections.items(), key=lambda x: x[1], reverse=True)[:5]
            insights['key_connectors'] = [{'entity': entity, 'connections': count} for entity, count in top_connectors]
            
            # Algorithm-specific insights
            if algorithm == 'breadth_first' and depth_levels:
                insights['entities_per_depth'] = {str(k): len(v) for k, v in depth_levels.items()}
                insights['max_depth_reached'] = max(depth_levels.keys()) if depth_levels else 0
            
            return insights
            
        except Exception as e:
            logger.warning(f"Insight extraction failed: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_path_patterns(self, paths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in shortest paths."""
        analysis = {
            'common_relationship_types': {},
            'path_length_distribution': {},
            'confidence_distribution': {
                'high': 0,    # > 0.8
                'medium': 0,  # 0.5 - 0.8
                'low': 0      # < 0.5
            }
        }
        
        try:
            # Analyze relationship types in paths
            rel_type_counts = defaultdict(int)
            length_counts = defaultdict(int)
            
            for path in paths:
                for rel_type in path['relationship_path']:
                    rel_type_counts[rel_type] += 1
                
                length_counts[path['path_length']] += 1
                
                # Confidence distribution
                confidence = path['path_confidence']
                if confidence > 0.8:
                    analysis['confidence_distribution']['high'] += 1
                elif confidence > 0.5:
                    analysis['confidence_distribution']['medium'] += 1
                else:
                    analysis['confidence_distribution']['low'] += 1
            
            analysis['common_relationship_types'] = dict(rel_type_counts)
            analysis['path_length_distribution'] = dict(length_counts)
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Path pattern analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _find_common_intermediates(self, paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find entities that appear frequently as intermediates in paths."""
        intermediate_counts = defaultdict(int)
        
        for path in paths:
            entity_path = path['entity_path']
            # Skip start and end entities, count intermediates
            for entity in entity_path[1:-1]:
                intermediate_counts[entity] += 1
        
        # Sort by frequency
        common_intermediates = sorted(
            intermediate_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return [
            {'entity': entity, 'frequency': count} 
            for entity, count in common_intermediates
        ]
    
    def _score_path_importance(self, paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score paths by importance (confidence, length, relationship types)."""
        for path in paths:
            # Calculate importance score
            confidence_score = path['path_confidence'] * 0.5
            length_penalty = max(0, (5 - path['path_length']) / 5) * 0.3  # Prefer shorter paths
            
            # Boost for certain relationship types
            rel_type_boost = 0
            important_rels = ['FOUNDED_BY', 'WORKS_FOR', 'LOCATED_IN', 'CREATES']
            for rel_type in path['relationship_path']:
                if rel_type in important_rels:
                    rel_type_boost += 0.05
            
            path['importance_score'] = min(confidence_score + length_penalty + rel_type_boost, 1.0)
        
        # Sort by importance score
        return sorted(paths, key=lambda x: x['importance_score'], reverse=True)
    
    def _detect_louvain_communities(self, graph: nx.Graph, resolution: float) -> List[Dict[str, Any]]:
        """Detect communities using Louvain algorithm."""
        try:
            import community as community_louvain
            
            partition = community_louvain.best_partition(graph, resolution=resolution)
            
            # Group nodes by community
            communities_dict = defaultdict(list)
            for node, community_id in partition.items():
                communities_dict[community_id].append(node)
            
            communities = []
            for community_id, members in communities_dict.items():
                communities.append({
                    'community_id': community_id,
                    'members': members,
                    'size': len(members)
                })
            
            return communities
            
        except ImportError:
            # Fallback to simple connected components
            logger.warning("python-louvain not available, using connected components")
            components = list(nx.connected_components(graph))
            return [
                {
                    'community_id': i,
                    'members': list(component),
                    'size': len(component)
                }
                for i, component in enumerate(components)
            ]
    
    def _detect_leiden_communities(self, graph: nx.Graph, resolution: float) -> List[Dict[str, Any]]:
        """Detect communities using Leiden algorithm."""
        # Simplified implementation - would require igraph/leidenalg
        logger.warning("Leiden algorithm not implemented, falling back to Louvain")
        return self._detect_louvain_communities(graph, resolution)
    
    def _detect_modularity_communities(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """Detect communities using modularity optimization."""
        try:
            communities_generator = nx.community.greedy_modularity_communities(graph)
            communities = []
            
            for i, community in enumerate(communities_generator):
                communities.append({
                    'community_id': i,
                    'members': list(community),
                    'size': len(community)
                })
            
            return communities
            
        except Exception as e:
            logger.warning(f"Modularity community detection failed: {str(e)}")
            return []
    
    def _analyze_communities(self, graph: nx.Graph, communities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze properties of detected communities."""
        analysis = {
            'community_sizes': [c['size'] for c in communities],
            'avg_community_size': 0,
            'modularity_scores': {},
            'inter_community_edges': 0,
            'intra_community_edges': 0
        }
        
        try:
            if communities:
                analysis['avg_community_size'] = np.mean([c['size'] for c in communities])
            
            # Calculate modularity
            partition = {}
            for i, community in enumerate(communities):
                for member in community['members']:
                    partition[member] = i
            
            if partition:
                analysis['overall_modularity'] = nx.community.modularity(
                    graph, 
                    [set(c['members']) for c in communities]
                )
            
            # Count inter vs intra community edges
            inter_edges = 0
            intra_edges = 0
            
            for edge in graph.edges():
                source_comm = partition.get(edge[0], -1)
                target_comm = partition.get(edge[1], -1)
                
                if source_comm == target_comm and source_comm != -1:
                    intra_edges += 1
                else:
                    inter_edges += 1
            
            analysis['inter_community_edges'] = inter_edges
            analysis['intra_community_edges'] = intra_edges
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Community analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _top_n_centrality(self, centrality_dict: Dict[str, float], n: int) -> List[Dict[str, Any]]:
        """Get top N entities by centrality score."""
        sorted_entities = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)
        return [
            {'entity': entity, 'score': round(score, 4)}
            for entity, score in sorted_entities[:n]
        ]
    
    def _find_influential_entities(self, centrality_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Find entities that appear in multiple centrality top rankings."""
        entity_appearances = defaultdict(list)
        
        for measure, rankings in centrality_results.items():
            for i, item in enumerate(rankings[:10]):  # Top 10 for each measure
                entity_appearances[item['entity']].append({
                    'measure': measure,
                    'rank': i + 1,
                    'score': item['score']
                })
        
        # Sort by number of appearances and average rank
        influential = []
        for entity, appearances in entity_appearances.items():
            if len(appearances) >= 2:  # Appears in at least 2 measures
                avg_rank = np.mean([a['rank'] for a in appearances])
                influential.append({
                    'entity': entity,
                    'appearances': len(appearances),
                    'avg_rank': round(avg_rank, 2),
                    'measures': appearances
                })
        
        return sorted(influential, key=lambda x: (-x['appearances'], x['avg_rank']))[:10]
    
    def _analyze_centrality_patterns(self, centrality_results: Dict[str, List[Dict[str, Any]]], graph: nx.Graph) -> Dict[str, Any]:
        """Analyze patterns in centrality measures."""
        analysis = {
            'measure_correlations': {},
            'centrality_distribution': {},
            'top_entities_overlap': {}
        }
        
        try:
            # Calculate correlations between measures
            measures = list(centrality_results.keys())
            if len(measures) >= 2:
                for i, measure1 in enumerate(measures):
                    for measure2 in measures[i+1:]:
                        # Get entities that appear in both top 10
                        entities1 = {item['entity'] for item in centrality_results[measure1][:10]}
                        entities2 = {item['entity'] for item in centrality_results[measure2][:10]}
                        
                        overlap = len(entities1.intersection(entities2))
                        analysis['top_entities_overlap'][f"{measure1}_vs_{measure2}"] = overlap
            
            # Distribution analysis
            for measure, rankings in centrality_results.items():
                scores = [item['score'] for item in rankings]
                if scores:
                    analysis['centrality_distribution'][measure] = {
                        'mean': round(np.mean(scores), 4),
                        'std': round(np.std(scores), 4),
                        'min': round(min(scores), 4),
                        'max': round(max(scores), 4)
                    }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Centrality pattern analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _update_traversal_stats(self, algorithm: str, nodes_explored: int, relationships_found: int, processing_time_ms: float) -> None:
        """Update traversal statistics."""
        self.traversal_stats['total_traversals'] += 1
        
        # Update algorithm usage counts
        self.traversal_stats['algorithm_usage_counts'][algorithm] = (
            self.traversal_stats['algorithm_usage_counts'].get(algorithm, 0) + 1
        )
        
        # Update average processing time
        total_time = (
            self.traversal_stats['avg_processing_time_ms'] * 
            (self.traversal_stats['total_traversals'] - 1) + 
            processing_time_ms
        )
        self.traversal_stats['avg_processing_time_ms'] = (
            total_time / self.traversal_stats['total_traversals']
        )
        
        # Update average nodes and relationships
        total_nodes = (
            self.traversal_stats['avg_nodes_explored'] * 
            (self.traversal_stats['total_traversals'] - 1) + 
            nodes_explored
        )
        self.traversal_stats['avg_nodes_explored'] = (
            total_nodes / self.traversal_stats['total_traversals']
        )
        
        total_rels = (
            self.traversal_stats['avg_relationships_found'] * 
            (self.traversal_stats['total_traversals'] - 1) + 
            relationships_found
        )
        self.traversal_stats['avg_relationships_found'] = (
            total_rels / self.traversal_stats['total_traversals']
        )
    
    async def get_traversal_statistics(self) -> Dict[str, Any]:
        """Get current traversal statistics."""
        return {
            'total_traversals': self.traversal_stats['total_traversals'],
            'avg_processing_time_ms': round(self.traversal_stats['avg_processing_time_ms'], 2),
            'algorithm_usage_distribution': self.traversal_stats['algorithm_usage_counts'],
            'avg_nodes_explored': round(self.traversal_stats['avg_nodes_explored'], 2),
            'avg_relationships_found': round(self.traversal_stats['avg_relationships_found'], 2),
            'available_algorithms': list(self.algorithms.keys()),
            'performance_target_met': self.traversal_stats['avg_processing_time_ms'] < 100
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check graph traversal system health."""
        try:
            if not self.is_initialized:
                return {
                    'status': 'unhealthy',
                    'error': 'Graph traversal system not initialized'
                }
            
            # Test basic traversal
            start_time = datetime.now()
            
            # Test with a simple breadth-first traversal
            test_result = await self.traverse_breadth_first(
                'test_entity',
                max_depth=1,
                max_nodes=5,
                min_confidence=0.1
            )
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Get Neo4j optimizer health
            neo4j_health = await self.neo4j_optimizer.health_check()
            
            return {
                'status': 'healthy' if response_time < 200 else 'degraded',
                'response_time_ms': round(response_time, 2),
                'neo4j_optimizer_status': neo4j_health['status'],
                'algorithms_available': list(self.algorithms.keys()),
                'total_traversals_performed': self.traversal_stats['total_traversals'],
                'avg_processing_time_ms': round(self.traversal_stats['avg_processing_time_ms'], 2)
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def close(self) -> None:
        """Close graph traversal system and cleanup resources."""
        if self.neo4j_optimizer:
            await self.neo4j_optimizer.close()
        
        self.is_initialized = False
        logger.info("Graph traversal system closed")


# Convenience functions for common traversal operations

async def explore_entity_neighborhood(
    traversal_system: GraphTraversalStrategies,
    entity_name: str,
    exploration_depth: int = 2,
    max_entities: int = 25
) -> Dict[str, Any]:
    """Explore the immediate neighborhood of an entity using breadth-first search."""
    return await traversal_system.traverse_breadth_first(
        entity_name,
        max_depth=exploration_depth,
        max_nodes=max_entities,
        min_confidence=0.5
    )


async def find_entity_connections(
    traversal_system: GraphTraversalStrategies,
    source_entity: str,
    target_entities: List[str],
    max_path_length: int = 4
) -> Dict[str, Any]:
    """Find connections between a source entity and multiple target entities."""
    return await traversal_system.find_shortest_paths(
        source_entity,
        target_entities,
        max_path_length,
        min_confidence=0.4
    )


async def discover_entity_communities(
    traversal_system: GraphTraversalStrategies,
    focus_entities: List[str],
    min_community_size: int = 4
) -> Dict[str, Any]:
    """Discover communities around a set of focus entities."""
    return await traversal_system.detect_communities(
        center_entities=focus_entities,
        min_community_size=min_community_size,
        algorithm='louvain'
    )


async def identify_key_entities(
    traversal_system: GraphTraversalStrategies,
    entity_subset: Optional[List[str]] = None,
    top_n: int = 15
) -> Dict[str, Any]:
    """Identify the most important entities using centrality analysis."""
    return await traversal_system.analyze_centrality(
        entity_subset,
        centrality_measures=['degree', 'betweenness', 'pagerank'],
        top_n=top_n
    )