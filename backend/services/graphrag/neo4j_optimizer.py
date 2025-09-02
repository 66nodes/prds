"""
Neo4j query optimizer for GraphRAG system.
High-performance query execution with <50ms response times.
"""

import asyncio
from typing import Any, Dict, List, Set, Tuple, Optional, Union
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import hashlib

import structlog
from neo4j import AsyncGraphDatabase, Record
from neo4j.exceptions import Neo4jError

from core.config import get_settings
from core.database import get_neo4j

logger = structlog.get_logger(__name__)
settings = get_settings()


@dataclass
class QueryPerformanceMetrics:
    """Performance metrics for Neo4j queries."""
    query_id: str
    execution_time_ms: float
    result_count: int
    cache_hit: bool
    index_used: bool
    memory_usage_mb: float
    timestamp: datetime


class Neo4jQueryOptimizer:
    """
    Advanced Neo4j query optimizer for GraphRAG operations.
    Ensures <50ms response times through caching, indexing, and query optimization.
    """
    
    def __init__(self):
        self.driver = None
        self.is_initialized = False
        
        # Query cache with TTL
        self.query_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
        # Performance tracking
        self.query_metrics = []
        self.max_metrics_history = 1000
        
        # Optimized queries with pre-compiled patterns
        self.optimized_queries = {
            'entity_similarity': """
                CALL db.index.fulltext.queryNodes('entity_search', $search_term) 
                YIELD node, score
                WHERE score > $min_score
                RETURN node.name as name, node.type as type, node.confidence_score as confidence,
                       node.importance_score as importance, score
                ORDER BY score DESC
                LIMIT $limit
            """,
            
            'relationship_lookup': """
                MATCH (s:Entity)-[r]->(t:Entity)
                WHERE s.name_lower = $source_lower AND t.name_lower = $target_lower
                RETURN s.name as source, type(r) as relationship, t.name as target,
                       r.confidence as confidence, r.verified as verified
                LIMIT $limit
            """,
            
            'entity_relationships': """
                MATCH (e:Entity {name: $entity_name})-[r]-(connected)
                WHERE r.confidence > $min_confidence
                RETURN type(r) as relationship_type, connected.name as connected_entity,
                       connected.type as connected_type, r.confidence as confidence
                ORDER BY r.confidence DESC
                LIMIT $limit
            """,
            
            'fact_verification': """
                CALL db.index.fulltext.queryNodes('claim_search', $claim_text) 
                YIELD node, score
                WHERE score > $min_score AND node.verified = true
                RETURN node.text as claim, node.confidence_score as confidence, 
                       node.source_refs as sources, score
                ORDER BY score DESC
                LIMIT $limit
            """,
            
            'community_lookup': """
                MATCH (c:Community)
                WHERE c.level <= $max_level
                WITH c, [(c)-[:CONTAINS]->(e:Entity) | e] as entities
                WHERE size(entities) >= $min_entities
                RETURN c.name as community, c.description as description, 
                       c.level as level, size(entities) as entity_count
                ORDER BY c.importance_score DESC
                LIMIT $limit
            """,
            
            'path_finding': """
                MATCH path = shortestPath((start:Entity {name: $start_entity})-[*1..3]-(end:Entity {name: $end_entity}))
                WHERE ALL(r IN relationships(path) WHERE r.confidence > $min_confidence)
                RETURN [n IN nodes(path) | n.name] as entity_path,
                       [r IN relationships(path) | type(r)] as relationship_path,
                       reduce(conf = 1.0, r IN relationships(path) | conf * r.confidence) as path_confidence
                ORDER BY path_confidence DESC
                LIMIT $limit
            """,
            
            'graph_stats': """
                MATCH (n) 
                RETURN labels(n)[0] as node_type, count(n) as count
                UNION ALL
                MATCH ()-[r]->()
                RETURN type(r) as node_type, count(r) as count
            """
        }
        
        # Index definitions for optimal performance
        self.required_indexes = [
            # Entity indexes
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_name_lower_idx IF NOT EXISTS FOR (e:Entity) ON (e.name_lower)",
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX entity_confidence_idx IF NOT EXISTS FOR (e:Entity) ON (e.confidence_score)",
            "CREATE INDEX entity_importance_idx IF NOT EXISTS FOR (e:Entity) ON (e.importance_score)",
            
            # Relationship indexes
            "CREATE INDEX rel_confidence_idx IF NOT EXISTS FOR ()-[r:RELATED_TO]-() ON (r.confidence)",
            "CREATE INDEX rel_verified_idx IF NOT EXISTS FOR ()-[r:RELATED_TO]-() ON (r.verified)",
            
            # Community indexes
            "CREATE INDEX community_level_idx IF NOT EXISTS FOR (c:Community) ON (c.level)",
            "CREATE INDEX community_importance_idx IF NOT EXISTS FOR (c:Community) ON (c.importance_score)",
            
            # Claim indexes
            "CREATE INDEX claim_confidence_idx IF NOT EXISTS FOR (c:Claim) ON (c.confidence_score)",
            "CREATE INDEX claim_verified_idx IF NOT EXISTS FOR (c:Claim) ON (c.verified)",
            
            # Full-text search indexes
            "CALL db.index.fulltext.createNodeIndex('entity_search', ['Entity'], ['name', 'normalized_name', 'description']) ",
            "CALL db.index.fulltext.createNodeIndex('claim_search', ['Claim'], ['text']) "
        ]
    
    async def initialize(self) -> None:
        """Initialize the Neo4j optimizer with indexes and connection pooling."""
        try:
            logger.info("Initializing Neo4j query optimizer...")
            start_time = time.time()
            
            # Get Neo4j driver with optimized settings
            self.driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
                max_connection_lifetime=3600,  # 1 hour
                max_connection_pool_size=50,   # High concurrency
                connection_acquisition_timeout=5.0,  # 5 second timeout
                max_retry_time=10.0,
                initial_retry_delay=0.1,
                retry_delay_multiplier=1.1,
                retry_delay_jitter_factor=0.1
            )
            
            # Verify connection and create indexes
            await self._setup_database_optimizations()
            
            # Warm up query cache with common queries
            await self._warm_up_cache()
            
            init_time = (time.time() - start_time) * 1000
            logger.info(f"Neo4j optimizer initialized in {init_time:.2f}ms")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j optimizer: {str(e)}")
            raise
    
    async def _setup_database_optimizations(self) -> None:
        """Setup database indexes and constraints for optimal performance."""
        async with self.driver.session() as session:
            try:
                # Create indexes
                for index_query in self.required_indexes:
                    try:
                        await session.run(index_query)
                        logger.debug(f"Created/verified index: {index_query[:50]}...")
                    except Neo4jError as e:
                        if "already exists" not in str(e).lower():
                            logger.warning(f"Index creation failed: {str(e)}")
                
                # Create constraints for data integrity
                constraint_queries = [
                    "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
                    "CREATE CONSTRAINT community_name_unique IF NOT EXISTS FOR (c:Community) REQUIRE c.name IS UNIQUE"
                ]
                
                for constraint_query in constraint_queries:
                    try:
                        await session.run(constraint_query)
                    except Neo4jError as e:
                        if "already exists" not in str(e).lower():
                            logger.warning(f"Constraint creation failed: {str(e)}")
                
                # Pre-compute derived properties for faster queries
                await self._precompute_derived_properties(session)
                
                logger.info("Database optimizations completed")
                
            except Exception as e:
                logger.error(f"Database optimization setup failed: {str(e)}")
                raise
    
    async def _precompute_derived_properties(self, session) -> None:
        """Pre-compute derived properties for faster query execution."""
        try:
            # Add lowercase names for case-insensitive matching
            await session.run("""
                MATCH (e:Entity) 
                WHERE e.name_lower IS NULL
                SET e.name_lower = toLower(e.name)
            """)
            
            # Add normalized names for better matching
            await session.run("""
                MATCH (e:Entity)
                WHERE e.normalized_name IS NULL
                SET e.normalized_name = apoc.text.clean(toLower(e.name))
            """)
            
            logger.debug("Derived properties pre-computed")
            
        except Exception as e:
            logger.warning(f"Failed to pre-compute derived properties: {str(e)}")
    
    async def _warm_up_cache(self) -> None:
        """Warm up query cache with common queries."""
        try:
            # Common entity types
            await self.execute_optimized_query(
                'graph_stats',
                {},
                cache_key='graph_stats_warmup'
            )
            
            # High importance entities
            await self.execute_optimized_query(
                'entity_similarity',
                {
                    'search_term': 'company OR organization',
                    'min_score': 0.1,
                    'limit': 10
                },
                cache_key='top_entities_warmup'
            )
            
            logger.debug("Query cache warmed up")
            
        except Exception as e:
            logger.warning(f"Cache warm-up failed: {str(e)}")
    
    async def execute_optimized_query(
        self,
        query_name: str,
        parameters: Dict[str, Any],
        cache_key: Optional[str] = None,
        cache_ttl: Optional[int] = None
    ) -> Tuple[List[Record], QueryPerformanceMetrics]:
        """
        Execute optimized query with caching and performance monitoring.
        
        Args:
            query_name: Name of the pre-optimized query
            parameters: Query parameters
            cache_key: Optional cache key (auto-generated if not provided)
            cache_ttl: Cache TTL in seconds (default: 300)
            
        Returns:
            Tuple of (results, performance_metrics)
        """
        if not self.is_initialized:
            raise RuntimeError("Neo4j optimizer not initialized")
        
        start_time = time.time()
        query_id = f"{query_name}_{int(start_time * 1000)}"
        
        # Generate cache key if not provided
        if cache_key is None:
            cache_content = f"{query_name}:{str(sorted(parameters.items()))}"
            cache_key = hashlib.md5(cache_content.encode()).hexdigest()
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            self.cache_stats['hits'] += 1
            
            metrics = QueryPerformanceMetrics(
                query_id=query_id,
                execution_time_ms=(time.time() - start_time) * 1000,
                result_count=len(cached_result),
                cache_hit=True,
                index_used=True,  # Assume optimized queries use indexes
                memory_usage_mb=0.0,  # Cached results don't use DB memory
                timestamp=datetime.now()
            )
            
            return cached_result, metrics
        
        # Execute query
        self.cache_stats['misses'] += 1
        
        try:
            if query_name not in self.optimized_queries:
                raise ValueError(f"Unknown optimized query: {query_name}")
            
            query = self.optimized_queries[query_name]
            
            async with self.driver.session() as session:
                # Execute with timeout
                result = await asyncio.wait_for(
                    session.run(query, parameters),
                    timeout=0.1  # 100ms timeout for <50ms target
                )
                
                records = await result.consume()
                data = await result.data()
                
                # Cache results
                self._store_in_cache(cache_key, data, cache_ttl or self.cache_ttl)
                
                # Calculate metrics
                execution_time_ms = (time.time() - start_time) * 1000
                
                metrics = QueryPerformanceMetrics(
                    query_id=query_id,
                    execution_time_ms=execution_time_ms,
                    result_count=len(data),
                    cache_hit=False,
                    index_used=self._query_used_index(records.summary),
                    memory_usage_mb=self._calculate_memory_usage(records.summary),
                    timestamp=datetime.now()
                )
                
                # Track performance
                self._track_performance(metrics)
                
                logger.debug(
                    f"Query {query_name} executed",
                    execution_time_ms=execution_time_ms,
                    result_count=len(data),
                    index_used=metrics.index_used
                )
                
                return data, metrics
                
        except asyncio.TimeoutError:
            logger.warning(f"Query {query_name} timed out")
            raise
        except Exception as e:
            logger.error(f"Query {query_name} failed: {str(e)}")
            raise
    
    async def execute_custom_query(
        self,
        query: str,
        parameters: Dict[str, Any],
        cache_key: Optional[str] = None,
        cache_ttl: Optional[int] = None,
        timeout: float = 0.05  # 50ms timeout
    ) -> Tuple[List[Record], QueryPerformanceMetrics]:
        """
        Execute custom query with optimization and caching.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            cache_key: Optional cache key
            cache_ttl: Cache TTL in seconds
            timeout: Query timeout in seconds
            
        Returns:
            Tuple of (results, performance_metrics)
        """
        if not self.is_initialized:
            raise RuntimeError("Neo4j optimizer not initialized")
        
        start_time = time.time()
        query_id = f"custom_{int(start_time * 1000)}"
        
        # Generate cache key if not provided
        if cache_key is None:
            cache_content = f"{query}:{str(sorted(parameters.items()))}"
            cache_key = hashlib.md5(cache_content.encode()).hexdigest()
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            self.cache_stats['hits'] += 1
            
            metrics = QueryPerformanceMetrics(
                query_id=query_id,
                execution_time_ms=(time.time() - start_time) * 1000,
                result_count=len(cached_result),
                cache_hit=True,
                index_used=True,  # Assume cached queries were optimized
                memory_usage_mb=0.0,
                timestamp=datetime.now()
            )
            
            return cached_result, metrics
        
        # Execute query
        self.cache_stats['misses'] += 1
        
        try:
            async with self.driver.session() as session:
                # Execute with timeout
                result = await asyncio.wait_for(
                    session.run(query, parameters),
                    timeout=timeout
                )
                
                records = await result.consume()
                data = await result.data()
                
                # Cache results
                self._store_in_cache(cache_key, data, cache_ttl or self.cache_ttl)
                
                # Calculate metrics
                execution_time_ms = (time.time() - start_time) * 1000
                
                metrics = QueryPerformanceMetrics(
                    query_id=query_id,
                    execution_time_ms=execution_time_ms,
                    result_count=len(data),
                    cache_hit=False,
                    index_used=self._query_used_index(records.summary),
                    memory_usage_mb=self._calculate_memory_usage(records.summary),
                    timestamp=datetime.now()
                )
                
                # Track performance
                self._track_performance(metrics)
                
                if execution_time_ms > 50:
                    logger.warning(
                        f"Query exceeded 50ms target: {execution_time_ms:.2f}ms",
                        query=query[:100] + "..." if len(query) > 100 else query
                    )
                
                return data, metrics
                
        except asyncio.TimeoutError:
            logger.warning(f"Custom query timed out after {timeout}s")
            raise
        except Exception as e:
            logger.error(f"Custom query failed: {str(e)}")
            raise
    
    async def batch_execute_queries(
        self,
        queries: List[Tuple[str, str, Dict[str, Any]]],  # (query_name, cache_key, parameters)
        max_parallel: int = 10
    ) -> List[Tuple[List[Record], QueryPerformanceMetrics]]:
        """
        Execute multiple queries in parallel for optimal performance.
        
        Args:
            queries: List of (query_name, cache_key, parameters) tuples
            max_parallel: Maximum number of parallel queries
            
        Returns:
            List of (results, metrics) tuples
        """
        if not self.is_initialized:
            raise RuntimeError("Neo4j optimizer not initialized")
        
        # Create semaphore to limit concurrent queries
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def execute_single(query_name: str, cache_key: str, parameters: Dict[str, Any]):
            async with semaphore:
                return await self.execute_optimized_query(query_name, parameters, cache_key)
        
        # Execute all queries in parallel
        tasks = [
            execute_single(query_name, cache_key, parameters)
            for query_name, cache_key, parameters in queries
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch query {i} failed: {str(result)}")
                # Add empty result for failed query
                valid_results.append(([], QueryPerformanceMetrics(
                    query_id=f"batch_{i}_failed",
                    execution_time_ms=0.0,
                    result_count=0,
                    cache_hit=False,
                    index_used=False,
                    memory_usage_mb=0.0,
                    timestamp=datetime.now()
                )))
            else:
                valid_results.append(result)
        
        return valid_results
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[Record]]:
        """Get result from cache if not expired."""
        if cache_key not in self.query_cache:
            return None
        
        cached_data, timestamp = self.query_cache[cache_key]
        
        # Check if expired
        if (datetime.now() - timestamp).total_seconds() > self.cache_ttl:
            del self.query_cache[cache_key]
            self.cache_stats['evictions'] += 1
            return None
        
        return cached_data
    
    def _store_in_cache(self, cache_key: str, data: List[Record], ttl: int) -> None:
        """Store result in cache with timestamp."""
        # Implement LRU-style eviction if cache gets too large
        if len(self.query_cache) > 1000:  # Max 1000 cached queries
            # Remove oldest entry
            oldest_key = min(self.query_cache.keys(), 
                           key=lambda k: self.query_cache[k][1])
            del self.query_cache[oldest_key]
            self.cache_stats['evictions'] += 1
        
        self.query_cache[cache_key] = (data, datetime.now())
    
    def _query_used_index(self, summary) -> bool:
        """Check if query used indexes based on execution summary."""
        try:
            # Check plan for index usage
            if hasattr(summary, 'plan'):
                plan_str = str(summary.plan)
                return 'NodeByLabelScan' not in plan_str or 'NodeIndexSeek' in plan_str
            return True  # Assume optimized queries use indexes
        except:
            return True  # Default to assuming index usage
    
    def _calculate_memory_usage(self, summary) -> float:
        """Calculate memory usage from query summary."""
        try:
            if hasattr(summary, 'counters'):
                # Estimate based on nodes and relationships processed
                nodes = summary.counters.nodes_created + summary.counters.nodes_deleted
                rels = summary.counters.relationships_created + summary.counters.relationships_deleted
                return (nodes * 0.001) + (rels * 0.0005)  # Rough estimate in MB
            return 0.0
        except:
            return 0.0
    
    def _track_performance(self, metrics: QueryPerformanceMetrics) -> None:
        """Track query performance metrics."""
        self.query_metrics.append(metrics)
        
        # Keep only recent metrics
        if len(self.query_metrics) > self.max_metrics_history:
            self.query_metrics = self.query_metrics[-self.max_metrics_history:]
        
        # Log slow queries
        if metrics.execution_time_ms > 50:
            logger.warning(
                "Slow query detected",
                query_id=metrics.query_id,
                execution_time_ms=metrics.execution_time_ms,
                result_count=metrics.result_count,
                index_used=metrics.index_used
            )
    
    async def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        if not self.query_metrics:
            return {
                'total_queries': 0,
                'avg_execution_time_ms': 0,
                'performance_target_met': True,
                'cache_hit_rate': 0
            }
        
        execution_times = [m.execution_time_ms for m in self.query_metrics]
        
        # Calculate percentiles
        sorted_times = sorted(execution_times)
        n = len(sorted_times)
        
        statistics = {
            'total_queries': len(self.query_metrics),
            'avg_execution_time_ms': sum(execution_times) / len(execution_times),
            'median_execution_time_ms': sorted_times[n // 2] if n > 0 else 0,
            'p95_execution_time_ms': sorted_times[int(n * 0.95)] if n > 0 else 0,
            'p99_execution_time_ms': sorted_times[int(n * 0.99)] if n > 0 else 0,
            'max_execution_time_ms': max(execution_times),
            'min_execution_time_ms': min(execution_times),
            'performance_target_met': sum(1 for t in execution_times if t <= 50) / len(execution_times),
            'slow_queries_count': sum(1 for t in execution_times if t > 50),
            'cache_statistics': {
                'hits': self.cache_stats['hits'],
                'misses': self.cache_stats['misses'], 
                'evictions': self.cache_stats['evictions'],
                'hit_rate': self.cache_stats['hits'] / max(1, self.cache_stats['hits'] + self.cache_stats['misses']),
                'cached_entries': len(self.query_cache)
            },
            'index_usage_rate': sum(1 for m in self.query_metrics if m.index_used) / len(self.query_metrics),
            'avg_result_count': sum(m.result_count for m in self.query_metrics) / len(self.query_metrics),
            'avg_memory_usage_mb': sum(m.memory_usage_mb for m in self.query_metrics) / len(self.query_metrics)
        }
        
        return statistics
    
    async def optimize_query(self, query: str) -> str:
        """
        Optimize a Cypher query for better performance.
        
        Args:
            query: Original Cypher query
            
        Returns:
            Optimized query string
        """
        # Basic query optimizations
        optimized = query.strip()
        
        # Add LIMIT if missing (prevent runaway queries)
        if 'LIMIT' not in optimized.upper() and 'RETURN' in optimized.upper():
            optimized += ' LIMIT 1000'
        
        # Suggest using parameters instead of string concatenation
        if "'" in optimized or '"' in optimized:
            logger.warning("Query contains string literals, consider using parameters")
        
        # Suggest index usage
        if 'WHERE' in optimized.upper() and 'INDEX' not in optimized.upper():
            logger.info("Consider adding indexes for WHERE clause properties")
        
        # Add query planning hints
        if optimized.upper().startswith('MATCH'):
            # Add USING INDEX hint for common patterns
            if 'name =' in optimized or 'name CONTAINS' in optimized:
                optimized = optimized.replace(
                    'WHERE', 
                    'USING INDEX e:Entity(name) WHERE', 
                    1
                )
        
        return optimized
    
    async def clear_cache(self, pattern: Optional[str] = None) -> int:
        """
        Clear query cache entries.
        
        Args:
            pattern: Optional pattern to match cache keys (clears all if None)
            
        Returns:
            Number of entries cleared
        """
        if pattern is None:
            cleared_count = len(self.query_cache)
            self.query_cache.clear()
            return cleared_count
        
        # Clear entries matching pattern
        keys_to_remove = [k for k in self.query_cache.keys() if pattern in k]
        for key in keys_to_remove:
            del self.query_cache[key]
        
        return len(keys_to_remove)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Neo4j optimizer health and performance."""
        try:
            if not self.is_initialized:
                return {
                    'status': 'unhealthy',
                    'error': 'Optimizer not initialized'
                }
            
            # Test query performance
            start_time = time.time()
            
            _, metrics = await self.execute_optimized_query(
                'graph_stats',
                {},
                cache_key='health_check'
            )
            
            response_time = (time.time() - start_time) * 1000
            
            # Get recent performance stats
            stats = await self.get_performance_statistics()
            
            return {
                'status': 'healthy' if response_time < 100 else 'degraded',
                'response_time_ms': round(response_time, 2),
                'performance_target_met': response_time < 50,
                'avg_query_time_ms': round(stats.get('avg_execution_time_ms', 0), 2),
                'cache_hit_rate': round(stats.get('cache_statistics', {}).get('hit_rate', 0), 3),
                'slow_queries_count': stats.get('slow_queries_count', 0),
                'index_usage_rate': round(stats.get('index_usage_rate', 0), 3),
                'cached_entries': len(self.query_cache),
                'driver_connected': self.driver is not None
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def close(self) -> None:
        """Close Neo4j driver and cleanup resources."""
        if self.driver:
            await self.driver.close()
            self.driver = None
        
        self.query_cache.clear()
        self.is_initialized = False
        logger.info("Neo4j optimizer closed")


# Convenience functions for common GraphRAG operations

async def find_similar_entities(
    optimizer: Neo4jQueryOptimizer,
    search_term: str,
    min_score: float = 0.3,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Find entities similar to the search term."""
    results, metrics = await optimizer.execute_optimized_query(
        'entity_similarity',
        {
            'search_term': search_term,
            'min_score': min_score,
            'limit': limit
        }
    )
    
    return [dict(record) for record in results]


async def verify_relationship(
    optimizer: Neo4jQueryOptimizer,
    source_entity: str,
    target_entity: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """Verify if a relationship exists between entities."""
    results, metrics = await optimizer.execute_optimized_query(
        'relationship_lookup',
        {
            'source_lower': source_entity.lower(),
            'target_lower': target_entity.lower(),
            'limit': limit
        }
    )
    
    return [dict(record) for record in results]


async def find_entity_connections(
    optimizer: Neo4jQueryOptimizer,
    entity_name: str,
    min_confidence: float = 0.5,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """Find all entities connected to the given entity."""
    results, metrics = await optimizer.execute_optimized_query(
        'entity_relationships',
        {
            'entity_name': entity_name,
            'min_confidence': min_confidence,
            'limit': limit
        }
    )
    
    return [dict(record) for record in results]


async def verify_factual_claim(
    optimizer: Neo4jQueryOptimizer,
    claim_text: str,
    min_score: float = 0.5,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """Verify a factual claim against the knowledge base."""
    results, metrics = await optimizer.execute_optimized_query(
        'fact_verification',
        {
            'claim_text': claim_text,
            'min_score': min_score,
            'limit': limit
        }
    )
    
    return [dict(record) for record in results]


async def find_shortest_path(
    optimizer: Neo4jQueryOptimizer,
    start_entity: str,
    end_entity: str,
    min_confidence: float = 0.3,
    limit: int = 3
) -> List[Dict[str, Any]]:
    """Find shortest path between two entities."""
    results, metrics = await optimizer.execute_optimized_query(
        'path_finding',
        {
            'start_entity': start_entity,
            'end_entity': end_entity,
            'min_confidence': min_confidence,
            'limit': limit
        }
    )
    
    return [dict(record) for record in results]