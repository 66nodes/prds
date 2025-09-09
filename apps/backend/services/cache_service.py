"""
Multi-tier caching service for performance optimization.

Implements caching for:
- LLM responses
- GraphRAG validation results
- Frequently accessed graph data
- Query results
"""

import asyncio
import functools
import hashlib
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar, Callable, Union
from functools import wraps

from pydantic import BaseModel

from ..core.redis import get_redis_cache, RedisCache
from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

T = TypeVar('T')


class CacheNamespace(str, Enum):
    """Cache namespaces for different data types."""
    LLM_RESPONSES = "llm"
    GRAPHRAG_VALIDATION = "graphrag"
    GRAPH_DATA = "graph"
    QUERY_RESULTS = "query"
    USER_SESSION = "session"
    API_RESPONSES = "api"
    DOCUMENT_CACHE = "doc"
    AGENT_STATE = "agent"


class CacheTTL(int, Enum):
    """Standard TTL values in seconds."""
    SHORT = 300  # 5 minutes
    MEDIUM = 3600  # 1 hour
    LONG = 86400  # 24 hours
    VERY_LONG = 604800  # 7 days
    
    # Specific TTLs for different data types
    LLM_RESPONSE = 7200  # 2 hours
    GRAPHRAG_RESULT = 3600  # 1 hour
    GRAPH_DATA = 1800  # 30 minutes
    SESSION_DATA = 900  # 15 minutes
    API_RESPONSE = 600  # 10 minutes


class CacheStats(BaseModel):
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    errors: int = 0
    evictions: int = 0
    hit_rate: float = 0.0
    avg_latency_ms: float = 0.0
    last_reset: datetime = datetime.utcnow()


class CacheService:
    """
    Advanced caching service with namespace support, TTL management,
    and performance monitoring.
    """
    
    def __init__(self):
        self._cache: Optional[RedisCache] = None
        self._stats: Dict[str, CacheStats] = {}
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize cache service."""
        try:
            self._cache = get_redis_cache()
            self._initialized = True
            logger.info("Cache service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize cache service: {e}")
            self._initialized = False
            
    @property
    def is_available(self) -> bool:
        """Check if cache service is available."""
        return self._initialized and self._cache is not None
    
    def _generate_key(
        self, 
        namespace: CacheNamespace, 
        identifier: str,
        version: str = "v1"
    ) -> str:
        """
        Generate a namespaced cache key.
        
        Args:
            namespace: Cache namespace
            identifier: Unique identifier
            version: Cache version for invalidation
            
        Returns:
            Formatted cache key
        """
        return f"{namespace.value}:{version}:{identifier}"
    
    def _hash_data(self, data: Any) -> str:
        """
        Generate a hash from data for cache key generation.
        
        Args:
            data: Data to hash
            
        Returns:
            SHA256 hash of the data
        """
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True, default=str)
        else:
            data_str = str(data)
        
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    async def get(
        self,
        namespace: CacheNamespace,
        key: str,
        deserialize: bool = True
    ) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            namespace: Cache namespace
            key: Cache key
            deserialize: Whether to deserialize JSON
            
        Returns:
            Cached value or None
        """
        if not self.is_available:
            return None
            
        try:
            full_key = self._generate_key(namespace, key)
            value = await self._cache.get(full_key)
            
            if value is None:
                self._record_miss(namespace)
                return None
                
            self._record_hit(namespace)
            
            if deserialize and value:
                return json.loads(value)
            return value
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self._record_error(namespace)
            return None
    
    async def set(
        self,
        namespace: CacheNamespace,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialize: bool = True
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            namespace: Cache namespace
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            serialize: Whether to serialize to JSON
            
        Returns:
            Success status
        """
        if not self.is_available:
            return False
            
        try:
            full_key = self._generate_key(namespace, key)
            
            if serialize:
                value = json.dumps(value, default=str)
            
            ttl = ttl or self._get_default_ttl(namespace)
            success = await self._cache.set(full_key, value, ttl)
            
            if success:
                logger.debug(f"Cached {full_key} with TTL {ttl}s")
                
            return success
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            self._record_error(namespace)
            return False
    
    async def delete(
        self,
        namespace: CacheNamespace,
        key: str
    ) -> bool:
        """
        Delete value from cache.
        
        Args:
            namespace: Cache namespace
            key: Cache key
            
        Returns:
            Success status
        """
        if not self.is_available:
            return False
            
        try:
            full_key = self._generate_key(namespace, key)
            return await self._cache.delete(full_key)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def invalidate_namespace(
        self,
        namespace: CacheNamespace
    ) -> int:
        """
        Invalidate all keys in a namespace.
        
        Args:
            namespace: Cache namespace to invalidate
            
        Returns:
            Number of keys deleted
        """
        if not self.is_available:
            return 0
            
        try:
            pattern = f"{namespace.value}:*"
            count = await self._cache.clear_pattern(pattern)
            logger.info(f"Invalidated {count} keys in namespace {namespace.value}")
            return count
        except Exception as e:
            logger.error(f"Namespace invalidation error: {e}")
            return 0
    
    # Specialized caching methods
    
    async def cache_llm_response(
        self,
        prompt: str,
        model: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Cache LLM response with metadata.
        
        Args:
            prompt: LLM prompt
            model: Model identifier
            response: LLM response
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        key_data = {
            "prompt": prompt,
            "model": model,
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens
        }
        
        cache_key = self._hash_data(key_data)
        
        cache_value = {
            "response": response,
            "model": model,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        return await self.set(
            CacheNamespace.LLM_RESPONSES,
            cache_key,
            cache_value,
            ttl=CacheTTL.LLM_RESPONSE
        )
    
    async def get_llm_response(
        self,
        prompt: str,
        model: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached LLM response.
        
        Args:
            prompt: LLM prompt
            model: Model identifier
            
        Returns:
            Cached response or None
        """
        key_data = {
            "prompt": prompt,
            "model": model,
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens
        }
        
        cache_key = self._hash_data(key_data)
        return await self.get(CacheNamespace.LLM_RESPONSES, cache_key)
    
    async def cache_graphrag_validation(
        self,
        content: str,
        project_id: str,
        validation_result: Dict[str, Any]
    ) -> bool:
        """
        Cache GraphRAG validation result.
        
        Args:
            content: Content that was validated
            project_id: Project identifier
            validation_result: Validation result
            
        Returns:
            Success status
        """
        key_data = {
            "content_hash": self._hash_data(content),
            "project_id": project_id
        }
        
        cache_key = self._hash_data(key_data)
        
        cache_value = {
            **validation_result,
            "cached_at": datetime.utcnow().isoformat()
        }
        
        return await self.set(
            CacheNamespace.GRAPHRAG_VALIDATION,
            cache_key,
            cache_value,
            ttl=CacheTTL.GRAPHRAG_RESULT
        )
    
    async def get_graphrag_validation(
        self,
        content: str,
        project_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached GraphRAG validation result.
        
        Args:
            content: Content to validate
            project_id: Project identifier
            
        Returns:
            Cached validation result or None
        """
        key_data = {
            "content_hash": self._hash_data(content),
            "project_id": project_id
        }
        
        cache_key = self._hash_data(key_data)
        return await self.get(CacheNamespace.GRAPHRAG_VALIDATION, cache_key)
    
    async def cache_graph_data(
        self,
        query: str,
        project_id: str,
        data: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache graph query results.
        
        Args:
            query: Graph query
            project_id: Project identifier
            data: Query result data
            ttl: Custom TTL
            
        Returns:
            Success status
        """
        key_data = {
            "query": query,
            "project_id": project_id
        }
        
        cache_key = self._hash_data(key_data)
        
        return await self.set(
            CacheNamespace.GRAPH_DATA,
            cache_key,
            data,
            ttl=ttl or CacheTTL.GRAPH_DATA
        )
    
    async def get_graph_data(
        self,
        query: str,
        project_id: str
    ) -> Optional[Any]:
        """
        Get cached graph query results.
        
        Args:
            query: Graph query
            project_id: Project identifier
            
        Returns:
            Cached data or None
        """
        key_data = {
            "query": query,
            "project_id": project_id
        }
        
        cache_key = self._hash_data(key_data)
        return await self.get(CacheNamespace.GRAPH_DATA, cache_key)
    
    # Statistics and monitoring
    
    def _get_default_ttl(self, namespace: CacheNamespace) -> int:
        """Get default TTL for namespace."""
        ttl_map = {
            CacheNamespace.LLM_RESPONSES: CacheTTL.LLM_RESPONSE,
            CacheNamespace.GRAPHRAG_VALIDATION: CacheTTL.GRAPHRAG_RESULT,
            CacheNamespace.GRAPH_DATA: CacheTTL.GRAPH_DATA,
            CacheNamespace.USER_SESSION: CacheTTL.SESSION_DATA,
            CacheNamespace.API_RESPONSES: CacheTTL.API_RESPONSE,
            CacheNamespace.QUERY_RESULTS: CacheTTL.MEDIUM,
            CacheNamespace.DOCUMENT_CACHE: CacheTTL.LONG,
            CacheNamespace.AGENT_STATE: CacheTTL.SHORT
        }
        return ttl_map.get(namespace, CacheTTL.MEDIUM)
    
    def _record_hit(self, namespace: CacheNamespace) -> None:
        """Record cache hit."""
        if namespace.value not in self._stats:
            self._stats[namespace.value] = CacheStats()
        self._stats[namespace.value].hits += 1
        self._update_hit_rate(namespace.value)
    
    def _record_miss(self, namespace: CacheNamespace) -> None:
        """Record cache miss."""
        if namespace.value not in self._stats:
            self._stats[namespace.value] = CacheStats()
        self._stats[namespace.value].misses += 1
        self._update_hit_rate(namespace.value)
    
    def _record_error(self, namespace: CacheNamespace) -> None:
        """Record cache error."""
        if namespace.value not in self._stats:
            self._stats[namespace.value] = CacheStats()
        self._stats[namespace.value].errors += 1
    
    def _update_hit_rate(self, namespace: str) -> None:
        """Update cache hit rate."""
        stats = self._stats[namespace]
        total = stats.hits + stats.misses
        if total > 0:
            stats.hit_rate = stats.hits / total
    
    def get_stats(self, namespace: Optional[CacheNamespace] = None) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Args:
            namespace: Specific namespace or all if None
            
        Returns:
            Cache statistics
        """
        if namespace:
            stats = self._stats.get(namespace.value, CacheStats())
            return stats.model_dump()
        
        return {
            ns: stats.model_dump() 
            for ns, stats in self._stats.items()
        }
    
    def reset_stats(self, namespace: Optional[CacheNamespace] = None) -> None:
        """Reset cache statistics."""
        if namespace:
            self._stats[namespace.value] = CacheStats()
        else:
            self._stats.clear()


# Decorators for easy caching

def cached(
    namespace: CacheNamespace = CacheNamespace.API_RESPONSES,
    ttl: Optional[int] = None,
    key_prefix: Optional[str] = None
):
    """
    Decorator for caching function results.
    
    Args:
        namespace: Cache namespace
        ttl: Time to live in seconds
        key_prefix: Optional key prefix
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_parts = [key_prefix or func.__name__]
            
            # Add positional arguments
            for arg in args:
                if hasattr(arg, '__dict__'):
                    # Skip complex objects
                    continue
                key_parts.append(str(arg))
            
            # Add keyword arguments
            for k, v in sorted(kwargs.items()):
                if hasattr(v, '__dict__'):
                    continue
                key_parts.append(f"{k}={v}")
            
            cache_key = ":".join(key_parts)
            cache_service = _get_cache_service()
            
            # Try to get from cache
            if cache_service.is_available:
                cached_value = await cache_service.get(namespace, cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_value
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            if cache_service.is_available and result is not None:
                await cache_service.set(namespace, cache_key, result, ttl)
                logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we can't use async cache
            # Just execute the function
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


def invalidate_on_update(namespace: CacheNamespace):
    """
    Decorator to invalidate cache when data is updated.
    
    Args:
        namespace: Cache namespace to invalidate
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Invalidate cache after successful update
            if result:
                cache_service = _get_cache_service()
                if cache_service.is_available:
                    await cache_service.invalidate_namespace(namespace)
                    logger.debug(f"Invalidated {namespace.value} cache after {func.__name__}")
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


# Global cache service instance
_cache_service: Optional[CacheService] = None


async def initialize_cache_service() -> CacheService:
    """Initialize global cache service."""
    global _cache_service
    
    if _cache_service is None:
        _cache_service = CacheService()
        await _cache_service.initialize()
    
    return _cache_service


def _get_cache_service() -> CacheService:
    """Get global cache service instance."""
    global _cache_service
    
    if _cache_service is None:
        _cache_service = CacheService()
    
    return _cache_service


def get_cache_service() -> CacheService:
    """Get cache service for external use."""
    return _get_cache_service()