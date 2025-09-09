"""
Redis connection manager for caching and message queuing.
"""

import asyncio
import logging
from typing import Optional, Any, Dict, List
from urllib.parse import urlparse

import redis.asyncio as aioredis
from redis.asyncio import Redis
from redis.exceptions import ConnectionError, RedisError

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RedisConnectionManager:
    """Manages Redis connections with connection pooling and health monitoring."""
    
    def __init__(self):
        self._redis_client: Optional[Redis] = None
        self._connection_pool: Optional[aioredis.ConnectionPool] = None
        self._health_check_interval = 30  # seconds
        self._max_connections = 10
        self._min_connections = 1
        
    async def initialize(self) -> None:
        """Initialize Redis connection pool."""
        try:
            # Parse Redis URL
            parsed_url = urlparse(settings.redis_url)
            
            # Create connection pool
            self._connection_pool = aioredis.ConnectionPool.from_url(
                settings.redis_url,
                password=settings.redis_password,
                db=settings.redis_db,
                max_connections=self._max_connections,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=self._health_check_interval
            )
            
            # Create Redis client
            self._redis_client = Redis(connection_pool=self._connection_pool)
            
            # Test connection
            await self._redis_client.ping()
            
            logger.info(f"Redis connection established: {parsed_url.hostname}:{parsed_url.port}")
            
        except ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
        except Exception as e:
            logger.error(f"Redis initialization error: {e}")
            raise
    
    async def close(self) -> None:
        """Close Redis connections."""
        if self._redis_client:
            await self._redis_client.close()
            logger.info("Redis connection closed")
        
        if self._connection_pool:
            await self._connection_pool.disconnect()
    
    @property
    def client(self) -> Redis:
        """Get Redis client."""
        if not self._redis_client:
            raise RuntimeError("Redis not initialized. Call initialize() first.")
        return self._redis_client
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform Redis health check."""
        try:
            if not self._redis_client:
                return {"status": "disconnected", "error": "Redis not initialized"}
            
            # Test basic connectivity
            start_time = asyncio.get_event_loop().time()
            pong = await self._redis_client.ping()
            latency = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Get Redis info
            info = await self._redis_client.info()
            
            return {
                "status": "healthy" if pong else "unhealthy",
                "latency_ms": round(latency, 2),
                "redis_version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "unknown"),
                "uptime_seconds": info.get("uptime_in_seconds", 0)
            }
            
        except RedisError as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis performance statistics."""
        try:
            info = await self._redis_client.info()
            stats = await self._redis_client.info("stats")
            
            return {
                "connections": {
                    "total": info.get("total_connections_received", 0),
                    "current": info.get("connected_clients", 0),
                    "rejected": info.get("rejected_connections", 0)
                },
                "memory": {
                    "used": info.get("used_memory", 0),
                    "used_human": info.get("used_memory_human", "0B"),
                    "peak": info.get("used_memory_peak", 0),
                    "peak_human": info.get("used_memory_peak_human", "0B")
                },
                "operations": {
                    "commands_processed": stats.get("total_commands_processed", 0),
                    "instantaneous_ops": stats.get("instantaneous_ops_per_sec", 0),
                    "keyspace_hits": stats.get("keyspace_hits", 0),
                    "keyspace_misses": stats.get("keyspace_misses", 0)
                },
                "persistence": {
                    "rdb_last_save": info.get("rdb_last_save_time", 0),
                    "rdb_changes_since_save": info.get("rdb_changes_since_last_save", 0)
                }
            }
            
        except RedisError as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {"error": str(e)}


class RedisCache:
    """Redis caching utilities."""
    
    def __init__(self, connection_manager: RedisConnectionManager):
        self._connection_manager = connection_manager
    
    @property
    def redis(self) -> Redis:
        """Get Redis client."""
        return self._connection_manager.client
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        try:
            return await self.redis.get(key)
        except RedisError as e:
            logger.error(f"Cache get error for key '{key}': {e}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: str, 
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache with optional TTL."""
        try:
            ttl = ttl or settings.cache_ttl
            result = await self.redis.set(key, value, ex=ttl)
            return bool(result)
        except RedisError as e:
            logger.error(f"Cache set error for key '{key}': {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            result = await self.redis.delete(key)
            return result > 0
        except RedisError as e:
            logger.error(f"Cache delete error for key '{key}': {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            result = await self.redis.exists(key)
            return result > 0
        except RedisError as e:
            logger.error(f"Cache exists error for key '{key}': {e}")
            return False
    
    async def get_many(self, keys: List[str]) -> Dict[str, Optional[str]]:
        """Get multiple values from cache."""
        try:
            values = await self.redis.mget(keys)
            return dict(zip(keys, values))
        except RedisError as e:
            logger.error(f"Cache get_many error: {e}")
            return {key: None for key in keys}
    
    async def set_many(
        self, 
        mapping: Dict[str, str], 
        ttl: Optional[int] = None
    ) -> bool:
        """Set multiple values in cache."""
        try:
            ttl = ttl or settings.cache_ttl
            pipe = self.redis.pipeline()
            
            for key, value in mapping.items():
                pipe.set(key, value, ex=ttl)
            
            results = await pipe.execute()
            return all(results)
        except RedisError as e:
            logger.error(f"Cache set_many error: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        try:
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                return await self.redis.delete(*keys)
            return 0
        except RedisError as e:
            logger.error(f"Cache clear_pattern error for '{pattern}': {e}")
            return 0


# Global Redis connection manager
_redis_manager: Optional[RedisConnectionManager] = None
_redis_cache: Optional[RedisCache] = None


async def initialize_redis() -> None:
    """Initialize global Redis connection."""
    global _redis_manager, _redis_cache
    
    _redis_manager = RedisConnectionManager()
    await _redis_manager.initialize()
    
    _redis_cache = RedisCache(_redis_manager)
    
    logger.info("Redis services initialized")


async def close_redis() -> None:
    """Close global Redis connection."""
    global _redis_manager, _redis_cache
    
    if _redis_manager:
        await _redis_manager.close()
        _redis_manager = None
        _redis_cache = None
    
    logger.info("Redis services closed")


def get_redis_manager() -> RedisConnectionManager:
    """Get Redis connection manager."""
    if not _redis_manager:
        raise RuntimeError("Redis not initialized. Call initialize_redis() first.")
    return _redis_manager


def get_redis_cache() -> RedisCache:
    """Get Redis cache service."""
    if not _redis_cache:
        raise RuntimeError("Redis not initialized. Call initialize_redis() first.")
    return _redis_cache


def get_redis_client() -> Redis:
    """Get Redis client."""
    return get_redis_manager().client