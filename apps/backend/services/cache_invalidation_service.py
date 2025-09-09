"""
Cache Invalidation Service for managing cache lifecycle and TTL policies.

Handles:
- Automatic TTL-based expiration
- Manual invalidation triggers
- Dependency-based invalidation
- Namespace-specific invalidation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from enum import Enum
import json

from pydantic import BaseModel

from ..core.redis import get_redis_cache
from ..core.config import get_settings
from .cache_service import CacheNamespace, CacheTTL, get_cache_service

logger = logging.getLogger(__name__)
settings = get_settings()


class InvalidationTrigger(str, Enum):
    """Types of cache invalidation triggers."""
    TTL_EXPIRY = "ttl_expiry"
    MANUAL = "manual"
    DATA_UPDATE = "data_update"
    DEPENDENCY = "dependency"
    SCHEDULED = "scheduled"
    ERROR = "error"


class InvalidationPolicy(BaseModel):
    """Cache invalidation policy configuration."""
    namespace: CacheNamespace
    ttl_seconds: int
    invalidate_on_update: bool = True
    invalidate_dependencies: bool = False
    max_age_seconds: Optional[int] = None
    refresh_on_access: bool = False
    priority: int = 5  # 1-10, higher = more important


class InvalidationEvent(BaseModel):
    """Record of cache invalidation event."""
    event_id: str
    namespace: CacheNamespace
    trigger: InvalidationTrigger
    keys_invalidated: int
    timestamp: datetime
    reason: Optional[str] = None
    metadata: Dict[str, Any] = {}


class CacheInvalidationService:
    """
    Service for managing cache invalidation with configurable TTL policies
    and dependency tracking.
    """
    
    def __init__(self):
        self._cache_service = get_cache_service()
        self._redis_cache = None
        self._policies: Dict[CacheNamespace, InvalidationPolicy] = {}
        self._dependencies: Dict[str, Set[str]] = {}
        self._invalidation_history: List[InvalidationEvent] = []
        self._max_history = 1000
        self._background_tasks: List[asyncio.Task] = []
        self._running = False
        
        # Default policies for each namespace
        self._setup_default_policies()
    
    def _setup_default_policies(self) -> None:
        """Set up default invalidation policies for each namespace."""
        self._policies = {
            CacheNamespace.LLM_RESPONSES: InvalidationPolicy(
                namespace=CacheNamespace.LLM_RESPONSES,
                ttl_seconds=CacheTTL.LLM_RESPONSE,
                invalidate_on_update=False,
                max_age_seconds=86400,  # 24 hours max
                priority=3
            ),
            CacheNamespace.GRAPHRAG_VALIDATION: InvalidationPolicy(
                namespace=CacheNamespace.GRAPHRAG_VALIDATION,
                ttl_seconds=CacheTTL.GRAPHRAG_RESULT,
                invalidate_on_update=True,
                invalidate_dependencies=True,
                max_age_seconds=7200,  # 2 hours max
                priority=8
            ),
            CacheNamespace.GRAPH_DATA: InvalidationPolicy(
                namespace=CacheNamespace.GRAPH_DATA,
                ttl_seconds=CacheTTL.GRAPH_DATA,
                invalidate_on_update=True,
                invalidate_dependencies=True,
                max_age_seconds=3600,  # 1 hour max
                refresh_on_access=True,
                priority=7
            ),
            CacheNamespace.USER_SESSION: InvalidationPolicy(
                namespace=CacheNamespace.USER_SESSION,
                ttl_seconds=CacheTTL.SESSION_DATA,
                invalidate_on_update=True,
                max_age_seconds=1800,  # 30 minutes max
                refresh_on_access=True,
                priority=9
            ),
            CacheNamespace.API_RESPONSES: InvalidationPolicy(
                namespace=CacheNamespace.API_RESPONSES,
                ttl_seconds=CacheTTL.API_RESPONSE,
                invalidate_on_update=True,
                max_age_seconds=1800,
                priority=5
            ),
            CacheNamespace.DOCUMENT_CACHE: InvalidationPolicy(
                namespace=CacheNamespace.DOCUMENT_CACHE,
                ttl_seconds=CacheTTL.LONG,
                invalidate_on_update=True,
                invalidate_dependencies=True,
                priority=4
            ),
            CacheNamespace.AGENT_STATE: InvalidationPolicy(
                namespace=CacheNamespace.AGENT_STATE,
                ttl_seconds=CacheTTL.SHORT,
                invalidate_on_update=True,
                max_age_seconds=600,  # 10 minutes max
                refresh_on_access=True,
                priority=10
            )
        }
    
    async def initialize(self) -> None:
        """Initialize the cache invalidation service."""
        try:
            self._redis_cache = get_redis_cache()
            
            # Start background monitoring task
            if not self._running:
                self._running = True
                task = asyncio.create_task(self._background_monitor())
                self._background_tasks.append(task)
                
            logger.info("Cache invalidation service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize cache invalidation service: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the cache invalidation service."""
        self._running = False
        
        # Cancel all background tasks
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        logger.info("Cache invalidation service shutdown")
    
    async def invalidate_namespace(
        self,
        namespace: CacheNamespace,
        trigger: InvalidationTrigger = InvalidationTrigger.MANUAL,
        reason: Optional[str] = None
    ) -> int:
        """
        Invalidate all keys in a namespace.
        
        Args:
            namespace: Cache namespace to invalidate
            trigger: Type of invalidation trigger
            reason: Optional reason for invalidation
            
        Returns:
            Number of keys invalidated
        """
        if not self._cache_service.is_available:
            logger.warning("Cache service unavailable for invalidation")
            return 0
        
        try:
            count = await self._cache_service.invalidate_namespace(namespace)
            
            # Record event
            event = InvalidationEvent(
                event_id=f"{namespace.value}_{datetime.utcnow().timestamp()}",
                namespace=namespace,
                trigger=trigger,
                keys_invalidated=count,
                timestamp=datetime.utcnow(),
                reason=reason
            )
            self._record_event(event)
            
            # Invalidate dependencies if configured
            if self._policies[namespace].invalidate_dependencies:
                await self._invalidate_dependencies(namespace)
            
            logger.info(
                f"Invalidated {count} keys in namespace {namespace.value}",
                trigger=trigger.value,
                reason=reason
            )
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to invalidate namespace {namespace.value}: {e}")
            return 0
    
    async def invalidate_key(
        self,
        namespace: CacheNamespace,
        key: str,
        trigger: InvalidationTrigger = InvalidationTrigger.MANUAL,
        reason: Optional[str] = None
    ) -> bool:
        """
        Invalidate a specific cache key.
        
        Args:
            namespace: Cache namespace
            key: Cache key to invalidate
            trigger: Type of invalidation trigger
            reason: Optional reason for invalidation
            
        Returns:
            Success status
        """
        if not self._cache_service.is_available:
            return False
        
        try:
            success = await self._cache_service.delete(namespace, key)
            
            if success:
                # Record event
                event = InvalidationEvent(
                    event_id=f"{key}_{datetime.utcnow().timestamp()}",
                    namespace=namespace,
                    trigger=trigger,
                    keys_invalidated=1,
                    timestamp=datetime.utcnow(),
                    reason=reason,
                    metadata={"key": key}
                )
                self._record_event(event)
                
                # Check and invalidate dependencies
                dep_key = f"{namespace.value}:{key}"
                if dep_key in self._dependencies:
                    for dependent in self._dependencies[dep_key]:
                        dep_namespace, dep_key = dependent.split(":", 1)
                        await self.invalidate_key(
                            CacheNamespace(dep_namespace),
                            dep_key,
                            InvalidationTrigger.DEPENDENCY,
                            f"Dependency on {key}"
                        )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to invalidate key {key}: {e}")
            return False
    
    def register_dependency(
        self,
        source_namespace: CacheNamespace,
        source_key: str,
        dependent_namespace: CacheNamespace,
        dependent_key: str
    ) -> None:
        """
        Register a cache dependency relationship.
        
        Args:
            source_namespace: Source cache namespace
            source_key: Source cache key
            dependent_namespace: Dependent cache namespace
            dependent_key: Dependent cache key
        """
        source = f"{source_namespace.value}:{source_key}"
        dependent = f"{dependent_namespace.value}:{dependent_key}"
        
        if source not in self._dependencies:
            self._dependencies[source] = set()
        
        self._dependencies[source].add(dependent)
        
        logger.debug(f"Registered dependency: {source} -> {dependent}")
    
    async def _invalidate_dependencies(self, namespace: CacheNamespace) -> None:
        """Invalidate all dependencies of a namespace."""
        keys_to_invalidate = []
        
        for key, dependents in self._dependencies.items():
            if key.startswith(f"{namespace.value}:"):
                keys_to_invalidate.extend(dependents)
        
        for dependent in keys_to_invalidate:
            dep_namespace, dep_key = dependent.split(":", 1)
            await self.invalidate_key(
                CacheNamespace(dep_namespace),
                dep_key,
                InvalidationTrigger.DEPENDENCY,
                f"Parent namespace {namespace.value} invalidated"
            )
    
    async def _background_monitor(self) -> None:
        """Background task to monitor and enforce cache policies."""
        while self._running:
            try:
                # Check each namespace policy
                for namespace, policy in self._policies.items():
                    if policy.max_age_seconds:
                        # Implement max age enforcement
                        # This would require tracking creation times
                        # For now, we rely on Redis TTL
                        pass
                
                # Clean up old history
                if len(self._invalidation_history) > self._max_history:
                    self._invalidation_history = self._invalidation_history[-self._max_history:]
                
                # Sleep for monitoring interval
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background monitor error: {e}")
                await asyncio.sleep(10)
    
    def _record_event(self, event: InvalidationEvent) -> None:
        """Record an invalidation event."""
        self._invalidation_history.append(event)
        
        # Trim history if needed
        if len(self._invalidation_history) > self._max_history:
            self._invalidation_history.pop(0)
    
    def get_policy(self, namespace: CacheNamespace) -> InvalidationPolicy:
        """Get invalidation policy for a namespace."""
        return self._policies.get(namespace)
    
    def update_policy(
        self,
        namespace: CacheNamespace,
        policy: InvalidationPolicy
    ) -> None:
        """Update invalidation policy for a namespace."""
        self._policies[namespace] = policy
        logger.info(f"Updated policy for namespace {namespace.value}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get invalidation statistics."""
        stats = {
            "total_events": len(self._invalidation_history),
            "events_by_namespace": {},
            "events_by_trigger": {},
            "recent_events": []
        }
        
        # Count events by namespace and trigger
        for event in self._invalidation_history:
            ns = event.namespace.value
            trigger = event.trigger.value
            
            if ns not in stats["events_by_namespace"]:
                stats["events_by_namespace"][ns] = 0
            stats["events_by_namespace"][ns] += 1
            
            if trigger not in stats["events_by_trigger"]:
                stats["events_by_trigger"][trigger] = 0
            stats["events_by_trigger"][trigger] += 1
        
        # Get recent events
        stats["recent_events"] = [
            {
                "namespace": event.namespace.value,
                "trigger": event.trigger.value,
                "keys_invalidated": event.keys_invalidated,
                "timestamp": event.timestamp.isoformat(),
                "reason": event.reason
            }
            for event in self._invalidation_history[-10:]
        ]
        
        return stats
    
    async def invalidate_on_data_update(
        self,
        namespace: CacheNamespace,
        update_type: str,
        affected_keys: Optional[List[str]] = None
    ) -> int:
        """
        Invalidate cache based on data updates.
        
        Args:
            namespace: Cache namespace affected
            update_type: Type of data update
            affected_keys: Specific keys affected (optional)
            
        Returns:
            Total keys invalidated
        """
        policy = self._policies.get(namespace)
        
        if not policy or not policy.invalidate_on_update:
            return 0
        
        total_invalidated = 0
        
        if affected_keys:
            # Invalidate specific keys
            for key in affected_keys:
                if await self.invalidate_key(
                    namespace,
                    key,
                    InvalidationTrigger.DATA_UPDATE,
                    f"Data update: {update_type}"
                ):
                    total_invalidated += 1
        else:
            # Invalidate entire namespace
            total_invalidated = await self.invalidate_namespace(
                namespace,
                InvalidationTrigger.DATA_UPDATE,
                f"Data update: {update_type}"
            )
        
        return total_invalidated


# Global invalidation service instance
_invalidation_service: Optional[CacheInvalidationService] = None


async def initialize_invalidation_service() -> CacheInvalidationService:
    """Initialize global cache invalidation service."""
    global _invalidation_service
    
    if _invalidation_service is None:
        _invalidation_service = CacheInvalidationService()
        await _invalidation_service.initialize()
    
    return _invalidation_service


def get_invalidation_service() -> CacheInvalidationService:
    """Get cache invalidation service instance."""
    global _invalidation_service
    
    if _invalidation_service is None:
        _invalidation_service = CacheInvalidationService()
    
    return _invalidation_service