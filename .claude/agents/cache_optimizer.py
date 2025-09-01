#!/usr/bin/env python3
"""
Intelligent Caching Strategy and Performance Optimization
For AI Multi-Agent Orchestration System
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
import pickle
import yaml
from enum import Enum


class CacheLevel(Enum):
    """Cache level priorities"""
    L1_MEMORY = "memory"      # In-memory cache (fastest)
    L2_REDIS = "redis"        # Redis cache (fast, distributed)
    L3_DISK = "disk"          # Local disk cache (persistent)
    L4_DATABASE = "database"  # Database cache (shared, persistent)


class CachePriority(Enum):
    """Cache priority levels"""
    CRITICAL = "critical"     # Always cache, never evict
    HIGH = "high"            # Cache with long TTL
    MEDIUM = "medium"        # Standard cache behavior  
    LOW = "low"              # Short TTL, first to evict


@dataclass
class CacheEntry:
    """Individual cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    ttl_seconds: int = 3600
    priority: CachePriority = CachePriority.MEDIUM
    size_bytes: int = 0
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        
        # Calculate approximate size
        if self.size_bytes == 0:
            try:
                self.size_bytes = len(pickle.dumps(self.value))
            except:
                self.size_bytes = len(str(self.value))
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        if self.priority == CachePriority.CRITICAL:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    @property 
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds"""
        return (datetime.now() - self.created_at).total_seconds()


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    total_size_bytes: int = 0
    avg_access_time_ms: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate"""
        return 1.0 - self.hit_rate


class IntelligentCacheManager:
    """
    Multi-level intelligent cache manager for AI agent orchestration
    Features:
    - Multi-level caching (Memory -> Redis -> Disk -> Database)
    - Intelligent TTL based on usage patterns
    - LRU eviction with priority weighting
    - Performance optimization and monitoring
    - Context-aware caching strategies
    """
    
    def __init__(self, config_path: str = ".claude/agents/cache_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Cache storage layers
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.disk_cache_dir = Path(self.config.get('disk_cache_dir', '.claude/cache'))
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.stats = CacheStats()
        self.access_patterns: Dict[str, List[datetime]] = {}
        
        # Configuration
        self.max_memory_size = self.config.get('max_memory_size_mb', 100) * 1024 * 1024
        self.max_disk_size = self.config.get('max_disk_size_mb', 1000) * 1024 * 1024
        self.default_ttl = self.config.get('default_ttl_seconds', 3600)
        
        # Setup logging
        self.setup_logging()
        
        # Background tasks
        self._cleanup_task = None
        self._start_background_tasks()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load caching configuration"""
        default_config = {
            'max_memory_size_mb': 100,
            'max_disk_size_mb': 1000,
            'default_ttl_seconds': 3600,
            'cleanup_interval_seconds': 300,
            'disk_cache_dir': '.claude/cache',
            'compression_enabled': True,
            'encryption_enabled': False,
            'preload_patterns': [
                'agent_*.yaml',
                'workflow_*.json',
                'context_*.md'
            ]
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    return {**default_config, **config}
            except Exception as e:
                logging.error(f"Failed to load cache config: {e}")
        
        return default_config
    
    def setup_logging(self):
        """Setup cache-specific logging"""
        self.logger = logging.getLogger('IntelligentCacheManager')
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory
        logs_dir = Path('.claude/logs')
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler
        handler = logging.FileHandler(logs_dir / 'cache.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        cleanup_interval = self.config.get('cleanup_interval_seconds', 300)
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup(cleanup_interval))
    
    def stop_background_tasks(self):
        """Stop background tasks"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
    
    async def _periodic_cleanup(self, interval: int):
        """Periodic cache cleanup task"""
        while True:
            try:
                await asyncio.sleep(interval)
                await self._cleanup_expired()
                await self._optimize_memory_usage()
                self._update_access_patterns()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {e}")
    
    def _generate_cache_key(self, key: str, context: Dict[str, Any] = None) -> str:
        """Generate a cache key with context hashing"""
        if context:
            context_hash = hashlib.md5(json.dumps(context, sort_keys=True).encode()).hexdigest()[:8]
            return f"{key}:{context_hash}"
        return key
    
    async def get(self, key: str, context: Dict[str, Any] = None) -> Optional[Any]:
        """
        Get value from cache with multi-level fallback
        Returns None if not found in any cache level
        """
        start_time = time.time()
        cache_key = self._generate_cache_key(key, context)
        
        # Track access pattern
        self._record_access(cache_key)
        
        try:
            # L1: Check memory cache first
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                if not entry.is_expired:
                    entry.accessed_at = datetime.now()
                    entry.access_count += 1
                    self._update_stats_hit(time.time() - start_time)
                    self.logger.debug(f"Cache hit (L1-Memory): {key}")
                    return entry.value
                else:
                    # Remove expired entry
                    del self.memory_cache[cache_key]
                    self.stats.evictions += 1
            
            # L2: Check Redis cache (if configured)
            redis_value = await self._get_from_redis(cache_key)
            if redis_value is not None:
                # Promote to L1 cache
                await self.put(key, redis_value, context=context, level=CacheLevel.L1_MEMORY)
                self._update_stats_hit(time.time() - start_time)
                self.logger.debug(f"Cache hit (L2-Redis): {key}")
                return redis_value
            
            # L3: Check disk cache
            disk_value = await self._get_from_disk(cache_key)
            if disk_value is not None:
                # Promote to higher cache levels
                await self.put(key, disk_value, context=context, level=CacheLevel.L1_MEMORY)
                self._update_stats_hit(time.time() - start_time)
                self.logger.debug(f"Cache hit (L3-Disk): {key}")
                return disk_value
            
            # L4: Database cache (if configured)
            db_value = await self._get_from_database(cache_key)
            if db_value is not None:
                # Promote to all cache levels
                await self.put(key, db_value, context=context)
                self._update_stats_hit(time.time() - start_time)
                self.logger.debug(f"Cache hit (L4-Database): {key}")
                return db_value
            
            # Cache miss
            self._update_stats_miss(time.time() - start_time)
            self.logger.debug(f"Cache miss: {key}")
            return None
            
        except Exception as e:
            self.logger.error(f"Cache get error for {key}: {e}")
            self._update_stats_miss(time.time() - start_time)
            return None
    
    async def put(self, 
                  key: str, 
                  value: Any, 
                  ttl_seconds: Optional[int] = None,
                  priority: CachePriority = CachePriority.MEDIUM,
                  context: Dict[str, Any] = None,
                  level: CacheLevel = None) -> bool:
        """
        Store value in cache with intelligent placement
        """
        try:
            cache_key = self._generate_cache_key(key, context)
            ttl = ttl_seconds or self._calculate_intelligent_ttl(key, value, context)
            
            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                value=value,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                ttl_seconds=ttl,
                priority=priority,
                context=context or {}
            )
            
            # Determine optimal cache level
            if level is None:
                level = self._determine_optimal_cache_level(entry)
            
            # Store in specified level(s)
            success = False
            
            if level == CacheLevel.L1_MEMORY or level is None:
                success = await self._put_in_memory(entry)
            
            if level == CacheLevel.L2_REDIS:
                success = await self._put_in_redis(entry)
            
            if level == CacheLevel.L3_DISK:
                success = await self._put_in_disk(entry)
            
            if level == CacheLevel.L4_DATABASE:
                success = await self._put_in_database(entry)
            
            if success:
                self.logger.debug(f"Cached {key} in {level.value if level else 'auto'}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Cache put error for {key}: {e}")
            return False
    
    async def _put_in_memory(self, entry: CacheEntry) -> bool:
        """Store entry in memory cache with eviction if needed"""
        try:
            # Check if we need to evict entries
            if self._get_memory_usage() + entry.size_bytes > self.max_memory_size:
                await self._evict_from_memory(entry.size_bytes)
            
            self.memory_cache[entry.key] = entry
            self.stats.total_size_bytes += entry.size_bytes
            return True
            
        except Exception as e:
            self.logger.error(f"Memory cache put error: {e}")
            return False
    
    async def _put_in_redis(self, entry: CacheEntry) -> bool:
        """Store entry in Redis cache (placeholder)"""
        # Redis implementation would go here
        self.logger.debug(f"Redis cache put: {entry.key}")
        return True
    
    async def _put_in_disk(self, entry: CacheEntry) -> bool:
        """Store entry in disk cache"""
        try:
            cache_file = self.disk_cache_dir / f"{entry.key}.cache"
            
            # Create cache data
            cache_data = {
                'value': entry.value,
                'metadata': {
                    'created_at': entry.created_at.isoformat(),
                    'ttl_seconds': entry.ttl_seconds,
                    'priority': entry.priority.value,
                    'context': entry.context
                }
            }
            
            # Write to disk with compression if enabled
            if self.config.get('compression_enabled', True):
                import gzip
                with gzip.open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
            else:
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Disk cache put error: {e}")
            return False
    
    async def _put_in_database(self, entry: CacheEntry) -> bool:
        """Store entry in database cache (placeholder)"""
        # Database implementation would go here
        self.logger.debug(f"Database cache put: {entry.key}")
        return True
    
    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get value from Redis cache (placeholder)"""
        # Redis implementation would go here
        return None
    
    async def _get_from_disk(self, key: str) -> Optional[Any]:
        """Get value from disk cache"""
        try:
            cache_file = self.disk_cache_dir / f"{key}.cache"
            
            if not cache_file.exists():
                return None
            
            # Read from disk
            if self.config.get('compression_enabled', True):
                import gzip
                with gzip.open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
            else:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
            
            # Check expiration
            created_at = datetime.fromisoformat(cache_data['metadata']['created_at'])
            ttl = cache_data['metadata']['ttl_seconds']
            
            if datetime.now() > created_at + timedelta(seconds=ttl):
                # Expired, remove file
                cache_file.unlink()
                return None
            
            return cache_data['value']
            
        except Exception as e:
            self.logger.error(f"Disk cache get error: {e}")
            return None
    
    async def _get_from_database(self, key: str) -> Optional[Any]:
        """Get value from database cache (placeholder)"""
        # Database implementation would go here
        return None
    
    def _calculate_intelligent_ttl(self, key: str, value: Any, context: Dict[str, Any] = None) -> int:
        """Calculate intelligent TTL based on key patterns and usage"""
        
        # Base TTL
        ttl = self.default_ttl
        
        # Adjust based on key patterns
        if 'agent' in key.lower():
            ttl = 7200  # 2 hours for agent data
        elif 'workflow' in key.lower():
            ttl = 3600  # 1 hour for workflow data
        elif 'context' in key.lower():
            ttl = 1800  # 30 minutes for context data
        elif 'validation' in key.lower():
            ttl = 300   # 5 minutes for validation results
        
        # Adjust based on access patterns
        if key in self.access_patterns:
            accesses = self.access_patterns[key]
            recent_accesses = [a for a in accesses if a > datetime.now() - timedelta(hours=1)]
            
            if len(recent_accesses) > 10:
                # Frequently accessed, increase TTL
                ttl *= 2
            elif len(recent_accesses) == 0 and len(accesses) > 0:
                # Not accessed recently, decrease TTL
                ttl //= 2
        
        # Adjust based on value size
        try:
            size = len(pickle.dumps(value))
            if size > 1024 * 1024:  # > 1MB
                ttl //= 2  # Shorter TTL for large objects
        except:
            pass
        
        return max(ttl, 60)  # Minimum 1 minute TTL
    
    def _determine_optimal_cache_level(self, entry: CacheEntry) -> CacheLevel:
        """Determine optimal cache level for entry"""
        
        # Critical priority always goes to memory
        if entry.priority == CachePriority.CRITICAL:
            return CacheLevel.L1_MEMORY
        
        # Large objects go to disk
        if entry.size_bytes > 100 * 1024:  # > 100KB
            return CacheLevel.L3_DISK
        
        # Frequently accessed goes to memory
        if entry.key in self.access_patterns:
            recent_accesses = len([
                a for a in self.access_patterns[entry.key] 
                if a > datetime.now() - timedelta(minutes=30)
            ])
            if recent_accesses > 5:
                return CacheLevel.L1_MEMORY
        
        # Short TTL goes to memory for fast access
        if entry.ttl_seconds < 300:  # < 5 minutes
            return CacheLevel.L1_MEMORY
        
        # Default to memory for smaller objects
        if entry.size_bytes < 10 * 1024:  # < 10KB
            return CacheLevel.L1_MEMORY
        
        return CacheLevel.L3_DISK
    
    async def _evict_from_memory(self, needed_bytes: int):
        """Evict entries from memory cache using LRU with priority weighting"""
        if not self.memory_cache:
            return
        
        # Sort entries by eviction score (lower score = evict first)
        entries = list(self.memory_cache.values())
        entries.sort(key=self._calculate_eviction_score)
        
        freed_bytes = 0
        evicted_keys = []
        
        for entry in entries:
            if freed_bytes >= needed_bytes:
                break
            
            # Don't evict critical entries
            if entry.priority == CachePriority.CRITICAL:
                continue
            
            # Evict entry
            freed_bytes += entry.size_bytes
            evicted_keys.append(entry.key)
            
            # Try to preserve in lower cache level
            if entry.priority in [CachePriority.HIGH, CachePriority.MEDIUM]:
                await self._put_in_disk(entry)
        
        # Remove evicted entries
        for key in evicted_keys:
            if key in self.memory_cache:
                del self.memory_cache[key]
                self.stats.evictions += 1
        
        self.stats.total_size_bytes -= freed_bytes
        
        if evicted_keys:
            self.logger.debug(f"Evicted {len(evicted_keys)} entries, freed {freed_bytes} bytes")
    
    def _calculate_eviction_score(self, entry: CacheEntry) -> float:
        """Calculate eviction score for LRU with priority weighting"""
        
        # Base score from last access time (older = higher score = evict first)
        time_score = entry.age_seconds / 3600  # Hours since creation
        
        # Priority weighting (lower priority = higher score)
        priority_weights = {
            CachePriority.CRITICAL: 0.0,
            CachePriority.HIGH: 0.25,
            CachePriority.MEDIUM: 0.5,
            CachePriority.LOW: 1.0
        }
        priority_score = priority_weights.get(entry.priority, 0.5)
        
        # Access frequency (less frequent = higher score)
        access_score = 1.0 / max(entry.access_count, 1)
        
        # Size factor (larger = slightly higher score)
        size_score = min(entry.size_bytes / (1024 * 1024), 1.0)  # Normalize to MB
        
        # Combined score
        return (time_score * 0.4 + 
                priority_score * 0.3 + 
                access_score * 0.2 + 
                size_score * 0.1)
    
    def _record_access(self, key: str):
        """Record access pattern for intelligent caching"""
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        self.access_patterns[key].append(datetime.now())
        
        # Keep only recent access history
        cutoff = datetime.now() - timedelta(hours=24)
        self.access_patterns[key] = [
            access for access in self.access_patterns[key]
            if access > cutoff
        ]
    
    def _update_access_patterns(self):
        """Clean up old access patterns"""
        cutoff = datetime.now() - timedelta(hours=24)
        
        for key in list(self.access_patterns.keys()):
            self.access_patterns[key] = [
                access for access in self.access_patterns[key]
                if access > cutoff
            ]
            
            if not self.access_patterns[key]:
                del self.access_patterns[key]
    
    async def _cleanup_expired(self):
        """Clean up expired cache entries"""
        # Memory cache cleanup
        expired_keys = [
            key for key, entry in self.memory_cache.items()
            if entry.is_expired
        ]
        
        for key in expired_keys:
            entry = self.memory_cache[key]
            self.stats.total_size_bytes -= entry.size_bytes
            del self.memory_cache[key]
            self.stats.evictions += 1
        
        # Disk cache cleanup
        await self._cleanup_disk_cache()
        
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
    
    async def _cleanup_disk_cache(self):
        """Clean up expired disk cache files"""
        try:
            for cache_file in self.disk_cache_dir.glob("*.cache"):
                try:
                    # Check if file is expired
                    if self.config.get('compression_enabled', True):
                        import gzip
                        with gzip.open(cache_file, 'rb') as f:
                            cache_data = pickle.load(f)
                    else:
                        with open(cache_file, 'rb') as f:
                            cache_data = pickle.load(f)
                    
                    created_at = datetime.fromisoformat(cache_data['metadata']['created_at'])
                    ttl = cache_data['metadata']['ttl_seconds']
                    
                    if datetime.now() > created_at + timedelta(seconds=ttl):
                        cache_file.unlink()
                        
                except Exception:
                    # If we can't read the file, it's probably corrupted
                    cache_file.unlink()
                    
        except Exception as e:
            self.logger.error(f"Disk cache cleanup error: {e}")
    
    async def _optimize_memory_usage(self):
        """Optimize memory usage by moving entries to appropriate levels"""
        if not self.memory_cache:
            return
        
        current_usage = self._get_memory_usage()
        target_usage = self.max_memory_size * 0.8  # 80% of max
        
        if current_usage <= target_usage:
            return
        
        # Move less critical entries to disk
        entries_to_move = []
        
        for entry in self.memory_cache.values():
            if entry.priority == CachePriority.LOW:
                entries_to_move.append(entry)
            elif entry.priority == CachePriority.MEDIUM and entry.access_count < 2:
                entries_to_move.append(entry)
        
        # Sort by access patterns and move to disk
        entries_to_move.sort(key=lambda e: e.access_count)
        
        moved_bytes = 0
        for entry in entries_to_move:
            if current_usage - moved_bytes <= target_usage:
                break
            
            # Move to disk
            if await self._put_in_disk(entry):
                del self.memory_cache[entry.key]
                moved_bytes += entry.size_bytes
        
        self.stats.total_size_bytes -= moved_bytes
        
        if moved_bytes > 0:
            self.logger.debug(f"Optimized memory usage: moved {moved_bytes} bytes to disk")
    
    def _get_memory_usage(self) -> int:
        """Get current memory cache usage in bytes"""
        return sum(entry.size_bytes for entry in self.memory_cache.values())
    
    def _update_stats_hit(self, access_time: float):
        """Update cache hit statistics"""
        self.stats.hits += 1
        self.stats.total_requests += 1
        
        # Update average access time
        total_time = self.stats.avg_access_time_ms * (self.stats.total_requests - 1)
        self.stats.avg_access_time_ms = (total_time + access_time * 1000) / self.stats.total_requests
    
    def _update_stats_miss(self, access_time: float):
        """Update cache miss statistics"""
        self.stats.misses += 1
        self.stats.total_requests += 1
        
        # Update average access time
        total_time = self.stats.avg_access_time_ms * (self.stats.total_requests - 1)
        self.stats.avg_access_time_ms = (total_time + access_time * 1000) / self.stats.total_requests
    
    async def invalidate(self, key: str, context: Dict[str, Any] = None) -> bool:
        """Invalidate cache entry across all levels"""
        cache_key = self._generate_cache_key(key, context)
        
        invalidated = False
        
        # Memory cache
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]
            self.stats.total_size_bytes -= entry.size_bytes
            del self.memory_cache[cache_key]
            invalidated = True
        
        # Disk cache
        cache_file = self.disk_cache_dir / f"{cache_key}.cache"
        if cache_file.exists():
            cache_file.unlink()
            invalidated = True
        
        # Redis cache (placeholder)
        # await self._invalidate_redis(cache_key)
        
        # Database cache (placeholder)
        # await self._invalidate_database(cache_key)
        
        if invalidated:
            self.logger.debug(f"Invalidated cache: {key}")
        
        return invalidated
    
    async def clear(self, pattern: str = None):
        """Clear cache entries matching pattern"""
        if pattern is None:
            # Clear all
            self.memory_cache.clear()
            self.stats.total_size_bytes = 0
            
            # Clear disk cache
            for cache_file in self.disk_cache_dir.glob("*.cache"):
                cache_file.unlink()
            
            self.logger.info("Cleared all cache entries")
        else:
            # Clear matching pattern
            import fnmatch
            
            matching_keys = [
                key for key in self.memory_cache.keys()
                if fnmatch.fnmatch(key, pattern)
            ]
            
            for key in matching_keys:
                await self.invalidate(key)
            
            self.logger.info(f"Cleared {len(matching_keys)} cache entries matching '{pattern}'")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive cache performance report"""
        current_usage = self._get_memory_usage()
        
        # Disk usage
        disk_usage = sum(
            f.stat().st_size for f in self.disk_cache_dir.glob("*.cache")
            if f.exists()
        )
        
        return {
            'cache_stats': asdict(self.stats),
            'memory_usage': {
                'current_bytes': current_usage,
                'max_bytes': self.max_memory_size,
                'utilization_percent': (current_usage / self.max_memory_size) * 100,
                'total_entries': len(self.memory_cache)
            },
            'disk_usage': {
                'current_bytes': disk_usage,
                'max_bytes': self.max_disk_size,
                'utilization_percent': (disk_usage / self.max_disk_size) * 100,
                'cache_files': len(list(self.disk_cache_dir.glob("*.cache")))
            },
            'access_patterns': {
                'tracked_keys': len(self.access_patterns),
                'most_accessed': self._get_most_accessed_keys(5),
                'least_accessed': self._get_least_accessed_keys(5)
            },
            'performance_metrics': {
                'hit_rate_percent': self.stats.hit_rate * 100,
                'avg_access_time_ms': self.stats.avg_access_time_ms,
                'total_requests': self.stats.total_requests,
                'evictions': self.stats.evictions
            },
            'recommendations': self._generate_performance_recommendations()
        }
    
    def _get_most_accessed_keys(self, limit: int) -> List[Dict[str, Any]]:
        """Get most accessed cache keys"""
        sorted_patterns = sorted(
            self.access_patterns.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        return [
            {'key': key, 'access_count': len(accesses)}
            for key, accesses in sorted_patterns[:limit]
        ]
    
    def _get_least_accessed_keys(self, limit: int) -> List[Dict[str, Any]]:
        """Get least accessed cache keys"""
        sorted_patterns = sorted(
            self.access_patterns.items(),
            key=lambda x: len(x[1])
        )
        
        return [
            {'key': key, 'access_count': len(accesses)}
            for key, accesses in sorted_patterns[:limit]
        ]
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Hit rate analysis
        if self.stats.hit_rate < 0.7:
            recommendations.append("Low cache hit rate (<70%). Consider increasing TTL for frequently accessed data.")
        
        # Memory utilization
        usage_percent = (self._get_memory_usage() / self.max_memory_size) * 100
        if usage_percent > 90:
            recommendations.append("High memory usage (>90%). Consider increasing max_memory_size or moving large objects to disk cache.")
        elif usage_percent < 20:
            recommendations.append("Low memory usage (<20%). Consider reducing max_memory_size to free system resources.")
        
        # Access patterns
        if len(self.access_patterns) > 1000:
            recommendations.append("High number of tracked keys. Consider implementing key pattern grouping.")
        
        # Evictions
        if self.stats.evictions > self.stats.hits * 0.1:
            recommendations.append("High eviction rate. Consider increasing cache size or optimizing TTL values.")
        
        return recommendations


# Configuration helper
async def create_cache_config():
    """Create default cache configuration file"""
    config = {
        'max_memory_size_mb': 100,
        'max_disk_size_mb': 1000,
        'default_ttl_seconds': 3600,
        'cleanup_interval_seconds': 300,
        'disk_cache_dir': '.claude/cache',
        'compression_enabled': True,
        'encryption_enabled': False,
        'preload_patterns': [
            'agent_*.yaml',
            'workflow_*.json',
            'context_*.md',
            'prd_*.json',
            'graphrag_*.cache'
        ],
        'intelligent_ttl_rules': {
            'agent_data': 7200,      # 2 hours
            'workflow_data': 3600,   # 1 hour
            'context_data': 1800,    # 30 minutes
            'validation_results': 300, # 5 minutes
            'temporary_data': 60      # 1 minute
        }
    }
    
    config_path = Path('.claude/agents/cache_config.yaml')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Cache configuration created at: {config_path}")


# CLI Interface
async def main():
    """Main cache manager demonstration"""
    cache_manager = IntelligentCacheManager()
    
    try:
        # Test cache operations
        print("Testing cache operations...")
        
        # Store some test data
        await cache_manager.put("test_key", "test_value", priority=CachePriority.HIGH)
        await cache_manager.put("agent_config", {"model": "claude-3-sonnet"}, ttl_seconds=7200)
        
        # Retrieve data
        value1 = await cache_manager.get("test_key")
        value2 = await cache_manager.get("agent_config")
        
        print(f"Retrieved: {value1}, {value2}")
        
        # Wait a bit for background tasks
        await asyncio.sleep(2)
        
        # Generate performance report
        report = cache_manager.get_performance_report()
        print(json.dumps(report, indent=2, default=str))
        
    finally:
        cache_manager.stop_background_tasks()


if __name__ == '__main__':
    asyncio.run(main())