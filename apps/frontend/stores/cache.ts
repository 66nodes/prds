/**
 * Frontend Cache Store - localStorage implementation for browser-side caching
 * Implements TTL-based caching strategy with automatic expiration
 */

import { defineStore } from 'pinia'

interface CacheItem<T = any> {
  data: T
  timestamp: number
  ttl: number // Time to live in milliseconds
  key: string
  namespace?: string
}

interface CacheStats {
  hits: number
  misses: number
  evictions: number
  totalRequests: number
}

export enum CacheNamespace {
  LLM_RESPONSES = 'llm_responses',
  API_RESPONSES = 'api_responses',
  USER_PREFERENCES = 'user_prefs',
  DOCUMENTS = 'documents',
  VALIDATION_RESULTS = 'validation',
  GRAPH_DATA = 'graph_data',
  COMPONENTS = 'components',
  FORMS = 'forms'
}

// TTL constants in milliseconds
export const CacheTTL = {
  SHORT: 5 * 60 * 1000,      // 5 minutes
  MEDIUM: 30 * 60 * 1000,    // 30 minutes
  LONG: 2 * 60 * 60 * 1000,  // 2 hours
  VERY_LONG: 24 * 60 * 60 * 1000, // 24 hours
  
  // Specific cache types
  LLM_RESPONSE: 2 * 60 * 60 * 1000,     // 2 hours
  API_RESPONSE: 15 * 60 * 1000,         // 15 minutes
  USER_PREFS: 24 * 60 * 60 * 1000,      // 24 hours
  DOCUMENTS: 60 * 60 * 1000,            // 1 hour
  VALIDATION: 30 * 60 * 1000,           // 30 minutes
  GRAPH_DATA: 30 * 60 * 1000,           // 30 minutes
  COMPONENTS: 60 * 60 * 1000,           // 1 hour
  FORMS: 5 * 60 * 1000,                 // 5 minutes
} as const

interface CacheState {
  stats: CacheStats
  maxItems: number
  cleanupInterval: number
  lastCleanup: number
}

export const useCacheStore = defineStore('cache', {
  state: (): CacheState => ({
    stats: {
      hits: 0,
      misses: 0,
      evictions: 0,
      totalRequests: 0
    },
    maxItems: 1000, // Maximum cache items
    cleanupInterval: 5 * 60 * 1000, // 5 minutes
    lastCleanup: Date.now()
  }),

  getters: {
    hitRate: (state) => {
      const { hits, totalRequests } = state.stats
      return totalRequests > 0 ? (hits / totalRequests) * 100 : 0
    },
    
    cacheSize: () => {
      if (!process.client) return 0
      return Object.keys(localStorage).filter(key => key.startsWith('cache:')).length
    },

    storageUsage: () => {
      if (!process.client) return 0
      let total = 0
      for (let key in localStorage) {
        if (key.startsWith('cache:')) {
          total += localStorage[key].length
        }
      }
      return Math.round(total / 1024) // Return KB
    }
  },

  actions: {
    /**
     * Store item in cache with TTL
     */
    set<T>(
      namespace: CacheNamespace,
      key: string,
      data: T,
      ttl: number = CacheTTL.MEDIUM
    ): boolean {
      if (!process.client) return false

      try {
        const cacheKey = this._buildCacheKey(namespace, key)
        const item: CacheItem<T> = {
          data,
          timestamp: Date.now(),
          ttl,
          key: cacheKey,
          namespace
        }

        localStorage.setItem(cacheKey, JSON.stringify(item))
        
        // Trigger cleanup if needed
        this._checkAndCleanup()
        
        return true
      } catch (error) {
        console.warn('Cache set failed:', error)
        return false
      }
    },

    /**
     * Get item from cache
     */
    get<T>(namespace: CacheNamespace, key: string): T | null {
      if (!process.client) return null

      this.stats.totalRequests++

      try {
        const cacheKey = this._buildCacheKey(namespace, key)
        const cached = localStorage.getItem(cacheKey)
        
        if (!cached) {
          this.stats.misses++
          return null
        }

        const item: CacheItem<T> = JSON.parse(cached)
        
        // Check if expired
        if (this._isExpired(item)) {
          localStorage.removeItem(cacheKey)
          this.stats.misses++
          this.stats.evictions++
          return null
        }

        this.stats.hits++
        return item.data
      } catch (error) {
        console.warn('Cache get failed:', error)
        this.stats.misses++
        return null
      }
    },

    /**
     * Remove item from cache
     */
    remove(namespace: CacheNamespace, key: string): boolean {
      if (!process.client) return false

      try {
        const cacheKey = this._buildCacheKey(namespace, key)
        localStorage.removeItem(cacheKey)
        return true
      } catch (error) {
        console.warn('Cache remove failed:', error)
        return false
      }
    },

    /**
     * Clear all cache items in a namespace
     */
    clearNamespace(namespace: CacheNamespace): number {
      if (!process.client) return 0

      const prefix = `cache:${namespace}:`
      const keysToRemove: string[] = []
      
      for (let key in localStorage) {
        if (key.startsWith(prefix)) {
          keysToRemove.push(key)
        }
      }

      keysToRemove.forEach(key => localStorage.removeItem(key))
      return keysToRemove.length
    },

    /**
     * Clear all cache items
     */
    clearAll(): number {
      if (!process.client) return 0

      const keysToRemove: string[] = []
      
      for (let key in localStorage) {
        if (key.startsWith('cache:')) {
          keysToRemove.push(key)
        }
      }

      keysToRemove.forEach(key => localStorage.removeItem(key))
      
      // Reset stats
      this.stats = {
        hits: 0,
        misses: 0,
        evictions: 0,
        totalRequests: 0
      }

      return keysToRemove.length
    },

    /**
     * Get or set cached data with fallback
     */
    async getOrSet<T>(
      namespace: CacheNamespace,
      key: string,
      fetchFn: () => Promise<T>,
      ttl: number = CacheTTL.MEDIUM
    ): Promise<T> {
      // Try to get from cache first
      const cached = this.get<T>(namespace, key)
      if (cached !== null) {
        return cached
      }

      // Fetch data
      const data = await fetchFn()
      
      // Store in cache
      this.set(namespace, key, data, ttl)
      
      return data
    },

    /**
     * Cache LLM responses
     */
    cacheLLMResponse(
      model: string,
      prompt: string,
      response: any,
      ttl: number = CacheTTL.LLM_RESPONSE
    ): boolean {
      const key = this._hashString(`${model}:${prompt}`)
      return this.set(CacheNamespace.LLM_RESPONSES, key, response, ttl)
    },

    /**
     * Get cached LLM response
     */
    getLLMResponse(model: string, prompt: string): any {
      const key = this._hashString(`${model}:${prompt}`)
      return this.get(CacheNamespace.LLM_RESPONSES, key)
    },

    /**
     * Cache API responses
     */
    cacheAPIResponse(
      endpoint: string,
      params: Record<string, any>,
      response: any,
      ttl: number = CacheTTL.API_RESPONSE
    ): boolean {
      const key = this._hashString(`${endpoint}:${JSON.stringify(params)}`)
      return this.set(CacheNamespace.API_RESPONSES, key, response, ttl)
    },

    /**
     * Get cached API response
     */
    getAPIResponse(endpoint: string, params: Record<string, any>): any {
      const key = this._hashString(`${endpoint}:${JSON.stringify(params)}`)
      return this.get(CacheNamespace.API_RESPONSES, key)
    },

    /**
     * Cache user preferences
     */
    cacheUserPreferences(userId: string, preferences: any): boolean {
      return this.set(
        CacheNamespace.USER_PREFERENCES,
        userId,
        preferences,
        CacheTTL.USER_PREFS
      )
    },

    /**
     * Get cached user preferences
     */
    getUserPreferences(userId: string): any {
      return this.get(CacheNamespace.USER_PREFERENCES, userId)
    },

    /**
     * Cache validation results
     */
    cacheValidationResult(
      contentHash: string,
      result: any,
      ttl: number = CacheTTL.VALIDATION
    ): boolean {
      return this.set(CacheNamespace.VALIDATION_RESULTS, contentHash, result, ttl)
    },

    /**
     * Get cached validation result
     */
    getValidationResult(contentHash: string): any {
      return this.get(CacheNamespace.VALIDATION_RESULTS, contentHash)
    },

    /**
     * Manual cleanup of expired items
     */
    cleanup(): number {
      if (!process.client) return 0

      let removed = 0
      const now = Date.now()

      for (let key in localStorage) {
        if (!key.startsWith('cache:')) continue

        try {
          const item: CacheItem = JSON.parse(localStorage[key])
          
          if (this._isExpired(item)) {
            localStorage.removeItem(key)
            removed++
          }
        } catch (error) {
          // Remove corrupted cache entries
          localStorage.removeItem(key)
          removed++
        }
      }

      this.stats.evictions += removed
      this.lastCleanup = now
      
      return removed
    },

    /**
     * Get cache statistics
     */
    getStats(): CacheStats & { hitRate: number; size: number; storageUsage: number } {
      return {
        ...this.stats,
        hitRate: this.hitRate,
        size: this.cacheSize,
        storageUsage: this.storageUsage
      }
    },

    // Private methods
    _buildCacheKey(namespace: CacheNamespace, key: string): string {
      return `cache:${namespace}:${key}`
    },

    _isExpired(item: CacheItem): boolean {
      return Date.now() > (item.timestamp + item.ttl)
    },

    _checkAndCleanup(): void {
      const now = Date.now()
      
      if (now - this.lastCleanup > this.cleanupInterval) {
        // Run cleanup in next tick to avoid blocking
        setTimeout(() => this.cleanup(), 0)
      }
      
      // Check if we're over the max items limit
      if (this.cacheSize > this.maxItems) {
        this._evictOldest(Math.floor(this.maxItems * 0.1)) // Remove 10%
      }
    },

    _evictOldest(count: number): void {
      if (!process.client) return

      const items: Array<{ key: string; timestamp: number }> = []
      
      for (let key in localStorage) {
        if (!key.startsWith('cache:')) continue
        
        try {
          const item: CacheItem = JSON.parse(localStorage[key])
          items.push({ key, timestamp: item.timestamp })
        } catch (error) {
          // Remove corrupted entries
          localStorage.removeItem(key)
        }
      }

      // Sort by timestamp (oldest first)
      items.sort((a, b) => a.timestamp - b.timestamp)
      
      // Remove oldest items
      for (let i = 0; i < Math.min(count, items.length); i++) {
        localStorage.removeItem(items[i].key)
        this.stats.evictions++
      }
    },

    _hashString(str: string): string {
      let hash = 0
      for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i)
        hash = ((hash << 5) - hash) + char
        hash = hash & hash // Convert to 32-bit integer
      }
      return Math.abs(hash).toString(16)
    }
  },

  persist: {
    enabled: true,
    strategies: [
      {
        key: 'cache-stats',
        storage: process.client ? localStorage : undefined,
        paths: ['stats', 'lastCleanup'], // Persist stats across sessions
      }
    ]
  }
})

// Cache decorator for Pinia actions
export function cached<T>(
  namespace: CacheNamespace,
  ttl: number = CacheTTL.MEDIUM
) {
  return function (
    target: any,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ) {
    const originalMethod = descriptor.value

    descriptor.value = async function (...args: any[]): Promise<T> {
      const cacheStore = useCacheStore()
      const key = `${propertyKey}:${JSON.stringify(args)}`
      
      return cacheStore.getOrSet(
        namespace,
        key,
        () => originalMethod.apply(this, args),
        ttl
      )
    }

    return descriptor
  }
}