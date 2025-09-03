import type { ApiResponse, PaginatedResponse } from '~/types'
import { useApiClient } from './useApiClient'

interface CacheEntry {
  data: any
  timestamp: number
  ttl: number
}

interface PendingRequest {
  promise: Promise<any>
  timestamp: number
}

export const useOptimizedApiClient = () => {
  const apiClient = useApiClient()
  
  // Request cache for deduplication
  const requestCache = new Map<string, CacheEntry>()
  const pendingRequests = new Map<string, PendingRequest>()
  
  // Clear expired cache entries
  const clearExpiredCache = () => {
    const now = Date.now()
    requestCache.forEach((entry, key) => {
      if (now - entry.timestamp > entry.ttl) {
        requestCache.delete(key)
      }
    })
  }
  
  // Generate cache key
  const getCacheKey = (method: string, endpoint: string, params?: any): string => {
    const paramStr = params ? JSON.stringify(params) : ''
    return `${method}:${endpoint}:${paramStr}`
  }
  
  // Check if request is cacheable
  const isCacheable = (method: string): boolean => {
    return method === 'GET'
  }
  
  // Deduplicated GET request
  const get = async <T>(
    endpoint: string,
    options: any = {},
    ttl: number = 5000 // 5 seconds default cache
  ): Promise<T> => {
    const cacheKey = getCacheKey('GET', endpoint, options.params)
    
    // Check cache first
    if (requestCache.has(cacheKey)) {
      const cached = requestCache.get(cacheKey)!
      if (Date.now() - cached.timestamp < cached.ttl) {
        return cached.data as T
      }
      requestCache.delete(cacheKey)
    }
    
    // Check for pending request (deduplication)
    if (pendingRequests.has(cacheKey)) {
      const pending = pendingRequests.get(cacheKey)!
      // Only deduplicate if request was made within last 100ms
      if (Date.now() - pending.timestamp < 100) {
        return pending.promise as Promise<T>
      }
      pendingRequests.delete(cacheKey)
    }
    
    // Make the request
    const requestPromise = apiClient.get<T>(endpoint, options)
    
    // Store as pending
    pendingRequests.set(cacheKey, {
      promise: requestPromise,
      timestamp: Date.now()
    })
    
    try {
      const result = await requestPromise
      
      // Cache successful result
      requestCache.set(cacheKey, {
        data: result,
        timestamp: Date.now(),
        ttl
      })
      
      // Clean up pending
      pendingRequests.delete(cacheKey)
      
      // Periodically clear expired cache
      if (requestCache.size > 100) {
        clearExpiredCache()
      }
      
      return result
    } catch (error) {
      // Clean up pending on error
      pendingRequests.delete(cacheKey)
      throw error
    }
  }
  
  // Batch multiple GET requests
  const batchGet = async <T>(
    requests: Array<{ endpoint: string; options?: any; ttl?: number }>
  ): Promise<T[]> => {
    return Promise.all(
      requests.map(req => get<T>(req.endpoint, req.options, req.ttl))
    )
  }
  
  // Clear cache for specific endpoint
  const invalidateCache = (endpoint?: string) => {
    if (endpoint) {
      // Clear all cache entries for this endpoint
      requestCache.forEach((_, key) => {
        if (key.includes(endpoint)) {
          requestCache.delete(key)
        }
      })
    } else {
      // Clear all cache
      requestCache.clear()
    }
  }
  
  // Prefetch data
  const prefetch = async <T>(
    endpoint: string,
    options: any = {},
    ttl: number = 30000 // 30 seconds for prefetched data
  ): Promise<void> => {
    await get<T>(endpoint, options, ttl)
  }
  
  return {
    ...apiClient,
    get,
    batchGet,
    invalidateCache,
    prefetch,
    clearCache: () => requestCache.clear(),
    getCacheSize: () => requestCache.size
  }
}