/**
 * Frontend Cache Composable - Easy-to-use caching utilities for Vue components
 */

import type { Ref } from 'vue'
import { useCacheStore, CacheNamespace, CacheTTL } from '~/stores/cache'

export interface CacheOptions {
  ttl?: number
  namespace?: CacheNamespace
  key?: string
  enabled?: boolean
}

export interface CachedAsyncOptions<T> extends CacheOptions {
  onError?: (error: Error) => void
  onSuccess?: (data: T) => void
  transform?: (data: any) => T
  immediate?: boolean
}

export const useCache = () => {
  const cacheStore = useCacheStore()

  /**
   * Basic cache operations
   */
  const cache = {
    set: <T>(namespace: CacheNamespace, key: string, data: T, ttl?: number) =>
      cacheStore.set(namespace, key, data, ttl),
    
    get: <T>(namespace: CacheNamespace, key: string): T | null =>
      cacheStore.get<T>(namespace, key),
    
    remove: (namespace: CacheNamespace, key: string) =>
      cacheStore.remove(namespace, key),
    
    clear: (namespace?: CacheNamespace) =>
      namespace ? cacheStore.clearNamespace(namespace) : cacheStore.clearAll(),
    
    getOrSet: <T>(
      namespace: CacheNamespace,
      key: string,
      fetchFn: () => Promise<T>,
      ttl?: number
    ) => cacheStore.getOrSet(namespace, key, fetchFn, ttl)
  }

  /**
   * Cache statistics
   */
  const stats = computed(() => cacheStore.getStats())

  /**
   * Cache health check
   */
  const isHealthy = computed(() => {
    const currentStats = stats.value
    return currentStats.hitRate > 20 && currentStats.storageUsage < 5000 // 5MB limit
  })

  return {
    cache,
    stats,
    isHealthy,
    
    // Direct store access for advanced usage
    store: cacheStore
  }
}

/**
 * Cached API composable - wraps API calls with automatic caching
 */
export const useCachedAPI = <T = any>(options: CachedAsyncOptions<T> = {}) => {
  const {
    ttl = CacheTTL.API_RESPONSE,
    namespace = CacheNamespace.API_RESPONSES,
    enabled = true,
    onError,
    onSuccess,
    transform,
    immediate = true
  } = options

  const { cache } = useCache()
  const { $api } = useNuxtApp()

  const data: Ref<T | null> = ref(null)
  const error: Ref<Error | null> = ref(null)
  const pending = ref(false)
  const cached = ref(false)

  const execute = async (
    endpoint: string,
    params: Record<string, any> = {},
    method: 'GET' | 'POST' | 'PUT' | 'DELETE' = 'GET'
  ): Promise<T | null> => {
    if (!enabled) {
      // Direct API call without caching
      try {
        pending.value = true
        error.value = null
        
        const response = await $api[method.toLowerCase()](endpoint, params)
        const result = transform ? transform(response) : response
        
        data.value = result
        onSuccess?.(result)
        
        return result
      } catch (err) {
        error.value = err as Error
        onError?.(err as Error)
        return null
      } finally {
        pending.value = false
      }
    }

    try {
      pending.value = true
      error.value = null
      cached.value = false

      const cacheKey = `${method}:${endpoint}:${JSON.stringify(params)}`
      
      const result = await cache.getOrSet(
        namespace,
        cacheKey,
        async () => {
          const response = await $api[method.toLowerCase()](endpoint, params)
          return transform ? transform(response) : response
        },
        ttl
      )

      // Check if this was a cache hit
      cached.value = cache.get(namespace, cacheKey) !== null

      data.value = result
      onSuccess?.(result)
      
      return result
    } catch (err) {
      error.value = err as Error
      onError?.(err as Error)
      return null
    } finally {
      pending.value = false
    }
  }

  const refresh = async (
    endpoint: string,
    params: Record<string, any> = {},
    method: 'GET' | 'POST' | 'PUT' | 'DELETE' = 'GET'
  ) => {
    // Clear cache first
    const cacheKey = `${method}:${endpoint}:${JSON.stringify(params)}`
    cache.remove(namespace, cacheKey)
    
    // Then execute fresh request
    return execute(endpoint, params, method)
  }

  return {
    data: readonly(data),
    error: readonly(error),
    pending: readonly(pending),
    cached: readonly(cached),
    execute,
    refresh
  }
}

/**
 * Cached LLM responses composable
 */
export const useCachedLLM = (options: Omit<CachedAsyncOptions<any>, 'namespace'> = {}) => {
  const {
    ttl = CacheTTL.LLM_RESPONSE,
    enabled = true,
    onError,
    onSuccess,
    transform
  } = options

  const cacheStore = useCacheStore()
  
  const data = ref(null)
  const error = ref(null)
  const pending = ref(false)
  const cached = ref(false)

  const generate = async (
    model: string,
    prompt: string,
    params: Record<string, any> = {}
  ) => {
    if (!enabled) {
      // Direct LLM call without caching
      try {
        pending.value = true
        error.value = null
        
        const { $api } = useNuxtApp()
        const response = await $api.post('/api/v1/llm/generate', {
          model,
          prompt,
          ...params
        })
        
        const result = transform ? transform(response) : response
        data.value = result
        onSuccess?.(result)
        
        return result
      } catch (err) {
        error.value = err
        onError?.(err as Error)
        return null
      } finally {
        pending.value = false
      }
    }

    try {
      pending.value = true
      error.value = null

      // Check cache first
      const cachedResponse = cacheStore.getLLMResponse(model, prompt)
      if (cachedResponse) {
        cached.value = true
        data.value = cachedResponse
        onSuccess?.(cachedResponse)
        return cachedResponse
      }

      // Generate new response
      const { $api } = useNuxtApp()
      const response = await $api.post('/api/v1/llm/generate', {
        model,
        prompt,
        ...params
      })

      const result = transform ? transform(response) : response
      
      // Cache the response
      cacheStore.cacheLLMResponse(model, prompt, result, ttl)
      
      cached.value = false
      data.value = result
      onSuccess?.(result)
      
      return result
    } catch (err) {
      error.value = err
      onError?.(err as Error)
      return null
    } finally {
      pending.value = false
    }
  }

  return {
    data: readonly(data),
    error: readonly(error),
    pending: readonly(pending),
    cached: readonly(cached),
    generate
  }
}

/**
 * Cached form data composable
 */
export const useCachedForm = <T extends Record<string, any>>(
  formId: string,
  initialData: T,
  options: { ttl?: number; enabled?: boolean } = {}
) => {
  const { ttl = CacheTTL.FORMS, enabled = true } = options
  const { cache } = useCache()

  const formData = ref<T>({ ...initialData })
  const isDirty = ref(false)

  // Load from cache on initialization
  onMounted(() => {
    if (enabled) {
      const cached = cache.get<T>(CacheNamespace.FORMS, formId)
      if (cached) {
        formData.value = { ...cached }
      }
    }
  })

  // Auto-save to cache when form data changes
  watch(
    formData,
    (newData) => {
      isDirty.value = true
      
      if (enabled) {
        cache.set(CacheNamespace.FORMS, formId, newData, ttl)
      }
    },
    { deep: true }
  )

  const clearCache = () => {
    cache.remove(CacheNamespace.FORMS, formId)
    formData.value = { ...initialData }
    isDirty.value = false
  }

  const resetForm = () => {
    formData.value = { ...initialData }
    isDirty.value = false
  }

  return {
    formData,
    isDirty: readonly(isDirty),
    clearCache,
    resetForm
  }
}

/**
 * User preferences caching composable
 */
export const useCachedUserPreferences = () => {
  const { user } = useAuth()
  const cacheStore = useCacheStore()

  const preferences = ref<Record<string, any>>({})
  const loading = ref(false)

  const loadPreferences = async () => {
    if (!user.value?.id) return

    loading.value = true
    
    try {
      // Try cache first
      const cached = cacheStore.getUserPreferences(user.value.id)
      if (cached) {
        preferences.value = cached
        return cached
      }

      // Fetch from API
      const { $api } = useNuxtApp()
      const response = await $api.get('/api/v1/user/preferences')
      
      preferences.value = response
      
      // Cache the preferences
      cacheStore.cacheUserPreferences(user.value.id, response)
      
      return response
    } catch (error) {
      console.error('Failed to load user preferences:', error)
      return {}
    } finally {
      loading.value = false
    }
  }

  const updatePreferences = async (updates: Record<string, any>) => {
    if (!user.value?.id) return

    try {
      loading.value = true
      
      const { $api } = useNuxtApp()
      const response = await $api.put('/api/v1/user/preferences', updates)
      
      preferences.value = { ...preferences.value, ...response }
      
      // Update cache
      cacheStore.cacheUserPreferences(user.value.id, preferences.value)
      
      return response
    } catch (error) {
      console.error('Failed to update user preferences:', error)
      throw error
    } finally {
      loading.value = false
    }
  }

  // Load preferences on mount
  onMounted(() => {
    if (user.value?.id) {
      loadPreferences()
    }
  })

  // Watch for user changes
  watch(
    () => user.value?.id,
    (userId) => {
      if (userId) {
        loadPreferences()
      } else {
        preferences.value = {}
      }
    }
  )

  return {
    preferences: readonly(preferences),
    loading: readonly(loading),
    loadPreferences,
    updatePreferences
  }
}