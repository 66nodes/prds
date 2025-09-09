/**
 * Frontend Cache Plugin - Initializes cache store and sets up automatic cleanup
 */

import { useCacheStore } from '~/stores/cache'

export default defineNuxtPlugin(() => {
  const cacheStore = useCacheStore()

  // Initialize cache store on client side
  if (process.client) {
    // Initial cleanup on app start
    cacheStore.cleanup()

    // Set up periodic cleanup every 5 minutes
    const cleanupInterval = setInterval(() => {
      try {
        const removed = cacheStore.cleanup()
        if (removed > 0) {
          console.log(`Cache cleanup: removed ${removed} expired items`)
        }
      } catch (error) {
        console.warn('Cache cleanup failed:', error)
      }
    }, 5 * 60 * 1000) // 5 minutes

    // Set up storage quota monitoring
    const checkStorageQuota = () => {
      if ('storage' in navigator && 'estimate' in navigator.storage) {
        navigator.storage.estimate().then(estimate => {
          const usedMB = (estimate.usage || 0) / (1024 * 1024)
          const quotaMB = (estimate.quota || 0) / (1024 * 1024)
          
          // If using more than 80% of quota, clean up aggressively
          if (usedMB / quotaMB > 0.8) {
            console.warn(`Storage quota usage high: ${usedMB.toFixed(2)}MB / ${quotaMB.toFixed(2)}MB`)
            
            // Clear old cache entries
            const cleared = cacheStore.clearAll()
            console.log(`Emergency cache clear: removed ${cleared} items`)
          }
        }).catch(error => {
          console.warn('Storage estimate failed:', error)
        })
      }
    }

    // Check storage quota every 10 minutes
    const storageCheckInterval = setInterval(checkStorageQuota, 10 * 60 * 1000)

    // Handle page visibility changes - cleanup when page becomes hidden
    let visibilityCleanupTimer: NodeJS.Timeout | null = null
    
    const handleVisibilityChange = () => {
      if (document.hidden) {
        // Start cleanup timer when page is hidden
        visibilityCleanupTimer = setTimeout(() => {
          try {
            const removed = cacheStore.cleanup()
            if (removed > 0) {
              console.log(`Background cache cleanup: removed ${removed} expired items`)
            }
          } catch (error) {
            console.warn('Background cache cleanup failed:', error)
          }
        }, 30000) // Clean up after 30 seconds of being hidden
      } else {
        // Cancel cleanup timer when page becomes visible
        if (visibilityCleanupTimer) {
          clearTimeout(visibilityCleanupTimer)
          visibilityCleanupTimer = null
        }
      }
    }

    document.addEventListener('visibilitychange', handleVisibilityChange)

    // Handle beforeunload - final cleanup
    const handleBeforeUnload = () => {
      try {
        // Quick cleanup of expired items before leaving
        cacheStore.cleanup()
      } catch (error) {
        // Ignore errors during unload
      }
    }

    window.addEventListener('beforeunload', handleBeforeUnload)

    // Clean up on app unmount
    onUnmounted(() => {
      clearInterval(cleanupInterval)
      clearInterval(storageCheckInterval)
      
      if (visibilityCleanupTimer) {
        clearTimeout(visibilityCleanupTimer)
      }
      
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      window.removeEventListener('beforeunload', handleBeforeUnload)
    })

    // Log cache statistics on development
    if (process.dev) {
      const logStats = () => {
        const stats = cacheStore.getStats()
        console.log('Cache Stats:', {
          hitRate: `${stats.hitRate.toFixed(1)}%`,
          size: `${stats.size} items`,
          storage: `${stats.storageUsage}KB`,
          hits: stats.hits,
          misses: stats.misses,
          evictions: stats.evictions
        })
      }

      // Log stats every minute in development
      const statsInterval = setInterval(logStats, 60 * 1000)
      
      onUnmounted(() => {
        clearInterval(statsInterval)
      })
    }

    // Provide global cache access for debugging
    if (process.dev) {
      ;(window as any).__cache = {
        store: cacheStore,
        stats: () => cacheStore.getStats(),
        clear: () => cacheStore.clearAll(),
        cleanup: () => cacheStore.cleanup()
      }
    }
  }

  return {
    provide: {
      cache: cacheStore
    }
  }
})