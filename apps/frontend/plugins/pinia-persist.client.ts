import { createPersistedState } from 'pinia-plugin-persistedstate'
import type { PiniaPluginContext } from 'pinia'

export default defineNuxtPlugin((nuxtApp) => {
  // Configure the persisted state plugin with security considerations
  const persistedState = createPersistedState({
    storage: localStorage,
    
    // Custom serializer for secure storage
    serializer: {
      serialize: (state) => {
        try {
          // Remove sensitive data before storing
          const cleanState = JSON.parse(JSON.stringify(state))
          
          // Don't persist sensitive auth tokens in plain text
          if (cleanState.auth?.token) {
            // Keep only non-sensitive user info
            cleanState.auth = {
              ...cleanState.auth,
              token: undefined, // Don't persist raw token
              refreshToken: undefined, // Don't persist refresh token
            }
          }
          
          return JSON.stringify(cleanState)
        } catch (error) {
          console.error('Failed to serialize state:', error)
          return JSON.stringify(state)
        }
      },
      deserialize: (value) => {
        try {
          return JSON.parse(value)
        } catch (error) {
          console.error('Failed to deserialize state:', error)
          return {}
        }
      }
    },
    
    // Configure which stores should be persisted
    key: (id: string) => `pinia-${id}`,
    
    // Handle storage errors gracefully
    beforeRestore: (context: PiniaPluginContext) => {
      console.debug(`Restoring state for store: ${context.store.$id}`)
    },
    
    afterRestore: (context: PiniaPluginContext) => {
      console.debug(`State restored for store: ${context.store.$id}`)
    }
  })

  // Add the plugin to Pinia
  nuxtApp.$pinia.use(persistedState)
})