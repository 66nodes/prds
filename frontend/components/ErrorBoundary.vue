<template>
  <div v-if="hasError" class="min-h-[400px] flex items-center justify-center">
    <div class="text-center p-8 max-w-md">
      <div class="mb-4">
        <svg class="mx-auto h-12 w-12 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
      </div>
      <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">
        {{ title || 'Something went wrong' }}
      </h3>
      <p class="text-sm text-gray-600 dark:text-gray-400 mb-4">
        {{ message || 'An unexpected error occurred. Please try again.' }}
      </p>
      <div class="flex gap-3 justify-center">
        <button
          @click="retry"
          class="px-4 py-2 text-sm font-medium text-white bg-indigo-600 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
        >
          Try Again
        </button>
        <button
          @click="reset"
          class="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 dark:bg-gray-700 dark:text-gray-300 dark:border-gray-600 dark:hover:bg-gray-600"
        >
          Reset
        </button>
      </div>
      <details v-if="isDevelopment" class="mt-4 text-left">
        <summary class="cursor-pointer text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200">
          Error Details
        </summary>
        <pre class="mt-2 p-2 bg-gray-100 dark:bg-gray-800 rounded text-xs overflow-auto">{{ error }}</pre>
      </details>
    </div>
  </div>
  <slot v-else />
</template>

<script setup lang="ts">
import { ref, onErrorCaptured } from 'vue'

interface Props {
  title?: string
  message?: string
  onError?: (error: Error) => void
  onRetry?: () => void
}

const props = defineProps<Props>()
const emit = defineEmits<{
  error: [error: Error]
  retry: []
}>()

const hasError = ref(false)
const error = ref<Error | null>(null)
const isDevelopment = process.env.NODE_ENV === 'development'

onErrorCaptured((err) => {
  hasError.value = true
  error.value = err
  
  // Log error to monitoring service in production
  if (!isDevelopment) {
    console.error('Error boundary caught:', err)
    // TODO: Send to error tracking service (e.g., Sentry)
  }
  
  props.onError?.(err)
  emit('error', err)
  
  return false // Prevent propagation
})

const retry = () => {
  hasError.value = false
  error.value = null
  props.onRetry?.()
  emit('retry')
}

const reset = () => {
  hasError.value = false
  error.value = null
  window.location.reload()
}
</script>