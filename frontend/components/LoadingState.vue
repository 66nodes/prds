<template>
  <div class="flex items-center justify-center" :class="containerClass">
    <div class="text-center">
      <!-- Spinner -->
      <div class="relative inline-flex">
        <div 
          class="animate-spin rounded-full border-b-2 border-indigo-600"
          :class="spinnerSizeClass"
        />
        <div 
          v-if="showPulse"
          class="absolute inset-0 rounded-full border-2 border-indigo-600 opacity-20 animate-ping"
          :class="spinnerSizeClass"
        />
      </div>
      
      <!-- Text -->
      <div v-if="text" class="mt-4">
        <p class="text-sm font-medium text-gray-900 dark:text-white">
          {{ text }}
        </p>
        <p v-if="subtext" class="mt-1 text-xs text-gray-500 dark:text-gray-400">
          {{ subtext }}
        </p>
      </div>
      
      <!-- Progress Bar -->
      <div v-if="progress !== undefined" class="mt-4 w-48 mx-auto">
        <div class="flex items-center justify-between text-xs text-gray-600 dark:text-gray-400 mb-1">
          <span>Progress</span>
          <span>{{ Math.round(progress) }}%</span>
        </div>
        <div class="w-full bg-gray-200 rounded-full h-1.5 dark:bg-gray-700">
          <div 
            class="bg-indigo-600 h-1.5 rounded-full transition-all duration-300"
            :style="{ width: `${Math.min(100, Math.max(0, progress))}%` }"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface Props {
  text?: string
  subtext?: string
  size?: 'sm' | 'md' | 'lg' | 'xl'
  fullscreen?: boolean
  progress?: number
  showPulse?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  size: 'md',
  fullscreen: false,
  showPulse: false
})

const containerClass = computed(() => {
  if (props.fullscreen) {
    return 'min-h-screen'
  }
  
  const heights = {
    sm: 'min-h-[200px]',
    md: 'min-h-[300px]',
    lg: 'min-h-[400px]',
    xl: 'min-h-[500px]'
  }
  
  return heights[props.size]
})

const spinnerSizeClass = computed(() => {
  const sizes = {
    sm: 'h-8 w-8',
    md: 'h-12 w-12',
    lg: 'h-16 w-16',
    xl: 'h-20 w-20'
  }
  
  return sizes[props.size]
})
</script>