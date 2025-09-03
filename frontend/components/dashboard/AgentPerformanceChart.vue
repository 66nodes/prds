<template>
  <div class="h-64">
    <div class="h-full flex items-center justify-center border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg">
      <div class="text-center">
        <svg class="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
          <path d="M9 19v6a2 2 0 002 2h2a2 2 0 002-2v-6a2 2 0 00-2-2H11a2 2 0 00-2 2zM13 19v6a2 2 0 002 2h2a2 2 0 002-2v-6a2 2 0 00-2-2h-2a2 2 0 00-2 2zM17 19v6a2 2 0 002 2h2a2 2 0 002-2v-6a2 2 0 00-2-2h-2a2 2 0 00-2 2z" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
        <h3 class="mt-2 text-sm font-medium text-gray-900 dark:text-white">Agent Performance Chart</h3>
        <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">Chart.js visualization would be rendered here</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
interface Props {
  data: any
}

const props = defineProps<Props>()

// Chart.js integration would go here
// For now, showing placeholder
</script>