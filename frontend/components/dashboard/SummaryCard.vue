<template>
  <div class="bg-white dark:bg-gray-800 overflow-hidden shadow rounded-lg">
    <div class="p-5">
      <div class="flex items-center">
        <div class="flex-shrink-0">
          <component :is="iconComponent" class="h-6 w-6 text-gray-400" />
        </div>
        <div class="ml-5 w-0 flex-1">
          <dl>
            <dt class="text-sm font-medium text-gray-500 dark:text-gray-400 truncate">
              {{ title }}
            </dt>
            <dd class="text-lg font-medium text-gray-900 dark:text-white">
              {{ value }}
            </dd>
          </dl>
        </div>
      </div>
    </div>
    <div v-if="change !== undefined" class="bg-gray-50 dark:bg-gray-700 px-5 py-3">
      <div class="text-sm">
        <span v-if="change > 0" class="text-green-600 dark:text-green-400 font-medium">
          +{{ change }}%
        </span>
        <span v-else-if="change < 0" class="text-red-600 dark:text-red-400 font-medium">
          {{ change }}%
        </span>
        <span v-else class="text-gray-500 dark:text-gray-400 font-medium">
          {{ change }}%
        </span>
        <span class="text-gray-500 dark:text-gray-400">
          from last period
        </span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
interface Props {
  title: string
  value: string
  change?: number
  changeType?: 'increase' | 'decrease' | 'neutral'
  icon: string
}

const props = defineProps<Props>()

const iconComponent = computed(() => {
  switch (props.icon) {
    case 'folder':
      return 'svg'
    case 'document':
      return 'svg'
    case 'cpu':
      return 'svg'
    case 'clock':
      return 'svg'
    default:
      return 'svg'
  }
})
</script>