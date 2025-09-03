<template>
  <div class="flow-root">
    <ul role="list" class="-mb-8">
      <li v-for="(activity, activityIdx) in activities" :key="activity.id">
        <div class="relative pb-8">
          <span v-if="activityIdx !== activities.length - 1" class="absolute top-4 left-4 -ml-px h-full w-0.5 bg-gray-200 dark:bg-gray-700" aria-hidden="true" />
          <div class="relative flex space-x-3">
            <div>
              <span class="h-8 w-8 rounded-full flex items-center justify-center ring-8 ring-white dark:ring-gray-800"
                    :class="getActivityIconClass(activity.type)">
                <component :is="getActivityIcon(activity.type)" class="h-4 w-4 text-white" />
              </span>
            </div>
            <div class="flex min-w-0 flex-1 justify-between space-x-4 pt-1.5">
              <div>
                <p class="text-sm text-gray-500 dark:text-gray-400">
                  {{ activity.description }}
                  <a href="#" class="font-medium text-gray-900 dark:text-white">{{ activity.user.name }}</a>
                </p>
              </div>
              <div class="whitespace-nowrap text-right text-sm text-gray-500 dark:text-gray-400">
                <time :datetime="activity.timestamp">{{ formatRelativeTime(activity.timestamp) }}</time>
              </div>
            </div>
          </div>
        </div>
      </li>
    </ul>
  </div>
</template>

<script setup lang="ts">
import type { Activity } from '~/types'

interface Props {
  activities: Activity[]
}

const props = defineProps<Props>()

const getActivityIconClass = (type: string) => {
  switch (type) {
    case 'prd_created':
      return 'bg-green-500'
    case 'prd_updated':
      return 'bg-blue-500'
    case 'prd_approved':
      return 'bg-purple-500'
    case 'project_created':
      return 'bg-indigo-500'
    case 'user_joined':
      return 'bg-yellow-500'
    case 'agent_task':
      return 'bg-red-500'
    default:
      return 'bg-gray-500'
  }
}

const getActivityIcon = (type: string) => {
  // Return SVG icon component based on activity type
  return 'svg'
}

const formatRelativeTime = (timestamp: string) => {
  const date = new Date(timestamp)
  const now = new Date()
  const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000)
  
  if (diffInSeconds < 60) return 'just now'
  if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m ago`
  if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h ago`
  return `${Math.floor(diffInSeconds / 86400)}d ago`
}
</script>