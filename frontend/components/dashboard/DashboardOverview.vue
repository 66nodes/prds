<template>
  <div class="space-y-6">
    <!-- Summary Cards -->
    <div class="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
      <SummaryCard
        v-for="card in summaryCards"
        :key="card.title"
        :title="card.title"
        :value="card.value"
        :change="card.change"
        :changeType="card.changeType"
        :icon="card.icon"
      />
    </div>

    <!-- Charts Section -->
    <div class="grid grid-cols-1 gap-6 lg:grid-cols-2">
      <!-- Hallucination Rate Chart -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Hallucination Rate Trend
        </h3>
        <HallucinationChart :data="hallucinationData" />
      </div>

      <!-- Agent Performance Chart -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Agent Performance
        </h3>
        <AgentPerformanceChart :data="agentPerformanceData" />
      </div>
    </div>

    <!-- Recent Activity -->
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow">
      <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <h3 class="text-lg font-medium text-gray-900 dark:text-white">
          Recent Activity
        </h3>
      </div>
      <div class="px-6 py-4">
        <ActivityFeed :activities="recentActivities" />
      </div>
    </div>

    <!-- System Status -->
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow">
      <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <h3 class="text-lg font-medium text-gray-900 dark:text-white">
          System Status
        </h3>
      </div>
      <div class="px-6 py-4">
        <SystemStatus :health="systemHealth" :metrics="systemMetrics" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useApiClient } from '~/composables/useApiClient'
import type { DashboardData, Activity, SystemHealth, PerformanceMetrics } from '~/types'

// Components
import SummaryCard from './SummaryCard.vue'
import HallucinationChart from './HallucinationChart.vue'
import AgentPerformanceChart from './AgentPerformanceChart.vue'
import ActivityFeed from './ActivityFeed.vue'
import SystemStatus from './SystemStatus.vue'

const { get } = useApiClient()

// State
const dashboardData = ref<DashboardData | null>(null)
const isLoading = ref(false)
const error = ref<string | null>(null)

// Computed properties
const summaryCards = computed(() => {
  if (!dashboardData.value) {
    return [
      { title: 'Total Projects', value: '0', change: 0, changeType: 'neutral' as const, icon: 'folder' },
      { title: 'Active PRDs', value: '0', change: 0, changeType: 'neutral' as const, icon: 'document' },
      { title: 'Active Agents', value: '0', change: 0, changeType: 'neutral' as const, icon: 'cpu' },
      { title: 'Avg Response Time', value: '0ms', change: 0, changeType: 'neutral' as const, icon: 'clock' }
    ]
  }

  const summary = dashboardData.value.summary
  return [
    {
      title: 'Total Projects',
      value: summary.totalProjects.toString(),
      change: 12,
      changeType: 'increase' as const,
      icon: 'folder'
    },
    {
      title: 'Active PRDs',
      value: summary.activePRDs.toString(),
      change: 8,
      changeType: 'increase' as const,
      icon: 'document'
    },
    {
      title: 'Active Agents',
      value: `${summary.totalAgents}/${summary.totalAgents}`,
      change: 0,
      changeType: 'neutral' as const,
      icon: 'cpu'
    },
    {
      title: 'Avg Response Time',
      value: `${summary.averageResponseTime}ms`,
      change: -15,
      changeType: 'decrease' as const,
      icon: 'clock'
    }
  ]
})

const hallucinationData = computed(() => {
  if (!dashboardData.value) return null
  
  return {
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    datasets: [{
      label: 'Hallucination Rate (%)',
      data: dashboardData.value.performanceMetrics.hallucinationRates.map(r => r * 100),
      borderColor: 'rgb(99, 102, 241)',
      backgroundColor: 'rgba(99, 102, 241, 0.1)',
      tension: 0.4
    }]
  }
})

const agentPerformanceData = computed(() => {
  if (!dashboardData.value) return null
  
  return {
    labels: ['Context Manager', 'Draft Agent', 'Judge Agent', 'Task Executor', 'Validator'],
    datasets: [{
      label: 'Tasks Completed',
      data: [145, 89, 234, 167, 98],
      backgroundColor: [
        'rgba(99, 102, 241, 0.8)',
        'rgba(34, 197, 94, 0.8)',
        'rgba(251, 146, 60, 0.8)',
        'rgba(147, 51, 234, 0.8)',
        'rgba(236, 72, 153, 0.8)'
      ]
    }]
  }
})

const recentActivities = computed(() => {
  return dashboardData.value?.recentActivity || []
})

const systemHealth = computed(() => {
  return dashboardData.value?.summary.systemHealth || 'healthy' as SystemHealth
})

const systemMetrics = computed(() => {
  if (!dashboardData.value) {
    return {
      apiLatency: 0,
      cacheHitRate: 0,
      errorRate: 0,
      agentUtilization: 0
    }
  }
  
  const metrics = dashboardData.value.performanceMetrics
  return {
    apiLatency: metrics.apiLatency[metrics.apiLatency.length - 1] || 0,
    cacheHitRate: metrics.cacheHitRate,
    errorRate: metrics.errorRate,
    agentUtilization: metrics.agentUtilization
  }
})

// Fetch dashboard data
const fetchDashboardData = async () => {
  isLoading.value = true
  error.value = null
  
  try {
    const data = await get<DashboardData>('/dashboard')
    dashboardData.value = data
  } catch (err: any) {
    error.value = err.message || 'Failed to fetch dashboard data'
    console.error('Dashboard fetch error:', err)
  } finally {
    isLoading.value = false
  }
}

// Auto-refresh dashboard data
const startAutoRefresh = () => {
  const interval = setInterval(() => {
    fetchDashboardData()
  }, 30000) // Refresh every 30 seconds
  
  onUnmounted(() => {
    clearInterval(interval)
  })
}

// Lifecycle
onMounted(() => {
  fetchDashboardData()
  startAutoRefresh()
})
</script>