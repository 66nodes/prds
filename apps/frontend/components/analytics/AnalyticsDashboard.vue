<template>
  <div class="space-y-6">
    <div class="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
      <h3
        class="text-lg leading-6 font-medium text-gray-900 dark:text-white mb-4"
      >
        Analytics Overview
      </h3>

      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div class="text-center">
          <div class="text-3xl font-bold text-indigo-600 dark:text-indigo-400">
            {{ metrics.totalPRDs }}
          </div>
          <div class="text-sm text-gray-500 dark:text-gray-400">Total PRDs</div>
        </div>

        <div class="text-center">
          <div class="text-3xl font-bold text-green-600 dark:text-green-400">
            {{ (metrics.averageValidationScore * 100).toFixed(1) }}%
          </div>
          <div class="text-sm text-gray-500 dark:text-gray-400">
            Avg Validation Score
          </div>
        </div>

        <div class="text-center">
          <div class="text-3xl font-bold text-red-600 dark:text-red-400">
            {{ (metrics.averageHallucinationRate * 100).toFixed(2) }}%
          </div>
          <div class="text-sm text-gray-500 dark:text-gray-400">
            Hallucination Rate
          </div>
        </div>

        <div class="text-center">
          <div class="text-3xl font-bold text-blue-600 dark:text-blue-400">
            {{ metrics.averageResponseTime }}ms
          </div>
          <div class="text-sm text-gray-500 dark:text-gray-400">
            Avg Response Time
          </div>
        </div>
      </div>
    </div>

    <!-- Trend Charts -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div class="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
        <h4 class="text-md font-medium text-gray-900 dark:text-white mb-4">
          Weekly PRD Generation
        </h4>
        <div
          class="h-64 flex items-center justify-center text-gray-500 dark:text-gray-400"
        >
          Chart visualization would be rendered here
        </div>
      </div>

      <div class="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
        <h4 class="text-md font-medium text-gray-900 dark:text-white mb-4">
          Quality Metrics Trend
        </h4>
        <div
          class="h-64 flex items-center justify-center text-gray-500 dark:text-gray-400"
        >
          Chart visualization would be rendered here
        </div>
      </div>
    </div>

    <!-- Performance Breakdown -->
    <div class="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
      <h4 class="text-md font-medium text-gray-900 dark:text-white mb-4">
        Agent Performance Breakdown
      </h4>

      <div class="space-y-4">
        <div
          v-for="agent in agentMetrics"
          :key="agent.name"
          class="flex items-center justify-between"
        >
          <div class="flex items-center">
            <div class="w-3 h-3 rounded-full mr-3" :class="agent.color"></div>
            <span class="text-sm font-medium text-gray-900 dark:text-white">{{
              agent.name
            }}</span>
          </div>
          <div class="flex items-center space-x-4">
            <span class="text-sm text-gray-500 dark:text-gray-400">
              {{ agent.tasksCompleted }} tasks
            </span>
            <span class="text-sm text-green-600 dark:text-green-400">
              {{ (agent.successRate * 100).toFixed(1) }}% success
            </span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const metrics = ref({
  totalPRDs: 156,
  averageValidationScore: 0.94,
  averageHallucinationRate: 0.018,
  averageResponseTime: 1250,
});

const agentMetrics = ref([
  {
    name: 'Context Manager',
    tasksCompleted: 142,
    successRate: 0.98,
    color: 'bg-purple-500',
  },
  {
    name: 'Draft Agent',
    tasksCompleted: 89,
    successRate: 0.94,
    color: 'bg-blue-500',
  },
  {
    name: 'Judge Agent',
    tasksCompleted: 234,
    successRate: 0.99,
    color: 'bg-green-500',
  },
  {
    name: 'Task Executor',
    tasksCompleted: 167,
    successRate: 0.96,
    color: 'bg-orange-500',
  },
  {
    name: 'Validation Agent',
    tasksCompleted: 98,
    successRate: 0.97,
    color: 'bg-red-500',
  },
]);
</script>
