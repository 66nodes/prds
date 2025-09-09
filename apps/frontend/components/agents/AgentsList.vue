<template>
  <div class="bg-white dark:bg-gray-800 shadow rounded-lg">
    <div class="px-4 py-5 sm:p-6">
      <h3 class="text-lg leading-6 font-medium text-gray-900 dark:text-white">
        AI Agents
      </h3>
      <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
        Monitor and manage your AI agent fleet
      </p>

      <div class="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <div
          v-for="agent in agents"
          :key="agent.id"
          class="border border-gray-200 dark:border-gray-700 rounded-lg p-4"
        >
          <div class="flex items-center justify-between">
            <div class="flex items-center">
              <div class="flex-shrink-0">
                <div
                  class="h-8 w-8 rounded-full flex items-center justify-center"
                  :class="getAgentIconClass(agent.type)"
                >
                  <svg
                    class="h-4 w-4 text-white"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
              </div>
              <div class="ml-3">
                <h4 class="text-sm font-medium text-gray-900 dark:text-white">
                  {{ agent.name }}
                </h4>
                <p class="text-xs text-gray-500 dark:text-gray-400">
                  {{ agent.type }}
                </p>
              </div>
            </div>
            <div class="flex items-center">
              <div
                class="h-2 w-2 rounded-full mr-2"
                :class="getStatusColor(agent.status)"
              ></div>
              <span
                class="text-xs text-gray-500 dark:text-gray-400 capitalize"
                >{{ agent.status }}</span
              >
            </div>
          </div>

          <div class="mt-4">
            <div
              class="flex justify-between text-xs text-gray-500 dark:text-gray-400"
            >
              <span>Success Rate</span>
              <span
                >{{ (agent.performance.successRate * 100).toFixed(1) }}%</span
              >
            </div>
            <div
              class="mt-1 w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1"
            >
              <div
                class="bg-green-500 h-1 rounded-full"
                :style="{ width: `${agent.performance.successRate * 100}%` }"
              ></div>
            </div>
          </div>

          <div
            class="mt-3 flex justify-between text-xs text-gray-500 dark:text-gray-400"
          >
            <span>Tasks: {{ agent.performance.tasksCompleted }}</span>
            <span>{{ agent.performance.averageResponseTime }}ms avg</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const agents = ref([
  {
    id: '1',
    name: 'Context Manager',
    type: 'context_manager',
    status: 'processing',
    performance: {
      tasksCompleted: 142,
      successRate: 0.98,
      averageResponseTime: 450,
    },
  },
  {
    id: '2',
    name: 'Draft Agent',
    type: 'draft_agent',
    status: 'idle',
    performance: {
      tasksCompleted: 89,
      successRate: 0.94,
      averageResponseTime: 1200,
    },
  },
  {
    id: '3',
    name: 'Judge Agent',
    type: 'judge_agent',
    status: 'processing',
    performance: {
      tasksCompleted: 234,
      successRate: 0.99,
      averageResponseTime: 280,
    },
  },
  {
    id: '4',
    name: 'Task Executor',
    type: 'task_executor',
    status: 'idle',
    performance: {
      tasksCompleted: 167,
      successRate: 0.96,
      averageResponseTime: 650,
    },
  },
  {
    id: '5',
    name: 'Validation Agent',
    type: 'validation_agent',
    status: 'processing',
    performance: {
      tasksCompleted: 98,
      successRate: 0.97,
      averageResponseTime: 380,
    },
  },
]);

const getAgentIconClass = (type: string) => {
  switch (type) {
    case 'context_manager':
      return 'bg-purple-500';
    case 'draft_agent':
      return 'bg-blue-500';
    case 'judge_agent':
      return 'bg-green-500';
    case 'task_executor':
      return 'bg-orange-500';
    case 'validation_agent':
      return 'bg-red-500';
    default:
      return 'bg-gray-500';
  }
};

const getStatusColor = (status: string) => {
  switch (status) {
    case 'processing':
      return 'bg-green-500';
    case 'idle':
      return 'bg-yellow-500';
    case 'error':
      return 'bg-red-500';
    case 'offline':
      return 'bg-gray-500';
    default:
      return 'bg-gray-500';
  }
};
</script>
