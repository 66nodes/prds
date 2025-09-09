<template>
  <div class="bg-white dark:bg-gray-800 shadow-sm rounded-lg border border-gray-200 dark:border-gray-700">
    <!-- Header -->
    <div class="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
      <div class="flex items-center justify-between">
        <h3 class="text-lg font-medium text-gray-900 dark:text-white flex items-center">
          <ShieldExclamationIcon class="h-5 w-5 mr-2" :class="riskLevelClass" />
          Risk Assessment
        </h3>
        <div class="flex items-center space-x-2">
          <span class="text-xs text-gray-500 dark:text-gray-400">
            Confidence: {{ Math.round(riskData.confidence * 100) }}%
          </span>
          <div class="w-2 h-2 rounded-full" :class="confidenceIndicatorClass"></div>
        </div>
      </div>
    </div>

    <!-- Risk Score Display -->
    <div class="p-6">
      <div class="text-center mb-6">
        <!-- Circular Progress Indicator -->
        <div class="relative inline-flex items-center justify-center">
          <svg class="w-32 h-32 transform -rotate-90" viewBox="0 0 100 100">
            <!-- Background circle -->
            <circle
              cx="50"
              cy="50"
              r="40"
              stroke="currentColor"
              stroke-width="8"
              fill="none"
              class="text-gray-200 dark:text-gray-700"
            />
            <!-- Progress circle -->
            <circle
              cx="50"
              cy="50"
              r="40"
              :stroke="riskColor"
              stroke-width="8"
              fill="none"
              stroke-linecap="round"
              :stroke-dasharray="circumference"
              :stroke-dashoffset="strokeDashoffset"
              class="transition-all duration-1000 ease-out"
            />
          </svg>
          <div class="absolute inset-0 flex items-center justify-center">
            <div class="text-center">
              <div class="text-3xl font-bold text-gray-900 dark:text-white">
                {{ riskScoreDisplay }}
              </div>
              <div class="text-sm font-medium" :class="riskLevelClass">
                {{ riskLevel }}
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Risk Level Description -->
      <div class="text-center mb-6">
        <p class="text-sm text-gray-600 dark:text-gray-400">
          {{ riskLevelDescription }}
        </p>
      </div>

      <!-- Quick Stats -->
      <div class="grid grid-cols-2 gap-4 mb-6">
        <div class="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
          <div class="text-2xl font-bold text-gray-900 dark:text-white">
            {{ riskData.risk_factors?.length || 0 }}
          </div>
          <div class="text-xs text-gray-500 dark:text-gray-400">Risk Factors</div>
        </div>
        <div class="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
          <div class="text-2xl font-bold text-gray-900 dark:text-white">
            {{ successProbabilityDisplay }}
          </div>
          <div class="text-xs text-gray-500 dark:text-gray-400">Success Rate</div>
        </div>
      </div>

      <!-- Action Buttons -->
      <div class="flex space-x-2">
        <button
          @click="$emit('view-details')"
          class="flex-1 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md text-sm font-medium transition-colors"
        >
          View Details
        </button>
        <button
          @click="$emit('view-mitigation')"
          class="flex-1 bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-md text-sm font-medium transition-colors"
        >
          Mitigation Plan
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ShieldExclamationIcon } from '@heroicons/vue/24/outline'

interface RiskData {
  overall_risk_score: number
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
  confidence: number
  risk_factors?: Array<any>
  success_probability?: number
  assessment_timestamp: string
}

interface Props {
  riskData: RiskData
  loading?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  loading: false
})

defineEmits<{
  'view-details': []
  'view-mitigation': []
}>()

// Computed properties
const riskScoreDisplay = computed(() => {
  return Math.round(props.riskData.overall_risk_score * 100)
})

const successProbabilityDisplay = computed(() => {
  const probability = props.riskData.success_probability || (1 - props.riskData.overall_risk_score)
  return Math.round(probability * 100) + '%'
})

const riskLevel = computed(() => {
  return props.riskData.risk_level?.toLowerCase().replace('_', ' ') || 'Unknown'
})

const riskLevelClass = computed(() => {
  const level = props.riskData.risk_level
  switch (level) {
    case 'LOW':
      return 'text-green-600 dark:text-green-400'
    case 'MEDIUM':
      return 'text-yellow-600 dark:text-yellow-400'
    case 'HIGH':
      return 'text-orange-600 dark:text-orange-400'
    case 'CRITICAL':
      return 'text-red-600 dark:text-red-400'
    default:
      return 'text-gray-600 dark:text-gray-400'
  }
})

const riskColor = computed(() => {
  const level = props.riskData.risk_level
  switch (level) {
    case 'LOW':
      return '#10b981' // green-500
    case 'MEDIUM':
      return '#f59e0b' // yellow-500
    case 'HIGH':
      return '#f97316' // orange-500
    case 'CRITICAL':
      return '#ef4444' // red-500
    default:
      return '#6b7280' // gray-500
  }
})

const riskLevelDescription = computed(() => {
  const level = props.riskData.risk_level
  switch (level) {
    case 'LOW':
      return 'Project has minimal risk factors and high success probability'
    case 'MEDIUM':
      return 'Project has moderate risks that should be monitored and managed'
    case 'HIGH':
      return 'Project has significant risks requiring active mitigation strategies'
    case 'CRITICAL':
      return 'Project has severe risks that may threaten success without immediate action'
    default:
      return 'Risk level assessment is pending'
  }
})

const confidenceIndicatorClass = computed(() => {
  const confidence = props.riskData.confidence
  if (confidence >= 0.8) return 'bg-green-400'
  if (confidence >= 0.6) return 'bg-yellow-400'
  return 'bg-red-400'
})

// Circle progress calculations
const circumference = 2 * Math.PI * 40 // radius = 40

const strokeDashoffset = computed(() => {
  const progress = props.riskData.overall_risk_score
  return circumference - (progress * circumference)
})
</script>

<style scoped>
/* Add any additional styles if needed */
</style>