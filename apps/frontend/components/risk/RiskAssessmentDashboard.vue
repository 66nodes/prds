<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-2xl font-bold text-gray-900 dark:text-white">Risk Assessment</h1>
        <p class="text-gray-600 dark:text-gray-400">
          Comprehensive risk analysis and historical insights for your project
        </p>
      </div>
      <div class="flex space-x-3">
        <button
          @click="refreshAssessment"
          :disabled="loading"
          class="inline-flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 shadow-sm text-sm font-medium rounded-md text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50"
        >
          <ArrowPathIcon class="h-4 w-4 mr-2" :class="{ 'animate-spin': loading }" />
          Refresh
        </button>
        <button
          @click="exportFullReport"
          class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
        >
          <DocumentArrowDownIcon class="h-4 w-4 mr-2" />
          Export Report
        </button>
      </div>
    </div>

    <!-- Loading State -->
    <div v-if="loading" class="flex items-center justify-center py-12">
      <div class="text-center">
        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
        <p class="text-gray-600 dark:text-gray-400">Analyzing project risks...</p>
      </div>
    </div>

    <!-- Error State -->
    <div v-else-if="error" class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
      <div class="flex">
        <XCircleIcon class="h-5 w-5 text-red-400 mr-3 mt-0.5" />
        <div>
          <h3 class="text-sm font-medium text-red-800 dark:text-red-200">Assessment Failed</h3>
          <p class="text-sm text-red-600 dark:text-red-400 mt-1">{{ error }}</p>
          <button
            @click="refreshAssessment"
            class="mt-2 text-sm text-red-600 dark:text-red-400 hover:text-red-500 dark:hover:text-red-300 underline"
          >
            Try again
          </button>
        </div>
      </div>
    </div>

    <!-- Assessment Results -->
    <div v-else-if="riskAssessment" class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Left Column: Risk Score and Quick Actions -->
      <div class="space-y-6">
        <!-- Risk Score Card -->
        <RiskScoreCard 
          :risk-data="riskAssessment"
          @view-details="showDetailsModal = true"
          @view-mitigation="showMitigationModal = true"
        />

        <!-- Quick Actions -->
        <div class="bg-white dark:bg-gray-800 shadow-sm rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <h3 class="text-sm font-medium text-gray-900 dark:text-white mb-3">Quick Actions</h3>
          <div class="space-y-2">
            <button
              @click="scheduleReview"
              class="w-full text-left px-3 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md transition-colors"
            >
              📅 Schedule Risk Review
            </button>
            <button
              @click="createMitigationPlan"
              class="w-full text-left px-3 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md transition-colors"
            >
              🛡️ Create Mitigation Plan
            </button>
            <button
              @click="shareAssessment"
              class="w-full text-left px-3 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md transition-colors"
            >
              📤 Share with Team
            </button>
          </div>
        </div>

        <!-- Assessment Info -->
        <div class="bg-white dark:bg-gray-800 shadow-sm rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <h3 class="text-sm font-medium text-gray-900 dark:text-white mb-3">Assessment Info</h3>
          <div class="space-y-2 text-sm text-gray-600 dark:text-gray-400">
            <div class="flex justify-between">
              <span>Completed:</span>
              <span>{{ formatDate(riskAssessment.assessment_timestamp) }}</span>
            </div>
            <div class="flex justify-between">
              <span>Confidence:</span>
              <span>{{ Math.round(riskAssessment.confidence * 100) }}%</span>
            </div>
            <div class="flex justify-between">
              <span>Factors Analyzed:</span>
              <span>{{ riskAssessment.risk_factors?.length || 0 }}</span>
            </div>
            <div class="flex justify-between">
              <span>Historical Data:</span>
              <span>{{ riskAssessment.historical_patterns?.length || 0 }} patterns</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Middle Column: Risk Factors -->
      <div class="space-y-6">
        <RiskFactorsList 
          :risk-factors="riskAssessment.risk_factors || []"
          @export-report="exportRiskFactorsReport"
        />
      </div>

      <!-- Right Column: Historical Insights -->
      <div class="space-y-6">
        <HistoricalInsights
          :historical-patterns="riskAssessment.historical_patterns || []"
          :recommended-templates="riskAssessment.recommended_templates || []"
          :lessons-learned="lessonsLearned"
          :historical-comparison="historicalComparison"
          :actionable-insights="riskAssessment.actionable_insights || []"
          @select-template="selectTemplate"
          @view-template="viewTemplate"
          @export-insights="exportInsightsReport"
          @view-similar-projects="viewSimilarProjects"
        />
      </div>
    </div>

    <!-- Empty State -->
    <div v-else class="text-center py-12">
      <ChartBarSquareIcon class="h-12 w-12 text-gray-400 mx-auto mb-4" />
      <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-2">No Risk Assessment Available</h3>
      <p class="text-gray-600 dark:text-gray-400 mb-4">
        Start a risk assessment to analyze potential project risks and get actionable insights.
      </p>
      <button
        @click="startAssessment"
        class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
      >
        <PlayIcon class="h-4 w-4 mr-2" />
        Start Risk Assessment
      </button>
    </div>

    <!-- Modals -->
    <RiskDetailsModal 
      v-if="showDetailsModal"
      :risk-data="riskAssessment"
      @close="showDetailsModal = false"
    />

    <MitigationPlanModal
      v-if="showMitigationModal"
      :risk-factors="riskAssessment?.risk_factors || []"
      @close="showMitigationModal = false"
      @save="saveMitigationPlan"
    />
  </div>
</template>

<script setup lang="ts">
import {
  ArrowPathIcon,
  DocumentArrowDownIcon,
  XCircleIcon,
  ChartBarSquareIcon,
  PlayIcon
} from '@heroicons/vue/24/outline'
import RiskScoreCard from './RiskScoreCard.vue'
import RiskFactorsList from './RiskFactorsList.vue'
import HistoricalInsights from './HistoricalInsights.vue'

interface RiskAssessmentData {
  overall_risk_score: number
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
  confidence: number
  risk_factors: any[]
  historical_patterns: any[]
  recommended_templates: any[]
  actionable_insights: string[]
  assessment_timestamp: string
}

interface Props {
  projectId?: string
  projectDescription?: string
  autoRun?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  autoRun: false
})

// Reactive state
const loading = ref(false)
const error = ref<string | null>(null)
const riskAssessment = ref<RiskAssessmentData | null>(null)
const lessonsLearned = ref([])
const historicalComparison = ref({
  similar_projects_count: 0,
  average_success_rate: 0,
  score_range: ''
})
const showDetailsModal = ref(false)
const showMitigationModal = ref(false)

// Lifecycle
onMounted(async () => {
  if (props.autoRun && props.projectDescription) {
    await runRiskAssessment()
  }
})

// Methods
const runRiskAssessment = async () => {
  if (!props.projectDescription) {
    error.value = 'Project description is required for risk assessment'
    return
  }

  loading.value = true
  error.value = null

  try {
    // Call risk assessment API
    const response = await $fetch('/api/risk-assessment', {
      method: 'POST',
      body: {
        project_description: props.projectDescription,
        project_id: props.projectId,
        include_historical: true,
        include_templates: true
      }
    })

    riskAssessment.value = response.assessment
    
    // Get additional data
    await Promise.all([
      loadLessonsLearned(),
      loadHistoricalComparison()
    ])

  } catch (err: any) {
    error.value = err.message || 'Failed to run risk assessment'
    console.error('Risk assessment failed:', err)
  } finally {
    loading.value = false
  }
}

const loadLessonsLearned = async () => {
  try {
    const response = await $fetch('/api/risk-assessment/lessons-learned')
    lessonsLearned.value = response.lessons
  } catch (err) {
    console.error('Failed to load lessons learned:', err)
  }
}

const loadHistoricalComparison = async () => {
  if (!riskAssessment.value) return

  try {
    const response = await $fetch('/api/risk-assessment/historical-comparison', {
      method: 'POST',
      body: {
        risk_score: riskAssessment.value.overall_risk_score,
        project_description: props.projectDescription
      }
    })
    historicalComparison.value = response.comparison
  } catch (err) {
    console.error('Failed to load historical comparison:', err)
  }
}

const refreshAssessment = async () => {
  await runRiskAssessment()
}

const startAssessment = async () => {
  await runRiskAssessment()
}

const exportFullReport = async () => {
  try {
    const response = await $fetch('/api/risk-assessment/export', {
      method: 'POST',
      body: {
        assessment: riskAssessment.value,
        format: 'pdf'
      }
    })
    
    // Download the report
    const link = document.createElement('a')
    link.href = response.download_url
    link.download = `risk-assessment-${Date.now()}.pdf`
    link.click()
    
  } catch (err: any) {
    console.error('Export failed:', err)
    // Show error toast
  }
}

const exportRiskFactorsReport = async () => {
  // Implementation for exporting risk factors report
  console.log('Exporting risk factors report...')
}

const exportInsightsReport = async () => {
  // Implementation for exporting insights report
  console.log('Exporting insights report...')
}

const scheduleReview = () => {
  // Implementation for scheduling risk review
  console.log('Scheduling risk review...')
}

const createMitigationPlan = () => {
  // Implementation for creating mitigation plan
  console.log('Creating mitigation plan...')
}

const shareAssessment = () => {
  // Implementation for sharing assessment
  console.log('Sharing assessment...')
}

const selectTemplate = (template: any) => {
  // Implementation for selecting template
  console.log('Selected template:', template)
}

const viewTemplate = (template: any) => {
  // Implementation for viewing template details
  console.log('Viewing template:', template)
}

const viewSimilarProjects = () => {
  // Implementation for viewing similar projects
  console.log('Viewing similar projects...')
}

const saveMitigationPlan = (plan: any) => {
  // Implementation for saving mitigation plan
  console.log('Saving mitigation plan:', plan)
  showMitigationModal.value = false
}

// Utility functions
const formatDate = (dateString: string) => {
  return new Date(dateString).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })
}

// Expose methods for parent components
defineExpose({
  runRiskAssessment,
  refreshAssessment
})
</script>