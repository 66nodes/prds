<template>
  <div class="bg-white dark:bg-gray-800 shadow-sm rounded-lg border border-gray-200 dark:border-gray-700">
    <!-- Header -->
    <div class="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
      <h3 class="text-lg font-medium text-gray-900 dark:text-white flex items-center">
        <ChartBarIcon class="h-5 w-5 mr-2 text-blue-500" />
        Historical Insights & Patterns
        <span class="ml-2 text-sm text-gray-500 dark:text-gray-400">
          ({{ historicalPatterns.length }} patterns found)
        </span>
      </h3>
    </div>

    <div class="p-4 space-y-6">
      <!-- Success Probability Comparison -->
      <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
        <h4 class="text-sm font-medium text-gray-900 dark:text-white mb-3 flex items-center">
          <TrendingUpIcon class="h-4 w-4 mr-2" />
          Success Rate Comparison
        </h4>
        
        <div class="flex items-center justify-between mb-2">
          <span class="text-sm text-gray-600 dark:text-gray-400">Similar Projects</span>
          <span class="text-sm font-medium text-gray-900 dark:text-white">
            {{ historicalComparison.average_success_rate ? Math.round(historicalComparison.average_success_rate * 100) + '%' : 'N/A' }}
          </span>
        </div>
        
        <div class="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2 mb-3">
          <div
            class="bg-blue-500 h-2 rounded-full transition-all duration-500"
            :style="{ width: `${(historicalComparison.average_success_rate || 0) * 100}%` }"
          ></div>
        </div>
        
        <div class="text-xs text-gray-500 dark:text-gray-400">
          Based on {{ historicalComparison.similar_projects_count || 0 }} similar projects
          <span v-if="historicalComparison.score_range">
            (risk score {{ historicalComparison.score_range }})
          </span>
        </div>
      </div>

      <!-- Historical Patterns -->
      <div v-if="historicalPatterns.length > 0">
        <h4 class="text-sm font-medium text-gray-900 dark:text-white mb-3 flex items-center">
          <SparklesIcon class="h-4 w-4 mr-2" />
          Identified Patterns
        </h4>
        
        <div class="space-y-3">
          <div
            v-for="pattern in historicalPatterns"
            :key="pattern.pattern_id"
            class="border border-gray-200 dark:border-gray-600 rounded-lg p-3 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          >
            <div class="flex items-start justify-between mb-2">
              <div class="flex-1">
                <div class="flex items-center space-x-2 mb-1">
                  <span class="text-sm font-medium text-gray-900 dark:text-white">
                    {{ pattern.description }}
                  </span>
                  <span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium"
                        :class="getPatternTypeClass(pattern.pattern_type)">
                    {{ formatPatternType(pattern.pattern_type) }}
                  </span>
                </div>
                
                <div class="flex items-center space-x-4 text-xs text-gray-500 dark:text-gray-400">
                  <span class="flex items-center">
                    <ClockIcon class="h-3 w-3 mr-1" />
                    {{ pattern.projects_count }} projects
                  </span>
                  <span class="flex items-center">
                    <ChartBarIcon class="h-3 w-3 mr-1" />
                    {{ Math.round(pattern.success_rate * 100) }}% success rate
                  </span>
                  <span class="flex items-center">
                    <ArrowTrendingUpIcon class="h-3 w-3 mr-1" />
                    {{ Math.round(pattern.frequency * 100) }}% frequency
                  </span>
                </div>
              </div>
            </div>
            
            <!-- Pattern Template Suggestions -->
            <div v-if="pattern.template_suggestions && pattern.template_suggestions.length > 0" 
                 class="mt-2 pt-2 border-t border-gray-100 dark:border-gray-600">
              <div class="text-xs text-gray-500 dark:text-gray-400 mb-1">Suggested Templates:</div>
              <div class="flex flex-wrap gap-1">
                <span
                  v-for="template in pattern.template_suggestions"
                  :key="template"
                  class="inline-flex items-center px-2 py-0.5 rounded text-xs bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
                >
                  {{ template }}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Template Recommendations -->
      <div v-if="recommendedTemplates.length > 0">
        <h4 class="text-sm font-medium text-gray-900 dark:text-white mb-3 flex items-center">
          <DocumentTextIcon class="h-4 w-4 mr-2" />
          Recommended Templates
        </h4>
        
        <div class="space-y-3">
          <div
            v-for="template in recommendedTemplates"
            :key="template.template_id"
            class="border border-gray-200 dark:border-gray-600 rounded-lg p-3 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors cursor-pointer"
            @click="$emit('select-template', template)"
          >
            <div class="flex items-start justify-between">
              <div class="flex-1">
                <div class="flex items-center space-x-2 mb-2">
                  <span class="text-sm font-medium text-gray-900 dark:text-white">
                    {{ template.name }}
                  </span>
                  <span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                    {{ Math.round(template.success_rate * 100) }}% success
                  </span>
                  <span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200">
                    -{{ Math.round(template.risk_reduction * 100) }}% risk
                  </span>
                </div>
                
                <p class="text-xs text-gray-600 dark:text-gray-400 mb-2">
                  {{ template.description }}
                </p>
                
                <div class="flex items-center justify-between">
                  <div class="text-xs text-gray-500 dark:text-gray-400">
                    Relevance: {{ Math.round(template.relevance_score * 100) }}%
                  </div>
                  <button
                    @click.stop="$emit('view-template', template)"
                    class="text-xs text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 font-medium"
                  >
                    View Template →
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Lessons Learned -->
      <div v-if="lessonsLearned.length > 0">
        <h4 class="text-sm font-medium text-gray-900 dark:text-white mb-3 flex items-center">
          <LightBulbIcon class="h-4 w-4 mr-2" />
          Lessons Learned
        </h4>
        
        <div class="space-y-3">
          <div
            v-for="lesson in lessonsLearned"
            :key="lesson.lesson_id"
            class="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-3"
          >
            <div class="flex items-start">
              <ExclamationCircleIcon class="h-4 w-4 text-yellow-500 mr-2 mt-0.5 flex-shrink-0" />
              <div class="flex-1">
                <div class="text-sm font-medium text-gray-900 dark:text-white mb-1">
                  {{ lesson.title }}
                </div>
                <p class="text-xs text-gray-600 dark:text-gray-400 mb-2">
                  {{ lesson.description }}
                </p>
                <div class="bg-white dark:bg-gray-800 rounded p-2 text-xs">
                  <div class="font-medium text-gray-900 dark:text-white mb-1">Recommendation:</div>
                  <div class="text-gray-600 dark:text-gray-400">{{ lesson.recommendation }}</div>
                </div>
                <div class="mt-2 flex items-center space-x-4 text-xs text-gray-500 dark:text-gray-400">
                  <span>Confidence: {{ Math.round(lesson.confidence * 100) }}%</span>
                  <span>{{ lesson.source_projects.length }} projects</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Actionable Insights -->
      <div v-if="actionableInsights.length > 0">
        <h4 class="text-sm font-medium text-gray-900 dark:text-white mb-3 flex items-center">
          <BoltIcon class="h-4 w-4 mr-2" />
          Key Recommendations
        </h4>
        
        <div class="space-y-2">
          <div
            v-for="(insight, index) in actionableInsights"
            :key="index"
            class="flex items-start p-2 bg-blue-50 dark:bg-blue-900/20 rounded-lg"
          >
            <CheckCircleIcon class="h-4 w-4 text-blue-500 mr-2 mt-0.5 flex-shrink-0" />
            <span class="text-sm text-gray-900 dark:text-white">{{ insight }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Footer Actions -->
    <div class="px-4 py-3 bg-gray-50 dark:bg-gray-700 border-t border-gray-200 dark:border-gray-600">
      <div class="flex items-center justify-between">
        <span class="text-xs text-gray-500 dark:text-gray-400">
          Analysis based on {{ totalHistoricalProjects }} historical projects
        </span>
        <div class="flex space-x-2">
          <button
            @click="$emit('export-insights')"
            class="text-xs text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 font-medium"
          >
            Export Report
          </button>
          <button
            @click="$emit('view-similar-projects')"
            class="text-xs text-gray-600 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-200 font-medium"
          >
            View Similar Projects
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import {
  ChartBarIcon,
  TrendingUpIcon,
  SparklesIcon,
  ClockIcon,
  ArrowTrendingUpIcon,
  DocumentTextIcon,
  LightBulbIcon,
  ExclamationCircleIcon,
  BoltIcon,
  CheckCircleIcon
} from '@heroicons/vue/24/outline'

interface HistoricalPattern {
  pattern_id: string
  pattern_type: string
  description: string
  frequency: number
  success_rate: number
  projects_count: number
  template_suggestions: string[]
}

interface RecommendedTemplate {
  template_id: string
  name: string
  description: string
  category: string
  relevance_score: number
  success_rate: number
  risk_reduction: number
}

interface LessonLearned {
  lesson_id: string
  title: string
  description: string
  recommendation: string
  confidence: number
  source_projects: string[]
}

interface HistoricalComparison {
  similar_projects_count: number
  average_success_rate: number
  score_range: string
}

interface Props {
  historicalPatterns: HistoricalPattern[]
  recommendedTemplates: RecommendedTemplate[]
  lessonsLearned: LessonLearned[]
  historicalComparison: HistoricalComparison
  actionableInsights: string[]
  loading?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  loading: false
})

defineEmits<{
  'select-template': [template: RecommendedTemplate]
  'view-template': [template: RecommendedTemplate]
  'export-insights': []
  'view-similar-projects': []
}>()

// Computed properties
const totalHistoricalProjects = computed(() => {
  return props.historicalPatterns.reduce((total, pattern) => total + pattern.projects_count, 0)
})

// Methods
const formatPatternType = (type: string) => {
  return type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
}

const getPatternTypeClass = (type: string) => {
  const classes = {
    'SUCCESS_FACTOR': 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    'FAILURE_INDICATOR': 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
    'COMPLEXITY_DRIVER': 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200',
    'RISK_CORRELATE': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    'TEMPLATE_TRIGGER': 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
  }
  return classes[type as keyof typeof classes] || 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
}
</script>