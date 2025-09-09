<template>
  <div class="bg-white dark:bg-gray-800 shadow-sm rounded-lg border border-gray-200 dark:border-gray-700">
    <!-- Header -->
    <div class="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
      <div class="flex items-center justify-between">
        <h3 class="text-lg font-medium text-gray-900 dark:text-white flex items-center">
          <ExclamationTriangleIcon class="h-5 w-5 mr-2 text-yellow-500" />
          Risk Factors
          <span class="ml-2 text-sm text-gray-500 dark:text-gray-400">
            ({{ riskFactors.length }})
          </span>
        </h3>
        <div class="flex items-center space-x-2">
          <button
            @click="sortBy = sortBy === 'risk_score' ? 'category' : 'risk_score'"
            class="text-xs text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
          >
            Sort by {{ sortBy === 'risk_score' ? 'Category' : 'Risk Score' }}
          </button>
          <button
            @click="showOnlyHigh = !showOnlyHigh"
            class="text-xs px-2 py-1 rounded"
            :class="showOnlyHigh 
              ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' 
              : 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'"
          >
            {{ showOnlyHigh ? 'Show All' : 'High Risk Only' }}
          </button>
        </div>
      </div>
    </div>

    <!-- Risk Factors List -->
    <div class="max-h-96 overflow-y-auto">
      <div v-if="filteredFactors.length === 0" class="p-6 text-center text-gray-500 dark:text-gray-400">
        <ClipboardDocumentCheckIcon class="h-12 w-12 mx-auto mb-2 text-gray-300 dark:text-gray-600" />
        <p>No risk factors found</p>
        <p class="text-sm">This project shows a clean risk profile</p>
      </div>
      
      <div v-else class="divide-y divide-gray-200 dark:divide-gray-700">
        <div
          v-for="factor in filteredFactors"
          :key="factor.id"
          class="p-4 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
        >
          <div class="flex items-start justify-between">
            <div class="flex-1 min-w-0">
              <!-- Risk Factor Header -->
              <div class="flex items-center mb-2">
                <div class="flex items-center space-x-2">
                  <component :is="categoryIcon(factor.category)" class="h-4 w-4 text-gray-500" />
                  <span class="text-sm font-medium text-gray-900 dark:text-white">
                    {{ factor.name }}
                  </span>
                  <span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium"
                        :class="categoryClass(factor.category)">
                    {{ factor.category }}
                  </span>
                </div>
                <div class="ml-auto flex items-center space-x-2">
                  <span class="text-xs text-gray-500 dark:text-gray-400">
                    {{ Math.round(factor.probability * 100) }}% chance
                  </span>
                  <div class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium"
                       :class="riskLevelClass(factor.level)">
                    {{ factor.level }}
                  </div>
                </div>
              </div>

              <!-- Risk Factor Description -->
              <p class="text-sm text-gray-600 dark:text-gray-400 mb-3">
                {{ factor.description }}
              </p>

              <!-- Risk Score Visualization -->
              <div class="mb-3">
                <div class="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400 mb-1">
                  <span>Risk Score</span>
                  <span>{{ Math.round(factor.risk_score * 100) }}/100</span>
                </div>
                <div class="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                  <div
                    class="h-2 rounded-full transition-all duration-500"
                    :style="{ 
                      width: `${factor.risk_score * 100}%`,
                      backgroundColor: getRiskColor(factor.level)
                    }"
                  ></div>
                </div>
              </div>

              <!-- Historical Frequency -->
              <div class="mb-3" v-if="factor.historical_frequency > 0">
                <div class="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400 mb-1">
                  <span>Historical Frequency</span>
                  <span>{{ Math.round(factor.historical_frequency * 100) }}% of projects</span>
                </div>
                <div class="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-1">
                  <div
                    class="bg-blue-500 h-1 rounded-full transition-all duration-500"
                    :style="{ width: `${factor.historical_frequency * 100}%` }"
                  ></div>
                </div>
              </div>

              <!-- Mitigation Strategies -->
              <div v-if="factor.mitigation_strategies && factor.mitigation_strategies.length > 0"
                   class="mt-3">
                <button
                  @click="toggleMitigation(factor.id)"
                  class="text-xs text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 flex items-center"
                >
                  <ChevronRightIcon 
                    class="h-3 w-3 mr-1 transition-transform"
                    :class="{ 'transform rotate-90': expandedMitigation.has(factor.id) }"
                  />
                  {{ expandedMitigation.has(factor.id) ? 'Hide' : 'Show' }} Mitigation Strategies
                </button>
                
                <div v-if="expandedMitigation.has(factor.id)" 
                     class="mt-2 ml-4 space-y-1">
                  <div
                    v-for="(strategy, index) in factor.mitigation_strategies"
                    :key="index"
                    class="flex items-start text-xs text-gray-600 dark:text-gray-400"
                  >
                    <CheckCircleIcon class="h-3 w-3 mr-2 mt-0.5 text-green-500 flex-shrink-0" />
                    <span>{{ strategy }}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Summary Footer -->
    <div class="px-4 py-3 bg-gray-50 dark:bg-gray-700 border-t border-gray-200 dark:border-gray-600">
      <div class="flex items-center justify-between text-sm">
        <span class="text-gray-600 dark:text-gray-400">
          {{ highRiskCount }} high-priority risks require attention
        </span>
        <button
          @click="$emit('export-report')"
          class="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 font-medium"
        >
          Export Risk Report
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { 
  ExclamationTriangleIcon, 
  ClipboardDocumentCheckIcon,
  ChevronRightIcon,
  CheckCircleIcon,
  CpuChipIcon,
  ClockIcon,
  DocumentTextIcon,
  UsersIcon,
  GlobeAltIcon,
  CurrencyDollarIcon,
  ChatBubbleLeftRightIcon,
  ServerIcon,
  CircleStackIcon,
  BeakerIcon
} from '@heroicons/vue/24/outline'

interface RiskFactor {
  id: string
  category: 'TECHNICAL' | 'SCHEDULE' | 'SCOPE' | 'TEAM' | 'EXTERNAL' | 'BUDGET' | 'STAKEHOLDER' | 'INTEGRATION' | 'DATA' | 'TECHNOLOGY'
  name: string
  description: string
  probability: number
  impact: number
  risk_score: number
  level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
  mitigation_strategies: string[]
  historical_frequency: number
}

interface Props {
  riskFactors: RiskFactor[]
  loading?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  loading: false
})

defineEmits<{
  'export-report': []
}>()

// Reactive state
const sortBy = ref<'risk_score' | 'category'>('risk_score')
const showOnlyHigh = ref(false)
const expandedMitigation = ref(new Set<string>())

// Computed properties
const filteredFactors = computed(() => {
  let factors = [...props.riskFactors]
  
  // Filter by risk level if needed
  if (showOnlyHigh.value) {
    factors = factors.filter(f => f.level === 'HIGH' || f.level === 'CRITICAL')
  }
  
  // Sort factors
  factors.sort((a, b) => {
    if (sortBy.value === 'risk_score') {
      return b.risk_score - a.risk_score
    } else {
      return a.category.localeCompare(b.category)
    }
  })
  
  return factors
})

const highRiskCount = computed(() => {
  return props.riskFactors.filter(f => f.level === 'HIGH' || f.level === 'CRITICAL').length
})

// Methods
const toggleMitigation = (factorId: string) => {
  if (expandedMitigation.value.has(factorId)) {
    expandedMitigation.value.delete(factorId)
  } else {
    expandedMitigation.value.add(factorId)
  }
}

const categoryIcon = (category: string) => {
  const icons = {
    TECHNICAL: CpuChipIcon,
    SCHEDULE: ClockIcon,
    SCOPE: DocumentTextIcon,
    TEAM: UsersIcon,
    EXTERNAL: GlobeAltIcon,
    BUDGET: CurrencyDollarIcon,
    STAKEHOLDER: ChatBubbleLeftRightIcon,
    INTEGRATION: ServerIcon,
    DATA: CircleStackIcon,
    TECHNOLOGY: BeakerIcon
  }
  return icons[category as keyof typeof icons] || ExclamationTriangleIcon
}

const categoryClass = (category: string) => {
  const classes = {
    TECHNICAL: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
    SCHEDULE: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    SCOPE: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
    TEAM: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    EXTERNAL: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200',
    BUDGET: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
    STAKEHOLDER: 'bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200',
    INTEGRATION: 'bg-cyan-100 text-cyan-800 dark:bg-cyan-900 dark:text-cyan-200',
    DATA: 'bg-teal-100 text-teal-800 dark:bg-teal-900 dark:text-teal-200',
    TECHNOLOGY: 'bg-pink-100 text-pink-800 dark:bg-pink-900 dark:text-pink-200'
  }
  return classes[category as keyof typeof classes] || 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
}

const riskLevelClass = (level: string) => {
  const classes = {
    LOW: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    MEDIUM: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    HIGH: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200',
    CRITICAL: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
  }
  return classes[level as keyof typeof classes] || 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
}

const getRiskColor = (level: string) => {
  const colors = {
    LOW: '#10b981',
    MEDIUM: '#f59e0b',
    HIGH: '#f97316',
    CRITICAL: '#ef4444'
  }
  return colors[level as keyof typeof colors] || '#6b7280'
}
</script>