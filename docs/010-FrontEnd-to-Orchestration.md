# Frontend to Orchestration API Integration Specification

## 1. API Client Architecture

### 1.1 Core API Client Configuration

```typescript
// composables/useApiClient.ts
import type { $Fetch } from 'ofetch'

interface ApiClientConfig {
  baseURL: string
  timeout: number
  retryAttempts: number
  retryDelay: number
}

export const useApiClient = () => {
  const config = useRuntimeConfig()
  const { $supabase } = useNuxtApp()
  
  const apiClient: $Fetch = $fetch.create({
    baseURL: config.public.apiBaseUrl || 'http://localhost:8000/api/v1',
    timeout: 30000,
    
    async onRequest({ request, options }) {
      // Add authentication header
      const session = await $supabase.auth.getSession()
      if (session?.data?.session?.access_token) {
        options.headers = {
          ...options.headers,
          'Authorization': `Bearer ${session.data.session.access_token}`,
          'X-Request-ID': generateRequestId(),
          'X-Client-Version': config.public.appVersion
        }
      }
    },
    
    async onResponseError({ response }) {
      if (response.status === 401) {
        // Token expired - refresh
        await $supabase.auth.refreshSession()
        // Retry the request
        return $fetch(response.url, response._data)
      }
      
      // Handle other errors
      const error = new ApiError(
        response._data?.message || 'API request failed',
        response.status,
        response._data?.code
      )
      
      // Log to monitoring service
      await logApiError(error)
      throw error
    }
  })
  
  return apiClient
}
```

### 1.2 API Service Layer

```typescript
// services/api/orchestration.service.ts
export class OrchestrationApiService {
  private client: $Fetch
  
  constructor() {
    this.client = useApiClient()
  }
  
  // Project Concept Submission
  async submitProjectConcept(data: ProjectConceptRequest): Promise<ProjectConceptResponse> {
    return await this.client('/concept/submit', {
      method: 'POST',
      body: data,
      retry: 3,
      retryDelay: 1000
    })
  }
  
  // Generate Clarification Questions
  async generateClarifications(projectId: string, graphGaps: string[]): Promise<ClarificationResponse> {
    return await this.client('/concept/clarify', {
      method: 'POST',
      body: {
        project_id: projectId,
        graph_gaps: graphGaps
      }
    })
  }
  
  // Generate PRD with GraphRAG Validation
  async generatePRD(request: PRDGenerationRequest): Promise<PRDGenerationResponse> {
    // Long-running operation with progress tracking
    const response = await this.client('/prd/generate', {
      method: 'POST',
      body: request,
      timeout: 600000, // 10 minutes for PRD generation
      onUploadProgress: (progress) => {
        useProgressStore().updateProgress('prd_generation', progress)
      }
    })
    
    return response
  }
  
  // Validate Content with GraphRAG
  async validateContent(request: ValidationRequest): Promise<ValidationResponse> {
    return await this.client('/validation/content', {
      method: 'POST',
      body: request,
      timeout: 5000 // 5 seconds for validation
    })
  }
  
  // Export PRD
  async exportPRD(prdId: string, format: ExportFormat): Promise<Blob> {
    const response = await this.client(`/export/${prdId}`, {
      method: 'GET',
      params: { format },
      responseType: 'blob'
    })
    
    return response
  }
  
  // Create GitHub Project
  async createGitHubProject(request: GitHubProjectRequest): Promise<GitHubProjectResponse> {
    return await this.client('/integrations/github/create', {
      method: 'POST',
      body: request
    })
  }
}
```

## 2. State Management with Pinia

### 2.1 Project Store

```typescript
// stores/project.store.ts
import { defineStore } from 'pinia'

interface ProjectState {
  currentProject: Project | null
  projects: Project[]
  isLoading: boolean
  error: ApiError | null
  validationResults: ValidationResult[]
}

export const useProjectStore = defineStore('project', {
  state: (): ProjectState => ({
    currentProject: null,
    projects: [],
    isLoading: false,
    error: null,
    validationResults: []
  }),
  
  getters: {
    getCurrentProject: (state) => state.currentProject,
    getProjectById: (state) => (id: string) => 
      state.projects.find(p => p.id === id),
    hasValidationErrors: (state) => 
      state.validationResults.some(v => v.confidence_score < 0.95)
  },
  
  actions: {
    async submitConcept(conceptText: string) {
      this.isLoading = true
      this.error = null
      
      try {
        const api = new OrchestrationApiService()
        const response = await api.submitProjectConcept({
          concept_text: conceptText,
          industry_context: useUserStore().industryContext,
          project_type: null
        })
        
        this.currentProject = {
          id: response.project_id,
          status: response.status,
          similarProjects: response.similar_projects
        }
        
        // Navigate to clarification phase
        await navigateTo(`/projects/${response.project_id}/clarify`)
        
      } catch (error) {
        this.error = error as ApiError
        useToast().error('Failed to submit project concept')
      } finally {
        this.isLoading = false
      }
    },
    
    async generatePRD(projectId: string, clarifications: any) {
      this.isLoading = true
      
      try {
        const api = new OrchestrationApiService()
        
        // Start SSE connection for real-time progress
        const eventSource = new EventSource(
          `/api/v1/prd/generate/stream?project_id=${projectId}`
        )
        
        eventSource.onmessage = (event) => {
          const data = JSON.parse(event.data)
          this.updateGenerationProgress(data)
        }
        
        const response = await api.generatePRD({
          project_id: projectId,
          clarifications,
          validation_threshold: 0.95
        })
        
        eventSource.close()
        
        this.currentProject = {
          ...this.currentProject,
          prd: response.document,
          wbs: response.wbs,
          quality_metrics: response.quality_metrics
        }
        
        return response
        
      } catch (error) {
        this.error = error as ApiError
        throw error
      } finally {
        this.isLoading = false
      }
    }
  }
})
```

### 2.2 Agent Orchestration Store

```typescript
// stores/agent.store.ts
export const useAgentStore = defineStore('agent', {
  state: () => ({
    activeAgents: [] as Agent[],
    agentLogs: [] as AgentLog[],
    currentPhase: null as WorkflowPhase | null,
    workflowStatus: 'idle' as WorkflowStatus
  }),
  
  actions: {
    async initializeWorkflow(projectId: string) {
      const api = new OrchestrationApiService()
      
      // Start WebSocket connection for real-time updates
      const ws = new WebSocket(`ws://localhost:8000/ws/workflow/${projectId}`)
      
      ws.onmessage = (event) => {
        const message = JSON.parse(event.data)
        
        switch (message.type) {
          case 'AGENT_STARTED':
            this.activeAgents.push(message.agent)
            break
            
          case 'AGENT_COMPLETED':
            this.updateAgentStatus(message.agent_id, 'completed')
            break
            
          case 'PHASE_TRANSITION':
            this.currentPhase = message.phase
            break
            
          case 'VALIDATION_RESULT':
            useProjectStore().validationResults.push(message.result)
            break
        }
      }
      
      return ws
    }
  }
})
```

## 3. Composables for API Integration

### 3.1 GraphRAG Validation Composable

```typescript
// composables/useGraphRAGValidation.ts
export const useGraphRAGValidation = () => {
  const api = new OrchestrationApiService()
  const { $supabase } = useNuxtApp()
  
  const validateSection = async (
    content: string,
    sectionType: string,
    projectId: string
  ): Promise<SectionValidation> => {
    try {
      // Get project context from graph
      const context = await api.getProjectContext(projectId)
      
      // Perform validation
      const validation = await api.validateContent({
        content,
        context: {
          project_id: projectId,
          requirement_ids: context.requirement_ids
        },
        validation_level: 'all'
      })
      
      // Store validation result
      await $supabase
        .from('validation_traces')
        .insert({
          document_id: projectId,
          section_name: sectionType,
          validation_type: 'graphrag',
          result: validation,
          timestamp: new Date().toISOString()
        })
      
      return {
        isValid: validation.confidence_score >= 0.95,
        score: validation.confidence_score,
        corrections: validation.corrections,
        provenance: validation.provenance
      }
      
    } catch (error) {
      console.error('Validation failed:', error)
      throw error
    }
  }
  
  const validateDocument = async (
    document: PRDDocument
  ): Promise<DocumentValidation> => {
    const validations = await Promise.all(
      document.sections.map(section => 
        validateSection(section.content, section.name, document.project_id)
      )
    )
    
    const overallScore = validations.reduce(
      (sum, v) => sum + v.score, 0
    ) / validations.length
    
    return {
      overallScore,
      sectionValidations: validations,
      requiresReview: overallScore < 0.95
    }
  }
  
  return {
    validateSection,
    validateDocument
  }
}
```

### 3.2 Real-time Progress Composable

```typescript
// composables/useRealtimeProgress.ts
export const useRealtimeProgress = (projectId: string) => {
  const progress = ref<GenerationProgress>({
    phase: 'initializing',
    percentage: 0,
    currentAgent: null,
    message: '',
    estimatedTimeRemaining: null
  })
  
  const eventSource = ref<EventSource | null>(null)
  
  const startTracking = () => {
    eventSource.value = new EventSource(
      `/api/v1/progress/stream?project_id=${projectId}`
    )
    
    eventSource.value.addEventListener('progress', (event) => {
      const data = JSON.parse(event.data)
      progress.value = {
        ...progress.value,
        ...data
      }
    })
    
    eventSource.value.addEventListener('complete', () => {
      stopTracking()
    })
    
    eventSource.value.addEventListener('error', (error) => {
      console.error('SSE Error:', error)
      stopTracking()
    })
  }
  
  const stopTracking = () => {
    if (eventSource.value) {
      eventSource.value.close()
      eventSource.value = null
    }
  }
  
  onUnmounted(() => {
    stopTracking()
  })
  
  return {
    progress: readonly(progress),
    startTracking,
    stopTracking
  }
}
```

## 4. Component Integration Examples

### 4.1 Project Concept Component

```vue
<!-- components/ProjectConcept.vue -->
<template>
  <div class="max-w-4xl mx-auto p-6">
    <UCard>
      <template #header>
        <h2 class="text-2xl font-bold">Describe Your Project</h2>
      </template>
      
      <UTextarea
        v-model="conceptText"
        :rows="6"
        :maxlength="2000"
        placeholder="Describe your project idea in a sentence or paragraph..."
        :error="validationError"
        @input="validateInput"
      />
      
      <template #footer>
        <div class="flex justify-between items-center">
          <span class="text-sm text-gray-500">
            {{ conceptText.length }}/2000 characters
          </span>
          
          <UButton
            @click="submitConcept"
            :loading="isSubmitting"
            :disabled="!isValid"
            size="lg"
          >
            Continue to Analysis
          </UButton>
        </div>
      </template>
    </UCard>
    
    <!-- Similar Projects Preview -->
    <USlideover v-model="showSimilarProjects">
      <SimilarProjectsList 
        :projects="similarProjects"
        @select="loadFromSimilar"
      />
    </USlideover>
  </div>
</template>

<script setup lang="ts">
const projectStore = useProjectStore()
const { submitConcept: submitToApi } = projectStore

const conceptText = ref('')
const validationError = ref('')
const isSubmitting = ref(false)
const showSimilarProjects = ref(false)

const isValid = computed(() => 
  conceptText.value.length >= 100 && 
  conceptText.value.length <= 2000 &&
  !validationError.value
)

const validateInput = debounce(() => {
  if (conceptText.value.length < 100) {
    validationError.value = 'Please provide at least 100 characters'
  } else {
    validationError.value = ''
  }
}, 300)

const submitConcept = async () => {
  isSubmitting.value = true
  
  try {
    await submitToApi(conceptText.value)
    
    // Track analytics
    await useAnalytics().track('project_concept_submitted', {
      length: conceptText.value.length,
      has_industry_context: !!useUserStore().industryContext
    })
    
  } catch (error) {
    useToast().error('Failed to submit concept. Please try again.')
  } finally {
    isSubmitting.value = false
  }
}
</script>
```

### 4.2 PRD Generation Dashboard

```vue
<!-- components/PRDGenerationDashboard.vue -->
<template>
  <div class="grid grid-cols-12 gap-6">
    <!-- Progress Sidebar -->
    <div class="col-span-3">
      <ProjectSpine 
        :sections="sections"
        :currentPhase="currentPhase"
        @navigate="navigateToSection"
      />
    </div>
    
    <!-- Main Content Area -->
    <div class="col-span-9">
      <UCard>
        <!-- Generation Progress -->
        <GenerationProgress 
          v-if="isGenerating"
          :progress="generationProgress"
        />
        
        <!-- Section Editor -->
        <SectionEditor
          v-else
          :section="currentSection"
          :validation="currentValidation"
          @save="saveSection"
          @validate="validateSection"
        />
        
        <!-- Quality Metrics -->
        <QualityMetricsPanel
          :metrics="qualityMetrics"
          :threshold="8.0"
        />
      </UCard>
    </div>
  </div>
</template>

<script setup lang="ts">
const route = useRoute()
const projectId = route.params.projectId as string

const { progress, startTracking } = useRealtimeProgress(projectId)
const { validateSection } = useGraphRAGValidation()

const isGenerating = ref(false)
const currentSection = ref<PRDSection | null>(null)
const currentValidation = ref<ValidationResult | null>(null)

onMounted(async () => {
  // Initialize WebSocket for agent updates
  const ws = await useAgentStore().initializeWorkflow(projectId)
  
  // Start progress tracking
  startTracking()
  
  // Load project data
  await useProjectStore().loadProject(projectId)
})
</script>
```

## 5. Error Handling & Retry Logic

### 5.1 API Error Handler

```typescript
// utils/apiErrorHandler.ts
export class ApiErrorHandler {
  private static retryableStatuses = [408, 429, 500, 502, 503, 504]
  
  static async handleWithRetry<T>(
    fn: () => Promise<T>,
    options: RetryOptions = {}
  ): Promise<T> {
    const {
      maxAttempts = 3,
      delay = 1000,
      backoff = 2,
      onRetry
    } = options
    
    let lastError: Error | null = null
    
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        return await fn()
      } catch (error) {
        lastError = error as Error
        
        if (!this.isRetryable(error) || attempt === maxAttempts) {
          throw error
        }
        
        const waitTime = delay * Math.pow(backoff, attempt - 1)
        
        if (onRetry) {
          onRetry(attempt, waitTime)
        }
        
        await new Promise(resolve => setTimeout(resolve, waitTime))
      }
    }
    
    throw lastError
  }
  
  private static isRetryable(error: any): boolean {
    if (error.response?.status) {
      return this.retryableStatuses.includes(error.response.status)
    }
    
    // Network errors are retryable
    return error.code === 'ECONNABORTED' || 
           error.code === 'ETIMEDOUT' ||
           error.message?.includes('Network')
  }
}
```

## 6. WebSocket Integration for Real-time Updates

```typescript
// composables/useWebSocketConnection.ts
export const useWebSocketConnection = () => {
  const ws = ref<WebSocket | null>(null)
  const isConnected = ref(false)
  const reconnectAttempts = ref(0)
  const maxReconnectAttempts = 5
  
  const connect = (projectId: string) => {
    const config = useRuntimeConfig()
    const wsUrl = `${config.public.wsBaseUrl}/ws/project/${projectId}`
    
    ws.value = new WebSocket(wsUrl)
    
    ws.value.onopen = () => {
      isConnected.value = true
      reconnectAttempts.value = 0
      console.log('WebSocket connected')
    }
    
    ws.value.onmessage = (event) => {
      const message = JSON.parse(event.data)
      handleMessage(message)
    }
    
    ws.value.onerror = (error) => {
      console.error('WebSocket error:', error)
    }
    
    ws.value.onclose = () => {
      isConnected.value = false
      
      if (reconnectAttempts.value < maxReconnectAttempts) {
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.value), 30000)
        setTimeout(() => {
          reconnectAttempts.value++
          connect(projectId)
        }, delay)
      }
    }
  }
  
  const handleMessage = (message: WebSocketMessage) => {
    switch (message.type) {
      case 'AGENT_UPDATE':
        useAgentStore().updateAgent(message.data)
        break
        
      case 'VALIDATION_RESULT':
        useProjectStore().addValidationResult(message.data)
        break
        
      case 'GENERATION_PROGRESS':
        useProgressStore().updateProgress(message.data)
        break
        
      case 'ERROR':
        useToast().error(message.data.message)
        break
    }
  }
  
  const send = (message: any) => {
    if (ws.value?.readyState === WebSocket.OPEN) {
      ws.value.send(JSON.stringify(message))
    }
  }
  
  const disconnect = () => {
    if (ws.value) {
      ws.value.close()
      ws.value = null
    }
  }
  
  onUnmounted(() => {
    disconnect()
  })
  
  return {
    connect,
    send,
    disconnect,
    isConnected: readonly(isConnected)
  }
}
```

## 7. Authentication Integration

```typescript
// middleware/auth.global.ts
export default defineNuxtRouteMiddleware(async (to) => {
  const { $supabase } = useNuxtApp()
  const publicRoutes = ['/login', '/register', '/forgot-password']
  
  if (publicRoutes.includes(to.path)) {
    return
  }
  
  const { data: { session } } = await $supabase.auth.getSession()
  
  if (!session) {
    return navigateTo('/login')
  }
  
  // Check RBAC permissions
  const userRole = session.user?.user_metadata?.role
  const requiredRole = to.meta.requiredRole as string
  
  if (requiredRole && !hasPermission(userRole, requiredRole)) {
    throw createError({
      statusCode: 403,
      statusMessage: 'Insufficient permissions'
    })
  }
})
```

This comprehensive integration specification provides a complete bridge between your Nuxt.js 4 frontend and Python FastAPI backend, including:

1. **Robust API client** with retry logic and error handling
2. **State management** using Pinia stores
3. **Real-time updates** via WebSocket and SSE
4. **GraphRAG validation** integration
5. **Authentication flow** with Supabase and Auth0
6. **Progress tracking** for long-running operations
7. **Error recovery** and resilience patterns
8. **Component examples** showing practical usage

The architecture ensures smooth communication between frontend and backend while maintaining type safety, error resilience, and real-time capabilities essential for the AI-powered planning platform.
