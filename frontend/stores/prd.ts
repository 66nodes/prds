import { defineStore } from 'pinia'
import type { PRDRequest, PRDResponse, PRDStatus, ValidationResult } from '~/types'

interface PRDState {
  prds: PRDResponse[]
  currentPRD: PRDResponse | null
  isGenerating: boolean
  isValidating: boolean
  isLoading: boolean
  error: string | null
  validationResult: ValidationResult | null
  generationProgress: number
}

export const usePRDStore = defineStore('prd', {
  state: (): PRDState => ({
    prds: [],
    currentPRD: null,
    isGenerating: false,
    isValidating: false,
    isLoading: false,
    error: null,
    validationResult: null,
    generationProgress: 0
  }),

  getters: {
    allPRDs: (state) => state.prds,
    
    activePRDs: (state) => 
      state.prds.filter(prd => prd.metadata.status !== 'archived'),
    
    draftPRDs: (state) => 
      state.prds.filter(prd => prd.metadata.status === 'draft'),
    
    publishedPRDs: (state) => 
      state.prds.filter(prd => prd.metadata.status === 'published'),
    
    prdById: (state) => (id: string) => 
      state.prds.find(prd => prd.id === id),
    
    currentPRDId: (state) => state.currentPRD?.id,
    
    averageHallucinationRate: (state) => {
      if (state.prds.length === 0) return 0
      const sum = state.prds.reduce((acc, prd) => acc + prd.hallucination_rate, 0)
      return sum / state.prds.length
    },
    
    isGeneratingPRD: (state) => state.isGenerating,
    isValidatingPRD: (state) => state.isValidating,
    hasError: (state) => !!state.error
  },

  actions: {
    setPRDs(prds: PRDResponse[]) {
      this.prds = prds
    },

    setCurrentPRD(prd: PRDResponse | null) {
      this.currentPRD = prd
    },

    addPRD(prd: PRDResponse) {
      this.prds.push(prd)
    },

    updatePRD(id: string, updates: Partial<PRDResponse>) {
      const index = this.prds.findIndex(prd => prd.id === id)
      if (index !== -1) {
        this.prds[index] = { ...this.prds[index], ...updates }
        
        if (this.currentPRD?.id === id) {
          this.currentPRD = { ...this.currentPRD, ...updates }
        }
      }
    },

    removePRD(id: string) {
      this.prds = this.prds.filter(prd => prd.id !== id)
      
      if (this.currentPRD?.id === id) {
        this.currentPRD = null
      }
    },

    setLoading(loading: boolean) {
      this.isLoading = loading
    },

    setGenerating(generating: boolean) {
      this.isGenerating = generating
      if (!generating) {
        this.generationProgress = 0
      }
    },

    setValidating(validating: boolean) {
      this.isValidating = validating
    },

    setError(error: string | null) {
      this.error = error
    },

    setValidationResult(result: ValidationResult | null) {
      this.validationResult = result
    },

    setGenerationProgress(progress: number) {
      this.generationProgress = Math.min(100, Math.max(0, progress))
    },

    async fetchPRDs(projectId: string) {
      this.setLoading(true)
      this.setError(null)
      
      try {
        const { get } = useApiClient()
        const prds = await get<PRDResponse[]>(`/projects/${projectId}/prds`)
        this.setPRDs(prds)
      } catch (error: any) {
        this.setError(error.message || 'Failed to fetch PRDs')
        throw error
      } finally {
        this.setLoading(false)
      }
    },

    async fetchPRD(projectId: string, prdId: string) {
      this.setLoading(true)
      this.setError(null)
      
      try {
        const { get } = useApiClient()
        const prd = await get<PRDResponse>(`/projects/${projectId}/prds/${prdId}`)
        this.setCurrentPRD(prd)
        
        // Update in list if exists
        const index = this.prds.findIndex(p => p.id === prdId)
        if (index !== -1) {
          this.prds[index] = prd
        } else {
          this.addPRD(prd)
        }
        
        return prd
      } catch (error: any) {
        this.setError(error.message || 'Failed to fetch PRD')
        throw error
      } finally {
        this.setLoading(false)
      }
    },

    async generatePRD(projectId: string, request: PRDRequest) {
      this.setGenerating(true)
      this.setError(null)
      this.setGenerationProgress(0)
      
      try {
        const { post } = useApiClient()
        const ws = useWebSocket()
        
        // Subscribe to real-time updates
        ws.on('prd_update' as any, (data: any) => {
          if (data.type === 'progress') {
            this.setGenerationProgress(data.progress)
          }
        })
        
        // Simulate progress for demo
        const progressInterval = setInterval(() => {
          this.setGenerationProgress(prev => Math.min(prev + 10, 90))
        }, 1000)
        
        const prd = await post<PRDResponse>(`/projects/${projectId}/prds/generate`, request)
        
        clearInterval(progressInterval)
        this.setGenerationProgress(100)
        
        // Validate the generated PRD
        await this.validatePRD(projectId, prd.id)
        
        this.addPRD(prd)
        this.setCurrentPRD(prd)
        
        return prd
      } catch (error: any) {
        this.setError(error.message || 'Failed to generate PRD')
        throw error
      } finally {
        this.setGenerating(false)
        this.setGenerationProgress(0)
      }
    },

    async validatePRD(projectId: string, prdId: string) {
      this.setValidating(true)
      this.setError(null)
      
      try {
        const { post } = useApiClient()
        const result = await post<ValidationResult>(
          `/projects/${projectId}/prds/${prdId}/validate`
        )
        
        this.setValidationResult(result)
        
        // Update PRD with validation results
        this.updatePRD(prdId, {
          hallucination_rate: result.hallucinationRate,
          validation_score: result.validationScore,
          graph_evidence: result.graphEvidence
        })
        
        return result
      } catch (error: any) {
        this.setError(error.message || 'Failed to validate PRD')
        throw error
      } finally {
        this.setValidating(false)
      }
    },

    async updatePRDStatus(projectId: string, prdId: string, status: PRDStatus) {
      this.setLoading(true)
      this.setError(null)
      
      try {
        const { patch } = useApiClient()
        const prd = await patch<PRDResponse>(
          `/projects/${projectId}/prds/${prdId}`,
          { status }
        )
        
        this.updatePRD(prdId, prd)
        
        if (this.currentPRD?.id === prdId) {
          this.setCurrentPRD(prd)
        }
        
        return prd
      } catch (error: any) {
        this.setError(error.message || 'Failed to update PRD status')
        throw error
      } finally {
        this.setLoading(false)
      }
    },

    async deletePRD(projectId: string, prdId: string) {
      this.setLoading(true)
      this.setError(null)
      
      try {
        const { delete: del } = useApiClient()
        await del(`/projects/${projectId}/prds/${prdId}`)
        this.removePRD(prdId)
      } catch (error: any) {
        this.setError(error.message || 'Failed to delete PRD')
        throw error
      } finally {
        this.setLoading(false)
      }
    },

    async exportPRD(projectId: string, prdId: string, format: 'pdf' | 'docx' | 'markdown') {
      this.setLoading(true)
      this.setError(null)
      
      try {
        const { download } = useApiClient()
        const prd = this.prdById(prdId)
        const filename = `${prd?.title || 'prd'}.${format}`
        
        await download(
          `/projects/${projectId}/prds/${prdId}/export?format=${format}`,
          filename
        )
      } catch (error: any) {
        this.setError(error.message || 'Failed to export PRD')
        throw error
      } finally {
        this.setLoading(false)
      }
    },

    reset() {
      this.prds = []
      this.currentPRD = null
      this.isGenerating = false
      this.isValidating = false
      this.isLoading = false
      this.error = null
      this.validationResult = null
      this.generationProgress = 0
    }
  }
})