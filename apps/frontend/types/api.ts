// Frontend API Types and Interfaces
// Links to backend API schemas for type safety

export interface ApiResponse<T = any> {
  success: boolean
  data: T
  message?: string
  errors?: string[]
  timestamp?: string
}

export interface PaginatedResponse<T> {
  data: T[]
  pagination: {
    page: number
    pageSize: number
    total: number
    totalPages: number
    hasMore: boolean
  }
}

// Authentication Types
export interface AuthTokens {
  accessToken: string
  refreshToken: string
  expiresIn: number
  tokenType?: string
}

export interface LoginRequest {
  email: string
  password: string
  rememberMe?: boolean
}

export interface RegisterRequest {
  email: string
  password: string
  name: string
  company?: string
  role?: string
}

export interface PasswordResetRequest {
  email: string
  token?: string
  newPassword?: string
}

// User Types
export interface UserProfile {
  id: string
  email: string
  name: string
  avatar?: string
  role: 'admin' | 'user' | 'auditor' | 'guest'
  company?: string
  title?: string
  department?: string
  preferences: UserPreferences
  isActive: boolean
  createdAt: string
  updatedAt: string
  lastLogin?: string
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'system'
  language: string
  timezone: string
  dateFormat: string
  notifications: NotificationSettings
}

// Notification Types
export interface NotificationSettings {
  email: boolean
  push: boolean
  inApp: boolean
  frequency: 'realtime' | 'hourly' | 'daily' | 'weekly'
  types: {
    agent: boolean
    prd: boolean
    project: boolean
    system: boolean
  }
}

// PRD Types (Product Requirements Document)
export interface PRDRequest {
  title: string
  description: string
  projectId: string
  requirements?: string[]
  constraints?: string[]
  targetAudience?: string
  successMetrics?: string[]
  priority?: 'low' | 'medium' | 'high' | 'critical'
  estimatedEffort?: string
  timeline?: string
}

export interface PRDResponse {
  id: string
  title: string
  content: string
  hallucinationRate: number
  confidenceScore?: number
  validationResult?: ValidationResult
  metadata: PRDMetadata
  createdAt: string
  updatedAt: string
}

export interface PRDMetadata {
  version: string
  status: PRDStatus
  author: string
  reviewers: string[]
  tags: string[]
  estimatedEffort?: string
  priority?: Priority
  categories?: string[]
}

export type PRDStatus =
  | 'draft'
  | 'review'
  | 'approved'
  | 'published'
  | 'archived'

export type Priority = 'low' | 'medium' | 'high' | 'critical'

// Agent Types
export interface AgentInfo {
  id: string
  name: string
  type: AgentType
  status: AgentStatus
  capabilities: string[]
  performance: AgentPerformance
  configuration?: Record<string, any>
  lastActive?: string
  description?: string
}

export type AgentType = keyof typeof AGENT_TYPES

export const AGENT_TYPES = {
  CONTEXT_MANAGER: 'context_manager',
  TASK_EXECUTOR: 'task_executor',
  JUDGE_AGENT: 'judge_agent',
  DOCUMENTATION_SPECIALIST: 'documentation_specialist',
  CODE_REVIEWER: 'code_reviewer',
  ARCHITECT: 'architect',
  PERFORMANCE_OPTIMIZER: 'performance_optimizer',
  SECURITY_AUDITOR: 'security_auditor',
  UI_DESIGNER: 'ui_designer',
  API_DESIGNER: 'api_designer'
} as const

export type AgentStatus =
  | 'idle'
  | 'processing'
  | 'error'
  | 'offline'
  | 'maintenance'

export interface AgentPerformance {
  totalTasks: number
  successfulTasks: number
  failedTasks: number
  averageResponseTime: number
  averageAccuracy: number
  totalProcessingTime: number
  taskCompletionRate: number
  lastPerformanceCheck?: string
}

// Project Types
export interface Project {
  id: string
  name: string
  description: string
  status: ProjectStatus
  priority: Priority
  owner: Pick<UserProfile, 'id' | 'name' | 'email'>
  members: ProjectMember[]
  prds: PRDSummary[]
  metrics: ProjectMetrics
  tags: string[]
  createdAt: string
  updatedAt: string
  dueDate?: string
  budget?: number
  progress: number
}

export type ProjectStatus =
  | 'planning'
  | 'active'
  | 'review'
  | 'completed'
  | 'on_hold'
  | 'cancelled'

export interface ProjectMember {
  user: Pick<UserProfile, 'id' | 'name' | 'email' | 'role'>
  role: ProjectRole
  joinedAt: string
  permissions: string[]
}

export type ProjectRole = 'owner' | 'admin' | 'contributor' | 'viewer'

export interface PRDSummary {
  id: string
  title: string
  status: PRDStatus
  author: string
  lastModified: string
  progress: number
}

export interface ProjectMetrics {
  totalPRDs: number
  completedPRDs: number
  averageHallucinationRate: number
  timeToCompletion: number
  automationRate: number
  userSatisfaction: number
  budgetUtilization?: number
}

// Validation Types
export interface ValidationRequest {
  content: string
  context?: {
    expectedEntities?: string[]
    hallucinationThreshold?: number
    contentType?: string
    metadata?: Record<string, any>
  }
  options?: {
    includeGraphRAG?: boolean
    includeSpellCheck?: boolean
    includeGrammarCheck?: boolean
    strictMode?: boolean
  }
}

export interface ValidationResult {
  id: string
  valid: boolean
  overallScore: number
  hallucinationRate: number
  confidence: number
  issues: ValidationIssue[]
  suggestions: ValidationSuggestion[]
  graphEvidence?: GraphEvidence[]
  processingTime: number
  validatedAt: string
}

export interface ValidationIssue {
  id: string
  type: ValidationIssueType
  severity: ValidationSeverity
  message: string
  location?: {
    line: number
    column?: number
    section?: string
  }
  source?: string
  solution?: string
  metadata?: Record<string, any>
}

export type ValidationIssueType =
  | 'hallucination'
  | 'inconsistency'
  | 'missing_evidence'
  | 'contradiction'
  | 'spelling'
  | 'grammar'
  | 'factual_error'
  | 'logical_error'

export type ValidationSeverity =
  | 'error'
  | 'warning'
  | 'info'

export interface ValidationSuggestion {
  id: string
  type: 'correction' | 'improvement' | 'addition' | 'removal'
  message: string
  priority: Priority
  confidence: number
  implementation?: {
    description: string
    codeSnippet?: string
    effort?: 'low' | 'medium' | 'high'
  }
}

// GraphRAG Types
export interface GraphEvidence {
  id: string
  nodeId: string
  nodeType: string
  content: string
  confidence: number
  source: string
  links: GraphLink[]
  metadata: Record<string, any>
}

export interface GraphLink {
  targetId: string
  relationship: string
  strength: number
  direction: 'outgoing' | 'incoming'
}

// WebSocket Types
export type WebSocketMessageType =
  | 'agent:response'
  | 'agent:status'
  | 'validation:start'
  | 'validation:progress'
  | 'validation:complete'
  | 'project:update'
  | 'user:presence'
  | 'error'
  | 'heartbeat'
  | 'notification'

export interface WebSocketMessage {
  type: WebSocketMessageType
  payload: any
  id: string
  timestamp: string
  source?: string
  target?: string
}

export interface WebSocketConfig {
  url: string
  protocols?: string[]
  heartbeat?: {
    interval: number
    timeout: number
  }
  reconnect?: {
    enabled: boolean
    maxAttempts: number
    initialDelay: number
    maxDelay: number
  }
}

// State Management Types
export interface GlobalState {
  user: UserProfile | null
  auth: AuthState
  isInitializing: boolean
  error: string | null
  version: string
  environment: 'development' | 'staging' | 'production'
}

export interface AuthState {
  isAuthenticated: boolean
  isLoading: boolean
  error: string | null
  user: UserProfile | null
  tokens: AuthTokens | null
}

// Error Types
export class ApiError extends Error {
  constructor(
    message: string,
    public statusCode: number = 500,
    public code?: string
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

export class NetworkError extends ApiError {
  constructor(message: string) {
    super(message, 0, 'NETWORK_ERROR')
  }
}

export class ValidationError extends ApiError {
  constructor(
    message: string,
    public field: string
  ) {
    super(message, 400, 'VALIDATION_ERROR')
  }
}

export class AuthenticationError extends ApiError {
  constructor(message = 'Authentication required') {
    super(message, 401, 'AUTHENTICATION_ERROR')
  }
}

export class AuthorizationError extends ApiError {
  constructor(message = 'Permission denied') {
    super(message, 403, 'AUTHORIZATION_ERROR')
  }
}

export class NotFoundError extends ApiError {
  constructor(resource = 'resource') {
    super(`${resource} not found`, 404, 'NOT_FOUND_ERROR')
  }
}

// Utility Types
export type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>
export type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P]
}
export type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends object ? DeepReadonly<T[P]> : T[P]
}

// Form Types
export interface FormField {
  name: string
  label: string
  type: FormFieldType
  value?: any
  required?: boolean
  placeholder?: string
  description?: string
  validation?: FormValidation[]
  options?: FormOption[]
  disabled?: boolean
  hidden?: boolean
}

export type FormFieldType =
  | 'text'
  | 'email'
  | 'password'
  | 'textarea'
  | 'number'
  | 'date'
  | 'time'
  | 'select'
  | 'multiselect'
  | 'checkbox'
  | 'radio'
  | 'file'

export interface FormValidation {
  rule: string
  value?: any
  message: string
}

export interface FormOption {
  label: string
  value: string | number
  description?: string
  disabled?: boolean
}

// Theme Types
export interface Theme {
  name: string
  colors: {
    primary: string
    secondary: string
    accent: string
    background: string
    foreground: string
    muted: string
    success: string
    warning: string
    error: string
    info: string
  }
  fontSize: Record<string, string>
  spacing: Record<string, string>
  borderRadius: Record<string, string>
}

// Plugin Types
export interface PluginConfig {
  enabled: boolean
  priority: number
  config: Record<string, any>
}

export interface PluginHook<T = any> {
  name: string
  before?: (...args: any[]) => Promise<T> | T
  after?: (...args: any[]) => Promise<T> | T
  execute?: (...args: any[]) => Promise<T> | T
}

// Analytics Types
export interface AnalyticsEvent {
  name: string
  properties?: Record<string, any>
  userId?: string
  sessionId?: string
  timestamp: Date
}

export interface UsageMetrics {
  totalRequests: number
  totalErrors: number
  averageResponseTime: number
  uptime: number
  last24Hours: {
    requests: number
    errors: number
    avgResponseTime: number
  }
}

// Export all types
export type * from './store'
export type * from './websocket'
export type * from './conversation'