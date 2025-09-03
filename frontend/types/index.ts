// Core Types for Strategic Planning Platform

// User and Authentication Types
export interface User {
  id: string
  email: string
  name: string
  role: UserRole
  avatar?: string
  createdAt: string
  updatedAt: string
}

export enum UserRole {
  ADMIN = 'admin',
  USER = 'user',
  VIEWER = 'viewer'
}

export interface AuthTokens {
  accessToken: string
  refreshToken: string
  expiresIn: number
}

export interface LoginRequest {
  email: string
  password: string
}

export interface RegisterRequest {
  email: string
  password: string
  name: string
}

// PRD Types
export interface PRDRequest {
  title: string
  description: string
  projectId: string
  requirements?: string[]
  constraints?: string[]
  targetAudience?: string
  successMetrics?: string[]
}

export interface PRDResponse {
  id: string
  title: string
  content: string
  hallucination_rate: number
  validation_score: number
  graph_evidence?: GraphEvidence[]
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
}

export enum PRDStatus {
  DRAFT = 'draft',
  IN_REVIEW = 'in_review',
  APPROVED = 'approved',
  PUBLISHED = 'published',
  ARCHIVED = 'archived'
}

export enum Priority {
  CRITICAL = 'critical',
  HIGH = 'high',
  MEDIUM = 'medium',
  LOW = 'low'
}

// Agent Types
export interface Agent {
  id: string
  name: string
  type: AgentType
  status: AgentStatus
  capabilities: string[]
  performance: AgentPerformance
  configuration: Record<string, any>
}

export enum AgentType {
  CONTEXT_MANAGER = 'context_manager',
  DRAFT_AGENT = 'draft_agent',
  JUDGE_AGENT = 'judge_agent',
  TASK_EXECUTOR = 'task_executor',
  VALIDATION_AGENT = 'validation_agent'
}

export enum AgentStatus {
  IDLE = 'idle',
  PROCESSING = 'processing',
  ERROR = 'error',
  OFFLINE = 'offline'
}

export interface AgentPerformance {
  tasksCompleted: number
  averageResponseTime: number
  successRate: number
  lastActive: string
}

// Project Types
export interface Project {
  id: string
  name: string
  description: string
  status: ProjectStatus
  owner: User
  members: ProjectMember[]
  prds: PRDSummary[]
  metrics: ProjectMetrics
  createdAt: string
  updatedAt: string
}

export enum ProjectStatus {
  PLANNING = 'planning',
  IN_PROGRESS = 'in_progress',
  REVIEW = 'review',
  COMPLETED = 'completed',
  ON_HOLD = 'on_hold'
}

export interface ProjectMember {
  user: User
  role: ProjectRole
  joinedAt: string
}

export enum ProjectRole {
  OWNER = 'owner',
  ADMIN = 'admin',
  CONTRIBUTOR = 'contributor',
  VIEWER = 'viewer'
}

export interface PRDSummary {
  id: string
  title: string
  status: PRDStatus
  lastModified: string
}

export interface ProjectMetrics {
  totalPRDs: number
  completedPRDs: number
  averageHallucinationRate: number
  timeToCompletion: number
}

// GraphRAG Types
export interface GraphEvidence {
  nodeId: string
  nodeType: string
  content: string
  confidence: number
  relationships: GraphRelationship[]
}

export interface GraphRelationship {
  type: string
  targetNodeId: string
  strength: number
}

export interface ValidationResult {
  content: string
  hallucinationRate: number
  validationScore: number
  graphEvidence: GraphEvidence[]
  issues: ValidationIssue[]
}

export interface ValidationIssue {
  type: IssueType
  severity: IssueSeverity
  message: string
  location?: string
}

export enum IssueType {
  HALLUCINATION = 'hallucination',
  INCONSISTENCY = 'inconsistency',
  MISSING_EVIDENCE = 'missing_evidence',
  CONTRADICTION = 'contradiction'
}

export enum IssueSeverity {
  ERROR = 'error',
  WARNING = 'warning',
  INFO = 'info'
}

// Dashboard & Analytics Types
export interface DashboardData {
  summary: DashboardSummary
  recentActivity: Activity[]
  performanceMetrics: PerformanceMetrics
  charts: ChartData[]
}

export interface DashboardSummary {
  totalProjects: number
  activePRDs: number
  totalAgents: number
  activeUsers: number
  averageResponseTime: number
  systemHealth: SystemHealth
}

export enum SystemHealth {
  HEALTHY = 'healthy',
  DEGRADED = 'degraded',
  DOWN = 'down'
}

export interface Activity {
  id: string
  type: ActivityType
  description: string
  user: User
  timestamp: string
  metadata?: Record<string, any>
}

export enum ActivityType {
  PRD_CREATED = 'prd_created',
  PRD_UPDATED = 'prd_updated',
  PRD_APPROVED = 'prd_approved',
  PROJECT_CREATED = 'project_created',
  USER_JOINED = 'user_joined',
  AGENT_TASK = 'agent_task'
}

export interface PerformanceMetrics {
  apiLatency: number[]
  hallucinationRates: number[]
  agentUtilization: number
  cacheHitRate: number
  errorRate: number
}

export interface ChartData {
  label: string
  datasets: ChartDataset[]
}

export interface ChartDataset {
  label: string
  data: number[]
  backgroundColor?: string
  borderColor?: string
}

// WebSocket Types
export interface WebSocketMessage {
  type: WebSocketMessageType
  payload: any
  timestamp: string
}

export enum WebSocketMessageType {
  CONNECT = 'connect',
  DISCONNECT = 'disconnect',
  PRD_UPDATE = 'prd_update',
  AGENT_STATUS = 'agent_status',
  VALIDATION_RESULT = 'validation_result',
  ERROR = 'error',
  HEARTBEAT = 'heartbeat'
}

// API Response Types
export interface ApiResponse<T> {
  data: T
  success: boolean
  message?: string
  errors?: string[]
}

export interface PaginatedResponse<T> {
  data: T[]
  total: number
  page: number
  pageSize: number
  hasMore: boolean
}

// Form Types
export interface FormField {
  name: string
  label: string
  type: FormFieldType
  value?: any
  required?: boolean
  placeholder?: string
  validation?: ValidationRule[]
  options?: SelectOption[]
}

export enum FormFieldType {
  TEXT = 'text',
  EMAIL = 'email',
  PASSWORD = 'password',
  TEXTAREA = 'textarea',
  SELECT = 'select',
  CHECKBOX = 'checkbox',
  RADIO = 'radio',
  DATE = 'date',
  NUMBER = 'number'
}

export interface ValidationRule {
  type: ValidationType
  value?: any
  message: string
}

export enum ValidationType {
  REQUIRED = 'required',
  MIN_LENGTH = 'min_length',
  MAX_LENGTH = 'max_length',
  PATTERN = 'pattern',
  EMAIL = 'email',
  CUSTOM = 'custom'
}

export interface SelectOption {
  label: string
  value: string | number
  disabled?: boolean
}

// Notification Types
export interface Notification {
  id: string
  type: NotificationType
  title: string
  message: string
  timestamp: string
  read: boolean
  actionUrl?: string
}

export enum NotificationType {
  SUCCESS = 'success',
  INFO = 'info',
  WARNING = 'warning',
  ERROR = 'error'
}

// Settings Types
export interface UserSettings {
  theme: 'light' | 'dark' | 'system'
  language: string
  notifications: NotificationSettings
  privacy: PrivacySettings
}

export interface NotificationSettings {
  email: boolean
  push: boolean
  inApp: boolean
  frequency: 'realtime' | 'hourly' | 'daily'
}

export interface PrivacySettings {
  profileVisibility: 'public' | 'private' | 'team'
  activityTracking: boolean
  dataSharing: boolean
}