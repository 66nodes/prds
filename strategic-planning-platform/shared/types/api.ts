/**
 * API Response Types
 * Standardized response formats, error handling, and HTTP interfaces
 */

// Standard API Response Wrapper
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: ApiError;
  metadata?: ResponseMetadata;
  timestamp: string;
  requestId: string;
}

export interface ResponseMetadata {
  page?: number;
  limit?: number;
  total?: number;
  totalPages?: number;
  hasNext?: boolean;
  hasPrev?: boolean;
  processingTime?: number;
  cacheHit?: boolean;
  version?: string;
}

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, any>;
  field?: string; // For validation errors
  timestamp: string;
  requestId?: string;
  stackTrace?: string; // Only in development
}

// HTTP Status Codes
export enum HttpStatus {
  OK = 200,
  CREATED = 201,
  NO_CONTENT = 204,
  BAD_REQUEST = 400,
  UNAUTHORIZED = 401,
  FORBIDDEN = 403,
  NOT_FOUND = 404,
  METHOD_NOT_ALLOWED = 405,
  CONFLICT = 409,
  UNPROCESSABLE_ENTITY = 422,
  TOO_MANY_REQUESTS = 429,
  INTERNAL_SERVER_ERROR = 500,
  BAD_GATEWAY = 502,
  SERVICE_UNAVAILABLE = 503,
  GATEWAY_TIMEOUT = 504
}

// Error Codes
export enum ApiErrorCode {
  // Authentication & Authorization
  INVALID_TOKEN = 'INVALID_TOKEN',
  TOKEN_EXPIRED = 'TOKEN_EXPIRED',
  INSUFFICIENT_PERMISSIONS = 'INSUFFICIENT_PERMISSIONS',
  USER_NOT_FOUND = 'USER_NOT_FOUND',
  INVALID_CREDENTIALS = 'INVALID_CREDENTIALS',
  
  // Validation
  VALIDATION_ERROR = 'VALIDATION_ERROR',
  MISSING_REQUIRED_FIELD = 'MISSING_REQUIRED_FIELD',
  INVALID_FORMAT = 'INVALID_FORMAT',
  VALUE_OUT_OF_RANGE = 'VALUE_OUT_OF_RANGE',
  
  // Business Logic
  PRD_NOT_FOUND = 'PRD_NOT_FOUND',
  PHASE_TRANSITION_INVALID = 'PHASE_TRANSITION_INVALID',
  SECTION_LOCKED = 'SECTION_LOCKED',
  VALIDATION_IN_PROGRESS = 'VALIDATION_IN_PROGRESS',
  INSUFFICIENT_CONTEXT = 'INSUFFICIENT_CONTEXT',
  
  // External Services
  GRAPHRAG_SERVICE_UNAVAILABLE = 'GRAPHRAG_SERVICE_UNAVAILABLE',
  LLM_SERVICE_ERROR = 'LLM_SERVICE_ERROR',
  DATABASE_CONNECTION_ERROR = 'DATABASE_CONNECTION_ERROR',
  EXTERNAL_API_ERROR = 'EXTERNAL_API_ERROR',
  
  // Rate Limiting & Quotas
  RATE_LIMIT_EXCEEDED = 'RATE_LIMIT_EXCEEDED',
  QUOTA_EXCEEDED = 'QUOTA_EXCEEDED',
  CONCURRENT_REQUESTS_EXCEEDED = 'CONCURRENT_REQUESTS_EXCEEDED',
  
  // System
  INTERNAL_SERVER_ERROR = 'INTERNAL_SERVER_ERROR',
  SERVICE_UNAVAILABLE = 'SERVICE_UNAVAILABLE',
  TIMEOUT_ERROR = 'TIMEOUT_ERROR',
  MAINTENANCE_MODE = 'MAINTENANCE_MODE'
}

// Pagination
export interface PaginationParams {
  page: number;
  limit: number;
  sort?: string;
  order?: 'asc' | 'desc';
}

export interface PaginatedResponse<T> {
  items: T[];
  pagination: PaginationInfo;
}

export interface PaginationInfo {
  currentPage: number;
  totalPages: number;
  totalItems: number;
  itemsPerPage: number;
  hasNext: boolean;
  hasPrev: boolean;
  nextPage?: number;
  prevPage?: number;
}

// Request Context
export interface RequestContext {
  userId: string;
  userRole: string;
  sessionId: string;
  ipAddress: string;
  userAgent: string;
  timestamp: string;
  requestId: string;
  correlationId?: string;
}

export interface ApiRequestOptions {
  timeout?: number;
  retries?: number;
  headers?: Record<string, string>;
  cache?: boolean;
  cacheTtl?: number;
}

// Health Check
export interface HealthCheckResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  timestamp: string;
  uptime: number;
  services: ServiceHealth[];
  metrics: HealthMetrics;
}

export interface ServiceHealth {
  name: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  responseTime?: number;
  lastCheck: string;
  error?: string;
  dependencies?: ServiceHealth[];
}

export interface HealthMetrics {
  requestsPerSecond: number;
  averageResponseTime: number;
  errorRate: number;
  activeConnections: number;
  memoryUsage: number;
  cpuUsage: number;
}

// Analytics & Metrics
export interface MetricsResponse {
  timeRange: {
    start: string;
    end: string;
  };
  metrics: Metric[];
  aggregations: Record<string, number>;
}

export interface Metric {
  name: string;
  value: number;
  unit: string;
  timestamp: string;
  labels?: Record<string, string>;
}

// File Upload
export interface FileUploadResponse {
  fileId: string;
  filename: string;
  size: number;
  mimeType: string;
  url: string;
  thumbnailUrl?: string;
  metadata: FileMetadata;
  uploadedAt: string;
}

export interface FileMetadata {
  originalName: string;
  encoding: string;
  fieldname: string;
  width?: number;
  height?: number;
  duration?: number;
  checksum: string;
}

// Bulk Operations
export interface BulkOperationRequest<T> {
  operations: BulkOperation<T>[];
  options?: BulkOptions;
}

export interface BulkOperation<T> {
  operation: 'create' | 'update' | 'delete';
  id?: string;
  data: T;
}

export interface BulkOptions {
  continueOnError?: boolean;
  validateOnly?: boolean;
  batchSize?: number;
}

export interface BulkOperationResponse<T> {
  success: boolean;
  results: BulkOperationResult<T>[];
  summary: BulkOperationSummary;
  errors: BulkOperationError[];
}

export interface BulkOperationResult<T> {
  operation: 'create' | 'update' | 'delete';
  success: boolean;
  data?: T;
  error?: string;
  index: number;
}

export interface BulkOperationSummary {
  total: number;
  successful: number;
  failed: number;
  skipped: number;
  processingTime: number;
}

export interface BulkOperationError {
  index: number;
  operation: 'create' | 'update' | 'delete';
  error: string;
  data?: any;
}

// Search & Filtering
export interface SearchRequest {
  query: string;
  filters: SearchFilter[];
  sort?: SearchSort[];
  pagination: PaginationParams;
  options?: SearchOptions;
}

export interface SearchFilter {
  field: string;
  operator: FilterOperator;
  value: any;
  values?: any[]; // For IN/NOT_IN operators
}

export enum FilterOperator {
  EQUALS = 'eq',
  NOT_EQUALS = 'ne',
  GREATER_THAN = 'gt',
  GREATER_THAN_OR_EQUAL = 'gte',
  LESS_THAN = 'lt',
  LESS_THAN_OR_EQUAL = 'lte',
  IN = 'in',
  NOT_IN = 'nin',
  CONTAINS = 'contains',
  STARTS_WITH = 'starts_with',
  ENDS_WITH = 'ends_with',
  IS_NULL = 'is_null',
  IS_NOT_NULL = 'is_not_null'
}

export interface SearchSort {
  field: string;
  direction: 'asc' | 'desc';
}

export interface SearchOptions {
  highlightMatches?: boolean;
  includeSnippets?: boolean;
  fuzzyMatching?: boolean;
  synonyms?: boolean;
  facets?: string[];
}

export interface SearchResponse<T> {
  results: SearchResult<T>[];
  totalResults: number;
  facets?: SearchFacet[];
  suggestions?: string[];
  queryTime: number;
  pagination: PaginationInfo;
}

export interface SearchResult<T> {
  item: T;
  score: number;
  highlights?: Record<string, string[]>;
  snippet?: string;
}

export interface SearchFacet {
  field: string;
  values: FacetValue[];
}

export interface FacetValue {
  value: string;
  count: number;
  selected: boolean;
}

// Export Operations
export interface ExportRequest {
  format: ExportFormat;
  filters?: SearchFilter[];
  fields?: string[];
  options?: ExportOptions;
}

export enum ExportFormat {
  CSV = 'csv',
  XLSX = 'xlsx',
  JSON = 'json',
  PDF = 'pdf'
}

export interface ExportOptions {
  includeHeaders?: boolean;
  dateFormat?: string;
  delimiter?: string; // For CSV
  sheetName?: string; // For XLSX
  template?: string; // For PDF
}

export interface ExportResponse {
  exportId: string;
  status: ExportStatus;
  format: ExportFormat;
  downloadUrl?: string;
  filename?: string;
  size?: number;
  recordCount?: number;
  createdAt: string;
  expiresAt: string;
}

export enum ExportStatus {
  QUEUED = 'queued',
  PROCESSING = 'processing',
  COMPLETED = 'completed',
  FAILED = 'failed',
  EXPIRED = 'expired'
}

// Audit Trail
export interface AuditLogEntry {
  id: string;
  userId: string;
  userEmail: string;
  action: string;
  resource: string;
  resourceId: string;
  changes?: Record<string, { old: any; new: any }>;
  metadata: Record<string, any>;
  ipAddress: string;
  userAgent: string;
  timestamp: string;
}

export interface AuditLogRequest {
  resource?: string;
  resourceId?: string;
  userId?: string;
  actions?: string[];
  dateRange?: {
    start: string;
    end: string;
  };
  pagination: PaginationParams;
}

// Rate Limiting
export interface RateLimitInfo {
  limit: number;
  remaining: number;
  resetTime: string;
  retryAfter?: number;
}

export interface RateLimitHeaders {
  'X-RateLimit-Limit': string;
  'X-RateLimit-Remaining': string;
  'X-RateLimit-Reset': string;
  'Retry-After'?: string;
}

// API Client Interface
export interface ApiClient {
  // Base HTTP methods
  get<T>(url: string, options?: ApiRequestOptions): Promise<ApiResponse<T>>;
  post<T>(url: string, data?: any, options?: ApiRequestOptions): Promise<ApiResponse<T>>;
  put<T>(url: string, data?: any, options?: ApiRequestOptions): Promise<ApiResponse<T>>;
  patch<T>(url: string, data?: any, options?: ApiRequestOptions): Promise<ApiResponse<T>>;
  delete<T>(url: string, options?: ApiRequestOptions): Promise<ApiResponse<T>>;
  
  // Specialized methods
  upload<T>(url: string, file: File, options?: ApiRequestOptions): Promise<ApiResponse<T>>;
  download(url: string, options?: ApiRequestOptions): Promise<Blob>;
  
  // Configuration
  setBaseUrl(url: string): void;
  setAuthToken(token: string): void;
  setDefaultHeaders(headers: Record<string, string>): void;
  
  // Interceptors
  addRequestInterceptor(interceptor: RequestInterceptor): void;
  addResponseInterceptor(interceptor: ResponseInterceptor): void;
}

export type RequestInterceptor = (request: RequestConfig) => RequestConfig | Promise<RequestConfig>;
export type ResponseInterceptor = (response: ApiResponse) => ApiResponse | Promise<ApiResponse>;

export interface RequestConfig {
  url: string;
  method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  headers: Record<string, string>;
  data?: any;
  params?: Record<string, any>;
  timeout?: number;
}

// WebHook Types
export interface WebHookPayload<T = any> {
  event: string;
  data: T;
  timestamp: string;
  webhookId: string;
  signature: string;
  version: string;
}

export interface WebHookConfig {
  url: string;
  events: string[];
  secret: string;
  active: boolean;
  retryPolicy: WebHookRetryPolicy;
}

export interface WebHookRetryPolicy {
  maxAttempts: number;
  backoffStrategy: 'exponential' | 'linear';
  initialDelay: number;
  maxDelay: number;
}