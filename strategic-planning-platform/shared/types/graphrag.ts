/**
 * GraphRAG Integration Types
 * Confidence scores, validation responses, and knowledge graph operations
 */

// Core GraphRAG Types
export enum ValidationType {
  ENTITY = 'entity',
  COMMUNITY = 'community',
  GLOBAL = 'global',
  CONSISTENCY = 'consistency'
}

export enum ConfidenceLevel {
  VERY_HIGH = 'very_high', // 0.9-1.0
  HIGH = 'high',          // 0.8-0.89
  MEDIUM = 'medium',      // 0.6-0.79
  LOW = 'low',           // 0.4-0.59
  VERY_LOW = 'very_low'  // 0.0-0.39
}

export interface ValidationRequest {
  content: string;
  context: ValidationContext;
  validationTypes: ValidationType[];
  options: ValidationOptions;
}

export interface ValidationContext {
  projectId?: string;
  prdId?: string;
  sectionType?: string;
  userId: string;
  organizationalContext?: OrganizationalContext;
  previousSections?: string[];
}

export interface OrganizationalContext {
  industryDomain: string;
  companySize: 'startup' | 'small' | 'medium' | 'large' | 'enterprise';
  technologyStack: string[];
  businessObjectives: string[];
  complianceRequirements: string[];
  culturalFactors: string[];
}

export interface ValidationOptions {
  confidenceThreshold: number;
  enableCorrections: boolean;
  includeAlternatives: boolean;
  maxAlternatives: number;
  timeoutMs: number;
}

export interface ValidationResponse {
  overallConfidence: number;
  confidenceLevel: ConfidenceLevel;
  isValid: boolean;
  validationResults: ValidationResult[];
  corrections?: CorrectionSuggestion[];
  alternatives?: AlternativeContent[];
  processingTime: number;
  timestamp: string;
}

export interface ValidationResult {
  type: ValidationType;
  confidence: number;
  isValid: boolean;
  evidence: Evidence[];
  issues: ValidationIssue[];
  metadata: ValidationMetadata;
}

export interface Evidence {
  id: string;
  source: EvidenceSource;
  content: string;
  relevanceScore: number;
  sourceMetadata: SourceMetadata;
  extractedAt: string;
}

export enum EvidenceSource {
  KNOWLEDGE_GRAPH = 'knowledge_graph',
  HISTORICAL_PRDS = 'historical_prds',
  ORGANIZATIONAL_DOCS = 'organizational_docs',
  INDUSTRY_PATTERNS = 'industry_patterns',
  EXTERNAL_SOURCES = 'external_sources'
}

export interface SourceMetadata {
  sourceId: string;
  sourceType: EvidenceSource;
  authority: number;
  freshness: number;
  reliability: number;
  lastVerified: string;
}

export interface ValidationIssue {
  id: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  type: IssueType;
  description: string;
  location?: ContentLocation;
  suggestedFix?: string;
  relatedEvidence: string[];
}

export enum IssueType {
  FACTUAL_INCONSISTENCY = 'factual_inconsistency',
  MISSING_CONTEXT = 'missing_context',
  CONFLICTING_INFORMATION = 'conflicting_information',
  INSUFFICIENT_DETAIL = 'insufficient_detail',
  OUTDATED_INFORMATION = 'outdated_information',
  SCOPE_MISMATCH = 'scope_mismatch',
  STAKEHOLDER_MISALIGNMENT = 'stakeholder_misalignment'
}

export interface ContentLocation {
  startOffset: number;
  endOffset: number;
  lineNumber?: number;
  columnNumber?: number;
}

export interface ValidationMetadata {
  queryComplexity: number;
  graphTraversalDepth: number;
  nodesExamined: number;
  relationshipsEvaluated: number;
  cacheHitRate: number;
  processingSteps: string[];
}

export interface CorrectionSuggestion {
  id: string;
  originalText: string;
  suggestedText: string;
  confidence: number;
  rationale: string;
  evidence: Evidence[];
  impactAssessment: ImpactAssessment;
}

export interface ImpactAssessment {
  changeType: 'minor' | 'moderate' | 'major';
  affectedSections: string[];
  stakeholderNotification: boolean;
  riskLevel: 'low' | 'medium' | 'high';
  approvalRequired: boolean;
}

export interface AlternativeContent {
  id: string;
  content: string;
  confidence: number;
  rationale: string;
  advantages: string[];
  disadvantages: string[];
  suitabilityScore: number;
}

// Knowledge Graph Types
export interface KnowledgeGraphQuery {
  query: string;
  parameters: Record<string, any>;
  queryType: QueryType;
  options: QueryOptions;
}

export enum QueryType {
  ENTITY_LOOKUP = 'entity_lookup',
  RELATIONSHIP_TRAVERSAL = 'relationship_traversal',
  PATTERN_MATCHING = 'pattern_matching',
  SIMILARITY_SEARCH = 'similarity_search',
  AGGREGATION = 'aggregation',
  PATH_FINDING = 'path_finding'
}

export interface QueryOptions {
  maxResults: number;
  timeoutMs: number;
  includeMetadata: boolean;
  enableCaching: boolean;
  cacheTtl: number;
}

export interface KnowledgeGraphResponse {
  results: GraphNode[];
  relationships: GraphRelationship[];
  metadata: QueryMetadata;
  executionTime: number;
}

export interface GraphNode {
  id: string;
  labels: string[];
  properties: Record<string, any>;
  score?: number;
  metadata: NodeMetadata;
}

export interface GraphRelationship {
  id: string;
  type: string;
  startNode: string;
  endNode: string;
  properties: Record<string, any>;
  score?: number;
}

export interface NodeMetadata {
  createdAt: string;
  updatedAt: string;
  source: string;
  reliability: number;
  lastValidated?: string;
  validationCount: number;
}

export interface QueryMetadata {
  cypherQuery: string;
  executionPlan: string;
  indexesUsed: string[];
  cacheHit: boolean;
  resultsCount: number;
  totalTime: number;
  dbHits: number;
}

// Vector Search Types
export interface VectorSearchRequest {
  query: string;
  embedding?: number[];
  filters: VectorFilters;
  options: VectorSearchOptions;
}

export interface VectorFilters {
  labels?: string[];
  properties?: Record<string, any>;
  dateRange?: DateRange;
  sourceTypes?: EvidenceSource[];
  minRelevance?: number;
}

export interface DateRange {
  start: string;
  end: string;
}

export interface VectorSearchOptions {
  k: number; // top k results
  similarityThreshold: number;
  includeMetadata: boolean;
  rerank: boolean;
  diversityFactor?: number;
}

export interface VectorSearchResponse {
  results: VectorSearchResult[];
  totalCount: number;
  processingTime: number;
  searchMetadata: SearchMetadata;
}

export interface VectorSearchResult {
  node: GraphNode;
  similarity: number;
  distance: number;
  explanation?: string;
  context: string[];
}

export interface SearchMetadata {
  queryEmbedding: number[];
  indexUsed: string;
  searchAlgorithm: string;
  approximateSearch: boolean;
  candidatesExamined: number;
}

// Community Detection Types
export interface CommunityDetectionRequest {
  nodeIds?: string[];
  algorithm: CommunityAlgorithm;
  parameters: AlgorithmParameters;
}

export enum CommunityAlgorithm {
  LOUVAIN = 'louvain',
  LABEL_PROPAGATION = 'label_propagation',
  LEIDEN = 'leiden',
  STRONGLY_CONNECTED = 'strongly_connected'
}

export interface AlgorithmParameters {
  resolution?: number;
  maxIterations?: number;
  tolerance?: number;
  randomSeed?: number;
  weightProperty?: string;
}

export interface CommunityDetectionResponse {
  communities: Community[];
  modularity: number;
  totalNodes: number;
  totalCommunities: number;
  algorithmMetadata: AlgorithmMetadata;
}

export interface Community {
  id: string;
  nodes: string[];
  size: number;
  density: number;
  summary: string;
  keyTopics: string[];
  representative: string;
  strength: number;
}

export interface AlgorithmMetadata {
  algorithm: CommunityAlgorithm;
  parameters: AlgorithmParameters;
  executionTime: number;
  iterations: number;
  converged: boolean;
}

// GraphRAG Configuration
export interface GraphRAGConfig {
  neo4jUri: string;
  neo4jUsername: string;
  neo4jPassword: string;
  vectorIndexName: string;
  embeddingDimensions: number;
  similarityMetric: 'cosine' | 'euclidean' | 'manhattan';
  defaultConfidenceThreshold: number;
  validationWeights: ValidationWeights;
  cacheConfiguration: CacheConfiguration;
  performanceSettings: PerformanceSettings;
}

export interface ValidationWeights {
  entityValidation: number;
  communityValidation: number;
  globalValidation: number;
  consistencyValidation: number;
}

export interface CacheConfiguration {
  enabled: boolean;
  ttlSeconds: number;
  maxSize: number;
  compressionEnabled: boolean;
}

export interface PerformanceSettings {
  maxConcurrentQueries: number;
  queryTimeoutMs: number;
  connectionPoolSize: number;
  retryAttempts: number;
  circuitBreakerThreshold: number;
}

// GraphRAG Service Interface
export interface GraphRAGService {
  validate(request: ValidationRequest): Promise<ValidationResponse>;
  search(request: VectorSearchRequest): Promise<VectorSearchResponse>;
  query(request: KnowledgeGraphQuery): Promise<KnowledgeGraphResponse>;
  detectCommunities(request: CommunityDetectionRequest): Promise<CommunityDetectionResponse>;
  updateKnowledgeGraph(updates: GraphUpdate[]): Promise<UpdateResponse>;
  getHealth(): Promise<HealthStatus>;
}

export interface GraphUpdate {
  operation: 'create' | 'update' | 'delete';
  nodeId?: string;
  relationshipId?: string;
  data: Record<string, any>;
}

export interface UpdateResponse {
  success: boolean;
  updatedNodes: number;
  updatedRelationships: number;
  errors: string[];
  processingTime: number;
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  database: {
    connected: boolean;
    responseTime: number;
    nodeCount: number;
    relationshipCount: number;
  };
  vectorIndex: {
    available: boolean;
    indexSize: number;
    lastUpdate: string;
  };
  cache: {
    hitRate: number;
    size: number;
    maxSize: number;
  };
}

// Error Types
export interface GraphRAGError {
  code: GraphRAGErrorCode;
  message: string;
  details?: Record<string, any>;
  timestamp: string;
  requestId?: string;
}

export enum GraphRAGErrorCode {
  VALIDATION_FAILED = 'VALIDATION_FAILED',
  QUERY_TIMEOUT = 'QUERY_TIMEOUT',
  DATABASE_CONNECTION = 'DATABASE_CONNECTION',
  INVALID_REQUEST = 'INVALID_REQUEST',
  INSUFFICIENT_CONFIDENCE = 'INSUFFICIENT_CONFIDENCE',
  RATE_LIMIT_EXCEEDED = 'RATE_LIMIT_EXCEEDED',
  INTERNAL_ERROR = 'INTERNAL_ERROR'
}