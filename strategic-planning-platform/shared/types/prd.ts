/**
 * PRD Workflow Types
 * Phase 0-4 interfaces, validation results, and workflow management
 */

export enum PRDPhase {
  PHASE_0 = 'phase_0', // Project Invitation
  PHASE_1 = 'phase_1', // Objective Clarification  
  PHASE_2 = 'phase_2', // Objective Drafting & Approval
  PHASE_3 = 'phase_3', // Section-by-Section Co-Creation
  PHASE_4 = 'phase_4'  // Synthesis & Finalization
}

export enum PRDStatus {
  DRAFT = 'draft',
  IN_PROGRESS = 'in_progress',
  UNDER_REVIEW = 'under_review',
  APPROVED = 'approved',
  COMPLETED = 'completed',
  ARCHIVED = 'archived'
}

export enum PriorityLevel {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

export enum SectionType {
  SCOPE = 'scope',
  DELIVERABLES = 'deliverables',
  TIMELINE = 'timeline',
  STAKEHOLDERS = 'stakeholders',
  BUDGET = 'budget',
  SUCCESS_METRICS = 'success_metrics',
  ASSUMPTIONS = 'assumptions',
  RISKS = 'risks'
}

export enum SectionStatus {
  PENDING = 'pending',
  IN_PROGRESS = 'in_progress',
  UNDER_REVIEW = 'under_review',
  APPROVED = 'approved',
  REVISION_REQUIRED = 'revision_required'
}

// Phase 0: Project Invitation
export interface Phase0Request {
  initialDescription: string;
  userId: string;
}

export interface Phase0Response {
  prdId: string;
  questions: ClarifyingQuestion[];
  similarProjects: ProjectSummary[];
  conceptAnalysis: ConceptExtractionResult;
  confidenceScore: number;
}

export interface ConceptExtractionResult {
  keyEntities: string[];
  domain: string;
  complexity: number;
  technologies: string[];
  businessObjectives: string[];
}

export interface ProjectSummary {
  id: string;
  title: string;
  description: string;
  similarity: number;
  outcomes: string[];
  lessonsLearned: string[];
}

// Phase 1: Objective Clarification
export interface ClarifyingQuestion {
  id: string;
  question: string;
  category: QuestionCategory;
  isRequired: boolean;
  helpText?: string;
  examples?: string[];
  dependsOn?: string[];
}

export enum QuestionCategory {
  BUSINESS_CONTEXT = 'business_context',
  USER_IMPACT = 'user_impact',
  TECHNICAL_SCOPE = 'technical_scope',
  SUCCESS_DEFINITION = 'success_definition',
  RESOURCE_CONTEXT = 'resource_context'
}

export interface Phase1Request {
  prdId: string;
  answers: Record<string, string>;
}

export interface Phase1Response {
  prdId: string;
  validations: QuestionValidation[];
  readyForPhase2: boolean;
  additionalQuestions?: ClarifyingQuestion[];
}

export interface QuestionValidation {
  questionId: string;
  confidence: number;
  isValid: boolean;
  suggestions?: string[];
  warnings?: string[];
}

// Phase 2: Objective Drafting & Approval
export interface Phase2Request {
  prdId: string;
  userFeedback?: string;
  iterationCount?: number;
}

export interface Phase2Response {
  prdId: string;
  objective: SMARTObjective;
  confidence: number;
  alternatives?: SMARTObjective[];
  improvements: string[];
}

export interface SMARTObjective {
  id: string;
  statement: string;
  specific: string;
  measurable: string[];
  achievable: string;
  relevant: string;
  timeBound: string;
  qualityScore: number;
  version: number;
  createdAt: string;
}

// Phase 3: Section-by-Section Co-Creation
export interface Phase3Request {
  prdId: string;
  sectionType: SectionType;
  userInput?: string;
  context: SectionContext;
}

export interface SectionContext {
  approvedSections: Record<SectionType, Section>;
  currentObjective: SMARTObjective;
  projectConstraints: string[];
  stakeholderRequirements: string[];
}

export interface Phase3Response {
  prdId: string;
  section: Section;
  crossSectionConsistency: ConsistencyCheck;
  dependencies: string[];
  recommendations: string[];
}

export interface Section {
  id: string;
  type: SectionType;
  title: string;
  content: string;
  status: SectionStatus;
  version: number;
  createdAt: string;
  updatedAt: string;
  author: string;
  reviewers: string[];
  metadata: SectionMetadata;
}

export interface SectionMetadata {
  wordCount: number;
  readabilityScore: number;
  completenessScore: number;
  qualityScore: number;
  estimatedReadingTime: number;
  tags: string[];
}

export interface ConsistencyCheck {
  overallScore: number;
  conflicts: ConflictItem[];
  alignmentScore: number;
  recommendations: string[];
}

export interface ConflictItem {
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  affectedSections: SectionType[];
  suggestedResolution: string;
}

// Phase 4: Synthesis & Finalization
export interface Phase4Request {
  prdId: string;
  exportFormat: ExportFormat[];
  includeTemplates: boolean;
}

export enum ExportFormat {
  PDF = 'pdf',
  WORD = 'word',
  MARKDOWN = 'markdown',
  HTML = 'html',
  JSON = 'json'
}

export interface Phase4Response {
  prdId: string;
  completeDocument: CompleteDocument;
  exports: ExportResult[];
  nextActions: NextAction[];
  qualityReport: QualityReport;
}

export interface CompleteDocument {
  id: string;
  title: string;
  executiveSummary: string;
  objective: SMARTObjective;
  sections: Section[];
  metadata: DocumentMetadata;
  version: string;
  finalizedAt: string;
}

export interface DocumentMetadata {
  totalWords: number;
  estimatedReadingTime: number;
  overallQualityScore: number;
  completenessScore: number;
  stakeholderCount: number;
  requirementCount: number;
  riskCount: number;
  milestoneCount: number;
}

export interface ExportResult {
  format: ExportFormat;
  url: string;
  size: number;
  generatedAt: string;
  expiresAt: string;
}

export interface NextAction {
  id: string;
  title: string;
  description: string;
  type: NextActionType;
  priority: PriorityLevel;
  assignee?: string;
  dueDate?: string;
  dependencies: string[];
}

export enum NextActionType {
  WBS_CREATION = 'wbs_creation',
  STAKEHOLDER_REVIEW = 'stakeholder_review',
  IMPLEMENTATION_PLANNING = 'implementation_planning',
  RESOURCE_ALLOCATION = 'resource_allocation',
  RISK_MITIGATION = 'risk_mitigation'
}

export interface QualityReport {
  overallScore: number;
  dimensionScores: QualityDimensions;
  strengths: string[];
  improvementAreas: string[];
  recommendations: string[];
  comparisonToBenchmark: BenchmarkComparison;
}

export interface QualityDimensions {
  clarity: number;
  completeness: number;
  consistency: number;
  feasibility: number;
  stakeholderAlignment: number;
  riskMitigation: number;
}

export interface BenchmarkComparison {
  industryAverage: number;
  organizationAverage: number;
  topPerformer: number;
  ranking: 'top_10_percent' | 'top_25_percent' | 'above_average' | 'average' | 'below_average';
}

// PRD Management
export interface PRD {
  id: string;
  title: string;
  description: string;
  phase: PRDPhase;
  status: PRDStatus;
  priority: PriorityLevel;
  createdBy: string;
  assignedTo?: string;
  createdAt: string;
  updatedAt: string;
  dueDate?: string;
  tags: string[];
  metadata: PRDMetadata;
  phases: PhaseProgress;
}

export interface PRDMetadata {
  estimatedHours: number;
  actualHours?: number;
  complexity: number;
  stakeholderCount: number;
  requirementCount: number;
  lastActivity: string;
  collaborators: string[];
}

export interface PhaseProgress {
  phase0: PhaseStatus;
  phase1: PhaseStatus;
  phase2: PhaseStatus;
  phase3: PhaseStatus;
  phase4: PhaseStatus;
}

export interface PhaseStatus {
  status: 'not_started' | 'in_progress' | 'completed' | 'skipped';
  completedAt?: string;
  duration?: number;
  qualityScore?: number;
  iterations?: number;
}

export interface PRDListItem {
  id: string;
  title: string;
  status: PRDStatus;
  phase: PRDPhase;
  priority: PriorityLevel;
  createdAt: string;
  updatedAt: string;
  createdBy: string;
  assignedTo?: string;
  progress: number;
  qualityScore?: number;
  tags: string[];
}

export interface PRDFilters {
  status?: PRDStatus[];
  phase?: PRDPhase[];
  priority?: PriorityLevel[];
  createdBy?: string[];
  assignedTo?: string[];
  tags?: string[];
  dateRange?: {
    start: string;
    end: string;
  };
  search?: string;
}

export interface PRDSortOptions {
  field: 'title' | 'createdAt' | 'updatedAt' | 'priority' | 'status' | 'progress';
  direction: 'asc' | 'desc';
}

export interface CreatePRDRequest {
  title: string;
  description: string;
  priority: PriorityLevel;
  assignedTo?: string;
  dueDate?: string;
  tags?: string[];
}

export interface UpdatePRDRequest {
  title?: string;
  description?: string;
  priority?: PriorityLevel;
  assignedTo?: string;
  dueDate?: string;
  tags?: string[];
  status?: PRDStatus;
}