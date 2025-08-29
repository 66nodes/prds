/**
 * Shared Types Index
 * Main export file for all TypeScript types and interfaces
 */

// Authentication & Authorization Types
export * from './auth';

// PRD Workflow Types
export * from './prd';

// GraphRAG Integration Types
export * from './graphrag';

// WebSocket Communication Types
export * from './websocket';

// API Response Types
export * from './api';

// Common Utility Types
export interface BaseEntity {
  id: string;
  createdAt: string;
  updatedAt: string;
  createdBy?: string;
  updatedBy?: string;
}

export interface TimestampedEntity extends BaseEntity {
  deletedAt?: string;
  deletedBy?: string;
  version: number;
}

export interface AuditableEntity extends TimestampedEntity {
  auditLog: AuditEntry[];
}

export interface AuditEntry {
  action: string;
  userId: string;
  timestamp: string;
  changes: Record<string, { old: any; new: any }>;
  metadata?: Record<string, any>;
}

// Generic Response Types
export type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;
export type Required<T, K extends keyof T> = Omit<T, K> & Required<Pick<T, K>>;
export type Nullable<T> = T | null;
export type Maybe<T> = T | undefined;

// Environment Types
export interface Environment {
  NODE_ENV: 'development' | 'test' | 'staging' | 'production';
  API_BASE_URL: string;
  WEBSOCKET_URL: string;
  
  // Database
  NEO4J_URI: string;
  NEO4J_USERNAME: string;
  NEO4J_PASSWORD: string;
  POSTGRES_URL: string;
  REDIS_URL: string;
  
  // Authentication
  JWT_SECRET: string;
  JWT_REFRESH_SECRET: string;
  JWT_EXPIRES_IN: string;
  JWT_REFRESH_EXPIRES_IN: string;
  
  // External Services
  OPENROUTER_API_KEY: string;
  OPENAI_API_KEY: string;
  ANTHROPIC_API_KEY: string;
  GOOGLE_API_KEY: string;
  
  // GraphRAG
  GRAPHRAG_ENDPOINT: string;
  GRAPHRAG_API_KEY: string;
  VECTOR_INDEX_NAME: string;
  EMBEDDING_DIMENSIONS: number;
  
  // Email Service
  SMTP_HOST: string;
  SMTP_PORT: string;
  SMTP_USER: string;
  SMTP_PASS: string;
  
  // Storage
  S3_BUCKET: string;
  S3_REGION: string;
  S3_ACCESS_KEY: string;
  S3_SECRET_KEY: string;
  
  // Monitoring
  SENTRY_DSN?: string;
  LOGFIRE_TOKEN?: string;
  NEW_RELIC_LICENSE_KEY?: string;
  
  // Feature Flags
  ENABLE_WEBSOCKETS: boolean;
  ENABLE_GRAPHRAG: boolean;
  ENABLE_REAL_TIME_COLLABORATION: boolean;
  ENABLE_ANALYTICS: boolean;
}

// Configuration Types
export interface AppConfig {
  app: {
    name: string;
    version: string;
    port: number;
    host: string;
    environment: Environment['NODE_ENV'];
  };
  
  database: {
    neo4j: {
      uri: string;
      username: string;
      password: string;
      maxConnectionLifetime: number;
      maxConnectionPoolSize: number;
    };
    postgres: {
      url: string;
      ssl: boolean;
      maxConnections: number;
      connectionTimeout: number;
    };
    redis: {
      url: string;
      maxRetriesPerRequest: number;
      retryDelayOnFailover: number;
    };
  };
  
  auth: {
    jwt: {
      secret: string;
      expiresIn: string;
      refreshSecret: string;
      refreshExpiresIn: string;
    };
    session: {
      maxAge: number;
      secure: boolean;
      sameSite: 'strict' | 'lax' | 'none';
    };
  };
  
  websocket: {
    enabled: boolean;
    port: number;
    heartbeatInterval: number;
    connectionTimeout: number;
    maxConnections: number;
  };
  
  graphrag: {
    enabled: boolean;
    endpoint: string;
    apiKey: string;
    timeout: number;
    maxRetries: number;
    confidenceThreshold: number;
  };
  
  llm: {
    primaryProvider: 'openrouter' | 'openai' | 'anthropic' | 'google';
    fallbackProviders: string[];
    maxTokens: number;
    temperature: number;
    timeout: number;
  };
  
  storage: {
    provider: 'local' | 's3' | 'gcs';
    bucket?: string;
    region?: string;
    maxFileSize: number;
    allowedMimeTypes: string[];
  };
  
  email: {
    provider: 'smtp' | 'sendgrid' | 'ses';
    from: string;
    replyTo: string;
    templates: {
      welcome: string;
      passwordReset: string;
      prdCompleted: string;
    };
  };
  
  monitoring: {
    enabled: boolean;
    sentry?: {
      dsn: string;
      environment: string;
      tracesSampleRate: number;
    };
    metrics: {
      enabled: boolean;
      port: number;
      path: string;
    };
  };
  
  rateLimiting: {
    enabled: boolean;
    windowMs: number;
    maxRequests: number;
    skipSuccessfulRequests: boolean;
  };
  
  cors: {
    origin: string | string[];
    credentials: boolean;
    methods: string[];
    allowedHeaders: string[];
  };
}

// Error Handling Types
export interface ErrorContext {
  userId?: string;
  requestId?: string;
  resource?: string;
  operation?: string;
  metadata?: Record<string, any>;
}

export interface ErrorDetails {
  code: string;
  message: string;
  context?: ErrorContext;
  stack?: string;
  timestamp: string;
}

export class AppError extends Error {
  public readonly code: string;
  public readonly statusCode: number;
  public readonly context?: ErrorContext;
  public readonly isOperational: boolean;

  constructor(
    code: string,
    message: string,
    statusCode: number = 500,
    context?: ErrorContext,
    isOperational: boolean = true
  ) {
    super(message);
    
    this.name = this.constructor.name;
    this.code = code;
    this.statusCode = statusCode;
    this.context = context;
    this.isOperational = isOperational;
    
    Error.captureStackTrace(this, this.constructor);
  }
}

// Validation Types
export interface ValidationRule {
  field: string;
  rules: ValidationConstraint[];
}

export interface ValidationConstraint {
  type: 'required' | 'minLength' | 'maxLength' | 'pattern' | 'min' | 'max' | 'email' | 'url' | 'custom';
  value?: any;
  message: string;
  validator?: (value: any) => boolean;
}

export interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
}

export interface ValidationError {
  field: string;
  message: string;
  value: any;
}

// Utility Functions Types
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type DeepRequired<T> = {
  [P in keyof T]-?: T[P] extends object ? DeepRequired<T[P]> : T[P];
};

export type KeysOfType<T, U> = {
  [K in keyof T]: T[K] extends U ? K : never;
}[keyof T];

export type Awaited<T> = T extends Promise<infer U> ? U : T;

export type NonNullable<T> = T extends null | undefined ? never : T;