/**
 * WebSocket Communication Types
 * Real-time collaboration, live updates, and socket event management
 */

// WebSocket Connection Types
export enum WebSocketEvent {
  // Connection Events
  CONNECT = 'connect',
  DISCONNECT = 'disconnect',
  RECONNECT = 'reconnect',
  ERROR = 'error',
  
  // Authentication Events
  AUTHENTICATE = 'authenticate',
  AUTHENTICATED = 'authenticated',
  AUTHENTICATION_FAILED = 'authentication_failed',
  
  // Room Management
  JOIN_ROOM = 'join_room',
  LEAVE_ROOM = 'leave_room',
  ROOM_JOINED = 'room_joined',
  ROOM_LEFT = 'room_left',
  
  // PRD Collaboration Events
  PRD_UPDATED = 'prd_updated',
  PRD_SECTION_CHANGED = 'prd_section_changed',
  PRD_STATUS_CHANGED = 'prd_status_changed',
  PRD_COMMENT_ADDED = 'prd_comment_added',
  
  // Real-time Editing
  CURSOR_MOVED = 'cursor_moved',
  SELECTION_CHANGED = 'selection_changed',
  TEXT_CHANGED = 'text_changed',
  TEXT_INSERTED = 'text_inserted',
  TEXT_DELETED = 'text_deleted',
  
  // User Presence
  USER_JOINED = 'user_joined',
  USER_LEFT = 'user_left',
  USER_TYPING = 'user_typing',
  USER_STOPPED_TYPING = 'user_stopped_typing',
  USER_PRESENCE_CHANGED = 'user_presence_changed',
  
  // Validation Events
  VALIDATION_STARTED = 'validation_started',
  VALIDATION_COMPLETED = 'validation_completed',
  VALIDATION_FAILED = 'validation_failed',
  
  // Progress Events
  PHASE_PROGRESS = 'phase_progress',
  TASK_PROGRESS = 'task_progress',
  GENERATION_PROGRESS = 'generation_progress',
  
  // Notification Events
  NOTIFICATION = 'notification',
  ALERT = 'alert',
  SYSTEM_MESSAGE = 'system_message'
}

export enum ConnectionState {
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  DISCONNECTED = 'disconnected',
  RECONNECTING = 'reconnecting',
  ERROR = 'error'
}

export enum UserPresenceStatus {
  ONLINE = 'online',
  AWAY = 'away',
  BUSY = 'busy',
  OFFLINE = 'offline'
}

export interface WebSocketConfig {
  url: string;
  protocols?: string[];
  heartbeatInterval: number;
  reconnectAttempts: number;
  reconnectDelay: number;
  timeout: number;
  enableCompression: boolean;
}

export interface WebSocketMessage<T = any> {
  event: WebSocketEvent;
  data: T;
  messageId: string;
  timestamp: string;
  userId?: string;
  roomId?: string;
  replyTo?: string;
}

export interface WebSocketResponse<T = any> {
  success: boolean;
  data?: T;
  error?: WebSocketError;
  messageId: string;
  timestamp: string;
}

export interface WebSocketError {
  code: string;
  message: string;
  details?: Record<string, any>;
}

// Authentication Types
export interface AuthenticateMessage {
  token: string;
  userId: string;
  sessionId: string;
}

export interface AuthenticationResult {
  success: boolean;
  userId: string;
  permissions: string[];
  sessionId: string;
}

// Room Management Types
export interface JoinRoomMessage {
  roomId: string;
  roomType: RoomType;
  userMetadata?: UserMetadata;
}

export enum RoomType {
  PRD_COLLABORATION = 'prd_collaboration',
  SECTION_EDITING = 'section_editing',
  VALIDATION_REVIEW = 'validation_review',
  GLOBAL_NOTIFICATIONS = 'global_notifications'
}

export interface Room {
  id: string;
  type: RoomType;
  name: string;
  participants: RoomParticipant[];
  metadata: RoomMetadata;
  permissions: RoomPermissions;
  createdAt: string;
  lastActivity: string;
}

export interface RoomParticipant {
  userId: string;
  username: string;
  role: string;
  presence: UserPresenceStatus;
  joinedAt: string;
  lastSeen: string;
  cursor?: CursorPosition;
  selection?: TextSelection;
  permissions: ParticipantPermissions;
}

export interface UserMetadata {
  name: string;
  avatar?: string;
  role: string;
  color?: string; // For cursor/selection display
  preferences?: UserCollaborationPreferences;
}

export interface UserCollaborationPreferences {
  showCursor: boolean;
  showSelection: boolean;
  enableNotifications: boolean;
  notificationSound: boolean;
}

export interface RoomMetadata {
  prdId?: string;
  sectionId?: string;
  documentVersion: number;
  lastModified: string;
  lockStatus?: LockStatus;
}

export interface LockStatus {
  isLocked: boolean;
  lockedBy?: string;
  lockedAt?: string;
  reason?: string;
  autoUnlockAt?: string;
}

export interface RoomPermissions {
  canRead: boolean;
  canWrite: boolean;
  canComment: boolean;
  canModerate: boolean;
  canInvite: boolean;
}

export interface ParticipantPermissions {
  canEdit: boolean;
  canComment: boolean;
  canView: boolean;
  canValidate: boolean;
}

// Real-time Editing Types
export interface CursorPosition {
  line: number;
  column: number;
  sectionId?: string;
  timestamp: string;
}

export interface TextSelection {
  start: CursorPosition;
  end: CursorPosition;
  text: string;
  timestamp: string;
}

export interface TextChange {
  id: string;
  type: TextChangeType;
  position: CursorPosition;
  length: number;
  content: string;
  userId: string;
  timestamp: string;
  sectionId?: string;
  version: number;
}

export enum TextChangeType {
  INSERT = 'insert',
  DELETE = 'delete',
  REPLACE = 'replace',
  FORMAT = 'format'
}

export interface CursorUpdate {
  userId: string;
  position: CursorPosition;
  timestamp: string;
}

export interface SelectionUpdate {
  userId: string;
  selection: TextSelection;
  timestamp: string;
}

export interface TypingIndicator {
  userId: string;
  isTyping: boolean;
  sectionId?: string;
  timestamp: string;
}

// PRD Collaboration Events
export interface PRDUpdateEvent {
  prdId: string;
  updateType: PRDUpdateType;
  updatedBy: string;
  changes: PRDChange[];
  timestamp: string;
  version: number;
}

export enum PRDUpdateType {
  METADATA_CHANGED = 'metadata_changed',
  SECTION_ADDED = 'section_added',
  SECTION_UPDATED = 'section_updated',
  SECTION_DELETED = 'section_deleted',
  STATUS_CHANGED = 'status_changed',
  PHASE_CHANGED = 'phase_changed',
  ASSIGNMENT_CHANGED = 'assignment_changed'
}

export interface PRDChange {
  field: string;
  oldValue: any;
  newValue: any;
  sectionId?: string;
}

export interface CommentEvent {
  commentId: string;
  prdId: string;
  sectionId?: string;
  content: string;
  author: string;
  timestamp: string;
  parentCommentId?: string;
  mentions: string[];
  resolved: boolean;
}

// Validation Events
export interface ValidationEvent {
  validationId: string;
  prdId: string;
  sectionId?: string;
  status: ValidationStatus;
  progress?: number;
  results?: ValidationResults;
  timestamp: string;
}

export enum ValidationStatus {
  QUEUED = 'queued',
  IN_PROGRESS = 'in_progress',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled'
}

export interface ValidationResults {
  overallScore: number;
  issues: ValidationIssueEvent[];
  suggestions: string[];
  confidence: number;
}

export interface ValidationIssueEvent {
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  location?: TextSelection;
  suggestedFix?: string;
}

// Progress Events
export interface ProgressEvent {
  id: string;
  type: ProgressType;
  prdId: string;
  progress: number; // 0-100
  message: string;
  details?: ProgressDetails;
  timestamp: string;
}

export enum ProgressType {
  PHASE_COMPLETION = 'phase_completion',
  TASK_EXECUTION = 'task_execution',
  DOCUMENT_GENERATION = 'document_generation',
  VALIDATION_PROCESSING = 'validation_processing',
  EXPORT_GENERATION = 'export_generation'
}

export interface ProgressDetails {
  currentStep: string;
  totalSteps: number;
  completedSteps: number;
  estimatedTimeRemaining?: number;
  errors?: string[];
}

// Notification Events
export interface NotificationEvent {
  id: string;
  type: NotificationType;
  title: string;
  message: string;
  priority: NotificationPriority;
  userId?: string; // If targeted to specific user
  prdId?: string;
  actionUrl?: string;
  actionText?: string;
  timestamp: string;
  expiresAt?: string;
  metadata?: Record<string, any>;
}

export enum NotificationType {
  PRD_ASSIGNED = 'prd_assigned',
  PRD_COMPLETED = 'prd_completed',
  REVIEW_REQUESTED = 'review_requested',
  COMMENT_MENTION = 'comment_mention',
  VALIDATION_COMPLETE = 'validation_complete',
  SYSTEM_UPDATE = 'system_update',
  ERROR_ALERT = 'error_alert'
}

export enum NotificationPriority {
  LOW = 'low',
  NORMAL = 'normal',
  HIGH = 'high',
  URGENT = 'urgent'
}

export interface SystemMessage {
  id: string;
  type: SystemMessageType;
  message: string;
  level: 'info' | 'warning' | 'error' | 'success';
  timestamp: string;
  dismissible: boolean;
  actions?: SystemMessageAction[];
}

export enum SystemMessageType {
  MAINTENANCE = 'maintenance',
  FEATURE_ANNOUNCEMENT = 'feature_announcement',
  SERVICE_DISRUPTION = 'service_disruption',
  PERFORMANCE_WARNING = 'performance_warning'
}

export interface SystemMessageAction {
  label: string;
  action: string;
  style: 'primary' | 'secondary' | 'danger';
}

// WebSocket Client Interface
export interface WebSocketClient {
  // Connection Management
  connect(): Promise<void>;
  disconnect(): void;
  reconnect(): void;
  
  // Event Handlers
  on<T = any>(event: WebSocketEvent, handler: (data: T) => void): void;
  off(event: WebSocketEvent, handler?: Function): void;
  
  // Message Sending
  send<T = any>(event: WebSocketEvent, data: T, roomId?: string): Promise<WebSocketResponse>;
  sendToUser<T = any>(event: WebSocketEvent, data: T, userId: string): Promise<WebSocketResponse>;
  
  // Room Management
  joinRoom(roomId: string, roomType: RoomType, metadata?: UserMetadata): Promise<void>;
  leaveRoom(roomId: string): Promise<void>;
  
  // Authentication
  authenticate(token: string): Promise<AuthenticationResult>;
  
  // State
  getConnectionState(): ConnectionState;
  getRooms(): Room[];
  getParticipants(roomId: string): RoomParticipant[];
  
  // Real-time Editing
  updateCursor(position: CursorPosition, roomId: string): void;
  updateSelection(selection: TextSelection, roomId: string): void;
  sendTextChange(change: TextChange, roomId: string): void;
  setTyping(isTyping: boolean, roomId: string, sectionId?: string): void;
}

// WebSocket Server Interface (for backend)
export interface WebSocketServer {
  // Client Management
  handleConnection(socket: WebSocket): void;
  handleDisconnection(socketId: string): void;
  
  // Room Management
  createRoom(room: Partial<Room>): Room;
  deleteRoom(roomId: string): void;
  addToRoom(socketId: string, roomId: string, metadata: UserMetadata): void;
  removeFromRoom(socketId: string, roomId: string): void;
  
  // Broadcasting
  broadcast(event: WebSocketEvent, data: any, roomId?: string): void;
  broadcastToUser(event: WebSocketEvent, data: any, userId: string): void;
  broadcastToRoom(event: WebSocketEvent, data: any, roomId: string, excludeUser?: string): void;
  
  // Authentication
  authenticateSocket(socketId: string, token: string): Promise<boolean>;
  
  // Health & Monitoring
  getConnectedClients(): number;
  getRoomStats(): Record<string, number>;
  getHealth(): { status: string; connections: number; rooms: number; };
}