import { defineStore } from 'pinia';
import { useWebSocket } from '~/composables/useWebSocket';

// Types for collaboration
interface UserPresence {
  user_id: string;
  username: string;
  cursor_position: number;
  selection_start?: number;
  selection_end?: number;
  color: string;
  last_activity: string;
  is_active: boolean;
}

interface DocumentState {
  document_id: string;
  content: string;
  version: number;
  last_modified: string;
  active_users: string[];
  checksum?: string;
}

interface CollaborationSession {
  session_id: string;
  document_id: string;
  document_state: DocumentState;
  active_users: UserPresence[];
  created_at: string;
  last_activity: string;
}

interface Operation {
  id?: string;
  type: 'insert' | 'delete' | 'replace' | 'format';
  position: number;
  content?: string;
  length?: number;
  user_id?: string;
  version?: number;
  parent_version?: number;
  metadata?: Record<string, any>;
}

interface PendingOperation extends Operation {
  timestamp: number;
  retryCount: number;
}

export const useCollaborationStore = defineStore('collaboration', {
  state: () => ({
    // Session management
    currentSession: null as CollaborationSession | null,
    sessions: new Map<string, CollaborationSession>(),
    
    // User presence
    activeUsers: new Map<string, UserPresence>(),
    localUser: null as UserPresence | null,
    
    // Document state
    documentContent: '',
    documentVersion: 0,
    localVersion: 0,
    
    // Operation management
    pendingOperations: [] as PendingOperation[],
    operationHistory: [] as Operation[],
    
    // UI state
    isConnected: false,
    isSyncing: false,
    hasConflict: false,
    lastSyncTime: null as Date | null,
    
    // Cursor and selection tracking
    remoteCursors: new Map<string, { position: number; color: string; username: string }>(),
    remoteSelections: new Map<string, { start: number; end: number; color: string }>(),
  }),

  getters: {
    isInSession: (state) => state.currentSession !== null,
    
    sessionId: (state) => state.currentSession?.session_id || null,
    
    documentId: (state) => state.currentSession?.document_id || null,
    
    otherUsers: (state) => {
      if (!state.localUser) return [];
      return Array.from(state.activeUsers.values()).filter(
        user => user.user_id !== state.localUser?.user_id
      );
    },
    
    userCount: (state) => state.activeUsers.size,
    
    hasPendingOperations: (state) => state.pendingOperations.length > 0,
    
    isOutOfSync: (state) => state.localVersion !== state.documentVersion,
  },

  actions: {
    // Initialize collaboration
    async initializeCollaboration() {
      const { isConnected, on, off } = useWebSocket();
      
      // Set connection status
      this.isConnected = isConnected.value;
      
      // Watch connection changes
      watch(isConnected, (connected) => {
        this.isConnected = connected;
        if (connected && this.currentSession) {
          // Rejoin session on reconnect
          this.rejoinSession();
        }
      });
      
      // Register WebSocket event handlers
      this.registerWebSocketHandlers();
    },
    
    // Join a collaboration session
    async joinSession(documentId: string, username: string, initialContent = '') {
      const { send } = useWebSocket();
      const { user } = useAuth();
      
      if (!user.value) {
        throw new Error('User must be authenticated to join collaboration');
      }
      
      // Send join request
      send('collaboration' as any, {
        action: 'join_session',
        document_id: documentId,
        username: username || user.value.name,
        initial_content: initialContent,
      });
      
      this.isSyncing = true;
    },
    
    // Leave current session
    async leaveSession() {
      if (!this.currentSession) return;
      
      const { send } = useWebSocket();
      
      send('collaboration' as any, {
        action: 'leave_session',
        session_id: this.currentSession.session_id,
      });
      
      // Clear local state
      this.clearSession();
    },
    
    // Rejoin session after reconnection
    async rejoinSession() {
      if (!this.currentSession) return;
      
      const { send } = useWebSocket();
      
      send('collaboration' as any, {
        action: 'get_session_state',
        session_id: this.currentSession.session_id,
      });
      
      this.isSyncing = true;
    },
    
    // Send edit operation
    async sendEdit(operation: Operation) {
      const { send, isConnected: wsConnected } = useWebSocket();
      
      if (!this.currentSession || !wsConnected.value) {
        // Queue operation if not connected
        this.queueOperation(operation);
        return;
      }
      
      // Add metadata
      const fullOperation: Operation = {
        ...operation,
        parent_version: this.documentVersion,
        user_id: this.localUser?.user_id,
      };
      
      // Send via WebSocket
      send('collaboration' as any, {
        action: 'edit',
        session_id: this.currentSession.session_id,
        operation: fullOperation,
      });
      
      // Queue for potential retry
      this.queueOperation(fullOperation);
    },
    
    // Queue operation for retry
    queueOperation(operation: Operation) {
      this.pendingOperations.push({
        ...operation,
        timestamp: Date.now(),
        retryCount: 0,
      });
      
      // Start retry timer
      this.startRetryTimer();
    },
    
    // Update cursor position
    updateCursorPosition(position: number, selectionStart?: number, selectionEnd?: number) {
      const { send, isConnected: wsConnected } = useWebSocket();
      
      if (!this.currentSession || !wsConnected.value) return;
      
      // Update local cursor
      if (this.localUser) {
        this.localUser.cursor_position = position;
        if (selectionStart !== undefined) this.localUser.selection_start = selectionStart;
        if (selectionEnd !== undefined) this.localUser.selection_end = selectionEnd;
      }
      
      // Send cursor update
      send('collaboration' as any, {
        action: 'cursor_update',
        session_id: this.currentSession.session_id,
        cursor_position: position,
        selection_start: selectionStart,
        selection_end: selectionEnd,
      });
    },
    
    // Apply remote operation with OT
    applyRemoteOperation(operation: Operation) {
      // Transform against pending operations
      let transformedOp = this.transformOperation(operation);
      
      // Apply to document
      this.applyOperationToDocument(transformedOp);
      
      // Update version
      this.documentVersion = operation.version || this.documentVersion + 1;
      
      // Add to history
      this.operationHistory.push(operation);
      
      // Limit history size
      if (this.operationHistory.length > 1000) {
        this.operationHistory = this.operationHistory.slice(-500);
      }
    },
    
    // Transform operation against pending operations
    transformOperation(operation: Operation): Operation {
      let transformed = { ...operation };
      
      for (const pending of this.pendingOperations) {
        // Transform based on operation types
        if (pending.type === 'insert' && transformed.type === 'insert') {
          if (transformed.position >= pending.position) {
            transformed.position += pending.content?.length || 0;
          }
        } else if (pending.type === 'delete' && transformed.type === 'insert') {
          if (transformed.position > pending.position) {
            const deleteLength = pending.length || 0;
            if (transformed.position >= pending.position + deleteLength) {
              transformed.position -= deleteLength;
            } else {
              transformed.position = pending.position;
            }
          }
        }
        // Add more transformation rules as needed
      }
      
      return transformed;
    },
    
    // Apply operation to local document
    applyOperationToDocument(operation: Operation) {
      const content = this.documentContent;
      
      switch (operation.type) {
        case 'insert':
          if (operation.content) {
            this.documentContent = 
              content.slice(0, operation.position) +
              operation.content +
              content.slice(operation.position);
          }
          break;
          
        case 'delete':
          if (operation.length) {
            this.documentContent = 
              content.slice(0, operation.position) +
              content.slice(operation.position + operation.length);
          }
          break;
          
        case 'replace':
          if (operation.content && operation.length) {
            this.documentContent = 
              content.slice(0, operation.position) +
              operation.content +
              content.slice(operation.position + operation.length);
          }
          break;
      }
      
      this.localVersion++;
    },
    
    // Register WebSocket event handlers
    registerWebSocketHandlers() {
      const { on } = useWebSocket();
      
      // Handle session joined
      on('user_join' as any, (data: any) => {
        if (data.session_state) {
          this.handleSessionJoined(data.session_state);
        }
      });
      
      // Handle document edits
      on('document_edit' as any, (data: any) => {
        if (data.operation) {
          this.handleRemoteEdit(data.operation, data.document_state);
        }
      });
      
      // Handle cursor updates
      on('user_cursor_update' as any, (data: any) => {
        this.handleRemoteCursor(data);
      });
      
      // Handle user presence updates
      on('presence_update' as any, (data: any) => {
        this.handlePresenceUpdate(data);
      });
      
      // Handle user leave
      on('user_leave' as any, (data: any) => {
        this.handleUserLeave(data.left_user_id);
      });
      
      // Handle conflicts
      on('conflict_detected' as any, (data: any) => {
        this.handleConflict(data);
      });
      
      // Handle document sync
      on('document_sync' as any, (data: any) => {
        if (data.session_state) {
          this.handleDocumentSync(data.session_state);
        }
      });
    },
    
    // Handle session joined event
    handleSessionJoined(sessionState: CollaborationSession) {
      this.currentSession = sessionState;
      this.documentContent = sessionState.document_state.content;
      this.documentVersion = sessionState.document_state.version;
      this.localVersion = sessionState.document_state.version;
      
      // Set active users
      this.activeUsers.clear();
      sessionState.active_users.forEach(user => {
        this.activeUsers.set(user.user_id, user);
        if (user.user_id === useAuth().user.value?.id) {
          this.localUser = user;
        }
      });
      
      this.isSyncing = false;
      this.lastSyncTime = new Date();
    },
    
    // Handle remote edit
    handleRemoteEdit(operation: Operation, documentState: DocumentState) {
      // Skip if it's our own operation
      if (operation.user_id === this.localUser?.user_id) {
        // Remove from pending operations
        this.pendingOperations = this.pendingOperations.filter(
          op => op.id !== operation.id
        );
        return;
      }
      
      // Apply remote operation
      this.applyRemoteOperation(operation);
      
      // Update document state if provided
      if (documentState) {
        // Verify content matches
        if (documentState.content !== this.documentContent) {
          // Resync if mismatch
          this.documentContent = documentState.content;
          this.documentVersion = documentState.version;
          this.localVersion = documentState.version;
        }
      }
    },
    
    // Handle remote cursor update
    handleRemoteCursor(data: any) {
      const { user_id, cursor_position, selection_start, selection_end, color } = data;
      
      // Skip if it's our own cursor
      if (user_id === this.localUser?.user_id) return;
      
      // Update remote cursor
      const user = this.activeUsers.get(user_id);
      if (user) {
        this.remoteCursors.set(user_id, {
          position: cursor_position,
          color: color || user.color,
          username: user.username,
        });
        
        // Update selection if provided
        if (selection_start !== undefined && selection_end !== undefined) {
          this.remoteSelections.set(user_id, {
            start: selection_start,
            end: selection_end,
            color: color || user.color,
          });
        } else {
          this.remoteSelections.delete(user_id);
        }
      }
    },
    
    // Handle presence update
    handlePresenceUpdate(data: any) {
      if (data.new_user) {
        const user = data.new_user as UserPresence;
        this.activeUsers.set(user.user_id, user);
      }
    },
    
    // Handle user leave
    handleUserLeave(userId: string) {
      this.activeUsers.delete(userId);
      this.remoteCursors.delete(userId);
      this.remoteSelections.delete(userId);
    },
    
    // Handle conflict
    handleConflict(data: any) {
      this.hasConflict = true;
      
      // Log conflict for debugging
      console.warn('Collaboration conflict detected:', data);
      
      // Request full sync
      this.rejoinSession();
    },
    
    // Handle document sync
    handleDocumentSync(sessionState: CollaborationSession) {
      this.currentSession = sessionState;
      this.documentContent = sessionState.document_state.content;
      this.documentVersion = sessionState.document_state.version;
      this.localVersion = sessionState.document_state.version;
      
      // Clear pending operations as they're obsolete
      this.pendingOperations = [];
      
      this.hasConflict = false;
      this.isSyncing = false;
      this.lastSyncTime = new Date();
    },
    
    // Start retry timer for pending operations
    startRetryTimer() {
      const retryInterval = setInterval(() => {
        if (this.pendingOperations.length === 0) {
          clearInterval(retryInterval);
          return;
        }
        
        const now = Date.now();
        const maxRetries = 3;
        const retryDelay = 2000; // 2 seconds
        
        this.pendingOperations = this.pendingOperations.filter(op => {
          // Check if should retry
          if (now - op.timestamp > retryDelay && op.retryCount < maxRetries) {
            op.retryCount++;
            op.timestamp = now;
            
            // Resend operation
            this.sendEdit(op);
            return true;
          }
          
          // Remove if max retries exceeded
          return op.retryCount < maxRetries;
        });
      }, 1000);
    },
    
    // Clear session data
    clearSession() {
      this.currentSession = null;
      this.activeUsers.clear();
      this.localUser = null;
      this.documentContent = '';
      this.documentVersion = 0;
      this.localVersion = 0;
      this.pendingOperations = [];
      this.operationHistory = [];
      this.remoteCursors.clear();
      this.remoteSelections.clear();
      this.hasConflict = false;
      this.isSyncing = false;
    },
  },
});