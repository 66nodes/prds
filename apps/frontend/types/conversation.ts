// Conversation type declarations
export interface ConversationMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant' | 'system';
  timestamp: string;
  metadata?: Record<string, any>;
  validationResults?: any[];
}

export interface ConversationState {
  messages: ConversationMessage[];
  currentConversationId: string | null;
  isProcessing: boolean;
  processingState: 'idle' | 'generating' | 'validating';
  error: string | null;
  progress: number;
  requiresValidation: boolean;
  validationResults: any[];
}

export interface ConversationStore {
  messages: ConversationMessage[];
  currentConversationId: string | null;
  isProcessing: boolean;
  processingState: 'idle' | 'generating' | 'validating';
  error: string | null;
  progress: number;
  requiresValidation: boolean;
  validationResults: any[];

  // Methods
  addMessage: (message: ConversationMessage) => void;
  updateMessage: (id: string, updates: Partial<ConversationMessage>) => void;
  deleteMessage: (id: string) => void;
  clearMessages: () => void;
  setCurrentConversation: (id: string) => void;
  startNewConversation: () => void;
  setProcessingState: (state: string) => void;
  setError: (error: string | null) => void;
  setProgress: (progress: number) => void;
  updateMetadata: (metadata: Record<string, any>) => void;
  validateCurrentMessage: () => Promise<void>;
}