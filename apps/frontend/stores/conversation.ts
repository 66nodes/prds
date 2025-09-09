import { defineStore } from 'pinia';
import type { 
  ConversationMessage, 
  ConversationState, 
  ConversationContext,
  ValidationRequest,
  HumanValidationPrompt
} from '~/types';

export const useConversationStore = defineStore('conversation', {
  state: (): ConversationState => ({
    currentConversation: null,
    messages: [],
    context: {
      projectId: null,
      currentStep: 'initial',
      extractedRequirements: [],
      validationPoints: [],
      metadata: {}
    },
    isProcessing: false,
    waitingForHumanValidation: false,
    pendingValidation: null,
    conversationHistory: []
  }),

  getters: {
    latestMessage: (state): ConversationMessage | null => 
      state.messages.length > 0 ? state.messages[state.messages.length - 1] : null,
    
    userMessages: (state): ConversationMessage[] => 
      state.messages.filter(msg => msg.role === 'user'),
    
    assistantMessages: (state): ConversationMessage[] => 
      state.messages.filter(msg => msg.role === 'assistant'),
    
    systemMessages: (state): ConversationMessage[] => 
      state.messages.filter(msg => msg.role === 'system'),

    requiresValidation: (state): boolean =>
      state.waitingForHumanValidation && state.pendingValidation !== null,

    conversationProgress: (state): number => {
      const steps = ['initial', 'requirements_gathering', 'validation', 'refinement', 'completion'];
      const currentIndex = steps.indexOf(state.context.currentStep);
      return currentIndex >= 0 ? (currentIndex / (steps.length - 1)) * 100 : 0;
    }
  },

  actions: {
    async startNewConversation(projectId: string) {
      this.currentConversation = {
        id: crypto.randomUUID(),
        projectId,
        title: 'New Planning Session',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        status: 'active'
      };

      this.context = {
        projectId,
        currentStep: 'initial',
        extractedRequirements: [],
        validationPoints: [],
        metadata: {}
      };

      this.messages = [{
        id: crypto.randomUUID(),
        role: 'assistant',
        content: 'Welcome! I\'ll help you create a comprehensive Product Requirements Document. Let\'s start by discussing your project vision. What are you looking to build?',
        timestamp: new Date().toISOString(),
        metadata: {
          type: 'greeting',
          step: 'initial'
        }
      }];

      this.saveToHistory();
    },

    async addMessage(message: Omit<ConversationMessage, 'id' | 'timestamp'>) {
      const newMessage: ConversationMessage = {
        ...message,
        id: crypto.randomUUID(),
        timestamp: new Date().toISOString()
      };

      this.messages.push(newMessage);
      
      if (this.currentConversation) {
        this.currentConversation.updatedAt = new Date().toISOString();
      }

      // Auto-advance conversation step based on message content
      await this.updateConversationStep();
      
      this.saveToHistory();
    },

    async addSystemMessage(content: string, metadata?: Record<string, any>) {
      await this.addMessage({
        role: 'system',
        content,
        metadata
      });
    },

    async updateConversationStep() {
      const currentMessages = this.messages.length;
      const userMessages = this.userMessages.length;

      // Simple step progression logic
      if (userMessages >= 1 && this.context.currentStep === 'initial') {
        this.context.currentStep = 'requirements_gathering';
      } else if (userMessages >= 3 && this.context.currentStep === 'requirements_gathering') {
        this.context.currentStep = 'validation';
      } else if (userMessages >= 5 && this.context.currentStep === 'validation') {
        this.context.currentStep = 'refinement';
      } else if (userMessages >= 7 && this.context.currentStep === 'refinement') {
        this.context.currentStep = 'completion';
      }
    },

    async extractRequirementsFromMessages() {
      // Extract potential requirements from user messages
      const requirements: string[] = [];
      
      this.userMessages.forEach(message => {
        // Simple keyword-based extraction (would be enhanced with NLP in production)
        const requirementKeywords = ['need', 'want', 'should', 'must', 'require', 'feature'];
        const sentences = message.content.split(/[.!?]+/);
        
        sentences.forEach(sentence => {
          const lowerSentence = sentence.toLowerCase();
          if (requirementKeywords.some(keyword => lowerSentence.includes(keyword))) {
            requirements.push(sentence.trim());
          }
        });
      });

      this.context.extractedRequirements = [...new Set(requirements)]; // Remove duplicates
    },

    async addValidationPoint(validationPoint: {
      type: 'requirement' | 'assumption' | 'constraint';
      content: string;
      confidence: number;
    }) {
      this.context.validationPoints.push({
        id: crypto.randomUUID(),
        ...validationPoint,
        timestamp: new Date().toISOString(),
        validated: false
      });
    },

    async requestHumanValidation(prompt: HumanValidationPrompt) {
      this.waitingForHumanValidation = true;
      this.pendingValidation = prompt;
      
      await this.addSystemMessage(
        `🤔 I need your input on: ${prompt.question}`,
        { type: 'validation_request', validationId: prompt.id }
      );
    },

    async submitValidationResponse(validationId: string, response: any, approved: boolean) {
      if (!this.pendingValidation || this.pendingValidation.id !== validationId) {
        throw new Error('No matching validation request found');
      }

      // Store validation result
      const validationResult = {
        id: validationId,
        response,
        approved,
        timestamp: new Date().toISOString()
      };

      this.context.metadata.validationResults = this.context.metadata.validationResults || [];
      this.context.metadata.validationResults.push(validationResult);

      // Add confirmation message
      await this.addMessage({
        role: 'user',
        content: approved ? '✅ Approved' : '❌ Needs revision',
        metadata: {
          type: 'validation_response',
          validationId,
          approved,
          response
        }
      });

      // Clear validation state
      this.waitingForHumanValidation = false;
      this.pendingValidation = null;

      // Continue conversation based on validation result
      if (approved) {
        await this.addSystemMessage('Great! Continuing with the approved approach...');
      } else {
        await this.addSystemMessage('I\'ll revise the approach based on your feedback...');
      }
    },

    async saveToHistory() {
      if (this.currentConversation) {
        const existingIndex = this.conversationHistory.findIndex(
          conv => conv.id === this.currentConversation?.id
        );

        const conversationSnapshot = {
          ...this.currentConversation,
          messageCount: this.messages.length,
          lastMessage: this.latestMessage?.content || ''
        };

        if (existingIndex >= 0) {
          this.conversationHistory[existingIndex] = conversationSnapshot;
        } else {
          this.conversationHistory.unshift(conversationSnapshot);
        }

        // Keep only latest 10 conversations in memory
        if (this.conversationHistory.length > 10) {
          this.conversationHistory = this.conversationHistory.slice(0, 10);
        }
      }
    },

    async loadConversation(conversationId: string) {
      const conversation = this.conversationHistory.find(conv => conv.id === conversationId);
      if (conversation) {
        this.currentConversation = conversation;
        // In a real app, you'd load the full messages from backend
        // For now, we'll start fresh but keep the conversation metadata
        this.messages = [];
        this.context = {
          projectId: conversation.projectId,
          currentStep: 'initial',
          extractedRequirements: [],
          validationPoints: [],
          metadata: {}
        };
      }
    },

    async endConversation() {
      if (this.currentConversation) {
        this.currentConversation.status = 'completed';
        this.currentConversation.updatedAt = new Date().toISOString();
        this.saveToHistory();
      }

      // Reset state
      this.currentConversation = null;
      this.messages = [];
      this.context = {
        projectId: null,
        currentStep: 'initial',
        extractedRequirements: [],
        validationPoints: [],
        metadata: {}
      };
      this.isProcessing = false;
      this.waitingForHumanValidation = false;
      this.pendingValidation = null;
    },

    async updateMetadata(key: string, value: any) {
      this.context.metadata[key] = value;
      this.saveToHistory();
    },

    async setProcessingState(processing: boolean) {
      this.isProcessing = processing;
    }
  },

  persist: {
    storage: persistedState.localStorage,
    paths: ['conversationHistory', 'context']
  }
});