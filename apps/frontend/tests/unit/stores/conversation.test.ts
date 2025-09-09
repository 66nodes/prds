import { describe, it, expect, beforeEach, vi } from 'vitest';
import { setActivePinia, createPinia } from 'pinia';
import { useConversationStore } from '~/stores/conversation';
import type { ConversationMessage, HumanValidationType } from '~/types';

// Mock crypto.randomUUID
Object.defineProperty(global, 'crypto', {
  value: {
    randomUUID: vi.fn(() => 'test-uuid-123')
  }
});

describe('Conversation Store', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    vi.clearAllMocks();
  });

  describe('Initial State', () => {
    it('should have correct initial state', () => {
      const store = useConversationStore();
      
      expect(store.currentConversation).toBeNull();
      expect(store.messages).toEqual([]);
      expect(store.context.projectId).toBeNull();
      expect(store.context.currentStep).toBe('initial');
      expect(store.context.extractedRequirements).toEqual([]);
      expect(store.context.validationPoints).toEqual([]);
      expect(store.isProcessing).toBe(false);
      expect(store.waitingForHumanValidation).toBe(false);
      expect(store.pendingValidation).toBeNull();
      expect(store.conversationHistory).toEqual([]);
    });
  });

  describe('Getters', () => {
    it('should return latest message correctly', () => {
      const store = useConversationStore();
      
      expect(store.latestMessage).toBeNull();
      
      store.messages = [
        {
          id: '1',
          role: 'user',
          content: 'Hello',
          timestamp: '2023-01-01T00:00:00Z'
        },
        {
          id: '2',
          role: 'assistant',
          content: 'Hi there!',
          timestamp: '2023-01-01T00:01:00Z'
        }
      ];
      
      expect(store.latestMessage?.id).toBe('2');
      expect(store.latestMessage?.content).toBe('Hi there!');
    });

    it('should filter messages by role correctly', () => {
      const store = useConversationStore();
      
      store.messages = [
        { id: '1', role: 'user', content: 'Hello', timestamp: '2023-01-01T00:00:00Z' },
        { id: '2', role: 'assistant', content: 'Hi!', timestamp: '2023-01-01T00:01:00Z' },
        { id: '3', role: 'system', content: 'System message', timestamp: '2023-01-01T00:02:00Z' },
        { id: '4', role: 'user', content: 'How are you?', timestamp: '2023-01-01T00:03:00Z' }
      ];
      
      expect(store.userMessages).toHaveLength(2);
      expect(store.assistantMessages).toHaveLength(1);
      expect(store.systemMessages).toHaveLength(1);
      
      expect(store.userMessages[0].content).toBe('Hello');
      expect(store.userMessages[1].content).toBe('How are you?');
    });

    it('should calculate conversation progress correctly', () => {
      const store = useConversationStore();
      
      expect(store.conversationProgress).toBe(0);
      
      store.context.currentStep = 'requirements_gathering';
      expect(store.conversationProgress).toBe(25);
      
      store.context.currentStep = 'validation';
      expect(store.conversationProgress).toBe(50);
      
      store.context.currentStep = 'completion';
      expect(store.conversationProgress).toBe(100);
    });

    it('should detect validation requirement correctly', () => {
      const store = useConversationStore();
      
      expect(store.requiresValidation).toBe(false);
      
      store.waitingForHumanValidation = true;
      store.pendingValidation = {
        id: 'test-validation',
        type: 'approval' as HumanValidationType,
        question: 'Test question?',
        context: 'Test context',
        required: true
      };
      
      expect(store.requiresValidation).toBe(true);
    });
  });

  describe('Actions', () => {
    describe('startNewConversation', () => {
      it('should initialize new conversation correctly', async () => {
        const store = useConversationStore();
        const projectId = 'project-123';
        
        await store.startNewConversation(projectId);
        
        expect(store.currentConversation).not.toBeNull();
        expect(store.currentConversation?.projectId).toBe(projectId);
        expect(store.currentConversation?.title).toBe('New Planning Session');
        expect(store.currentConversation?.status).toBe('active');
        
        expect(store.context.projectId).toBe(projectId);
        expect(store.context.currentStep).toBe('initial');
        
        expect(store.messages).toHaveLength(1);
        expect(store.messages[0].role).toBe('assistant');
        expect(store.messages[0].content).toContain('Welcome');
      });
    });

    describe('addMessage', () => {
      it('should add message with generated ID and timestamp', async () => {
        const store = useConversationStore();
        await store.startNewConversation('project-123');
        
        const initialMessageCount = store.messages.length;
        
        await store.addMessage({
          role: 'user',
          content: 'Test message'
        });
        
        expect(store.messages).toHaveLength(initialMessageCount + 1);
        
        const newMessage = store.messages[store.messages.length - 1];
        expect(newMessage.id).toBe('test-uuid-123');
        expect(newMessage.role).toBe('user');
        expect(newMessage.content).toBe('Test message');
        expect(newMessage.timestamp).toBeDefined();
      });

      it('should update conversation step based on message count', async () => {
        const store = useConversationStore();
        await store.startNewConversation('project-123');
        
        expect(store.context.currentStep).toBe('initial');
        
        // Add first user message
        await store.addMessage({ role: 'user', content: 'Message 1' });
        expect(store.context.currentStep).toBe('requirements_gathering');
        
        // Add more messages to progress steps
        await store.addMessage({ role: 'user', content: 'Message 2' });
        await store.addMessage({ role: 'user', content: 'Message 3' });
        expect(store.context.currentStep).toBe('validation');
        
        // Continue progression
        await store.addMessage({ role: 'user', content: 'Message 4' });
        await store.addMessage({ role: 'user', content: 'Message 5' });
        expect(store.context.currentStep).toBe('refinement');
        
        // Final step
        await store.addMessage({ role: 'user', content: 'Message 6' });
        await store.addMessage({ role: 'user', content: 'Message 7' });
        expect(store.context.currentStep).toBe('completion');
      });

      it('should update conversation timestamp', async () => {
        const store = useConversationStore();
        await store.startNewConversation('project-123');
        
        const originalTimestamp = store.currentConversation?.updatedAt;
        
        // Wait a bit to ensure timestamp difference
        await new Promise(resolve => setTimeout(resolve, 10));
        
        await store.addMessage({ role: 'user', content: 'Test' });
        
        expect(store.currentConversation?.updatedAt).not.toBe(originalTimestamp);
      });
    });

    describe('addSystemMessage', () => {
      it('should add system message with metadata', async () => {
        const store = useConversationStore();
        await store.startNewConversation('project-123');
        
        await store.addSystemMessage('System notification', { type: 'info' });
        
        const systemMessage = store.messages.find(m => m.role === 'system');
        expect(systemMessage).toBeDefined();
        expect(systemMessage?.content).toBe('System notification');
        expect(systemMessage?.metadata?.type).toBe('info');
      });
    });

    describe('extractRequirementsFromMessages', () => {
      it('should extract requirements from user messages', async () => {
        const store = useConversationStore();
        await store.startNewConversation('project-123');
        
        await store.addMessage({
          role: 'user',
          content: 'I need a user authentication system. The app should have real-time notifications.'
        });
        
        await store.addMessage({
          role: 'user',
          content: 'We must support offline mode and want social media integration.'
        });
        
        await store.extractRequirementsFromMessages();
        
        expect(store.context.extractedRequirements.length).toBeGreaterThan(0);
        expect(store.context.extractedRequirements.some(req => 
          req.includes('authentication') || req.includes('need')
        )).toBe(true);
      });

      it('should remove duplicate requirements', async () => {
        const store = useConversationStore();
        await store.startNewConversation('project-123');
        
        await store.addMessage({
          role: 'user',
          content: 'I need user authentication. I need user authentication.'
        });
        
        await store.extractRequirementsFromMessages();
        
        const duplicateRequirements = store.context.extractedRequirements.filter(req => 
          req.includes('authentication')
        );
        expect(duplicateRequirements.length).toBeLessThanOrEqual(1);
      });
    });

    describe('Human Validation', () => {
      describe('requestHumanValidation', () => {
        it('should set validation state correctly', async () => {
          const store = useConversationStore();
          await store.startNewConversation('project-123');
          
          const prompt = {
            id: 'validation-123',
            type: 'approval' as HumanValidationType,
            question: 'Do you approve this approach?',
            context: 'We suggest using React Native.',
            required: true
          };
          
          await store.requestHumanValidation(prompt);
          
          expect(store.waitingForHumanValidation).toBe(true);
          expect(store.pendingValidation).toEqual(prompt);
          
          // Should add system message
          const systemMessage = store.messages.find(m => 
            m.role === 'system' && m.content.includes('I need your input on')
          );
          expect(systemMessage).toBeDefined();
        });
      });

      describe('submitValidationResponse', () => {
        it('should handle approval correctly', async () => {
          const store = useConversationStore();
          await store.startNewConversation('project-123');
          
          const prompt = {
            id: 'validation-123',
            type: 'approval' as HumanValidationType,
            question: 'Test question?',
            context: 'Test context',
            required: true
          };
          
          await store.requestHumanValidation(prompt);
          
          const initialMessageCount = store.messages.length;
          
          await store.submitValidationResponse('validation-123', { approved: true }, true);
          
          expect(store.waitingForHumanValidation).toBe(false);
          expect(store.pendingValidation).toBeNull();
          expect(store.messages.length).toBe(initialMessageCount + 2); // User response + system confirmation
          
          // Check approval message
          const approvalMessage = store.messages.find(m => 
            m.role === 'user' && m.content === '✅ Approved'
          );
          expect(approvalMessage).toBeDefined();
          expect(approvalMessage?.metadata?.approved).toBe(true);
          
          // Check system confirmation
          const confirmationMessage = store.messages.find(m => 
            m.role === 'system' && m.content.includes('approved approach')
          );
          expect(confirmationMessage).toBeDefined();
        });

        it('should handle rejection correctly', async () => {
          const store = useConversationStore();
          await store.startNewConversation('project-123');
          
          const prompt = {
            id: 'validation-123',
            type: 'approval' as HumanValidationType,
            question: 'Test question?',
            context: 'Test context',
            required: true
          };
          
          await store.requestHumanValidation(prompt);
          
          await store.submitValidationResponse('validation-123', { 
            feedback: 'I disagree with this approach'
          }, false);
          
          expect(store.waitingForHumanValidation).toBe(false);
          expect(store.pendingValidation).toBeNull();
          
          // Check rejection message
          const rejectionMessage = store.messages.find(m => 
            m.role === 'user' && m.content === '❌ Needs revision'
          );
          expect(rejectionMessage).toBeDefined();
          expect(rejectionMessage?.metadata?.approved).toBe(false);
          
          // Check system response
          const responseMessage = store.messages.find(m => 
            m.role === 'system' && m.content.includes('revise the approach')
          );
          expect(responseMessage).toBeDefined();
        });

        it('should store validation results in metadata', async () => {
          const store = useConversationStore();
          await store.startNewConversation('project-123');
          
          const prompt = {
            id: 'validation-123',
            type: 'approval' as HumanValidationType,
            question: 'Test question?',
            context: 'Test context',
            required: true
          };
          
          await store.requestHumanValidation(prompt);
          
          const response = { approved: true, reason: 'Looks good' };
          await store.submitValidationResponse('validation-123', response, true);
          
          expect(store.context.metadata.validationResults).toBeDefined();
          expect(store.context.metadata.validationResults[0]).toEqual({
            id: 'validation-123',
            response,
            approved: true,
            timestamp: expect.any(String)
          });
        });

        it('should throw error for non-matching validation', async () => {
          const store = useConversationStore();
          await store.startNewConversation('project-123');
          
          await expect(
            store.submitValidationResponse('non-existent', {}, true)
          ).rejects.toThrow('No matching validation request found');
        });
      });
    });

    describe('addValidationPoint', () => {
      it('should add validation point correctly', async () => {
        const store = useConversationStore();
        await store.startNewConversation('project-123');
        
        await store.addValidationPoint({
          type: 'requirement',
          content: 'User authentication is required',
          confidence: 0.9
        });
        
        expect(store.context.validationPoints).toHaveLength(1);
        
        const validationPoint = store.context.validationPoints[0];
        expect(validationPoint.id).toBeDefined();
        expect(validationPoint.type).toBe('requirement');
        expect(validationPoint.content).toBe('User authentication is required');
        expect(validationPoint.confidence).toBe(0.9);
        expect(validationPoint.validated).toBe(false);
        expect(validationPoint.timestamp).toBeDefined();
      });
    });

    describe('saveToHistory and loadConversation', () => {
      it('should save conversation to history', async () => {
        const store = useConversationStore();
        await store.startNewConversation('project-123');
        
        await store.addMessage({ role: 'user', content: 'Test message' });
        
        expect(store.conversationHistory).toHaveLength(1);
        
        const historyItem = store.conversationHistory[0];
        expect(historyItem.id).toBe(store.currentConversation?.id);
        expect(historyItem.messageCount).toBe(store.messages.length);
        expect(historyItem.lastMessage).toBe('Test message');
      });

      it('should limit history to 10 items', async () => {
        const store = useConversationStore();
        
        // Create 12 conversations
        for (let i = 0; i < 12; i++) {
          await store.startNewConversation(`project-${i}`);
          await store.addMessage({ role: 'user', content: `Message ${i}` });
        }
        
        expect(store.conversationHistory).toHaveLength(10);
        
        // Most recent should be first
        expect(store.conversationHistory[0].lastMessage).toBe('Message 11');
      });

      it('should load conversation from history', async () => {
        const store = useConversationStore();
        await store.startNewConversation('project-123');
        
        const conversationId = store.currentConversation?.id;
        await store.addMessage({ role: 'user', content: 'Test message' });
        
        // End current conversation
        await store.endConversation();
        
        expect(store.currentConversation).toBeNull();
        expect(store.messages).toHaveLength(0);
        
        // Load from history
        await store.loadConversation(conversationId!);
        
        expect(store.currentConversation?.id).toBe(conversationId);
        expect(store.context.projectId).toBe('project-123');
      });
    });

    describe('endConversation', () => {
      it('should end conversation and reset state', async () => {
        const store = useConversationStore();
        await store.startNewConversation('project-123');
        await store.addMessage({ role: 'user', content: 'Test' });
        
        await store.endConversation();
        
        expect(store.currentConversation).toBeNull();
        expect(store.messages).toHaveLength(0);
        expect(store.context.projectId).toBeNull();
        expect(store.context.currentStep).toBe('initial');
        expect(store.isProcessing).toBe(false);
        expect(store.waitingForHumanValidation).toBe(false);
        expect(store.pendingValidation).toBeNull();
      });

      it('should mark conversation as completed in history', async () => {
        const store = useConversationStore();
        await store.startNewConversation('project-123');
        
        const conversationId = store.currentConversation?.id;
        
        await store.endConversation();
        
        const historyItem = store.conversationHistory.find(conv => conv.id === conversationId);
        expect(historyItem?.status).toBe('completed');
      });
    });

    describe('updateMetadata', () => {
      it('should update conversation metadata', async () => {
        const store = useConversationStore();
        await store.startNewConversation('project-123');
        
        await store.updateMetadata('testKey', 'testValue');
        
        expect(store.context.metadata.testKey).toBe('testValue');
      });
    });

    describe('setProcessingState', () => {
      it('should update processing state', async () => {
        const store = useConversationStore();
        
        expect(store.isProcessing).toBe(false);
        
        await store.setProcessingState(true);
        expect(store.isProcessing).toBe(true);
        
        await store.setProcessingState(false);
        expect(store.isProcessing).toBe(false);
      });
    });
  });
});