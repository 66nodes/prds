<template>
  <div class="conversation-interface flex flex-col h-full">
    <!-- Conversation Header -->
    <div class="conversation-header border-b border-gray-200 dark:border-gray-700 p-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center space-x-3">
          <UIcon name="i-heroicons-chat-bubble-left-right" class="w-6 h-6 text-primary" />
          <div>
            <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
              {{ conversation.currentConversation?.title || 'Planning Session' }}
            </h3>
            <p class="text-sm text-gray-500 dark:text-gray-400">
              Step: {{ formatStep(conversation.context.currentStep) }}
              <span v-if="conversation.conversationProgress > 0" class="ml-2">
                ({{ Math.round(conversation.conversationProgress) }}% complete)
              </span>
            </p>
          </div>
        </div>
        
        <div class="flex items-center space-x-2">
          <!-- Progress Indicator -->
          <div class="w-32 bg-gray-200 rounded-full h-2 dark:bg-gray-700">
            <div 
              class="bg-primary h-2 rounded-full transition-all duration-300"
              :style="{ width: `${conversation.conversationProgress}%` }"
            ></div>
          </div>
          
          <!-- Connection Status -->
          <div class="flex items-center space-x-1">
            <div 
              :class="[
                'w-2 h-2 rounded-full',
                webSocket.isConnected ? 'bg-green-500' : 'bg-red-500'
              ]"
            ></div>
            <span class="text-xs text-gray-500">
              {{ webSocket.isConnected ? 'Connected' : 'Disconnected' }}
            </span>
          </div>
        </div>
      </div>
      
      <!-- Validation Alert -->
      <UAlert
        v-if="conversation.requiresValidation"
        icon="i-heroicons-exclamation-triangle"
        color="amber"
        variant="soft"
        :title="'Human Input Required'"
        :description="conversation.pendingValidation?.question || 'Please review and approve the current step'"
        class="mt-4"
      />
    </div>

    <!-- Messages Container -->
    <div 
      ref="messagesContainer"
      class="conversation-messages flex-1 overflow-y-auto p-4 space-y-4"
    >
      <TransitionGroup
        name="message"
        tag="div"
        class="space-y-4"
      >
        <ConversationMessage
          v-for="message in conversation.messages"
          :key="message.id"
          :message="message"
          :is-processing="conversation.isProcessing && isLastMessage(message)"
        />
      </TransitionGroup>
      
      <!-- Processing Indicator -->
      <div v-if="conversation.isProcessing" class="flex items-center space-x-2 text-gray-500">
        <UIcon name="i-heroicons-arrow-path" class="w-4 h-4 animate-spin" />
        <span class="text-sm">AI is thinking...</span>
      </div>
    </div>

    <!-- Human Validation Prompt -->
    <ValidationPrompt
      v-if="conversation.requiresValidation && conversation.pendingValidation"
      :prompt="conversation.pendingValidation"
      @approve="handleValidationResponse($event, true)"
      @reject="handleValidationResponse($event, false)"
      class="border-t border-gray-200 dark:border-gray-700"
    />

    <!-- Message Input -->
    <div class="conversation-input border-t border-gray-200 dark:border-gray-700 p-4">
      <div class="flex space-x-2">
        <UTextarea
          v-model="currentMessage"
          :disabled="conversation.isProcessing || conversation.requiresValidation"
          placeholder="Describe your project requirements..."
          :rows="1"
          autoresize
          class="flex-1"
          @keydown.enter.exact.prevent="sendMessage"
          @keydown.enter.shift.exact="$event.target.value += '\n'"
        />
        <UButton
          :disabled="!canSendMessage"
          :loading="conversation.isProcessing"
          @click="sendMessage"
          icon="i-heroicons-paper-airplane"
          size="lg"
        />
      </div>
      
      <!-- Quick Actions -->
      <div class="flex items-center justify-between mt-2">
        <div class="flex space-x-2">
          <UButton
            v-if="conversation.context.extractedRequirements.length > 0"
            variant="ghost"
            size="xs"
            icon="i-heroicons-list-bullet"
            @click="showExtractedRequirements = !showExtractedRequirements"
          >
            Requirements ({{ conversation.context.extractedRequirements.length }})
          </UButton>
          <UButton
            v-if="conversation.context.validationPoints.length > 0"
            variant="ghost"
            size="xs"
            icon="i-heroicons-check-circle"
            @click="showValidationPoints = !showValidationPoints"
          >
            Validations ({{ conversation.context.validationPoints.length }})
          </UButton>
        </div>
        
        <div class="flex space-x-2">
          <UButton
            variant="ghost"
            size="xs"
            icon="i-heroicons-arrow-path"
            @click="resetConversation"
          >
            Reset
          </UButton>
          <UButton
            v-if="canComplete"
            color="green"
            size="xs"
            icon="i-heroicons-check"
            @click="completeConversation"
          >
            Generate PRD
          </UButton>
        </div>
      </div>
    </div>

    <!-- Extracted Requirements Panel -->
    <USlideover v-model="showExtractedRequirements" side="right">
      <UCard>
        <template #header>
          <div class="flex items-center justify-between">
            <h3 class="text-lg font-semibold">Extracted Requirements</h3>
            <UButton
              color="gray"
              variant="ghost"
              icon="i-heroicons-x-mark"
              @click="showExtractedRequirements = false"
            />
          </div>
        </template>
        
        <div class="space-y-2">
          <div
            v-for="(requirement, index) in conversation.context.extractedRequirements"
            :key="index"
            class="p-3 bg-gray-50 dark:bg-gray-800 rounded-md"
          >
            <p class="text-sm">{{ requirement }}</p>
          </div>
        </div>
      </UCard>
    </USlideover>

    <!-- Validation Points Panel -->
    <USlideover v-model="showValidationPoints" side="right">
      <UCard>
        <template #header>
          <div class="flex items-center justify-between">
            <h3 class="text-lg font-semibold">Validation Points</h3>
            <UButton
              color="gray"
              variant="ghost"
              icon="i-heroicons-x-mark"
              @click="showValidationPoints = false"
            />
          </div>
        </template>
        
        <div class="space-y-3">
          <div
            v-for="point in conversation.context.validationPoints"
            :key="point.id"
            class="p-3 border rounded-md"
            :class="[
              point.validated 
                ? 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20'
                : 'border-yellow-200 bg-yellow-50 dark:border-yellow-800 dark:bg-yellow-900/20'
            ]"
          >
            <div class="flex items-center justify-between mb-2">
              <UBadge 
                :color="point.type === 'requirement' ? 'blue' : point.type === 'assumption' ? 'yellow' : 'red'"
                size="xs"
              >
                {{ point.type }}
              </UBadge>
              <span class="text-xs text-gray-500">{{ point.confidence }}% confidence</span>
            </div>
            <p class="text-sm">{{ point.content }}</p>
          </div>
        </div>
      </UCard>
    </USlideover>
  </div>
</template>

<script setup lang="ts">
import type { ConversationMessage } from '~/types';

interface Props {
  projectId: string;
}

const props = defineProps<Props>();

// Store references
const conversation = useConversationStore();
const webSocket = useWebSocket();
const { $api } = useApiClient();

// Component state
const currentMessage = ref('');
const messagesContainer = ref<HTMLElement>();
const showExtractedRequirements = ref(false);
const showValidationPoints = ref(false);

// Computed properties
const canSendMessage = computed(() => 
  currentMessage.value.trim().length > 0 && 
  !conversation.isProcessing && 
  !conversation.requiresValidation
);

const canComplete = computed(() => 
  conversation.context.currentStep === 'completion' ||
  (conversation.context.currentStep === 'refinement' && 
   conversation.context.extractedRequirements.length >= 3)
);

// Methods
const formatStep = (step: string): string => {
  return step.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
};

const isLastMessage = (message: ConversationMessage): boolean => {
  return conversation.latestMessage?.id === message.id;
};

const sendMessage = async () => {
  if (!canSendMessage.value) return;

  const messageContent = currentMessage.value.trim();
  currentMessage.value = '';

  try {
    conversation.setProcessingState(true);

    // Add user message
    await conversation.addMessage({
      role: 'user',
      content: messageContent
    });

    // Extract requirements from the message
    await conversation.extractRequirementsFromMessages();

    // Send to AI via WebSocket
    webSocket.send('conversation_message', {
      conversationId: conversation.currentConversation?.id,
      projectId: props.projectId,
      message: messageContent,
      context: conversation.context
    });

    // Scroll to bottom
    await nextTick();
    scrollToBottom();

  } catch (error) {
    console.error('Failed to send message:', error);
    useToast().add({
      title: 'Error',
      description: 'Failed to send message. Please try again.',
      color: 'red'
    });
  }
};

const handleValidationResponse = async (response: any, approved: boolean) => {
  if (!conversation.pendingValidation) return;

  try {
    await conversation.submitValidationResponse(
      conversation.pendingValidation.id,
      response,
      approved
    );

    // Send validation result via WebSocket
    webSocket.send('human_validation_request', {
      validationId: conversation.pendingValidation.id,
      response,
      approved,
      conversationId: conversation.currentConversation?.id
    });

  } catch (error) {
    console.error('Failed to submit validation:', error);
    useToast().add({
      title: 'Error',
      description: 'Failed to submit validation. Please try again.',
      color: 'red'
    });
  }
};

const resetConversation = async () => {
  if (confirm('Are you sure you want to reset the conversation?')) {
    await conversation.startNewConversation(props.projectId);
    scrollToBottom();
  }
};

const completeConversation = async () => {
  try {
    conversation.setProcessingState(true);

    // Generate PRD based on conversation
    const response = await $api.post(`/projects/${props.projectId}/prd/from-conversation`, {
      conversationId: conversation.currentConversation?.id,
      messages: conversation.messages,
      context: conversation.context
    });

    await conversation.endConversation();
    
    useToast().add({
      title: 'Success',
      description: 'PRD generated successfully from conversation!',
      color: 'green'
    });

    // Navigate to the generated PRD
    await navigateTo(`/projects/${props.projectId}/prd/${response.data.id}`);

  } catch (error) {
    console.error('Failed to generate PRD:', error);
    useToast().add({
      title: 'Error',
      description: 'Failed to generate PRD. Please try again.',
      color: 'red'
    });
  } finally {
    conversation.setProcessingState(false);
  }
};

const scrollToBottom = () => {
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight;
  }
};

// WebSocket event handlers
const handleConversationMessage = (data: any) => {
  conversation.addMessage({
    role: 'assistant',
    content: data.content,
    metadata: data.metadata
  });
  
  conversation.setProcessingState(false);
  nextTick(() => scrollToBottom());
};

const handleValidationRequest = (data: any) => {
  conversation.requestHumanValidation(data);
};

const handleStepUpdate = (data: any) => {
  conversation.context.currentStep = data.step;
  conversation.updateMetadata('stepData', data);
};

// Initialize conversation on mount
onMounted(async () => {
  // Start new conversation if none exists
  if (!conversation.currentConversation) {
    await conversation.startNewConversation(props.projectId);
  }

  // Setup WebSocket listeners
  webSocket.on('conversation_message', handleConversationMessage);
  webSocket.on('human_validation_request', handleValidationRequest);
  webSocket.on('conversation_step_update', handleStepUpdate);

  scrollToBottom();
});

onUnmounted(() => {
  // Cleanup WebSocket listeners
  webSocket.off('conversation_message', handleConversationMessage);
  webSocket.off('human_validation_request', handleValidationRequest);
  webSocket.off('conversation_step_update', handleStepUpdate);
});

// Watch for new messages to auto-scroll
watch(
  () => conversation.messages.length,
  () => {
    nextTick(() => scrollToBottom());
  }
);
</script>

<style scoped>
.conversation-interface {
  min-height: 600px;
}

.conversation-messages {
  scroll-behavior: smooth;
}

/* Message animations */
.message-enter-active,
.message-leave-active {
  transition: all 0.3s ease;
}

.message-enter-from {
  opacity: 0;
  transform: translateY(20px);
}

.message-leave-to {
  opacity: 0;
  transform: translateX(-20px);
}
</style>