<template>
  <div 
    class="conversation-message"
    :class="[
      message.role === 'user' ? 'message-user' : 'message-assistant',
      message.role === 'system' ? 'message-system' : ''
    ]"
  >
    <div class="message-container">
      <!-- Avatar -->
      <div class="message-avatar">
        <div
          :class="[
            'w-8 h-8 rounded-full flex items-center justify-center',
            message.role === 'user' 
              ? 'bg-primary text-white' 
              : message.role === 'system'
              ? 'bg-yellow-500 text-white'
              : 'bg-gray-200 dark:bg-gray-700'
          ]"
        >
          <UIcon
            :name="getMessageIcon(message.role)"
            class="w-4 h-4"
          />
        </div>
      </div>

      <!-- Message Content -->
      <div class="message-content">
        <div class="message-header">
          <span class="message-role">
            {{ message.role === 'user' ? 'You' : message.role === 'system' ? 'System' : 'AI Assistant' }}
          </span>
          <span class="message-timestamp">
            {{ formatTimestamp(message.timestamp) }}
          </span>
        </div>

        <div 
          class="message-body"
          :class="[
            message.role === 'user' ? 'user-message' : 'assistant-message',
            message.role === 'system' ? 'system-message' : ''
          ]"
        >
          <!-- Processing Indicator -->
          <div v-if="isProcessing" class="processing-indicator">
            <div class="typing-dots">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>

          <!-- Message Text -->
          <div v-else class="message-text">
            <!-- System message with special styling -->
            <div v-if="message.role === 'system'" class="system-message-content">
              <UIcon 
                :name="getSystemMessageIcon()" 
                class="w-4 h-4 inline-block mr-2"
              />
              <span v-html="formatMessageContent(message.content)"></span>
            </div>
            
            <!-- Regular message -->
            <div v-else v-html="formatMessageContent(message.content)"></div>
          </div>

          <!-- Message Metadata -->
          <div v-if="message.metadata && hasVisibleMetadata(message.metadata)" class="message-metadata">
            <UBadge
              v-if="message.metadata.type"
              :color="getMetadataColor(message.metadata.type)"
              size="xs"
              class="mr-2"
            >
              {{ formatMetadataType(message.metadata.type) }}
            </UBadge>
            
            <UBadge
              v-if="message.metadata.step"
              color="gray"
              size="xs"
              class="mr-2"
            >
              {{ formatStep(message.metadata.step) }}
            </UBadge>

            <UBadge
              v-if="message.metadata.confidence"
              :color="getConfidenceColor(message.metadata.confidence)"
              size="xs"
            >
              {{ message.metadata.confidence }}% confidence
            </UBadge>
          </div>

          <!-- Validation Response Indicator -->
          <div 
            v-if="message.metadata?.type === 'validation_response'" 
            class="validation-response"
          >
            <div 
              :class="[
                'inline-flex items-center px-2 py-1 rounded text-xs',
                message.metadata.approved 
                  ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
                  : 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
              ]"
            >
              <UIcon 
                :name="message.metadata.approved ? 'i-heroicons-check' : 'i-heroicons-x-mark'" 
                class="w-3 h-3 mr-1"
              />
              {{ message.metadata.approved ? 'Approved' : 'Needs Revision' }}
            </div>
          </div>
        </div>

        <!-- Action Buttons -->
        <div v-if="showActions" class="message-actions">
          <UButton
            v-if="message.role === 'assistant'"
            variant="ghost"
            size="xs"
            icon="i-heroicons-clipboard-document"
            @click="copyMessage"
          >
            Copy
          </UButton>
          
          <UButton
            v-if="message.role === 'assistant'"
            variant="ghost"
            size="xs"
            icon="i-heroicons-arrow-path"
            @click="regenerateMessage"
          >
            Regenerate
          </UButton>

          <UButton
            v-if="message.role !== 'system'"
            variant="ghost"
            size="xs"
            icon="i-heroicons-thumbs-up"
            :color="feedback === 'positive' ? 'green' : 'gray'"
            @click="submitFeedback('positive')"
          />

          <UButton
            v-if="message.role !== 'system'"
            variant="ghost"
            size="xs"
            icon="i-heroicons-thumbs-down"
            :color="feedback === 'negative' ? 'red' : 'gray'"
            @click="submitFeedback('negative')"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { format } from 'date-fns';
import type { ConversationMessage } from '~/types';

interface Props {
  message: ConversationMessage;
  isProcessing?: boolean;
  showActions?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  isProcessing: false,
  showActions: true
});

// Component state
const feedback = ref<'positive' | 'negative' | null>(null);

// Methods
const getMessageIcon = (role: string): string => {
  switch (role) {
    case 'user': return 'i-heroicons-user';
    case 'system': return 'i-heroicons-cog-6-tooth';
    default: return 'i-heroicons-cpu-chip';
  }
};

const getSystemMessageIcon = (): string => {
  const type = props.message.metadata?.type;
  switch (type) {
    case 'validation_request': return 'i-heroicons-exclamation-triangle';
    case 'greeting': return 'i-heroicons-hand-raised';
    case 'step_change': return 'i-heroicons-arrow-right';
    default: return 'i-heroicons-information-circle';
  }
};

const formatTimestamp = (timestamp: string): string => {
  return format(new Date(timestamp), 'HH:mm');
};

const formatMessageContent = (content: string): string => {
  // Basic markdown-like formatting
  return content
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`(.*?)`/g, '<code class="bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded text-sm">$1</code>')
    .replace(/\n/g, '<br>');
};

const formatStep = (step: string): string => {
  return step.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
};

const formatMetadataType = (type: string): string => {
  return type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
};

const getMetadataColor = (type: string): string => {
  switch (type) {
    case 'greeting': return 'blue';
    case 'validation_request': return 'amber';
    case 'validation_response': return 'green';
    case 'step_change': return 'purple';
    case 'requirement_extraction': return 'cyan';
    default: return 'gray';
  }
};

const getConfidenceColor = (confidence: number): string => {
  if (confidence >= 80) return 'green';
  if (confidence >= 60) return 'yellow';
  return 'red';
};

const hasVisibleMetadata = (metadata: Record<string, any>): boolean => {
  return metadata.type !== 'validation_response' && 
         (metadata.type || metadata.step || metadata.confidence);
};

const copyMessage = async () => {
  try {
    await navigator.clipboard.writeText(props.message.content);
    useToast().add({
      title: 'Copied',
      description: 'Message copied to clipboard',
      color: 'green'
    });
  } catch (error) {
    console.error('Failed to copy message:', error);
  }
};

const regenerateMessage = async () => {
  // Emit event to parent component to handle regeneration
  await $emit('regenerate', props.message.id);
};

const submitFeedback = async (type: 'positive' | 'negative') => {
  feedback.value = feedback.value === type ? null : type;
  
  // TODO: Send feedback to backend
  console.log(`Feedback for message ${props.message.id}: ${feedback.value}`);
};

const $emit = defineEmits<{
  regenerate: [messageId: string];
  feedback: [messageId: string, feedback: 'positive' | 'negative' | null];
}>();
</script>

<style scoped>
.conversation-message {
  @apply w-full;
}

.message-user {
  @apply flex justify-end;
}

.message-user .message-container {
  @apply flex-row-reverse;
}

.message-assistant,
.message-system {
  @apply flex justify-start;
}

.message-container {
  @apply flex space-x-3 max-w-4xl;
}

.message-user .message-container {
  @apply space-x-reverse space-x-3;
}

.message-avatar {
  @apply flex-shrink-0;
}

.message-content {
  @apply flex-1 min-w-0;
}

.message-header {
  @apply flex items-center justify-between mb-1;
}

.message-role {
  @apply text-sm font-medium text-gray-900 dark:text-white;
}

.message-timestamp {
  @apply text-xs text-gray-500 dark:text-gray-400;
}

.message-body {
  @apply rounded-lg p-3 shadow-sm;
}

.user-message {
  @apply bg-primary text-white;
}

.assistant-message {
  @apply bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white;
}

.system-message {
  @apply bg-yellow-50 dark:bg-yellow-900/20 text-yellow-900 dark:text-yellow-400 border border-yellow-200 dark:border-yellow-800;
}

.system-message-content {
  @apply flex items-start;
}

.message-text {
  @apply leading-relaxed;
}

.message-metadata {
  @apply flex flex-wrap gap-1 mt-2;
}

.message-actions {
  @apply flex items-center space-x-2 mt-2 opacity-0 group-hover:opacity-100 transition-opacity;
}

.conversation-message:hover .message-actions {
  @apply opacity-100;
}

.processing-indicator {
  @apply flex items-center;
}

.typing-dots {
  @apply flex space-x-1;
}

.typing-dots span {
  @apply w-2 h-2 bg-gray-400 rounded-full animate-pulse;
  animation-delay: calc(var(--i) * 0.2s);
}

.typing-dots span:nth-child(1) { --i: 0; }
.typing-dots span:nth-child(2) { --i: 1; }
.typing-dots span:nth-child(3) { --i: 2; }

.validation-response {
  @apply mt-2;
}

/* Code styling */
:deep(code) {
  @apply font-mono text-sm;
}

/* Link styling */
:deep(a) {
  @apply text-blue-600 dark:text-blue-400 underline hover:no-underline;
}
</style>