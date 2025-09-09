<template>
  <div class="validation-prompt bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 p-4">
    <div class="flex items-start space-x-3">
      <!-- Icon -->
      <div class="flex-shrink-0">
        <UIcon 
          name="i-heroicons-exclamation-triangle" 
          class="w-6 h-6 text-amber-600 dark:text-amber-400"
        />
      </div>

      <!-- Content -->
      <div class="flex-1 min-w-0">
        <h4 class="text-lg font-medium text-amber-900 dark:text-amber-100 mb-2">
          Human Input Required
        </h4>
        
        <p class="text-amber-800 dark:text-amber-200 mb-4">
          {{ prompt.question }}
        </p>

        <!-- Context -->
        <div v-if="prompt.context" class="mb-4 p-3 bg-white dark:bg-gray-800 rounded-md border">
          <h5 class="text-sm font-medium text-gray-900 dark:text-white mb-2">Context:</h5>
          <p class="text-sm text-gray-700 dark:text-gray-300">{{ prompt.context }}</p>
        </div>

        <!-- Different validation types -->
        <div class="validation-input">
          <!-- Approval Type -->
          <div v-if="prompt.type === 'approval'" class="space-y-3">
            <div class="flex space-x-3">
              <UButton
                color="green"
                @click="handleApprove"
                :loading="isSubmitting"
              >
                <UIcon name="i-heroicons-check" class="w-4 h-4 mr-2" />
                Approve
              </UButton>
              <UButton
                color="red"
                variant="outline"
                @click="handleReject"
                :loading="isSubmitting"
              >
                <UIcon name="i-heroicons-x-mark" class="w-4 h-4 mr-2" />
                Reject
              </UButton>
            </div>
            
            <!-- Optional feedback -->
            <UTextarea
              v-model="feedback"
              placeholder="Optional feedback or comments..."
              :rows="2"
              class="mt-2"
            />
          </div>

          <!-- Choice Type -->
          <div v-else-if="prompt.type === 'choice' && prompt.options" class="space-y-3">
            <URadioGroup 
              v-model="selectedChoice"
              :options="prompt.options.map(opt => ({ 
                label: opt.label, 
                value: opt.value, 
                description: opt.description 
              }))"
            />
            
            <div class="flex space-x-3 pt-2">
              <UButton
                :disabled="!selectedChoice"
                @click="handleChoiceSubmit"
                :loading="isSubmitting"
              >
                Submit Choice
              </UButton>
            </div>
          </div>

          <!-- Input Type -->
          <div v-else-if="prompt.type === 'input'" class="space-y-3">
            <UTextarea
              v-model="inputValue"
              placeholder="Please provide your input..."
              :rows="3"
              :required="prompt.required"
            />
            
            <div class="flex space-x-3">
              <UButton
                :disabled="prompt.required && !inputValue.trim()"
                @click="handleInputSubmit"
                :loading="isSubmitting"
              >
                Submit
              </UButton>
            </div>
          </div>

          <!-- Review Type -->
          <div v-else-if="prompt.type === 'review'" class="space-y-3">
            <div class="bg-white dark:bg-gray-800 p-4 rounded-md border">
              <div v-if="reviewContent" v-html="formatReviewContent(reviewContent)"></div>
              <div v-else class="text-gray-500">No content to review</div>
            </div>
            
            <UTextarea
              v-model="reviewFeedback"
              placeholder="Review comments and suggestions..."
              :rows="3"
            />
            
            <div class="flex space-x-3">
              <UButton
                color="green"
                @click="handleReviewApprove"
                :loading="isSubmitting"
              >
                <UIcon name="i-heroicons-check" class="w-4 h-4 mr-2" />
                Approve
              </UButton>
              <UButton
                color="yellow"
                @click="handleReviewRequestChanges"
                :loading="isSubmitting"
              >
                <UIcon name="i-heroicons-pencil" class="w-4 h-4 mr-2" />
                Request Changes
              </UButton>
              <UButton
                color="red"
                variant="outline"
                @click="handleReviewReject"
                :loading="isSubmitting"
              >
                <UIcon name="i-heroicons-x-mark" class="w-4 h-4 mr-2" />
                Reject
              </UButton>
            </div>
          </div>

          <!-- Confirmation Type -->
          <div v-else-if="prompt.type === 'confirmation'" class="space-y-3">
            <UCheckbox 
              v-model="confirmed"
              :label="prompt.question"
            />
            
            <div class="flex space-x-3 pt-2">
              <UButton
                :disabled="!confirmed"
                @click="handleConfirm"
                :loading="isSubmitting"
              >
                Confirm
              </UButton>
              <UButton
                variant="outline"
                @click="handleCancel"
                :loading="isSubmitting"
              >
                Cancel
              </UButton>
            </div>
          </div>
        </div>

        <!-- Timeout indicator -->
        <div v-if="prompt.timeout && timeRemaining > 0" class="mt-4 flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-400">
          <UIcon name="i-heroicons-clock" class="w-4 h-4" />
          <span>Time remaining: {{ formatTimeRemaining(timeRemaining) }}</span>
          <div class="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <div 
              class="bg-amber-500 h-2 rounded-full transition-all duration-1000"
              :style="{ width: `${timeProgress}%` }"
            ></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { HumanValidationPrompt } from '~/types';

interface Props {
  prompt: HumanValidationPrompt;
}

const props = defineProps<Props>();

// Component state
const isSubmitting = ref(false);
const feedback = ref('');
const selectedChoice = ref('');
const inputValue = ref('');
const reviewFeedback = ref('');
const confirmed = ref(false);
const timeRemaining = ref(0);

// Computed
const reviewContent = computed(() => {
  return props.prompt.metadata?.reviewContent || '';
});

const timeProgress = computed(() => {
  if (!props.prompt.timeout) return 100;
  return (timeRemaining.value / props.prompt.timeout) * 100;
});

// Emits
const $emit = defineEmits<{
  approve: [response: any];
  reject: [response: any];
}>();

// Methods
const handleApprove = async () => {
  await submitValidation({ feedback: feedback.value }, true);
};

const handleReject = async () => {
  await submitValidation({ feedback: feedback.value }, false);
};

const handleChoiceSubmit = async () => {
  if (selectedChoice.value) {
    await submitValidation({ choice: selectedChoice.value }, true);
  }
};

const handleInputSubmit = async () => {
  if (!props.prompt.required || inputValue.value.trim()) {
    await submitValidation({ input: inputValue.value }, true);
  }
};

const handleReviewApprove = async () => {
  await submitValidation({ 
    feedback: reviewFeedback.value,
    decision: 'approved' 
  }, true);
};

const handleReviewRequestChanges = async () => {
  await submitValidation({ 
    feedback: reviewFeedback.value,
    decision: 'changes_requested' 
  }, false);
};

const handleReviewReject = async () => {
  await submitValidation({ 
    feedback: reviewFeedback.value,
    decision: 'rejected' 
  }, false);
};

const handleConfirm = async () => {
  await submitValidation({ confirmed: true }, true);
};

const handleCancel = async () => {
  await submitValidation({ confirmed: false }, false);
};

const submitValidation = async (response: any, approved: boolean) => {
  if (isSubmitting.value) return;
  
  isSubmitting.value = true;
  
  try {
    $emit(approved ? 'approve' : 'reject', response);
  } catch (error) {
    console.error('Failed to submit validation:', error);
  } finally {
    isSubmitting.value = false;
  }
};

const formatReviewContent = (content: string): string => {
  // Basic markdown-like formatting
  return content
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`(.*?)`/g, '<code class="bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded text-sm">$1</code>')
    .replace(/\n/g, '<br>');
};

const formatTimeRemaining = (ms: number): string => {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  
  if (minutes > 0) {
    return `${minutes}m ${remainingSeconds}s`;
  }
  return `${remainingSeconds}s`;
};

// Timeout management
let timeoutInterval: NodeJS.Timeout | null = null;

const startTimeout = () => {
  if (!props.prompt.timeout) return;
  
  timeRemaining.value = props.prompt.timeout;
  
  timeoutInterval = setInterval(() => {
    timeRemaining.value -= 1000;
    
    if (timeRemaining.value <= 0) {
      clearTimeout();
      // Auto-reject on timeout
      submitValidation({ reason: 'timeout' }, false);
    }
  }, 1000);
};

const clearTimeout = () => {
  if (timeoutInterval) {
    clearInterval(timeoutInterval);
    timeoutInterval = null;
  }
};

// Lifecycle
onMounted(() => {
  startTimeout();
});

onUnmounted(() => {
  clearTimeout();
});

// Watch for prompt changes
watch(() => props.prompt, () => {
  clearTimeout();
  startTimeout();
  
  // Reset component state
  feedback.value = '';
  selectedChoice.value = '';
  inputValue.value = '';
  reviewFeedback.value = '';
  confirmed.value = false;
}, { deep: true });
</script>

<style scoped>
.validation-prompt {
  animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

:deep(code) {
  @apply font-mono text-sm;
}
</style>