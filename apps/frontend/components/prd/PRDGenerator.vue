<template>
  <div class="max-w-4xl mx-auto p-6">
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow-lg">
      <!-- Header -->
      <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <h2 class="text-2xl font-bold text-gray-900 dark:text-white">
          Generate Product Requirements Document
        </h2>
        <p class="mt-1 text-sm text-gray-600 dark:text-gray-400">
          AI-powered PRD generation with real-time validation
        </p>
      </div>

      <!-- Form -->
      <form @submit.prevent="handleSubmit" class="p-6 space-y-6">
        <!-- Title -->
        <div>
          <label
            for="title"
            class="block text-sm font-medium text-gray-700 dark:text-gray-300"
          >
            PRD Title *
          </label>
          <input
            id="title"
            v-model="formData.title"
            type="text"
            required
            class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white sm:text-sm"
            placeholder="Enter PRD title"
            :disabled="isGenerating"
          />
        </div>

        <!-- Description -->
        <div>
          <label
            for="description"
            class="block text-sm font-medium text-gray-700 dark:text-gray-300"
          >
            Description *
          </label>
          <textarea
            id="description"
            v-model="formData.description"
            rows="4"
            required
            class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white sm:text-sm"
            placeholder="Describe the product or feature"
            :disabled="isGenerating"
          />
        </div>

        <!-- Target Audience -->
        <div>
          <label
            for="targetAudience"
            class="block text-sm font-medium text-gray-700 dark:text-gray-300"
          >
            Target Audience
          </label>
          <input
            id="targetAudience"
            v-model="formData.targetAudience"
            type="text"
            class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white sm:text-sm"
            placeholder="Who is this for?"
            :disabled="isGenerating"
          />
        </div>

        <!-- Requirements -->
        <div>
          <label
            class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
          >
            Requirements
          </label>
          <div class="space-y-2">
            <div
              v-for="(req, index) in formData.requirements"
              :key="`req-${index}`"
              class="flex items-center space-x-2"
            >
              <input
                v-model="formData.requirements[index]"
                type="text"
                class="flex-1 rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white sm:text-sm"
                placeholder="Enter requirement"
                :disabled="isGenerating"
              />
              <button
                type="button"
                @click="removeRequirement(index)"
                class="text-red-600 hover:text-red-800 dark:text-red-400 dark:hover:text-red-300"
                :disabled="isGenerating"
              >
                <svg
                  class="h-5 w-5"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>
          </div>
          <button
            type="button"
            @click="addRequirement"
            class="mt-2 text-sm text-indigo-600 hover:text-indigo-500 dark:text-indigo-400 dark:hover:text-indigo-300"
            :disabled="isGenerating"
          >
            + Add Requirement
          </button>
        </div>

        <!-- Constraints -->
        <div>
          <label
            class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
          >
            Constraints
          </label>
          <div class="space-y-2">
            <div
              v-for="(constraint, index) in formData.constraints"
              :key="`constraint-${index}`"
              class="flex items-center space-x-2"
            >
              <input
                v-model="formData.constraints[index]"
                type="text"
                class="flex-1 rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white sm:text-sm"
                placeholder="Enter constraint"
                :disabled="isGenerating"
              />
              <button
                type="button"
                @click="removeConstraint(index)"
                class="text-red-600 hover:text-red-800 dark:text-red-400 dark:hover:text-red-300"
                :disabled="isGenerating"
              >
                <svg
                  class="h-5 w-5"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>
          </div>
          <button
            type="button"
            @click="addConstraint"
            class="mt-2 text-sm text-indigo-600 hover:text-indigo-500 dark:text-indigo-400 dark:hover:text-indigo-300"
            :disabled="isGenerating"
          >
            + Add Constraint
          </button>
        </div>

        <!-- Success Metrics -->
        <div>
          <label
            class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
          >
            Success Metrics
          </label>
          <div class="space-y-2">
            <div
              v-for="(metric, index) in formData.successMetrics"
              :key="`metric-${index}`"
              class="flex items-center space-x-2"
            >
              <input
                v-model="formData.successMetrics[index]"
                type="text"
                class="flex-1 rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white sm:text-sm"
                placeholder="Enter success metric"
                :disabled="isGenerating"
              />
              <button
                type="button"
                @click="removeSuccessMetric(index)"
                class="text-red-600 hover:text-red-800 dark:text-red-400 dark:hover:text-red-300"
                :disabled="isGenerating"
              >
                <svg
                  class="h-5 w-5"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>
          </div>
          <button
            type="button"
            @click="addSuccessMetric"
            class="mt-2 text-sm text-indigo-600 hover:text-indigo-500 dark:text-indigo-400 dark:hover:text-indigo-300"
            :disabled="isGenerating"
          >
            + Add Success Metric
          </button>
        </div>

        <!-- Generation Progress -->
        <div v-if="isGenerating" class="space-y-2">
          <div
            class="flex items-center justify-between text-sm text-gray-600 dark:text-gray-400"
          >
            <span>Generating PRD...</span>
            <span>{{ generationProgress }}%</span>
          </div>
          <div class="w-full bg-gray-200 rounded-full h-2 dark:bg-gray-700">
            <div
              class="bg-indigo-600 h-2 rounded-full transition-all duration-300"
              :style="{ width: `${generationProgress}%` }"
            />
          </div>
        </div>

        <!-- Validation Result -->
        <div
          v-if="validationResult"
          class="p-4 rounded-lg"
          :class="validationResultClass"
        >
          <div class="flex">
            <div class="flex-shrink-0">
              <svg
                v-if="validationResult.hallucinationRate < 0.02"
                class="h-5 w-5 text-green-400"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fill-rule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                  clip-rule="evenodd"
                />
              </svg>
              <svg
                v-else
                class="h-5 w-5 text-yellow-400"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fill-rule="evenodd"
                  d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                  clip-rule="evenodd"
                />
              </svg>
            </div>
            <div class="ml-3">
              <h3 class="text-sm font-medium">
                Validation
                {{
                  validationResult.hallucinationRate < 0.02
                    ? 'Passed'
                    : 'Warning'
                }}
              </h3>
              <div class="mt-2 text-sm">
                <p>
                  Hallucination Rate:
                  {{ (validationResult.hallucinationRate * 100).toFixed(2) }}%
                </p>
                <p>
                  Validation Score:
                  {{ (validationResult.validationScore * 100).toFixed(1) }}%
                </p>
              </div>
            </div>
          </div>
        </div>

        <!-- Error Message -->
        <div v-if="error" class="p-4 rounded-lg bg-red-50 dark:bg-red-900/20">
          <div class="flex">
            <div class="flex-shrink-0">
              <svg
                class="h-5 w-5 text-red-400"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fill-rule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                  clip-rule="evenodd"
                />
              </svg>
            </div>
            <div class="ml-3">
              <h3 class="text-sm font-medium text-red-800 dark:text-red-200">
                Error
              </h3>
              <div class="mt-2 text-sm text-red-700 dark:text-red-300">
                {{ error }}
              </div>
            </div>
          </div>
        </div>

        <!-- Submit Button -->
        <div class="flex justify-end space-x-3">
          <button
            type="button"
            @click="handleReset"
            class="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 dark:bg-gray-700 dark:text-gray-300 dark:border-gray-600 dark:hover:bg-gray-600"
            :disabled="isGenerating"
          >
            Reset
          </button>
          <button
            type="submit"
            class="px-4 py-2 text-sm font-medium text-white bg-indigo-600 border border-transparent rounded-md shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
            :disabled="isGenerating || !isFormValid"
          >
            <span v-if="!isGenerating">Generate PRD</span>
            <span v-else>Generating...</span>
          </button>
        </div>
      </form>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import { useDebounceFn } from '@vueuse/core';
import { usePRDStore } from '~/stores/prd';
import { useFormValidation, validators } from '~/composables/useFormValidation';
import ErrorBoundary from '~/components/ErrorBoundary.vue';
import LoadingState from '~/components/LoadingState.vue';
import type { PRDRequest, ValidationResult } from '~/types';

const props = defineProps<{
  projectId: string;
}>();

const emit = defineEmits<{
  generated: [prd: any];
}>();

const prdStore = usePRDStore();
const router = useRouter();

// Form data with validation
const formData = ref<PRDRequest>({
  title: '',
  description: '',
  projectId: props.projectId,
  requirements: [''],
  constraints: [''],
  targetAudience: '',
  successMetrics: [''],
});

// Form validation setup
const {
  isValid: isFormValid,
  errors,
  validate,
  validateField,
  reset: resetValidation,
} = useFormValidation({
  title: {
    value: computed(() => formData.value.title),
    rules: [
      validators.required('Title is required'),
      validators.minLength(3, 'Title must be at least 3 characters'),
      validators.maxLength(100, 'Title must be less than 100 characters'),
    ],
  },
  description: {
    value: computed(() => formData.value.description),
    rules: [
      validators.required('Description is required'),
      validators.minLength(10, 'Description must be at least 10 characters'),
      validators.maxLength(
        1000,
        'Description must be less than 1000 characters'
      ),
    ],
  },
  requirements: {
    value: computed(() => formData.value.requirements),
    rules: [validators.arrayMinLength(1, 'At least one requirement is needed')],
  },
});

// Computed properties
const isGenerating = computed(() => prdStore.isGenerating);
const generationProgress = computed(() => prdStore.generationProgress);
const validationResult = computed(() => prdStore.validationResult);
const error = computed(() => prdStore.error);

// Debounce validation for better UX
const debouncedValidate = useDebounceFn(() => {
  validate();
}, 300);

const validationResultClass = computed(() => {
  if (!validationResult.value) return '';

  if (validationResult.value.hallucinationRate < 0.02) {
    return 'bg-green-50 dark:bg-green-900/20';
  } else {
    return 'bg-yellow-50 dark:bg-yellow-900/20';
  }
});

// Methods
const addRequirement = () => {
  formData.value.requirements?.push('');
};

const removeRequirement = (index: number) => {
  formData.value.requirements?.splice(index, 1);
};

const addConstraint = () => {
  formData.value.constraints?.push('');
};

const removeConstraint = (index: number) => {
  formData.value.constraints?.splice(index, 1);
};

const addSuccessMetric = () => {
  formData.value.successMetrics?.push('');
};

const removeSuccessMetric = (index: number) => {
  formData.value.successMetrics?.splice(index, 1);
};

const handleSubmit = async () => {
  // Validate form before submission
  if (!validate()) {
    return;
  }

  try {
    // Filter out empty values
    const requestData: PRDRequest = {
      ...formData.value,
      requirements: formData.value.requirements?.filter(r => r.trim() !== ''),
      constraints: formData.value.constraints?.filter(c => c.trim() !== ''),
      successMetrics: formData.value.successMetrics?.filter(
        m => m.trim() !== ''
      ),
    };

    const prd = await prdStore.generatePRD(props.projectId, requestData);

    emit('generated', prd);

    // Navigate to the generated PRD
    await router.push(`/projects/${props.projectId}/prds/${prd.id}`);
  } catch (error) {
    console.error('Failed to generate PRD:', error);
  }
};

const handleReset = () => {
  formData.value = {
    title: '',
    description: '',
    projectId: props.projectId,
    requirements: [''],
    constraints: [''],
    targetAudience: '',
    successMetrics: [''],
  };

  resetValidation();
  prdStore.setError(null);
  prdStore.setValidationResult(null);
};
</script>
