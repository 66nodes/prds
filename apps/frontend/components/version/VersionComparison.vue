<template>
  <div
    class="fixed inset-0 z-50 overflow-y-auto"
    @click.self="$emit('close')"
  >
    <div class="flex min-h-screen items-end justify-center px-4 pt-4 pb-20 text-center sm:block sm:p-0">
      <!-- Background overlay -->
      <div class="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity"></div>

      <!-- Modal panel -->
      <div class="relative inline-block w-full max-w-6xl transform overflow-hidden rounded-lg bg-white dark:bg-gray-800 px-4 pt-5 pb-4 text-left align-bottom shadow-xl transition-all sm:my-8 sm:align-middle sm:p-6">
        <div class="absolute top-0 right-0 hidden pt-4 pr-4 sm:block">
          <button
            @click="$emit('close')"
            type="button"
            class="rounded-md bg-white dark:bg-gray-800 text-gray-400 hover:text-gray-500 dark:hover:text-gray-300 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
          >
            <span class="sr-only">Close</span>
            <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div class="w-full">
          <!-- Header -->
          <div class="mb-6 border-b border-gray-200 dark:border-gray-700 pb-4">
            <h3 class="text-lg font-medium leading-6 text-gray-900 dark:text-white">
              Version Comparison
            </h3>
            <div class="mt-2 flex items-center justify-between">
              <div class="text-sm text-gray-500 dark:text-gray-400">
                Comparing version {{ fromVersion.version_number }} with version {{ toVersion.version_number }}
              </div>
              <div class="flex items-center space-x-2">
                <button
                  @click="swapVersions"
                  class="inline-flex items-center px-2 py-1 border border-gray-300 dark:border-gray-600 shadow-sm text-xs leading-4 font-medium rounded text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                >
                  <svg class="h-3 w-3 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                  </svg>
                  Swap
                </button>
                <button
                  @click="loadComparison"
                  :disabled="loading"
                  class="inline-flex items-center px-2 py-1 border border-gray-300 dark:border-gray-600 shadow-sm text-xs leading-4 font-medium rounded text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
                >
                  <svg
                    class="h-3 w-3 mr-1"
                    :class="{ 'animate-spin': loading }"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-width="2"
                      d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                    />
                  </svg>
                  Refresh
                </button>
              </div>
            </div>
          </div>

          <!-- Loading State -->
          <div v-if="loading" class="flex justify-center py-8">
            <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
          </div>

          <!-- Comparison Content -->
          <div v-else-if="diff" class="space-y-6">
            <!-- Summary Stats -->
            <div class="grid grid-cols-1 gap-4 sm:grid-cols-4">
              <div class="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 text-center">
                <div class="text-2xl font-bold text-gray-900 dark:text-white">
                  {{ diff.total_changes }}
                </div>
                <div class="text-sm text-gray-500 dark:text-gray-400">Total Changes</div>
              </div>
              <div class="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 text-center">
                <div class="text-2xl font-bold text-green-600 dark:text-green-400">
                  {{ diff.lines_added }}
                </div>
                <div class="text-sm text-green-600 dark:text-green-400">Lines Added</div>
              </div>
              <div class="bg-red-50 dark:bg-red-900/20 rounded-lg p-4 text-center">
                <div class="text-2xl font-bold text-red-600 dark:text-red-400">
                  {{ diff.lines_deleted }}
                </div>
                <div class="text-sm text-red-600 dark:text-red-400">Lines Deleted</div>
              </div>
              <div class="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4 text-center">
                <div class="text-2xl font-bold text-yellow-600 dark:text-yellow-400">
                  {{ diff.modifications.length }}
                </div>
                <div class="text-sm text-yellow-600 dark:text-yellow-400">Modifications</div>
              </div>
            </div>

            <!-- Change Details -->
            <div class="space-y-6">
              <!-- Additions -->
              <div v-if="diff.additions.length > 0" class="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                <h4 class="text-sm font-medium text-green-800 dark:text-green-200 mb-3">
                  Additions ({{ diff.additions.length }})
                </h4>
                <div class="space-y-2">
                  <div
                    v-for="(addition, index) in diff.additions"
                    :key="index"
                    class="bg-green-100 dark:bg-green-800/30 rounded p-3"
                  >
                    <div v-if="addition.field" class="text-xs font-mono text-green-700 dark:text-green-300 mb-1">
                      + {{ addition.field }}
                    </div>
                    <pre
                      class="text-sm text-green-800 dark:text-green-200 whitespace-pre-wrap break-words"
                    >{{ formatValue(addition.value || addition.line) }}</pre>
                  </div>
                </div>
              </div>

              <!-- Deletions -->
              <div v-if="diff.deletions.length > 0" class="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
                <h4 class="text-sm font-medium text-red-800 dark:text-red-200 mb-3">
                  Deletions ({{ diff.deletions.length }})
                </h4>
                <div class="space-y-2">
                  <div
                    v-for="(deletion, index) in diff.deletions"
                    :key="index"
                    class="bg-red-100 dark:bg-red-800/30 rounded p-3"
                  >
                    <div v-if="deletion.field" class="text-xs font-mono text-red-700 dark:text-red-300 mb-1">
                      - {{ deletion.field }}
                    </div>
                    <pre
                      class="text-sm text-red-800 dark:text-red-200 whitespace-pre-wrap break-words"
                    >{{ formatValue(deletion.value || deletion.line) }}</pre>
                  </div>
                </div>
              </div>

              <!-- Modifications -->
              <div v-if="diff.modifications.length > 0" class="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
                <h4 class="text-sm font-medium text-yellow-800 dark:text-yellow-200 mb-3">
                  Modifications ({{ diff.modifications.length }})
                </h4>
                <div class="space-y-4">
                  <div
                    v-for="(modification, index) in diff.modifications"
                    :key="index"
                    class="bg-yellow-100 dark:bg-yellow-800/30 rounded p-3"
                  >
                    <div v-if="modification.field" class="text-xs font-mono text-yellow-700 dark:text-yellow-300 mb-2">
                      ~ {{ modification.field }}
                    </div>
                    
                    <!-- Before (Old Value) -->
                    <div class="mb-2">
                      <div class="text-xs text-red-600 dark:text-red-400 mb-1">- Before:</div>
                      <pre class="text-sm text-red-700 dark:text-red-300 bg-red-50 dark:bg-red-900/30 p-2 rounded whitespace-pre-wrap break-words">{{ formatValue(modification.old_value) }}</pre>
                    </div>

                    <!-- After (New Value) -->
                    <div>
                      <div class="text-xs text-green-600 dark:text-green-400 mb-1">+ After:</div>
                      <pre class="text-sm text-green-700 dark:text-green-300 bg-green-50 dark:bg-green-900/30 p-2 rounded whitespace-pre-wrap break-words">{{ formatValue(modification.new_value) }}</pre>
                    </div>
                  </div>
                </div>
              </div>

              <!-- No Changes -->
              <div v-if="diff.total_changes === 0" class="text-center py-8">
                <svg
                  class="mx-auto h-12 w-12 text-gray-400"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z"
                  />
                </svg>
                <h3 class="mt-2 text-sm font-medium text-gray-900 dark:text-white">
                  No differences found
                </h3>
                <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
                  These versions have identical content.
                </p>
              </div>
            </div>
          </div>

          <!-- Error State -->
          <div v-else-if="error" class="text-center py-8">
            <svg
              class="mx-auto h-12 w-12 text-red-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.464 0L4.35 16.5c-.77.833.192 2.5 1.732 2.5z"
              />
            </svg>
            <h3 class="mt-2 text-sm font-medium text-gray-900 dark:text-white">
              Failed to load comparison
            </h3>
            <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
              {{ error }}
            </p>
          </div>

          <!-- Actions -->
          <div class="mt-6 flex justify-end space-x-3">
            <button
              @click="$emit('close')"
              type="button"
              class="rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 py-2 px-4 text-sm font-medium text-gray-700 dark:text-gray-200 shadow-sm hover:bg-gray-50 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
            >
              Close
            </button>
            <button
              v-if="diff"
              @click="downloadDiff"
              type="button"
              class="inline-flex justify-center rounded-md border border-transparent bg-indigo-600 py-2 px-4 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
            >
              Download Diff
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { useApiClient } from '~/composables/useApiClient';

interface DocumentVersion {
  id: string;
  document_id: string;
  version_number: number;
  title: string;
  content: Record<string, any>;
  created_by: string;
  created_at: string;
}

interface DocumentDiff {
  from_version_id: string;
  to_version_id: string;
  from_version_number: number;
  to_version_number: number;
  additions: Array<{ field?: string; value?: any; line?: string }>;
  deletions: Array<{ field?: string; value?: any; line?: string }>;
  modifications: Array<{ field?: string; old_value: any; new_value: any }>;
  total_changes: number;
  lines_added: number;
  lines_deleted: number;
  generated_at: string;
  generated_by: string;
}

interface Props {
  fromVersion: DocumentVersion;
  toVersion: DocumentVersion;
  documentId: string;
}

const props = defineProps<Props>();

// State
const diff = ref<DocumentDiff | null>(null);
const loading = ref(false);
const error = ref<string | null>(null);

// Composables
const { $api } = useApiClient();

// Methods
const loadComparison = async () => {
  loading.value = true;
  error.value = null;

  try {
    const response = await $api.post('/api/v1/versions/compare', {
      document_id: props.documentId,
      from_version_id: props.fromVersion.id,
      to_version_id: props.toVersion.id,
      include_metadata: false,
    });

    diff.value = response;
  } catch (err) {
    console.error('Failed to load comparison:', err);
    error.value = 'Failed to load version comparison';
  } finally {
    loading.value = false;
  }
};

const swapVersions = () => {
  // Emit event to parent to swap the versions
  emit('swap-versions');
};

const formatValue = (value: any): string => {
  if (typeof value === 'string') {
    return value;
  }
  return JSON.stringify(value, null, 2);
};

const downloadDiff = () => {
  if (!diff.value) return;

  const dataStr = JSON.stringify(diff.value, null, 2);
  const dataBlob = new Blob([dataStr], { type: 'application/json' });
  const url = URL.createObjectURL(dataBlob);
  
  const link = document.createElement('a');
  link.href = url;
  link.download = `diff_v${props.fromVersion.version_number}_to_v${props.toVersion.version_number}.json`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  
  URL.revokeObjectURL(url);
};

// Emits
const emit = defineEmits<{
  close: [];
  'swap-versions': [];
}>();

// Lifecycle
onMounted(() => {
  loadComparison();
});
</script>