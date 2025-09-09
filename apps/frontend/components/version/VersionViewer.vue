<template>
  <div
    class="fixed inset-0 z-50 overflow-y-auto"
    @click.self="$emit('close')"
  >
    <div class="flex min-h-screen items-end justify-center px-4 pt-4 pb-20 text-center sm:block sm:p-0">
      <!-- Background overlay -->
      <div class="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity"></div>

      <!-- Modal panel -->
      <div class="relative inline-block w-full max-w-4xl transform overflow-hidden rounded-lg bg-white dark:bg-gray-800 px-4 pt-5 pb-4 text-left align-bottom shadow-xl transition-all sm:my-8 sm:align-middle sm:p-6">
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

        <div class="sm:flex sm:items-start">
          <div class="w-full">
            <!-- Header -->
            <div class="mb-6 border-b border-gray-200 dark:border-gray-700 pb-4">
              <h3 class="text-lg font-medium leading-6 text-gray-900 dark:text-white">
                Version {{ version.version_number }}
              </h3>
              <div class="mt-2 flex items-center space-x-4 text-sm text-gray-500 dark:text-gray-400">
                <div>
                  <span class="font-medium">Created by:</span> {{ version.created_by }}
                </div>
                <div>
                  <span class="font-medium">Date:</span> {{ formatDateTime(version.created_at) }}
                </div>
                <div v-if="version.validation_score">
                  <span class="font-medium">Quality Score:</span>
                  <span class="ml-1 text-green-600 dark:text-green-400">
                    {{ (version.validation_score * 10).toFixed(1) }}/10
                  </span>
                </div>
              </div>
              <div v-if="version.comment" class="mt-2">
                <span class="font-medium text-gray-700 dark:text-gray-300">Comment:</span>
                <p class="mt-1 text-gray-600 dark:text-gray-400">{{ version.comment }}</p>
              </div>
            </div>

            <!-- Content Tabs -->
            <div class="mb-4">
              <nav class="flex space-x-8" aria-label="Tabs">
                <button
                  v-for="tab in tabs"
                  :key="tab.id"
                  @click="activeTab = tab.id"
                  :class="[
                    activeTab === tab.id
                      ? 'border-indigo-500 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
                      : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:border-gray-300 dark:hover:border-gray-600',
                    'whitespace-nowrap py-2 px-1 border-b-2 font-medium text-sm'
                  ]"
                >
                  {{ tab.name }}
                </button>
              </nav>
            </div>

            <!-- Tab Content -->
            <div class="max-h-96 overflow-y-auto">
              <!-- Content Tab -->
              <div v-if="activeTab === 'content'" class="space-y-4">
                <div class="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                  <pre class="text-sm text-gray-800 dark:text-gray-200 whitespace-pre-wrap">{{
                    formatContent(version.content)
                  }}</pre>
                </div>
              </div>

              <!-- Metadata Tab -->
              <div v-if="activeTab === 'metadata'" class="space-y-4">
                <div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <div class="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                    <h4 class="text-sm font-medium text-gray-900 dark:text-white mb-2">
                      Version Info
                    </h4>
                    <dl class="space-y-2 text-sm">
                      <div class="flex justify-between">
                        <dt class="text-gray-600 dark:text-gray-400">ID:</dt>
                        <dd class="text-gray-900 dark:text-white font-mono text-xs">{{ version.id }}</dd>
                      </div>
                      <div class="flex justify-between">
                        <dt class="text-gray-600 dark:text-gray-400">Version:</dt>
                        <dd class="text-gray-900 dark:text-white">{{ version.version_number }}</dd>
                      </div>
                      <div class="flex justify-between">
                        <dt class="text-gray-600 dark:text-gray-400">Document Type:</dt>
                        <dd class="text-gray-900 dark:text-white">{{ version.document_type }}</dd>
                      </div>
                      <div class="flex justify-between">
                        <dt class="text-gray-600 dark:text-gray-400">Validated:</dt>
                        <dd>
                          <span
                            :class="[
                              version.is_validated
                                ? 'text-green-600 dark:text-green-400'
                                : 'text-red-600 dark:text-red-400'
                            ]"
                          >
                            {{ version.is_validated ? 'Yes' : 'No' }}
                          </span>
                        </dd>
                      </div>
                    </dl>
                  </div>

                  <div class="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                    <h4 class="text-sm font-medium text-gray-900 dark:text-white mb-2">
                      Content Stats
                    </h4>
                    <dl class="space-y-2 text-sm">
                      <div class="flex justify-between">
                        <dt class="text-gray-600 dark:text-gray-400">Content Size:</dt>
                        <dd class="text-gray-900 dark:text-white">
                          {{ formatBytes(version.metadata?.size_bytes || 0) }}
                        </dd>
                      </div>
                      <div class="flex justify-between">
                        <dt class="text-gray-600 dark:text-gray-400">Fields Count:</dt>
                        <dd class="text-gray-900 dark:text-white">
                          {{ Object.keys(version.content).length }}
                        </dd>
                      </div>
                      <div v-if="version.metadata?.content_hash" class="flex justify-between">
                        <dt class="text-gray-600 dark:text-gray-400">Content Hash:</dt>
                        <dd class="text-gray-900 dark:text-white font-mono text-xs">
                          {{ version.metadata.content_hash.substring(0, 16) }}...
                        </dd>
                      </div>
                    </dl>
                  </div>
                </div>
              </div>

              <!-- Changes Tab -->
              <div v-if="activeTab === 'changes'" class="space-y-4">
                <div v-if="version.changes_summary" class="grid grid-cols-1 gap-4 sm:grid-cols-3">
                  <!-- Fields Added -->
                  <div v-if="version.changes_summary.fields_added?.length" class="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                    <h4 class="text-sm font-medium text-green-800 dark:text-green-200 mb-2">
                      Added Fields ({{ version.changes_summary.fields_added.length }})
                    </h4>
                    <ul class="space-y-1 text-sm text-green-700 dark:text-green-300">
                      <li
                        v-for="field in version.changes_summary.fields_added"
                        :key="field"
                        class="font-mono text-xs bg-green-100 dark:bg-green-800/30 px-2 py-1 rounded"
                      >
                        {{ field }}
                      </li>
                    </ul>
                  </div>

                  <!-- Fields Modified -->
                  <div v-if="version.changes_summary.fields_modified?.length" class="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
                    <h4 class="text-sm font-medium text-yellow-800 dark:text-yellow-200 mb-2">
                      Modified Fields ({{ version.changes_summary.fields_modified.length }})
                    </h4>
                    <ul class="space-y-1 text-sm text-yellow-700 dark:text-yellow-300">
                      <li
                        v-for="field in version.changes_summary.fields_modified"
                        :key="field"
                        class="font-mono text-xs bg-yellow-100 dark:bg-yellow-800/30 px-2 py-1 rounded"
                      >
                        {{ field }}
                      </li>
                    </ul>
                  </div>

                  <!-- Fields Removed -->
                  <div v-if="version.changes_summary.fields_removed?.length" class="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
                    <h4 class="text-sm font-medium text-red-800 dark:text-red-200 mb-2">
                      Removed Fields ({{ version.changes_summary.fields_removed.length }})
                    </h4>
                    <ul class="space-y-1 text-sm text-red-700 dark:text-red-300">
                      <li
                        v-for="field in version.changes_summary.fields_removed"
                        :key="field"
                        class="font-mono text-xs bg-red-100 dark:bg-red-800/30 px-2 py-1 rounded"
                      >
                        {{ field }}
                      </li>
                    </ul>
                  </div>
                </div>

                <div v-if="!version.changes_summary || version.changes_summary.total_changes === 0" class="text-center py-8">
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
                    No changes tracked
                  </h3>
                  <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
                    This version doesn't have detailed change information.
                  </p>
                </div>
              </div>
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
                @click="downloadVersion"
                type="button"
                class="inline-flex justify-center rounded-md border border-transparent bg-indigo-600 py-2 px-4 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
              >
                Download JSON
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';

interface DocumentVersion {
  id: string;
  document_id: string;
  document_type: string;
  version_number: number;
  title: string;
  content: Record<string, any>;
  metadata?: Record<string, any>;
  created_by: string;
  created_at: string;
  comment?: string;
  changes_summary?: {
    total_changes: number;
    fields_added: string[];
    fields_removed: string[];
    fields_modified: string[];
  };
  is_validated: boolean;
  validation_score?: number;
}

interface Props {
  version: DocumentVersion;
  documentId: string;
}

const props = defineProps<Props>();

// State
const activeTab = ref('content');

const tabs = [
  { id: 'content', name: 'Content' },
  { id: 'metadata', name: 'Metadata' },
  { id: 'changes', name: 'Changes' },
];

// Methods
const formatDateTime = (dateString: string) => {
  const date = new Date(dateString);
  return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });
};

const formatContent = (content: Record<string, any>) => {
  return JSON.stringify(content, null, 2);
};

const formatBytes = (bytes: number) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

const downloadVersion = () => {
  const dataStr = JSON.stringify(props.version, null, 2);
  const dataBlob = new Blob([dataStr], { type: 'application/json' });
  const url = URL.createObjectURL(dataBlob);
  
  const link = document.createElement('a');
  link.href = url;
  link.download = `${props.version.document_type}_v${props.version.version_number}_${props.version.id}.json`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  
  URL.revokeObjectURL(url);
};

// Emits
defineEmits<{
  close: [];
}>();
</script>