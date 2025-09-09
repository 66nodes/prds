<template>
  <div class="bg-white dark:bg-gray-800 shadow rounded-lg">
    <div class="px-4 py-5 sm:p-6">
      <div class="flex items-center justify-between mb-6">
        <div>
          <h3 class="text-lg leading-6 font-medium text-gray-900 dark:text-white">
            Version History
          </h3>
          <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Track all changes and revisions for this document
          </p>
        </div>
        <div class="flex items-center space-x-2">
          <button
            @click="refreshVersions"
            :disabled="loading"
            class="inline-flex items-center px-3 py-2 border border-gray-300 dark:border-gray-600 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
          >
            <svg
              class="h-4 w-4 mr-2"
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

      <!-- Version List -->
      <div class="flow-root">
        <ul role="list" class="-mb-8">
          <li
            v-for="(version, versionIdx) in versions"
            :key="version.id"
          >
            <div class="relative pb-8">
              <span
                v-if="versionIdx !== versions.length - 1"
                class="absolute top-4 left-4 -ml-px h-full w-0.5 bg-gray-200 dark:bg-gray-600"
                aria-hidden="true"
              />
              <div class="relative flex space-x-3">
                <div>
                  <span
                    :class="[
                      version.is_validated ? 'bg-green-500' : 'bg-gray-400',
                      'h-8 w-8 rounded-full flex items-center justify-center ring-8 ring-white dark:ring-gray-800'
                    ]"
                  >
                    <svg
                      v-if="version.is_validated"
                      class="h-5 w-5 text-white"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fill-rule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clip-rule="evenodd"
                      />
                    </svg>
                    <svg
                      v-else
                      class="h-5 w-5 text-white"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        d="M10 2a8 8 0 100 16 8 8 0 000-16zM9 9a1 1 0 012 0v4a1 1 0 11-2 0V9zm1-3a1 1 0 100 2 1 1 0 000-2z"
                      />
                    </svg>
                  </span>
                </div>
                <div class="flex-1 min-w-0">
                  <div>
                    <div class="text-sm">
                      <span class="font-medium text-gray-900 dark:text-white">
                        Version {{ version.version_number }}
                      </span>
                      <span class="ml-2 text-gray-500 dark:text-gray-400">
                        by {{ version.created_by }}
                      </span>
                    </div>
                    <p class="mt-0.5 text-xs text-gray-500 dark:text-gray-400">
                      {{ formatDateTime(version.created_at) }}
                    </p>
                  </div>
                  <div class="mt-2 text-sm text-gray-700 dark:text-gray-300">
                    <p class="font-medium">{{ version.title }}</p>
                    <p v-if="version.comment" class="mt-1 text-gray-600 dark:text-gray-400">
                      {{ version.comment }}
                    </p>
                    <div class="mt-2 flex items-center space-x-4">
                      <div class="flex items-center text-xs text-gray-500 dark:text-gray-400">
                        <span>
                          {{ version.changes_summary?.total_changes || 0 }} changes
                        </span>
                        <span v-if="version.validation_score" class="ml-2">
                          • {{ (version.validation_score * 10).toFixed(1) }}/10 quality
                        </span>
                      </div>
                    </div>
                  </div>
                  <div class="mt-2 flex items-center space-x-2">
                    <button
                      @click="viewVersion(version)"
                      class="text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-200 text-sm font-medium"
                    >
                      View
                    </button>
                    <button
                      v-if="versionIdx > 0"
                      @click="compareVersion(version)"
                      class="text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-200 text-sm font-medium"
                    >
                      Compare
                    </button>
                    <button
                      v-if="versionIdx > 0"
                      @click="restoreVersion(version)"
                      class="text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-200 text-sm font-medium"
                    >
                      Restore
                    </button>
                    <button
                      v-if="versionIdx === 0"
                      class="text-green-600 dark:text-green-400 text-sm font-medium cursor-default"
                    >
                      Current
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </li>
        </ul>
      </div>

      <!-- Loading State -->
      <div v-if="loading" class="flex justify-center py-8">
        <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
      </div>

      <!-- Empty State -->
      <div v-if="!loading && versions.length === 0" class="text-center py-8">
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
            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
          />
        </svg>
        <h3 class="mt-2 text-sm font-medium text-gray-900 dark:text-white">
          No versions found
        </h3>
        <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
          This document doesn't have any versions yet.
        </p>
      </div>

      <!-- Pagination -->
      <div
        v-if="totalCount > pageSize && !loading"
        class="mt-6 flex items-center justify-between border-t border-gray-200 dark:border-gray-700 pt-4"
      >
        <div class="flex-1 flex justify-between">
          <button
            @click="previousPage"
            :disabled="currentPage === 1"
            class="relative inline-flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 text-sm font-medium rounded-md text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Previous
          </button>
          <span class="text-sm text-gray-700 dark:text-gray-300">
            Page {{ currentPage }} of {{ totalPages }}
          </span>
          <button
            @click="nextPage"
            :disabled="currentPage === totalPages"
            class="ml-3 relative inline-flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 text-sm font-medium rounded-md text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  </div>

  <!-- Version Viewer Modal -->
  <VersionViewer
    v-if="selectedVersion"
    :version="selectedVersion"
    :document-id="documentId"
    @close="selectedVersion = null"
  />

  <!-- Version Comparison Modal -->
  <VersionComparison
    v-if="comparisonVersion"
    :from-version="comparisonVersion"
    :to-version="versions[0]"
    :document-id="documentId"
    @close="comparisonVersion = null"
  />
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue';
import { useApiClient } from '~/composables/useApiClient';
import VersionViewer from './VersionViewer.vue';
import VersionComparison from './VersionComparison.vue';

interface DocumentVersion {
  id: string;
  document_id: string;
  version_number: number;
  title: string;
  content: Record<string, any>;
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
  documentId: string;
}

const props = defineProps<Props>();

// State
const versions = ref<DocumentVersion[]>([]);
const loading = ref(false);
const currentPage = ref(1);
const pageSize = ref(20);
const totalCount = ref(0);
const selectedVersion = ref<DocumentVersion | null>(null);
const comparisonVersion = ref<DocumentVersion | null>(null);

// Computed
const totalPages = computed(() => Math.ceil(totalCount.value / pageSize.value));

// Composables
const { $api } = useApiClient();

// Methods
const loadVersions = async () => {
  loading.value = true;
  try {
    const response = await $api.get(`/api/v1/versions/document/${props.documentId}`, {
      params: {
        page: currentPage.value,
        page_size: pageSize.value,
      },
    });

    versions.value = response.versions || [];
    totalCount.value = response.total_count || 0;
  } catch (error) {
    console.error('Failed to load versions:', error);
    // Show error notification
  } finally {
    loading.value = false;
  }
};

const refreshVersions = () => {
  currentPage.value = 1;
  loadVersions();
};

const previousPage = () => {
  if (currentPage.value > 1) {
    currentPage.value--;
    loadVersions();
  }
};

const nextPage = () => {
  if (currentPage.value < totalPages.value) {
    currentPage.value++;
    loadVersions();
  }
};

const viewVersion = (version: DocumentVersion) => {
  selectedVersion.value = version;
};

const compareVersion = (version: DocumentVersion) => {
  comparisonVersion.value = version;
};

const restoreVersion = async (version: DocumentVersion) => {
  if (confirm(`Are you sure you want to restore to version ${version.version_number}? This will create a new version with the restored content.`)) {
    try {
      await $api.post('/api/v1/versions/restore', {
        document_id: props.documentId,
        version_id: version.id,
        comment: `Restored from version ${version.version_number}`,
      });
      
      // Refresh the version list
      refreshVersions();
      
      // Emit event to parent component
      emit('version-restored', version);
      
      // Show success notification
    } catch (error) {
      console.error('Failed to restore version:', error);
      // Show error notification
    }
  }
};

const formatDateTime = (dateString: string) => {
  const date = new Date(dateString);
  return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { 
    hour: '2-digit', 
    minute: '2-digit' 
  });
};

// Emits
const emit = defineEmits<{
  'version-restored': [version: DocumentVersion];
  'version-selected': [version: DocumentVersion];
}>();

// Watchers
watch(() => props.documentId, (newId) => {
  if (newId) {
    refreshVersions();
  }
});

// Lifecycle
onMounted(() => {
  if (props.documentId) {
    loadVersions();
  }
});
</script>