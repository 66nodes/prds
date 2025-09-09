/**
 * Version Control composable for managing document versions and change history
 */

import { ref, computed } from 'vue';
import { useApiClient } from './useApiClient';

export interface DocumentVersion {
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
  parent_version_id?: string;
  is_validated: boolean;
  validation_score?: number;
}

export interface VersionCreateRequest {
  document_id: string;
  document_type: string;
  content: Record<string, any>;
  comment?: string;
  user_id: string;
}

export interface VersionRestoreRequest {
  document_id: string;
  version_id: string;
  comment?: string;
  user_id: string;
}

export interface VersionComparisonRequest {
  document_id: string;
  from_version_id?: string;
  to_version_id?: string;
  include_metadata?: boolean;
}

export interface DocumentDiff {
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

export interface ChangeHistoryEntry {
  id: string;
  document_id: string;
  version_id: string;
  change_type: 'create' | 'update' | 'delete' | 'restore' | 'merge';
  field_path?: string;
  old_value?: any;
  new_value?: any;
  changed_by: string;
  changed_at: string;
  comment?: string;
  session_id?: string;
}

export interface VersionListResponse {
  document_id: string;
  versions: DocumentVersion[];
  total_count: number;
  current_version_id: string;
  page: number;
  page_size: number;
}

export interface ChangeHistoryResponse {
  document_id: string;
  changes: ChangeHistoryEntry[];
  total_count: number;
  page: number;
  page_size: number;
}

export const useVersionControl = () => {
  const { $api } = useApiClient();

  // State
  const loading = ref(false);
  const error = ref<string | null>(null);
  const versions = ref<DocumentVersion[]>([]);
  const currentVersion = ref<DocumentVersion | null>(null);
  const changeHistory = ref<ChangeHistoryEntry[]>([]);

  // Computed
  const hasVersions = computed(() => versions.value.length > 0);
  const latestVersion = computed(() => 
    versions.value.length > 0 ? versions.value[0] : null
  );

  // Methods
  const createVersion = async (request: VersionCreateRequest): Promise<DocumentVersion> => {
    loading.value = true;
    error.value = null;

    try {
      const version = await $api.post<DocumentVersion>('/api/v1/versions/create', request);
      
      // Add to local versions list if it's for the same document
      if (versions.value.length > 0 && versions.value[0].document_id === request.document_id) {
        versions.value.unshift(version);
      }

      return version;
    } catch (err) {
      error.value = 'Failed to create version';
      throw err;
    } finally {
      loading.value = false;
    }
  };

  const getVersion = async (versionId: string): Promise<DocumentVersion> => {
    loading.value = true;
    error.value = null;

    try {
      const version = await $api.get<DocumentVersion>(`/api/v1/versions/${versionId}`);
      return version;
    } catch (err) {
      error.value = 'Failed to fetch version';
      throw err;
    } finally {
      loading.value = false;
    }
  };

  const getLatestVersion = async (documentId: string): Promise<DocumentVersion> => {
    loading.value = true;
    error.value = null;

    try {
      const version = await $api.get<DocumentVersion>(`/api/v1/versions/latest/${documentId}`);
      currentVersion.value = version;
      return version;
    } catch (err) {
      error.value = 'Failed to fetch latest version';
      throw err;
    } finally {
      loading.value = false;
    }
  };

  const listVersions = async (
    documentId: string, 
    page = 1, 
    pageSize = 20
  ): Promise<VersionListResponse> => {
    loading.value = true;
    error.value = null;

    try {
      const response = await $api.get<VersionListResponse>(
        `/api/v1/versions/document/${documentId}`,
        {
          params: { page, page_size: pageSize }
        }
      );

      versions.value = response.versions;
      return response;
    } catch (err) {
      error.value = 'Failed to list versions';
      throw err;
    } finally {
      loading.value = false;
    }
  };

  const restoreVersion = async (request: VersionRestoreRequest): Promise<DocumentVersion> => {
    loading.value = true;
    error.value = null;

    try {
      const newVersion = await $api.post<DocumentVersion>('/api/v1/versions/restore', request);
      
      // Add to local versions list if it's for the same document
      if (versions.value.length > 0 && versions.value[0].document_id === request.document_id) {
        versions.value.unshift(newVersion);
      }

      return newVersion;
    } catch (err) {
      error.value = 'Failed to restore version';
      throw err;
    } finally {
      loading.value = false;
    }
  };

  const compareVersions = async (request: VersionComparisonRequest): Promise<DocumentDiff> => {
    loading.value = true;
    error.value = null;

    try {
      const diff = await $api.post<DocumentDiff>('/api/v1/versions/compare', request);
      return diff;
    } catch (err) {
      error.value = 'Failed to compare versions';
      throw err;
    } finally {
      loading.value = false;
    }
  };

  const getChangeHistory = async (
    documentId: string,
    page = 1,
    pageSize = 50
  ): Promise<ChangeHistoryResponse> => {
    loading.value = true;
    error.value = null;

    try {
      const response = await $api.get<ChangeHistoryResponse>(
        `/api/v1/versions/history/${documentId}`,
        {
          params: { page, page_size: pageSize }
        }
      );

      changeHistory.value = response.changes;
      return response;
    } catch (err) {
      error.value = 'Failed to fetch change history';
      throw err;
    } finally {
      loading.value = false;
    }
  };

  const deleteVersion = async (versionId: string): Promise<void> => {
    loading.value = true;
    error.value = null;

    try {
      await $api.delete(`/api/v1/versions/${versionId}`);
      
      // Remove from local versions list
      const index = versions.value.findIndex(v => v.id === versionId);
      if (index !== -1) {
        versions.value.splice(index, 1);
      }
    } catch (err) {
      error.value = 'Failed to delete version';
      throw err;
    } finally {
      loading.value = false;
    }
  };

  // Utility methods
  const clearError = () => {
    error.value = null;
  };

  const clearVersions = () => {
    versions.value = [];
    currentVersion.value = null;
  };

  const clearChangeHistory = () => {
    changeHistory.value = [];
  };

  const formatVersionDate = (dateString: string): string => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getVersionSummary = (version: DocumentVersion): string => {
    const changes = version.changes_summary?.total_changes || 0;
    const quality = version.validation_score ? (version.validation_score * 10).toFixed(1) : 'N/A';
    return `${changes} changes, ${quality}/10 quality`;
  };

  const isCurrentVersion = (version: DocumentVersion): boolean => {
    return versions.value.length > 0 && versions.value[0].id === version.id;
  };

  // Auto-save functionality for documents
  let autoSaveTimeout: NodeJS.Timeout | null = null;

  const scheduleAutoSave = (
    documentId: string,
    content: Record<string, any>,
    documentType: string,
    delay = 30000 // 30 seconds
  ) => {
    if (autoSaveTimeout) {
      clearTimeout(autoSaveTimeout);
    }

    autoSaveTimeout = setTimeout(async () => {
      try {
        await createVersion({
          document_id: documentId,
          document_type: documentType,
          content: content,
          comment: 'Auto-saved',
          user_id: 'current-user' // This would come from auth context
        });
      } catch (error) {
        console.warn('Auto-save failed:', error);
      }
    }, delay);
  };

  const cancelAutoSave = () => {
    if (autoSaveTimeout) {
      clearTimeout(autoSaveTimeout);
      autoSaveTimeout = null;
    }
  };

  return {
    // State
    loading: readonly(loading),
    error: readonly(error),
    versions: readonly(versions),
    currentVersion: readonly(currentVersion),
    changeHistory: readonly(changeHistory),

    // Computed
    hasVersions,
    latestVersion,

    // Methods
    createVersion,
    getVersion,
    getLatestVersion,
    listVersions,
    restoreVersion,
    compareVersions,
    getChangeHistory,
    deleteVersion,

    // Utility methods
    clearError,
    clearVersions,
    clearChangeHistory,
    formatVersionDate,
    getVersionSummary,
    isCurrentVersion,

    // Auto-save
    scheduleAutoSave,
    cancelAutoSave,
  };
};