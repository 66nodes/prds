<template>
  <div class="comment-list">
    <!-- Header -->
    <div class="comment-list-header">
      <div class="header-info">
        <h3 class="comment-title">
          Comments ({{ comments.total_count }})
        </h3>
        <div class="comment-stats">
          <span class="stat-item open">{{ comments.open_count }} open</span>
          <span class="stat-item resolved">{{ comments.resolved_count }} resolved</span>
        </div>
      </div>
      
      <div class="header-controls">
        <button 
          @click="toggleCommentForm"
          class="add-comment-btn"
        >
          <Icon name="plus" />
          Add Comment
        </button>
      </div>
    </div>

    <!-- Filters and Search -->
    <div class="comment-filters">
      <div class="filter-row">
        <div class="search-container">
          <Icon name="search" class="search-icon" />
          <input
            v-model="searchQuery"
            @input="debouncedSearch"
            type="text"
            placeholder="Search comments..."
            class="search-input"
          />
          <button
            v-if="searchQuery"
            @click="clearSearch"
            class="clear-search-btn"
            type="button"
          >
            <Icon name="x" />
          </button>
        </div>

        <div class="filter-controls">
          <!-- Status Filter -->
          <select v-model="selectedStatus" @change="applyFilters" class="filter-select">
            <option value="">All Status</option>
            <option value="open">Open</option>
            <option value="in_progress">In Progress</option>
            <option value="resolved">Resolved</option>
            <option value="closed">Closed</option>
          </select>

          <!-- Type Filter -->
          <select v-model="selectedType" @change="applyFilters" class="filter-select">
            <option value="">All Types</option>
            <option value="comment">Comments</option>
            <option value="suggestion">Suggestions</option>
            <option value="question">Questions</option>
            <option value="concern">Concerns</option>
            <option value="annotation">Annotations</option>
          </select>

          <!-- Sort Options -->
          <select v-model="sortBy" @change="applyFilters" class="filter-select">
            <option value="created_at_desc">Newest First</option>
            <option value="created_at_asc">Oldest First</option>
            <option value="priority_desc">Priority High to Low</option>
            <option value="updated_at_desc">Recently Updated</option>
          </select>
        </div>
      </div>

      <!-- Active Filters Display -->
      <div v-if="hasActiveFilters" class="active-filters">
        <span class="filters-label">Active filters:</span>
        <span v-if="searchQuery" class="filter-chip">
          Search: "{{ searchQuery }}"
          <button @click="clearSearch" class="remove-filter">×</button>
        </span>
        <span v-if="selectedStatus" class="filter-chip">
          Status: {{ selectedStatus }}
          <button @click="selectedStatus = ''; applyFilters()" class="remove-filter">×</button>
        </span>
        <span v-if="selectedType" class="filter-chip">
          Type: {{ selectedType }}
          <button @click="selectedType = ''; applyFilters()" class="remove-filter">×</button>
        </span>
        <button @click="clearAllFilters" class="clear-all-filters">Clear all</button>
      </div>
    </div>

    <!-- Loading State -->
    <div v-if="loading" class="loading-state">
      <Icon name="loading" class="animate-spin" />
      <span>Loading comments...</span>
    </div>

    <!-- Error State -->
    <div v-else-if="error" class="error-state">
      <Icon name="alert-circle" />
      <span>{{ error }}</span>
      <button @click="loadComments" class="retry-btn">Retry</button>
    </div>

    <!-- Empty State -->
    <div v-else-if="!comments.comments.length && !loading" class="empty-state">
      <Icon name="message-square" class="empty-icon" />
      <h4>No comments yet</h4>
      <p>Be the first to add a comment or suggestion to this document.</p>
      <button @click="toggleCommentForm" class="start-discussion-btn">
        Start Discussion
      </button>
    </div>

    <!-- Comment Form -->
    <div v-if="showCommentForm" class="comment-form-container">
      <CommentForm
        :document-id="documentId"
        :document-type="documentType"
        @submitted="onCommentSubmitted"
        @cancelled="showCommentForm = false"
        placeholder="Share your thoughts on this document..."
      />
    </div>

    <!-- Comments List -->
    <div v-else class="comments-container">
      <!-- Threaded View -->
      <div v-if="viewMode === 'threaded'" class="threaded-view">
        <div
          v-for="thread in comments.threads"
          :key="thread.root_comment.id"
          class="comment-thread"
        >
          <CommentComponent
            :comment="thread.root_comment"
            :is-root="true"
            :depth="0"
            @updated="handleCommentUpdated"
            @deleted="handleCommentDeleted"
          />
          
          <!-- Thread Replies -->
          <div v-if="thread.replies.length" class="thread-replies">
            <CommentThreadView
              v-for="reply in thread.replies"
              :key="reply.root_comment.id"
              :thread="reply"
              :depth="1"
              @updated="handleCommentUpdated"
              @deleted="handleCommentDeleted"
            />
          </div>
        </div>
      </div>

      <!-- Flat View -->
      <div v-else class="flat-view">
        <CommentComponent
          v-for="comment in comments.comments"
          :key="comment.id"
          :comment="comment"
          :is-root="true"
          @updated="handleCommentUpdated"
          @deleted="handleCommentDeleted"
        />
      </div>

      <!-- Load More -->
      <div v-if="comments.has_more" class="load-more-container">
        <button
          @click="loadMoreComments"
          :disabled="loadingMore"
          class="load-more-btn"
        >
          <Icon v-if="loadingMore" name="loading" class="animate-spin" />
          <Icon v-else name="chevron-down" />
          {{ loadingMore ? 'Loading...' : `Load More (${comments.total_count - comments.comments.length} remaining)` }}
        </button>
      </div>
    </div>

    <!-- View Mode Toggle -->
    <div class="view-mode-controls">
      <div class="view-mode-tabs">
        <button
          @click="viewMode = 'threaded'"
          :class="{ active: viewMode === 'threaded' }"
          class="view-mode-tab"
        >
          <Icon name="git-branch" />
          Threaded
        </button>
        <button
          @click="viewMode = 'flat'"
          :class="{ active: viewMode === 'flat' }"
          class="view-mode-tab"
        >
          <Icon name="list" />
          Flat
        </button>
      </div>

      <div class="view-options">
        <label class="checkbox-option">
          <input
            v-model="includeResolved"
            @change="applyFilters"
            type="checkbox"
          />
          Show resolved comments
        </label>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch, nextTick } from 'vue'
import { useApiClient } from '~/composables/useApiClient'
import { useAuth } from '~/composables/useAuth'
import { debounce } from 'lodash-es'
import type { CommentListResponse, Comment } from '~/types'

interface Props {
  documentId: string
  documentType?: string
  initialFilters?: {
    status?: string
    type?: string
    searchQuery?: string
  }
  autoRefresh?: boolean
  refreshInterval?: number
}

const props = withDefaults(defineProps<Props>(), {
  documentType: 'prd',
  initialFilters: () => ({}),
  autoRefresh: false,
  refreshInterval: 30000 // 30 seconds
})

const emit = defineEmits<{
  commentCreated: [comment: Comment]
  commentUpdated: [comment: Comment]
  commentDeleted: [commentId: string]
  filtersChanged: [filters: any]
}>()

const { $api } = useApiClient()
const { user } = useAuth()

// Component state
const loading = ref(false)
const loadingMore = ref(false)
const error = ref<string | null>(null)
const showCommentForm = ref(false)
const viewMode = ref<'threaded' | 'flat'>('threaded')

// Comments data
const comments = ref<CommentListResponse>({
  comments: [],
  threads: [],
  total_count: 0,
  open_count: 0,
  resolved_count: 0,
  page: 1,
  page_size: 20,
  has_more: false
})

// Filters and search
const searchQuery = ref(props.initialFilters.searchQuery || '')
const selectedStatus = ref(props.initialFilters.status || '')
const selectedType = ref(props.initialFilters.type || '')
const sortBy = ref('created_at_desc')
const includeResolved = ref(true)
const currentPage = ref(1)

// Auto-refresh
let refreshInterval: NodeJS.Timeout | null = null

// Computed properties
const hasActiveFilters = computed(() => {
  return !!(searchQuery.value || selectedStatus.value || selectedType.value)
})

// Methods
const loadComments = async (reset = true) => {
  if (reset) {
    loading.value = true
    currentPage.value = 1
    error.value = null
  } else {
    loadingMore.value = true
  }

  try {
    const params = buildQueryParams(reset ? 1 : currentPage.value + 1)
    
    const response = await $api.get(`/comments/document/${props.documentId}`, {
      params
    })

    if (reset) {
      comments.value = response
    } else {
      // Append to existing comments
      comments.value.comments.push(...response.comments)
      comments.value.threads.push(...response.threads)
      comments.value.has_more = response.has_more
      comments.value.page = response.page
      currentPage.value = response.page
    }

    // Emit filters changed event
    emit('filtersChanged', {
      searchQuery: searchQuery.value,
      selectedStatus: selectedStatus.value,
      selectedType: selectedType.value,
      sortBy: sortBy.value
    })

  } catch (err: any) {
    console.error('Failed to load comments:', err)
    error.value = err.response?.data?.detail || 'Failed to load comments'
  } finally {
    loading.value = false
    loadingMore.value = false
  }
}

const loadMoreComments = async () => {
  if (!comments.value.has_more || loadingMore.value) return
  await loadComments(false)
}

const buildQueryParams = (page: number) => {
  const params: any = {
    page,
    page_size: comments.value.page_size,
    threaded: viewMode.value === 'threaded',
    include_resolved: includeResolved.value
  }

  if (searchQuery.value) {
    // For text search, use search endpoint instead
    return null // Will trigger search endpoint
  }

  if (selectedStatus.value) {
    params.status = selectedStatus.value
  }

  if (selectedType.value) {
    params.comment_type = selectedType.value
  }

  return params
}

const searchComments = async () => {
  if (!searchQuery.value.trim()) {
    await loadComments()
    return
  }

  loading.value = true
  error.value = null

  try {
    const searchRequest = {
      query: searchQuery.value,
      document_id: props.documentId,
      status: selectedStatus.value || undefined,
      comment_type: selectedType.value || undefined,
      include_resolved: includeResolved.value,
      sort_by: sortBy.value.split('_')[0],
      sort_order: sortBy.value.split('_')[1]
    }

    const response = await $api.post('/comments/search', searchRequest, {
      params: {
        page: 1,
        page_size: comments.value.page_size
      }
    })

    comments.value = {
      ...response,
      threads: [] // Search doesn't return threaded structure
    }

    // Force flat view for search results
    if (searchQuery.value) {
      viewMode.value = 'flat'
    }

  } catch (err: any) {
    console.error('Failed to search comments:', err)
    error.value = err.response?.data?.detail || 'Failed to search comments'
  } finally {
    loading.value = false
  }
}

const debouncedSearch = debounce(searchComments, 500)

const applyFilters = async () => {
  currentPage.value = 1
  if (searchQuery.value) {
    await searchComments()
  } else {
    await loadComments()
  }
}

const clearSearch = () => {
  searchQuery.value = ''
  applyFilters()
}

const clearAllFilters = () => {
  searchQuery.value = ''
  selectedStatus.value = ''
  selectedType.value = ''
  includeResolved.value = true
  sortBy.value = 'created_at_desc'
  applyFilters()
}

const toggleCommentForm = () => {
  showCommentForm.value = !showCommentForm.value
}

const onCommentSubmitted = (comment: Comment) => {
  showCommentForm.value = false
  
  // Add comment to current list
  comments.value.comments.unshift(comment as any) // Add to beginning
  comments.value.total_count += 1
  comments.value.open_count += 1

  // If in threaded mode, refresh to get proper threading
  if (viewMode.value === 'threaded') {
    loadComments()
  }

  emit('commentCreated', comment)
}

const handleCommentUpdated = (comment?: Comment) => {
  // Refresh comments to get updated counts and structure
  loadComments()
  
  if (comment) {
    emit('commentUpdated', comment)
  }
}

const handleCommentDeleted = (commentId: string) => {
  // Remove comment from current list
  comments.value.comments = comments.value.comments.filter(c => c.id !== commentId)
  comments.value.total_count = Math.max(0, comments.value.total_count - 1)
  
  // Refresh to get updated threading and counts
  loadComments()
  
  emit('commentDeleted', commentId)
}

const startAutoRefresh = () => {
  if (props.autoRefresh && props.refreshInterval > 0) {
    refreshInterval = setInterval(() => {
      // Only refresh if user is not actively interacting
      if (!showCommentForm.value && !loading.value) {
        loadComments()
      }
    }, props.refreshInterval)
  }
}

const stopAutoRefresh = () => {
  if (refreshInterval) {
    clearInterval(refreshInterval)
    refreshInterval = null
  }
}

// Watchers
watch(() => props.documentId, () => {
  loadComments()
})

watch(viewMode, () => {
  applyFilters()
})

watch(() => props.autoRefresh, (newValue) => {
  if (newValue) {
    startAutoRefresh()
  } else {
    stopAutoRefresh()
  }
})

// Lifecycle
onMounted(async () => {
  await loadComments()
  startAutoRefresh()
})

onUnmounted(() => {
  stopAutoRefresh()
})
</script>

<style scoped>
.comment-list {
  @apply space-y-6;
}

.comment-list-header {
  @apply flex items-start justify-between;
}

.header-info {
  @apply space-y-2;
}

.comment-title {
  @apply text-xl font-semibold text-gray-900;
}

.comment-stats {
  @apply flex items-center space-x-4 text-sm;
}

.stat-item {
  @apply font-medium;
}

.stat-item.open {
  @apply text-green-600;
}

.stat-item.resolved {
  @apply text-blue-600;
}

.header-controls {
  @apply flex items-center space-x-3;
}

.add-comment-btn {
  @apply flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700;
}

.comment-filters {
  @apply space-y-4 p-4 bg-gray-50 border border-gray-200 rounded-lg;
}

.filter-row {
  @apply flex items-center space-x-4;
}

.search-container {
  @apply relative flex-1 max-w-md;
}

.search-icon {
  @apply absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400;
}

.search-input {
  @apply w-full pl-10 pr-10 py-2 border border-gray-300 rounded-md text-sm;
}

.search-input:focus {
  @apply ring-2 ring-blue-500 border-blue-500 outline-none;
}

.clear-search-btn {
  @apply absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600;
}

.filter-controls {
  @apply flex items-center space-x-2;
}

.filter-select {
  @apply px-3 py-2 border border-gray-300 rounded-md text-sm;
}

.active-filters {
  @apply flex items-center flex-wrap gap-2;
}

.filters-label {
  @apply text-sm font-medium text-gray-700;
}

.filter-chip {
  @apply inline-flex items-center px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full;
}

.remove-filter {
  @apply ml-1 text-blue-600 hover:text-blue-800;
}

.clear-all-filters {
  @apply text-sm text-gray-600 hover:text-gray-900 underline;
}

.loading-state {
  @apply flex items-center justify-center space-x-3 py-8 text-gray-600;
}

.error-state {
  @apply flex items-center justify-center space-x-3 py-8 text-red-600;
}

.retry-btn {
  @apply px-3 py-1 bg-red-100 text-red-700 rounded text-sm hover:bg-red-200;
}

.empty-state {
  @apply text-center py-12 space-y-4;
}

.empty-icon {
  @apply w-16 h-16 text-gray-300 mx-auto;
}

.empty-state h4 {
  @apply text-lg font-medium text-gray-900;
}

.empty-state p {
  @apply text-gray-600;
}

.start-discussion-btn {
  @apply px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700;
}

.comment-form-container {
  @apply border border-gray-200 rounded-lg p-4 bg-white;
}

.comments-container {
  @apply space-y-4;
}

.threaded-view {
  @apply space-y-6;
}

.comment-thread {
  @apply space-y-4;
}

.thread-replies {
  @apply ml-8 space-y-4;
}

.flat-view {
  @apply space-y-4;
}

.load-more-container {
  @apply text-center py-4;
}

.load-more-btn {
  @apply flex items-center space-x-2 px-6 py-3 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 mx-auto;
}

.view-mode-controls {
  @apply flex items-center justify-between p-4 bg-gray-50 border border-gray-200 rounded-lg;
}

.view-mode-tabs {
  @apply flex space-x-1 bg-gray-200 rounded-md p-1;
}

.view-mode-tab {
  @apply flex items-center space-x-1 px-3 py-1 text-sm rounded;
}

.view-mode-tab.active {
  @apply bg-white text-gray-900 shadow-sm;
}

.view-mode-tab:not(.active) {
  @apply text-gray-600 hover:text-gray-900;
}

.view-options {
  @apply flex items-center space-x-4;
}

.checkbox-option {
  @apply flex items-center space-x-2 text-sm text-gray-700;
}

.checkbox-option input[type="checkbox"] {
  @apply rounded border-gray-300;
}
</style>