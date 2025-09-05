<template>
  <div class="comment-thread-view" :class="{ 
    'thread-collapsed': isCollapsed,
    'max-depth-reached': depth >= maxDepth 
  }">
    <!-- Thread Header (for collapsible threads) -->
    <div v-if="depth > 0" class="thread-header">
      <button
        @click="toggleCollapsed"
        class="collapse-toggle"
        :title="isCollapsed ? 'Expand thread' : 'Collapse thread'"
      >
        <Icon :name="isCollapsed ? 'chevron-right' : 'chevron-down'" class="collapse-icon" />
        <span class="thread-info">
          {{ formatThreadInfo() }}
        </span>
      </button>
      
      <div v-if="!isCollapsed" class="thread-meta">
        <span class="participant-count">
          {{ thread.participants.length }} participant{{ thread.participants.length !== 1 ? 's' : '' }}
        </span>
        <span class="last-activity">
          Last activity {{ formatTime(thread.last_activity) }}
        </span>
      </div>
    </div>

    <!-- Thread Content (collapsed/expanded) -->
    <div v-if="!isCollapsed" class="thread-content">
      <!-- Root Comment -->
      <div class="thread-root">
        <CommentComponent
          :comment="thread.root_comment"
          :is-root="depth === 0"
          :depth="depth"
          :is-highlighted="highlightedCommentId === thread.root_comment.id"
          @updated="$emit('updated')"
          @deleted="$emit('deleted', thread.root_comment.id)"
          @reply-created="handleReplyCreated"
        />
      </div>

      <!-- Nested Replies -->
      <div v-if="thread.replies.length > 0" class="thread-replies">
        <div 
          v-for="(reply, index) in visibleReplies" 
          :key="reply.root_comment.id" 
          class="reply-item"
          :class="{
            'reply-border': depth < 3,
            'reply-compact': depth >= 3
          }"
        >
          <CommentThreadView
            :thread="reply"
            :depth="depth + 1"
            :max-depth="maxDepth"
            :highlighted-comment-id="highlightedCommentId"
            @updated="$emit('updated')"
            @deleted="handleReplyDeleted"
          />
        </div>

        <!-- Show More Replies -->
        <div v-if="hasMoreReplies" class="show-more-replies">
          <button
            @click="showAllReplies = !showAllReplies"
            class="show-more-btn"
          >
            <Icon name="message-square" />
            {{ showAllReplies ? 'Show fewer' : `Show ${hiddenRepliesCount} more` }} replies
          </button>
        </div>

        <!-- Load More (for paginated replies) -->
        <div v-if="canLoadMore" class="load-more-replies">
          <button
            @click="loadMoreReplies"
            :disabled="loadingMore"
            class="load-more-btn"
          >
            <Icon v-if="loadingMore" name="loading" class="animate-spin" />
            <Icon v-else name="chevron-down" />
            {{ loadingMore ? 'Loading...' : 'Load more replies' }}
          </button>
        </div>
      </div>

      <!-- Thread Actions (for deep threads) -->
      <div v-if="depth >= maxDepth - 1" class="thread-actions">
        <button
          @click="expandInNewView"
          class="expand-thread-btn"
        >
          <Icon name="external-link" />
          View full thread
        </button>
      </div>
    </div>

    <!-- Collapsed Summary -->
    <div v-else class="collapsed-summary">
      <div class="collapsed-content">
        <span class="collapsed-text">
          "{{ truncateText(thread.root_comment.content, 60) }}"
        </span>
        <div class="collapsed-meta">
          <span class="author">by {{ thread.root_comment.author_name }}</span>
          <span class="reply-count">
            {{ thread.total_replies }} {{ thread.total_replies === 1 ? 'reply' : 'replies' }}
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useApiClient } from '~/composables/useApiClient'
import type { CommentThread, Comment } from '~/types'

interface Props {
  thread: CommentThread
  depth?: number
  maxDepth?: number
  highlightedCommentId?: string
  initiallyCollapsed?: boolean
  repliesPerPage?: number
}

const props = withDefaults(defineProps<Props>(), {
  depth: 0,
  maxDepth: 8,
  initiallyCollapsed: false,
  repliesPerPage: 10
})

const emit = defineEmits<{
  updated: []
  deleted: [commentId: string]
  threadExpanded: [threadId: string]
}>()

const { $api } = useApiClient()

// Component state
const isCollapsed = ref(props.initiallyCollapsed || props.depth >= props.maxDepth)
const showAllReplies = ref(false)
const loadingMore = ref(false)
const currentPage = ref(1)

// Computed properties
const visibleReplies = computed(() => {
  if (showAllReplies.value) {
    return props.thread.replies
  }
  
  // Show first N replies, then allow expansion
  const limit = Math.max(3, Math.floor(props.repliesPerPage / (props.depth + 1)))
  return props.thread.replies.slice(0, limit)
})

const hasMoreReplies = computed(() => {
  return !showAllReplies.value && props.thread.replies.length > visibleReplies.value.length
})

const hiddenRepliesCount = computed(() => {
  return props.thread.replies.length - visibleReplies.value.length
})

const canLoadMore = computed(() => {
  // This would be true if we're loading replies from API with pagination
  // For now, assuming all replies are loaded
  return false
})

// Methods
const toggleCollapsed = () => {
  isCollapsed.value = !isCollapsed.value
  
  if (!isCollapsed.value) {
    emit('threadExpanded', props.thread.root_comment.thread_id)
  }
}

const formatThreadInfo = () => {
  const replies = props.thread.total_replies
  const participants = props.thread.participants.length
  
  return `${replies} ${replies === 1 ? 'reply' : 'replies'} • ${participants} ${participants === 1 ? 'participant' : 'participants'}`
}

const formatTime = (timestamp: string) => {
  const date = new Date(timestamp)
  const now = new Date()
  const diff = now.getTime() - date.getTime()
  const minutes = Math.floor(diff / 60000)
  
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes}m ago`
  if (minutes < 1440) return `${Math.floor(minutes / 60)}h ago`
  return date.toLocaleDateString()
}

const truncateText = (text: string, maxLength: number) => {
  if (text.length <= maxLength) return text
  return text.substring(0, maxLength).trim() + '...'
}

const handleReplyCreated = (reply: Comment) => {
  // Add the new reply to the thread
  const newReplyThread: CommentThread = {
    root_comment: reply,
    replies: [],
    total_replies: 0,
    participants: [reply.author_id],
    last_activity: reply.created_at
  }
  
  props.thread.replies.push(newReplyThread)
  props.thread.total_replies += 1
  props.thread.last_activity = reply.created_at
  
  // Add author to participants if not already present
  if (!props.thread.participants.includes(reply.author_id)) {
    props.thread.participants.push(reply.author_id)
  }
  
  emit('updated')
}

const handleReplyDeleted = (commentId: string) => {
  // Remove reply from thread
  props.thread.replies = props.thread.replies.filter(r => r.root_comment.id !== commentId)
  props.thread.total_replies = Math.max(0, props.thread.total_replies - 1)
  
  emit('updated')
  emit('deleted', commentId)
}

const loadMoreReplies = async () => {
  if (loadingMore.value) return
  
  loadingMore.value = true
  
  try {
    // This would load more replies from the API
    // For now, just show more from existing data
    showAllReplies.value = true
  } catch (error) {
    console.error('Failed to load more replies:', error)
  } finally {
    loadingMore.value = false
  }
}

const expandInNewView = () => {
  // This could open the thread in a modal or navigate to a dedicated thread view
  const threadId = props.thread.root_comment.thread_id
  
  // For now, just expand the current thread
  isCollapsed.value = false
  showAllReplies.value = true
  
  emit('threadExpanded', threadId)
}

// Auto-collapse deeply nested threads
onMounted(() => {
  if (props.depth >= 4) {
    isCollapsed.value = true
  }
})
</script>

<style scoped>
.comment-thread-view {
  @apply relative;
}

.thread-collapsed {
  @apply bg-gray-50 border border-gray-200 rounded-lg;
}

.max-depth-reached {
  @apply bg-yellow-50 border-yellow-200;
}

.thread-header {
  @apply flex items-center justify-between p-3 bg-gray-100 rounded-t-lg border-b border-gray-200;
}

.collapse-toggle {
  @apply flex items-center space-x-2 text-gray-600 hover:text-gray-900;
}

.collapse-icon {
  @apply w-4 h-4;
}

.thread-info {
  @apply text-sm font-medium;
}

.thread-meta {
  @apply flex items-center space-x-3 text-xs text-gray-500;
}

.thread-content {
  @apply space-y-4;
}

.thread-root {
  @apply relative;
}

.thread-replies {
  @apply space-y-3;
}

.reply-item {
  @apply relative;
}

.reply-border {
  @apply border-l-2 border-gray-200 pl-4 ml-4;
}

.reply-compact {
  @apply ml-2 pl-2 border-l border-gray-300;
}

.show-more-replies {
  @apply mt-3;
}

.show-more-btn {
  @apply flex items-center space-x-2 px-3 py-2 text-sm text-blue-600 hover:text-blue-800 hover:bg-blue-50 rounded;
}

.load-more-replies {
  @apply mt-3 text-center;
}

.load-more-btn {
  @apply flex items-center space-x-2 px-4 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-50 rounded mx-auto;
}

.thread-actions {
  @apply mt-4 pt-3 border-t border-gray-200;
}

.expand-thread-btn {
  @apply flex items-center space-x-2 px-3 py-2 text-sm text-blue-600 hover:text-blue-800 hover:bg-blue-50 rounded;
}

.collapsed-summary {
  @apply p-3;
}

.collapsed-content {
  @apply space-y-2;
}

.collapsed-text {
  @apply text-sm text-gray-600 italic;
}

.collapsed-meta {
  @apply flex items-center space-x-3 text-xs text-gray-500;
}

.author {
  @apply font-medium;
}

.reply-count {
  @apply text-blue-600;
}

/* Depth-based styling */
.comment-thread-view[data-depth="0"] {
  @apply border-l-4 border-transparent;
}

.comment-thread-view[data-depth="1"] {
  @apply border-l-4 border-blue-200;
}

.comment-thread-view[data-depth="2"] {
  @apply border-l-4 border-green-200;
}

.comment-thread-view[data-depth="3"] {
  @apply border-l-4 border-purple-200;
}

.comment-thread-view[data-depth="4"] {
  @apply border-l-4 border-orange-200;
}

/* Animation for expand/collapse */
.thread-content {
  transition: all 0.3s ease;
}

.thread-collapsed .thread-content {
  @apply opacity-0 max-h-0 overflow-hidden;
}

/* Hover effects */
.comment-thread-view:hover {
  @apply shadow-sm;
}

.thread-collapsed:hover {
  @apply bg-gray-100;
}

/* Mobile responsive */
@media (max-width: 640px) {
  .reply-border,
  .reply-compact {
    @apply ml-2 pl-2;
  }
  
  .thread-meta {
    @apply flex-col items-start space-x-0 space-y-1;
  }
  
  .collapsed-meta {
    @apply flex-col items-start space-x-0 space-y-1;
  }
}
</style>