<template>
  <div class="comment-component" :class="{
    'comment-thread': !isRoot,
    'comment-root': isRoot,
    'comment-highlighted': isHighlighted
  }">
    <!-- Comment Header -->
    <div class="comment-header">
      <div class="comment-author">
        <div class="author-avatar">
          <img v-if="comment.author_avatar" :src="comment.author_avatar" :alt="comment.author_name" />
          <div v-else class="avatar-placeholder">
            {{ comment.author_name?.charAt(0).toUpperCase() }}
          </div>
        </div>
        <div class="author-info">
          <span class="author-name">{{ comment.author_name }}</span>
          <span class="comment-time">{{ formatTime(comment.created_at) }}</span>
          <span v-if="comment.updated_at && comment.updated_at !== comment.created_at" class="edited-indicator">
            (edited {{ formatTime(comment.updated_at) }})
          </span>
        </div>
      </div>
      
      <div class="comment-meta">
        <span class="comment-type" :class="`type-${comment.comment_type}`">
          {{ formatCommentType(comment.comment_type) }}
        </span>
        <span v-if="comment.priority && comment.priority !== 'medium'" 
              class="comment-priority" :class="`priority-${comment.priority}`">
          {{ comment.priority.toUpperCase() }}
        </span>
        <span class="comment-status" :class="`status-${comment.status}`">
          {{ formatStatus(comment.status) }}
        </span>
      </div>
    </div>

    <!-- Comment Content -->
    <div class="comment-content">
      <div v-if="!isEditing" class="comment-text" v-html="formatContent(comment.content)"></div>
      <div v-else class="comment-edit-form">
        <textarea 
          v-model="editContent" 
          class="edit-textarea"
          :placeholder="'Edit your ' + comment.comment_type + '...'"
          rows="3"
        ></textarea>
        <div class="edit-actions">
          <button @click="saveEdit" class="save-btn" :disabled="!editContent.trim()">Save</button>
          <button @click="cancelEdit" class="cancel-btn">Cancel</button>
        </div>
      </div>
    </div>

    <!-- Text Selection Context (for annotations) -->
    <div v-if="comment.selection_range" class="comment-selection">
      <div class="selection-context">
        <span class="selection-label">Selected text:</span>
        <span class="selection-text">"{{ comment.selection_range.selected_text }}"</span>
      </div>
    </div>

    <!-- Comment Tags -->
    <div v-if="comment.tags && comment.tags.length" class="comment-tags">
      <span v-for="tag in comment.tags" :key="tag" class="tag">{{ tag }}</span>
    </div>

    <!-- Mentions and Assignees -->
    <div v-if="comment.mentions?.length || comment.assignees?.length" class="comment-references">
      <div v-if="comment.mentions?.length" class="mentions">
        <span class="ref-label">@</span>
        <span v-for="(mention, i) in comment.mentions" :key="mention" class="mention">
          {{ mention }}<span v-if="i < comment.mentions.length - 1">, </span>
        </span>
      </div>
      <div v-if="comment.assignees?.length" class="assignees">
        <span class="ref-label">Assigned:</span>
        <span v-for="(assignee, i) in comment.assignees" :key="assignee" class="assignee">
          {{ assignee }}<span v-if="i < comment.assignees.length - 1">, </span>
        </span>
      </div>
    </div>

    <!-- Comment Actions -->
    <div class="comment-actions">
      <button @click="toggleReply" class="action-btn reply-btn">
        <Icon name="reply" /> Reply
      </button>
      
      <button v-if="canEdit" @click="toggleEdit" class="action-btn edit-btn">
        <Icon name="edit" /> Edit
      </button>
      
      <button @click="toggleReactions" class="action-btn reaction-btn">
        <Icon name="thumb-up" /> 
        React
        <span v-if="reactionCount" class="reaction-count">({{ reactionCount }})</span>
      </button>
      
      <div class="reaction-panel" v-if="showReactions">
        <button 
          v-for="reaction in availableReactions" 
          :key="reaction.type"
          @click="addReaction(reaction.type)"
          class="reaction-option"
          :class="{ active: userReactions.includes(reaction.type) }"
        >
          {{ reaction.emoji }} {{ reaction.label }}
        </button>
      </div>

      <button v-if="canDelete" @click="confirmDelete" class="action-btn delete-btn">
        <Icon name="trash" /> Delete
      </button>

      <div class="comment-status-actions" v-if="canChangeStatus">
        <select @change="updateStatus" v-model="selectedStatus" class="status-select">
          <option value="open">Open</option>
          <option value="in_progress">In Progress</option>
          <option value="resolved">Resolved</option>
          <option value="closed">Closed</option>
        </select>
      </div>
    </div>

    <!-- Reply Form -->
    <div v-if="showReplyForm" class="reply-form">
      <CommentForm
        :document-id="comment.document_id"
        :parent-id="comment.id"
        :thread-id="comment.thread_id"
        @submitted="onReplySubmitted"
        @cancelled="showReplyForm = false"
        placeholder="Write a reply..."
      />
    </div>

    <!-- Nested Replies -->
    <div v-if="replies.length" class="comment-replies">
      <CommentComponent
        v-for="reply in replies"
        :key="reply.id"
        :comment="reply"
        :is-root="false"
        :depth="depth + 1"
        @updated="$emit('updated')"
        @deleted="onReplyDeleted"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useApiClient } from '~/composables/useApiClient'
import { useAuth } from '~/composables/useAuth'
import type { Comment, CommentReaction } from '~/types'

interface Props {
  comment: Comment
  isRoot?: boolean
  depth?: number
  isHighlighted?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  isRoot: true,
  depth: 0,
  isHighlighted: false
})

const emit = defineEmits<{
  updated: []
  deleted: [commentId: string]
}>()

const { $api } = useApiClient()
const { user } = useAuth()

// Component state
const isEditing = ref(false)
const editContent = ref('')
const showReplyForm = ref(false)
const showReactions = ref(false)
const selectedStatus = ref(props.comment.status)
const replies = ref<Comment[]>([])
const userReactions = ref<string[]>([])
const reactionCount = ref(0)

// Available reaction types
const availableReactions = [
  { type: 'like', emoji: '👍', label: 'Like' },
  { type: 'agree', emoji: '✅', label: 'Agree' },
  { type: 'disagree', emoji: '❌', label: 'Disagree' },
  { type: 'question', emoji: '❓', label: 'Question' },
  { type: 'important', emoji: '❗', label: 'Important' }
]

// Computed properties
const canEdit = computed(() => {
  return user.value && (
    user.value.id === props.comment.author_id ||
    user.value.is_superuser
  )
})

const canDelete = computed(() => {
  return user.value && (
    user.value.id === props.comment.author_id ||
    user.value.is_superuser
  )
})

const canChangeStatus = computed(() => {
  return user.value && (
    user.value.id === props.comment.author_id ||
    user.value.is_superuser ||
    props.comment.assignees?.includes(user.value.id)
  )
})

// Methods
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

const formatCommentType = (type: string) => {
  return type.charAt(0).toUpperCase() + type.slice(1).replace('_', ' ')
}

const formatStatus = (status: string) => {
  return status.charAt(0).toUpperCase() + status.slice(1).replace('_', ' ')
}

const formatContent = (content: string) => {
  // Basic markdown-like formatting
  return content
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/\n/g, '<br>')
}

const toggleEdit = () => {
  if (!isEditing.value) {
    editContent.value = props.comment.content
  }
  isEditing.value = !isEditing.value
}

const saveEdit = async () => {
  if (!editContent.value.trim()) return
  
  try {
    await $api.put(`/comments/${props.comment.id}`, {
      content: editContent.value.trim()
    })
    
    props.comment.content = editContent.value.trim()
    props.comment.updated_at = new Date().toISOString()
    isEditing.value = false
    emit('updated')
  } catch (error) {
    console.error('Failed to update comment:', error)
    alert('Failed to update comment. Please try again.')
  }
}

const cancelEdit = () => {
  isEditing.value = false
  editContent.value = ''
}

const toggleReply = () => {
  showReplyForm.value = !showReplyForm.value
}

const onReplySubmitted = (reply: Comment) => {
  replies.value.push(reply)
  showReplyForm.value = false
  emit('updated')
}

const onReplyDeleted = (replyId: string) => {
  replies.value = replies.value.filter(r => r.id !== replyId)
  emit('updated')
}

const toggleReactions = () => {
  showReactions.value = !showReactions.value
}

const addReaction = async (reactionType: string) => {
  try {
    await $api.post(`/comments/${props.comment.id}/reactions`, {}, {
      params: { reaction_type: reactionType }
    })
    
    if (!userReactions.value.includes(reactionType)) {
      userReactions.value.push(reactionType)
      reactionCount.value++
    }
    
    showReactions.value = false
  } catch (error) {
    console.error('Failed to add reaction:', error)
  }
}

const updateStatus = async () => {
  try {
    await $api.put(`/comments/${props.comment.id}`, {
      status: selectedStatus.value
    })
    
    props.comment.status = selectedStatus.value
    emit('updated')
  } catch (error) {
    console.error('Failed to update status:', error)
    selectedStatus.value = props.comment.status
  }
}

const confirmDelete = () => {
  if (confirm('Are you sure you want to delete this comment?')) {
    deleteComment()
  }
}

const deleteComment = async () => {
  try {
    await $api.delete(`/comments/${props.comment.id}`)
    emit('deleted', props.comment.id)
  } catch (error) {
    console.error('Failed to delete comment:', error)
    alert('Failed to delete comment. Please try again.')
  }
}

const loadReplies = async () => {
  if (props.comment.reply_count > 0) {
    try {
      const response = await $api.get(`/comments/thread/${props.comment.thread_id}`)
      replies.value = response.replies?.filter((r: Comment) => r.parent_id === props.comment.id) || []
    } catch (error) {
      console.error('Failed to load replies:', error)
    }
  }
}

const loadReactions = async () => {
  try {
    // Load user's reactions and total count
    const response = await $api.get(`/comments/${props.comment.id}/reactions`)
    userReactions.value = response.user_reactions || []
    reactionCount.value = response.total_count || 0
  } catch (error) {
    console.error('Failed to load reactions:', error)
  }
}

// Lifecycle
onMounted(() => {
  loadReplies()
  loadReactions()
})

// Watch for changes
watch(() => props.comment.status, (newStatus) => {
  selectedStatus.value = newStatus
})
</script>

<style scoped>
.comment-component {
  @apply border border-gray-200 rounded-lg p-4 bg-white shadow-sm;
  margin-bottom: 1rem;
}

.comment-thread {
  @apply ml-8 border-l-2 border-blue-200 pl-4 border-t-0 border-r-0 border-b-0;
}

.comment-highlighted {
  @apply border-yellow-300 bg-yellow-50;
}

.comment-header {
  @apply flex items-start justify-between mb-3;
}

.comment-author {
  @apply flex items-start space-x-3;
}

.author-avatar img,
.avatar-placeholder {
  @apply w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium;
}

.avatar-placeholder {
  @apply bg-gray-300 text-gray-700;
}

.author-info {
  @apply flex flex-col;
}

.author-name {
  @apply font-medium text-gray-900;
}

.comment-time,
.edited-indicator {
  @apply text-sm text-gray-500;
}

.comment-meta {
  @apply flex items-center space-x-2;
}

.comment-type,
.comment-priority,
.comment-status {
  @apply px-2 py-1 text-xs font-medium rounded;
}

.type-comment { @apply bg-blue-100 text-blue-800; }
.type-suggestion { @apply bg-green-100 text-green-800; }
.type-question { @apply bg-yellow-100 text-yellow-800; }
.type-approval { @apply bg-purple-100 text-purple-800; }
.type-concern { @apply bg-red-100 text-red-800; }
.type-annotation { @apply bg-indigo-100 text-indigo-800; }

.priority-high { @apply bg-orange-100 text-orange-800; }
.priority-critical { @apply bg-red-100 text-red-800; }

.status-open { @apply bg-green-100 text-green-800; }
.status-in_progress { @apply bg-yellow-100 text-yellow-800; }
.status-resolved { @apply bg-blue-100 text-blue-800; }
.status-closed { @apply bg-gray-100 text-gray-800; }

.comment-content {
  @apply mb-4;
}

.comment-text {
  @apply text-gray-900 leading-relaxed;
}

.edit-textarea {
  @apply w-full p-3 border border-gray-300 rounded-md resize-y min-h-[80px];
}

.edit-actions {
  @apply flex space-x-2 mt-2;
}

.save-btn {
  @apply px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50;
}

.cancel-btn {
  @apply px-4 py-2 bg-gray-300 text-gray-700 rounded-md hover:bg-gray-400;
}

.comment-selection {
  @apply mb-3 p-2 bg-blue-50 border-l-4 border-blue-400 rounded;
}

.selection-label {
  @apply text-sm font-medium text-blue-900;
}

.selection-text {
  @apply text-sm text-blue-800 italic;
}

.comment-tags {
  @apply flex flex-wrap gap-1 mb-3;
}

.tag {
  @apply px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded-full;
}

.comment-references {
  @apply flex flex-wrap items-center gap-4 mb-3 text-sm text-gray-600;
}

.mention,
.assignee {
  @apply text-blue-600;
}

.comment-actions {
  @apply flex items-center space-x-3 relative;
}

.action-btn {
  @apply flex items-center space-x-1 px-3 py-1 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded;
}

.reaction-panel {
  @apply absolute top-8 left-0 bg-white border border-gray-200 rounded-lg shadow-lg p-2 z-10;
}

.reaction-option {
  @apply flex items-center space-x-1 px-2 py-1 text-sm hover:bg-gray-100 rounded;
}

.reaction-option.active {
  @apply bg-blue-100 text-blue-800;
}

.reaction-count {
  @apply text-blue-600 font-medium;
}

.status-select {
  @apply px-2 py-1 text-sm border border-gray-300 rounded;
}

.reply-form {
  @apply mt-4 pt-4 border-t border-gray-200;
}

.comment-replies {
  @apply mt-4;
}
</style>