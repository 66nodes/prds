<template>
  <div class="comment-form">
    <form @submit.prevent="submitComment" class="form-container">
      <!-- Comment Type Selection -->
      <div class="form-header">
        <div class="comment-type-selector">
          <label for="comment-type" class="form-label">Type:</label>
          <select 
            id="comment-type" 
            v-model="formData.comment_type" 
            class="type-select"
            required
          >
            <option value="comment">Comment</option>
            <option value="suggestion">Suggestion</option>
            <option value="question">Question</option>
            <option value="approval">Approval</option>
            <option value="concern">Concern</option>
            <option value="annotation">Annotation</option>
            <option value="highlight">Highlight</option>
            <option value="note">Note</option>
          </select>
        </div>

        <div class="priority-selector">
          <label for="priority" class="form-label">Priority:</label>
          <select 
            id="priority" 
            v-model="formData.priority" 
            class="priority-select"
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
            <option value="critical">Critical</option>
          </select>
        </div>
      </div>

      <!-- Selected Text Display (for annotations) -->
      <div v-if="selectionRange" class="selected-text-display">
        <div class="selection-header">
          <Icon name="quote" class="selection-icon" />
          <span class="selection-label">Commenting on selected text:</span>
        </div>
        <div class="selected-text">
          "{{ selectionRange.selected_text }}"
        </div>
        <button 
          @click="clearSelection" 
          type="button" 
          class="clear-selection-btn"
        >
          <Icon name="x" /> Clear Selection
        </button>
      </div>

      <!-- Main Comment Content -->
      <div class="content-section">
        <textarea
          v-model="formData.content"
          :placeholder="contentPlaceholder"
          class="comment-textarea"
          rows="4"
          required
          maxlength="5000"
        ></textarea>
        <div class="character-count">
          {{ formData.content.length }}/5000
        </div>
      </div>

      <!-- Tags Input -->
      <div class="tags-section">
        <label class="form-label">Tags:</label>
        <div class="tags-input-container">
          <div class="tags-display">
            <span 
              v-for="tag in formData.tags" 
              :key="tag" 
              class="tag-chip"
            >
              {{ tag }}
              <button 
                @click="removeTag(tag)" 
                type="button" 
                class="tag-remove"
              >×</button>
            </span>
          </div>
          <input
            v-model="newTag"
            @keydown.enter.prevent="addTag"
            @keydown.comma.prevent="addTag"
            placeholder="Add tags (press Enter or comma)"
            class="tag-input"
            maxlength="20"
          />
        </div>
      </div>

      <!-- Mentions Input -->
      <div class="mentions-section">
        <label class="form-label">Mentions:</label>
        <div class="mentions-input-container">
          <div class="mentions-display">
            <span 
              v-for="mention in formData.mentions" 
              :key="mention" 
              class="mention-chip"
            >
              @{{ mention }}
              <button 
                @click="removeMention(mention)" 
                type="button" 
                class="mention-remove"
              >×</button>
            </span>
          </div>
          <input
            v-model="newMention"
            @keydown.enter.prevent="addMention"
            @keydown="@"
            @input="searchUsers"
            placeholder="@username"
            class="mention-input"
            maxlength="50"
          />
          
          <!-- User Suggestions Dropdown -->
          <div v-if="userSuggestions.length" class="user-suggestions">
            <button
              v-for="user in userSuggestions"
              :key="user.id"
              @click="selectUser(user)"
              type="button"
              class="user-suggestion"
            >
              <img v-if="user.avatar" :src="user.avatar" :alt="user.name" class="user-avatar" />
              <div class="user-info">
                <span class="user-name">{{ user.name }}</span>
                <span class="user-email">{{ user.email }}</span>
              </div>
            </button>
          </div>
        </div>
      </div>

      <!-- Assignees Section -->
      <div v-if="formData.comment_type === 'concern' || formData.comment_type === 'suggestion'" 
           class="assignees-section">
        <label class="form-label">Assign to:</label>
        <div class="assignees-input-container">
          <div class="assignees-display">
            <span 
              v-for="assignee in formData.assignees" 
              :key="assignee" 
              class="assignee-chip"
            >
              {{ assignee }}
              <button 
                @click="removeAssignee(assignee)" 
                type="button" 
                class="assignee-remove"
              >×</button>
            </span>
          </div>
          <input
            v-model="newAssignee"
            @keydown.enter.prevent="addAssignee"
            @input="searchUsers"
            placeholder="Assign to user"
            class="assignee-input"
            maxlength="50"
          />
        </div>
      </div>

      <!-- Form Actions -->
      <div class="form-actions">
        <button 
          type="submit" 
          class="submit-btn"
          :disabled="!canSubmit || isSubmitting"
        >
          <Icon v-if="isSubmitting" name="loading" class="animate-spin" />
          <Icon v-else name="send" />
          {{ isReply ? 'Reply' : 'Comment' }}
        </button>
        
        <button 
          @click="$emit('cancelled')" 
          type="button" 
          class="cancel-btn"
        >
          Cancel
        </button>

        <div class="advanced-options">
          <button
            @click="toggleAdvanced"
            type="button"
            class="advanced-toggle"
          >
            <Icon name="settings" />
            {{ showAdvanced ? 'Basic' : 'Advanced' }}
          </button>
        </div>
      </div>

      <!-- Advanced Options -->
      <div v-if="showAdvanced" class="advanced-section">
        <div class="advanced-grid">
          <!-- Privacy Setting -->
          <div class="privacy-setting">
            <label class="checkbox-label">
              <input 
                v-model="formData.is_private" 
                type="checkbox" 
                class="checkbox"
              />
              Private comment (only assignees and mentions can see)
            </label>
          </div>

          <!-- Notification Setting -->
          <div class="notification-setting">
            <label class="checkbox-label">
              <input 
                v-model="formData.send_notifications" 
                type="checkbox" 
                class="checkbox"
              />
              Send email notifications to mentions and assignees
            </label>
          </div>

          <!-- Due Date (for concerns/suggestions) -->
          <div v-if="formData.comment_type === 'concern' || formData.comment_type === 'suggestion'" 
               class="due-date-setting">
            <label for="due-date" class="form-label">Due Date:</label>
            <input 
              id="due-date"
              v-model="formData.due_date" 
              type="datetime-local" 
              class="date-input"
            />
          </div>
        </div>
      </div>
    </form>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useApiClient } from '~/composables/useApiClient'
import { useAuth } from '~/composables/useAuth'
import type { CommentCreate, User, SelectionRange } from '~/types'

interface Props {
  documentId: string
  documentType?: string
  parentId?: string
  threadId?: string
  selectionRange?: SelectionRange
  placeholder?: string
  initialType?: string
}

const props = withDefaults(defineProps<Props>(), {
  documentType: 'prd',
  placeholder: 'Write your comment...',
  initialType: 'comment'
})

const emit = defineEmits<{
  submitted: [comment: any]
  cancelled: []
}>()

const { $api } = useApiClient()
const { user } = useAuth()

// Form state
const formData = ref<CommentCreate>({
  document_id: props.documentId,
  document_type: props.documentType as any,
  content: '',
  comment_type: props.initialType as any,
  priority: 'medium',
  tags: [],
  mentions: [],
  assignees: [],
  parent_id: props.parentId,
  selection_range: props.selectionRange,
  is_private: false,
  send_notifications: true
})

const isSubmitting = ref(false)
const showAdvanced = ref(false)

// Tag and mention input states
const newTag = ref('')
const newMention = ref('')
const newAssignee = ref('')
const userSuggestions = ref<User[]>([])

// Computed properties
const isReply = computed(() => !!props.parentId)

const contentPlaceholder = computed(() => {
  if (props.placeholder !== 'Write your comment...') {
    return props.placeholder
  }
  
  const typeMap = {
    comment: 'Share your thoughts...',
    suggestion: 'What would you suggest?',
    question: 'What would you like to know?',
    approval: 'Provide your approval or feedback...',
    concern: 'What concerns you?',
    annotation: 'Add your annotation...',
    highlight: 'Why is this important?',
    note: 'Add a personal note...'
  }
  
  return typeMap[formData.value.comment_type] || 'Write your comment...'
})

const canSubmit = computed(() => {
  return formData.value.content.trim().length > 0 && !isSubmitting.value
})

// Methods
const submitComment = async () => {
  if (!canSubmit.value) return
  
  isSubmitting.value = true
  
  try {
    // Clean up form data
    const submitData = {
      ...formData.value,
      content: formData.value.content.trim(),
      tags: formData.value.tags.filter(tag => tag.trim()),
      mentions: formData.value.mentions.filter(mention => mention.trim()),
      assignees: formData.value.assignees.filter(assignee => assignee.trim())
    }

    const response = await $api.post('/comments/', submitData)
    
    emit('submitted', response)
    resetForm()
  } catch (error) {
    console.error('Failed to create comment:', error)
    alert('Failed to create comment. Please try again.')
  } finally {
    isSubmitting.value = false
  }
}

const resetForm = () => {
  formData.value = {
    document_id: props.documentId,
    document_type: props.documentType as any,
    content: '',
    comment_type: props.initialType as any,
    priority: 'medium',
    tags: [],
    mentions: [],
    assignees: [],
    parent_id: props.parentId,
    selection_range: props.selectionRange,
    is_private: false,
    send_notifications: true
  }
  showAdvanced.value = false
}

const addTag = () => {
  const tag = newTag.value.trim().replace(/[,\s]+/g, ' ').toLowerCase()
  if (tag && !formData.value.tags.includes(tag) && formData.value.tags.length < 10) {
    formData.value.tags.push(tag)
    newTag.value = ''
  }
}

const removeTag = (tag: string) => {
  formData.value.tags = formData.value.tags.filter(t => t !== tag)
}

const addMention = () => {
  const mention = newMention.value.trim().replace('@', '')
  if (mention && !formData.value.mentions.includes(mention)) {
    formData.value.mentions.push(mention)
    newMention.value = ''
    userSuggestions.value = []
  }
}

const removeMention = (mention: string) => {
  formData.value.mentions = formData.value.mentions.filter(m => m !== mention)
}

const addAssignee = () => {
  const assignee = newAssignee.value.trim()
  if (assignee && !formData.value.assignees.includes(assignee)) {
    formData.value.assignees.push(assignee)
    newAssignee.value = ''
    userSuggestions.value = []
  }
}

const removeAssignee = (assignee: string) => {
  formData.value.assignees = formData.value.assignees.filter(a => a !== assignee)
}

const searchUsers = async (event?: Event) => {
  const query = (event?.target as HTMLInputElement)?.value || newMention.value || newAssignee.value
  if (query.length < 2) {
    userSuggestions.value = []
    return
  }

  try {
    const response = await $api.get('/users/search', {
      params: { q: query.replace('@', ''), limit: 5 }
    })
    userSuggestions.value = response.users || []
  } catch (error) {
    console.error('Failed to search users:', error)
    userSuggestions.value = []
  }
}

const selectUser = (selectedUser: User) => {
  if (newMention.value) {
    if (!formData.value.mentions.includes(selectedUser.username)) {
      formData.value.mentions.push(selectedUser.username)
    }
    newMention.value = ''
  } else if (newAssignee.value) {
    if (!formData.value.assignees.includes(selectedUser.username)) {
      formData.value.assignees.push(selectedUser.username)
    }
    newAssignee.value = ''
  }
  userSuggestions.value = []
}

const clearSelection = () => {
  formData.value.selection_range = undefined
  emit('cancelled')
}

const toggleAdvanced = () => {
  showAdvanced.value = !showAdvanced.value
}

// Watch for selection range changes
watch(() => props.selectionRange, (newSelection) => {
  formData.value.selection_range = newSelection
}, { immediate: true })

// Auto-focus on mount
onMounted(() => {
  const textarea = document.querySelector('.comment-textarea') as HTMLTextAreaElement
  if (textarea) {
    textarea.focus()
  }
})
</script>

<style scoped>
.comment-form {
  @apply bg-white border border-gray-200 rounded-lg p-4;
}

.form-container {
  @apply space-y-4;
}

.form-header {
  @apply flex items-center space-x-4;
}

.form-label {
  @apply text-sm font-medium text-gray-700;
}

.type-select,
.priority-select {
  @apply px-3 py-2 border border-gray-300 rounded-md text-sm;
}

.selected-text-display {
  @apply p-3 bg-blue-50 border border-blue-200 rounded-md;
}

.selection-header {
  @apply flex items-center space-x-2 mb-2;
}

.selection-icon {
  @apply w-4 h-4 text-blue-600;
}

.selection-label {
  @apply text-sm font-medium text-blue-900;
}

.selected-text {
  @apply text-sm text-blue-800 italic bg-white p-2 rounded border border-blue-200;
}

.clear-selection-btn {
  @apply mt-2 flex items-center space-x-1 text-sm text-blue-600 hover:text-blue-800;
}

.content-section {
  @apply relative;
}

.comment-textarea {
  @apply w-full p-3 border border-gray-300 rounded-md resize-y min-h-[100px] text-sm;
}

.comment-textarea:focus {
  @apply ring-2 ring-blue-500 border-blue-500 outline-none;
}

.character-count {
  @apply absolute bottom-2 right-2 text-xs text-gray-500;
}

.tags-section,
.mentions-section,
.assignees-section {
  @apply space-y-2;
}

.tags-input-container,
.mentions-input-container,
.assignees-input-container {
  @apply relative;
}

.tags-display,
.mentions-display,
.assignees-display {
  @apply flex flex-wrap gap-1 mb-2;
}

.tag-chip,
.mention-chip,
.assignee-chip {
  @apply inline-flex items-center px-2 py-1 text-xs rounded-full;
}

.tag-chip {
  @apply bg-gray-100 text-gray-800;
}

.mention-chip {
  @apply bg-blue-100 text-blue-800;
}

.assignee-chip {
  @apply bg-green-100 text-green-800;
}

.tag-remove,
.mention-remove,
.assignee-remove {
  @apply ml-1 text-sm hover:text-red-600;
}

.tag-input,
.mention-input,
.assignee-input {
  @apply w-full px-3 py-2 border border-gray-300 rounded-md text-sm;
}

.user-suggestions {
  @apply absolute top-full left-0 w-full bg-white border border-gray-300 rounded-md shadow-lg z-10 mt-1;
}

.user-suggestion {
  @apply flex items-center space-x-2 p-2 hover:bg-gray-100 w-full text-left;
}

.user-avatar {
  @apply w-6 h-6 rounded-full;
}

.user-info {
  @apply flex flex-col;
}

.user-name {
  @apply text-sm font-medium text-gray-900;
}

.user-email {
  @apply text-xs text-gray-600;
}

.form-actions {
  @apply flex items-center justify-between;
}

.submit-btn {
  @apply flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed;
}

.cancel-btn {
  @apply px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300;
}

.advanced-toggle {
  @apply flex items-center space-x-1 px-3 py-2 text-sm text-gray-600 hover:text-gray-900;
}

.advanced-section {
  @apply border-t border-gray-200 pt-4;
}

.advanced-grid {
  @apply space-y-3;
}

.checkbox-label {
  @apply flex items-center space-x-2 text-sm text-gray-700;
}

.checkbox {
  @apply rounded border-gray-300;
}

.date-input {
  @apply px-3 py-2 border border-gray-300 rounded-md text-sm;
}
</style>