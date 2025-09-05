<template>
  <div class="comment-system-example">
    <!-- Document Header -->
    <div class="document-header">
      <h1 class="document-title">{{ documentTitle }}</h1>
      <div class="document-meta">
        <span class="document-type">{{ documentType.toUpperCase() }}</span>
        <span class="document-id">ID: {{ documentId }}</span>
        <div class="collaboration-status">
          <Icon name="users" />
          <span>{{ activeUsersCount }} active users</span>
          <div v-if="connectionState === 'connected'" class="connection-indicator connected">
            <Icon name="wifi" />
            Connected
          </div>
          <div v-else-if="connectionState === 'connecting'" class="connection-indicator connecting">
            <Icon name="loader" class="animate-spin" />
            Connecting...
          </div>
          <div v-else class="connection-indicator disconnected">
            <Icon name="wifi-off" />
            Disconnected
          </div>
        </div>
      </div>
    </div>

    <!-- Document Content with Annotations -->
    <div class="document-content-container">
      <div class="document-content" ref="documentContentRef">
        <!-- Annotation Overlay -->
        <AnnotationOverlay
          :document-id="documentId"
          :document-type="documentType"
          :annotations="annotations"
          container-selector=".document-content"
          :auto-highlight="true"
          :show-sidebar-by-default="showAnnotationSidebar"
          @annotation-created="handleAnnotationCreated"
          @annotation-updated="handleAnnotationUpdated"
          @annotation-deleted="handleAnnotationDeleted"
          @selection-changed="handleSelectionChanged"
        />

        <!-- Sample Document Content -->
        <div class="document-text">
          <h2>Product Requirements Document</h2>
          
          <section class="section" id="overview">
            <h3>1. Overview</h3>
            <p>
              This product requirements document outlines the specifications for the new 
              <span class="highlightable">collaborative feedback system</span> that will enable 
              teams to provide inline comments and annotations on planning documents.
            </p>
            
            <p>
              The system must support <span class="highlightable">real-time collaboration</span> 
              with WebSocket connections, threaded discussions, and user permissions management.
            </p>
          </section>

          <section class="section" id="features">
            <h3>2. Core Features</h3>
            <ul>
              <li>Inline comments and annotations</li>
              <li>Threaded reply system</li>
              <li>Real-time collaboration</li>
              <li>User mentions and notifications</li>
              <li>Comment reactions and voting</li>
              <li>Status tracking (open/resolved/closed)</li>
            </ul>
          </section>

          <section class="section" id="technical">
            <h3>3. Technical Requirements</h3>
            <p>
              The frontend will be built with <span class="highlightable">Nuxt 3 and Vue 3</span> 
              using TypeScript for type safety. The backend API will use FastAPI with 
              WebSocket support for real-time updates.
            </p>
            
            <p>
              Comment data will be stored in a relational database with proper indexing 
              for efficient querying. The system must handle <span class="highlightable">high 
              concurrency</span> with multiple users commenting simultaneously.
            </p>
          </section>
        </div>
      </div>

      <!-- Typing Indicators -->
      <div v-if="typingUsers.length > 0" class="typing-indicators">
        <div class="typing-indicator">
          <Icon name="edit" class="animate-pulse" />
          <span>
            {{ formatTypingUsers(typingUsers) }} {{ typingUsers.length === 1 ? 'is' : 'are' }} typing...
          </span>
        </div>
      </div>
    </div>

    <!-- Comments Section -->
    <div class="comments-section">
      <div class="section-header">
        <h2 class="section-title">Discussion</h2>
        <div class="section-controls">
          <button
            @click="refreshComments"
            :disabled="loadingComments"
            class="refresh-btn"
          >
            <Icon :name="loadingComments ? 'loader' : 'refresh-cw'" :class="{ 'animate-spin': loadingComments }" />
            Refresh
          </button>
          
          <button
            @click="showAnnotationSidebar = !showAnnotationSidebar"
            :class="{ active: showAnnotationSidebar }"
            class="annotations-toggle-btn"
          >
            <Icon name="sticky-note" />
            Annotations ({{ annotations.length }})
          </button>
        </div>
      </div>

      <!-- Comment List -->
      <CommentList
        :document-id="documentId"
        :document-type="documentType"
        :auto-refresh="true"
        :refresh-interval="30000"
        @comment-created="handleCommentCreated"
        @comment-updated="handleCommentUpdated"
        @comment-deleted="handleCommentDeleted"
        @filters-changed="handleFiltersChanged"
      />
    </div>

    <!-- Debug Panel (Development Only) -->
    <div v-if="showDebugPanel" class="debug-panel">
      <div class="debug-header">
        <h3>Debug Information</h3>
        <button @click="showDebugPanel = false" class="close-debug">×</button>
      </div>
      
      <div class="debug-content">
        <div class="debug-section">
          <h4>WebSocket Status</h4>
          <pre>{{ {
            connectionState,
            lastError,
            reconnectAttempts,
            subscribedDocuments: Array.from(subscribedDocuments),
            activeUsersCount,
            typingUsersCount: typingUsers.length
          } }}</pre>
        </div>
        
        <div class="debug-section">
          <h4>Permissions</h4>
          <pre>{{ {
            isAuthenticated,
            roles: currentUser?.roles || [],
            canCreateComments: permissions.canCreateComments,
            canModerateComments: permissions.canModerateComments
          } }}</pre>
        </div>

        <div class="debug-section">
          <h4>Comments Stats</h4>
          <pre>{{ {
            totalComments: comments.length,
            annotations: annotations.length,
            activeSelection: !!activeSelection
          } }}</pre>
        </div>
      </div>
    </div>

    <!-- Debug Toggle (Development Only) -->
    <button
      v-if="isDevelopment"
      @click="showDebugPanel = !showDebugPanel"
      class="debug-toggle"
    >
      <Icon name="bug" />
    </button>

    <!-- Notifications -->
    <div v-if="notifications.length > 0" class="notifications-container">
      <div
        v-for="notification in notifications"
        :key="notification.id"
        class="notification"
        :class="`notification-${notification.type}`"
      >
        <Icon :name="getNotificationIcon(notification.type)" />
        <span>{{ notification.message }}</span>
        <button
          @click="dismissNotification(notification.id)"
          class="dismiss-notification"
        >
          ×
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useCommentWebSocket } from '~/composables/useCommentWebSocket'
import { useCommentPermissions } from '~/composables/useCommentPermissions'
import type { Comment, SelectionRange, CommentNotification } from '~/types'

interface Props {
  documentId: string
  documentType?: string
  documentTitle?: string
}

const props = withDefaults(defineProps<Props>(), {
  documentType: 'prd',
  documentTitle: 'Sample Document'
})

// Component refs
const documentContentRef = ref<HTMLElement>()

// Component state
const comments = ref<Comment[]>([])
const annotations = ref<Comment[]>([])
const loadingComments = ref(false)
const showAnnotationSidebar = ref(false)
const showDebugPanel = ref(false)
const activeSelection = ref<SelectionRange | null>(null)

// Notifications
const notifications = ref<Array<{
  id: string
  type: 'info' | 'success' | 'warning' | 'error'
  message: string
  timestamp: number
}>>([])

// Environment
const isDevelopment = process.env.NODE_ENV === 'development'

// WebSocket integration
const {
  connectionState,
  lastError,
  reconnectAttempts,
  subscribedDocuments,
  getActiveUsersCount,
  getTypingUsers,
  startTyping,
  stopTyping
} = useCommentWebSocket(props.documentId, {
  onCommentCreated: handleWebSocketCommentCreated,
  onCommentUpdated: handleWebSocketCommentUpdated,
  onCommentDeleted: handleWebSocketCommentDeleted,
  onUserMentioned: handleUserMentioned,
  onCommentAssigned: handleCommentAssigned,
  onTypingStarted: handleTypingStarted,
  onTypingStopped: handleTypingStopped,
  onConnectionStateChange: handleConnectionStateChange
})

// Permissions integration
const {
  permissions,
  isAuthenticated,
  currentUser
} = useCommentPermissions()

// Computed properties
const activeUsersCount = computed(() => getActiveUsersCount())
const typingUsers = computed(() => getTypingUsers())

// Methods
const loadComments = async () => {
  loadingComments.value = true
  
  try {
    // This would typically be handled by the CommentList component
    // Here we're just simulating the data structure
    
    // Load regular comments
    const commentsResponse = await $fetch(`/api/comments/document/${props.documentId}`)
    comments.value = commentsResponse.comments || []
    
    // Filter annotations
    annotations.value = comments.value.filter(comment => 
      ['annotation', 'highlight', 'note'].includes(comment.comment_type)
    )
    
  } catch (error) {
    console.error('Failed to load comments:', error)
    addNotification('error', 'Failed to load comments')
  } finally {
    loadingComments.value = false
  }
}

const refreshComments = () => {
  loadComments()
  addNotification('info', 'Comments refreshed')
}

// Comment event handlers
const handleCommentCreated = (comment: Comment) => {
  comments.value.unshift(comment)
  
  if (['annotation', 'highlight', 'note'].includes(comment.comment_type)) {
    annotations.value.unshift(comment)
  }
  
  addNotification('success', `New ${comment.comment_type} added`)
}

const handleCommentUpdated = (comment: Comment) => {
  const index = comments.value.findIndex(c => c.id === comment.id)
  if (index !== -1) {
    comments.value[index] = comment
  }
  
  const annotationIndex = annotations.value.findIndex(c => c.id === comment.id)
  if (annotationIndex !== -1) {
    annotations.value[annotationIndex] = comment
  }
  
  addNotification('info', `${comment.comment_type} updated`)
}

const handleCommentDeleted = (commentId: string) => {
  comments.value = comments.value.filter(c => c.id !== commentId)
  annotations.value = annotations.value.filter(c => c.id !== commentId)
  
  addNotification('info', 'Comment deleted')
}

// Annotation event handlers
const handleAnnotationCreated = (annotation: Comment) => {
  annotations.value.unshift(annotation)
  comments.value.unshift(annotation)
  
  addNotification('success', 'Annotation created')
}

const handleAnnotationUpdated = (annotation: Comment) => {
  handleCommentUpdated(annotation)
}

const handleAnnotationDeleted = (annotationId: string) => {
  handleCommentDeleted(annotationId)
}

const handleSelectionChanged = (selection: SelectionRange | null) => {
  activeSelection.value = selection
}

// WebSocket event handlers
const handleWebSocketCommentCreated = (comment: Comment) => {
  // Only add if we don't already have it (avoid duplicates)
  if (!comments.value.find(c => c.id === comment.id)) {
    handleCommentCreated(comment)
  }
}

const handleWebSocketCommentUpdated = (comment: Comment, changes: string[]) => {
  handleCommentUpdated(comment)
  
  if (changes.includes('status') && comment.status === 'resolved') {
    addNotification('info', `Comment resolved by ${comment.author_name}`)
  }
}

const handleWebSocketCommentDeleted = (commentId: string) => {
  handleCommentDeleted(commentId)
}

const handleUserMentioned = (notification: CommentNotification) => {
  addNotification('info', `You were mentioned by ${notification.sender_name}`)
}

const handleCommentAssigned = (notification: CommentNotification) => {
  addNotification('info', `You were assigned a comment by ${notification.sender_name}`)
}

const handleTypingStarted = (userId: string, commentId?: string) => {
  // Typing indicators are handled by the computed property
}

const handleTypingStopped = (userId: string, commentId?: string) => {
  // Typing indicators are handled by the computed property
}

const handleConnectionStateChange = (state: string) => {
  if (state === 'connected') {
    addNotification('success', 'Real-time collaboration connected')
  } else if (state === 'disconnected') {
    addNotification('warning', 'Real-time collaboration disconnected')
  } else if (state === 'error') {
    addNotification('error', 'Connection error occurred')
  }
}

const handleFiltersChanged = (filters: any) => {
  console.log('Filters changed:', filters)
}

// Notification methods
const addNotification = (type: 'info' | 'success' | 'warning' | 'error', message: string) => {
  const notification = {
    id: Date.now().toString(),
    type,
    message,
    timestamp: Date.now()
  }
  
  notifications.value.push(notification)
  
  // Auto-dismiss after 5 seconds
  setTimeout(() => {
    dismissNotification(notification.id)
  }, 5000)
}

const dismissNotification = (id: string) => {
  notifications.value = notifications.value.filter(n => n.id !== id)
}

const getNotificationIcon = (type: string) => {
  const iconMap = {
    info: 'info',
    success: 'check-circle',
    warning: 'alert-triangle',
    error: 'alert-circle'
  }
  return iconMap[type] || 'info'
}

// Utility methods
const formatTypingUsers = (users: any[]) => {
  if (users.length === 0) return ''
  if (users.length === 1) return users[0].name || 'Someone'
  if (users.length === 2) return `${users[0].name} and ${users[1].name}`
  return `${users[0].name} and ${users.length - 1} others`
}

// Keyboard shortcuts
const handleKeydown = (event: KeyboardEvent) => {
  // Ctrl/Cmd + Enter to add comment quickly
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
    // Focus on comment form if available
    const commentForm = document.querySelector('.comment-form textarea')
    if (commentForm) {
      (commentForm as HTMLElement).focus()
    }
  }
  
  // Escape to close modals/forms
  if (event.key === 'Escape') {
    activeSelection.value = null
  }
}

// Lifecycle
onMounted(() => {
  loadComments()
  document.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeydown)
})
</script>

<style scoped>
.comment-system-example {
  @apply max-w-full mx-auto space-y-8;
}

/* Document Header */
.document-header {
  @apply border-b border-gray-200 pb-6;
}

.document-title {
  @apply text-3xl font-bold text-gray-900 mb-2;
}

.document-meta {
  @apply flex items-center justify-between;
}

.document-type {
  @apply inline-flex items-center px-2 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded;
}

.document-id {
  @apply text-sm text-gray-500;
}

.collaboration-status {
  @apply flex items-center space-x-4 text-sm;
}

.connection-indicator {
  @apply flex items-center space-x-1;
}

.connection-indicator.connected {
  @apply text-green-600;
}

.connection-indicator.connecting {
  @apply text-yellow-600;
}

.connection-indicator.disconnected {
  @apply text-red-600;
}

/* Document Content */
.document-content-container {
  @apply relative;
}

.document-content {
  @apply relative bg-white border border-gray-200 rounded-lg p-8 shadow-sm;
}

.document-text {
  @apply prose prose-gray max-w-none;
}

.document-text .section {
  @apply mb-8;
}

.document-text .highlightable {
  @apply cursor-pointer hover:bg-yellow-100 transition-colors;
}

/* Typing Indicators */
.typing-indicators {
  @apply mt-4;
}

.typing-indicator {
  @apply flex items-center space-x-2 px-3 py-2 bg-blue-50 text-blue-700 text-sm rounded-md;
}

/* Comments Section */
.comments-section {
  @apply space-y-6;
}

.section-header {
  @apply flex items-center justify-between;
}

.section-title {
  @apply text-2xl font-bold text-gray-900;
}

.section-controls {
  @apply flex items-center space-x-3;
}

.refresh-btn {
  @apply flex items-center space-x-2 px-3 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200;
}

.annotations-toggle-btn {
  @apply flex items-center space-x-2 px-3 py-2 bg-yellow-100 text-yellow-700 rounded-md hover:bg-yellow-200;
}

.annotations-toggle-btn.active {
  @apply bg-yellow-200;
}

/* Debug Panel */
.debug-panel {
  @apply fixed bottom-4 right-4 w-96 max-h-96 bg-gray-900 text-white rounded-lg shadow-xl overflow-hidden;
  z-index: 1000;
}

.debug-header {
  @apply flex items-center justify-between p-3 bg-gray-800 border-b border-gray-700;
}

.debug-header h3 {
  @apply text-sm font-medium;
}

.close-debug {
  @apply text-gray-400 hover:text-white;
}

.debug-content {
  @apply p-3 space-y-3 overflow-y-auto max-h-80;
}

.debug-section {
  @apply space-y-1;
}

.debug-section h4 {
  @apply text-xs font-medium text-gray-300 uppercase tracking-wide;
}

.debug-section pre {
  @apply text-xs bg-gray-800 p-2 rounded whitespace-pre-wrap overflow-x-auto;
}

.debug-toggle {
  @apply fixed bottom-4 left-4 w-12 h-12 bg-gray-800 text-white rounded-full flex items-center justify-center hover:bg-gray-700 shadow-lg;
  z-index: 999;
}

/* Notifications */
.notifications-container {
  @apply fixed top-4 right-4 space-y-2 z-50;
}

.notification {
  @apply flex items-center space-x-3 px-4 py-3 rounded-lg shadow-lg max-w-sm;
}

.notification-info {
  @apply bg-blue-100 text-blue-800 border border-blue-200;
}

.notification-success {
  @apply bg-green-100 text-green-800 border border-green-200;
}

.notification-warning {
  @apply bg-yellow-100 text-yellow-800 border border-yellow-200;
}

.notification-error {
  @apply bg-red-100 text-red-800 border border-red-200;
}

.dismiss-notification {
  @apply text-current opacity-60 hover:opacity-100 ml-auto;
}

/* Mobile Responsive */
@media (max-width: 768px) {
  .comment-system-example {
    @apply px-4;
  }
  
  .document-meta {
    @apply flex-col items-start space-y-2;
  }
  
  .collaboration-status {
    @apply flex-wrap;
  }
  
  .debug-panel {
    @apply w-full max-w-sm bottom-0 right-0 rounded-none;
  }
  
  .notifications-container {
    @apply left-4 right-4;
  }
  
  .notification {
    @apply max-w-none;
  }
}

/* Animation */
.notification {
  animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
  from {
    transform: translateX(100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}
</style>