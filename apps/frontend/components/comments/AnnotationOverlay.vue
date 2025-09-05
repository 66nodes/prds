<template>
  <div class="annotation-overlay" ref="overlayRef">
    <!-- Selection Indicator -->
    <div
      v-if="activeSelection"
      class="selection-indicator"
      :style="selectionStyle"
      @click="showAnnotationForm"
    >
      <div class="selection-highlight"></div>
      <button class="annotation-btn" title="Add annotation">
        <Icon name="message-square" />
      </button>
    </div>

    <!-- Existing Annotations -->
    <div
      v-for="annotation in visibleAnnotations"
      :key="annotation.id"
      class="annotation-marker"
      :style="getAnnotationStyle(annotation)"
      :class="{
        'annotation-active': activeAnnotationId === annotation.id,
        'annotation-hover': hoveredAnnotationId === annotation.id,
        [`annotation-${annotation.comment_type}`]: true,
        [`priority-${annotation.priority}`]: true
      }"
      @click="toggleAnnotation(annotation.id)"
      @mouseenter="hoveredAnnotationId = annotation.id"
      @mouseleave="hoveredAnnotationId = null"
      :title="getAnnotationTooltip(annotation)"
    >
      <div class="annotation-icon">
        <Icon :name="getAnnotationIcon(annotation.comment_type)" />
      </div>
      <div v-if="annotation.reply_count > 0" class="reply-badge">
        {{ annotation.reply_count }}
      </div>
    </div>

    <!-- Annotation Popover -->
    <div
      v-if="activeAnnotation"
      class="annotation-popover"
      :style="getPopoverStyle(activeAnnotation)"
    >
      <div class="popover-arrow" :style="getArrowStyle(activeAnnotation)"></div>
      
      <div class="annotation-content">
        <!-- Selected Text Context -->
        <div v-if="activeAnnotation.selection_range" class="selected-text-context">
          <div class="context-label">
            <Icon name="quote" class="context-icon" />
            Selected text:
          </div>
          <div class="selected-text">
            "{{ activeAnnotation.selection_range.selected_text }}"
          </div>
        </div>

        <!-- Annotation Comments -->
        <div class="annotation-comments">
          <CommentComponent
            :comment="activeAnnotation"
            :is-root="true"
            :depth="0"
            @updated="handleAnnotationUpdated"
            @deleted="handleAnnotationDeleted"
          />
        </div>

        <!-- Close Button -->
        <button
          @click="closeAnnotation"
          class="close-annotation-btn"
          title="Close annotation"
        >
          <Icon name="x" />
        </button>
      </div>
    </div>

    <!-- Annotation Form -->
    <div
      v-if="showForm && activeSelection"
      class="annotation-form"
      :style="getFormStyle()"
    >
      <div class="form-arrow"></div>
      
      <CommentForm
        :document-id="documentId"
        :document-type="documentType"
        :selection-range="activeSelection"
        initial-type="annotation"
        placeholder="Add your annotation..."
        @submitted="handleAnnotationSubmitted"
        @cancelled="hideAnnotationForm"
      />
    </div>

    <!-- Annotation Sidebar Toggle -->
    <div v-if="annotations.length > 0" class="annotation-sidebar-toggle">
      <button
        @click="toggleSidebar"
        class="sidebar-toggle-btn"
        :class="{ active: showSidebar }"
        :title="showSidebar ? 'Hide annotations sidebar' : 'Show annotations sidebar'"
      >
        <Icon name="message-square" />
        <span class="annotation-count">{{ annotations.length }}</span>
      </button>
    </div>

    <!-- Annotation Sidebar -->
    <div
      v-if="showSidebar"
      class="annotation-sidebar"
      :class="{ 'sidebar-visible': showSidebar }"
    >
      <div class="sidebar-header">
        <h3 class="sidebar-title">Annotations</h3>
        <button
          @click="toggleSidebar"
          class="sidebar-close-btn"
        >
          <Icon name="x" />
        </button>
      </div>

      <div class="sidebar-content">
        <div class="sidebar-filters">
          <select v-model="annotationFilter" @change="filterAnnotations" class="filter-select">
            <option value="">All Annotations</option>
            <option value="annotation">Annotations</option>
            <option value="highlight">Highlights</option>
            <option value="note">Notes</option>
          </select>

          <select v-model="statusFilter" @change="filterAnnotations" class="filter-select">
            <option value="">All Status</option>
            <option value="open">Open</option>
            <option value="resolved">Resolved</option>
          </select>
        </div>

        <div class="annotation-list">
          <div
            v-for="annotation in filteredAnnotations"
            :key="annotation.id"
            class="annotation-item"
            :class="{
              'item-active': activeAnnotationId === annotation.id,
              [`item-${annotation.comment_type}`]: true
            }"
            @click="focusAnnotation(annotation.id)"
          >
            <div class="item-header">
              <div class="item-icon">
                <Icon :name="getAnnotationIcon(annotation.comment_type)" />
              </div>
              <div class="item-meta">
                <span class="item-type">{{ formatCommentType(annotation.comment_type) }}</span>
                <span class="item-priority priority-{{ annotation.priority }}">
                  {{ annotation.priority.toUpperCase() }}
                </span>
              </div>
              <div class="item-status">
                <span class="status-badge status-{{ annotation.status }}">
                  {{ formatStatus(annotation.status) }}
                </span>
              </div>
            </div>

            <div class="item-content">
              <div v-if="annotation.selection_range" class="item-selection">
                "{{ truncateText(annotation.selection_range.selected_text, 40) }}"
              </div>
              <div class="item-text">
                {{ truncateText(annotation.content, 100) }}
              </div>
            </div>

            <div class="item-footer">
              <span class="item-author">{{ annotation.author_name }}</span>
              <span class="item-time">{{ formatTime(annotation.created_at) }}</span>
              <span v-if="annotation.reply_count > 0" class="item-replies">
                {{ annotation.reply_count }} {{ annotation.reply_count === 1 ? 'reply' : 'replies' }}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Annotation Highlights in Text -->
    <div class="text-highlights">
      <span
        v-for="annotation in visibleAnnotations"
        :key="`highlight-${annotation.id}`"
        class="text-highlight"
        :class="{
          'highlight-active': activeAnnotationId === annotation.id,
          [`highlight-${annotation.comment_type}`]: true
        }"
        :style="getHighlightStyle(annotation)"
        @click="toggleAnnotation(annotation.id)"
      ></span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useApiClient } from '~/composables/useApiClient'
import { useAuth } from '~/composables/useAuth'
import type { Comment, SelectionRange, CommentPosition } from '~/types'

interface Props {
  documentId: string
  documentType?: string
  annotations?: Comment[]
  containerSelector?: string
  autoHighlight?: boolean
  showSidebarByDefault?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  documentType: 'prd',
  annotations: () => [],
  containerSelector: '.document-content',
  autoHighlight: true,
  showSidebarByDefault: false
})

const emit = defineEmits<{
  annotationCreated: [annotation: Comment]
  annotationUpdated: [annotation: Comment]
  annotationDeleted: [annotationId: string]
  selectionChanged: [selection: SelectionRange | null]
}>()

const { $api } = useApiClient()
const { user } = useAuth()

// Component refs
const overlayRef = ref<HTMLElement>()

// Component state
const activeSelection = ref<SelectionRange | null>(null)
const activeAnnotationId = ref<string | null>(null)
const hoveredAnnotationId = ref<string | null>(null)
const showForm = ref(false)
const showSidebar = ref(props.showSidebarByDefault)
const annotationFilter = ref('')
const statusFilter = ref('')

// Document interaction
let documentContainer: HTMLElement | null = null
let selectionTimeout: NodeJS.Timeout | null = null

// Computed properties
const visibleAnnotations = computed(() => {
  return props.annotations.filter(annotation => 
    annotation.selection_range && 
    (annotation.comment_type === 'annotation' || 
     annotation.comment_type === 'highlight' || 
     annotation.comment_type === 'note')
  )
})

const filteredAnnotations = computed(() => {
  let filtered = visibleAnnotations.value

  if (annotationFilter.value) {
    filtered = filtered.filter(a => a.comment_type === annotationFilter.value)
  }

  if (statusFilter.value) {
    filtered = filtered.filter(a => a.status === statusFilter.value)
  }

  return filtered.sort((a, b) => {
    // Sort by position in document
    if (a.selection_range && b.selection_range) {
      return a.selection_range.start_offset - b.selection_range.start_offset
    }
    return 0
  })
})

const activeAnnotation = computed(() => {
  return activeAnnotationId.value 
    ? visibleAnnotations.value.find(a => a.id === activeAnnotationId.value)
    : null
})

// Methods
const initializeTextSelection = () => {
  documentContainer = document.querySelector(props.containerSelector)
  
  if (documentContainer) {
    documentContainer.addEventListener('mouseup', handleTextSelection)
    documentContainer.addEventListener('selectstart', clearActiveSelection)
  }
}

const cleanupTextSelection = () => {
  if (documentContainer) {
    documentContainer.removeEventListener('mouseup', handleTextSelection)
    documentContainer.removeEventListener('selectstart', clearActiveSelection)
  }
}

const handleTextSelection = (event: MouseEvent) => {
  // Clear any existing timeout
  if (selectionTimeout) {
    clearTimeout(selectionTimeout)
  }

  // Delay to allow selection to complete
  selectionTimeout = setTimeout(() => {
    const selection = window.getSelection()
    
    if (!selection || selection.rangeCount === 0) {
      clearActiveSelection()
      return
    }

    const range = selection.getRangeAt(0)
    const selectedText = range.toString().trim()

    if (selectedText.length < 3) {
      clearActiveSelection()
      return
    }

    // Calculate selection position
    const containerRect = documentContainer!.getBoundingClientRect()
    const rangeRect = range.getBoundingClientRect()

    const selectionRange: SelectionRange = {
      start_offset: getTextOffset(range.startContainer, range.startOffset),
      end_offset: getTextOffset(range.endContainer, range.endOffset),
      selected_text: selectedText,
      container_element: range.commonAncestorContainer.nodeName.toLowerCase(),
      xpath: generateXPath(range.commonAncestorContainer)
    }

    activeSelection.value = selectionRange
    
    // Position for annotation button
    const position = {
      x: rangeRect.right - containerRect.left + 10,
      y: rangeRect.top - containerRect.top,
      width: rangeRect.width,
      height: rangeRect.height
    }

    nextTick(() => {
      updateSelectionStyle(position)
    })

    emit('selectionChanged', selectionRange)
  }, 100)
}

const clearActiveSelection = () => {
  activeSelection.value = null
  showForm.value = false
  
  // Clear browser selection
  if (window.getSelection) {
    window.getSelection()?.removeAllRanges()
  }

  emit('selectionChanged', null)
}

const getTextOffset = (node: Node, offset: number): number => {
  let textOffset = 0
  const walker = document.createTreeWalker(
    documentContainer!,
    NodeFilter.SHOW_TEXT,
    null,
    false
  )

  let currentNode
  while (currentNode = walker.nextNode()) {
    if (currentNode === node) {
      return textOffset + offset
    }
    textOffset += currentNode.textContent?.length || 0
  }

  return textOffset
}

const generateXPath = (node: Node): string => {
  if (node.nodeType === Node.DOCUMENT_NODE) return ''
  
  const parent = node.parentNode
  if (!parent) return ''
  
  const siblings = Array.from(parent.children)
  const index = siblings.indexOf(node as Element)
  
  const tagName = (node as Element).tagName?.toLowerCase() || 'text()'
  const xpath = generateXPath(parent)
  
  return xpath + '/' + tagName + (index >= 0 ? `[${index + 1}]` : '')
}

const updateSelectionStyle = (position: any) => {
  // Update CSS custom properties for positioning
  if (overlayRef.value) {
    overlayRef.value.style.setProperty('--selection-x', `${position.x}px`)
    overlayRef.value.style.setProperty('--selection-y', `${position.y}px`)
    overlayRef.value.style.setProperty('--selection-width', `${position.width}px`)
    overlayRef.value.style.setProperty('--selection-height', `${position.height}px`)
  }
}

const showAnnotationForm = () => {
  if (activeSelection.value) {
    showForm.value = true
    closeAnnotation()
  }
}

const hideAnnotationForm = () => {
  showForm.value = false
  clearActiveSelection()
}

const handleAnnotationSubmitted = (annotation: Comment) => {
  showForm.value = false
  clearActiveSelection()
  emit('annotationCreated', annotation)
}

const toggleAnnotation = (annotationId: string) => {
  if (activeAnnotationId.value === annotationId) {
    closeAnnotation()
  } else {
    activeAnnotationId.value = annotationId
    showForm.value = false
  }
}

const closeAnnotation = () => {
  activeAnnotationId.value = null
}

const focusAnnotation = (annotationId: string) => {
  activeAnnotationId.value = annotationId
  
  // Scroll annotation into view
  const annotation = visibleAnnotations.value.find(a => a.id === annotationId)
  if (annotation && annotation.selection_range) {
    // Scroll to the annotation position
    scrollToAnnotation(annotation)
  }
}

const scrollToAnnotation = (annotation: Comment) => {
  // Implementation would scroll to the annotation position in the document
  // This is a simplified version
  const marker = document.querySelector(`[data-annotation-id="${annotation.id}"]`)
  if (marker) {
    marker.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }
}

const handleAnnotationUpdated = (annotation?: Comment) => {
  if (annotation) {
    emit('annotationUpdated', annotation)
  }
}

const handleAnnotationDeleted = (annotationId: string) => {
  if (activeAnnotationId.value === annotationId) {
    closeAnnotation()
  }
  emit('annotationDeleted', annotationId)
}

const toggleSidebar = () => {
  showSidebar.value = !showSidebar.value
}

const filterAnnotations = () => {
  // Computed property handles filtering automatically
}

// Styling methods
const selectionStyle = computed(() => {
  return {
    position: 'absolute',
    left: 'var(--selection-x, 0)',
    top: 'var(--selection-y, 0)',
    width: 'var(--selection-width, 0)',
    height: 'var(--selection-height, 0)'
  }
})

const getAnnotationStyle = (annotation: Comment) => {
  if (!annotation.selection_range || !annotation.position) {
    return { display: 'none' }
  }

  return {
    position: 'absolute',
    left: `${annotation.position.x}px`,
    top: `${annotation.position.y}px`,
    zIndex: 10
  }
}

const getPopoverStyle = (annotation: Comment) => {
  if (!annotation.position) {
    return { display: 'none' }
  }

  return {
    position: 'absolute',
    left: `${annotation.position.x + 30}px`,
    top: `${annotation.position.y}px`,
    zIndex: 20
  }
}

const getArrowStyle = (annotation: Comment) => {
  return {
    left: '-8px',
    top: '20px'
  }
}

const getFormStyle = () => {
  return {
    position: 'absolute',
    left: 'var(--selection-x, 0)',
    top: 'calc(var(--selection-y, 0) + var(--selection-height, 0) + 10px)',
    zIndex: 30
  }
}

const getHighlightStyle = (annotation: Comment) => {
  if (!annotation.selection_range || !documentContainer) {
    return { display: 'none' }
  }

  // This would calculate the actual position of the highlight in the text
  // Implementation depends on the document structure
  return {
    position: 'absolute',
    backgroundColor: getHighlightColor(annotation),
    opacity: 0.3,
    pointerEvents: 'none'
  }
}

const getHighlightColor = (annotation: Comment) => {
  const colorMap = {
    annotation: '#3b82f6', // blue
    highlight: '#fbbf24', // yellow
    note: '#10b981', // green
    suggestion: '#8b5cf6', // purple
    concern: '#ef4444' // red
  }
  
  return colorMap[annotation.comment_type] || '#6b7280'
}

const getAnnotationIcon = (type: string) => {
  const iconMap = {
    annotation: 'message-square',
    highlight: 'highlighter',
    note: 'sticky-note',
    suggestion: 'lightbulb',
    concern: 'alert-triangle'
  }
  
  return iconMap[type] || 'message-square'
}

const getAnnotationTooltip = (annotation: Comment) => {
  return `${formatCommentType(annotation.comment_type)} by ${annotation.author_name}\n"${truncateText(annotation.content, 60)}"`
}

// Utility methods
const formatCommentType = (type: string) => {
  return type.charAt(0).toUpperCase() + type.slice(1)
}

const formatStatus = (status: string) => {
  return status.charAt(0).toUpperCase() + status.slice(1).replace('_', ' ')
}

const formatTime = (timestamp: string) => {
  const date = new Date(timestamp)
  const now = new Date()
  const diff = now.getTime() - date.getTime()
  const minutes = Math.floor(diff / 60000)
  
  if (minutes < 1) return 'now'
  if (minutes < 60) return `${minutes}m`
  if (minutes < 1440) return `${Math.floor(minutes / 60)}h`
  return date.toLocaleDateString()
}

const truncateText = (text: string, maxLength: number) => {
  if (text.length <= maxLength) return text
  return text.substring(0, maxLength).trim() + '...'
}

// Lifecycle
onMounted(() => {
  initializeTextSelection()
  
  // Add click outside listener to close popovers
  document.addEventListener('click', (event) => {
    if (!overlayRef.value?.contains(event.target as Node)) {
      closeAnnotation()
      hideAnnotationForm()
    }
  })
})

onUnmounted(() => {
  cleanupTextSelection()
  
  if (selectionTimeout) {
    clearTimeout(selectionTimeout)
  }
})

// Watchers
watch(() => props.annotations, () => {
  // Update highlights when annotations change
  nextTick(() => {
    // Re-render highlights
  })
})
</script>

<style scoped>
.annotation-overlay {
  @apply relative pointer-events-none;
  --selection-x: 0;
  --selection-y: 0;
  --selection-width: 0;
  --selection-height: 0;
}

.annotation-overlay * {
  @apply pointer-events-auto;
}

/* Selection Indicator */
.selection-indicator {
  @apply absolute;
}

.selection-highlight {
  @apply absolute inset-0 bg-blue-200 bg-opacity-30 border border-blue-400 rounded;
  pointer-events: none;
}

.annotation-btn {
  @apply absolute -right-2 -top-2 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center shadow-lg hover:bg-blue-700 transition-colors;
}

/* Annotation Markers */
.annotation-marker {
  @apply w-6 h-6 rounded-full border-2 border-white shadow-lg flex items-center justify-center cursor-pointer transition-all;
}

.annotation-marker.annotation-comment { @apply bg-blue-500; }
.annotation-marker.annotation-annotation { @apply bg-indigo-500; }
.annotation-marker.annotation-highlight { @apply bg-yellow-500; }
.annotation-marker.annotation-note { @apply bg-green-500; }
.annotation-marker.annotation-suggestion { @apply bg-purple-500; }
.annotation-marker.annotation-concern { @apply bg-red-500; }

.annotation-marker.priority-high { @apply ring-2 ring-orange-400; }
.annotation-marker.priority-critical { @apply ring-2 ring-red-400 animate-pulse; }

.annotation-marker:hover,
.annotation-marker.annotation-hover {
  @apply scale-110 z-20;
}

.annotation-marker.annotation-active {
  @apply scale-125 z-30 ring-4 ring-blue-200;
}

.annotation-icon {
  @apply w-3 h-3 text-white;
}

.reply-badge {
  @apply absolute -top-1 -right-1 w-4 h-4 bg-red-500 text-white text-xs rounded-full flex items-center justify-center;
}

/* Annotation Popover */
.annotation-popover {
  @apply bg-white border border-gray-200 rounded-lg shadow-xl max-w-sm w-80 relative;
  max-height: 400px;
  overflow-y: auto;
}

.popover-arrow {
  @apply absolute w-4 h-4 bg-white border-l border-t border-gray-200 transform rotate-45;
}

.annotation-content {
  @apply p-4 relative;
}

.selected-text-context {
  @apply mb-3 p-3 bg-blue-50 border-l-4 border-blue-400 rounded;
}

.context-label {
  @apply flex items-center space-x-1 text-sm font-medium text-blue-900 mb-1;
}

.context-icon {
  @apply w-3 h-3;
}

.selected-text {
  @apply text-sm text-blue-800 italic;
}

.annotation-comments {
  @apply space-y-3;
}

.close-annotation-btn {
  @apply absolute top-2 right-2 w-6 h-6 text-gray-400 hover:text-gray-600 flex items-center justify-center;
}

/* Annotation Form */
.annotation-form {
  @apply bg-white border border-gray-200 rounded-lg shadow-xl max-w-md w-96 relative;
}

.form-arrow {
  @apply absolute -top-2 left-6 w-4 h-4 bg-white border-l border-t border-gray-200 transform rotate-45;
}

/* Sidebar */
.annotation-sidebar-toggle {
  @apply fixed right-4 top-1/2 transform -translate-y-1/2 z-40;
}

.sidebar-toggle-btn {
  @apply relative w-12 h-12 bg-blue-600 text-white rounded-full shadow-lg hover:bg-blue-700 flex items-center justify-center transition-all;
}

.sidebar-toggle-btn.active {
  @apply bg-blue-700;
}

.annotation-count {
  @apply absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-xs rounded-full flex items-center justify-center;
}

.annotation-sidebar {
  @apply fixed right-0 top-0 h-full w-80 bg-white border-l border-gray-200 shadow-xl z-30 transform translate-x-full transition-transform;
}

.annotation-sidebar.sidebar-visible {
  @apply translate-x-0;
}

.sidebar-header {
  @apply flex items-center justify-between p-4 border-b border-gray-200;
}

.sidebar-title {
  @apply text-lg font-semibold text-gray-900;
}

.sidebar-close-btn {
  @apply w-8 h-8 text-gray-400 hover:text-gray-600 flex items-center justify-center;
}

.sidebar-content {
  @apply flex-1 overflow-y-auto p-4 space-y-4;
}

.sidebar-filters {
  @apply flex space-x-2;
}

.filter-select {
  @apply flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm;
}

.annotation-list {
  @apply space-y-3;
}

.annotation-item {
  @apply p-3 border border-gray-200 rounded-lg cursor-pointer hover:bg-gray-50 transition-colors;
}

.annotation-item.item-active {
  @apply bg-blue-50 border-blue-300;
}

.annotation-item.item-annotation { @apply border-l-4 border-l-indigo-400; }
.annotation-item.item-highlight { @apply border-l-4 border-l-yellow-400; }
.annotation-item.item-note { @apply border-l-4 border-l-green-400; }

.item-header {
  @apply flex items-center justify-between mb-2;
}

.item-icon {
  @apply w-5 h-5 text-gray-600;
}

.item-meta {
  @apply flex items-center space-x-2;
}

.item-type {
  @apply text-sm font-medium text-gray-900;
}

.item-priority {
  @apply text-xs px-2 py-1 rounded;
}

.item-priority.priority-high { @apply bg-orange-100 text-orange-800; }
.item-priority.priority-critical { @apply bg-red-100 text-red-800; }

.status-badge {
  @apply text-xs px-2 py-1 rounded;
}

.status-badge.status-open { @apply bg-green-100 text-green-800; }
.status-badge.status-resolved { @apply bg-blue-100 text-blue-800; }

.item-content {
  @apply space-y-2;
}

.item-selection {
  @apply text-sm text-blue-600 italic;
}

.item-text {
  @apply text-sm text-gray-700;
}

.item-footer {
  @apply flex items-center justify-between text-xs text-gray-500 mt-2;
}

.item-author {
  @apply font-medium;
}

.item-replies {
  @apply text-blue-600;
}

/* Text Highlights */
.text-highlights {
  @apply absolute inset-0 pointer-events-none;
}

.text-highlight {
  @apply absolute pointer-events-auto cursor-pointer transition-opacity;
}

.text-highlight:hover,
.text-highlight.highlight-active {
  @apply opacity-60;
}

.text-highlight.highlight-annotation { @apply bg-indigo-400; }
.text-highlight.highlight-highlight { @apply bg-yellow-400; }
.text-highlight.highlight-note { @apply bg-green-400; }

/* Mobile Responsive */
@media (max-width: 768px) {
  .annotation-sidebar {
    @apply w-full;
  }
  
  .annotation-popover {
    @apply w-screen max-w-none left-0 mx-4;
  }
  
  .annotation-form {
    @apply w-screen max-w-none left-0 mx-4;
  }
}
</style>