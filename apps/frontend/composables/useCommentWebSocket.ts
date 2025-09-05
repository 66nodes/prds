import { ref, onMounted, onUnmounted, watch } from 'vue'
import { useAuth } from './useAuth'
import type { Comment, CommentNotification } from '~/types'

interface WebSocketMessage {
  type: string
  data: any
  timestamp: string
  user_id?: string
}

interface CommentWebSocketEvents {
  onCommentCreated?: (comment: Comment) => void
  onCommentUpdated?: (comment: Comment, changes: string[]) => void
  onCommentDeleted?: (commentId: string) => void
  onCommentReaction?: (commentId: string, reaction: any) => void
  onUserMentioned?: (notification: CommentNotification) => void
  onCommentAssigned?: (notification: CommentNotification) => void
  onTypingStarted?: (userId: string, commentId?: string) => void
  onTypingStopped?: (userId: string, commentId?: string) => void
  onUserJoined?: (userId: string, documentId: string) => void
  onUserLeft?: (userId: string, documentId: string) => void
  onConnectionStateChange?: (state: 'connecting' | 'connected' | 'disconnected' | 'error') => void
}

export const useCommentWebSocket = (documentId?: string, events: CommentWebSocketEvents = {}) => {
  const { user } = useAuth()
  
  // Connection state
  const socket = ref<WebSocket | null>(null)
  const connectionState = ref<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected')
  const lastError = ref<string | null>(null)
  const reconnectAttempts = ref(0)
  const maxReconnectAttempts = 5
  const baseReconnectDelay = 1000 // 1 second
  
  // Subscriptions
  const subscribedDocuments = ref<Set<string>>(new Set())
  const activeUsers = ref<Map<string, any>>(new Map())
  const typingUsers = ref<Map<string, { userId: string; commentId?: string; timestamp: number }>>(new Map())
  
  // Message queue for offline messages
  const messageQueue = ref<any[]>([])
  
  // Typing indicators
  const typingTimeout = ref<NodeJS.Timeout | null>(null)
  const isTyping = ref(false)
  
  let heartbeatInterval: NodeJS.Timeout | null = null
  let reconnectTimeout: NodeJS.Timeout | null = null

  // WebSocket URL construction
  const getWebSocketUrl = () => {
    const config = useRuntimeConfig()
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = config.public.apiBaseUrl?.replace(/^https?:\/\//, '') || 'localhost:8000'
    return `${protocol}//${host}/ws/comments`
  }

  // Connect to WebSocket
  const connect = async () => {
    if (!user.value?.id) {
      console.warn('Cannot connect to WebSocket: No authenticated user')
      return
    }

    if (socket.value?.readyState === WebSocket.CONNECTING || socket.value?.readyState === WebSocket.OPEN) {
      return
    }

    try {
      connectionState.value = 'connecting'
      events.onConnectionStateChange?.('connecting')
      
      const wsUrl = getWebSocketUrl()
      const token = await $fetch('/auth/ws-token', {
        headers: {
          Authorization: `Bearer ${user.value.token}`
        }
      })
      
      socket.value = new WebSocket(`${wsUrl}?token=${token}`)
      
      socket.value.onopen = handleOpen
      socket.value.onmessage = handleMessage
      socket.value.onclose = handleClose
      socket.value.onerror = handleError
      
    } catch (error) {
      console.error('Failed to establish WebSocket connection:', error)
      connectionState.value = 'error'
      lastError.value = error instanceof Error ? error.message : 'Connection failed'
      events.onConnectionStateChange?.('error')
      scheduleReconnect()
    }
  }

  // Handle connection open
  const handleOpen = () => {
    console.log('WebSocket connected')
    connectionState.value = 'connected'
    events.onConnectionStateChange?.('connected')
    reconnectAttempts.value = 0
    lastError.value = null
    
    // Start heartbeat
    startHeartbeat()
    
    // Subscribe to document if provided
    if (documentId) {
      subscribeToDocument(documentId)
    }
    
    // Process queued messages
    processMessageQueue()
  }

  // Handle incoming messages
  const handleMessage = (event: MessageEvent) => {
    try {
      const message: WebSocketMessage = JSON.parse(event.data)
      
      switch (message.type) {
        case 'comment_created':
          events.onCommentCreated?.(message.data.comment)
          break
          
        case 'comment_updated':
          events.onCommentUpdated?.(message.data.comment, message.data.changes || [])
          break
          
        case 'comment_deleted':
          events.onCommentDeleted?.(message.data.comment_id)
          break
          
        case 'comment_reaction':
          events.onCommentReaction?.(message.data.comment_id, message.data.reaction)
          break
          
        case 'user_mentioned':
          events.onUserMentioned?.(message.data)
          break
          
        case 'comment_assigned':
          events.onCommentAssigned?.(message.data)
          break
          
        case 'typing_started':
          handleTypingStarted(message.data)
          break
          
        case 'typing_stopped':
          handleTypingStopped(message.data)
          break
          
        case 'user_joined':
          handleUserJoined(message.data)
          break
          
        case 'user_left':
          handleUserLeft(message.data)
          break
          
        case 'heartbeat_response':
          // Keep connection alive
          break
          
        case 'subscription_confirmed':
          console.log('Subscription confirmed:', message.data)
          break
          
        case 'error':
          console.error('WebSocket error message:', message.data)
          lastError.value = message.data.message
          break
          
        default:
          console.warn('Unknown WebSocket message type:', message.type)
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error)
    }
  }

  // Handle connection close
  const handleClose = (event: CloseEvent) => {
    console.log('WebSocket disconnected:', event.code, event.reason)
    connectionState.value = 'disconnected'
    events.onConnectionStateChange?.('disconnected')
    
    // Stop heartbeat
    stopHeartbeat()
    
    // Clear active users and typing indicators
    activeUsers.value.clear()
    typingUsers.value.clear()
    
    // Attempt reconnection if not a clean close
    if (event.code !== 1000 && event.code !== 1001) {
      scheduleReconnect()
    }
  }

  // Handle connection error
  const handleError = (event: Event) => {
    console.error('WebSocket error:', event)
    connectionState.value = 'error'
    lastError.value = 'Connection error occurred'
    events.onConnectionStateChange?.('error')
  }

  // Schedule reconnection
  const scheduleReconnect = () => {
    if (reconnectAttempts.value >= maxReconnectAttempts) {
      console.error('Max reconnection attempts reached')
      return
    }
    
    const delay = baseReconnectDelay * Math.pow(2, reconnectAttempts.value)
    console.log(`Scheduling reconnection in ${delay}ms (attempt ${reconnectAttempts.value + 1})`)
    
    reconnectTimeout = setTimeout(() => {
      reconnectAttempts.value++
      connect()
    }, delay)
  }

  // Cancel reconnection
  const cancelReconnect = () => {
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout)
      reconnectTimeout = null
    }
  }

  // Start heartbeat to keep connection alive
  const startHeartbeat = () => {
    stopHeartbeat()
    heartbeatInterval = setInterval(() => {
      if (socket.value?.readyState === WebSocket.OPEN) {
        sendMessage({
          type: 'heartbeat',
          data: { timestamp: Date.now() }
        })
      }
    }, 30000) // 30 seconds
  }

  // Stop heartbeat
  const stopHeartbeat = () => {
    if (heartbeatInterval) {
      clearInterval(heartbeatInterval)
      heartbeatInterval = null
    }
  }

  // Send message to WebSocket
  const sendMessage = (message: any) => {
    if (socket.value?.readyState === WebSocket.OPEN) {
      socket.value.send(JSON.stringify({
        ...message,
        timestamp: new Date().toISOString(),
        user_id: user.value?.id
      }))
    } else {
      // Queue message for when connection is restored
      messageQueue.value.push(message)
    }
  }

  // Process queued messages
  const processMessageQueue = () => {
    while (messageQueue.value.length > 0) {
      const message = messageQueue.value.shift()
      sendMessage(message)
    }
  }

  // Subscribe to document comments
  const subscribeToDocument = (docId: string) => {
    if (!docId) return
    
    subscribedDocuments.value.add(docId)
    sendMessage({
      type: 'subscribe_to_document',
      data: { document_id: docId }
    })
  }

  // Unsubscribe from document comments
  const unsubscribeFromDocument = (docId: string) => {
    if (!docId) return
    
    subscribedDocuments.value.delete(docId)
    sendMessage({
      type: 'unsubscribe_from_document',
      data: { document_id: docId }
    })
  }

  // Typing indicators
  const startTyping = (commentId?: string) => {
    if (isTyping.value) return
    
    isTyping.value = true
    sendMessage({
      type: 'typing_started',
      data: { 
        comment_id: commentId,
        document_id: documentId
      }
    })
    
    // Clear existing timeout
    if (typingTimeout.value) {
      clearTimeout(typingTimeout.value)
    }
    
    // Auto-stop typing after 3 seconds of inactivity
    typingTimeout.value = setTimeout(() => {
      stopTyping(commentId)
    }, 3000)
  }

  const stopTyping = (commentId?: string) => {
    if (!isTyping.value) return
    
    isTyping.value = false
    sendMessage({
      type: 'typing_stopped',
      data: { 
        comment_id: commentId,
        document_id: documentId
      }
    })
    
    if (typingTimeout.value) {
      clearTimeout(typingTimeout.value)
      typingTimeout.value = null
    }
  }

  // Handle typing events
  const handleTypingStarted = (data: { user_id: string; comment_id?: string }) => {
    typingUsers.value.set(data.user_id, {
      userId: data.user_id,
      commentId: data.comment_id,
      timestamp: Date.now()
    })
    
    events.onTypingStarted?.(data.user_id, data.comment_id)
  }

  const handleTypingStopped = (data: { user_id: string; comment_id?: string }) => {
    typingUsers.value.delete(data.user_id)
    events.onTypingStopped?.(data.user_id, data.comment_id)
  }

  // Handle user presence
  const handleUserJoined = (data: { user_id: string; user_name: string; document_id: string }) => {
    activeUsers.value.set(data.user_id, {
      id: data.user_id,
      name: data.user_name,
      joinedAt: Date.now()
    })
    
    events.onUserJoined?.(data.user_id, data.document_id)
  }

  const handleUserLeft = (data: { user_id: string; document_id: string }) => {
    activeUsers.value.delete(data.user_id)
    typingUsers.value.delete(data.user_id)
    
    events.onUserLeft?.(data.user_id, data.document_id)
  }

  // Disconnect from WebSocket
  const disconnect = () => {
    cancelReconnect()
    stopHeartbeat()
    
    if (typingTimeout.value) {
      clearTimeout(typingTimeout.value)
    }
    
    if (socket.value) {
      socket.value.close(1000, 'Client disconnect')
      socket.value = null
    }
    
    connectionState.value = 'disconnected'
    events.onConnectionStateChange?.('disconnected')
    
    // Clear state
    subscribedDocuments.value.clear()
    activeUsers.value.clear()
    typingUsers.value.clear()
    messageQueue.value = []
  }

  // Force reconnect
  const reconnect = () => {
    disconnect()
    reconnectAttempts.value = 0
    connect()
  }

  // Get typing users for a specific comment
  const getTypingUsers = (commentId?: string) => {
    return Array.from(typingUsers.value.values()).filter(typing => 
      typing.commentId === commentId && typing.userId !== user.value?.id
    )
  }

  // Get active users count
  const getActiveUsersCount = () => {
    return activeUsers.value.size
  }

  // Get active users list
  const getActiveUsers = () => {
    return Array.from(activeUsers.value.values())
  }

  // Auto-connect when user is authenticated
  watch(() => user.value?.id, (newUserId, oldUserId) => {
    if (newUserId && newUserId !== oldUserId) {
      connect()
    } else if (!newUserId) {
      disconnect()
    }
  }, { immediate: true })

  // Auto-subscribe to document changes
  watch(() => documentId, (newDocId, oldDocId) => {
    if (oldDocId) {
      unsubscribeFromDocument(oldDocId)
    }
    if (newDocId && connectionState.value === 'connected') {
      subscribeToDocument(newDocId)
    }
  })

  // Lifecycle
  onMounted(() => {
    if (user.value?.id) {
      connect()
    }
  })

  onUnmounted(() => {
    disconnect()
  })

  return {
    // Connection state
    connectionState: readonly(connectionState),
    lastError: readonly(lastError),
    reconnectAttempts: readonly(reconnectAttempts),
    
    // Active state
    activeUsers: readonly(activeUsers),
    typingUsers: readonly(typingUsers),
    subscribedDocuments: readonly(subscribedDocuments),
    
    // Connection methods
    connect,
    disconnect,
    reconnect,
    
    // Subscription methods
    subscribeToDocument,
    unsubscribeFromDocument,
    
    // Typing methods
    startTyping,
    stopTyping,
    getTypingUsers,
    
    // User presence methods
    getActiveUsersCount,
    getActiveUsers,
    
    // Utility methods
    sendMessage
  }
}