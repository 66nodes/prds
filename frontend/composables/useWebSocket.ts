import type { WebSocketMessage, WebSocketMessageType } from '~/types'

interface WebSocketOptions {
  autoReconnect?: boolean
  reconnectInterval?: number
  maxReconnectAttempts?: number
  heartbeatInterval?: number
}

export const useWebSocket = (options: WebSocketOptions = {}) => {
  const config = useRuntimeConfig()
  const { isAuthenticated } = useAuth()
  
  // WebSocket state
  const socket = ref<WebSocket | null>(null)
  const isConnected = ref(false)
  const reconnectAttempts = ref(0)
  const messageQueue = ref<WebSocketMessage[]>([])
  
  // Event handlers
  const eventHandlers = new Map<WebSocketMessageType, Set<(data: any) => void>>()
  
  // Options with defaults
  const wsOptions = {
    autoReconnect: options.autoReconnect ?? true,
    reconnectInterval: options.reconnectInterval ?? 3000,
    maxReconnectAttempts: options.maxReconnectAttempts ?? 10,
    heartbeatInterval: options.heartbeatInterval ?? 30000
  }
  
  let heartbeatTimer: NodeJS.Timeout | null = null
  let reconnectTimer: NodeJS.Timeout | null = null

  // Get WebSocket URL with authentication
  const getWebSocketUrl = (): string => {
    const baseUrl = config.public.wsBase
    const accessToken = useCookie('access_token').value
    
    if (accessToken) {
      return `${baseUrl}/ws?token=${accessToken}`
    }
    
    return `${baseUrl}/ws`
  }

  // Connect to WebSocket
  const connect = (): void => {
    if (!isAuthenticated.value) {
      console.warn('Cannot connect to WebSocket: User not authenticated')
      return
    }

    if (socket.value?.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected')
      return
    }

    try {
      const url = getWebSocketUrl()
      socket.value = new WebSocket(url)

      // Connection opened
      socket.value.onopen = () => {
        console.log('WebSocket connected')
        isConnected.value = true
        reconnectAttempts.value = 0

        // Start heartbeat
        startHeartbeat()

        // Send queued messages
        flushMessageQueue()

        // Emit connect event
        emitEvent('connect' as WebSocketMessageType, null)
      }

      // Message received
      socket.value.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          handleMessage(message)
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }

      // Connection closed
      socket.value.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason)
        isConnected.value = false
        stopHeartbeat()

        // Emit disconnect event
        emitEvent('disconnect' as WebSocketMessageType, { code: event.code, reason: event.reason })

        // Auto reconnect if enabled
        if (wsOptions.autoReconnect && reconnectAttempts.value < wsOptions.maxReconnectAttempts) {
          scheduleReconnect()
        }
      }

      // Connection error
      socket.value.onerror = (error) => {
        console.error('WebSocket error:', error)
        emitEvent('error' as WebSocketMessageType, error)
      }
    } catch (error) {
      console.error('Failed to connect to WebSocket:', error)
      emitEvent('error' as WebSocketMessageType, error)
    }
  }

  // Disconnect from WebSocket
  const disconnect = (): void => {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }

    stopHeartbeat()

    if (socket.value) {
      socket.value.close(1000, 'Client disconnect')
      socket.value = null
    }

    isConnected.value = false
    reconnectAttempts.value = 0
  }

  // Schedule reconnection
  const scheduleReconnect = (): void => {
    if (reconnectTimer) return

    reconnectAttempts.value++
    const delay = Math.min(
      wsOptions.reconnectInterval * Math.pow(1.5, reconnectAttempts.value - 1),
      30000
    )

    console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts.value})`)

    reconnectTimer = setTimeout(() => {
      reconnectTimer = null
      connect()
    }, delay)
  }

  // Start heartbeat
  const startHeartbeat = (): void => {
    stopHeartbeat()

    heartbeatTimer = setInterval(() => {
      if (isConnected.value) {
        send('heartbeat' as WebSocketMessageType, { timestamp: Date.now() })
      }
    }, wsOptions.heartbeatInterval)
  }

  // Stop heartbeat
  const stopHeartbeat = (): void => {
    if (heartbeatTimer) {
      clearInterval(heartbeatTimer)
      heartbeatTimer = null
    }
  }

  // Send message
  const send = (type: WebSocketMessageType, payload: any): void => {
    const message: WebSocketMessage = {
      type,
      payload,
      timestamp: new Date().toISOString()
    }

    if (isConnected.value && socket.value?.readyState === WebSocket.OPEN) {
      socket.value.send(JSON.stringify(message))
    } else {
      // Queue message if not connected
      messageQueue.value.push(message)
    }
  }

  // Flush message queue
  const flushMessageQueue = (): void => {
    while (messageQueue.value.length > 0 && isConnected.value) {
      const message = messageQueue.value.shift()
      if (message && socket.value?.readyState === WebSocket.OPEN) {
        socket.value.send(JSON.stringify(message))
      }
    }
  }

  // Handle incoming message
  const handleMessage = (message: WebSocketMessage): void => {
    // Skip heartbeat messages
    if (message.type === 'heartbeat') {
      return
    }

    // Emit to registered handlers
    emitEvent(message.type, message.payload)
  }

  // Register event handler
  const on = (type: WebSocketMessageType, handler: (data: any) => void): void => {
    if (!eventHandlers.has(type)) {
      eventHandlers.set(type, new Set())
    }
    eventHandlers.get(type)?.add(handler)
  }

  // Unregister event handler
  const off = (type: WebSocketMessageType, handler: (data: any) => void): void => {
    eventHandlers.get(type)?.delete(handler)
  }

  // Emit event to handlers
  const emitEvent = (type: WebSocketMessageType, data: any): void => {
    const handlers = eventHandlers.get(type)
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(data)
        } catch (error) {
          console.error(`Error in WebSocket event handler for ${type}:`, error)
        }
      })
    }
  }

  // Subscribe to PRD updates
  const subscribeToPRDUpdates = (prdId: string): void => {
    send('prd_update' as WebSocketMessageType, { action: 'subscribe', prdId })
  }

  // Unsubscribe from PRD updates
  const unsubscribeFromPRDUpdates = (prdId: string): void => {
    send('prd_update' as WebSocketMessageType, { action: 'unsubscribe', prdId })
  }

  // Subscribe to agent status updates
  const subscribeToAgentStatus = (agentId?: string): void => {
    send('agent_status' as WebSocketMessageType, { action: 'subscribe', agentId })
  }

  // Unsubscribe from agent status updates
  const unsubscribeFromAgentStatus = (agentId?: string): void => {
    send('agent_status' as WebSocketMessageType, { action: 'unsubscribe', agentId })
  }

  // Auto-connect when authenticated
  watch(isAuthenticated, (newValue) => {
    if (newValue && !isConnected.value) {
      connect()
    } else if (!newValue && isConnected.value) {
      disconnect()
    }
  })

  // Cleanup on unmount
  onUnmounted(() => {
    disconnect()
  })

  return {
    isConnected: readonly(isConnected),
    connect,
    disconnect,
    send,
    on,
    off,
    subscribeToPRDUpdates,
    unsubscribeFromPRDUpdates,
    subscribeToAgentStatus,
    unsubscribeFromAgentStatus
  }
}