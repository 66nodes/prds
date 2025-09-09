// WebSocket type declarations
export interface WebSocketMessage {
  type: string;
  data: any;
  id?: string;
  timestamp?: string;
}

export interface WebSocketConnection {
  connected: boolean;
  reconnecting: boolean;
  reconnectAttempts: number;
  lastUsed: Date;
}

export interface WebSocketState {
  connection: WebSocketConnection;
  subscriptions: Set<string>;
  messageQueue: WebSocketMessage[];
  heartbeatTimer?: NodeJS.Timeout;
}