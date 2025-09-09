// CSS Properties type declarations for Vue components
interface CSSProperties {
  position?: 'absolute' | 'relative' | 'static' | 'fixed' | 'sticky' | undefined;
  left?: string | number | undefined;
  top?: string | number | undefined;
  zIndex?: string | number | undefined;
  display?: string | undefined;
  width?: string | number | undefined;
  height?: string | number | undefined;
  // Add other common CSS properties as needed
}

// Mock WebSocket for testing
interface MockWebSocket {
  url?: string;
  readyState?: number;
  send?: (data: string) => void;
  onopen?: (event: Event) => void;
  onclose?: (event: Event) => void;
  onmessage?: (event: MessageEvent) => void;
  onerror?: (event: Event) => void;
  close?: () => void;
}