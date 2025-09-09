// Comprehensive Nuxt.js type declarations for frontend
declare module '#app' {
  export function useRuntimeConfig(): {
    public: Record<string, any>;
    [key: string]: any;
  };

  export function useNuxtApp(): {
    $fetch: typeof fetch;
    payload: any;
    provide: (key: string, value: any) => void;
    [key: string]: any;
  };

  export function useRouter(): any;
  export function useRoute(): any;
  export function navigateTo(to: string): Promise<void>;
  export function defineNuxtRouteMiddleware(fn: any): any;
  export function defineNuxtPlugin(fn: any): any;
  export function useState<T>(key: string, init?: () => T): Ref<T>;
  export function useCookie<T>(key: string, opts?: any): Ref<T>;
  export function ref<T>(value: T): Ref<T>;
  export function computed<T>(getter: () => T): Ref<T>;
  export function readonly<T>(ref: Ref<T>): Readonly<Ref<T>>;
  export function watch<T>(source: Ref<T> | (() => T), callback: (value: T, oldValue: T) => void): void;
  export const onMounted: (fn: () => void) => void;
  export const onUnmounted: (fn: () => void) => void;
}

declare interface Ref<T> {
  value: T;
  readonly [symbol: symbol]: true;
}

// Pinia store types
declare module 'pinia' {
  export interface StoreDefinition {
    [key: string]: any;
  }

  export function defineStore(
    id: string,
    storeSetup: () => any
  ): () => Store;

  export function defineStore(
    id: string,
    options: {
      state: () => any;
      getters?: Record<string, Function>;
      actions?: Record<string, Function>;
    }
  ): () => Store;

  export function createPinia(): any;
}

declare interface Store {
  [key: string]: any;
  $reset?: () => void;
  $state: any;
}

// Mock WebSocket for testing
declare interface MockWebSocket {
  url?: string;
  readyState?: number;
  send?: (data: string) => void;
  onopen?: (event: Event) => void;
  onclose?: (event: Event) => void;
  onmessage?: (event: MessageEvent) => void;
  onerror?: (event: Event) => void;
  close?: () => void;
}

// Extend global types
declare global {
  const process: {
    env: Record<string, string | undefined>;
    client?: boolean;
    dev?: boolean;
    [key: string]: any;
  };
}