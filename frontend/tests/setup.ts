import { vi } from 'vitest'
import { config } from '@vue/test-utils'

// Mock Nuxt composables
vi.mock('#app', () => ({
  navigateTo: vi.fn(),
  useState: vi.fn((key, init) => {
    const state = ref(typeof init === 'function' ? init() : init)
    return state
  }),
  useCookie: vi.fn((name, opts) => {
    return ref(null)
  }),
  useRuntimeConfig: vi.fn(() => ({
    public: {
      apiBase: 'http://localhost:8000',
      wsBase: 'ws://localhost:8000',
      appName: 'Strategic Planning Platform',
      appVersion: '1.0.0'
    }
  })),
  useRouter: vi.fn(() => ({
    push: vi.fn(),
    replace: vi.fn(),
    back: vi.fn(),
    forward: vi.fn()
  })),
  useRoute: vi.fn(() => ({
    path: '/',
    params: {},
    query: {},
    fullPath: '/'
  })),
  useHead: vi.fn()
}))

// Mock fetch
global.fetch = vi.fn()

// Mock $fetch
global.$fetch = vi.fn()

// Setup global test plugins
config.global.plugins = []

// Mock IntersectionObserver
global.IntersectionObserver = vi.fn().mockImplementation((callback) => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn()
}))

// Mock ResizeObserver
global.ResizeObserver = vi.fn().mockImplementation((callback) => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn()
}))