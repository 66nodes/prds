// Nuxt.js 4 Configuration for Strategic Planning Platform
export default defineNuxtConfig({
  // Core modules
  modules: [
    '@nuxt/ui',
    '@nuxtjs/tailwindcss', 
    '@pinia/nuxt',
    '@vueuse/nuxt',
    '@nuxtjs/google-fonts',
    '@nuxtjs/color-mode'
  ],

  // Development tools
  devtools: { enabled: true },

  // TypeScript configuration
  typescript: {
    strict: true,
    typeCheck: true
  },

  // Nuxt UI configuration
  ui: {
    global: true,
    icons: ['heroicons', 'lucide', 'mdi']
  },

  // Color mode configuration
  colorMode: {
    preference: 'system',
    fallback: 'light',
    classSuffix: ''
  },

  // CSS imports
  css: [
    '~/assets/css/main.css',
    '~/assets/css/theme.css'
  ],

  // Google Fonts
  googleFonts: {
    families: {
      Inter: [400, 500, 600, 700],
      'JetBrains Mono': [400, 500]
    },
    display: 'swap'
  },

  // Runtime config
  runtimeConfig: {
    // Private keys (only available on server-side)
    apiSecret: process.env.API_SECRET,
    
    // Public keys (exposed to client-side)
    public: {
      apiBase: process.env.API_BASE_URL || 'http://localhost:8000',
      appName: 'Strategic Planning Platform',
      appVersion: '1.0.0'
    }
  },

  // Server-side rendering
  ssr: true,

  // Build configuration
  build: {
    transpile: ['@headlessui/vue']
  },

  // App configuration
  app: {
    head: {
      title: 'Strategic Planning Platform',
      meta: [
        { charset: 'utf-8' },
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
        { name: 'description', content: 'AI-Powered Strategic Planning Platform - Transform weeks of planning into hours' },
        { name: 'theme-color', content: '#6366f1' }
      ],
      link: [
        { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' }
      ]
    }
  },

  // Experimental features
  experimental: {
    typedPages: true
  }
})