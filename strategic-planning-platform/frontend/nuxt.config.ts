export default defineNuxtConfig({
  devtools: { enabled: true },
  
  // Modules
  modules: [
    '@nuxt/ui',
    '@nuxtjs/tailwindcss', 
    '@pinia/nuxt',
    '@vueuse/nuxt',
    '@nuxtjs/google-fonts'
  ],

  // TypeScript configuration
  typescript: {
    strict: true,
    typeCheck: true
  },

  // CSS configuration
  css: [
    '~/assets/css/main.css'
  ],

  // UI configuration
  ui: {
    global: true,
    icons: ['heroicons', 'lucide']
  },

  // Tailwind configuration
  tailwindcss: {
    cssPath: '~/assets/css/tailwind.css',
    configPath: 'tailwind.config.ts'
  },

  // Google Fonts
  googleFonts: {
    families: {
      Inter: [400, 500, 600, 700],
      'JetBrains Mono': [400, 500]
    }
  },

  // Runtime config
  runtimeConfig: {
    // Private keys (only available on the server-side)
    jwtSecret: process.env.JWT_SECRET,
    
    // Public keys (exposed to the client-side)
    public: {
      apiBase: process.env.API_BASE_URL || 'http://localhost:8000',
      graphragEndpoint: process.env.GRAPHRAG_ENDPOINT,
      appName: 'Strategic Planning Platform',
      socketUrl: process.env.SOCKET_URL || 'http://localhost:8000'
    }
  },

  // Build configuration
  nitro: {
    experimental: {
      wasm: true
    }
  },

  // Vite configuration
  vite: {
    optimizeDeps: {
      include: ['socket.io-client']
    }
  },

  // App configuration
  app: {
    head: {
      title: 'Strategic Planning Platform',
      meta: [
        { name: 'description', content: 'AI-Powered Strategic Planning Platform' },
        { name: 'viewport', content: 'width=device-width, initial-scale=1' }
      ],
      link: [
        { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' }
      ]
    }
  },

  // Development configuration
  devServer: {
    port: 3000,
    host: '0.0.0.0'
  },

  // Security headers
  routeRules: {
    '/**': { 
      headers: { 
        'X-Frame-Options': 'DENY',
        'X-Content-Type-Options': 'nosniff',
        'Referrer-Policy': 'strict-origin-when-cross-origin'
      } 
    }
  },

  // Experimental features
  experimental: {
    payloadExtraction: false
  }
})