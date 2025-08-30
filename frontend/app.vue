<template>
  <div id="app" class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <!-- Global Loading State -->
    <UNotifications />
    
    <!-- Main Application Layout -->
    <NuxtLayout>
      <NuxtPage />
    </NuxtLayout>
    
    <!-- Global Modals -->
    <UModals />
  </div>
</template>

<script setup lang="ts">
// Strategic Planning Platform - Root App Component

// SEO and Meta Configuration
useHead({
  titleTemplate: (title) => title ? `${title} | Strategic Planning Platform` : 'Strategic Planning Platform',
  meta: [
    {
      name: 'description',
      content: 'Transform weeks of strategic planning into hours with AI-powered PRD generation and GraphRAG validation'
    },
    {
      name: 'keywords', 
      content: 'strategic planning, PRD generation, AI, GraphRAG, project management, requirements documentation'
    },
    {
      name: 'author',
      content: 'Strategic Planning Platform'
    },
    {
      property: 'og:type',
      content: 'website'
    },
    {
      property: 'og:title',
      content: 'Strategic Planning Platform'
    },
    {
      property: 'og:description',
      content: 'AI-Powered Strategic Planning Platform - Transform weeks of planning into hours'
    },
    {
      property: 'og:site_name',
      content: 'Strategic Planning Platform'
    },
    {
      name: 'twitter:card',
      content: 'summary_large_image'
    }
  ],
  link: [
    {
      rel: 'icon',
      type: 'image/svg+xml',
      href: '/favicon.svg'
    }
  ]
})

// Global error handling
const handleError = (error: any) => {
  console.error('Application error:', error)
  
  // Report to monitoring service in production
  if (process.env.NODE_ENV === 'production') {
    // Add error reporting logic here
  }
}

// Handle unhandled promise rejections
onMounted(() => {
  window.addEventListener('unhandledrejection', (event) => {
    handleError(event.reason)
  })
  
  window.addEventListener('error', (event) => {
    handleError(event.error)
  })
})
</script>

<style>
/* Global styles are imported via nuxt.config.ts css array */
</style>