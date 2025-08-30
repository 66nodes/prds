<template>
  <div class="min-h-screen">
    <!-- Hero Section -->
    <section class="section-spacing bg-gradient-to-br from-white via-gray-50 to-indigo-50 dark:from-gray-900 dark:via-gray-800 dark:to-indigo-950">
      <div class="page-container">
        <div class="mx-auto max-w-4xl text-center">
          <h1 class="text-4xl font-bold tracking-tight sm:text-6xl">
            <span class="text-gradient">AI-Powered</span>
            <br>
            Strategic Planning
          </h1>
          
          <p class="mt-6 text-lg leading-8 text-gray-600 dark:text-gray-300">
            Transform weeks of strategic planning into hours through intelligent 
            human-AI collaboration. Generate comprehensive PRDs with 
            <strong>&lt;2% hallucination rate</strong> through GraphRAG validation.
          </p>
          
          <div class="mt-10 flex items-center justify-center gap-x-6">
            <UButton
              size="lg"
              class="btn-solid"
              @click="navigateTo('/prd/create')"
            >
              Create Your First PRD
              <UIcon name="i-heroicons-arrow-right" class="ml-2 h-4 w-4" />
            </UButton>
            
            <UButton
              variant="ghost"
              size="lg"
              @click="navigateTo('/dashboard')"
            >
              View Dashboard
            </UButton>
          </div>
        </div>
      </div>
    </section>

    <!-- Key Features -->
    <section class="section-spacing bg-white dark:bg-gray-900">
      <div class="page-container">
        <div class="mx-auto max-w-2xl text-center">
          <h2 class="text-3xl font-bold">
            Why Choose Strategic Planning Platform?
          </h2>
          <p class="mt-4 text-lg text-gray-600 dark:text-gray-300">
            Built with cutting-edge AI technology for enterprise-grade results
          </p>
        </div>
        
        <div class="mx-auto mt-16 max-w-7xl">
          <div class="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-3">
            <div
              v-for="feature in features"
              :key="feature.title"
              class="card p-8 hover:shadow-lg transition-shadow duration-300"
            >
              <div class="flex h-12 w-12 items-center justify-center rounded-lg bg-indigo-100 dark:bg-indigo-900/30">
                <UIcon :name="feature.icon" class="h-6 w-6 text-indigo-600 dark:text-indigo-400" />
              </div>
              
              <h3 class="mt-6 text-lg font-semibold">
                {{ feature.title }}
              </h3>
              
              <p class="mt-2 text-gray-600 dark:text-gray-300">
                {{ feature.description }}
              </p>
              
              <div class="mt-4">
                <span class="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-indigo-100 text-indigo-800 dark:bg-indigo-900/30 dark:text-indigo-300">
                  {{ feature.metric }}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- How It Works -->
    <section class="section-spacing bg-gray-50 dark:bg-gray-800">
      <div class="page-container">
        <div class="mx-auto max-w-2xl text-center">
          <h2 class="text-3xl font-bold">
            4-Phase PRD Creation Workflow
          </h2>
          <p class="mt-4 text-lg text-gray-600 dark:text-gray-300">
            Guided AI collaboration from concept to comprehensive document
          </p>
        </div>
        
        <div class="mx-auto mt-16 max-w-6xl">
          <div class="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-4">
            <div
              v-for="(phase, index) in phases"
              :key="phase.title"
              class="text-center"
            >
              <div class="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-indigo-600 text-white text-xl font-bold">
                {{ index + 1 }}
              </div>
              
              <h3 class="mt-6 text-lg font-semibold">
                {{ phase.title }}
              </h3>
              
              <p class="mt-2 text-sm text-gray-600 dark:text-gray-300">
                {{ phase.description }}
              </p>
            </div>
          </div>
        </div>
        
        <div class="mt-12 text-center">
          <UButton
            size="lg" 
            @click="navigateTo('/prd/create')"
          >
            Start Creating PRD
          </UButton>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
// Strategic Planning Platform - Landing Page

// Page meta
definePageMeta({
  title: 'Home',
  description: 'Transform weeks of strategic planning into hours with AI-powered PRD generation'
})

// Features data
const features = [
  {
    title: 'Hallucination Prevention',
    description: 'Industry-leading GraphRAG validation ensures <2% false positive rate with three-tier verification.',
    icon: 'i-heroicons-shield-check',
    metric: '<2% Error Rate'
  },
  {
    title: 'Lightning Fast',
    description: 'Generate comprehensive PRDs in under 10 minutes with sub-200ms API response times.',
    icon: 'i-heroicons-bolt',
    metric: '<200ms Response'
  },
  {
    title: 'Enterprise Scale',
    description: 'Built for 100+ concurrent users with 99.9% uptime SLA and robust security.',
    icon: 'i-heroicons-building-office',
    metric: '99.9% Uptime'
  },
  {
    title: 'Quality Assurance',
    description: '90% stakeholder satisfaction through AI-human collaboration and iterative refinement.',
    icon: 'i-heroicons-star',
    metric: '90% Satisfaction'
  },
  {
    title: 'Time Reduction',
    description: 'Transform weeks of manual planning into hours of intelligent AI-assisted creation.',
    icon: 'i-heroicons-clock',
    metric: '80% Time Saved'
  },
  {
    title: 'Smart Integration',
    description: 'Seamless GitHub integration with automated project setup and task generation.',
    icon: 'i-heroicons-puzzle-piece',
    metric: 'Zero Setup'
  }
]

// Workflow phases
const phases = [
  {
    title: 'Project Invitation',
    description: 'Describe your project idea in natural language with AI-powered similar project suggestions.'
  },
  {
    title: 'Objective Clarification', 
    description: 'Answer 3-5 targeted questions generated by AI to enhance project understanding.'
  },
  {
    title: 'Section Co-Creation',
    description: 'Systematically build comprehensive sections with real-time GraphRAG validation.'
  },
  {
    title: 'Synthesis & Export',
    description: 'Generate complete document with multiple export formats and next actions.'
  }
]

// Page head configuration
useHead({
  title: 'AI-Powered Strategic Planning Platform',
  meta: [
    {
      name: 'description',
      content: 'Transform weeks of strategic planning into hours with AI-powered PRD generation and GraphRAG validation. <2% hallucination rate, 80% time reduction.'
    }
  ]
})
</script>