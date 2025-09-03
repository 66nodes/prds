<template>
  <div class="bg-white dark:bg-gray-800 shadow rounded-lg">
    <div class="px-4 py-5 sm:p-6">
      <h3 class="text-lg leading-6 font-medium text-gray-900 dark:text-white">
        Product Requirements Documents
      </h3>
      <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
        View and manage all PRDs across your projects
      </p>
      
      <div class="mt-6">
        <div class="grid gap-4 sm:grid-cols-1 lg:grid-cols-2">
          <div v-for="prd in prds" :key="prd.id" class="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
            <div class="flex items-center justify-between">
              <h4 class="text-sm font-medium text-gray-900 dark:text-white">{{ prd.title }}</h4>
              <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium"
                    :class="getStatusClass(prd.status)">
                {{ prd.status }}
              </span>
            </div>
            <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">{{ prd.description }}</p>
            <div class="mt-3 flex items-center justify-between">
              <div class="flex items-center text-sm text-gray-500 dark:text-gray-400">
                <svg class="mr-1.5 h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M6 2a1 1 0 00-1 1v1H4a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2h-1V3a1 1 0 10-2 0v1H7V3a1 1 0 00-1-1zm0 5a1 1 0 000 2h8a1 1 0 100-2H6z" clip-rule="evenodd" />
                </svg>
                {{ formatDate(prd.updatedAt) }}
              </div>
              <div class="text-sm">
                <span class="text-green-600 dark:text-green-400">{{ (prd.validationScore * 100).toFixed(0) }}% valid</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const prds = ref([
  {
    id: '1',
    title: 'User Authentication System',
    description: 'Secure login and registration functionality',
    status: 'published',
    validationScore: 0.98,
    updatedAt: '2024-01-15T10:30:00Z'
  },
  {
    id: '2',
    title: 'Shopping Cart Feature',
    description: 'Add to cart and checkout process',
    status: 'in_review',
    validationScore: 0.92,
    updatedAt: '2024-01-14T15:45:00Z'
  },
  {
    id: '3',
    title: 'Payment Integration',
    description: 'Stripe and PayPal payment processing',
    status: 'draft',
    validationScore: 0.85,
    updatedAt: '2024-01-13T09:15:00Z'
  },
  {
    id: '4',
    title: 'Search Functionality',
    description: 'Product search and filtering',
    status: 'published',
    validationScore: 0.95,
    updatedAt: '2024-01-12T14:20:00Z'
  }
])

const getStatusClass = (status: string) => {
  switch (status) {
    case 'published':
      return 'bg-green-100 text-green-800 dark:bg-green-800 dark:text-green-100'
    case 'in_review':
      return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-800 dark:text-yellow-100'
    case 'draft':
      return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-100'
    case 'approved':
      return 'bg-blue-100 text-blue-800 dark:bg-blue-800 dark:text-blue-100'
    default:
      return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-100'
  }
}

const formatDate = (dateString: string) => {
  return new Date(dateString).toLocaleDateString()
}
</script>