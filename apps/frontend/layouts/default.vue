<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <!-- Navigation Header -->
    <header
      class="sticky top-0 z-50 bg-white/80 backdrop-blur-lg dark:bg-gray-900/80 border-b border-gray-200 dark:border-gray-800"
    >
      <nav class="page-container">
        <div class="flex h-16 items-center justify-between">
          <!-- Logo and Brand -->
          <div class="flex items-center">
            <NuxtLink to="/" class="flex items-center space-x-2">
              <div
                class="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600"
              >
                <UIcon
                  name="i-heroicons-lightning-bolt"
                  class="h-5 w-5 text-white"
                />
              </div>
              <span class="text-xl font-bold text-gray-900 dark:text-white">
                Strategic Planning
              </span>
            </NuxtLink>
          </div>

          <!-- Desktop Navigation -->
          <div class="hidden md:block">
            <div class="ml-10 flex items-baseline space-x-4">
              <NuxtLink
                v-for="item in navigation"
                :key="item.name"
                :to="item.href"
                class="px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200"
                :class="[
                  $route.path === item.href
                    ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100 dark:text-gray-300 dark:hover:text-white dark:hover:bg-gray-800',
                ]"
              >
                {{ item.name }}
              </NuxtLink>
            </div>
          </div>

          <!-- Right side actions -->
          <div class="flex items-center space-x-4">
            <!-- Theme Toggle -->
            <UButton variant="ghost" size="sm" square @click="toggleColorMode">
              <UIcon
                :name="
                  $colorMode.value === 'dark'
                    ? 'i-heroicons-sun'
                    : 'i-heroicons-moon'
                "
                class="h-5 w-5"
              />
            </UButton>

            <!-- User Menu or Auth -->
            <div v-if="isAuthenticated" class="relative">
              <UDropdown :items="userMenuItems">
                <UAvatar
                  :src="user?.avatar"
                  :alt="user?.name"
                  size="sm"
                  class="cursor-pointer"
                />
              </UDropdown>
            </div>

            <div v-else class="flex items-center space-x-2">
              <UButton
                variant="ghost"
                size="sm"
                @click="navigateTo('/auth/login')"
              >
                Sign In
              </UButton>

              <UButton size="sm" @click="navigateTo('/auth/register')">
                Get Started
              </UButton>
            </div>

            <!-- Mobile menu button -->
            <UButton
              variant="ghost"
              size="sm"
              square
              class="md:hidden"
              @click="mobileMenuOpen = !mobileMenuOpen"
            >
              <UIcon name="i-heroicons-bars-3" class="h-5 w-5" />
            </UButton>
          </div>
        </div>

        <!-- Mobile Navigation -->
        <div v-show="mobileMenuOpen" class="md:hidden">
          <div
            class="px-2 pt-2 pb-3 space-y-1 sm:px-3 border-t border-gray-200 dark:border-gray-700"
          >
            <NuxtLink
              v-for="item in navigation"
              :key="item.name"
              :to="item.href"
              class="block px-3 py-2 rounded-md text-base font-medium transition-colors duration-200"
              :class="[
                $route.path === item.href
                  ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100 dark:text-gray-300 dark:hover:text-white dark:hover:bg-gray-800',
              ]"
              @click="mobileMenuOpen = false"
            >
              {{ item.name }}
            </NuxtLink>
          </div>
        </div>
      </nav>
    </header>

    <!-- Main Content -->
    <main>
      <slot />
    </main>

    <!-- Footer -->
    <footer
      class="bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800"
    >
      <div class="page-container py-12">
        <div class="grid grid-cols-1 md:grid-cols-4 gap-8">
          <!-- Brand Section -->
          <div class="col-span-1 md:col-span-2">
            <div class="flex items-center space-x-2 mb-4">
              <div
                class="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600"
              >
                <UIcon
                  name="i-heroicons-lightning-bolt"
                  class="h-5 w-5 text-white"
                />
              </div>
              <span class="text-xl font-bold text-gray-900 dark:text-white">
                Strategic Planning
              </span>
            </div>
            <p class="text-gray-600 dark:text-gray-300 max-w-md">
              Transform weeks of strategic planning into hours with AI-powered
              PRD generation and GraphRAG validation. Built for enterprise scale
              and reliability.
            </p>
          </div>

          <!-- Quick Links -->
          <div>
            <h3 class="font-semibold text-gray-900 dark:text-white mb-4">
              Product
            </h3>
            <ul class="space-y-2">
              <li>
                <NuxtLink
                  to="/prd/create"
                  class="text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white"
                >
                  Create PRD
                </NuxtLink>
              </li>
              <li>
                <NuxtLink
                  to="/dashboard"
                  class="text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white"
                >
                  Dashboard
                </NuxtLink>
              </li>
              <li>
                <NuxtLink
                  to="/templates"
                  class="text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white"
                >
                  Templates
                </NuxtLink>
              </li>
            </ul>
          </div>

          <!-- Support -->
          <div>
            <h3 class="font-semibold text-gray-900 dark:text-white mb-4">
              Support
            </h3>
            <ul class="space-y-2">
              <li>
                <NuxtLink
                  to="/docs"
                  class="text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white"
                >
                  Documentation
                </NuxtLink>
              </li>
              <li>
                <NuxtLink
                  to="/help"
                  class="text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white"
                >
                  Help Center
                </NuxtLink>
              </li>
              <li>
                <NuxtLink
                  to="/contact"
                  class="text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white"
                >
                  Contact Us
                </NuxtLink>
              </li>
            </ul>
          </div>
        </div>

        <div class="mt-8 pt-8 border-t border-gray-200 dark:border-gray-700">
          <p class="text-center text-gray-600 dark:text-gray-300">
            © {{ currentYear }} Strategic Planning Platform. Built with AI for
            the future of project management.
          </p>
        </div>
      </div>
    </footer>
  </div>
</template>

<script setup lang="ts">
// Strategic Planning Platform - Default Layout

// Reactive data
const mobileMenuOpen = ref(false);
const currentYear = new Date().getFullYear();

// Navigation items
const navigation = [
  { name: 'Dashboard', href: '/dashboard' },
  { name: 'Create PRD', href: '/prd/create' },
  { name: 'Templates', href: '/templates' },
  { name: 'Analytics', href: '/analytics' },
];

// Mock authentication state (replace with real auth)
const isAuthenticated = ref(false);
const user = ref(null);

// User menu items
const userMenuItems = [
  [
    {
      label: 'Profile',
      icon: 'i-heroicons-user',
      click: () => navigateTo('/profile'),
    },
    {
      label: 'Settings',
      icon: 'i-heroicons-cog-6-tooth',
      click: () => navigateTo('/settings'),
    },
  ],
  [
    {
      label: 'Sign Out',
      icon: 'i-heroicons-arrow-left-on-rectangle',
      click: () => signOut(),
    },
  ],
];

// Color mode toggle
const colorMode = useColorMode();

const toggleColorMode = () => {
  colorMode.preference = colorMode.value === 'dark' ? 'light' : 'dark';
};

// Sign out function
const signOut = () => {
  // Implement sign out logic
  isAuthenticated.value = false;
  user.value = null;
  navigateTo('/');
};

// Close mobile menu when route changes
watch(
  () => route.path,
  () => {
    mobileMenuOpen.value = false;
  }
);

// Get current route
const route = useRoute();
</script>
