<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <!-- Navigation Header -->
    <header class="bg-white dark:bg-gray-800 shadow">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between items-center py-6">
          <div class="flex items-center">
            <h1 class="text-2xl font-bold text-gray-900 dark:text-white">
              Dashboard
            </h1>
          </div>

          <div class="flex items-center space-x-4">
            <!-- Notifications -->
            <button
              type="button"
              class="p-2 rounded-md text-gray-400 hover:text-gray-500 dark:hover:text-gray-300"
              @click="toggleNotifications"
            >
              <span class="sr-only">View notifications</span>
              <svg
                class="h-6 w-6"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
                />
              </svg>
              <span
                v-if="unreadNotifications > 0"
                class="absolute -mt-5 -mr-2 h-2 w-2 rounded-full bg-red-500"
              />
            </button>

            <!-- User Menu -->
            <div class="relative">
              <button
                type="button"
                class="flex items-center text-sm rounded-full focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                @click="toggleUserMenu"
              >
                <span class="sr-only">Open user menu</span>
                <div
                  class="h-8 w-8 rounded-full bg-indigo-600 flex items-center justify-center"
                >
                  <span class="text-white font-medium">
                    {{ userInitials }}
                  </span>
                </div>
              </button>

              <!-- User Dropdown -->
              <transition
                enter-active-class="transition ease-out duration-100"
                enter-from-class="transform opacity-0 scale-95"
                enter-to-class="transform opacity-100 scale-100"
                leave-active-class="transition ease-in duration-75"
                leave-from-class="transform opacity-100 scale-100"
                leave-to-class="transform opacity-0 scale-95"
              >
                <div
                  v-if="userMenuOpen"
                  class="origin-top-right absolute right-0 mt-2 w-48 rounded-md shadow-lg py-1 bg-white dark:bg-gray-700 ring-1 ring-black ring-opacity-5 focus:outline-none"
                >
                  <NuxtLink
                    to="/profile"
                    class="block px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-600"
                    @click="userMenuOpen = false"
                  >
                    Your Profile
                  </NuxtLink>
                  <NuxtLink
                    to="/settings"
                    class="block px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-600"
                    @click="userMenuOpen = false"
                  >
                    Settings
                  </NuxtLink>
                  <hr class="my-1 border-gray-200 dark:border-gray-600" />
                  <button
                    type="button"
                    class="block w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-600"
                    @click="handleLogout"
                  >
                    Sign out
                  </button>
                </div>
              </transition>
            </div>
          </div>
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
      <!-- Tab Navigation -->
      <div class="border-b border-gray-200 dark:border-gray-700 mb-6">
        <nav class="-mb-px flex space-x-8" aria-label="Tabs">
          <button
            v-for="tab in tabs"
            :key="tab.id"
            @click="activeTab = tab.id"
            :class="[
              activeTab === tab.id
                ? 'border-indigo-500 text-indigo-600 dark:text-indigo-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300',
              'whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm',
            ]"
          >
            {{ tab.name }}
            <span
              v-if="tab.count"
              :class="[
                activeTab === tab.id
                  ? 'bg-indigo-100 text-indigo-600 dark:bg-indigo-900 dark:text-indigo-400'
                  : 'bg-gray-100 text-gray-900 dark:bg-gray-700 dark:text-gray-300',
                'ml-2 py-0.5 px-2.5 rounded-full text-xs font-medium',
              ]"
            >
              {{ tab.count }}
            </span>
          </button>
        </nav>
      </div>

      <!-- Tab Content -->
      <div class="px-4 py-6 sm:px-0">
        <!-- Overview Tab -->
        <div v-if="activeTab === 'overview'">
          <DashboardOverview />
        </div>

        <!-- Projects Tab -->
        <div v-if="activeTab === 'projects'">
          <ProjectsList />
        </div>

        <!-- PRDs Tab -->
        <div v-if="activeTab === 'prds'">
          <PRDsList />
        </div>

        <!-- Agents Tab -->
        <div v-if="activeTab === 'agents'">
          <AgentsList />
        </div>

        <!-- Analytics Tab -->
        <div v-if="activeTab === 'analytics'">
          <AnalyticsDashboard />
        </div>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import { useAuthStore } from '~/stores/auth';
import DashboardOverview from '~/components/dashboard/DashboardOverview.vue';
import ProjectsList from '~/components/projects/ProjectsList.vue';
import PRDsList from '~/components/prd/PRDsList.vue';
import AgentsList from '~/components/agents/AgentsList.vue';
import AnalyticsDashboard from '~/components/analytics/AnalyticsDashboard.vue';

// Middleware
definePageMeta({
  middleware: 'auth',
});

const authStore = useAuthStore();
const router = useRouter();

// State
const activeTab = ref('overview');
const userMenuOpen = ref(false);
const notificationsOpen = ref(false);
const unreadNotifications = ref(3);

// Tabs configuration
const tabs = ref([
  { id: 'overview', name: 'Overview' },
  { id: 'projects', name: 'Projects', count: 12 },
  { id: 'prds', name: 'PRDs', count: 28 },
  { id: 'agents', name: 'Agents', count: 5 },
  { id: 'analytics', name: 'Analytics' },
]);

// Computed
const userInitials = computed(() => {
  const user = authStore.currentUser;
  if (!user) return 'U';

  const names = user.name.split(' ');
  if (names.length >= 2) {
    return `${names[0][0]}${names[1][0]}`.toUpperCase();
  }
  return user.name.substring(0, 2).toUpperCase();
});

// Methods
const toggleUserMenu = () => {
  userMenuOpen.value = !userMenuOpen.value;
  notificationsOpen.value = false;
};

const toggleNotifications = () => {
  notificationsOpen.value = !notificationsOpen.value;
  userMenuOpen.value = false;
};

const handleLogout = async () => {
  await authStore.logout();
  await router.push('/login');
};

// Close dropdowns when clicking outside
onMounted(() => {
  document.addEventListener('click', e => {
    const target = e.target as HTMLElement;
    if (!target.closest('.relative')) {
      userMenuOpen.value = false;
      notificationsOpen.value = false;
    }
  });
});
</script>
