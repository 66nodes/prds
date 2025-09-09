<template>
  <div class="container mx-auto p-8">
    <h1 class="text-3xl font-bold mb-8">Pinia Persistence Test</h1>
    
    <!-- Auth Store Test -->
    <section class="mb-8 p-6 bg-white rounded-lg shadow">
      <h2 class="text-2xl font-semibold mb-4">Auth Store (sessionStorage)</h2>
      
      <div class="mb-4">
        <p class="text-sm text-gray-600">Current User:</p>
        <pre class="bg-gray-100 p-2 rounded">{{ authStore.user || 'No user' }}</pre>
      </div>
      
      <div class="mb-4">
        <p class="text-sm text-gray-600">Is Authenticated: {{ authStore.isAuthenticated }}</p>
      </div>
      
      <div class="flex gap-2">
        <button 
          @click="setMockUser"
          class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Set Mock User
        </button>
        <button 
          @click="clearAuthStore"
          class="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
        >
          Clear Auth
        </button>
      </div>
    </section>
    
    <!-- PRD Store Test -->
    <section class="mb-8 p-6 bg-white rounded-lg shadow">
      <h2 class="text-2xl font-semibold mb-4">PRD Store (localStorage)</h2>
      
      <div class="mb-4">
        <p class="text-sm text-gray-600">PRDs Count: {{ prdStore.prds.length }}</p>
        <p class="text-sm text-gray-600">Current PRD: {{ prdStore.currentPRD?.title || 'None' }}</p>
      </div>
      
      <div class="mb-4">
        <h3 class="font-semibold">PRDs List:</h3>
        <ul class="list-disc pl-6">
          <li v-for="prd in prdStore.prds" :key="prd.id">
            {{ prd.title }} ({{ prd.metadata.status }})
          </li>
        </ul>
      </div>
      
      <div class="flex gap-2">
        <button 
          @click="addMockPRD"
          class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Add Mock PRD
        </button>
        <button 
          @click="clearPRDStore"
          class="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
        >
          Clear PRDs
        </button>
      </div>
    </section>
    
    <!-- Projects Store Test -->
    <section class="mb-8 p-6 bg-white rounded-lg shadow">
      <h2 class="text-2xl font-semibold mb-4">Projects Store (localStorage)</h2>
      
      <div class="mb-4">
        <p class="text-sm text-gray-600">Projects Count: {{ projectsStore.projects.length }}</p>
        <p class="text-sm text-gray-600">Current Project: {{ projectsStore.currentProject?.name || 'None' }}</p>
        <p class="text-sm text-gray-600">Sort By: {{ projectsStore.sortBy }} ({{ projectsStore.sortOrder }})</p>
      </div>
      
      <div class="mb-4">
        <h3 class="font-semibold">Projects List:</h3>
        <ul class="list-disc pl-6">
          <li v-for="project in projectsStore.projects" :key="project.id">
            {{ project.name }} ({{ project.status }})
          </li>
        </ul>
      </div>
      
      <div class="mb-4">
        <h3 class="font-semibold">Filters:</h3>
        <pre class="bg-gray-100 p-2 rounded text-sm">{{ projectsStore.filters }}</pre>
      </div>
      
      <div class="flex gap-2 mb-4">
        <button 
          @click="addMockProject"
          class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Add Mock Project
        </button>
        <button 
          @click="toggleSort"
          class="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
        >
          Toggle Sort
        </button>
        <button 
          @click="clearProjectsStore"
          class="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
        >
          Clear Projects
        </button>
      </div>
    </section>
    
    <!-- Storage Inspection -->
    <section class="mb-8 p-6 bg-white rounded-lg shadow">
      <h2 class="text-2xl font-semibold mb-4">Storage Inspection</h2>
      
      <div class="mb-4">
        <h3 class="font-semibold">SessionStorage Keys:</h3>
        <ul class="list-disc pl-6">
          <li v-for="key in sessionStorageKeys" :key="key">
            {{ key }}
            <button 
              @click="viewStorageItem('session', key)"
              class="ml-2 text-blue-500 hover:underline"
            >
              View
            </button>
          </li>
        </ul>
      </div>
      
      <div class="mb-4">
        <h3 class="font-semibold">LocalStorage Keys:</h3>
        <ul class="list-disc pl-6">
          <li v-for="key in localStorageKeys" :key="key">
            {{ key }}
            <button 
              @click="viewStorageItem('local', key)"
              class="ml-2 text-blue-500 hover:underline"
            >
              View
            </button>
          </li>
        </ul>
      </div>
      
      <div v-if="selectedStorageItem" class="mt-4">
        <h3 class="font-semibold">{{ selectedStorageItem.key }} Content:</h3>
        <pre class="bg-gray-100 p-2 rounded text-xs overflow-x-auto">{{ selectedStorageItem.value }}</pre>
      </div>
      
      <div class="mt-4 flex gap-2">
        <button 
          @click="refreshStorageKeys"
          class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Refresh Storage Keys
        </button>
        <button 
          @click="clearAllStorage"
          class="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
        >
          Clear All Storage
        </button>
      </div>
    </section>
    
    <div class="p-6 bg-yellow-50 rounded-lg">
      <h2 class="text-lg font-semibold mb-2">Test Instructions:</h2>
      <ol class="list-decimal pl-6 space-y-2">
        <li>Add some test data using the buttons above</li>
        <li>Note the values in each store</li>
        <li>Refresh the page (F5 or browser refresh)</li>
        <li>Check if the data persists:
          <ul class="list-disc pl-6 mt-1">
            <li>Auth store should clear (sessionStorage)</li>
            <li>PRD store should persist (localStorage)</li>
            <li>Projects store should persist (localStorage)</li>
          </ul>
        </li>
        <li>Close the browser tab and reopen to test sessionStorage clearing</li>
      </ol>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useAuthStore } from '~/stores/auth';
import { usePRDStore } from '~/stores/prd';
import { useProjectsStore, ProjectStatus, ProjectPriority } from '~/stores/projects';

// Store instances
const authStore = useAuthStore();
const prdStore = usePRDStore();
const projectsStore = useProjectsStore();

// Storage keys
const sessionStorageKeys = ref<string[]>([]);
const localStorageKeys = ref<string[]>([]);
const selectedStorageItem = ref<{ key: string; value: string } | null>(null);

// Mock data counter
let mockCounter = 0;

// Auth store methods
const setMockUser = () => {
  authStore.setUser({
    id: '123',
    email: 'testexample.com',
    name: 'Test User',
    role: 'admin',
    avatar: null,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  });
};

const clearAuthStore = () => {
  authStore.reset();
};

// PRD store methods
const addMockPRD = () => {
  mockCounter++;
  prdStore.addPRD({
    id: `prd-${mockCounter}`,
    title: `Mock PRD ${mockCounter}`,
    description: `This is a mock PRD #${mockCounter}`,
    project_id: 'project-1',
    content: {
      executive_summary: 'Test summary',
      problem_statement: 'Test problem',
      goals_objectives: 'Test goals',
      target_audience: 'Test audience',
      user_stories: [],
      functional_requirements: [],
      non_functional_requirements: [],
      technical_requirements: [],
      success_metrics: [],
      timeline: {},
      risks: [],
      dependencies: [],
    },
    metadata: {
      version: '1.0',
      status: 'draft',
      author: 'Test User',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      tags: ['test', 'mock'],
      category: 'test',
      priority: 'medium',
      stakeholders: [],
      approval_status: 'pending',
      approved_by: null,
      approved_at: null,
    },
    hallucination_rate: 0.01,
    validation_score: 0.98,
    sources: [],
    graph_evidence: {},
    generated_by: 'test',
    review_status: 'pending',
    reviewedBy: null,
    reviewedAt: null,
    feedbackIncorporated: false,
  });
};

const clearPRDStore = () => {
  prdStore.reset();
};

// Projects store methods
const addMockProject = () => {
  mockCounter++;
  projectsStore.addProject({
    id: `project-${mockCounter}`,
    name: `Mock Project ${mockCounter}`,
    description: `This is mock project #${mockCounter}`,
    status: ProjectStatus.Active,
    priority: ProjectPriority.Medium,
    createdAt: new Date(),
    updatedAt: new Date(),
    owner: 'test-user',
    team: ['user1', 'user2'],
    tags: ['test', 'mock'],
    deadline: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000), // 30 days from now
    progress: Math.floor(Math.random() * 100),
    metadata: {
      prdCount: 3,
      taskCount: 10,
      completedTasks: 5,
      lastActivity: new Date(),
    },
  });
};

const toggleSort = () => {
  const sortOptions = ['name', 'createdAt', 'updatedAt', 'priority', 'deadline'] as const;
  const currentIndex = sortOptions.indexOf(projectsStore.sortBy as any);
  const nextIndex = (currentIndex + 1) % sortOptions.length;
  projectsStore.setSorting(sortOptions[nextIndex] as any);
};

const clearProjectsStore = () => {
  projectsStore.reset();
};

// Storage inspection methods
const refreshStorageKeys = () => {
  if (process.client) {
    // Get sessionStorage keys
    sessionStorageKeys.value = Object.keys(sessionStorage).filter(key => 
      key.startsWith('pinia')
    );
    
    // Get localStorage keys
    localStorageKeys.value = Object.keys(localStorage).filter(key => 
      key.startsWith('pinia')
    );
  }
};

const viewStorageItem = (type: 'session' | 'local', key: string) => {
  if (process.client) {
    const storage = type === 'session' ? sessionStorage : localStorage;
    const value = storage.getItem(key);
    if (value) {
      try {
        const parsed = JSON.parse(value);
        selectedStorageItem.value = {
          key,
          value: JSON.stringify(parsed, null, 2),
        };
      } catch {
        selectedStorageItem.value = { key, value };
      }
    }
  }
};

const clearAllStorage = () => {
  if (process.client) {
    // Clear Pinia-related items from sessionStorage
    Object.keys(sessionStorage)
      .filter(key => key.startsWith('pinia'))
      .forEach(key => sessionStorage.removeItem(key));
    
    // Clear Pinia-related items from localStorage
    Object.keys(localStorage)
      .filter(key => key.startsWith('pinia'))
      .forEach(key => localStorage.removeItem(key));
    
    // Reset all stores
    authStore.reset();
    prdStore.reset();
    projectsStore.reset();
    
    // Refresh keys display
    refreshStorageKeys();
    selectedStorageItem.value = null;
  }
};

// Initialize storage keys on mount
onMounted(() => {
  refreshStorageKeys();
});
</script>