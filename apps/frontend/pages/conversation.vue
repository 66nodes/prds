<template>
  <div class="conversation-page min-h-screen bg-gray-50 dark:bg-gray-900">
    <div class="container mx-auto px-4 py-8 max-w-6xl">
      <!-- Page Header -->
      <div class="mb-8">
        <h1 class="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Conversational Planning
        </h1>
        <p class="text-gray-600 dark:text-gray-400">
          Collaborate with AI to create comprehensive Product Requirements Documents through natural conversation.
        </p>
      </div>

      <!-- Project Selection -->
      <div v-if="!selectedProject" class="mb-8">
        <UCard>
          <template #header>
            <h2 class="text-xl font-semibold">Select a Project</h2>
          </template>
          
          <div v-if="loadingProjects" class="flex justify-center py-8">
            <UIcon name="i-heroicons-arrow-path" class="w-8 h-8 animate-spin text-primary" />
          </div>
          
          <div v-else-if="projects.length === 0" class="text-center py-8">
            <UIcon name="i-heroicons-folder-plus" class="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <p class="text-gray-500 mb-4">No projects found. Create a project first.</p>
            <UButton @click="navigateTo('/projects')">
              Create Project
            </UButton>
          </div>
          
          <div v-else class="space-y-4">
            <div
              v-for="project in projects"
              :key="project.id"
              @click="selectProject(project)"
              class="p-4 border border-gray-200 dark:border-gray-700 rounded-lg cursor-pointer hover:border-primary transition-colors"
            >
              <h3 class="font-semibold text-gray-900 dark:text-white">{{ project.name }}</h3>
              <p class="text-sm text-gray-600 dark:text-gray-400 mt-1">{{ project.description }}</p>
              <div class="flex items-center space-x-4 mt-2">
                <UBadge :color="getStatusColor(project.status)" size="xs">
                  {{ project.status }}
                </UBadge>
                <span class="text-xs text-gray-500">
                  {{ project.prds?.length || 0 }} PRDs
                </span>
              </div>
            </div>
          </div>
        </UCard>
      </div>

      <!-- Conversation Interface -->
      <div v-else class="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <!-- Main Conversation Panel -->
        <div class="lg:col-span-3">
          <UCard class="h-full">
            <ConversationInterface :project-id="selectedProject.id" />
          </UCard>
        </div>

        <!-- Sidebar with Project Info and History -->
        <div class="lg:col-span-1 space-y-6">
          <!-- Selected Project Info -->
          <UCard>
            <template #header>
              <div class="flex items-center justify-between">
                <h3 class="text-lg font-semibold">Current Project</h3>
                <UButton
                  variant="ghost"
                  size="xs"
                  icon="i-heroicons-x-mark"
                  @click="selectedProject = null"
                />
              </div>
            </template>
            
            <div class="space-y-3">
              <div>
                <h4 class="font-medium text-gray-900 dark:text-white">{{ selectedProject.name }}</h4>
                <p class="text-sm text-gray-600 dark:text-gray-400">{{ selectedProject.description }}</p>
              </div>
              
              <div class="flex items-center space-x-2">
                <UBadge :color="getStatusColor(selectedProject.status)" size="xs">
                  {{ selectedProject.status }}
                </UBadge>
                <span class="text-xs text-gray-500">
                  {{ selectedProject.prds?.length || 0 }} PRDs
                </span>
              </div>
              
              <div class="pt-2 border-t border-gray-200 dark:border-gray-700">
                <UButton
                  variant="outline"
                  size="xs"
                  block
                  @click="navigateTo(`/projects/${selectedProject.id}`)"
                >
                  View Project Details
                </UButton>
              </div>
            </div>
          </UCard>

          <!-- Conversation History -->
          <UCard>
            <template #header>
              <h3 class="text-lg font-semibold">Recent Conversations</h3>
            </template>
            
            <div v-if="conversationHistory.length === 0" class="text-center py-4">
              <UIcon name="i-heroicons-chat-bubble-left-right" class="w-8 h-8 text-gray-400 mx-auto mb-2" />
              <p class="text-sm text-gray-500">No conversations yet</p>
            </div>
            
            <div v-else class="space-y-3">
              <div
                v-for="conv in conversationHistory.slice(0, 5)"
                :key="conv.id"
                @click="loadConversation(conv.id)"
                class="p-3 border border-gray-200 dark:border-gray-700 rounded-md cursor-pointer hover:border-primary transition-colors"
              >
                <h4 class="text-sm font-medium text-gray-900 dark:text-white truncate">
                  {{ conv.title }}
                </h4>
                <p class="text-xs text-gray-500 mt-1">
                  {{ formatDate(conv.updatedAt) }}
                </p>
                <p class="text-xs text-gray-600 dark:text-gray-400 mt-1 truncate">
                  {{ conv.lastMessage }}
                </p>
              </div>
            </div>
          </UCard>

          <!-- Human Validation Status -->
          <UCard v-if="activeValidations.length > 0">
            <template #header>
              <div class="flex items-center space-x-2">
                <UIcon name="i-heroicons-exclamation-triangle" class="w-5 h-5 text-amber-500" />
                <h3 class="text-lg font-semibold">Pending Validations</h3>
              </div>
            </template>
            
            <div class="space-y-3">
              <div
                v-for="validation in activeValidations"
                :key="validation.id"
                class="p-3 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-md"
              >
                <h4 class="text-sm font-medium text-amber-900 dark:text-amber-100">
                  Validation Required
                </h4>
                <p class="text-xs text-amber-700 dark:text-amber-300 mt-1">
                  {{ validation.prompt.question }}
                </p>
                <div class="mt-2">
                  <UButton
                    size="xs"
                    @click="scrollToValidation(validation.id)"
                  >
                    Review & Respond
                  </UButton>
                </div>
              </div>
            </div>
          </UCard>

          <!-- Tips & Help -->
          <UCard>
            <template #header>
              <div class="flex items-center space-x-2">
                <UIcon name="i-heroicons-light-bulb" class="w-5 h-5 text-yellow-500" />
                <h3 class="text-lg font-semibold">Tips</h3>
              </div>
            </template>
            
            <div class="space-y-3 text-sm text-gray-600 dark:text-gray-400">
              <div class="flex items-start space-x-2">
                <UIcon name="i-heroicons-check-circle" class="w-4 h-4 text-green-500 mt-0.5" />
                <p>Be specific about your requirements and constraints</p>
              </div>
              <div class="flex items-start space-x-2">
                <UIcon name="i-heroicons-check-circle" class="w-4 h-4 text-green-500 mt-0.5" />
                <p>Ask follow-up questions to clarify details</p>
              </div>
              <div class="flex items-start space-x-2">
                <UIcon name="i-heroicons-check-circle" class="w-4 h-4 text-green-500 mt-0.5" />
                <p>Review extracted requirements as you go</p>
              </div>
              <div class="flex items-start space-x-2">
                <UIcon name="i-heroicons-check-circle" class="w-4 h-4 text-green-500 mt-0.5" />
                <p>Approve or reject AI suggestions when prompted</p>
              </div>
            </div>
          </UCard>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { format } from 'date-fns';
import type { Project } from '~/types';

// Meta
definePageMeta({
  title: 'Conversational Planning',
  description: 'Create PRDs through AI-powered conversation',
  middleware: ['auth']
});

// Store references
const conversation = useConversationStore();
const { $api } = useApiClient();
const toast = useToast();

// Component state
const selectedProject = ref<Project | null>(null);
const projects = ref<Project[]>([]);
const loadingProjects = ref(true);
const activeValidations = ref([]);

// Computed
const conversationHistory = computed(() => conversation.conversationHistory || []);

// Methods
const loadProjects = async () => {
  try {
    loadingProjects.value = true;
    const response = await $api.get<Project[]>('/projects');
    projects.value = response.data || [];
  } catch (error) {
    console.error('Failed to load projects:', error);
    toast.add({
      title: 'Error',
      description: 'Failed to load projects',
      color: 'red'
    });
  } finally {
    loadingProjects.value = false;
  }
};

const selectProject = async (project: Project) => {
  selectedProject.value = project;
  
  // Load active validations for the project
  await loadActiveValidations();
  
  toast.add({
    title: 'Project Selected',
    description: `Starting conversation for ${project.name}`,
    color: 'green'
  });
};

const loadActiveValidations = async () => {
  try {
    const response = await $api.get('/human-validation/active');
    activeValidations.value = response.data || [];
  } catch (error) {
    console.error('Failed to load active validations:', error);
  }
};

const loadConversation = async (conversationId: string) => {
  try {
    await conversation.loadConversation(conversationId);
    toast.add({
      title: 'Conversation Loaded',
      description: 'Previous conversation loaded successfully',
      color: 'green'
    });
  } catch (error) {
    console.error('Failed to load conversation:', error);
    toast.add({
      title: 'Error',
      description: 'Failed to load conversation',
      color: 'red'
    });
  }
};

const scrollToValidation = (validationId: string) => {
  // Scroll to the validation prompt in the conversation interface
  const validationElement = document.querySelector(`[data-validation-id="${validationId}"]`);
  if (validationElement) {
    validationElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
};

const getStatusColor = (status: string): string => {
  switch (status.toLowerCase()) {
    case 'planning': return 'blue';
    case 'in_progress': return 'yellow';
    case 'review': return 'orange';
    case 'completed': return 'green';
    case 'on_hold': return 'gray';
    default: return 'gray';
  }
};

const formatDate = (dateString: string): string => {
  return format(new Date(dateString), 'MMM d, HH:mm');
};

// Lifecycle
onMounted(async () => {
  await loadProjects();
  
  // Check if there's a project ID in the route query
  const route = useRoute();
  if (route.query.projectId && projects.value.length > 0) {
    const project = projects.value.find(p => p.id === route.query.projectId);
    if (project) {
      await selectProject(project);
    }
  }
});

// Watch for validation updates
watch(
  () => conversation.requiresValidation,
  async (requiresValidation) => {
    if (requiresValidation) {
      await loadActiveValidations();
    }
  }
);
</script>

<style scoped>
.conversation-page {
  min-height: calc(100vh - 64px); /* Account for navigation */
}
</style>