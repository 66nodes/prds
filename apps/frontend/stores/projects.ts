import { defineStore } from 'pinia';

// Project status enum
export enum ProjectStatus {
  Draft = 'draft',
  Active = 'active',
  InProgress = 'in_progress',
  Review = 'review',
  Completed = 'completed',
  Archived = 'archived',
}

// Project priority enum
export enum ProjectPriority {
  Low = 'low',
  Medium = 'medium',
  High = 'high',
  Urgent = 'urgent',
}

// Project interface
export interface Project {
  id: string;
  name: string;
  description: string;
  status: ProjectStatus;
  priority: ProjectPriority;
  createdAt: Date;
  updatedAt: Date;
  owner: string;
  team: string[];
  tags: string[];
  deadline?: Date;
  progress: number;
  metadata: {
    prdCount: number;
    taskCount: number;
    completedTasks: number;
    lastActivity: Date;
  };
}

// State interface
interface ProjectsState {
  projects: Project[];
  currentProject: Project | null;
  isLoading: boolean;
  error: string | null;
  filters: {
    status: ProjectStatus | null;
    priority: ProjectPriority | null;
    tags: string[];
    searchQuery: string;
  };
  sortBy: 'name' | 'createdAt' | 'updatedAt' | 'priority' | 'deadline';
  sortOrder: 'asc' | 'desc';
}

export const useProjectsStore = defineStore('projects', {
  state: (): ProjectsState => ({
    projects: [],
    currentProject: null,
    isLoading: false,
    error: null,
    filters: {
      status: null,
      priority: null,
      tags: [],
      searchQuery: '',
    },
    sortBy: 'updatedAt',
    sortOrder: 'desc',
  }),

  getters: {
    allProjects: state => state.projects,
    
    activeProjects: state =>
      state.projects.filter(p => p.status !== ProjectStatus.Archived),
    
    filteredProjects: (state) => {
      let filtered = [...state.projects];
      
      // Apply status filter
      if (state.filters.status) {
        filtered = filtered.filter(p => p.status === state.filters.status);
      }
      
      // Apply priority filter
      if (state.filters.priority) {
        filtered = filtered.filter(p => p.priority === state.filters.priority);
      }
      
      // Apply tags filter
      if (state.filters.tags.length > 0) {
        filtered = filtered.filter(p =>
          state.filters.tags.some(tag => p.tags.includes(tag))
        );
      }
      
      // Apply search query
      if (state.filters.searchQuery) {
        const query = state.filters.searchQuery.toLowerCase();
        filtered = filtered.filter(p =>
          p.name.toLowerCase().includes(query) ||
          p.description.toLowerCase().includes(query) ||
          p.tags.some(tag => tag.toLowerCase().includes(query))
        );
      }
      
      // Apply sorting
      filtered.sort((a, b) => {
        let compareValue = 0;
        
        switch (state.sortBy) {
          case 'name':
            compareValue = a.name.localeCompare(b.name);
            break;
          case 'createdAt':
            compareValue = new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();
            break;
          case 'updatedAt':
            compareValue = new Date(a.updatedAt).getTime() - new Date(b.updatedAt).getTime();
            break;
          case 'priority':
            const priorityOrder = { urgent: 4, high: 3, medium: 2, low: 1 };
            compareValue = priorityOrder[a.priority] - priorityOrder[b.priority];
            break;
          case 'deadline':
            if (!a.deadline) return 1;
            if (!b.deadline) return -1;
            compareValue = new Date(a.deadline).getTime() - new Date(b.deadline).getTime();
            break;
        }
        
        return state.sortOrder === 'asc' ? compareValue : -compareValue;
      });
      
      return filtered;
    },
    
    projectById: state => (id: string) =>
      state.projects.find(p => p.id === id),
    
    currentProjectId: state => state.currentProject?.id,
    
    projectsByStatus: state => (status: ProjectStatus) =>
      state.projects.filter(p => p.status === status),
    
    projectsByPriority: state => (priority: ProjectPriority) =>
      state.projects.filter(p => p.priority === priority),
    
    overallProgress: (state) => {
      if (state.projects.length === 0) return 0;
      const totalProgress = state.projects.reduce((sum, p) => sum + p.progress, 0);
      return Math.round(totalProgress / state.projects.length);
    },
    
    upcomingDeadlines: (state) => {
      const now = new Date();
      const oneWeekFromNow = new Date(now.getTime() + 7 * 24 * 60 * 60 * 1000);
      
      return state.projects
        .filter(p => p.deadline && new Date(p.deadline) <= oneWeekFromNow && new Date(p.deadline) >= now)
        .sort((a, b) => new Date(a.deadline!).getTime() - new Date(b.deadline!).getTime());
    },
  },

  actions: {
    setProjects(projects: Project[]) {
      this.projects = projects;
    },
    
    setCurrentProject(project: Project | null) {
      this.currentProject = project;
    },
    
    addProject(project: Project) {
      this.projects.push(project);
    },
    
    updateProject(id: string, updates: Partial<Project>) {
      const index = this.projects.findIndex(p => p.id === id);
      if (index !== -1) {
        this.projects[index] = {
          ...this.projects[index],
          ...updates,
          updatedAt: new Date(),
        };
        
        if (this.currentProject?.id === id) {
          this.currentProject = this.projects[index];
        }
      }
    },
    
    removeProject(id: string) {
      this.projects = this.projects.filter(p => p.id !== id);
      
      if (this.currentProject?.id === id) {
        this.currentProject = null;
      }
    },
    
    setLoading(loading: boolean) {
      this.isLoading = loading;
    },
    
    setError(error: string | null) {
      this.error = error;
    },
    
    setFilter(filter: keyof ProjectsState['filters'], value: any) {
      this.filters[filter] = value;
    },
    
    clearFilters() {
      this.filters = {
        status: null,
        priority: null,
        tags: [],
        searchQuery: '',
      };
    },
    
    setSorting(sortBy: ProjectsState['sortBy'], sortOrder?: ProjectsState['sortOrder']) {
      this.sortBy = sortBy;
      if (sortOrder) {
        this.sortOrder = sortOrder;
      } else {
        // Toggle sort order if same field
        if (this.sortBy === sortBy) {
          this.sortOrder = this.sortOrder === 'asc' ? 'desc' : 'asc';
        }
      }
    },
    
    async fetchProjects() {
      this.setLoading(true);
      this.setError(null);
      
      try {
        const { get } = useApiClient();
        const projects = await get<Project[]>('/projects');
        this.setProjects(projects);
      } catch (error: any) {
        this.setError(error.message || 'Failed to fetch projects');
        throw error;
      } finally {
        this.setLoading(false);
      }
    },
    
    async fetchProject(id: string) {
      this.setLoading(true);
      this.setError(null);
      
      try {
        const { get } = useApiClient();
        const project = await get<Project>(`/projects/${id}`);
        this.setCurrentProject(project);
        
        // Update in list if exists
        const index = this.projects.findIndex(p => p.id === id);
        if (index !== -1) {
          this.projects[index] = project;
        } else {
          this.addProject(project);
        }
        
        return project;
      } catch (error: any) {
        this.setError(error.message || 'Failed to fetch project');
        throw error;
      } finally {
        this.setLoading(false);
      }
    },
    
    async createProject(projectData: Partial<Project>) {
      this.setLoading(true);
      this.setError(null);
      
      try {
        const { post } = useApiClient();
        const project = await post<Project>('/projects', projectData);
        this.addProject(project);
        this.setCurrentProject(project);
        return project;
      } catch (error: any) {
        this.setError(error.message || 'Failed to create project');
        throw error;
      } finally {
        this.setLoading(false);
      }
    },
    
    async updateProjectStatus(id: string, status: ProjectStatus) {
      this.setLoading(true);
      this.setError(null);
      
      try {
        const { patch } = useApiClient();
        const project = await patch<Project>(`/projects/${id}`, { status });
        this.updateProject(id, project);
        return project;
      } catch (error: any) {
        this.setError(error.message || 'Failed to update project status');
        throw error;
      } finally {
        this.setLoading(false);
      }
    },
    
    async deleteProject(id: string) {
      this.setLoading(true);
      this.setError(null);
      
      try {
        const { delete: del } = useApiClient();
        await del(`/projects/${id}`);
        this.removeProject(id);
      } catch (error: any) {
        this.setError(error.message || 'Failed to delete project');
        throw error;
      } finally {
        this.setLoading(false);
      }
    },
    
    reset() {
      this.projects = [];
      this.currentProject = null;
      this.isLoading = false;
      this.error = null;
      this.filters = {
        status: null,
        priority: null,
        tags: [],
        searchQuery: '',
      };
      this.sortBy = 'updatedAt';
      this.sortOrder = 'desc';
    },
  },
  
  persist: {
    enabled: true,
    strategies: [
      {
        key: 'projects',
        storage: process.client ? localStorage : undefined,
        paths: [
          'projects',
          'currentProject',
          'filters',
          'sortBy',
          'sortOrder'
        ], // Persist projects and user preferences
        // Don't persist loading states or errors
      },
    ],
  },
});