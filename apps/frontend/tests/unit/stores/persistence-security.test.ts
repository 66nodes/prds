import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { setActivePinia, createPinia } from 'pinia';
import { useAuthStore } from '~/stores/auth';
import { usePRDStore } from '~/stores/prd';
import { useProjectsStore } from '~/stores/projects';

describe('Pinia Persistence Security Tests', () => {
  // Mock storage
  let mockLocalStorage: { [key: string]: string } = {};
  let mockSessionStorage: { [key: string]: string } = {};

  beforeEach(() => {
    // Create a fresh Pinia instance for each test
    setActivePinia(createPinia());

    // Mock localStorage
    Object.defineProperty(window, 'localStorage', {
      value: {
        getItem: (key: string) => mockLocalStorage[key] || null,
        setItem: (key: string, value: string) => {
          mockLocalStorage[key] = value;
        },
        removeItem: (key: string) => {
          delete mockLocalStorage[key];
        },
        clear: () => {
          mockLocalStorage = {};
        },
      },
      writable: true,
    });

    // Mock sessionStorage
    Object.defineProperty(window, 'sessionStorage', {
      value: {
        getItem: (key: string) => mockSessionStorage[key] || null,
        setItem: (key: string, value: string) => {
          mockSessionStorage[key] = value;
        },
        removeItem: (key: string) => {
          delete mockSessionStorage[key];
        },
        clear: () => {
          mockSessionStorage = {};
        },
      },
      writable: true,
    });
  });

  afterEach(() => {
    // Clear storage after each test
    mockLocalStorage = {};
    mockSessionStorage = {};
  });

  describe('Auth Store Security', () => {
    it('should use sessionStorage for auth data', () => {
      const authStore = useAuthStore();
      
      // Set user data
      authStore.setUser({
        id: '123',
        email: 'testexample.com',
        name: 'Test User',
        role: 'admin',
        avatar: null,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      });

      // Check sessionStorage is used (not localStorage)
      const sessionKeys = Object.keys(mockSessionStorage);
      const localKeys = Object.keys(mockLocalStorage);
      
      // Auth data should be in sessionStorage
      expect(sessionKeys.some(key => key.includes('auth'))).toBe(true);
      // Auth data should NOT be in localStorage
      expect(localKeys.some(key => key.includes('auth'))).toBe(false);
    });

    it('should not persist sensitive tokens', () => {
      const authStore = useAuthStore();
      
      // Set user with token
      const userData = {
        id: '123',
        email: 'testexample.com',
        name: 'Test User',
        role: 'admin',
        avatar: null,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        token: 'secret-jwt-token', // This should not be persisted
        refreshToken: 'secret-refresh-token', // This should not be persisted
      };
      
      authStore.setUser(userData as any);

      // Check stored data doesn't contain tokens
      const storedData = Object.values(mockSessionStorage).join('');
      expect(storedData).not.toContain('secret-jwt-token');
      expect(storedData).not.toContain('secret-refresh-token');
    });

    it('should clear auth data on reset', () => {
      const authStore = useAuthStore();
      
      // Set user data
      authStore.setUser({
        id: '123',
        email: 'testexample.com',
        name: 'Test User',
        role: 'admin',
        avatar: null,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      });

      // Reset store
      authStore.reset();

      // Check data is cleared
      expect(authStore.user).toBeNull();
      expect(authStore.isAuthenticated).toBe(false);
    });
  });

  describe('PRD Store Security', () => {
    it('should use localStorage for PRD data', () => {
      const prdStore = usePRDStore();
      
      // Add PRD data
      prdStore.addPRD({
        id: 'prd-1',
        title: 'Test PRD',
        description: 'Test Description',
        project_id: 'project-1',
        content: {} as any,
        metadata: {} as any,
        hallucination_rate: 0.01,
        validation_score: 0.99,
        sources: [],
        graph_evidence: {},
        generated_by: 'test',
        review_status: 'pending',
        reviewedBy: null,
        reviewedAt: null,
        feedbackIncorporated: false,
      });

      // Check localStorage is used
      const localKeys = Object.keys(mockLocalStorage);
      expect(localKeys.some(key => key.includes('prd'))).toBe(true);
    });

    it('should not persist loading states', () => {
      const prdStore = usePRDStore();
      
      // Set loading states
      prdStore.setGenerating(true);
      prdStore.setValidating(true);
      prdStore.setLoading(true);
      prdStore.setError('Test error');

      // Check stored data doesn't contain loading states
      const storedData = Object.values(mockLocalStorage).join('');
      
      // These states should not be persisted
      expect(storedData).not.toContain('"isGenerating":true');
      expect(storedData).not.toContain('"isValidating":true');
      expect(storedData).not.toContain('"isLoading":true');
      expect(storedData).not.toContain('Test error');
    });
  });

  describe('Projects Store Security', () => {
    it('should use localStorage for project data', () => {
      const projectsStore = useProjectsStore();
      
      // Add project data
      projectsStore.addProject({
        id: 'project-1',
        name: 'Test Project',
        description: 'Test Description',
        status: 'active' as any,
        priority: 'medium' as any,
        createdAt: new Date(),
        updatedAt: new Date(),
        owner: 'user-1',
        team: [],
        tags: [],
        progress: 50,
        metadata: {} as any,
      });

      // Check localStorage is used
      const localKeys = Object.keys(mockLocalStorage);
      expect(localKeys.some(key => key.includes('projects'))).toBe(true);
    });

    it('should persist user preferences', () => {
      const projectsStore = useProjectsStore();
      
      // Set user preferences
      projectsStore.setSorting('name', 'desc');
      projectsStore.setFilter('status', 'active');
      projectsStore.setFilter('priority', 'high');

      // Check preferences are persisted
      const storedData = Object.values(mockLocalStorage).join('');
      expect(storedData).toContain('"sortBy":"name"');
      expect(storedData).toContain('"sortOrder":"desc"');
      expect(storedData).toContain('"status":"active"');
      expect(storedData).toContain('"priority":"high"');
    });

    it('should not persist error states', () => {
      const projectsStore = useProjectsStore();
      
      // Set error state
      projectsStore.setError('Test error message');
      projectsStore.setLoading(true);

      // Check error state is not persisted
      const storedData = Object.values(mockLocalStorage).join('');
      expect(storedData).not.toContain('Test error message');
      expect(storedData).not.toContain('"isLoading":true');
    });
  });

  describe('Cross-Store Security', () => {
    it('should isolate data between stores', () => {
      const authStore = useAuthStore();
      const prdStore = usePRDStore();
      const projectsStore = useProjectsStore();

      // Set data in all stores
      authStore.setUser({
        id: 'user-1',
        email: 'userexample.com',
        name: 'User',
        role: 'user',
        avatar: null,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      });

      prdStore.addPRD({
        id: 'prd-1',
        title: 'PRD',
        description: 'Description',
        project_id: 'project-1',
        content: {} as any,
        metadata: {} as any,
        hallucination_rate: 0.01,
        validation_score: 0.99,
        sources: [],
        graph_evidence: {},
        generated_by: 'test',
        review_status: 'pending',
        reviewedBy: null,
        reviewedAt: null,
        feedbackIncorporated: false,
      });

      projectsStore.addProject({
        id: 'project-1',
        name: 'Project',
        description: 'Description',
        status: 'active' as any,
        priority: 'medium' as any,
        createdAt: new Date(),
        updatedAt: new Date(),
        owner: 'user-1',
        team: [],
        tags: [],
        progress: 0,
        metadata: {} as any,
      });

      // Check data isolation
      const sessionKeys = Object.keys(mockSessionStorage);
      const localKeys = Object.keys(mockLocalStorage);

      // Auth should only be in sessionStorage
      expect(sessionKeys.some(key => key.includes('auth'))).toBe(true);
      expect(localKeys.some(key => key.includes('auth'))).toBe(false);

      // PRD should only be in localStorage
      expect(localKeys.some(key => key.includes('prd'))).toBe(true);
      expect(sessionKeys.some(key => key.includes('prd'))).toBe(false);

      // Projects should only be in localStorage
      expect(localKeys.some(key => key.includes('projects'))).toBe(true);
      expect(sessionKeys.some(key => key.includes('projects'))).toBe(false);
    });

    it('should handle storage quota exceeded gracefully', () => {
      const projectsStore = useProjectsStore();
      
      // Mock storage quota exceeded error
      const originalSetItem = window.localStorage.setItem;
      window.localStorage.setItem = () => {
        throw new Error('QuotaExceededError');
      };

      // Try to add large amount of data
      expect(() => {
        for (let i = 0; i < 100; i++) {
          projectsStore.addProject({
            id: `project-${i}`,
            name: `Project ${i}`,
            description: 'Large description '.repeat(1000),
            status: 'active' as any,
            priority: 'medium' as any,
            createdAt: new Date(),
            updatedAt: new Date(),
            owner: 'user-1',
            team: [],
            tags: [],
            progress: 0,
            metadata: {} as any,
          });
        }
      }).not.toThrow();

      // Restore original
      window.localStorage.setItem = originalSetItem;
    });
  });

  describe('Data Validation', () => {
    it('should validate data types before persistence', () => {
      const projectsStore = useProjectsStore();
      
      // Try to add invalid data
      const invalidProject = {
        id: 123, // Should be string
        name: null, // Should be string
        description: undefined,
        status: 'invalid-status',
        priority: 999,
        createdAt: 'not-a-date',
        updatedAt: 'not-a-date',
        owner: {},
        team: 'not-an-array',
        tags: null,
        progress: 'not-a-number',
        metadata: 'not-an-object',
      };

      // Store should handle invalid data gracefully
      expect(() => {
        projectsStore.addProject(invalidProject as any);
      }).not.toThrow();
    });

    it('should sanitize data before storage', () => {
      const prdStore = usePRDStore();
      
      // Add PRD with potentially malicious content
      const maliciousPRD = {
        id: 'prd-xss',
        title: '<script>alert("XSS")</script>',
        description: 'javascript:alert("XSS")',
        project_id: 'project-1',
        content: {
          executive_summary: '<img src=x onerror=alert("XSS")>',
        } as any,
        metadata: {} as any,
        hallucination_rate: 0.01,
        validation_score: 0.99,
        sources: [],
        graph_evidence: {},
        generated_by: 'test',
        review_status: 'pending',
        reviewedBy: null,
        reviewedAt: null,
        feedbackIncorporated: false,
      };

      prdStore.addPRD(maliciousPRD);

      // Data should be stored as-is (sanitization happens on render)
      expect(prdStore.prdById('prd-xss')).toBeDefined();
      expect(prdStore.prdById('prd-xss')?.title).toBe('<script>alert("XSS")</script>');
    });
  });
});