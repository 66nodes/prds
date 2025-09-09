import { defineStore } from 'pinia';
import type { User, LoginRequest, RegisterRequest } from '~/types';

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

export const useAuthStore = defineStore('auth', {
  state: (): AuthState => ({
    user: null,
    isAuthenticated: false,
    isLoading: false,
    error: null,
  }),

  getters: {
    currentUser: state => state.user,
    isLoggedIn: state => state.isAuthenticated,
    userRole: state => state.user?.role,
    userId: state => state.user?.id,
    userName: state => state.user?.name,
    userEmail: state => state.user?.email,
    hasRole: state => (role: string) => state.user?.role === role,
    isAdmin: state => state.user?.role === 'admin',
    authError: state => state.error,
  },

  actions: {
    setUser(user: User | null) {
      this.user = user;
      this.isAuthenticated = !!user;
    },

    setLoading(loading: boolean) {
      this.isLoading = loading;
    },

    setError(error: string | null) {
      this.error = error;
    },

    async login(credentials: LoginRequest) {
      this.setLoading(true);
      this.setError(null);

      try {
        const { login } = useAuth();
        await login(credentials);
        // User will be set by the composable
      } catch (error: any) {
        this.setError(error.message || 'Login failed');
        throw error;
      } finally {
        this.setLoading(false);
      }
    },

    async register(userData: RegisterRequest) {
      this.setLoading(true);
      this.setError(null);

      try {
        const { register } = useAuth();
        await register(userData);
        // User will be set by the composable
      } catch (error: any) {
        this.setError(error.message || 'Registration failed');
        throw error;
      } finally {
        this.setLoading(false);
      }
    },

    async logout() {
      this.setLoading(true);

      try {
        const { logout } = useAuth();
        await logout();
        this.reset();
      } catch (error: any) {
        console.error('Logout error:', error);
      } finally {
        this.setLoading(false);
      }
    },

    async checkAuth() {
      this.setLoading(true);

      try {
        const { checkAuth, user } = useAuth();
        await checkAuth();

        if (user.value) {
          this.setUser(user.value);
        }
      } catch (error: any) {
        console.error('Auth check error:', error);
        this.reset();
      } finally {
        this.setLoading(false);
      }
    },

    async refreshToken() {
      try {
        const { refreshAccessToken } = useAuth();
        const success = await refreshAccessToken();

        if (!success) {
          this.reset();
        }

        return success;
      } catch (error: any) {
        console.error('Token refresh error:', error);
        this.reset();
        return false;
      }
    },

    async updateProfile(updates: Partial<User>) {
      this.setLoading(true);
      this.setError(null);

      try {
        const { updateProfile } = useAuth();
        await updateProfile(updates);

        if (this.user) {
          this.user = { ...this.user, ...updates };
        }
      } catch (error: any) {
        this.setError(error.message || 'Profile update failed');
        throw error;
      } finally {
        this.setLoading(false);
      }
    },

    async changePassword(currentPassword: string, newPassword: string) {
      this.setLoading(true);
      this.setError(null);

      try {
        const { changePassword } = useAuth();
        await changePassword(currentPassword, newPassword);
      } catch (error: any) {
        this.setError(error.message || 'Password change failed');
        throw error;
      } finally {
        this.setLoading(false);
      }
    },

    reset() {
      this.user = null;
      this.isAuthenticated = false;
      this.error = null;
    },
  },

  persist: {
    enabled: true,
    strategies: [
      {
        key: 'auth',
        storage: process.client ? sessionStorage : undefined, // Use sessionStorage for better security
        paths: ['user', 'isAuthenticated'], // Only persist non-sensitive data
        // Token handling is managed by the useAuth composable with secure httpOnly cookies
      },
    ],
  },
});
