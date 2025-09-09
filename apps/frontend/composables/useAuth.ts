import { jwtDecode } from 'jwt-decode';
import type { User, AuthTokens, LoginRequest, RegisterRequest } from '~/types';

interface JWTPayload {
  sub: string;
  email: string;
  name: string;
  role: string;
  exp: number;
  iat: number;
}

export const useAuth = () => {
  const config = useRuntimeConfig();
  const router = useRouter();

  // State
  const user = useState<User | null>('auth.user', () => null);
  const isAuthenticated = useState<boolean>(
    'auth.isAuthenticated',
    () => false
  );
  const accessToken = useCookie('access_token', {
    httpOnly: true,
    secure: true,
    sameSite: 'lax',
    maxAge: 60 * 60 * 24 * 7, // 7 days
  });
  const refreshToken = useCookie('refresh_token', {
    httpOnly: true,
    secure: true,
    sameSite: 'lax',
    maxAge: 60 * 60 * 24 * 30, // 30 days
  });

  // Check if token is expired
  const isTokenExpired = (token: string): boolean => {
    try {
      const decoded = jwtDecode<JWTPayload>(token);
      return decoded.exp * 1000 < Date.now();
    } catch {
      return true;
    }
  };

  // Decode token and extract user info
  const decodeToken = (token: string): User | null => {
    try {
      const decoded = jwtDecode<JWTPayload>(token);
      return {
        id: decoded.sub,
        email: decoded.email,
        name: decoded.name,
        role: decoded.role as any,
        createdAt: new Date(decoded.iat * 1000).toISOString(),
        updatedAt: new Date().toISOString(),
      };
    } catch {
      return null;
    }
  };

  // Login
  const login = async (credentials: LoginRequest): Promise<void> => {
    try {
      const { data } = await $fetch<AuthTokens>(
        `${config.public.apiBase}/auth/login`,
        {
          method: 'POST',
          body: credentials,
        }
      );

      // Store tokens
      accessToken.value = data.accessToken;
      refreshToken.value = data.refreshToken;

      // Decode and store user
      const decodedUser = decodeToken(data.accessToken);
      if (decodedUser) {
        user.value = decodedUser;
        isAuthenticated.value = true;
      }

      // Redirect to dashboard
      await router.push('/dashboard');
    } catch (error: any) {
      throw new Error(error?.data?.message || 'Login failed');
    }
  };

  // Register
  const register = async (userData: RegisterRequest): Promise<void> => {
    try {
      const { data } = await $fetch<AuthTokens>(
        `${config.public.apiBase}/auth/register`,
        {
          method: 'POST',
          body: userData,
        }
      );

      // Store tokens
      accessToken.value = data.accessToken;
      refreshToken.value = data.refreshToken;

      // Decode and store user
      const decodedUser = decodeToken(data.accessToken);
      if (decodedUser) {
        user.value = decodedUser;
        isAuthenticated.value = true;
      }

      // Redirect to dashboard
      await router.push('/dashboard');
    } catch (error: any) {
      throw new Error(error?.data?.message || 'Registration failed');
    }
  };

  // Refresh access token
  const refreshAccessToken = async (): Promise<boolean> => {
    if (!refreshToken.value) return false;

    try {
      const { data } = await $fetch<AuthTokens>(
        `${config.public.apiBase}/auth/refresh`,
        {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${refreshToken.value}`,
          },
        }
      );

      // Update tokens
      accessToken.value = data.accessToken;
      refreshToken.value = data.refreshToken;

      // Update user
      const decodedUser = decodeToken(data.accessToken);
      if (decodedUser) {
        user.value = decodedUser;
        isAuthenticated.value = true;
      }

      return true;
    } catch {
      // Clear auth state on refresh failure
      await logout();
      return false;
    }
  };

  // Logout
  const logout = async (): Promise<void> => {
    try {
      // Call logout endpoint if token exists
      if (accessToken.value) {
        await $fetch(`${config.public.apiBase}/auth/logout`, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${accessToken.value}`,
          },
        }).catch(() => {
          // Ignore logout endpoint errors
        });
      }
    } finally {
      // Clear local state
      user.value = null;
      isAuthenticated.value = false;
      accessToken.value = null;
      refreshToken.value = null;

      // Redirect to login
      await router.push('/login');
    }
  };

  // Check and restore authentication on app load
  const checkAuth = async (): Promise<void> => {
    // Check for access token
    if (!accessToken.value) {
      isAuthenticated.value = false;
      return;
    }

    // Check if token is expired
    if (isTokenExpired(accessToken.value)) {
      // Try to refresh
      const refreshed = await refreshAccessToken();
      if (!refreshed) {
        isAuthenticated.value = false;
        return;
      }
    }

    // Decode and set user
    const decodedUser = decodeToken(accessToken.value);
    if (decodedUser) {
      user.value = decodedUser;
      isAuthenticated.value = true;
    } else {
      isAuthenticated.value = false;
    }
  };

  // Get current user
  const getCurrentUser = async (): Promise<User | null> => {
    if (!accessToken.value) return null;

    try {
      const { data } = await $fetch<User>(`${config.public.apiBase}/auth/me`, {
        headers: {
          Authorization: `Bearer ${accessToken.value}`,
        },
      });

      user.value = data;
      return data;
    } catch {
      return null;
    }
  };

  // Update user profile
  const updateProfile = async (updates: Partial<User>): Promise<void> => {
    if (!accessToken.value) throw new Error('Not authenticated');

    try {
      const { data } = await $fetch<User>(
        `${config.public.apiBase}/auth/profile`,
        {
          method: 'PATCH',
          headers: {
            Authorization: `Bearer ${accessToken.value}`,
          },
          body: updates,
        }
      );

      user.value = data;
    } catch (error: any) {
      throw new Error(error?.data?.message || 'Profile update failed');
    }
  };

  // Change password
  const changePassword = async (
    currentPassword: string,
    newPassword: string
  ): Promise<void> => {
    if (!accessToken.value) throw new Error('Not authenticated');

    try {
      await $fetch(`${config.public.apiBase}/auth/change-password`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${accessToken.value}`,
        },
        body: {
          currentPassword,
          newPassword,
        },
      });
    } catch (error: any) {
      throw new Error(error?.data?.message || 'Password change failed');
    }
  };

  // Request password reset
  const requestPasswordReset = async (email: string): Promise<void> => {
    try {
      await $fetch(`${config.public.apiBase}/auth/forgot-password`, {
        method: 'POST',
        body: { email },
      });
    } catch (error: any) {
      throw new Error(error?.data?.message || 'Password reset request failed');
    }
  };

  // Reset password with token
  const resetPassword = async (
    token: string,
    newPassword: string
  ): Promise<void> => {
    try {
      await $fetch(`${config.public.apiBase}/auth/reset-password`, {
        method: 'POST',
        body: {
          token,
          newPassword,
        },
      });
    } catch (error: any) {
      throw new Error(error?.data?.message || 'Password reset failed');
    }
  };

  return {
    user: readonly(user),
    isAuthenticated: readonly(isAuthenticated),
    login,
    register,
    logout,
    checkAuth,
    getCurrentUser,
    updateProfile,
    changePassword,
    requestPasswordReset,
    resetPassword,
    refreshAccessToken,
  };
};
