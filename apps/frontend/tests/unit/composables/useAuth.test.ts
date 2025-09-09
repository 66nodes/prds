import { describe, it, expect, vi, beforeEach } from 'vitest';
import { useAuth } from '~/composables/useAuth';

// Mock dependencies
vi.mock('jwt-decode', () => ({
  jwtDecode: vi.fn((token: string) => ({
    sub: 'user-123',
    email: 'testexample.com',
    name: 'Test User',
    role: 'user',
    exp: Date.now() / 1000 + 3600,
    iat: Date.now() / 1000,
  })),
}));

vi.mock('#app', () => ({
  useRuntimeConfig: vi.fn(() => ({
    public: {
      apiBase: 'http://localhost:8000',
    },
  })),
  useRouter: vi.fn(() => ({
    push: vi.fn(),
  })),
  useState: vi.fn((key, init) => {
    const state = ref(typeof init === 'function' ? init() : init);
    return state;
  }),
  useCookie: vi.fn(name => ({
    value: null,
  })),
  readonly: vi.fn(ref => ref),
  $fetch: vi.fn(),
}));

describe('useAuth', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should initialize with unauthenticated state', () => {
    const { user, isAuthenticated } = useAuth();

    expect(user.value).toBeNull();
    expect(isAuthenticated.value).toBe(false);
  });

  it('should login successfully with valid credentials', async () => {
    const mockTokens = {
      accessToken: 'mock-access-token',
      refreshToken: 'mock-refresh-token',
      expiresIn: 3600,
    };

    global.$fetch = vi.fn().mockResolvedValue({ data: mockTokens });

    const { login, user, isAuthenticated } = useAuth();

    await login({
      email: 'testexample.com',
      password: 'password',
    });

    expect(user.value).toBeDefined();
    expect(user.value?.email).toBe('testexample.com');
    expect(isAuthenticated.value).toBe(true);
  });

  it('should handle login failure', async () => {
    global.$fetch = vi.fn().mockRejectedValue(new Error('Invalid credentials'));

    const { login } = useAuth();

    await expect(
      login({
        email: 'testexample.com',
        password: 'wrong-password',
      })
    ).rejects.toThrow('Invalid credentials');
  });

  it('should logout and clear state', async () => {
    const { logout, user, isAuthenticated } = useAuth();

    // Mock $fetch to resolve for logout call
    global.$fetch = vi.fn().mockResolvedValue({});

    await logout();

    expect(user.value).toBeNull();
    expect(isAuthenticated.value).toBe(false);
  });
});
