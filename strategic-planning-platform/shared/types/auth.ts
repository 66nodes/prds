/**
 * Authentication and Authorization Types
 * Shared types for user management, roles, and JWT tokens
 */

export enum UserRole {
  ADMIN = 'admin',
  PROJECT_MANAGER = 'project_manager',
  CONTRIBUTOR = 'contributor',
  VIEWER = 'viewer'
}

export enum UserStatus {
  ACTIVE = 'active',
  INACTIVE = 'inactive',
  PENDING = 'pending',
  SUSPENDED = 'suspended'
}

export interface User {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  role: UserRole;
  status: UserStatus;
  createdAt: string;
  updatedAt: string;
  lastLoginAt?: string;
  profileImageUrl?: string;
  department?: string;
  timezone?: string;
  preferences: UserPreferences;
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'system';
  language: string;
  notifications: NotificationSettings;
  dashboard: DashboardPreferences;
}

export interface NotificationSettings {
  email: boolean;
  inApp: boolean;
  prdUpdates: boolean;
  projectAssignments: boolean;
  systemAlerts: boolean;
}

export interface DashboardPreferences {
  defaultView: 'list' | 'grid' | 'kanban';
  itemsPerPage: number;
  showCompletedTasks: boolean;
  autoRefresh: boolean;
}

export interface JWTTokens {
  accessToken: string;
  refreshToken: string;
  tokenType: 'Bearer';
  expiresIn: number;
  scope: string[];
}

export interface JWTPayload {
  sub: string; // user ID
  email: string;
  role: UserRole;
  iat: number;
  exp: number;
  scope: string[];
  sessionId: string;
}

export interface LoginRequest {
  email: string;
  password: string;
  rememberMe?: boolean;
}

export interface LoginResponse {
  user: User;
  tokens: JWTTokens;
  sessionId: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  firstName: string;
  lastName: string;
  department?: string;
  inviteCode?: string;
}

export interface PasswordResetRequest {
  email: string;
}

export interface PasswordResetConfirm {
  token: string;
  newPassword: string;
}

export interface RefreshTokenRequest {
  refreshToken: string;
}

export interface UserUpdateRequest {
  firstName?: string;
  lastName?: string;
  department?: string;
  timezone?: string;
  preferences?: Partial<UserPreferences>;
}

export interface RolePermissions {
  [key: string]: {
    read: boolean;
    write: boolean;
    delete: boolean;
    admin: boolean;
  };
}

export const DEFAULT_PERMISSIONS: Record<UserRole, RolePermissions> = {
  [UserRole.ADMIN]: {
    users: { read: true, write: true, delete: true, admin: true },
    prds: { read: true, write: true, delete: true, admin: true },
    projects: { read: true, write: true, delete: true, admin: true },
    system: { read: true, write: true, delete: true, admin: true }
  },
  [UserRole.PROJECT_MANAGER]: {
    users: { read: true, write: false, delete: false, admin: false },
    prds: { read: true, write: true, delete: true, admin: false },
    projects: { read: true, write: true, delete: false, admin: false },
    system: { read: false, write: false, delete: false, admin: false }
  },
  [UserRole.CONTRIBUTOR]: {
    users: { read: false, write: false, delete: false, admin: false },
    prds: { read: true, write: true, delete: false, admin: false },
    projects: { read: true, write: false, delete: false, admin: false },
    system: { read: false, write: false, delete: false, admin: false }
  },
  [UserRole.VIEWER]: {
    users: { read: false, write: false, delete: false, admin: false },
    prds: { read: true, write: false, delete: false, admin: false },
    projects: { read: true, write: false, delete: false, admin: false },
    system: { read: false, write: false, delete: false, admin: false }
  }
};

export interface AuthContextState {
  user: User | null;
  tokens: JWTTokens | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  sessionId: string | null;
}

export interface AuthActions {
  login: (credentials: LoginRequest) => Promise<LoginResponse>;
  logout: () => Promise<void>;
  register: (userData: RegisterRequest) => Promise<User>;
  refreshTokens: () => Promise<JWTTokens>;
  updateUser: (updates: UserUpdateRequest) => Promise<User>;
  resetPassword: (email: string) => Promise<void>;
  confirmPasswordReset: (data: PasswordResetConfirm) => Promise<void>;
  checkPermission: (resource: string, action: string) => boolean;
}