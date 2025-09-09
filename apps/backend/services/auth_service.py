"""
Authentication and Authorization Service for the Strategic Planning Platform.

Provides JWT-based authentication, role-based access control, and user management.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import uuid
import hashlib
import secrets

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

from models.user import User, UserCreate, UserLogin, TokenResponse
from core.config import get_settings
from core.database import get_postgres, get_redis

logger = structlog.get_logger(__name__)
settings = get_settings()


class UserRole:
    """User role definitions."""
    ADMIN = "admin"
    USER = "user"  
    VIEWER = "viewer"
    SYSTEM = "system"


class Permission:
    """Permission definitions."""
    
    # PRD permissions
    CREATE_PRD = "create_prd"
    READ_PRD = "read_prd"
    UPDATE_PRD = "update_prd"
    DELETE_PRD = "delete_prd"
    
    # Validation permissions
    VALIDATE_CONTENT = "validate_content"
    VIEW_VALIDATION_RESULTS = "view_validation_results"
    
    # System permissions
    ADMIN_ACCESS = "admin_access"
    VIEW_SYSTEM_METRICS = "view_system_metrics"
    MANAGE_USERS = "manage_users"
    
    # Agent permissions
    EXECUTE_AGENTS = "execute_agents"
    VIEW_AGENT_STATUS = "view_agent_status"


class RolePermissions:
    """Role-based permission mapping."""
    
    ROLE_PERMISSIONS = {
        UserRole.ADMIN: [
            Permission.CREATE_PRD,
            Permission.READ_PRD,
            Permission.UPDATE_PRD,
            Permission.DELETE_PRD,
            Permission.VALIDATE_CONTENT,
            Permission.VIEW_VALIDATION_RESULTS,
            Permission.ADMIN_ACCESS,
            Permission.VIEW_SYSTEM_METRICS,
            Permission.MANAGE_USERS,
            Permission.EXECUTE_AGENTS,
            Permission.VIEW_AGENT_STATUS,
        ],
        UserRole.USER: [
            Permission.CREATE_PRD,
            Permission.READ_PRD,
            Permission.UPDATE_PRD,
            Permission.VALIDATE_CONTENT,
            Permission.VIEW_VALIDATION_RESULTS,
            Permission.EXECUTE_AGENTS,
            Permission.VIEW_AGENT_STATUS,
        ],
        UserRole.VIEWER: [
            Permission.READ_PRD,
            Permission.VIEW_VALIDATION_RESULTS,
            Permission.VIEW_AGENT_STATUS,
        ],
        UserRole.SYSTEM: [
            Permission.CREATE_PRD,
            Permission.READ_PRD,
            Permission.UPDATE_PRD,
            Permission.VALIDATE_CONTENT,
            Permission.EXECUTE_AGENTS,
            Permission.VIEW_AGENT_STATUS,
        ]
    }
    
    @classmethod
    def get_permissions(cls, role: str) -> List[str]:
        """Get permissions for a role."""
        return cls.ROLE_PERMISSIONS.get(role, [])
    
    @classmethod
    def has_permission(cls, role: str, permission: str) -> bool:
        """Check if role has permission."""
        return permission in cls.get_permissions(role)


class AuthService:
    """
    Authentication and authorization service.
    
    Features:
    - JWT token generation and validation
    - Password hashing and verification
    - Role-based access control
    - Session management
    - Rate limiting for auth endpoints
    - Account lockout protection
    """
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.postgres = None
        self.redis = None
        self.is_initialized = False
        
        # Security settings
        self.token_expire_minutes = settings.access_token_expire_minutes
        self.refresh_token_expire_days = 30
        self.max_login_attempts = 5
        self.lockout_duration_minutes = 30
        self.session_timeout_hours = 24
        
    async def initialize(self) -> None:
        """Initialize the authentication service."""
        try:
            self.postgres = await get_postgres()
            self.redis = await get_redis()
            
            # Create users table if it doesn't exist
            await self._create_users_table()
            
            self.is_initialized = True
            logger.info("Authentication service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize authentication service: {str(e)}")
            raise
    
    async def _create_users_table(self) -> None:
        """Create users table if it doesn't exist."""
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS users (
            id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
            email VARCHAR(255) UNIQUE NOT NULL,
            full_name VARCHAR(100) NOT NULL,
            company VARCHAR(100),
            role VARCHAR(50) NOT NULL DEFAULT 'user',
            hashed_password VARCHAR(255) NOT NULL,
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            failed_login_attempts INTEGER DEFAULT 0,
            locked_until TIMESTAMP,
            password_changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
        CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);
        """
        
        try:
            await self.postgres.execute(create_table_query)
            logger.info("Users table created/verified successfully")
        except Exception as e:
            logger.error(f"Failed to create users table: {str(e)}")
            raise
    
    # Password operations
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    # JWT operations
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create JWT access token."""
        
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.token_expire_minutes)
        to_encode.update({"exp": expire, "type": "access"})
        
        encoded_jwt = jwt.encode(
            to_encode, 
            settings.secret_key, 
            algorithm=settings.algorithm
        )
        
        return encoded_jwt
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token."""
        
        data = {
            "sub": user_id,
            "type": "refresh",
            "exp": datetime.utcnow() + timedelta(days=self.refresh_token_expire_days),
            "jti": str(uuid.uuid4())  # JWT ID for token revocation
        }
        
        encoded_jwt = jwt.encode(
            data,
            settings.secret_key,
            algorithm=settings.algorithm
        )
        
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        
        try:
            payload = jwt.decode(
                token,
                settings.secret_key,
                algorithms=[settings.algorithm]
            )
            return payload
            
        except JWTError as e:
            logger.warning(f"JWT verification failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    # User management
    async def create_user(self, user_create: UserCreate) -> User:
        """Create a new user account."""
        
        if not self.is_initialized:
            raise RuntimeError("Auth service not initialized")
        
        # Check if user already exists
        existing_user = await self.get_user_by_email(user_create.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User with this email already exists"
            )
        
        # Hash password
        hashed_password = self.hash_password(user_create.password)
        
        # Create user
        user_id = str(uuid.uuid4())
        
        insert_query = """
        INSERT INTO users (id, email, full_name, company, role, hashed_password, is_active)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        RETURNING id, email, full_name, company, role, is_active, created_at, last_login
        """
        
        try:
            result = await self.postgres.fetchrow(
                insert_query,
                user_id,
                user_create.email,
                user_create.full_name,
                user_create.company,
                user_create.role,
                hashed_password,
                user_create.is_active
            )
            
            user = User(
                id=result["id"],
                email=result["email"],
                full_name=result["full_name"],
                company=result["company"],
                role=result["role"],
                is_active=result["is_active"],
                created_at=result["created_at"],
                last_login=result["last_login"]
            )
            
            logger.info(f"User created successfully: {user.email}")
            return user
            
        except Exception as e:
            logger.error(f"Failed to create user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )
    
    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email address."""
        
        query = """
        SELECT id, email, full_name, company, role, hashed_password, is_active, 
               created_at, last_login, failed_login_attempts, locked_until
        FROM users 
        WHERE email = $1
        """
        
        try:
            result = await self.postgres.fetchrow(query, email)
            return dict(result) if result else None
            
        except Exception as e:
            logger.error(f"Failed to get user by email: {str(e)}")
            return None
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        
        query = """
        SELECT id, email, full_name, company, role, is_active, created_at, last_login
        FROM users 
        WHERE id = $1 AND is_active = true
        """
        
        try:
            result = await self.postgres.fetchrow(query, user_id)
            if result:
                return User(
                    id=result["id"],
                    email=result["email"],
                    full_name=result["full_name"],
                    company=result["company"],
                    role=result["role"],
                    is_active=result["is_active"],
                    created_at=result["created_at"],
                    last_login=result["last_login"]
                )
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user by ID: {str(e)}")
            return None
    
    # Authentication operations
    async def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with email and password."""
        
        # Check if account is locked
        if await self._is_account_locked(email):
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account is temporarily locked due to multiple failed login attempts"
            )
        
        user = await self.get_user_by_email(email)
        if not user:
            await self._record_failed_login(email)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        if not user["is_active"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is deactivated"
            )
        
        if not self.verify_password(password, user["hashed_password"]):
            await self._record_failed_login(email)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Reset failed login attempts on successful login
        await self._reset_failed_login_attempts(email)
        await self._update_last_login(user["id"])
        
        return user
    
    async def login(self, user_login: UserLogin) -> TokenResponse:
        """Login user and return tokens."""
        
        user = await self.authenticate_user(user_login.email, user_login.password)
        
        # Create tokens
        token_data = {
            "sub": user["id"],
            "email": user["email"],
            "role": user["role"],
            "full_name": user["full_name"]
        }
        
        access_token = self.create_access_token(token_data)
        refresh_token = self.create_refresh_token(user["id"])
        
        # Store refresh token in Redis
        await self._store_refresh_token(user["id"], refresh_token)
        
        user_obj = User(
            id=user["id"],
            email=user["email"],
            full_name=user["full_name"],
            company=user["company"],
            role=user["role"],
            is_active=user["is_active"],
            created_at=user["created_at"],
            last_login=user["last_login"]
        )
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=self.token_expire_minutes * 60,
            user=user_obj
        )
    
    async def refresh_access_token(self, refresh_token: str) -> TokenResponse:
        """Refresh access token using refresh token."""
        
        try:
            payload = self.verify_token(refresh_token)
            
            if payload.get("type") != "refresh":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            user_id = payload.get("sub")
            jti = payload.get("jti")
            
            # Verify refresh token is still valid in Redis
            stored_token = await self._get_stored_refresh_token(user_id)
            if not stored_token or stored_token != refresh_token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token"
                )
            
            # Get current user data
            user = await self.get_user_by_id(user_id)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )
            
            # Create new tokens
            token_data = {
                "sub": user.id,
                "email": user.email,
                "role": user.role,
                "full_name": user.full_name
            }
            
            access_token = self.create_access_token(token_data)
            new_refresh_token = self.create_refresh_token(user.id)
            
            # Store new refresh token
            await self._store_refresh_token(user.id, new_refresh_token)
            
            return TokenResponse(
                access_token=access_token,
                refresh_token=new_refresh_token,
                token_type="bearer",
                expires_in=self.token_expire_minutes * 60,
                user=user
            )
            
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
    
    async def logout(self, user_id: str) -> None:
        """Logout user and invalidate tokens."""
        
        try:
            # Remove refresh token from Redis
            if self.redis:
                await self.redis.delete(f"refresh_token:{user_id}")
            
            logger.info(f"User logged out: {user_id}")
            
        except Exception as e:
            logger.error(f"Logout error: {str(e)}")
    
    # Authorization operations
    def check_permission(self, user_role: str, required_permission: str) -> bool:
        """Check if user role has required permission."""
        return RolePermissions.has_permission(user_role, required_permission)
    
    def require_permission(self, user_role: str, required_permission: str) -> None:
        """Require user to have specific permission."""
        if not self.check_permission(user_role, required_permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {required_permission}"
            )
    
    def require_role(self, user_role: str, required_roles: List[str]) -> None:
        """Require user to have one of the specified roles."""
        if user_role not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient role. Required one of: {required_roles}"
            )
    
    # Helper methods
    async def _is_account_locked(self, email: str) -> bool:
        """Check if account is locked due to failed login attempts."""
        
        query = """
        SELECT failed_login_attempts, locked_until
        FROM users 
        WHERE email = $1
        """
        
        try:
            result = await self.postgres.fetchrow(query, email)
            if not result:
                return False
            
            # Check if account is locked and lock hasn't expired
            if result["locked_until"]:
                if datetime.utcnow() < result["locked_until"]:
                    return True
                else:
                    # Lock expired, reset attempts
                    await self._reset_failed_login_attempts(email)
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking account lock: {str(e)}")
            return False
    
    async def _record_failed_login(self, email: str) -> None:
        """Record failed login attempt."""
        
        update_query = """
        UPDATE users 
        SET failed_login_attempts = failed_login_attempts + 1,
            locked_until = CASE 
                WHEN failed_login_attempts + 1 >= $1 
                THEN $2 
                ELSE locked_until 
            END
        WHERE email = $3
        """
        
        try:
            lock_until = datetime.utcnow() + timedelta(minutes=self.lockout_duration_minutes)
            await self.postgres.execute(
                update_query,
                self.max_login_attempts,
                lock_until,
                email
            )
            
        except Exception as e:
            logger.error(f"Error recording failed login: {str(e)}")
    
    async def _reset_failed_login_attempts(self, email: str) -> None:
        """Reset failed login attempts."""
        
        update_query = """
        UPDATE users 
        SET failed_login_attempts = 0, locked_until = NULL
        WHERE email = $1
        """
        
        try:
            await self.postgres.execute(update_query, email)
        except Exception as e:
            logger.error(f"Error resetting failed login attempts: {str(e)}")
    
    async def _update_last_login(self, user_id: str) -> None:
        """Update user's last login timestamp."""
        
        update_query = """
        UPDATE users 
        SET last_login = CURRENT_TIMESTAMP
        WHERE id = $1
        """
        
        try:
            await self.postgres.execute(update_query, user_id)
        except Exception as e:
            logger.error(f"Error updating last login: {str(e)}")
    
    async def _store_refresh_token(self, user_id: str, refresh_token: str) -> None:
        """Store refresh token in Redis."""
        
        try:
            if self.redis:
                await self.redis.setex(
                    f"refresh_token:{user_id}",
                    self.refresh_token_expire_days * 24 * 3600,  # seconds
                    refresh_token
                )
        except Exception as e:
            logger.error(f"Error storing refresh token: {str(e)}")
    
    async def _get_stored_refresh_token(self, user_id: str) -> Optional[str]:
        """Get stored refresh token from Redis."""
        
        try:
            if self.redis:
                token = await self.redis.get(f"refresh_token:{user_id}")
                return token.decode() if token else None
            return None
        except Exception as e:
            logger.error(f"Error getting stored refresh token: {str(e)}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check authentication service health."""
        
        try:
            return {
                "status": "healthy" if self.is_initialized else "initializing",
                "initialized": self.is_initialized,
                "postgres_connected": self.postgres is not None,
                "redis_connected": self.redis is not None,
                "security_settings": {
                    "token_expire_minutes": self.token_expire_minutes,
                    "max_login_attempts": self.max_login_attempts,
                    "lockout_duration_minutes": self.lockout_duration_minutes
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Global auth service instance
auth_service = AuthService()


async def get_auth_service() -> AuthService:
    """Get the global authentication service instance."""
    if not auth_service.is_initialized:
        await auth_service.initialize()
    return auth_service