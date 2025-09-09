"""
Authentication endpoints for the Strategic Planning Platform API.

Provides JWT-based authentication, user registration, login, token refresh,
and role-based access control endpoints using the comprehensive AuthService.
"""

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

from models.user import User, UserCreate, UserLogin, TokenResponse
from services.auth_service import get_auth_service, AuthService, Permission
from core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()
security = HTTPBearer()

router = APIRouter()


# Dependency for getting current user from JWT token
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
) -> User:
    """Get current user from JWT token."""
    
    try:
        token = credentials.credentials
        payload = auth_service.verify_token(token)
        
        # Verify token type
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        # Get user from database
        user = await auth_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


# Dependency for admin access
async def require_admin(
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
) -> User:
    """Require admin role."""
    auth_service.require_permission(current_user.role, Permission.ADMIN_ACCESS)
    return current_user


# Dependency for user management access
async def require_user_management(
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
) -> User:
    """Require user management permissions."""
    auth_service.require_permission(current_user.role, Permission.MANAGE_USERS)
    return current_user


@router.post("/register", response_model=TokenResponse)
async def register_user(
    user_create: UserCreate,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Register a new user account.
    
    Creates a new user with the provided information and returns
    authentication tokens for immediate login.
    """
    
    try:
        # Create user
        user = await auth_service.create_user(user_create)
        
        # Generate tokens for immediate login
        login_request = UserLogin(
            email=user_create.email,
            password=user_create.password
        )
        
        token_response = await auth_service.login(login_request)
        
        logger.info(
            "User registered successfully",
            user_id=user.id,
            email=user.email,
            role=user.role
        )
        
        return token_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User registration failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=TokenResponse)
async def login_user(
    user_login: UserLogin,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Authenticate user and return tokens.
    
    Validates user credentials and returns JWT access and refresh tokens
    for authenticated API access.
    """
    
    try:
        token_response = await auth_service.login(user_login)
        
        logger.info(
            "User login successful",
            user_id=token_response.user.id,
            email=token_response.user.email
        )
        
        return token_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_token: str,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Refresh access token using refresh token.
    
    Generates a new access token and refresh token pair using a valid
    refresh token. The old refresh token is invalidated.
    """
    
    try:
        token_response = await auth_service.refresh_access_token(refresh_token)
        
        logger.info(
            "Token refresh successful",
            user_id=token_response.user.id
        )
        
        return token_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post("/logout")
async def logout_user(
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Logout current user.
    
    Invalidates the user's refresh token and logs them out of the system.
    The access token will continue to work until it expires naturally.
    """
    
    try:
        await auth_service.logout(current_user.id)
        
        logger.info(
            "User logout successful",
            user_id=current_user.id,
            email=current_user.email
        )
        
        return {
            "message": "Successfully logged out",
            "timestamp": settings.current_timestamp()
        }
        
    except Exception as e:
        logger.error(f"Logout failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/me", response_model=User)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get current user information.
    
    Returns the profile information for the currently authenticated user.
    """
    
    return current_user


@router.get("/permissions")
async def get_user_permissions(
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Get current user's permissions.
    
    Returns the list of permissions available to the current user
    based on their role.
    """
    
    from services.auth_service import RolePermissions
    
    permissions = RolePermissions.get_permissions(current_user.role)
    
    return {
        "user_id": current_user.id,
        "role": current_user.role,
        "permissions": permissions,
        "timestamp": settings.current_timestamp()
    }


@router.post("/verify-permission")
async def verify_permission(
    permission: str,
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Verify if current user has a specific permission.
    
    Checks whether the current user's role includes the specified permission.
    """
    
    has_permission = auth_service.check_permission(current_user.role, permission)
    
    return {
        "user_id": current_user.id,
        "role": current_user.role,
        "permission": permission,
        "has_permission": has_permission,
        "timestamp": settings.current_timestamp()
    }


@router.get("/users")
async def list_users(
    current_user: User = Depends(require_user_management),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    List all users (Admin/User Management only).
    
    Returns a list of all users in the system. Requires MANAGE_USERS permission.
    """
    
    # This would need to be implemented in AuthService
    # For now, return a placeholder response
    return {
        "message": "User listing not yet implemented",
        "requested_by": current_user.email,
        "timestamp": settings.current_timestamp()
    }


@router.put("/users/{user_id}/role")
async def update_user_role(
    user_id: str,
    new_role: str,
    current_user: User = Depends(require_admin),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Update user role (Admin only).
    
    Changes the role of a specified user. Requires ADMIN_ACCESS permission.
    """
    
    from services.auth_service import UserRole
    
    # Validate role
    valid_roles = [UserRole.ADMIN, UserRole.USER, UserRole.VIEWER, UserRole.SYSTEM]
    if new_role not in valid_roles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Must be one of: {valid_roles}"
        )
    
    # This would need to be implemented in AuthService
    # For now, return a placeholder response
    
    logger.info(
        "User role update requested",
        target_user_id=user_id,
        new_role=new_role,
        updated_by=current_user.id
    )
    
    return {
        "message": "User role update not yet implemented",
        "user_id": user_id,
        "new_role": new_role,
        "updated_by": current_user.email,
        "timestamp": settings.current_timestamp()
    }


@router.delete("/users/{user_id}")
async def deactivate_user(
    user_id: str,
    current_user: User = Depends(require_admin),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Deactivate user account (Admin only).
    
    Deactivates a user account, preventing further login.
    Requires ADMIN_ACCESS permission.
    """
    
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own account"
        )
    
    # This would need to be implemented in AuthService
    # For now, return a placeholder response
    
    logger.warning(
        "User deactivation requested",
        target_user_id=user_id,
        deactivated_by=current_user.id
    )
    
    return {
        "message": "User deactivation not yet implemented",
        "user_id": user_id,
        "deactivated_by": current_user.email,
        "timestamp": settings.current_timestamp()
    }


@router.get("/health")
async def auth_health_check(
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Authentication service health check.
    
    Returns the health status of the authentication service
    and its dependencies.
    """
    
    try:
        health_status = await auth_service.health_check()
        return health_status
        
    except Exception as e:
        logger.error(f"Auth health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": settings.current_timestamp()
        }