"""
Enterprise Single Sign-On (SSO) Service for OAuth2/OIDC Integration.

Provides integration with enterprise identity providers including:
- Azure AD (Microsoft Entra ID)
- Okta
- Generic OIDC providers

Features:
- OAuth2/OIDC authentication flows
- User provisioning and role mapping
- Enterprise API integrations
- Comprehensive audit logging
- Multi-provider support
"""

import asyncio
import base64
import json
import secrets
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urlparse, parse_qs
import hashlib

import aiohttp
from fastapi import HTTPException, status
from jose import jwt, JWTError
import structlog

from models.user import User, UserCreate
from services.auth_service import AuthService, UserRole
from core.config import get_settings
from core.database import get_redis

logger = structlog.get_logger(__name__)
settings = get_settings()


class ProviderType:
    """Supported identity provider types."""
    AZURE_AD = "azure_ad"
    OKTA = "okta"
    GENERIC_OIDC = "generic_oidc"


class ProviderConfig:
    """Identity provider configuration."""
    
    def __init__(
        self,
        provider_type: str,
        client_id: str,
        client_secret: str,
        tenant_id: Optional[str] = None,
        issuer_url: Optional[str] = None,
        auth_url: Optional[str] = None,
        token_url: Optional[str] = None,
        userinfo_url: Optional[str] = None,
        jwks_url: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        role_mapping: Optional[Dict[str, str]] = None
    ):
        self.provider_type = provider_type
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.issuer_url = issuer_url
        self.auth_url = auth_url
        self.token_url = token_url
        self.userinfo_url = userinfo_url
        self.jwks_url = jwks_url
        self.scopes = scopes or ["openid", "profile", "email"]
        self.role_mapping = role_mapping or {}
    
    @classmethod
    def azure_ad(
        cls,
        client_id: str,
        client_secret: str,
        tenant_id: str,
        role_mapping: Optional[Dict[str, str]] = None
    ):
        """Create Azure AD configuration."""
        base_url = f"https://login.microsoftonline.com/{tenant_id}"
        return cls(
            provider_type=ProviderType.AZURE_AD,
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
            issuer_url=f"{base_url}/v2.0",
            auth_url=f"{base_url}/oauth2/v2.0/authorize",
            token_url=f"{base_url}/oauth2/v2.0/token",
            userinfo_url="https://graph.microsoft.com/oidc/userinfo",
            jwks_url=f"{base_url}/discovery/v2.0/keys",
            scopes=["openid", "profile", "email", "User.Read"],
            role_mapping=role_mapping
        )
    
    @classmethod
    def okta(
        cls,
        client_id: str,
        client_secret: str,
        domain: str,
        role_mapping: Optional[Dict[str, str]] = None
    ):
        """Create Okta configuration."""
        base_url = f"https://{domain}"
        return cls(
            provider_type=ProviderType.OKTA,
            client_id=client_id,
            client_secret=client_secret,
            issuer_url=base_url,
            auth_url=f"{base_url}/oauth2/v1/authorize",
            token_url=f"{base_url}/oauth2/v1/token",
            userinfo_url=f"{base_url}/oauth2/v1/userinfo",
            jwks_url=f"{base_url}/oauth2/v1/keys",
            scopes=["openid", "profile", "email", "groups"],
            role_mapping=role_mapping
        )


class EnterpriseSSOService:
    """
    Enterprise SSO service for OAuth2/OIDC integration.
    
    Handles authentication with enterprise identity providers,
    user provisioning, role mapping, and audit logging.
    """
    
    def __init__(self):
        self.providers: Dict[str, ProviderConfig] = {}
        self.auth_service: Optional[AuthService] = None
        self.redis = None
        self.is_initialized = False
        
        # OAuth2 settings
        self.state_expire_minutes = 10
        self.code_challenge_method = "S256"
        self.token_cache_ttl = 3600  # 1 hour
        
        # Audit settings
        self.audit_events = []
        self.max_audit_events = 10000
        
    async def initialize(self, auth_service: AuthService) -> None:
        """Initialize the enterprise SSO service."""
        try:
            self.auth_service = auth_service
            self.redis = await get_redis()
            
            # Load provider configurations from environment
            await self._load_provider_configs()
            
            self.is_initialized = True
            logger.info("Enterprise SSO service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enterprise SSO service: {str(e)}")
            raise
    
    async def _load_provider_configs(self) -> None:
        """Load identity provider configurations from environment variables."""
        
        # Azure AD configuration
        azure_client_id = getattr(settings, 'azure_ad_client_id', None)
        azure_client_secret = getattr(settings, 'azure_ad_client_secret', None)
        azure_tenant_id = getattr(settings, 'azure_ad_tenant_id', None)
        
        if azure_client_id and azure_client_secret and azure_tenant_id:
            azure_role_mapping = {
                "Global Administrator": UserRole.ADMIN,
                "Application Administrator": UserRole.ADMIN,
                "User": UserRole.USER,
                "Guest": UserRole.VIEWER
            }
            
            self.providers["azure_ad"] = ProviderConfig.azure_ad(
                client_id=azure_client_id,
                client_secret=azure_client_secret,
                tenant_id=azure_tenant_id,
                role_mapping=azure_role_mapping
            )
            logger.info("Azure AD provider configured")
        
        # Okta configuration
        okta_client_id = getattr(settings, 'okta_client_id', None)
        okta_client_secret = getattr(settings, 'okta_client_secret', None)
        okta_domain = getattr(settings, 'okta_domain', None)
        
        if okta_client_id and okta_client_secret and okta_domain:
            okta_role_mapping = {
                "Admin": UserRole.ADMIN,
                "Manager": UserRole.USER,
                "User": UserRole.USER,
                "Guest": UserRole.VIEWER
            }
            
            self.providers["okta"] = ProviderConfig.okta(
                client_id=okta_client_id,
                client_secret=okta_client_secret,
                domain=okta_domain,
                role_mapping=okta_role_mapping
            )
            logger.info("Okta provider configured")
    
    # OAuth2/OIDC Flow Implementation
    
    def generate_auth_url(
        self,
        provider_name: str,
        redirect_uri: str,
        state: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Generate OAuth2 authorization URL for the specified provider.
        
        Returns:
            Tuple of (auth_url, state) where state should be stored for validation
        """
        
        provider = self.providers.get(provider_name)
        if not provider:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Provider '{provider_name}' not configured"
            )
        
        # Generate state for CSRF protection
        if not state:
            state = secrets.token_urlsafe(32)
        
        # Generate code challenge for PKCE (optional but recommended)
        code_verifier = secrets.token_urlsafe(96)
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).decode().rstrip('=')
        
        # Build authorization parameters
        auth_params = {
            "client_id": provider.client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": " ".join(provider.scopes),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": self.code_challenge_method,
            "prompt": "select_account"  # Force account selection
        }
        
        # Provider-specific parameters
        if provider.provider_type == ProviderType.AZURE_AD:
            auth_params["response_mode"] = "query"
        
        # Construct authorization URL
        auth_url = f"{provider.auth_url}?{urlencode(auth_params)}"
        
        # Store state and code_verifier in Redis for later validation
        asyncio.create_task(self._store_oauth_state(
            state, provider_name, redirect_uri, code_verifier
        ))
        
        logger.info(
            "Generated OAuth2 authorization URL",
            provider=provider_name,
            state=state[:8] + "..."
        )
        
        return auth_url, state
    
    async def handle_oauth_callback(
        self,
        provider_name: str,
        code: str,
        state: str,
        redirect_uri: str
    ) -> Dict[str, Any]:
        """
        Handle OAuth2 callback and exchange code for tokens.
        
        Returns user information and tokens from the identity provider.
        """
        
        provider = self.providers.get(provider_name)
        if not provider:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Provider '{provider_name}' not configured"
            )
        
        # Validate state parameter
        stored_state = await self._get_oauth_state(state)
        if not stored_state:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired state parameter"
            )
        
        if stored_state["provider"] != provider_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="State parameter provider mismatch"
            )
        
        try:
            # Exchange authorization code for tokens
            token_data = await self._exchange_code_for_tokens(
                provider, code, redirect_uri, stored_state["code_verifier"]
            )
            
            # Get user information from the provider
            user_info = await self._get_user_info(provider, token_data["access_token"])
            
            # Clean up state
            await self._cleanup_oauth_state(state)
            
            # Log successful OAuth callback
            await self._log_audit_event(
                "oauth_callback_success",
                {
                    "provider": provider_name,
                    "user_email": user_info.get("email"),
                    "user_id": user_info.get("sub"),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            return {
                "provider": provider_name,
                "user_info": user_info,
                "tokens": token_data
            }
            
        except Exception as e:
            # Log failed OAuth callback
            await self._log_audit_event(
                "oauth_callback_failed",
                {
                    "provider": provider_name,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.error(f"OAuth callback failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"OAuth callback failed: {str(e)}"
            )
    
    async def authenticate_or_create_user(
        self,
        provider_name: str,
        user_info: Dict[str, Any],
        tokens: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Authenticate existing user or create new user from SSO information.
        
        Handles user provisioning, role mapping, and token generation.
        """
        
        provider = self.providers.get(provider_name)
        if not provider:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Provider '{provider_name}' not configured"
            )
        
        email = user_info.get("email")
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email address not provided by identity provider"
            )
        
        try:
            # Check if user already exists
            existing_user = await self.auth_service.get_user_by_email(email)
            
            if existing_user:
                # Update existing user's last login
                await self.auth_service._update_last_login(existing_user["id"])
                
                # Log successful SSO authentication
                await self._log_audit_event(
                    "sso_authentication_success",
                    {
                        "provider": provider_name,
                        "user_email": email,
                        "user_id": existing_user["id"],
                        "existing_user": True,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                # Generate our own tokens
                token_data = {
                    "sub": existing_user["id"],
                    "email": existing_user["email"],
                    "role": existing_user["role"],
                    "full_name": existing_user["full_name"],
                    "sso_provider": provider_name
                }
                
                access_token = self.auth_service.create_access_token(token_data)
                refresh_token = self.auth_service.create_refresh_token(existing_user["id"])
                
                # Store refresh token
                await self.auth_service._store_refresh_token(existing_user["id"], refresh_token)
                
                return {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "token_type": "bearer",
                    "expires_in": self.auth_service.token_expire_minutes * 60,
                    "user": {
                        "id": existing_user["id"],
                        "email": existing_user["email"],
                        "full_name": existing_user["full_name"],
                        "company": existing_user["company"],
                        "role": existing_user["role"],
                        "is_active": existing_user["is_active"],
                        "sso_provider": provider_name
                    }
                }
            
            else:
                # Create new user from SSO information
                mapped_role = self._map_user_role(provider, user_info)
                
                # Extract user details from provider info
                full_name = self._extract_full_name(user_info)
                company = self._extract_company(user_info)
                
                # Create user (SSO users don't need passwords)
                user_create = UserCreate(
                    email=email,
                    full_name=full_name,
                    company=company,
                    role=mapped_role,
                    password=secrets.token_urlsafe(32),  # Random password, won't be used
                    is_active=True
                )
                
                new_user = await self.auth_service.create_user(user_create)
                
                # Log successful user provisioning
                await self._log_audit_event(
                    "sso_user_provisioned",
                    {
                        "provider": provider_name,
                        "user_email": email,
                        "user_id": new_user.id,
                        "role": mapped_role,
                        "full_name": full_name,
                        "company": company,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                # Generate tokens for new user
                token_data = {
                    "sub": new_user.id,
                    "email": new_user.email,
                    "role": new_user.role,
                    "full_name": new_user.full_name,
                    "sso_provider": provider_name
                }
                
                access_token = self.auth_service.create_access_token(token_data)
                refresh_token = self.auth_service.create_refresh_token(new_user.id)
                
                # Store refresh token
                await self.auth_service._store_refresh_token(new_user.id, refresh_token)
                
                return {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "token_type": "bearer",
                    "expires_in": self.auth_service.token_expire_minutes * 60,
                    "user": {
                        "id": new_user.id,
                        "email": new_user.email,
                        "full_name": new_user.full_name,
                        "company": new_user.company,
                        "role": new_user.role,
                        "is_active": new_user.is_active,
                        "sso_provider": provider_name
                    }
                }
                
        except Exception as e:
            # Log authentication failure
            await self._log_audit_event(
                "sso_authentication_failed",
                {
                    "provider": provider_name,
                    "user_email": email,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.error(f"SSO authentication failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication failed"
            )
    
    # Helper Methods
    
    async def _exchange_code_for_tokens(
        self,
        provider: ProviderConfig,
        code: str,
        redirect_uri: str,
        code_verifier: str
    ) -> Dict[str, Any]:
        """Exchange authorization code for access and ID tokens."""
        
        token_data = {
            "grant_type": "authorization_code",
            "client_id": provider.client_id,
            "client_secret": provider.client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                provider.token_url,
                data=token_data,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json"
                }
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Token exchange failed: {error_text}"
                    )
                
                return await response.json()
    
    async def _get_user_info(
        self,
        provider: ProviderConfig,
        access_token: str
    ) -> Dict[str, Any]:
        """Get user information from the identity provider."""
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                provider.userinfo_url,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json"
                }
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to get user info: {error_text}"
                    )
                
                return await response.json()
    
    def _map_user_role(
        self,
        provider: ProviderConfig,
        user_info: Dict[str, Any]
    ) -> str:
        """Map provider role/group information to internal user role."""
        
        # Extract roles/groups from user info
        roles = []
        
        if provider.provider_type == ProviderType.AZURE_AD:
            # Azure AD roles might be in different locations
            roles.extend(user_info.get("roles", []))
            roles.extend(user_info.get("groups", []))
            
        elif provider.provider_type == ProviderType.OKTA:
            # Okta groups
            roles.extend(user_info.get("groups", []))
            
        # Apply role mapping
        for role in roles:
            mapped_role = provider.role_mapping.get(role)
            if mapped_role:
                return mapped_role
        
        # Default role if no mapping found
        return UserRole.USER
    
    def _extract_full_name(self, user_info: Dict[str, Any]) -> str:
        """Extract full name from user info."""
        
        # Try different name fields
        name_fields = ["name", "displayName", "full_name", "fullName"]
        
        for field in name_fields:
            if field in user_info and user_info[field]:
                return user_info[field]
        
        # Fallback to first + last name
        given_name = user_info.get("given_name", user_info.get("givenName", ""))
        family_name = user_info.get("family_name", user_info.get("familyName", ""))
        
        if given_name and family_name:
            return f"{given_name} {family_name}"
        
        # Final fallback to email prefix
        email = user_info.get("email", "")
        if email:
            return email.split("@")[0]
        
        return "SSO User"
    
    def _extract_company(self, user_info: Dict[str, Any]) -> Optional[str]:
        """Extract company information from user info."""
        
        company_fields = ["companyName", "company", "organization", "org"]
        
        for field in company_fields:
            if field in user_info and user_info[field]:
                return user_info[field]
        
        return None
    
    # State Management
    
    async def _store_oauth_state(
        self,
        state: str,
        provider: str,
        redirect_uri: str,
        code_verifier: str
    ) -> None:
        """Store OAuth state information in Redis."""
        
        if not self.redis:
            return
        
        state_data = {
            "provider": provider,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
            "timestamp": time.time()
        }
        
        try:
            await self.redis.setex(
                f"oauth_state:{state}",
                self.state_expire_minutes * 60,
                json.dumps(state_data)
            )
        except Exception as e:
            logger.error(f"Failed to store OAuth state: {str(e)}")
    
    async def _get_oauth_state(self, state: str) -> Optional[Dict[str, Any]]:
        """Get OAuth state information from Redis."""
        
        if not self.redis:
            return None
        
        try:
            state_json = await self.redis.get(f"oauth_state:{state}")
            if state_json:
                return json.loads(state_json.decode())
        except Exception as e:
            logger.error(f"Failed to get OAuth state: {str(e)}")
        
        return None
    
    async def _cleanup_oauth_state(self, state: str) -> None:
        """Clean up OAuth state from Redis."""
        
        if not self.redis:
            return
        
        try:
            await self.redis.delete(f"oauth_state:{state}")
        except Exception as e:
            logger.error(f"Failed to cleanup OAuth state: {str(e)}")
    
    # Audit Logging
    
    async def _log_audit_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Log audit event for compliance and monitoring."""
        
        audit_event = {
            "event_type": event_type,
            "event_data": event_data,
            "timestamp": datetime.utcnow().isoformat(),
            "event_id": str(uuid.uuid4())
        }
        
        # Store in memory (in production, this should go to a persistent audit log)
        self.audit_events.append(audit_event)
        
        # Keep only recent events to prevent memory bloat
        if len(self.audit_events) > self.max_audit_events:
            self.audit_events = self.audit_events[-self.max_audit_events:]
        
        # Log to structured logger
        logger.info(
            f"SSO Audit Event: {event_type}",
            audit_event=audit_event
        )
        
        # Store in Redis for persistence (optional)
        if self.redis:
            try:
                await self.redis.lpush(
                    "sso_audit_events",
                    json.dumps(audit_event)
                )
                # Keep only last 1000 events
                await self.redis.ltrim("sso_audit_events", 0, 999)
            except Exception as e:
                logger.error(f"Failed to store audit event in Redis: {str(e)}")
    
    # Provider Management
    
    def list_providers(self) -> List[Dict[str, Any]]:
        """List all configured identity providers."""
        
        return [
            {
                "name": name,
                "type": provider.provider_type,
                "scopes": provider.scopes,
                "configured": True
            }
            for name, provider in self.providers.items()
        ]
    
    def get_provider_info(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific provider."""
        
        provider = self.providers.get(provider_name)
        if not provider:
            return None
        
        return {
            "name": provider_name,
            "type": provider.provider_type,
            "scopes": provider.scopes,
            "auth_url": provider.auth_url,
            "issuer_url": provider.issuer_url,
            "configured": True
        }
    
    # Health Check
    
    async def health_check(self) -> Dict[str, Any]:
        """Check enterprise SSO service health."""
        
        try:
            return {
                "status": "healthy" if self.is_initialized else "initializing",
                "initialized": self.is_initialized,
                "providers_configured": len(self.providers),
                "providers": list(self.providers.keys()),
                "redis_connected": self.redis is not None,
                "audit_events_count": len(self.audit_events)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Global enterprise SSO service instance
enterprise_sso_service = EnterpriseSSOService()


async def get_enterprise_sso_service() -> EnterpriseSSOService:
    """Get the global enterprise SSO service instance."""
    if not enterprise_sso_service.is_initialized:
        from services.auth_service import get_auth_service
        auth_service = await get_auth_service()
        await enterprise_sso_service.initialize(auth_service)
    return enterprise_sso_service