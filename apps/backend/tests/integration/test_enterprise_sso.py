"""
Integration tests for Enterprise SSO functionality.

Tests OAuth2/OIDC flows, user provisioning, role mapping,
and audit logging for Azure AD and Okta integrations.
"""
import uuid
from tests.utilities.test_data_factory import test_data_factory

import asyncio
import json
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any

from fastapi.testclient import TestClient
from fastapi import status

from services.enterprise_sso import EnterpriseSSOService, ProviderConfig, ProviderType
from services.auth_service import AuthService, UserRole
from api.endpoints.enterprise_api import router


class TestEnterpriseSSOService:
    """Test Enterprise SSO Service functionality."""
    
    @pytest.fixture
    async def sso_service(self):
        """Create SSO service instance for testing."""
        service = EnterpriseSSOService()
        
        # Mock auth service
        auth_service = Mock(spec=AuthService)
        auth_service.get_user_by_email = AsyncMock(return_value=None)
        auth_service.create_user = AsyncMock()
        auth_service.create_access_token = Mock(return_value="test_access_token")
        auth_service.create_refresh_token = Mock(return_value="test_refresh_token")
        auth_service._store_refresh_token = AsyncMock()
        auth_service._update_last_login = AsyncMock()
        auth_service.token_expire_minutes = 30
        
        # Mock Redis
        redis_mock = AsyncMock()
        
        # Initialize service
        service.auth_service = auth_service
        service.redis = redis_mock
        service.is_initialized = True
        
        # Add test providers
        service.providers = {
            "azure_ad": ProviderConfig.azure_ad(
                client_id="test_azure_client_id",
                client_secret="test_azure_client_secret", 
                tenant_id="test_tenant_id"
            ),
            "okta": ProviderConfig.okta(
                client_id="test_okta_client_id",
                client_secret="test_okta_client_secret",
                domain="dev-test.okta.com"
            )
        }
        
        return service
    
    @pytest.mark.asyncio
    async def test_azure_ad_provider_configuration(self, sso_service):
        """Test Azure AD provider configuration."""
        
        provider = sso_service.providers["azure_ad"]
        
        assert provider.provider_type == ProviderType.AZURE_AD
        assert provider.client_id == "test_azure_client_id"
        assert provider.tenant_id == "test_tenant_id"
        assert "https://login.microsoftonline.com" in provider.auth_url
        assert "User.Read" in provider.scopes
    
    @pytest.mark.asyncio
    async def test_okta_provider_configuration(self, sso_service):
        """Test Okta provider configuration."""
        
        provider = sso_service.providers["okta"]
        
        assert provider.provider_type == ProviderType.OKTA
        assert provider.client_id == "test_okta_client_id"
        assert "dev-test.okta.com" in provider.auth_url
        assert "groups" in provider.scopes
    
    @pytest.mark.asyncio
    async def test_generate_auth_url(self, sso_service):
        """Test OAuth2 authorization URL generation."""
        
        redirect_uri = "https://app.example.com/auth/callback"
        
        auth_url, state = sso_service.generate_auth_url(
            provider_name="azure_ad",
            redirect_uri=redirect_uri
        )
        
        assert "login.microsoftonline.com" in auth_url
        assert "client_id=test_azure_client_id" in auth_url
        assert "redirect_uri=" + redirect_uri.replace(":", "%3A").replace("/", "%2F") in auth_url
        assert "response_type=code" in auth_url
        assert f"state={state}" in auth_url
        assert "code_challenge=" in auth_url
        assert len(state) > 10  # State should be sufficiently long
    
    @pytest.mark.asyncio 
    async def test_generate_auth_url_invalid_provider(self, sso_service):
        """Test auth URL generation with invalid provider."""
        
        with pytest.raises(Exception) as exc_info:
            sso_service.generate_auth_url(
                provider_name="invalid_provider",
                redirect_uri="https://app.example.com/callback"
            )
        
        assert "not configured" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_oauth_state_management(self, sso_service):
        """Test OAuth state storage and retrieval."""
        
        state = "test_state_12345"
        provider = "azure_ad"
        redirect_uri = "https://app.example.com/callback"
        code_verifier = "test_code_verifier"
        
        # Store state
        await sso_service._store_oauth_state(state, provider, redirect_uri, code_verifier)
        
        # Verify Redis was called correctly
        sso_service.redis.setex.assert_called_once()
        call_args = sso_service.redis.setex.call_args
        assert call_args[0][0] == f"oauth_state:{state}"
        assert call_args[0][1] == 600  # 10 minutes
        
        # Test state retrieval
        sso_service.redis.get.return_value = json.dumps({
            "provider": provider,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
            "timestamp": 1234567890
        }).encode()
        
        retrieved_state = await sso_service._get_oauth_state(state)
        
        assert retrieved_state["provider"] == provider
        assert retrieved_state["redirect_uri"] == redirect_uri
        assert retrieved_state["code_verifier"] == code_verifier
    
    @pytest.mark.asyncio
    async def test_user_role_mapping_azure_ad(self, sso_service):
        """Test user role mapping for Azure AD."""
        
        provider = sso_service.providers["azure_ad"]
        
        # Test admin role mapping
        user_info_admin = {
            "roles": ["Global Administrator"],
            "email": "admin@strategic-planning.ai"
        }
        
        role = sso_service._map_user_role(provider, user_info_admin)
        assert role == UserRole.ADMIN
        
        # Test default user role
        user_info_user = {
            "roles": [],
            email=f"testuser{str(uuid.uuid4())[:8]}@{self.fake.domain_name()}","
        }
        
        role = sso_service._map_user_role(provider, user_info_user)
        assert role == UserRole.USER
    
    @pytest.mark.asyncio
    async def test_user_role_mapping_okta(self, sso_service):
        """Test user role mapping for Okta."""
        
        provider = sso_service.providers["okta"]
        
        # Test admin role mapping
        user_info_admin = {
            "groups": ["Admin"],
            "email": "admin@strategic-planning.ai"
        }
        
        role = sso_service._map_user_role(provider, user_info_admin)
        assert role == UserRole.ADMIN
        
        # Test viewer role mapping
        user_info_viewer = {
            "groups": ["Guest"],
            email=f"testuser{str(uuid.uuid4())[:8]}@{self.fake.domain_name()}","
        }
        
        role = sso_service._map_user_role(provider, user_info_viewer)
        assert role == UserRole.VIEWER
    
    @pytest.mark.asyncio
    async def test_extract_user_info(self, sso_service):
        """Test user information extraction from provider data."""
        
        # Test full name extraction
        user_info = {
            name=self.fake.name(),",
            email=f"testuser{str(uuid.uuid4())[:8]}@{self.fake.domain_name()}",",
            "companyName": "Acme Corp"
        }
        
        full_name = sso_service._extract_full_name(user_info)
        assert full_name == "Joshua Lawson"
        
        company = sso_service._extract_company(user_info)
        assert company == "Acme Corp"
        
        # Test fallback name extraction
        user_info_fallback = {
            "given_name": "Jane",
            "family_name": "Smith",
            email=f"testuser{str(uuid.uuid4())[:8]}@{self.fake.domain_name()}","
        }
        
        full_name = sso_service._extract_full_name(user_info_fallback)
        assert full_name == "Jane Smith"
    
    @pytest.mark.asyncio
    async def test_audit_logging(self, sso_service):
        """Test audit event logging."""
        
        event_type = "test_sso_event"
        event_data = {
            "user_email": "user@company.local",
            "provider": "azure_ad",
            "action": "login"
        }
        
        # Log audit event
        await sso_service._log_audit_event(event_type, event_data)
        
        # Verify event was stored
        assert len(sso_service.audit_events) == 1
        audit_event = sso_service.audit_events[0]
        
        assert audit_event["event_type"] == event_type
        assert audit_event["event_data"]["user_email"] == "user@company.local"
        assert audit_event["event_data"]["provider"] == "azure_ad"
        assert "event_id" in audit_event
        assert "timestamp" in audit_event
        
        # Verify Redis logging
        sso_service.redis.lpush.assert_called_once()
        sso_service.redis.ltrim.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_authenticate_existing_user(self, sso_service):
        """Test authentication of existing user."""
        
        # Mock existing user
        existing_user = {
            "id": "user123",
            email=f"testuser{str(uuid.uuid4())[:8]}@{self.fake.domain_name()}",",
            "full_name": "Existing User",
            "company": "Test Corp",
            "role": "user",
            "is_active": True
        }
        
        sso_service.auth_service.get_user_by_email.return_value = existing_user
        
        user_info = {
            email=f"testuser{str(uuid.uuid4())[:8]}@{self.fake.domain_name()}",",
            "name": "Existing User",
            "sub": "ext_user_123"
        }
        
        tokens = {
            "access_token": "provider_access_token",
            "id_token": "provider_id_token"
        }
        
        result = await sso_service.authenticate_or_create_user(
            provider_name="azure_ad",
            user_info=user_info,
            tokens=tokens
        )
        
        # Verify authentication result
        assert result["access_token"] == "test_access_token"
        assert result["refresh_token"] == "test_refresh_token"
        assert result["user"]["id"] == "user123"
        assert result["user"]["sso_provider"] == "azure_ad"
        
        # Verify audit event was logged
        assert len(sso_service.audit_events) == 1
        assert sso_service.audit_events[0]["event_type"] == "sso_authentication_success"
    
    @pytest.mark.asyncio
    async def test_create_new_user_from_sso(self, sso_service):
        """Test creation of new user from SSO information."""
        
        # Mock new user creation
        new_user = Mock()
        new_user.id = "new_user_123"
        new_user.email = "newuserexample.com"
        new_user.full_name = "New User"
        new_user.company = "New Corp"
        new_user.role = "user"
        new_user.is_active = True
        
        sso_service.auth_service.get_user_by_email.return_value = None
        sso_service.auth_service.create_user.return_value = new_user
        
        user_info = {
            email=f"testuser{str(uuid.uuid4())[:8]}@{self.fake.domain_name()}",",
            "name": "New User",
            "companyName": "New Corp",
            "sub": "new_ext_user_123"
        }
        
        tokens = {
            "access_token": "provider_access_token",
            "id_token": "provider_id_token"
        }
        
        result = await sso_service.authenticate_or_create_user(
            provider_name="okta",
            user_info=user_info,
            tokens=tokens
        )
        
        # Verify user creation
        sso_service.auth_service.create_user.assert_called_once()
        created_user_data = sso_service.auth_service.create_user.call_args[0][0]
        assert created_user_data.email == "newuserexample.com"
        assert created_user_data.full_name == "New User"
        assert created_user_data.company == "New Corp"
        
        # Verify authentication result
        assert result["access_token"] == "test_access_token"
        assert result["user"]["id"] == "new_user_123"
        assert result["user"]["sso_provider"] == "okta"
        
        # Verify audit events
        assert len(sso_service.audit_events) == 1
        assert sso_service.audit_events[0]["event_type"] == "sso_user_provisioned"
    
    @pytest.mark.asyncio
    async def test_health_check(self, sso_service):
        """Test SSO service health check."""
        
        health_status = await sso_service.health_check()
        
        assert health_status["status"] == "healthy"
        assert health_status["initialized"] is True
        assert health_status["providers_configured"] == 2
        assert "azure_ad" in health_status["providers"]
        assert "okta" in health_status["providers"]


class TestEnterpriseAPIEndpoints:
    """Test Enterprise API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)
    
    @pytest.fixture
    def mock_sso_service(self):
        """Mock SSO service."""
        service = Mock(spec=EnterpriseSSOService)
        
        # Mock provider info
        service.get_provider_info.return_value = {
            "name": "azure_ad",
            "type": "azure_ad",
            "scopes": ["openid", "profile", "email"],
            "configured": True
        }
        
        # Mock auth URL generation
        service.generate_auth_url.return_value = (
            "https://login.microsoftonline.com/test/oauth2/v2.0/authorize?...",
            "test_state_123"
        )
        
        # Mock provider list
        service.list_providers.return_value = [
            {
                "name": "azure_ad",
                "type": "azure_ad",
                "scopes": ["openid", "profile", "email"],
                "configured": True
            }
        ]
        
        # Mock audit events
        service.audit_events = [
            {
                "event_id": "event_123",
                "event_type": "sso_authentication_success",
                "timestamp": "2024-03-15T10:00:00Z",
                "event_data": {
                    "provider": "azure_ad",
                    "user_email": "user@company.local"
                }
            }
        ]
        
        return service
    
    @patch('api.endpoints.enterprise_api.get_enterprise_sso_service')
    def test_list_identity_providers(self, mock_get_service, client, mock_sso_service):
        """Test listing identity providers."""
        
        mock_get_service.return_value = mock_sso_service
        
        response = client.get("/sso/providers")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert len(data) == 1
        assert data[0]["name"] == "azure_ad"
        assert data[0]["type"] == "azure_ad"
        assert data[0]["configured"] is True
    
    @patch('api.endpoints.enterprise_api.get_enterprise_sso_service')
    def test_get_provider_info(self, mock_get_service, client, mock_sso_service):
        """Test getting specific provider information."""
        
        mock_get_service.return_value = mock_sso_service
        
        response = client.get("/sso/providers/azure_ad")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["name"] == "azure_ad"
        assert data["type"] == "azure_ad"
        assert data["configured"] is True
    
    @patch('api.endpoints.enterprise_api.get_enterprise_sso_service')
    def test_initiate_sso(self, mock_get_service, client, mock_sso_service):
        """Test SSO initiation."""
        
        mock_get_service.return_value = mock_sso_service
        
        request_data = {
            "provider": "azure_ad",
            "redirect_uri": "https://app.example.com/auth/callback"
        }
        
        response = client.post("/sso/initiate", json=request_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "authorization_url" in data
        assert "state" in data
        assert data["provider"] == "azure_ad"
        assert data["expires_in"] == 600
    
    @patch('api.endpoints.enterprise_api.get_enterprise_sso_service')
    @patch('api.endpoints.enterprise_api.require_admin')
    def test_get_audit_events(self, mock_require_admin, mock_get_service, client, mock_sso_service):
        """Test retrieving audit events."""
        
        # Mock admin user
        mock_admin = Mock()
        mock_admin.id = "admin_123"
        mock_admin.email = "admin@strategic-planning.ai"
        mock_require_admin.return_value = mock_admin
        
        mock_get_service.return_value = mock_sso_service
        
        response = client.get("/enterprise/audit/events")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert len(data) == 1
        assert data[0]["event_type"] == "sso_authentication_success"
        assert data[0]["event_id"] == "event_123"
        assert data[0]["event_data"]["provider"] == "azure_ad"
    
    @patch('api.endpoints.enterprise_api.get_current_user')
    def test_list_enterprise_projects(self, mock_get_user, client):
        """Test listing enterprise projects."""
        
        # Mock current user
        mock_user = Mock()
        mock_user.id = "user_123"
        mock_user.email = "userexample.com"
        mock_get_user.return_value = mock_user
        
        response = client.get("/enterprise/projects")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "projects" in data
        assert "total" in data
        assert data["user"] == "userexample.com"
        assert len(data["projects"]) >= 2  # Mock projects
    
    @patch('api.endpoints.enterprise_api.get_enterprise_sso_service')
    def test_enterprise_health_check(self, mock_get_service, client, mock_sso_service):
        """Test enterprise API health check."""
        
        mock_sso_service.health_check.return_value = {
            "status": "healthy",
            "initialized": True,
            "providers_configured": 2
        }
        
        mock_get_service.return_value = mock_sso_service
        
        response = client.get("/enterprise/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "enterprise_sso" in data
        assert data["enterprise_sso"]["initialized"] is True


class TestSSOIntegrationFlows:
    """Test complete SSO integration flows."""
    
    @pytest.mark.asyncio
    async def test_complete_azure_ad_flow(self):
        """Test complete Azure AD authentication flow."""
        
        # This would be an end-to-end test in a real scenario
        # For now, we'll test the major components working together
        
        sso_service = EnterpriseSSOService()
        
        # Mock dependencies
        auth_service = Mock(spec=AuthService)
        redis_mock = AsyncMock()
        
        sso_service.auth_service = auth_service
        sso_service.redis = redis_mock
        sso_service.is_initialized = True
        
        # Add Azure AD provider
        sso_service.providers["azure_ad"] = ProviderConfig.azure_ad(
            client_id="test_client_id",
            client_secret="test_client_secret",
            tenant_id="test_tenant_id"
        )
        
        # Step 1: Generate authorization URL
        auth_url, state = sso_service.generate_auth_url(
            provider_name="azure_ad",
            redirect_uri="https://app.example.com/callback"
        )
        
        assert "login.microsoftonline.com" in auth_url
        assert len(state) > 10
        
        # Step 2: Mock OAuth callback handling
        with patch.object(sso_service, '_exchange_code_for_tokens') as mock_exchange, \
             patch.object(sso_service, '_get_user_info') as mock_get_info, \
             patch.object(sso_service, '_get_oauth_state') as mock_get_state, \
             patch.object(sso_service, '_cleanup_oauth_state') as mock_cleanup:
            
            # Mock stored state
            mock_get_state.return_value = {
                "provider": "azure_ad",
                "redirect_uri": "https://app.example.com/callback",
                "code_verifier": "test_verifier"
            }
            
            # Mock token exchange
            mock_exchange.return_value = {
                "access_token": "azure_access_token",
                "id_token": "azure_id_token",
                "expires_in": 3600
            }
            
            # Mock user info
            mock_get_info.return_value = {
                "sub": "azure_user_123",
                "email": "testuser@company.com",
                "name": "Test User",
                "companyName": "Test Company"
            }
            
            # Mock new user creation
            new_user = Mock()
            new_user.id = "internal_user_123"
            new_user.email = "testuser@company.com"
            new_user.full_name = "Test User"
            new_user.company = "Test Company"
            new_user.role = "user"
            new_user.is_active = True
            
            auth_service.get_user_by_email.return_value = None
            auth_service.create_user.return_value = new_user
            auth_service.create_access_token.return_value = "internal_access_token"
            auth_service.create_refresh_token.return_value = "internal_refresh_token"
            auth_service._store_refresh_token = AsyncMock()
            auth_service.token_expire_minutes = 30
            
            # Process callback
            oauth_result = await sso_service.handle_oauth_callback(
                provider_name="azure_ad",
                code="test_auth_code",
                state=state,
                redirect_uri="https://app.example.com/callback"
            )
            
            auth_result = await sso_service.authenticate_or_create_user(
                provider_name="azure_ad",
                user_info=oauth_result["user_info"],
                tokens=oauth_result["tokens"]
            )
            
            # Verify complete flow
            assert auth_result["access_token"] == "internal_access_token"
            assert auth_result["user"]["email"] == "testuser@company.com"
            assert auth_result["user"]["sso_provider"] == "azure_ad"
            
            # Verify audit events were logged
            assert len(sso_service.audit_events) == 2  # callback success + user provisioned
            
            # Verify cleanup was called
            mock_cleanup.assert_called_once_with(state)
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_state(self):
        """Test error handling for invalid OAuth state."""
        
        sso_service = EnterpriseSSOService()
        sso_service.is_initialized = True
        sso_service.providers["azure_ad"] = ProviderConfig.azure_ad(
            client_id="test", client_secret="test", tenant_id="test"
        )
        
        with patch.object(sso_service, '_get_oauth_state', return_value=None):
            with pytest.raises(Exception) as exc_info:
                await sso_service.handle_oauth_callback(
                    provider_name="azure_ad",
                    code="test_code",
                    state="invalid_state",
                    redirect_uri="https://app.example.com/callback"
                )
            
            assert "Invalid or expired state" in str(exc_info.value)
    
    @pytest.mark.asyncio  
    async def test_concurrent_authentication_requests(self):
        """Test handling concurrent authentication requests."""
        
        sso_service = EnterpriseSSOService()
        sso_service.is_initialized = True
        sso_service.providers["azure_ad"] = ProviderConfig.azure_ad(
            client_id="test", client_secret="test", tenant_id="test"
        )
        sso_service.redis = AsyncMock()
        
        # Generate multiple auth URLs concurrently
        tasks = []
        for i in range(5):
            task = sso_service.generate_auth_url(
                provider_name="azure_ad",
                redirect_uri=f"https://app.example.com/callback/{i}"
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all requests succeeded
        for result in results:
            assert not isinstance(result, Exception)
            auth_url, state = result
            assert "login.microsoftonline.com" in auth_url
            assert len(state) > 10
        
        # Verify all states are unique
        states = [result[1] for result in results]
        assert len(set(states)) == 5  # All states should be unique