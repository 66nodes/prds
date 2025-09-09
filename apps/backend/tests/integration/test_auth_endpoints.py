"""
Integration tests for authentication endpoints.
"""
import uuid
from tests.utilities.test_data_factory import test_data_factory

import pytest
import json
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import jwt

from main import app


class TestAuthEndpoints:
    """Integration tests for authentication API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def test_user_data(self):
        """Test user registration data."""
        return {
            email=f"testuser{str(uuid.uuid4())[:8]}@{self.fake.domain_name()}",",
            "password": "SecureTestPassword123!",
            "name": "Test User"
        }

    @pytest.fixture
    def login_data(self, test_user_data):
        """Login credentials."""
        return {
            "email": test_user_data["email"],
            "password": test_user_data["password"]
        }

    def test_register_user_success(self, client, test_user_data):
        """Test successful user registration."""
        response = client.post("/api/auth/register", json=test_user_data)
        
        assert response.status_code == 201
        data = response.json()
        
        assert data["success"] is True
        assert "data" in data
        assert data["data"]["email"] == test_user_data["email"]
        assert data["data"]["name"] == test_user_data["name"]
        assert "id" in data["data"]
        assert "password" not in data["data"]  # Password should not be returned

    def test_register_user_duplicate_email(self, client, test_user_data):
        """Test user registration with duplicate email."""
        # Register user first time
        client.post("/api/auth/register", json=test_user_data)
        
        # Try to register again with same email
        response = client.post("/api/auth/register", json=test_user_data)
        
        assert response.status_code == 409
        data = response.json()
        assert data["success"] is False
        assert "already exists" in data["message"].lower()

    def test_register_user_invalid_email(self, client):
        """Test user registration with invalid email."""
        invalid_data = {
            "email": "invalid-email",
            "password": "SecurePassword123!",
            "name": "Test User"
        }
        
        response = client.post("/api/auth/register", json=invalid_data)
        
        assert response.status_code == 422
        data = response.json()
        assert "validation error" in data["message"].lower() or "email" in str(data).lower()

    def test_register_user_weak_password(self, client):
        """Test user registration with weak password."""
        weak_password_data = {
            "email": "user@company.local",
            "password": "weak",
            "name": "Test User"
        }
        
        response = client.post("/api/auth/register", json=weak_password_data)
        
        assert response.status_code == 400
        data = response.json()
        assert "password" in data["message"].lower()

    def test_login_success(self, client, test_user_data, login_data):
        """Test successful user login."""
        # Register user first
        client.post("/api/auth/register", json=test_user_data)
        
        # Now login
        response = client.post("/api/auth/login", json=login_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "data" in data
        assert "access_token" in data["data"]
        assert "refresh_token" in data["data"]
        assert "token_type" in data["data"]
        assert data["data"]["token_type"] == "bearer"
        assert "expires_in" in data["data"]

    def test_login_invalid_email(self, client):
        """Test login with non-existent email."""
        login_data = {
            email=f"testuser{str(uuid.uuid4())[:8]}@{self.fake.domain_name()}",",
            "password": "SomePassword123!"
        }
        
        response = client.post("/api/auth/login", json=login_data)
        
        assert response.status_code == 401
        data = response.json()
        assert data["success"] is False
        assert "invalid credentials" in data["message"].lower()

    def test_login_wrong_password(self, client, test_user_data):
        """Test login with wrong password."""
        # Register user first
        client.post("/api/auth/register", json=test_user_data)
        
        # Try login with wrong password
        wrong_login_data = {
            "email": test_user_data["email"],
            "password": "WrongPassword123!"
        }
        
        response = client.post("/api/auth/login", json=wrong_login_data)
        
        assert response.status_code == 401
        data = response.json()
        assert data["success"] is False
        assert "invalid credentials" in data["message"].lower()

    def test_refresh_token_success(self, client, test_user_data, login_data):
        """Test successful token refresh."""
        # Register and login user
        client.post("/api/auth/register", json=test_user_data)
        login_response = client.post("/api/auth/login", json=login_data)
        login_data_response = login_response.json()["data"]
        
        refresh_token = login_data_response["refresh_token"]
        
        # Refresh the token
        response = client.post("/api/auth/refresh", json={"refresh_token": refresh_token})
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "data" in data
        assert "access_token" in data["data"]
        assert "token_type" in data["data"]
        assert data["data"]["token_type"] == "bearer"

    def test_refresh_token_invalid(self, client):
        """Test token refresh with invalid token."""
        invalid_refresh_data = {"refresh_token": "invalid.jwt.token"}
        
        response = client.post("/api/auth/refresh", json=invalid_refresh_data)
        
        assert response.status_code == 401
        data = response.json()
        assert data["success"] is False
        assert "invalid" in data["message"].lower() or "expired" in data["message"].lower()

    def test_get_current_user_success(self, client, test_user_data, login_data):
        """Test getting current user with valid token."""
        # Register and login user
        client.post("/api/auth/register", json=test_user_data)
        login_response = client.post("/api/auth/login", json=login_data)
        access_token = login_response.json()["data"]["access_token"]
        
        # Get current user
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.get("/api/auth/me", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "data" in data
        assert data["data"]["email"] == test_user_data["email"]
        assert data["data"]["name"] == test_user_data["name"]
        assert "id" in data["data"]

    def test_get_current_user_no_token(self, client):
        """Test getting current user without token."""
        response = client.get("/api/auth/me")
        
        assert response.status_code == 401
        data = response.json()
        assert data["success"] is False

    def test_get_current_user_invalid_token(self, client):
        """Test getting current user with invalid token."""
        headers = {"Authorization": "Bearer invalid.jwt.token"}
        response = client.get("/api/auth/me", headers=headers)
        
        assert response.status_code == 401
        data = response.json()
        assert data["success"] is False

    def test_logout_success(self, client, test_user_data, login_data):
        """Test successful logout."""
        # Register and login user
        client.post("/api/auth/register", json=test_user_data)
        login_response = client.post("/api/auth/login", json=login_data)
        access_token = login_response.json()["data"]["access_token"]
        
        # Logout
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.post("/api/auth/logout", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "logged out" in data["message"].lower()

    def test_change_password_success(self, client, test_user_data, login_data):
        """Test successful password change."""
        # Register and login user
        client.post("/api/auth/register", json=test_user_data)
        login_response = client.post("/api/auth/login", json=login_data)
        access_token = login_response.json()["data"]["access_token"]
        
        # Change password
        headers = {"Authorization": f"Bearer {access_token}"}
        change_password_data = {
            "current_password": test_user_data["password"],
            "new_password": "NewSecurePassword456!"
        }
        
        response = client.put("/api/auth/change-password", json=change_password_data, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        
        # Test login with new password
        new_login_data = {
            "email": test_user_data["email"],
            "password": change_password_data["new_password"]
        }
        login_response = client.post("/api/auth/login", json=new_login_data)
        assert login_response.status_code == 200

    def test_change_password_wrong_current(self, client, test_user_data, login_data):
        """Test password change with wrong current password."""
        # Register and login user
        client.post("/api/auth/register", json=test_user_data)
        login_response = client.post("/api/auth/login", json=login_data)
        access_token = login_response.json()["data"]["access_token"]
        
        # Try to change password with wrong current password
        headers = {"Authorization": f"Bearer {access_token}"}
        change_password_data = {
            "current_password": "WrongCurrentPassword123!",
            "new_password": "NewSecurePassword456!"
        }
        
        response = client.put("/api/auth/change-password", json=change_password_data, headers=headers)
        
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False

    def test_request_password_reset_success(self, client, test_user_data):
        """Test successful password reset request."""
        # Register user first
        client.post("/api/auth/register", json=test_user_data)
        
        # Request password reset
        reset_request_data = {"email": test_user_data["email"]}
        response = client.post("/api/auth/request-password-reset", json=reset_request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "reset instructions" in data["message"].lower()

    def test_request_password_reset_nonexistent_email(self, client):
        """Test password reset request with non-existent email."""
        reset_request_data = {email=f"testuser{str(uuid.uuid4())[:8]}@{self.fake.domain_name()}","}
        response = client.post("/api/auth/request-password-reset", json=reset_request_data)
        
        # Should still return success for security reasons (don't reveal if email exists)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_token_expiration(self, client, test_user_data, login_data):
        """Test token expiration handling."""
        # This would require manipulating system time or creating expired tokens
        # For now, we'll test the token structure and basic validation
        
        # Register and login user
        client.post("/api/auth/register", json=test_user_data)
        login_response = client.post("/api/auth/login", json=login_data)
        access_token = login_response.json()["data"]["access_token"]
        
        # Decode token to check expiration
        # Note: In real implementation, we'd need the secret key
        try:
            # This will fail without the secret, but tests the token format
            decoded = jwt.decode(access_token, options={"verify_signature": False})
            assert "exp" in decoded
            assert "sub" in decoded
            assert decoded["sub"] == test_user_data["email"]
        except jwt.InvalidTokenError:
            pytest.fail("Token should be decodable without signature verification")

    def test_concurrent_login_attempts(self, client, test_user_data, login_data):
        """Test handling of concurrent login attempts."""
        import concurrent.futures
        import threading
        
        # Register user first
        client.post("/api/auth/register", json=test_user_data)
        
        def login_attempt():
            return client.post("/api/auth/login", json=login_data)
        
        # Perform concurrent login attempts
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(login_attempt) for _ in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should succeed (no rate limiting expected in basic case)
        for response in results:
            assert response.status_code == 200

    def test_auth_headers_validation(self, client, test_user_data, login_data):
        """Test various authorization header formats."""
        # Register and login user
        client.post("/api/auth/register", json=test_user_data)
        login_response = client.post("/api/auth/login", json=login_data)
        access_token = login_response.json()["data"]["access_token"]
        
        # Test valid header
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.get("/api/auth/me", headers=headers)
        assert response.status_code == 200
        
        # Test invalid header formats
        invalid_headers = [
            {"Authorization": access_token},  # Missing "Bearer "
            {"Authorization": f"Token {access_token}"},  # Wrong scheme
            {"Authorization": f"Bearer"},  # Missing token
            {"Authorization": f"Bearer {access_token} extra"},  # Extra content
        ]
        
        for header in invalid_headers:
            response = client.get("/api/auth/me", headers=header)
            assert response.status_code == 401

    def test_user_profile_update(self, client, test_user_data, login_data):
        """Test user profile update functionality."""
        # Register and login user
        client.post("/api/auth/register", json=test_user_data)
        login_response = client.post("/api/auth/login", json=login_data)
        access_token = login_response.json()["data"]["access_token"]
        
        # Update profile
        headers = {"Authorization": f"Bearer {access_token}"}
        update_data = {
            "name": "Updated Test User",
            "bio": "This is my updated bio"
        }
        
        response = client.put("/api/auth/profile", json=update_data, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["name"] == update_data["name"]

    @pytest.mark.parametrize("invalid_email", [
        "plaintext",
        "@missinglocalpart.com", 
        "missing@.com",
        "missing@domain",
        "spaces in@email.com",
        "too.long.email.address.that.exceeds.maximum.lengthexample.com"
    ])
    def test_register_invalid_email_formats(self, client, invalid_email):
        """Test registration with various invalid email formats."""
        invalid_data = {
            "email": invalid_email,
            "password": "SecurePassword123!",
            "name": "Test User"
        }
        
        response = client.post("/api/auth/register", json=invalid_data)
        assert response.status_code == 422

    @pytest.mark.parametrize("weak_password", [
        "short",
        "nouppercase123",
        "NOLOWERCASE123",
        "NoNumbers!",
        "NoSpecialChars123",
        "a" * 129  # Too long
    ])
    def test_register_weak_passwords(self, client, weak_password):
        """Test registration with various weak passwords."""
        invalid_data = {
            "email": "user@company.local",
            "password": weak_password,
            "name": "Test User"
        }
        
        response = client.post("/api/auth/register", json=invalid_data)
        assert response.status_code == 400