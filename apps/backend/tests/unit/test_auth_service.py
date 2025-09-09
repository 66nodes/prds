"""
Unit tests for authentication service.
"""
import uuid
from tests.utilities.test_data_factory import test_data_factory

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext

from services.auth_service import AuthService, TokenData, User


class TestAuthService:
    """Test suite for AuthService."""

    @pytest.fixture
    def auth_service(self):
        """Create an AuthService instance for testing."""
        return AuthService()

    @pytest.fixture
    def mock_user(self):
        """Mock user data for testing."""
        return {
            "id": "user-123",
            "email": "user@company.local",
            "name": "Test User",
            "hashed_password": "$2b$12$test.hash.here",
            "is_active": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

    def test_hash_password(self, auth_service):
        """Test password hashing."""
        password = "test_password_123"
        hashed = auth_service.hash_password(password)
        
        assert hashed != password
        assert auth_service.verify_password(password, hashed)
        assert not auth_service.verify_password("wrong_password", hashed)

    def test_verify_password_valid(self, auth_service):
        """Test password verification with valid password."""
        password = "test_password_123"
        hashed = auth_service.hash_password(password)
        
        assert auth_service.verify_password(password, hashed)

    def test_verify_password_invalid(self, auth_service):
        """Test password verification with invalid password."""
        password = "test_password_123"
        hashed = auth_service.hash_password(password)
        
        assert not auth_service.verify_password("wrong_password", hashed)

    def test_create_access_token(self, auth_service):
        """Test access token creation."""
        data = {"sub": "userexample.com"}
        token = auth_service.create_access_token(data)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Decode and verify token
        decoded = jwt.decode(token, auth_service.secret_key, algorithms=[auth_service.algorithm])
        assert decoded["sub"] == data["sub"]
        assert "exp" in decoded

    def test_create_access_token_with_expiry(self, auth_service):
        """Test access token creation with custom expiry."""
        data = {"sub": "userexample.com"}
        expires_delta = timedelta(minutes=30)
        token = auth_service.create_access_token(data, expires_delta)
        
        decoded = jwt.decode(token, auth_service.secret_key, algorithms=[auth_service.algorithm])
        
        # Check that expiry is approximately 30 minutes from now
        exp_time = datetime.fromtimestamp(decoded["exp"])
        expected_exp = datetime.utcnow() + expires_delta
        assert abs((exp_time - expected_exp).total_seconds()) < 60  # Within 1 minute tolerance

    def test_create_refresh_token(self, auth_service):
        """Test refresh token creation."""
        data = {"sub": "userexample.com"}
        token = auth_service.create_refresh_token(data)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        decoded = jwt.decode(token, auth_service.secret_key, algorithms=[auth_service.algorithm])
        assert decoded["sub"] == data["sub"]
        assert decoded["type"] == "refresh"

    def test_verify_token_valid(self, auth_service):
        """Test token verification with valid token."""
        data = {"sub": "userexample.com"}
        token = auth_service.create_access_token(data)
        
        token_data = auth_service.verify_token(token)
        assert token_data is not None
        assert token_data.email == data["sub"]

    def test_verify_token_expired(self, auth_service):
        """Test token verification with expired token."""
        data = {"sub": "userexample.com"}
        expires_delta = timedelta(seconds=-1)  # Already expired
        token = auth_service.create_access_token(data, expires_delta)
        
        token_data = auth_service.verify_token(token)
        assert token_data is None

    def test_verify_token_invalid(self, auth_service):
        """Test token verification with invalid token."""
        invalid_token = "invalid.token.here"
        
        token_data = auth_service.verify_token(invalid_token)
        assert token_data is None

    @pytest.mark.asyncio
    async def test_authenticate_user_valid(self, auth_service, mock_user):
        """Test user authentication with valid credentials."""
        email = mock_user["email"]
        password = "test_password_123"
        
        # Hash the test password
        mock_user["hashed_password"] = auth_service.hash_password(password)
        
        with patch.object(auth_service, 'get_user_by_email', new_callable=AsyncMock) as mock_get_user:
            mock_get_user.return_value = mock_user
            
            user = await auth_service.authenticate_user(email, password)
            
            assert user is not None
            assert user["email"] == email
            mock_get_user.assert_called_once_with(email)

    @pytest.mark.asyncio
    async def test_authenticate_user_invalid_email(self, auth_service):
        """Test user authentication with invalid email."""
        email = "nonexistentexample.com"
        password = "test_password_123"
        
        with patch.object(auth_service, 'get_user_by_email', new_callable=AsyncMock) as mock_get_user:
            mock_get_user.return_value = None
            
            user = await auth_service.authenticate_user(email, password)
            
            assert user is None
            mock_get_user.assert_called_once_with(email)

    @pytest.mark.asyncio
    async def test_authenticate_user_invalid_password(self, auth_service, mock_user):
        """Test user authentication with invalid password."""
        email = mock_user["email"]
        password = "wrong_password"
        
        # Hash a different password
        mock_user["hashed_password"] = auth_service.hash_password("correct_password")
        
        with patch.object(auth_service, 'get_user_by_email', new_callable=AsyncMock) as mock_get_user:
            mock_get_user.return_value = mock_user
            
            user = await auth_service.authenticate_user(email, password)
            
            assert user is None

    @pytest.mark.asyncio
    async def test_authenticate_user_inactive(self, auth_service, mock_user):
        """Test user authentication with inactive user."""
        email = mock_user["email"]
        password = "test_password_123"
        
        # Set user as inactive
        mock_user["is_active"] = False
        mock_user["hashed_password"] = auth_service.hash_password(password)
        
        with patch.object(auth_service, 'get_user_by_email', new_callable=AsyncMock) as mock_get_user:
            mock_get_user.return_value = mock_user
            
            user = await auth_service.authenticate_user(email, password)
            
            assert user is None

    @pytest.mark.asyncio
    async def test_register_user_success(self, auth_service):
        """Test successful user registration."""
        user_data = {
            "email": f"testuser{str(uuid.uuid4())[:8]}@{self.fake.domain_name()}",
            "password": "secure_password_123",
            "name": "New User"
        }
        
        with patch.object(auth_service, 'get_user_by_email', new_callable=AsyncMock) as mock_get_user, \
             patch.object(auth_service, 'create_user', new_callable=AsyncMock) as mock_create_user:
            
            # User doesn't exist yet
            mock_get_user.return_value = None
            
            # Mock successful user creation
            created_user = {
                "id": "user-456",
                "email": user_data["email"],
                "name": user_data["name"],
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            mock_create_user.return_value = created_user
            
            user = await auth_service.register_user(user_data)
            
            assert user is not None
            assert user["email"] == user_data["email"]
            assert user["name"] == user_data["name"]
            
            mock_get_user.assert_called_once_with(user_data["email"])
            mock_create_user.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_user_already_exists(self, auth_service, mock_user):
        """Test user registration when user already exists."""
        user_data = {
            "email": mock_user["email"],
            "password": "secure_password_123",
            "name": "Test User"
        }
        
        with patch.object(auth_service, 'get_user_by_email', new_callable=AsyncMock) as mock_get_user:
            mock_get_user.return_value = mock_user
            
            with pytest.raises(ValueError, match="User already exists"):
                await auth_service.register_user(user_data)

    @pytest.mark.asyncio
    async def test_refresh_access_token_valid(self, auth_service):
        """Test access token refresh with valid refresh token."""
        data = {"sub": "userexample.com"}
        refresh_token = auth_service.create_refresh_token(data)
        
        new_access_token = auth_service.refresh_access_token(refresh_token)
        
        assert new_access_token is not None
        assert isinstance(new_access_token, str)
        
        # Verify the new token
        token_data = auth_service.verify_token(new_access_token)
        assert token_data is not None
        assert token_data.email == data["sub"]

    def test_refresh_access_token_invalid(self, auth_service):
        """Test access token refresh with invalid refresh token."""
        invalid_token = "invalid.refresh.token"
        
        new_access_token = auth_service.refresh_access_token(invalid_token)
        assert new_access_token is None

    def test_refresh_access_token_not_refresh_type(self, auth_service):
        """Test access token refresh with non-refresh token."""
        data = {"sub": "userexample.com"}
        access_token = auth_service.create_access_token(data)  # Not a refresh token
        
        new_access_token = auth_service.refresh_access_token(access_token)
        assert new_access_token is None

    @pytest.mark.asyncio
    async def test_get_current_user_valid_token(self, auth_service, mock_user):
        """Test getting current user with valid token."""
        data = {"sub": mock_user["email"]}
        token = auth_service.create_access_token(data)
        
        with patch.object(auth_service, 'get_user_by_email', new_callable=AsyncMock) as mock_get_user:
            mock_get_user.return_value = mock_user
            
            user = await auth_service.get_current_user(token)
            
            assert user is not None
            assert user["email"] == mock_user["email"]
            mock_get_user.assert_called_once_with(mock_user["email"])

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self, auth_service):
        """Test getting current user with invalid token."""
        invalid_token = "invalid.token.here"
        
        user = await auth_service.get_current_user(invalid_token)
        assert user is None

    @pytest.mark.asyncio
    async def test_get_current_user_nonexistent_user(self, auth_service):
        """Test getting current user when user no longer exists."""
        data = {"sub": "nonexistentexample.com"}
        token = auth_service.create_access_token(data)
        
        with patch.object(auth_service, 'get_user_by_email', new_callable=AsyncMock) as mock_get_user:
            mock_get_user.return_value = None
            
            user = await auth_service.get_current_user(token)
            assert user is None

    def test_validate_password_strength_valid(self, auth_service):
        """Test password strength validation with valid passwords."""
        valid_passwords = [
            "SecurePass123!",
            "MyStr0ngP@ssw0rd",
            "Complex1ty!",
            "T3st1ng@Pass"
        ]
        
        for password in valid_passwords:
            assert auth_service.validate_password_strength(password)

    def test_validate_password_strength_invalid(self, auth_service):
        """Test password strength validation with invalid passwords."""
        invalid_passwords = [
            "short",           # Too short
            "toolongpasswordwithoutanyuppercase",  # Too long, no uppercase
            "NoNumbers!",      # No numbers
            "nonumbers1",      # No special characters
            "NOCAPS123!",      # No lowercase
        ]
        
        for password in invalid_passwords:
            assert not auth_service.validate_password_strength(password)

    @pytest.mark.asyncio
    async def test_change_password_success(self, auth_service, mock_user):
        """Test successful password change."""
        old_password = "old_password_123"
        new_password = "new_password_456"
        
        # Set up mock user with hashed old password
        mock_user["hashed_password"] = auth_service.hash_password(old_password)
        
        with patch.object(auth_service, 'get_user_by_id', new_callable=AsyncMock) as mock_get_user, \
             patch.object(auth_service, 'update_user_password', new_callable=AsyncMock) as mock_update_password:
            
            mock_get_user.return_value = mock_user
            mock_update_password.return_value = True
            
            success = await auth_service.change_password(
                mock_user["id"], old_password, new_password
            )
            
            assert success
            mock_get_user.assert_called_once_with(mock_user["id"])
            mock_update_password.assert_called_once()

    @pytest.mark.asyncio
    async def test_change_password_wrong_old_password(self, auth_service, mock_user):
        """Test password change with wrong old password."""
        old_password = "wrong_old_password"
        new_password = "new_password_456"
        
        # Set up mock user with different hashed password
        mock_user["hashed_password"] = auth_service.hash_password("correct_old_password")
        
        with patch.object(auth_service, 'get_user_by_id', new_callable=AsyncMock) as mock_get_user:
            mock_get_user.return_value = mock_user
            
            success = await auth_service.change_password(
                mock_user["id"], old_password, new_password
            )
            
            assert not success

    def test_generate_password_reset_token(self, auth_service):
        """Test password reset token generation."""
        email = "user@company.local"
        token = auth_service.generate_password_reset_token(email)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Verify token can be decoded
        decoded = jwt.decode(token, auth_service.secret_key, algorithms=[auth_service.algorithm])
        assert decoded["sub"] == email
        assert decoded["type"] == "password_reset"

    def test_verify_password_reset_token_valid(self, auth_service):
        """Test password reset token verification with valid token."""
        email = "user@company.local"
        token = auth_service.generate_password_reset_token(email)
        
        verified_email = auth_service.verify_password_reset_token(token)
        assert verified_email == email

    def test_verify_password_reset_token_invalid(self, auth_service):
        """Test password reset token verification with invalid token."""
        invalid_token = "invalid.token.here"
        
        verified_email = auth_service.verify_password_reset_token(invalid_token)
        assert verified_email is None

    @pytest.mark.asyncio
    async def test_reset_password_success(self, auth_service, mock_user):
        """Test successful password reset."""
        email = mock_user["email"]
        new_password = "new_secure_password_123"
        reset_token = auth_service.generate_password_reset_token(email)
        
        with patch.object(auth_service, 'get_user_by_email', new_callable=AsyncMock) as mock_get_user, \
             patch.object(auth_service, 'update_user_password', new_callable=AsyncMock) as mock_update_password:
            
            mock_get_user.return_value = mock_user
            mock_update_password.return_value = True
            
            success = await auth_service.reset_password(reset_token, new_password)
            
            assert success
            mock_get_user.assert_called_once_with(email)
            mock_update_password.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_password_invalid_token(self, auth_service):
        """Test password reset with invalid token."""
        invalid_token = "invalid.token.here"
        new_password = "new_secure_password_123"
        
        success = await auth_service.reset_password(invalid_token, new_password)
        assert not success

    @pytest.mark.benchmark
    def test_password_hashing_performance(self, benchmark, auth_service):
        """Benchmark password hashing performance."""
        password = "test_password_123"
        
        result = benchmark(auth_service.hash_password, password)
        assert len(result) > 0

    @pytest.mark.benchmark
    def test_token_creation_performance(self, benchmark, auth_service):
        """Benchmark token creation performance."""
        data = {"sub": "userexample.com"}
        
        result = benchmark(auth_service.create_access_token, data)
        assert len(result) > 0