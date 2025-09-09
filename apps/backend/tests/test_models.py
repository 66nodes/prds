"""
Tests for Pydantic models.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError
from models.user import User, UserCreate, UserLogin, TokenResponse
from models.prd import PRDGenerationRequest, PRDGenerationResponse
from models.validation import ValidationRequest, ValidationResult


class TestUserModels:
    """Test user-related models."""
    
    def test_user_create_valid(self):
        """Test valid user creation."""
        user_data = UserCreate(
            email="user@company.local",
            password="password123",
            full_name="Test User",
            company="Test Corp"
        )
        
        assert user_data.email == "user@company.local"
        assert user_data.full_name == "Test User"
        assert user_data.company == "Test Corp"
    
    def test_user_create_validation(self):
        """Test user creation validation."""
        # Invalid email
        with pytest.raises(ValidationError):
            UserCreate(
                email="invalid-email",
                password="password123",
                full_name="Test User"
            )
        
        # Short password
        with pytest.raises(ValidationError):
            UserCreate(
                email="user@company.local",
                password="short",
                full_name="Test User"
            )
        
        # Short name
        with pytest.raises(ValidationError):
            UserCreate(
                email="user@company.local",
                password="password123",
                full_name="T"
            )
    
    def test_user_login_model(self):
        """Test user login model."""
        login_data = UserLogin(
            email="user@company.local",
            password="password123"
        )
        
        assert login_data.email == "user@company.local"
        assert login_data.password == "password123"
    
    def test_user_model(self):
        """Test user model."""
        user = User(
            id="user-123",
            email="user@company.local",
            full_name="Test User",
            role="user",
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        assert user.id == "user-123"
        assert user.email == "user@company.local"
        assert user.role == "user"
        assert user.is_active is True
    
    def test_token_response_model(self):
        """Test token response model."""
        user = User(
            id="user-123",
            email="user@company.local",
            full_name="Test User",
            role="user",
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        token_response = TokenResponse(
            access_token="access-token",
            refresh_token="refresh-token",
            expires_in=1800,
            user=user
        )
        
        assert token_response.token_type == "bearer"  # default value
        assert token_response.expires_in == 1800
        assert token_response.user.id == "user-123"


class TestPRDModels:
    """Test PRD-related models."""
    
    def test_prd_generation_request_valid(self):
        """Test valid PRD generation request."""
        request = PRDGenerationRequest(
            title="Test Feature PRD",
            description="A comprehensive description of the test feature that is long enough to meet validation requirements.",
            user_id="user-123",
            priority="high"
        )
        
        assert request.title == "Test Feature PRD"
        assert request.priority == "high"
        assert len(request.description) >= 100
    
    def test_prd_generation_request_validation(self):
        """Test PRD generation request validation."""
        # Title too short
        with pytest.raises(ValidationError):
            PRDGenerationRequest(
                title="Short",
                description="A comprehensive description of the test feature that is long enough to meet validation requirements.",
                user_id="user-123"
            )
        
        # Description too short  
        with pytest.raises(ValidationError):
            PRDGenerationRequest(
                title="Valid Title Here",
                description="Short description",
                user_id="user-123"
            )
    
    def test_prd_generation_response(self):
        """Test PRD generation response model."""
        response = PRDGenerationResponse(
            prd_id="prd-123",
            title="Test PRD",
            status="completed",
            quality_score=8.5,
            sections_count=5,
            validation_summary={"overall": "passed"}
        )
        
        assert response.prd_id == "prd-123"
        assert response.quality_score == 8.5
        assert 0 <= response.quality_score <= 10


class TestValidationModels:
    """Test validation-related models."""
    
    def test_validation_request_valid(self):
        """Test valid validation request."""
        request = ValidationRequest(
            content="This is test content to validate",
            section_type="overview",
            project_id="project-123"
        )
        
        assert request.content == "This is test content to validate"
        assert request.section_type == "overview"
    
    def test_validation_request_validation(self):
        """Test validation request validation."""
        # Content too short (less than 10 chars)
        with pytest.raises(ValidationError):
            ValidationRequest(content="Short")
    
    def test_validation_result(self):
        """Test validation result model."""
        result = ValidationResult(
            validation_id="val-123",
            confidence=0.95,
            passes_threshold=True,
            entity_validation={"score": 0.9},
            community_validation={"score": 0.92},
            global_validation={"score": 0.88},
            requires_human_review=False,
            processing_time_ms=150,
            timestamp=datetime.utcnow(),
            validation_level="standard"
        )
        
        assert result.validation_id == "val-123"
        assert result.confidence == 0.95
        assert 0 <= result.confidence <= 1
        assert result.passes_threshold is True