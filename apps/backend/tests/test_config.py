"""
Tests for core configuration.
"""

import pytest
from pydantic import ValidationError
from core.config import Settings, get_settings


class TestSettings:
    """Test configuration settings."""
    
    def test_settings_default_values(self):
        """Test that settings have reasonable defaults."""
        settings = Settings(
            secret_key="test-key",
            neo4j_password="test-password"
        )
        
        assert settings.app_name == "Strategic Planning Platform API"
        assert settings.version == "1.0.0"
        assert settings.environment == "development"
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert not settings.debug
        
    def test_settings_validation(self):
        """Test settings validation."""
        # Test valid environment
        settings = Settings(
            secret_key="test-key",
            neo4j_password="test-password",
            environment="production"
        )
        assert settings.environment == "production"
        
        # Test invalid environment
        with pytest.raises(ValidationError):
            Settings(
                secret_key="test-key", 
                neo4j_password="test-password",
                environment="invalid"
            )
    
    def test_is_production_property(self):
        """Test production environment check."""
        settings = Settings(
            secret_key="test-key",
            neo4j_password="test-password",
            environment="production"
        )
        assert settings.is_production is True
        
        settings.environment = "development"
        assert settings.is_production is False
    
    def test_is_development_property(self):
        """Test development environment check."""
        settings = Settings(
            secret_key="test-key",
            neo4j_password="test-password",
            environment="development"
        )
        assert settings.is_development is True
        
        settings.environment = "production"
        assert settings.is_development is False
    
    def test_database_url_property(self):
        """Test database URL construction."""
        settings = Settings(
            secret_key="test-key",
            neo4j_password="test-password",
            neo4j_uri="bolt://localhost:7687",
            neo4j_database="testdb"
        )
        assert settings.database_url == "bolt://localhost:7687/testdb"
    
    def test_get_api_key_for_model(self):
        """Test API key selection for different models."""
        settings = Settings(
            secret_key="test-key",
            neo4j_password="test-password",
            openai_api_key="openai-key",
            anthropic_api_key="anthropic-key",
            openrouter_api_key="openrouter-key"
        )
        
        assert settings.get_api_key_for_model("gpt-4") == "openai-key"
        assert settings.get_api_key_for_model("claude-3") == "anthropic-key"
        assert settings.get_api_key_for_model("anthropic/claude-3") == "openrouter-key"
        assert settings.get_api_key_for_model("unknown-model") is None
    
    def test_validate_api_keys(self):
        """Test API key validation."""
        # No API keys
        settings = Settings(
            secret_key="test-key",
            neo4j_password="test-password"
        )
        missing = settings.validate_api_keys()
        assert "At least one AI service API key is required" in missing
        
        # With API key
        settings = Settings(
            secret_key="test-key",
            neo4j_password="test-password",
            openai_api_key="test-key"
        )
        missing = settings.validate_api_keys()
        assert "At least one AI service API key is required" not in missing
    
    def test_get_settings_cached(self):
        """Test that get_settings returns cached instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2