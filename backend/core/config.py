"""
Core configuration management using Pydantic Settings
"""

import os
from datetime import datetime
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application
    app_name: str = Field(default="Strategic Planning Platform API", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(default="development", description="Environment: development, staging, production")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    
    # Security
    secret_key: str = Field(..., description="Secret key for JWT token generation")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiration in minutes")
    refresh_token_expire_days: int = Field(default=7, description="Refresh token expiration in days")
    allowed_hosts: List[str] = Field(default=["*"], description="Allowed hosts for CORS")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "https://strategic-planning.ai"],
        description="Allowed CORS origins"
    )
    
    # Database Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j database URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(..., description="Neo4j password")
    neo4j_database: str = Field(default="neo4j", description="Neo4j database name")
    
    # Milvus Configuration
    milvus_host: str = Field(default="localhost", description="Milvus server host")
    milvus_port: int = Field(default=19530, description="Milvus server port")
    milvus_user: Optional[str] = Field(default=None, description="Milvus username")
    milvus_password: Optional[str] = Field(default=None, description="Milvus password")
    milvus_db_name: str = Field(default="default", description="Milvus database name")
    
    # PostgreSQL Configuration (for structured data)
    postgres_url: str = Field(default="postgresql://postgres:development@localhost:5432/aiplatform", description="PostgreSQL connection URL")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_db: int = Field(default=0, description="Redis database number")
    cache_ttl: int = Field(default=3600, description="Default cache TTL in seconds")
    
    # AI Service Configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    openrouter_api_key: Optional[str] = Field(default=None, description="OpenRouter API key")
    default_model: str = Field(default="gpt-4o", description="Default LLM model")
    fallback_model: str = Field(default="gpt-4o-mini", description="Fallback LLM model")
    max_tokens: int = Field(default=4000, description="Maximum tokens per request")
    temperature: float = Field(default=0.1, description="LLM temperature setting")
    
    # GraphRAG Configuration
    graphrag_enabled: bool = Field(default=True, description="Enable GraphRAG validation")
    entity_validation_weight: float = Field(default=0.5, description="Entity validation weight")
    community_validation_weight: float = Field(default=0.3, description="Community validation weight")
    global_validation_weight: float = Field(default=0.2, description="Global validation weight")
    validation_threshold: float = Field(default=0.8, description="Minimum validation score threshold")
    
    # GitHub Integration
    github_token: Optional[str] = Field(default=None, description="GitHub personal access token")
    github_org: Optional[str] = Field(default=None, description="Default GitHub organization")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, description="Requests per minute per IP")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format: json or text")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    # Monitoring and Tracing
    enable_tracing: bool = Field(default=False, description="Enable OpenTelemetry tracing")
    jaeger_host: str = Field(default="localhost", description="Jaeger collector host")
    jaeger_port: int = Field(default=14268, description="Jaeger collector port")
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    
    # File Storage
    upload_dir: str = Field(default="uploads", description="File upload directory")
    max_file_size: int = Field(default=10485760, description="Maximum file size in bytes (10MB)")
    allowed_file_types: List[str] = Field(
        default=[".pdf", ".docx", ".txt", ".md"],
        description="Allowed file extensions"
    )
    
    @validator("environment")
    def validate_environment(cls, v):
        if v not in ["development", "staging", "production"]:
            raise ValueError("Environment must be development, staging, or production")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @validator("validation_threshold")
    def validate_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Validation threshold must be between 0.0 and 1.0")
        return v
    
    @validator("temperature")
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    @property
    def database_url(self) -> str:
        """Construct Neo4j database URL."""
        return f"{self.neo4j_uri}/{self.neo4j_database}"
    
    def current_timestamp(self) -> str:
        """Get current ISO timestamp."""
        return datetime.utcnow().isoformat()
    
    def get_api_key_for_model(self, model: str) -> Optional[str]:
        """Get appropriate API key for a given model."""
        if model.startswith("gpt") or model.startswith("o1"):
            return self.openai_api_key
        elif model.startswith("claude"):
            return self.anthropic_api_key
        elif "/" in model:  # OpenRouter models
            return self.openrouter_api_key
        return None
    
    def validate_api_keys(self) -> List[str]:
        """Validate that at least one AI service API key is configured."""
        missing_keys = []
        
        if not any([self.openai_api_key, self.anthropic_api_key, self.openrouter_api_key]):
            missing_keys.append("At least one AI service API key is required")
        
        if not self.secret_key:
            missing_keys.append("SECRET_KEY is required")
        
        if not self.neo4j_password:
            missing_keys.append("NEO4J_PASSWORD is required")
        
        return missing_keys


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()