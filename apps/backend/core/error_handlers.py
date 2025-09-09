"""
Comprehensive error handling for the Strategic Planning Platform API.

Provides structured error responses, exception handling, monitoring integration,
and recovery mechanisms for different error types.
"""

import traceback
from typing import Any, Dict, Optional, Union
from datetime import datetime, timezone

import structlog
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from redis.exceptions import ConnectionError as RedisConnectionError
from neo4j.exceptions import ServiceUnavailable, TransientError

from core.config import get_settings
from core.logging_config import log_security_event, log_performance_event

logger = structlog.get_logger(__name__)
settings = get_settings()


class PlatformError(Exception):
    """Base exception for Strategic Planning Platform errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.user_message = user_message or message
        super().__init__(self.message)


class DatabaseError(PlatformError):
    """Database-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            status_code=500,
            details=details,
            user_message="A database error occurred. Please try again later."
        )


class ValidationError(PlatformError):
    """Input validation errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            details=details,
            user_message=message
        )


class AuthenticationError(PlatformError):
    """Authentication-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=401,
            details=details,
            user_message="Authentication failed. Please check your credentials."
        )


class AuthorizationError(PlatformError):
    """Authorization-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403,
            details=details,
            user_message="You don't have permission to perform this action."
        )


class GraphRAGError(PlatformError):
    """GraphRAG validation errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="GRAPHRAG_ERROR",
            status_code=422,
            details=details,
            user_message="Content validation failed. Please review your input."
        )


class AgentOrchestrationError(PlatformError):
    """Agent orchestration errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AGENT_ERROR",
            status_code=500,
            details=details,
            user_message="An error occurred during AI processing. Please try again."
        )


class ExternalServiceError(PlatformError):
    """External service integration errors."""
    
    def __init__(self, service: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"{service}: {message}",
            error_code="EXTERNAL_SERVICE_ERROR",
            status_code=503,
            details={**details or {}, "service": service},
            user_message="An external service is temporarily unavailable. Please try again later."
        )


class RateLimitError(PlatformError):
    """Rate limiting errors."""
    
    def __init__(self, message: str, retry_after: int, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            status_code=429,
            details={**details or {}, "retry_after": retry_after},
            user_message="Too many requests. Please try again later."
        )


def create_error_response(
    error: Union[Exception, PlatformError],
    request: Request,
    include_details: bool = False
) -> JSONResponse:
    """Create standardized error response."""
    
    # Extract error information
    if isinstance(error, PlatformError):
        status_code = error.status_code
        error_code = error.error_code
        message = error.user_message
        details = error.details if include_details else {}
    elif isinstance(error, HTTPException):
        status_code = error.status_code
        error_code = f"HTTP_{status_code}"
        message = error.detail
        details = {}
    else:
        status_code = 500
        error_code = "INTERNAL_SERVER_ERROR"
        message = "An internal error occurred"
        details = {"error_type": type(error).__name__} if include_details else {}
    
    # Add common error fields
    error_response = {
        "error": {
            "code": error_code,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "path": request.url.path,
            "method": request.method
        }
    }
    
    # Add details if requested and available
    if include_details and details:
        error_response["error"]["details"] = details
    
    # Add correlation ID if available
    correlation_id = getattr(request.state, "correlation_id", None)
    if correlation_id:
        error_response["error"]["correlation_id"] = correlation_id
    
    # Add retry information for specific errors
    if isinstance(error, RateLimitError):
        headers = {"Retry-After": str(error.details.get("retry_after", 60))}
        return JSONResponse(
            status_code=status_code,
            content=error_response,
            headers=headers
        )
    
    return JSONResponse(status_code=status_code, content=error_response)


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    
    # Log the error
    logger.error(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method,
        correlation_id=getattr(request.state, "correlation_id", None)
    )
    
    # Log security events for authentication/authorization errors
    if exc.status_code in (401, 403):
        log_security_event(
            event_type="access_denied",
            details={
                "status_code": exc.status_code,
                "path": request.url.path,
                "method": request.method,
                "detail": exc.detail
            },
            request_ip=get_client_ip(request)
        )
    
    return create_error_response(
        error=exc,
        request=request,
        include_details=settings.environment == "development"
    )


async def validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Handle Pydantic validation errors."""
    
    # Extract validation details
    validation_details = []
    for error in exc.errors():
        validation_details.append({
            "field": ".".join(str(x) for x in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    # Log validation error
    logger.warning(
        "Validation error occurred",
        path=request.url.path,
        method=request.method,
        validation_errors=validation_details,
        correlation_id=getattr(request.state, "correlation_id", None)
    )
    
    # Create validation error response
    platform_error = ValidationError(
        message="Input validation failed",
        details={
            "validation_errors": validation_details,
            "error_count": len(validation_details)
        }
    )
    
    return create_error_response(
        error=platform_error,
        request=request,
        include_details=True
    )


async def database_exception_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    """Handle SQLAlchemy database errors."""
    
    # Classify database error
    if isinstance(exc, IntegrityError):
        error_type = "integrity_constraint_violation"
        user_message = "A data integrity constraint was violated"
        status_code = 409
    elif isinstance(exc, OperationalError):
        error_type = "database_operational_error"
        user_message = "Database is temporarily unavailable"
        status_code = 503
    else:
        error_type = "database_error"
        user_message = "A database error occurred"
        status_code = 500
    
    # Log database error
    logger.error(
        "Database error occurred",
        error_type=error_type,
        error_message=str(exc),
        path=request.url.path,
        method=request.method,
        correlation_id=getattr(request.state, "correlation_id", None),
        exc_info=True
    )
    
    # Create database error response
    platform_error = DatabaseError(
        message=str(exc),
        details={
            "error_type": error_type,
            "database_error": True
        }
    )
    platform_error.status_code = status_code
    platform_error.user_message = user_message
    
    return create_error_response(
        error=platform_error,
        request=request,
        include_details=settings.environment == "development"
    )


async def redis_exception_handler(request: Request, exc: RedisConnectionError) -> JSONResponse:
    """Handle Redis connection errors."""
    
    logger.error(
        "Redis connection error",
        error_message=str(exc),
        path=request.url.path,
        method=request.method,
        correlation_id=getattr(request.state, "correlation_id", None)
    )
    
    platform_error = ExternalServiceError(
        service="Redis Cache",
        message=str(exc),
        details={"cache_unavailable": True}
    )
    
    return create_error_response(
        error=platform_error,
        request=request,
        include_details=settings.environment == "development"
    )


async def neo4j_exception_handler(request: Request, exc: Union[ServiceUnavailable, TransientError]) -> JSONResponse:
    """Handle Neo4j database errors."""
    
    logger.error(
        "Neo4j database error",
        error_type=type(exc).__name__,
        error_message=str(exc),
        path=request.url.path,
        method=request.method,
        correlation_id=getattr(request.state, "correlation_id", None)
    )
    
    platform_error = ExternalServiceError(
        service="Neo4j Graph Database",
        message=str(exc),
        details={
            "graph_database_unavailable": True,
            "error_type": type(exc).__name__
        }
    )
    
    return create_error_response(
        error=platform_error,
        request=request,
        include_details=settings.environment == "development"
    )


async def platform_exception_handler(request: Request, exc: PlatformError) -> JSONResponse:
    """Handle custom platform errors."""
    
    # Log platform error
    logger.error(
        "Platform error occurred",
        error_code=exc.error_code,
        error_message=exc.message,
        status_code=exc.status_code,
        path=request.url.path,
        method=request.method,
        correlation_id=getattr(request.state, "correlation_id", None),
        **exc.details
    )
    
    return create_error_response(
        error=exc,
        request=request,
        include_details=settings.environment == "development"
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all unhandled exceptions."""
    
    # Generate error ID for tracking
    import uuid
    error_id = str(uuid.uuid4())
    
    # Log the full exception
    logger.critical(
        "Unhandled exception occurred",
        error_id=error_id,
        error_type=type(exc).__name__,
        error_message=str(exc),
        path=request.url.path,
        method=request.method,
        correlation_id=getattr(request.state, "correlation_id", None),
        traceback=traceback.format_exc(),
        exc_info=True
    )
    
    # Create generic error response
    platform_error = PlatformError(
        message=f"Internal server error (ID: {error_id})",
        error_code="INTERNAL_SERVER_ERROR",
        status_code=500,
        details={
            "error_id": error_id,
            "error_type": type(exc).__name__
        }
    )
    
    return create_error_response(
        error=platform_error,
        request=request,
        include_details=settings.environment == "development"
    )


def get_client_ip(request: Request) -> str:
    """Extract client IP address from request."""
    # Check X-Forwarded-For header first (for proxies)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    # Check X-Real-IP header (Nginx)
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to direct client
    if hasattr(request, "client") and request.client:
        return request.client.host
    
    return "unknown"


def setup_error_handlers(app) -> None:
    """Setup all error handlers for the FastAPI application."""
    
    # HTTP exceptions
    app.add_exception_handler(HTTPException, http_exception_handler)
    
    # Pydantic validation errors
    app.add_exception_handler(ValidationError, validation_exception_handler)
    
    # Database errors
    app.add_exception_handler(SQLAlchemyError, database_exception_handler)
    
    # Redis errors
    app.add_exception_handler(RedisConnectionError, redis_exception_handler)
    
    # Neo4j errors
    app.add_exception_handler(ServiceUnavailable, neo4j_exception_handler)
    app.add_exception_handler(TransientError, neo4j_exception_handler)
    
    # Platform-specific errors
    app.add_exception_handler(PlatformError, platform_exception_handler)
    app.add_exception_handler(DatabaseError, platform_exception_handler)
    app.add_exception_handler(AuthenticationError, platform_exception_handler)
    app.add_exception_handler(AuthorizationError, platform_exception_handler)
    app.add_exception_handler(GraphRAGError, platform_exception_handler)
    app.add_exception_handler(AgentOrchestrationError, platform_exception_handler)
    app.add_exception_handler(ExternalServiceError, platform_exception_handler)
    app.add_exception_handler(RateLimitError, platform_exception_handler)
    
    # Catch-all for unhandled exceptions
    app.add_exception_handler(Exception, general_exception_handler)
    
    logger.info("Error handlers configured successfully")


# Context managers for error handling in services
class ErrorContext:
    """Context manager for handling errors in service methods."""
    
    def __init__(
        self,
        operation: str,
        logger_name: str = __name__,
        reraise: bool = True,
        fallback_value: Any = None
    ):
        self.operation = operation
        self.logger = structlog.get_logger(logger_name)
        self.reraise = reraise
        self.fallback_value = fallback_value
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now(timezone.utc)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Operation succeeded
            duration_ms = (datetime.now(timezone.utc) - self.start_time).total_seconds() * 1000
            self.logger.info(
                f"{self.operation} completed successfully",
                operation=self.operation,
                duration_ms=round(duration_ms, 2)
            )
            return False
        
        # Operation failed
        duration_ms = (datetime.now(timezone.utc) - self.start_time).total_seconds() * 1000
        
        self.logger.error(
            f"{self.operation} failed",
            operation=self.operation,
            error_type=exc_type.__name__,
            error_message=str(exc_val),
            duration_ms=round(duration_ms, 2),
            exc_info=True
        )
        
        # Log performance event for failed operations
        log_performance_event(
            operation=self.operation,
            duration_ms=duration_ms,
            metadata={
                "success": False,
                "error_type": exc_type.__name__
            }
        )
        
        if not self.reraise:
            return True  # Suppress the exception
        
        return False  # Re-raise the exception


# Export commonly used items
__all__ = [
    "PlatformError",
    "DatabaseError", 
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "GraphRAGError",
    "AgentOrchestrationError",
    "ExternalServiceError",
    "RateLimitError",
    "setup_error_handlers",
    "ErrorContext"
]