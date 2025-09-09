"""
Comprehensive logging configuration for the Strategic Planning Platform.

Provides structured logging with OpenTelemetry tracing, performance metrics,
security audit logging, and error monitoring integration.
"""

import logging
import sys
import json
from typing import Any, Dict, Optional
from datetime import datetime, timezone
from pathlib import Path
import os

import structlog
from structlog import contextvars
from structlog.types import EventDict
from structlog.stdlib import LoggerFactory
from pythonjsonlogger import jsonlogger

from .config import get_settings

settings = get_settings()


class SecurityAuditProcessor:
    """Processor for security-related log events."""
    
    SECURITY_EVENTS = {
        "login_success", "login_failure", "token_refresh", "logout",
        "permission_denied", "account_locked", "password_change",
        "admin_action", "data_access", "configuration_change"
    }
    
    def __call__(self, logger, method_name, event_dict):
        """Process security audit events."""
        
        # Check if this is a security event
        event_type = event_dict.get("event_type")
        if event_type in self.SECURITY_EVENTS:
            event_dict["security_audit"] = True
            event_dict["audit_timestamp"] = datetime.now(timezone.utc).isoformat()
        
        return event_dict


class PerformanceProcessor:
    """Processor for performance metrics and monitoring."""
    
    def __call__(self, logger, method_name, event_dict):
        """Add performance metrics to log events."""
        
        # Add performance markers
        if "duration_ms" in event_dict:
            duration = event_dict["duration_ms"]
            if duration > 5000:  # > 5 seconds
                event_dict["performance_alert"] = "slow_operation"
            elif duration > 1000:  # > 1 second
                event_dict["performance_warning"] = "elevated_duration"
        
        return event_dict


class ErrorEnrichmentProcessor:
    """Processor to enrich error logs with context and metadata."""
    
    def __call__(self, logger, method_name, event_dict):
        """Enrich error events with additional context."""
        
        if method_name in ("error", "critical", "exception"):
            # Add error classification
            error = event_dict.get("error", "")
            if isinstance(error, str):
                if "database" in error.lower() or "connection" in error.lower():
                    event_dict["error_category"] = "database"
                elif "auth" in error.lower() or "token" in error.lower():
                    event_dict["error_category"] = "authentication"
                elif "validation" in error.lower():
                    event_dict["error_category"] = "validation"
                elif "permission" in error.lower():
                    event_dict["error_category"] = "authorization"
                else:
                    event_dict["error_category"] = "application"
            
            # Add severity level
            if method_name == "critical":
                event_dict["severity"] = "critical"
                event_dict["requires_immediate_attention"] = True
            elif method_name == "error":
                event_dict["severity"] = "high"
            
            # Add timestamp for error tracking
            event_dict["error_timestamp"] = datetime.now(timezone.utc).isoformat()
        
        return event_dict


def setup_stdlib_logging() -> None:
    """Configure standard library logging."""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO if settings.environment != "development" else logging.DEBUG)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler for development
    if settings.environment == "development":
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # JSON file handler for production
    if settings.environment in ("production", "staging"):
        file_handler = logging.FileHandler(log_dir / "application.json")
        json_formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s"
        )
        file_handler.setFormatter(json_formatter)
        root_logger.addHandler(file_handler)
    
    # Error file handler
    error_handler = logging.FileHandler(log_dir / "errors.json")
    error_handler.setLevel(logging.ERROR)
    error_formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    error_handler.setFormatter(error_formatter)
    root_logger.addHandler(error_handler)


def setup_logging() -> structlog.BoundLogger:
    """Initialize the complete logging system."""
    
    try:
        # Setup standard library logging
        setup_stdlib_logging()
        
        # Configure structlog processors
        processors = [
            # Add context variables
            contextvars.merge_contextvars,
            
            # Add timestamp and log level
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            
            # Add context
            add_app_context,
            
            # Custom processors
            SecurityAuditProcessor(),
            PerformanceProcessor(),
            ErrorEnrichmentProcessor(),
            
            # Stack info for exceptions
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
        ]
        
        # Add formatting based on environment
        if settings.environment == "production" or getattr(settings, 'log_format', 'json') == "json":
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
        
        # Configure structlog
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            logger_factory=LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Get logger instance
        logger = structlog.get_logger("startup")
        logger.info(
            "Comprehensive logging system initialized",
            environment=settings.environment,
            log_level=getattr(settings, 'log_level', 'INFO')
        )
        
        return logger
        
    except Exception as e:
        # Fallback to basic logging if setup fails
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logger = logging.getLogger("logging_setup")
        logger.error(f"Failed to setup advanced logging: {str(e)}")
        logger.info("Using fallback logging configuration")
        
        return structlog.get_logger("fallback")


def add_app_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add application context to all log entries."""
    event_dict.update({
        "app": getattr(settings, 'app_name', 'strategic-planning-api'),
        "version": getattr(settings, 'version', '1.0.0'),
        "environment": settings.environment,
    })
    
    return event_dict


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a configured logger instance."""
    return structlog.get_logger(name)


def log_security_event(
    event_type: str,
    user_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    request_ip: Optional[str] = None
) -> None:
    """Log a security audit event."""
    
    logger = get_logger("security")
    
    event_data = {
        "event_type": event_type,
        "user_id": user_id,
        "source_ip": request_ip,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **(details or {})
    }
    
    logger.info("Security audit event", **event_data)


def log_performance_event(
    operation: str,
    duration_ms: float,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Log a performance monitoring event."""
    
    logger = get_logger("performance")
    
    event_data = {
        "operation": operation,
        "duration_ms": duration_ms,
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **(metadata or {})
    }
    
    logger.info("Performance event", **event_data)


def log_business_event(
    event_type: str,
    user_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Log a business logic event."""
    
    logger = get_logger("business")
    
    event_data = {
        "event_type": event_type,
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **(details or {})
    }
    
    logger.info("Business event", **event_data)


class LoggingMiddleware:
    """FastAPI middleware for request/response logging."""
    
    def __init__(self, app):
        self.app = app
        self.logger = structlog.get_logger(__name__)
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Extract request information
        request_info = {
            "method": scope["method"],
            "path": scope["path"],
            "query_string": scope.get("query_string", b"").decode(),
            "client_host": scope.get("client", ["unknown", None])[0],
            "user_agent": self.get_header(scope, b"user-agent"),
        }
        
        # Start timer
        import time
        start_time = time.time()
        
        # Process request
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Log response
                duration_ms = (time.time() - start_time) * 1000
                
                self.logger.info(
                    "HTTP request completed",
                    **request_info,
                    status_code=message["status"],
                    duration_ms=round(duration_ms, 2),
                    response_size=message.get("body", b"").__len__() if message.get("body") else 0
                )
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)
    
    def get_header(self, scope: Dict[str, Any], name: bytes) -> Optional[str]:
        """Extract header value from ASGI scope."""
        for header_name, header_value in scope.get("headers", []):
            if header_name == name:
                return header_value.decode()
        return None


def get_correlation_id() -> str:
    """Generate correlation ID for request tracing."""
    import uuid
    return str(uuid.uuid4())[:8]


class CorrelationMiddleware:
    """Middleware to add correlation ID to requests."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Add correlation ID to scope
        correlation_id = get_correlation_id()
        scope["state"] = getattr(scope, "state", {})
        scope["state"]["correlation_id"] = correlation_id
        
        # Add correlation ID to response headers
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-correlation-id", correlation_id.encode()))
                message["headers"] = headers
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)