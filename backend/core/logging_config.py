"""
Structured logging configuration with JSON output
"""

import logging
import sys
from typing import Any, Dict, Optional

import structlog
from structlog.types import EventDict

from .config import get_settings

settings = get_settings()


def setup_logging() -> structlog.BoundLogger:
    """Configure structured logging with JSON output."""
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level),
    )
    
    # Configure structlog processors
    processors = [
        # Add log level and timestamp
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="ISO"),
        
        # Add context
        add_app_context,
        
        # Stack info for errors
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    # Add JSON formatting for production
    if settings.environment == "production" or settings.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Development-friendly console output
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True),
        ])
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )
    
    # Get logger instance
    logger = structlog.get_logger(__name__)
    
    logger.info(
        "Logging configured",
        level=settings.log_level,
        format=settings.log_format,
        environment=settings.environment
    )
    
    return logger


def add_app_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add application context to all log entries."""
    event_dict.update({
        "app": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
    })
    
    return event_dict


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