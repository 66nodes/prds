"""
Custom middleware for the Strategic Planning Platform API
"""

import asyncio
import time
from typing import Any, Dict, Optional
from collections import defaultdict

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using in-memory storage."""
    
    def __init__(self, app):
        super().__init__(app)
        self.requests = defaultdict(list)
        self.cleanup_interval = 300  # Clean up old entries every 5 minutes
        self.last_cleanup = time.time()
    
    async def dispatch(self, request: Request, call_next):
        # Get client identifier
        client_ip = self.get_client_ip(request)
        
        # Check rate limit
        if await self.is_rate_limited(client_ip):
            logger.warning(
                "Rate limit exceeded",
                client_ip=client_ip,
                path=request.url.path,
                method=request.method
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {settings.rate_limit_requests} requests per {settings.rate_limit_window} seconds",
                    "retry_after": settings.rate_limit_window
                },
                headers={"Retry-After": str(settings.rate_limit_window)}
            )
        
        # Record request
        await self.record_request(client_ip)
        
        # Periodic cleanup
        if time.time() - self.last_cleanup > self.cleanup_interval:
            await self.cleanup_old_requests()
            self.last_cleanup = time.time()
        
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = await self.get_remaining_requests(client_ip)
        response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Window"] = str(settings.rate_limit_window)
        
        return response
    
    def get_client_ip(self, request: Request) -> str:
        """Get client IP address, considering proxies."""
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
    
    async def is_rate_limited(self, client_ip: str) -> bool:
        """Check if client has exceeded rate limit."""
        now = time.time()
        window_start = now - settings.rate_limit_window
        
        # Count recent requests
        recent_requests = [
            req_time for req_time in self.requests[client_ip] 
            if req_time > window_start
        ]
        
        return len(recent_requests) >= settings.rate_limit_requests
    
    async def record_request(self, client_ip: str) -> None:
        """Record a request timestamp for rate limiting."""
        self.requests[client_ip].append(time.time())
    
    async def get_remaining_requests(self, client_ip: str) -> int:
        """Get remaining requests for client."""
        now = time.time()
        window_start = now - settings.rate_limit_window
        
        recent_requests = [
            req_time for req_time in self.requests[client_ip]
            if req_time > window_start
        ]
        
        return max(0, settings.rate_limit_requests - len(recent_requests))
    
    async def cleanup_old_requests(self) -> None:
        """Clean up old request records to prevent memory leak."""
        now = time.time()
        cutoff = now - (settings.rate_limit_window * 2)  # Keep extra buffer
        
        for client_ip in list(self.requests.keys()):
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if req_time > cutoff
            ]
            
            # Remove empty entries
            if not self.requests[client_ip]:
                del self.requests[client_ip]


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Content Security Policy
        if settings.environment == "production":
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self'; "
                "connect-src 'self' wss: https:; "
                "frame-ancestors 'none'"
            )
            response.headers["Content-Security-Policy"] = csp
        
        # HSTS for HTTPS
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Add request context and correlation ID."""
    
    async def dispatch(self, request: Request, call_next):
        # Generate correlation ID
        import uuid
        correlation_id = str(uuid.uuid4())
        
        # Add to request state
        request.state.correlation_id = correlation_id
        request.state.start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Add correlation ID to response
        response.headers["X-Correlation-ID"] = correlation_id
        
        # Add response time
        duration = time.time() - request.state.start_time
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Enhanced request/response logging."""
    
    async def dispatch(self, request: Request, call_next):
        # Skip logging for health checks
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)
        
        # Extract request info
        start_time = time.time()
        
        # Log request start
        logger.info(
            "Request started",
            method=request.method,
            path=request.url.path,
            query_params=str(request.query_params) if request.query_params else None,
            client_ip=self.get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            correlation_id=getattr(request.state, "correlation_id", None)
        )
        
        # Process request
        try:
            response = await call_next(request)
            duration_ms = (time.time() - start_time) * 1000
            
            # Log successful response
            logger.info(
                "Request completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
                correlation_id=getattr(request.state, "correlation_id", None)
            )
            
            return response
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            # Log error
            logger.error(
                "Request failed",
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration_ms, 2),
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=getattr(request.state, "correlation_id", None),
                exc_info=True
            )
            
            raise
    
    def get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        # Same logic as rate limiting middleware
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"


class ValidationMiddleware(BaseHTTPMiddleware):
    """Request validation middleware."""
    
    async def dispatch(self, request: Request, call_next):
        # Validate content type for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            
            if not content_type:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Content-Type header is required"}
                )
            
            # Check for supported content types
            supported_types = [
                "application/json",
                "multipart/form-data",
                "application/x-www-form-urlencoded"
            ]
            
            if not any(supported_type in content_type for supported_type in supported_types):
                return JSONResponse(
                    status_code=415,
                    content={
                        "error": "Unsupported Media Type",
                        "supported_types": supported_types
                    }
                )
        
        return await call_next(request)


# Combine all custom middleware
LoggingMiddleware = RequestLoggingMiddleware