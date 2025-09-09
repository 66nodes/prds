"""
Audit Middleware for Automatic Event Logging

Comprehensive middleware that automatically captures and logs audit events
for HTTP requests, authentication, authorization, data access, and system events.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable, List
from urllib.parse import urlparse

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from core.config import get_settings
from services.comprehensive_audit_service import (
    get_comprehensive_audit_service,
    ComprehensiveAuditService,
    AuditEventType,
    AuditSeverity
)

logger = structlog.get_logger(__name__)
settings = get_settings()


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive audit logging of HTTP requests and responses.
    
    Automatically captures:
    - All HTTP requests and responses
    - Authentication events
    - Authorization decisions
    - Data access patterns
    - Error conditions
    - Performance metrics
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.audit_service: Optional[ComprehensiveAuditService] = None
        
        # Configuration
        self.log_request_body = settings.environment != "production"
        self.log_response_body = False  # Generally not recommended for audit
        self.max_body_size = 1024 * 10  # 10KB limit for request body logging
        
        # Sensitive endpoints that require enhanced logging
        self.sensitive_endpoints = {
            "/api/v1/auth/login",
            "/api/v1/auth/logout", 
            "/api/v1/auth/refresh",
            "/api/v1/users",
            "/api/v1/admin",
            "/api/v1/prd",
            "/api/v1/validation",
            "/api/v1/enterprise"
        }
        
        # Endpoints to exclude from audit logging
        self.excluded_endpoints = {
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process HTTP request and response with comprehensive audit logging."""
        
        # Skip excluded endpoints
        if any(request.url.path.startswith(excluded) for excluded in self.excluded_endpoints):
            return await call_next(request)
        
        # Initialize audit service if needed
        if not self.audit_service:
            try:
                self.audit_service = await get_comprehensive_audit_service()
            except Exception as e:
                logger.warning(f"Failed to initialize audit service: {str(e)}")
        
        # Start timing
        start_time = time.time()
        
        # Extract request information
        request_info = await self._extract_request_info(request)
        
        # Determine if this is a sensitive operation
        is_sensitive = any(request.url.path.startswith(endpoint) for endpoint in self.sensitive_endpoints)
        
        # Log request initiation for sensitive endpoints
        if is_sensitive and self.audit_service:
            await self._log_request_initiation(request_info)
        
        # Process request
        response = None
        error_occurred = False
        error_message = None
        
        try:
            response = await call_next(request)
        except Exception as e:
            error_occurred = True
            error_message = str(e)
            # Re-raise to maintain normal error handling
            raise
        finally:
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Extract response information
            response_info = self._extract_response_info(response) if response else {}
            
            # Log comprehensive audit event
            if self.audit_service:
                await self._log_request_completion(
                    request_info,
                    response_info,
                    duration_ms,
                    error_occurred,
                    error_message,
                    is_sensitive
                )
        
        return response
    
    async def _extract_request_info(self, request: Request) -> Dict[str, Any]:
        """Extract comprehensive request information for audit logging."""
        
        # Basic request information
        request_info = {
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client_host": getattr(request.client, "host", None) if request.client else None,
            "client_port": getattr(request.client, "port", None) if request.client else None,
            "scheme": request.url.scheme,
            "server_host": request.get("server", ("unknown", None))[0],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # Extract user information if available
        if hasattr(request.state, "user"):
            user = request.state.user
            request_info.update({
                "user_id": getattr(user, "id", None),
                "user_email": getattr(user, "email", None),
                "user_role": getattr(user, "role", None)
            })
        
        # Extract session information
        if hasattr(request.state, "session_id"):
            request_info["session_id"] = request.state.session_id
        
        # Extract correlation ID
        if hasattr(request.state, "correlation_id"):
            request_info["correlation_id"] = request.state.correlation_id
        
        # Extract request body for sensitive operations (if enabled)
        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if len(body) <= self.max_body_size:
                    # Only log non-sensitive body content
                    if not self._contains_sensitive_data(body):
                        request_info["body_size"] = len(body)
                        if body:
                            try:
                                # Try to parse as JSON
                                request_info["body_preview"] = json.loads(body.decode())
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                request_info["body_preview"] = f"<binary-data-{len(body)}-bytes>"
                else:
                    request_info["body_size"] = len(body)
                    request_info["body_truncated"] = True
            except Exception:
                request_info["body_error"] = "failed_to_read"
        
        # Clean sensitive headers
        request_info["headers"] = self._sanitize_headers(request_info["headers"])
        
        return request_info
    
    def _extract_response_info(self, response: Response) -> Dict[str, Any]:
        """Extract response information for audit logging."""
        
        response_info = {
            "status_code": response.status_code,
            "headers": dict(response.headers) if hasattr(response, "headers") else {},
            "content_type": response.headers.get("content-type") if hasattr(response, "headers") else None,
        }
        
        # Estimate response size
        if hasattr(response, "body") and response.body:
            response_info["body_size"] = len(response.body)
        
        return response_info
    
    async def _log_request_initiation(self, request_info: Dict[str, Any]) -> None:
        """Log the initiation of a sensitive request."""
        
        try:
            # Determine event type based on endpoint
            event_type = self._determine_event_type(request_info["path"], request_info["method"])
            
            await self.audit_service.log_audit_event(
                event_type=event_type,
                user_id=request_info.get("user_id"),
                user_email=request_info.get("user_email"),
                ip_address=request_info.get("client_host"),
                user_agent=request_info["headers"].get("user-agent"),
                session_id=request_info.get("session_id"),
                correlation_id=request_info.get("correlation_id"),
                resource_type="http_endpoint",
                resource_id=request_info["path"],
                action=f"{request_info['method']} {request_info['path']}",
                api_endpoint=request_info["path"],
                http_method=request_info["method"],
                severity=AuditSeverity.MEDIUM,
                metadata={
                    "request_initiated": True,
                    "query_params": request_info.get("query_params", {}),
                    "body_size": request_info.get("body_size", 0)
                }
            )
        except Exception as e:
            logger.error(f"Failed to log request initiation: {str(e)}")
    
    async def _log_request_completion(
        self,
        request_info: Dict[str, Any],
        response_info: Dict[str, Any],
        duration_ms: float,
        error_occurred: bool,
        error_message: Optional[str],
        is_sensitive: bool
    ) -> None:
        """Log the completion of an HTTP request."""
        
        try:
            # Determine event type and outcome
            if error_occurred:
                event_type = AuditEventType.SYSTEM_ERROR
                outcome = "failure"
                severity = AuditSeverity.HIGH
            else:
                event_type = self._determine_event_type(request_info["path"], request_info["method"])
                outcome = self._determine_outcome(response_info.get("status_code", 500))
                severity = self._determine_severity(response_info.get("status_code", 500), is_sensitive)
            
            # Determine if this was a data access operation
            data_operation = self._is_data_operation(request_info["path"], request_info["method"])
            
            await self.audit_service.log_audit_event(
                event_type=event_type,
                user_id=request_info.get("user_id"),
                user_email=request_info.get("user_email"),
                user_role=request_info.get("user_role"),
                ip_address=request_info.get("client_host"),
                user_agent=request_info["headers"].get("user-agent"),
                session_id=request_info.get("session_id"),
                correlation_id=request_info.get("correlation_id"),
                resource_type="http_endpoint",
                resource_id=request_info["path"],
                action=f"{request_info['method']} {request_info['path']}",
                outcome=outcome,
                error_message=error_message,
                duration_ms=duration_ms,
                api_endpoint=request_info["path"],
                http_method=request_info["method"],
                response_status=response_info.get("status_code"),
                data_volume_bytes=response_info.get("body_size", 0),
                severity=severity,
                sensitive_data=is_sensitive or data_operation,
                compliance_tags=self._get_compliance_tags(request_info["path"], is_sensitive, data_operation),
                metadata={
                    "request_url": request_info["url"],
                    "query_params": request_info.get("query_params", {}),
                    "request_body_size": request_info.get("body_size", 0),
                    "response_content_type": response_info.get("content_type"),
                    "server_host": request_info.get("server_host"),
                    "client_port": request_info.get("client_port"),
                    "scheme": request_info.get("scheme")
                }
            )
            
            # Log additional events for specific operations
            await self._log_specialized_events(request_info, response_info, outcome)
            
        except Exception as e:
            logger.error(f"Failed to log request completion: {str(e)}")
    
    def _determine_event_type(self, path: str, method: str) -> AuditEventType:
        """Determine audit event type based on endpoint and method."""
        
        # Authentication endpoints
        if "/auth/login" in path:
            return AuditEventType.USER_LOGIN
        elif "/auth/logout" in path:
            return AuditEventType.USER_LOGOUT
        elif "/auth/refresh" in path:
            return AuditEventType.TOKEN_REFRESH
        
        # Admin operations
        elif "/admin" in path:
            return AuditEventType.ADMIN_ACTION
        
        # Data operations
        elif method == "POST" and any(resource in path for resource in ["/prd", "/users", "/documents"]):
            return AuditEventType.DATA_CREATED
        elif method in ["PUT", "PATCH"] and any(resource in path for resource in ["/prd", "/users", "/documents"]):
            return AuditEventType.DATA_UPDATED
        elif method == "DELETE" and any(resource in path for resource in ["/prd", "/users", "/documents"]):
            return AuditEventType.DATA_DELETED
        elif method == "GET" and any(resource in path for resource in ["/prd", "/users", "/documents"]):
            return AuditEventType.DATA_READ
        
        # AI operations
        elif "/ai" in path or "/llm" in path or "/agents" in path:
            return AuditEventType.AI_AGENT_INVOKED
        elif "/validation" in path:
            return AuditEventType.AI_VALIDATION_PERFORMED
        
        # Export operations
        elif "/export" in path or "download" in path:
            return AuditEventType.DATA_EXPORTED
        
        # Default to system event
        else:
            return AuditEventType.DATA_READ if method == "GET" else AuditEventType.DATA_UPDATED
    
    def _determine_outcome(self, status_code: int) -> str:
        """Determine operation outcome based on HTTP status code."""
        
        if 200 <= status_code < 300:
            return "success"
        elif 400 <= status_code < 500:
            return "failure"
        elif status_code >= 500:
            return "error"
        else:
            return "unknown"
    
    def _determine_severity(self, status_code: int, is_sensitive: bool) -> AuditSeverity:
        """Determine audit event severity."""
        
        if status_code >= 500:
            return AuditSeverity.CRITICAL
        elif status_code == 401 or status_code == 403:
            return AuditSeverity.HIGH
        elif status_code >= 400:
            return AuditSeverity.MEDIUM
        elif is_sensitive:
            return AuditSeverity.MEDIUM
        else:
            return AuditSeverity.LOW
    
    def _is_data_operation(self, path: str, method: str) -> bool:
        """Determine if this is a data access/modification operation."""
        
        data_endpoints = ["/prd", "/users", "/documents", "/enterprise", "/validation"]
        return any(endpoint in path for endpoint in data_endpoints) and method in ["GET", "POST", "PUT", "PATCH", "DELETE"]
    
    def _get_compliance_tags(self, path: str, is_sensitive: bool, is_data_operation: bool) -> List[str]:
        """Get compliance tags based on the operation."""
        
        tags = []
        
        # SOC 2 relevant operations
        if any(endpoint in path for endpoint in ["/auth", "/admin", "/users"]):
            tags.append("soc2")
        
        # GDPR relevant operations
        if is_data_operation:
            tags.append("gdpr")
            tags.append("data_protection")
        
        # Security relevant operations
        if "/auth" in path or "/admin" in path or is_sensitive:
            tags.append("security")
        
        # AI governance
        if any(endpoint in path for endpoint in ["/ai", "/llm", "/agents", "/validation"]):
            tags.append("ai_governance")
        
        return tags
    
    async def _log_specialized_events(
        self,
        request_info: Dict[str, Any],
        response_info: Dict[str, Any],
        outcome: str
    ) -> None:
        """Log additional specialized events for specific operations."""
        
        path = request_info["path"]
        method = request_info["method"]
        status_code = response_info.get("status_code", 500)
        
        try:
            # Failed authentication attempts
            if "/auth/login" in path and status_code == 401:
                await self.audit_service.log_authentication_event(
                    event_type=AuditEventType.USER_LOGIN_FAILED,
                    user_email=request_info.get("user_email"),
                    ip_address=request_info.get("client_host"),
                    user_agent=request_info["headers"].get("user-agent"),
                    outcome="failure",
                    error_message="Invalid credentials"
                )
            
            # Permission denied events
            elif status_code == 403:
                await self.audit_service.log_audit_event(
                    event_type=AuditEventType.PERMISSION_DENIED,
                    user_id=request_info.get("user_id"),
                    user_email=request_info.get("user_email"),
                    ip_address=request_info.get("client_host"),
                    resource_type="http_endpoint",
                    resource_id=path,
                    action=f"{method} {path}",
                    outcome="denied",
                    severity=AuditSeverity.HIGH,
                    compliance_tags=["security", "soc2"]
                )
            
            # Rate limiting events
            elif status_code == 429:
                await self.audit_service.log_audit_event(
                    event_type=AuditEventType.RATE_LIMIT_EXCEEDED,
                    ip_address=request_info.get("client_host"),
                    resource_type="rate_limit",
                    action="rate_limit_exceeded",
                    severity=AuditSeverity.MEDIUM,
                    compliance_tags=["security"]
                )
            
        except Exception as e:
            logger.error(f"Failed to log specialized event: {str(e)}")
    
    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Remove or mask sensitive headers."""
        
        sensitive_headers = {
            "authorization", "cookie", "x-api-key", "x-auth-token",
            "proxy-authorization", "www-authenticate"
        }
        
        sanitized = {}
        for key, value in headers.items():
            if key.lower() in sensitive_headers:
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _contains_sensitive_data(self, body: bytes) -> bool:
        """Check if request body contains sensitive data patterns."""
        
        try:
            body_str = body.decode().lower()
            
            # Common sensitive field names
            sensitive_patterns = [
                "password", "passwd", "pwd", "secret", "token", "key",
                "ssn", "social", "credit", "card", "cvv", "pin",
                "private", "confidential", "sensitive"
            ]
            
            return any(pattern in body_str for pattern in sensitive_patterns)
        
        except UnicodeDecodeError:
            # If we can't decode, assume it might be sensitive
            return True


class AuthenticationAuditMiddleware:
    """
    Specialized middleware for authentication and authorization audit events.
    
    Captures detailed authentication flows, token operations, and permission checks.
    """
    
    def __init__(self, app: ASGIApp):
        self.app = app
        self.audit_service: Optional[ComprehensiveAuditService] = None
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Process ASGI requests with authentication audit logging."""
        
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Initialize audit service if needed
        if not self.audit_service:
            try:
                self.audit_service = await get_comprehensive_audit_service()
            except Exception as e:
                logger.warning(f"Failed to initialize audit service: {str(e)}")
        
        # Track authentication state changes
        original_user = getattr(scope.get("state", {}), "user", None)
        
        async def send_wrapper(message):
            # Check if authentication state changed
            if message["type"] == "http.response.start":
                current_user = getattr(scope.get("state", {}), "user", None)
                
                # Log authentication state changes
                if self.audit_service and original_user != current_user:
                    if current_user and not original_user:
                        # User authenticated
                        await self._log_authentication_success(scope, current_user)
                    elif not current_user and original_user:
                        # User logged out
                        await self._log_authentication_logout(scope, original_user)
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)
    
    async def _log_authentication_success(self, scope: Scope, user: Any) -> None:
        """Log successful authentication event."""
        
        try:
            request_info = self._extract_request_info_from_scope(scope)
            
            await self.audit_service.log_authentication_event(
                event_type=AuditEventType.USER_LOGIN,
                user_id=getattr(user, "id", None),
                user_email=getattr(user, "email", None),
                ip_address=request_info.get("client_host"),
                user_agent=request_info.get("user_agent"),
                outcome="success"
            )
        except Exception as e:
            logger.error(f"Failed to log authentication success: {str(e)}")
    
    async def _log_authentication_logout(self, scope: Scope, user: Any) -> None:
        """Log user logout event."""
        
        try:
            request_info = self._extract_request_info_from_scope(scope)
            
            await self.audit_service.log_authentication_event(
                event_type=AuditEventType.USER_LOGOUT,
                user_id=getattr(user, "id", None),
                user_email=getattr(user, "email", None),
                ip_address=request_info.get("client_host"),
                user_agent=request_info.get("user_agent"),
                outcome="success"
            )
        except Exception as e:
            logger.error(f"Failed to log authentication logout: {str(e)}")
    
    def _extract_request_info_from_scope(self, scope: Scope) -> Dict[str, Any]:
        """Extract request information from ASGI scope."""
        
        headers = dict(scope.get("headers", []))
        client = scope.get("client", ("unknown", None))
        
        return {
            "client_host": client[0] if client else "unknown",
            "user_agent": headers.get(b"user-agent", b"").decode()
        }


class SecurityEventMiddleware:
    """
    Middleware for detecting and logging security-related events.
    
    Monitors for suspicious patterns, potential attacks, and security violations.
    """
    
    def __init__(self, app: ASGIApp):
        self.app = app
        self.audit_service: Optional[ComprehensiveAuditService] = None
        
        # Security monitoring patterns
        self.suspicious_patterns = [
            # SQL injection patterns
            r"(?i)(union|select|insert|update|delete|drop|create|alter|exec|execute)",
            # XSS patterns
            r"(?i)(<script|javascript:|onerror=|onload=|alert\()",
            # Path traversal patterns
            r"(\.\./|\.\.\x5c|%2e%2e%2f|%2e%2e%5c)",
            # Command injection patterns
            r"(?i)(;|&&|\|\||`|\$\(|\${|exec|system|eval)"
        ]
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Monitor requests for security threats."""
        
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Initialize audit service if needed
        if not self.audit_service:
            try:
                self.audit_service = await get_comprehensive_audit_service()
            except Exception:
                pass
        
        # Monitor request for suspicious patterns
        if self.audit_service:
            await self._monitor_request_security(scope)
        
        await self.app(scope, receive, send)
    
    async def _monitor_request_security(self, scope: Scope) -> None:
        """Monitor request for security threats and log suspicious activity."""
        
        try:
            path = scope.get("path", "")
            query_string = scope.get("query_string", b"").decode()
            headers = dict(scope.get("headers", []))
            client = scope.get("client", ("unknown", None))
            
            # Check for suspicious patterns in URL and query parameters
            suspicious_content = f"{path} {query_string}"
            
            import re
            for pattern in self.suspicious_patterns:
                if re.search(pattern, suspicious_content):
                    await self.audit_service.log_audit_event(
                        event_type=AuditEventType.SUSPICIOUS_ACTIVITY,
                        ip_address=client[0] if client else "unknown",
                        user_agent=headers.get(b"user-agent", b"").decode(),
                        resource_type="security_threat",
                        action="potential_attack_detected",
                        severity=AuditSeverity.HIGH,
                        compliance_tags=["security", "threat_detection"],
                        metadata={
                            "pattern_matched": pattern,
                            "suspicious_content": suspicious_content[:500],  # Limit size
                            "path": path,
                            "query_string": query_string
                        }
                    )
                    break
        
        except Exception as e:
            logger.error(f"Failed to monitor request security: {str(e)}")