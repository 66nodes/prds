"""
Strategic Planning Platform - FastAPI Backend
AI-Powered PRD Generation with GraphRAG Validation
"""

from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from api.endpoints import auth, dashboard, prd, validation, websocket
from core.config import get_settings
from core.database import init_all_databases, close_all_databases, check_all_databases_health
from core.logging_config import setup_logging
from core.middleware import LoggingMiddleware, RateLimitMiddleware

# Initialize settings and logging
settings = get_settings()
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    logger.info("Starting Strategic Planning Platform API")
    
    # Initialize all databases
    await init_all_databases()
    logger.info("All databases initialized successfully")
    
    # Setup OpenTelemetry tracing
    if settings.enable_tracing:
        setup_tracing()
        logger.info("OpenTelemetry tracing enabled")
    
    logger.info("Application startup complete")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down Strategic Planning Platform API")
    await close_all_databases()
    logger.info("All database connections closed")


# Create FastAPI application
app = FastAPI(
    title="Strategic Planning Platform API",
    description="""
    AI-Powered Strategic Planning Platform with GraphRAG validation.
    
    ## Features
    
    * **4-Phase PRD Creation Workflow**: From concept to comprehensive document
    * **GraphRAG Validation**: <2% hallucination rate through three-tier validation
    * **PydanticAI Agents**: Intelligent PRD processing and task generation
    * **GitHub Integration**: Automated project setup and task management
    * **Enterprise Security**: JWT auth, RBAC, and comprehensive audit logging
    
    ## Performance Targets
    
    * API Response Time: <200ms P95
    * Document Generation: <60 seconds
    * Concurrent Users: 100+ with stable performance
    * Uptime: 99.9% SLA
    """,
    version="1.0.0",
    contact={
        "name": "Strategic Planning Platform Team",
        "email": "support@strategicplanning.ai",
    },
    license_info={
        "name": "Proprietary",
    },
    lifespan=lifespan,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
)

# Configure middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)

# Include API routers
app.include_router(
    auth.router,
    prefix="/api/v1/auth",
    tags=["Authentication"]
)

app.include_router(
    prd.router,
    prefix="/api/v1/prd",
    tags=["PRD Generation"]
)

app.include_router(
    validation.router,
    prefix="/api/v1/validation",
    tags=["GraphRAG Validation"]
)

app.include_router(
    dashboard.router,
    prefix="/api/v1/dashboard",
    tags=["Dashboard & Analytics"]
)

app.include_router(
    websocket.router,
    prefix="/api/v1",
    tags=["WebSocket & Real-time"]
)


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Strategic Planning Platform API",
        "version": "1.0.0",
        "status": "operational",
        "features": [
            "AI-Powered PRD Generation",
            "GraphRAG Validation (<2% hallucination rate)",
            "PydanticAI Agent Processing", 
            "GitHub Integration",
            "Enterprise Security"
        ],
        "docs": "/docs" if settings.environment != "production" else None
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        # Check all database components
        components_health = await check_all_databases_health()
        
        health_status = {
            "status": "healthy",
            "timestamp": settings.current_timestamp(),
            "version": "1.0.0",
            "components": {
                **components_health,
                "api": {"status": "healthy", "response_time_ms": "<1"}
            }
        }
        
        # Determine overall health
        if any(comp["status"] != "healthy" for comp in health_status["components"].values()):
            health_status["status"] = "degraded"
            
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": settings.current_timestamp(),
            "error": str(e)
        }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Global HTTP exception handler."""
    logger.error(
        f"HTTP {exc.status_code} error",
        extra={
            "path": request.url.path,
            "method": request.method,
            "error": exc.detail,
            "user": getattr(request.state, "user_id", None)
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": settings.current_timestamp(),
            "path": request.url.path
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(
        f"Unhandled exception: {str(exc)}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "error_type": type(exc).__name__,
            "user": getattr(request.state, "user_id", None)
        },
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": settings.current_timestamp(),
            "path": request.url.path,
            "request_id": getattr(request.state, "request_id", None)
        }
    )


def setup_tracing():
    """Setup OpenTelemetry distributed tracing."""
    resource = Resource(attributes={
        SERVICE_NAME: "strategic-planning-api"
    })
    
    provider = TracerProvider(resource=resource)
    
    jaeger_exporter = JaegerExporter(
        agent_host_name=settings.jaeger_host,
        agent_port=settings.jaeger_port,
    )
    
    processor = BatchSpanProcessor(jaeger_exporter)
    provider.add_span_processor(processor)
    
    trace.set_tracer_provider(provider)
    
    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)




if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
        log_config=None,  # Use our custom logging
        access_log=False  # We handle this in middleware
    )