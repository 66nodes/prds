"""
OpenAPI/Swagger Documentation Generator for Enterprise AI Agent Platform
Auto-generates comprehensive API documentation for all endpoints.
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from typing import Dict, Any
import json
from main import app

# Create docs router
docs_router = APIRouter(tags=["API Documentation"])

@docs_router.get("/swagger-json", tags=["API Documentation"])
async def get_openapi_json():
    """Get OpenAPI JSON specification."""
    return get_openapi(
        title="Enterprise AI Agent Platform API",
        version="1.0.0",
        description="""
        Comprehensive REST API for Enterprise AI Agent Orchestration Platform.

        ## Features
        - Multi-agent orchestration with 250+ specialized agents
        - GraphRAG validation with 2% hallucination threshold
        - Real-time WebSocket communication
        - Enterprise authentication and RBAC
        - Advanced performance monitoring and analytics

        ## Authentication
        All endpoints require Bearer token authentication:
        ```
        Authorization: Bearer <your_jwt_token>
        ```

        ## Response Format
        All successful responses follow this format:
        ```json
        {
          "success": true,
          "data": {...},
          "message": "Optional success message"
        }
        ```

        ## Error Handling
        Error responses follow this format:
        ```json
        {
          "success": false,
          "data": null,
          "message": "Error description",
          "errors": ["Detailed error messages"]
        }
        ```
        """,
        routes=app.routes,
    )

@docs_router.get("/swagger", tags=["API Documentation"])
async def get_swagger_docs():
    """Serve Swagger UI documentation interface."""
    return get_swagger_ui_html(
        openapi_url="/api/v1/docs/swagger-json",
        title="Enterprise AI Agent Platform - API Documentation",
        swagger_ui_parameters={
            "presets": ["apis"],
            "layout": "StandaloneLayout",
            "displayRequestDuration": True,
            "defaultModelsExpandDepth": 3,
            "defaultModelExpandDepth": 3,
            "filter": True,
            "showExtensions": True,
            "showCommonExtensions": True,
            "supportedSubmitMethods": ["get", "post", "put", "patch", "delete"]
        }
    )

@docs_router.get("/redoc", tags=["API Documentation"])
async def get_redoc_docs():
    """Serve ReDoc documentation interface."""
    return get_redoc_html(
        openapi_url="/api/v1/docs/swagger-json",
        title="Enterprise AI Agent Platform - API Documentation"
    )

# Enhanced endpoint documentation with detailed schemas
USER_SCHEMA = {
    "User": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "format": "uuid",
                "description": "Unique user identifier",
                "example": "123e4567-e89b-12d3-a456-426614174000"
            },
            "email": {
                "type": "string",
                "format": "email",
                "description": "User email address",
                "example": "user@company.local"
            },
            "name": {
                "type": "string",
                "description": "Full user name",
                "example": "Alex Johnson"
            },
            "role": {
                "type": "string",
                "enum": ["admin", "user", "auditor", "guest"],
                "description": "User role in the system",
                "example": "user"
            },
            "isActive": {
                "type": "boolean",
                "description": "Whether the user account is active",
                "default": True
            },
            "createdAt": {
                "type": "string",
                "format": "date-time",
                "description": "Account creation timestamp",
                "example": "2024-01-15T10:30:00Z"
            },
            "lastLogin": {
                "type": "string",
                "format": "date-time",
                "description": "Last login timestamp",
                "example": "2024-09-09T14:22:00Z"
            }
        },
        "required": ["id", "email", "name", "role"]
    }
}

PRD_SCHEMA = {
    "PRDRequest": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "PRD title",
                "example": "Enhanced AI Agent Orchestration",
                "minLength": 3,
                "maxLength": 200
            },
            "description": {
                "type": "string",
                "description": "Detailed PRD description and requirements",
                "example": "Create comprehensive AI agent orchestration system with multi-LLM support..."
            },
            "projectId": {
                "type": "string",
                "format": "uuid",
                "description": "Associated project identifier"
            },
            "requirements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Functional requirements list",
                "example": ["Multi-agent coordination", "GraphRAG validation"]
            },
            "constraints": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Technical and business constraints",
                "example": ["Must support 250+ agents", "99.9% uptime SLA"]
            },
            "targetAudience": {
                "type": "string",
                "description": "Target user personas",
                "example": "Enterprise developers, AI/ML engineers"
            },
            "successMetrics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Success criteria and KPIs",
                "example": ["PRD generation <10min", "Hallucination rate <2%"]
            }
        },
        "required": ["title", "description"]
    },
    "PRDResponse": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "format": "uuid",
                "description": "Generated PRD identifier"
            },
            "title": {
                "type": "string",
                "description": "Generated PRD title"
            },
            "content": {
                "type": "string",
                "description": "Complete generated PRD content in markdown format"
            },
            "hallucination_rate": {
                "type": "number",
                "description": "GraphRAG hallucination validation score",
                "minimum": 0,
                "maximum": 1,
                "example": 0.015
            },
            "confidenceScore": {
                "type": "number",
                "description": "Overall generation confidence score",
                "minimum": 0,
                "maximum": 1,
                "example": 0.94
            },
            "graphEvidence": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/GraphEvidence"},
                "description": "Supporting graph evidence from knowledge base"
            },
            "metadata": {
                "type": "object",
                "description": "PRD generation metadata",
                "properties": {
                    "version": {"type": "string", "example": "1.0.0"},
                    "status": {
                        "type": "string",
                        "enum": ["draft", "completed"],
                        "example": "completed"
                    },
                    "author": {"type": "string", "example": "Agent Orchestrator"}
                }
            }
        }
    }
}

AGENT_SCHEMA = {
    "AgentType": {
        "type": "string",
        "enum": [
            "context_manager", "task_executor", "judge_agent",
            "documentation_specialist", "code_reviewer", "architect",
            "performance_optimizer", "security_auditor", "ui_designer",
            "api_designer", "database_optimizer", "ci_cd_specialist"
        ],
        "description": "Available agent specializations in the platform"
    },
    "AgentStatus": {
        "type": "string",
        "enum": ["idle", "processing", "error", "offline"],
        "description": "Current agent operational status"
    },
    "AgentCapabilities": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Agent capabilities and skills",
        "example": ["code_generation", "testing", "documentation"]
    },
    "AgentPerformance": {
        "type": "object",
        "properties": {
            "taskCompletionRate": {
                "type": "number",
                "description": "Percentage of successfully completed tasks",
                "minimum": 0,
                "maximum": 1,
                "example": 0.95
            },
            "averageResponseTime": {
                "type": "number",
                "description": "Average task completion time in seconds",
                "example": 45.2
            },
            "totalTasksProcessed": {
                "type": "integer",
                "description": "Total number of tasks processed",
                "example": 1247
            },
            "satisfactionScore": {
                "type": "number",
                "description": "User satisfaction rating",
                "minimum": 0,
                "maximum": 5,
                "example": 4.8
            }
        }
    }
}

PROJECT_SCHEMA = {
    "ProjectStatus": {
        "type": "string",
        "enum": ["planning", "active", "review", "completed", "cancelled"],
        "description": "Current project lifecycle stage"
    },
    "ProjectMetrics": {
        "type": "object",
        "properties": {
            "totalPRDs": {
                "type": "integer",
                "description": "Total PRDs created",
                "example": 12
            },
            "completedPRDs": {
                "type": "integer",
                "description": "Completed PRDs",
                "example": 8
            },
            "averageHallucinationRate": {
                "type": "number",
                "description": "Average hallucination detection rate",
                "example": 0.012
            },
            "timeToCompletion": {
                "type": "number",
                "description": "Average PRD completion time in hours",
                "example": 6.5
            }
        }
    }
}

VALIDATION_SCHEMA = {
    "ValidationRequest": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Content to validate against knowledge graph",
                "example": "Generated PRD content for AI platform..."
            },
            "context": {
                "type": "object",
                "description": "Validation context and constraints",
                "properties": {
                    "expectedEntities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Expected entity types to validate"
                    },
                    "hallucinationThreshold": {
                        "type": "number",
                        "description": "Maximum allowed hallucination rate",
                        "default": 0.02,
                        "maximum": 1.0
                    }
                }
            }
        },
        "required": ["content"]
    },
    "ValidationResponse": {
        "type": "object",
        "properties": {
            "valid": {
                "type": "boolean",
                "description": "Overall validation result"
            },
            "hallucinationRate": {
                "type": "number",
                "description": "Calculated hallucination rate",
                "example": 0.014
            },
            "confidence": {
                "type": "number",
                "description": "Overall validation confidence",
                "example": 0.92
            },
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "example": "inconsistency"},
                        "severity": {"type": "string", "example": "warning"},
                        "message": {"type": "string", "example": "Potential factual discrepancy"}
                    }
                },
                "description": "Validation issues found"
            },
            "graphEvidence": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/GraphEvidence"},
                "description": "Supporting evidence from knowledge graph"
            }
        }
    }
}

GRAPH_SCHEMA = {
    "GraphEvidence": {
        "type": "object",
        "properties": {
            "nodeId": {
                "type": "string",
                "description": "Knowledge graph node identifier"
            },
            "nodeType": {
                "type": "string",
                "description": "Node classification (requirement, constraint, technical_spec)",
                "example": "requirement"
            },
            "content": {
                "type": "string",
                "description": "Node content and description"
            },
            "confidence": {
                "type": "number",
                "description": "Relationship confidence score",
                "example": 0.87
            },
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "example": "supports"},
                        "targetNodeId": {"type": "string"},
                        "strength": {"type": "number", "example": 0.95}
                    }
                },
                "description": "Node relationships in knowledge graph"
            }
        }
    }
}

# Complete schema collection
API_SCHEMA = {
    **USER_SCHEMA,
    **PRD_SCHEMA,
    **AGENT_SCHEMA,
    **PROJECT_SCHEMA,
    **VALIDATION_SCHEMA,
    **GRAPH_SCHEMA,

    # Response Wrappers
    "SuccessResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean", "enum": [True]},
            "data": {"type": "object", "description": "Response payload"},
            "message": {"type": "string", "description": "Optional success message"}
        },
        "required": ["success"]
    },
    "ErrorResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean", "enum": [False]},
            "data": {"type": "object", "nullable": True},
            "message": {"type": "string", "description": "Error description"},
            "errors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Detailed error messages"
            }
        },
        "required": ["success", "message"]
    },

    # Pagination
    "PaginatedResponse": {
        "type": "object",
        "properties": {
            "data": {"type": "array", "items": {"type": "object"}},
            "pagination": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer", "minimum": 1},
                    "pageSize": {"type": "integer", "minimum": 1},
                    "total": {"type": "integer", "minimum": 0},
                    "totalPages": {"type": "integer", "minimum": 1},
                    "hasMore": {"type": "boolean"}
                }
            }
        }
    }
}

def get_enhanced_openapi_specification() -> Dict[str, Any]:
    """Generate enhanced OpenAPI specification with comprehensive schemas."""
    spec = get_openapi(
        title="Enterprise AI Agent Platform API",
        version="1.0.0",
        description="""
        Enterprise AI Agent Orchestration Platform - Complete REST API

        ## 🎯 Platform Overview
        Advanced AI-powered platform for product requirements generation with enterprise-grade
        multi-agent orchestration and GraphRAG validation (2% hallucination threshold).

        ## 🤖 Agent System
        - 250+ specialized agents across technical, business, and analysis domains
        - Intelligent agent selection using context-aware algorithms
        - Parallel execution with advanced dependency management
        - Real-time performance monitoring and optimization

        ## 🧠 GraphRAG Validation
        - Comprehensive knowledge graph validation
        - Hallucination detection with 98% accuracy
        - Entity extraction and relationship mapping
        - Continuous learning and model improvement

        ## 🚀 Enterprise Features
        - JWT-based authentication with RBAC
        - Multi-tenant architecture
        - High-availability with 99.9% uptime SLA
        - Comprehensive audit logging and compliance
        """,
        routes=app.routes,
    )

    # Add enhanced components
    if 'components' not in spec:
        spec['components'] = {}

    spec['components']['schemas'] = {**spec['components'].get('schemas', {}), **API_SCHEMA}

    # Add security schemes
    spec['components']['securitySchemes'] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT Authentication using Bearer token"
        },
        "apiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API Key authentication for programatic access"
        }
    }

    # Enhance existing endpoints with detailed documentation
    for path_name, path_item in spec.get('paths', {}).items():
        for method_name, operation in path_item.items():
            if method_name.upper() not in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                continue

            # Add comprehensive operation documentation
            operation['description'] = (
                operation.get('description', '') +
                f"\n\n**Endpoint:** {method_name.upper()} {path_name}\n" +
                "**Authentication:** Bearer/JWT token required\n" +
                "**Content-Type:** application/json\n\n"
            )

            # Add response examples
            if 'responses' in operation:
                for status_code, response in operation['responses'].items():
                    if status_code == '200':
                        response['description'] = response.get('description', '') + (
                            "\n\n### Success Response Format\n"
                            "```json\n"
                            "{\n"
                            '  "success": true,\n'
                            '  "data": {\n'
                            '    // Response data here\n'
                            "  },\n"
                            '  "message": "Operation completed successfully"\n'
                            "}\n"
                            "```"
                        )
                    elif status_code == '401':
                        response['description'] = "Authentication required or invalid token"

    # Add global security
    spec['security'] = [{"bearerAuth": []}]

    return spec

@docs_router.get("/enhanced-docs", tags=["API Documentation"])
async def get_enhanced_openapi():
    """Get enhanced OpenAPI specification with comprehensive schemas."""
    try:
        spec = get_enhanced_openapi_specification()

        return {
            "openapi": "3.0.3",
            "info": spec["info"],
            "servers": spec.get("servers", []),
            "security": spec["security"],
            "paths": spec["paths"],
            "components": spec["components"],
            "tags": spec.get("tags", [])
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate enhanced documentation: {str(e)}"
        )

@docs_router.get("/postman-collection", tags=["API Documentation"])
async def get_postman_collection():
    """Generate Postman collection for API testing."""
    try:
        from openapi_to_postman import OpenApiToPostman

        spec = get_enhanced_openapi_specification()
        converter = OpenApiToPostman(spec)
        collection = converter.convert()

        return {
            "collection": collection,
            "info": {
                "name": "Enterprise AI Agent Platform API",
                "description": "Complete Postman collection for API testing",
                "schema": "https://schema.getpostman.com/collection/v2.1.0/collection.json"
            }
        }

    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Postman collection generation requires 'openapi-to-postmanv2' package"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate Postman collection: {str(e)}"
        )

# Export router for main app
__all__ = ['docs_router', 'get_enhanced_openapi_specification']