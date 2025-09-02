"""
Task Executor Agent - Specialized agent for executing implementation tasks.

This agent handles practical implementation tasks like GitHub repository creation,
project setup, API integration, and other operational activities.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import json

from pydantic_ai import Agent as PydanticAIAgent
from pydantic import BaseModel, Field
import structlog

from .base_agent import BaseAgent, AgentResult
from core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class TaskExecutorAgent(BaseAgent):
    """
    Task Executor Agent specializing in implementation and operational tasks.
    
    Capabilities:
    - GitHub repository creation and setup
    - Project structure creation
    - Configuration file generation
    - API integration setup
    - Deployment preparation
    - Task management and tracking
    """
    
    def _initialize_agent(self) -> None:
        """Initialize the PydanticAI agent for task execution."""
        self.pydantic_agent = PydanticAIAgent(
            model_name='openai:gpt-4o-mini',  # Use faster model for implementation tasks
            system_prompt="""You are the Task Executor Agent for an AI-powered strategic planning platform.

You specialize in executing practical implementation tasks and operational activities. Your role is to take strategic plans and requirements and execute the concrete steps needed to implement them.

**Core Capabilities:**

1. **GitHub Repository Management**:
   - Create repository structures
   - Generate README files and documentation
   - Set up issue templates and project boards
   - Configure repository settings and permissions

2. **Project Setup and Configuration**:
   - Generate project structure and scaffolding
   - Create configuration files (package.json, requirements.txt, etc.)
   - Set up development environments
   - Configure build and deployment pipelines

3. **API Integration and Setup**:
   - Create API endpoint structures
   - Generate OpenAPI/Swagger specifications
   - Set up authentication and authorization
   - Configure database connections and models

4. **Task Management and Tracking**:
   - Break down high-level requirements into specific tasks
   - Create implementation timelines and milestones
   - Track progress and dependencies
   - Generate status reports and updates

5. **Documentation and Communication**:
   - Create implementation guides and tutorials
   - Generate technical documentation
   - Prepare deployment instructions
   - Create stakeholder communication materials

**Execution Principles:**
- Practical and actionable: All outputs should be immediately implementable
- Standards-compliant: Follow industry best practices and conventions
- Well-documented: Provide clear documentation and instructions
- Error-resistant: Include error handling and validation
- Scalable: Design for growth and future enhancement

**Response Format:**
For implementation tasks, provide:
- Step-by-step instructions
- Code examples and templates
- Configuration files
- Validation criteria
- Troubleshooting guidance

You focus on turning strategic plans into concrete, executable actions.""",
            deps_type=Dict[str, Any]
        )
    
    async def execute(self, operation: str, context: Dict[str, Any]) -> AgentResult:
        """Execute a task implementation operation."""
        start_time = self._log_operation_start(operation, context)
        
        try:
            if operation == "create_github_repository":
                result = await self._create_github_repository(context)
            elif operation == "setup_project_structure":
                result = await self._setup_project_structure(context)
            elif operation == "generate_api_endpoints":
                result = await self._generate_api_endpoints(context)
            elif operation == "create_deployment_config":
                result = await self._create_deployment_config(context)
            elif operation == "break_down_requirements":
                result = await self._break_down_requirements(context)
            elif operation == "create_implementation_plan":
                result = await self._create_implementation_plan(context)
            elif operation == "generate_documentation":
                result = await self._generate_documentation(context)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            processing_time_ms = self._log_operation_complete(operation, start_time, True)
            
            return self._create_success_result(
                result=result,
                processing_time_ms=processing_time_ms,
                confidence_score=0.9  # High confidence for concrete implementation tasks
            )
            
        except Exception as e:
            processing_time_ms = self._log_operation_complete(operation, start_time, False, str(e))
            return self._create_error_result(
                error=str(e),
                metadata={"processing_time_ms": processing_time_ms}
            )
    
    async def _create_github_repository(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create GitHub repository structure and configuration."""
        
        project_name = self._extract_context_parameter(context, "project_name")
        project_description = self._extract_context_parameter(context, "project_description")
        project_type = self._extract_context_parameter(context, "project_type", required=False, default="web_application")
        tech_stack = self._extract_context_parameter(context, "tech_stack", required=False, default=[])
        
        github_prompt = f"""Create a comprehensive GitHub repository setup for:

Project Name: {project_name}
Description: {project_description}
Project Type: {project_type}
Technology Stack: {json.dumps(tech_stack, indent=2)}

Generate complete repository configuration including:

1. **Repository Structure**:
   - Directory layout and organization
   - Standard files and folders
   - Development vs production structure
   - Documentation organization

2. **README.md**:
   - Project overview and description
   - Installation and setup instructions
   - Usage examples and documentation links
   - Contributing guidelines
   - License and contact information

3. **Configuration Files**:
   - .gitignore appropriate for the tech stack
   - Package configuration (package.json, requirements.txt, etc.)
   - Environment configuration templates
   - CI/CD pipeline configuration

4. **Issue Templates**:
   - Bug report template
   - Feature request template
   - Question/support template
   - Pull request template

5. **Project Board Setup**:
   - Column structure for project management
   - Initial issue categories and labels
   - Milestone planning structure

6. **Repository Settings**:
   - Branch protection rules
   - Collaborator permissions
   - Integration recommendations
   - Security settings

Provide specific, implementable configurations and file contents."""

        result = await self.pydantic_agent.run(
            github_prompt,
            deps={
                "project_name": project_name,
                "project_description": project_description,
                "project_type": project_type,
                "tech_stack": tech_stack
            }
        )
        
        return {
            "github_setup": result.data,
            "project_name": project_name,
            "setup_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _setup_project_structure(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Set up project structure and scaffolding."""
        
        project_requirements = self._extract_context_parameter(context, "project_requirements")
        architecture_type = self._extract_context_parameter(context, "architecture_type", required=False, default="monolithic")
        tech_stack = self._extract_context_parameter(context, "tech_stack", required=False, default=[])
        
        structure_prompt = f"""Create a complete project structure and scaffolding:

Project Requirements: {json.dumps(project_requirements, indent=2)}
Architecture Type: {architecture_type}
Technology Stack: {json.dumps(tech_stack, indent=2)}

Generate comprehensive project structure including:

1. **Directory Structure**:
   - Complete folder hierarchy
   - Purpose of each directory
   - Organization principles
   - Scalability considerations

2. **Core Configuration Files**:
   - Build configuration (webpack, vite, etc.)
   - Dependency management
   - Environment configuration
   - Development vs production settings

3. **Application Structure**:
   - Entry points and main modules
   - Routing and navigation setup
   - State management configuration
   - API client configuration

4. **Development Environment**:
   - Local development setup
   - Development server configuration
   - Hot reloading and debugging setup
   - Testing environment configuration

5. **Code Organization Patterns**:
   - Naming conventions
   - File and folder organization
   - Import/export patterns
   - Code splitting strategies

6. **Quality Assurance Setup**:
   - Linting configuration (ESLint, Prettier, etc.)
   - Testing framework setup
   - Code coverage configuration
   - Pre-commit hooks

Provide specific file contents and configuration details for immediate implementation."""

        result = await self.pydantic_agent.run(
            structure_prompt,
            deps={
                "project_requirements": project_requirements,
                "architecture_type": architecture_type,
                "tech_stack": tech_stack
            }
        )
        
        return {
            "project_structure": result.data,
            "architecture_type": architecture_type,
            "structure_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _generate_api_endpoints(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate API endpoint structure and implementation."""
        
        api_requirements = self._extract_context_parameter(context, "api_requirements")
        framework = self._extract_context_parameter(context, "framework", required=False, default="fastapi")
        authentication_type = self._extract_context_parameter(context, "authentication_type", required=False, default="jwt")
        
        api_prompt = f"""Generate complete API endpoint structure and implementation:

API Requirements: {json.dumps(api_requirements, indent=2)}
Framework: {framework}
Authentication Type: {authentication_type}

Create comprehensive API implementation including:

1. **Endpoint Structure**:
   - RESTful endpoint design
   - URL patterns and routing
   - HTTP methods and status codes
   - Request/response schemas

2. **Authentication & Authorization**:
   - Authentication middleware
   - JWT token handling
   - Role-based access control
   - Permission management

3. **Data Models and Schemas**:
   - Request/response models
   - Data validation schemas
   - Database model definitions
   - Serialization/deserialization

4. **Business Logic Implementation**:
   - Service layer architecture
   - Business rule implementation
   - Data processing logic
   - Error handling patterns

5. **API Documentation**:
   - OpenAPI/Swagger specification
   - Endpoint documentation
   - Request/response examples
   - Authentication flow documentation

6. **Testing and Validation**:
   - Unit test structure
   - Integration test examples
   - API testing strategies
   - Validation and error handling

Provide specific code examples and implementation details for the specified framework."""

        result = await self.pydantic_agent.run(
            api_prompt,
            deps={
                "api_requirements": api_requirements,
                "framework": framework,
                "authentication_type": authentication_type
            }
        )
        
        return {
            "api_implementation": result.data,
            "framework": framework,
            "api_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _create_deployment_config(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create deployment configuration and infrastructure setup."""
        
        deployment_requirements = self._extract_context_parameter(context, "deployment_requirements")
        deployment_target = self._extract_context_parameter(context, "deployment_target", required=False, default="cloud")
        scalability_requirements = self._extract_context_parameter(context, "scalability_requirements", required=False, default={})
        
        deployment_prompt = f"""Create comprehensive deployment configuration:

Deployment Requirements: {json.dumps(deployment_requirements, indent=2)}
Deployment Target: {deployment_target}
Scalability Requirements: {json.dumps(scalability_requirements, indent=2)}

Generate complete deployment setup including:

1. **Infrastructure Configuration**:
   - Container configuration (Dockerfile, docker-compose)
   - Kubernetes manifests (if applicable)
   - Cloud provider configuration
   - Networking and security setup

2. **CI/CD Pipeline**:
   - Build pipeline configuration
   - Testing and validation stages
   - Deployment automation
   - Environment promotion strategy

3. **Environment Configuration**:
   - Production environment setup
   - Staging environment configuration
   - Development environment parity
   - Environment variable management

4. **Monitoring and Logging**:
   - Application monitoring setup
   - Log aggregation configuration
   - Health check endpoints
   - Performance monitoring

5. **Security Configuration**:
   - SSL/TLS certificate management
   - Security headers and policies
   - Access control and firewalls
   - Secret management

6. **Backup and Recovery**:
   - Database backup strategy
   - Disaster recovery procedures
   - Data retention policies
   - Rollback procedures

Provide specific configuration files and deployment scripts ready for implementation."""

        result = await self.pydantic_agent.run(
            deployment_prompt,
            deps={
                "deployment_requirements": deployment_requirements,
                "deployment_target": deployment_target,
                "scalability_requirements": scalability_requirements
            }
        )
        
        return {
            "deployment_config": result.data,
            "deployment_target": deployment_target,
            "deployment_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _break_down_requirements(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Break down high-level requirements into specific implementation tasks."""
        
        high_level_requirements = self._extract_context_parameter(context, "high_level_requirements")
        project_context = self._extract_context_parameter(context, "project_context", required=False, default={})
        team_size = self._extract_context_parameter(context, "team_size", required=False, default="small")
        
        breakdown_prompt = f"""Break down high-level requirements into specific implementation tasks:

High-Level Requirements: {json.dumps(high_level_requirements, indent=2)}
Project Context: {json.dumps(project_context, indent=2)}
Team Size: {team_size}

Create detailed task breakdown including:

1. **Task Decomposition**:
   - Specific, actionable tasks
   - Task dependencies and relationships
   - Estimated effort and complexity
   - Required skills and expertise

2. **Implementation Phases**:
   - Phase organization and sequencing
   - Milestone definitions
   - Deliverables for each phase
   - Success criteria

3. **Resource Requirements**:
   - Developer roles and responsibilities
   - Required tools and technologies
   - Infrastructure needs
   - Third-party dependencies

4. **Timeline and Scheduling**:
   - Task duration estimates
   - Critical path identification
   - Parallel work opportunities
   - Buffer time allocation

5. **Quality Assurance**:
   - Testing requirements for each task
   - Code review processes
   - Documentation requirements
   - Acceptance criteria

6. **Risk Assessment**:
   - Technical risks and mitigation
   - Resource availability risks
   - Dependency risks
   - Timeline risks

Format tasks as actionable items suitable for project management tools (Jira, GitHub Issues, etc.)."""

        result = await self.pydantic_agent.run(
            breakdown_prompt,
            deps={
                "high_level_requirements": high_level_requirements,
                "project_context": project_context,
                "team_size": team_size
            }
        )
        
        return {
            "task_breakdown": result.data,
            "team_size": team_size,
            "breakdown_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _create_implementation_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive implementation plan."""
        
        project_requirements = self._extract_context_parameter(context, "project_requirements")
        timeline_constraints = self._extract_context_parameter(context, "timeline_constraints", required=False, default={})
        resource_constraints = self._extract_context_parameter(context, "resource_constraints", required=False, default={})
        
        plan_prompt = f"""Create a comprehensive implementation plan:

Project Requirements: {json.dumps(project_requirements, indent=2)}
Timeline Constraints: {json.dumps(timeline_constraints, indent=2)}
Resource Constraints: {json.dumps(resource_constraints, indent=2)}

Develop detailed implementation plan including:

1. **Project Overview**:
   - Scope and objectives
   - Success criteria and metrics
   - Key stakeholders and roles
   - Communication plan

2. **Implementation Phases**:
   - Phase breakdown and objectives
   - Deliverables and milestones
   - Dependencies between phases
   - Go/no-go criteria

3. **Resource Planning**:
   - Team structure and roles
   - Skill requirements and gaps
   - Tool and technology needs
   - Budget and cost considerations

4. **Timeline and Scheduling**:
   - Detailed project timeline
   - Critical path analysis
   - Risk buffer allocation
   - Dependency management

5. **Quality Assurance Plan**:
   - Testing strategy and approach
   - Code review processes
   - Documentation requirements
   - Performance benchmarks

6. **Risk Management**:
   - Risk identification and assessment
   - Mitigation strategies
   - Contingency planning
   - Risk monitoring process

7. **Communication and Reporting**:
   - Status reporting schedule
   - Stakeholder communication plan
   - Issue escalation procedures
   - Progress tracking methods

Create an actionable plan that can be immediately implemented by the development team."""

        result = await self.pydantic_agent.run(
            plan_prompt,
            deps={
                "project_requirements": project_requirements,
                "timeline_constraints": timeline_constraints,
                "resource_constraints": resource_constraints
            }
        )
        
        return {
            "implementation_plan": result.data,
            "plan_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _generate_documentation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive project documentation."""
        
        project_details = self._extract_context_parameter(context, "project_details")
        documentation_type = self._extract_context_parameter(context, "documentation_type", required=False, default="complete")
        target_audience = self._extract_context_parameter(context, "target_audience", required=False, default="developers")
        
        documentation_prompt = f"""Generate comprehensive project documentation:

Project Details: {json.dumps(project_details, indent=2)}
Documentation Type: {documentation_type}
Target Audience: {target_audience}

Create thorough documentation including:

1. **Project Overview**:
   - Project description and objectives
   - Architecture overview
   - Technology stack and dependencies
   - Getting started guide

2. **Development Documentation**:
   - Setup and installation instructions
   - Development workflow and conventions
   - Code structure and organization
   - API documentation and examples

3. **Deployment Documentation**:
   - Deployment procedures and requirements
   - Environment configuration
   - Monitoring and maintenance
   - Troubleshooting guide

4. **User Documentation**:
   - User guide and tutorials
   - Feature documentation
   - FAQ and common issues
   - Best practices and tips

5. **Technical Reference**:
   - API reference documentation
   - Configuration options
   - Database schema and models
   - Security and performance considerations

6. **Maintenance Documentation**:
   - Backup and recovery procedures
   - Update and upgrade processes
   - Performance optimization
   - Security maintenance

Format documentation for easy navigation and maintenance, using appropriate markup and structure."""

        result = await self.pydantic_agent.run(
            documentation_prompt,
            deps={
                "project_details": project_details,
                "documentation_type": documentation_type,
                "target_audience": target_audience
            }
        )
        
        return {
            "project_documentation": result.data,
            "documentation_type": documentation_type,
            "target_audience": target_audience,
            "documentation_timestamp": datetime.utcnow().isoformat()
        }