### `EXECUTION_PROMPTS.md`

```markdown
# Sequential Execution Prompts for Claude

## 🎯 Master Execution Sequence

### PROMPT 1: Project Initialization
```

Initialize the AI Agent Platform project with Hybrid RAG architecture:

1. Load PROJECT_EXECUTION_MANIFEST.yaml
2. Create all directory structures from the manifest
3. Initialize git repository with .gitignore
4. Setup environment variables template
5. Validate all prerequisites are met
6. Update manifest status for completed tasks

Start with infrastructure phase and create docker-stack.yml for Milvus + Neo4j deployment.

```

### PROMPT 2: Infrastructure Deployment
```

Deploy the complete infrastructure stack:

1. Execute docker-stack.yml deployment
2. Verify all services are healthy (Milvus, Neo4j, Redis, Pulsar)
3. Initialize Neo4j with GraphRAG schema from database/neo4j/schema.cypher
4. Create Milvus collections with proper indexes
5. Setup monitoring with Prometheus and Grafana
6. Run infrastructure validation tests
7. Update PROJECT_EXECUTION_MANIFEST.yaml with completed tasks

Provide health check results for all services.

```

### PROMPT 3: Backend Services Implementation
```

Implement all backend services for the platform:

1. Create FastAPI application structure with proper routing
2. Implement HybridRAGService with Milvus and Neo4j integration
3. Build agent orchestration system with Context Manager
4. Create PRD generation pipeline with validation stages
5. Setup WebSocket for real-time updates
6. Implement authentication and authorization
7. Add comprehensive error handling and logging
8. Create all API endpoints as defined in docs/openapi.yaml
9. Run backend validation: pytest --cov=backend --cov-fail-under=90

Ensure all tests pass before proceeding.

```

### PROMPT 4: GraphRAG Implementation
```

Build the complete GraphRAG system:

1. Implement entity extraction pipeline using spaCy/transformers
2. Create relationship extraction for knowledge graph
3. Build hallucination detection with <2% threshold
4. Optimize Neo4j queries for <50ms response time
5. Implement graph traversal strategies
6. Create validation pipeline for content
7. Setup fact-checking against knowledge base
8. Run hallucination rate validation: python scripts/validate_hallucination_rate.py

Provide metrics showing hallucination rate is below 2%.

```

### PROMPT 5: Frontend Development
```

Create the complete Nuxt 4 frontend application:

1. Setup Nuxt 4 with TypeScript and Tailwind CSS
2. Implement authentication with JWT and refresh tokens
3. Create PRD generation workflow UI with real-time validation
4. Build dashboard with analytics and metrics
5. Develop agent management interface
6. Implement WebSocket integration for real-time updates
7. Create all components with proper TypeScript types
8. Setup Pinia stores for state management
9. Run frontend validation: npm run build && npm run typecheck && npm run test:e2e

Ensure zero TypeScript errors and all E2E tests pass.

```

### PROMPT 6: Agent System Implementation
```

Implement the complete 100+ agent system:

1. Create all agent definitions in .claude/agents/
2. Implement Context Manager for orchestration
3. Build task distribution and execution system
4. Create agent state management
5. Implement prompt engineering system
6. Setup agent communication protocols
7. Create agent monitoring and metrics
8. Validate all agents: python scripts/validate_agent_definitions.py

Show successful orchestration of a complex multi-agent workflow.

```

### PROMPT 7: Testing Suite
```

Create comprehensive testing suite:

1. Write unit tests for all services (>90% coverage)
2. Create integration tests for API endpoints
3. Implement E2E tests for critical user journeys
4. Setup performance testing with Locust
5. Create hallucination validation tests
6. Implement security testing
7. Run full test suite: ./scripts/run_all_tests.sh

Provide test coverage report and performance metrics.

```

### PROMPT 8: Deployment & Production
```

Prepare for production deployment:

1. Create optimized Docker images
2. Setup CI/CD with GitHub Actions
3. Create Kubernetes manifests
4. Configure production environment variables
5. Setup monitoring and alerting
6. Create backup and recovery procedures
7. Document deployment process
8. Run production readiness check: ./scripts/production_check.sh

Confirm all production criteria are met.

```

### PROMPT 9: Final Validation
```

Perform final validation of the entire system:

1. Review PROJECT_EXECUTION_MANIFEST.yaml - ensure 100% completion
2. Run comprehensive system tests
3. Validate performance metrics (<200ms API, <2% hallucination)
4. Check security vulnerabilities
5. Verify all documentation is complete
6. Run final acceptance tests
7. Generate project completion report

```

```
