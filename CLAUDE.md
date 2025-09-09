# CLAUDE.md Review & Enhancement

Your draft is solid but could be enhanced for optimal Claude Code performance. Here's my analysis
and improved version based on best practices:

## 🎯 Key Improvements Made

### 1. **Added Explicit Success Patterns**

- Clear validation criteria for each operation
- Specific error patterns to watch for
- Recovery strategies for common failures

### 2. **Enhanced Decision Trees**

- When to use which agent
- Clear escalation paths
- Specific trigger conditions

### 3. **Better Code Context**

- Common code patterns with examples
- Anti-patterns to avoid
- Performance optimization hints

---

# CLAUDE.md - Enhanced Version

```markdown
# CLAUDE.md - Claude Code Context for AI Agent Platform

Last Updated: 2025-01-20 | Version: v2.2.0

## 🎯 PRIMARY OBJECTIVE

You are Claude Code working on an enterprise AI agent orchestration platform. Your success is
measured by:

- **PRD Generation**: <10 min completion, <2% hallucination rate
- **Code Quality**: 90%+ test coverage, <200ms API latency
- **User Satisfaction**: 80%+ stakeholder approval on first generation

## 🚀 IMMEDIATE ACTION PATTERNS

### When User Says... → You Should...
```

"Create a PRD for..." → Invoke Context Manager → Draft Agent → GraphRAG Validator "Fix the bug
in..." → Analyze error → Check tests → Propose fix with validation "Optimize performance..." →
Profile code → Identify bottlenecks → Implement caching "Add new feature..." → Check architecture →
Create API contract → Implement with tests

````

## 🏗️ SYSTEM ARCHITECTURE - INTERACTIVE MAP

```mermaid
graph TB
    subgraph "Frontend [Nuxt 4]"
        FE[Components] --> API[API Client]
        API --> AUTH[JWT/RBAC]
    end

    subgraph "Backend [FastAPI]"
        GATEWAY[API Gateway] --> SERVICES[Services]
        SERVICES --> GRAPHRAG[GraphRAG Validator]
        GRAPHRAG --> NEO4J[(Neo4j Graph)]
    end

    subgraph "AI Layer"
        AGENTS[100+ Agents] --> ORCHESTRATOR[Context Manager]
        ORCHESTRATOR --> LLM[OpenRouter/Multi-LLM]
    end

    FE --> GATEWAY
    SERVICES --> AGENTS
````

## 🎮 QUICK COMMAND REFERENCE

```bash
# MOST USED COMMANDS - Copy & Paste Ready
npm run dev                    # Start frontend
uvicorn main:app --reload      # Start backend
docker-compose up -d           # Full stack
npm run test:e2e              # Run E2E tests
python -m pytest -xvs         # Debug backend tests

# VALIDATION & QUALITY
npm run typecheck && npm run lint:fix  # Frontend validation
python scripts/validate_graphrag.py    # GraphRAG check
curl -X POST http://localhost:8000/health  # API health

# DEBUG HELPERS
tail -f logs/api.log | grep ERROR     # Error monitoring
wscat -c ws://localhost:8000/ws       # WebSocket debug
redis-cli MONITOR                      # Cache monitoring
```

## 🧠 AGENT ORCHESTRATION PLAYBOOK

### Decision Tree for Agent Selection

```python
def select_agent(task_type: str, complexity: int) -> Agent:
    """
    Use this logic when determining which agent to invoke
    """
    if task_type == "strategic_planning":
        if complexity > 8:
            return ContextManager()  # Opus - orchestrates multiple agents
        return DraftAgent()  # Sonnet - single document generation

    elif task_type == "technical_implementation":
        if "graphrag" in requirements:
            return AIEngineer()  # Specialized for GraphRAG
        elif "frontend" in requirements:
            return FrontendDeveloper()  # Nuxt/Vue specialist
        elif "api" in requirements:
            return BackendArchitect()  # FastAPI expert

    elif task_type == "validation":
        return JudgeAgent()  # Always validate outputs
```

### Multi-Agent Coordination Pattern

```typescript
// Frontend pattern for agent coordination
const executeWorkflow = async (requirement: string) => {
  // 1. Initialize context
  const context = await contextManager.initialize(requirement);

  // 2. Orchestrate agents
  const tasks = await taskOrchestrator.plan(context);

  // 3. Execute in parallel where possible
  const results = await Promise.all(tasks.map(task => taskExecutor.run(task)));

  // 4. Validate outputs
  const validated = await judgeAgent.validate(results);

  // 5. Store if valid
  if (validated.score > 0.95) {
    await documentationLibrarian.store(validated);
  }

  return validated;
};
```

## 🔥 COMMON PATTERNS & SOLUTIONS

### Pattern 1: GraphRAG Validation Implementation

```python
# backend/services/graphrag_validator.py
async def validate_with_graphrag(content: str, project_id: str) -> ValidationResult:
    """
    ALWAYS use this pattern for content validation
    """
    # 1. Extract entities
    entities = await extract_entities(content)

    # 2. Query knowledge graph
    graph_data = await neo4j_client.query(
        """
        MATCH (n:Entity)-[r:RELATES_TO]->(m:Entity)
        WHERE n.project_id = $project_id
        AND n.name IN $entities
        RETURN n, r, m
        """,
        {"project_id": project_id, "entities": entities}
    )

    # 3. Calculate hallucination score
    hallucination_rate = calculate_hallucination_rate(content, graph_data)

    # 4. Enforce threshold
    if hallucination_rate > 0.02:  # 2% threshold
        raise HallucinationThresholdExceeded(
            f"Rate: {hallucination_rate:.2%}, Max: 2%"
        )

    return ValidationResult(
        content=content,
        hallucination_rate=hallucination_rate,
        graph_evidence=graph_data
    )
```

### Pattern 2: Nuxt 4 Component with TypeScript

```vue
<!-- frontend/components/PrdGenerator.vue -->
<script setup lang="ts">
// ALWAYS use this pattern for new components
import { useApiClient } from '~/composables/useApiClient';
import type { PRDRequest, PRDResponse } from '~/types/prd';

const props = defineProps<{
  projectId: string;
}>();

const { $api } = useApiClient();
const loading = ref(false);
const error = ref<Error | null>(null);

const generatePRD = async (request: PRDRequest): Promise<PRDResponse> => {
  loading.value = true;
  error.value = null;

  try {
    // Always validate input first
    if (!request.title || request.title.length < 3) {
      throw new Error('Title must be at least 3 characters');
    }

    // Call API with proper error handling
    const response = await $api.post<PRDResponse>(`/projects/${props.projectId}/prd`, request);

    // Validate response
    if (response.hallucination_rate > 0.02) {
      throw new Error('Content failed validation');
    }

    return response;
  } catch (e) {
    error.value = e as Error;
    throw e;
  } finally {
    loading.value = false;
  }
};
</script>
```

## ⚠️ CRITICAL RULES - NEVER VIOLATE

### 🔴 NEVER DO

```typescript
// ❌ NEVER modify these without approval
const PROTECTED_FILES = [
  'backend/core/security.py', // Security core
  '.github/workflows/*', // CI/CD
  '.env.production', // Production secrets
  'database/migrations/*', // Schema changes
];

// ❌ NEVER use these patterns
const FORBIDDEN_PATTERNS = {
  auth: 'Never bypass authentication checks',
  sync: 'Never use synchronous DB calls in API routes',
  any: 'Never use "any" type in TypeScript production code',
  console: 'Never leave console.log in production',
  secrets: 'Never hardcode API keys or secrets',
};
```

### ✅ ALWAYS DO

```typescript
// ✅ ALWAYS follow these patterns
const REQUIRED_PATTERNS = {
  validation: 'Always validate with GraphRAG for strategic docs',
  types: 'Always use TypeScript strict mode',
  tests: 'Always write tests for new features',
  errors: 'Always handle errors explicitly',
  auth: 'Always check permissions before operations',
};
```

## 🚨 ERROR RECOVERY PLAYBOOK

### Common Errors & Solutions

```bash
# Error: GraphRAG validation fails
Solution:
1. Check Neo4j connection: neo4j-admin memrec
2. Verify graph data: MATCH (n) RETURN count(n)
3. Rebuild index: python scripts/rebuild_graph_index.py

# Error: Frontend build fails
Solution:
1. Clear cache: rm -rf .nuxt node_modules/.cache
2. Check types: npm run typecheck
3. Reinstall: rm -rf node_modules && npm install

# Error: API timeout
Solution:
1. Check Redis: redis-cli ping
2. Profile endpoint: python -m cProfile -o profile.stats main.py
3. Add caching: @cache(ttl=300) decorator

# Error: WebSocket disconnection
Solution:
1. Check connection: wscat -c ws://localhost:8000/ws
2. Verify heartbeat: See backend/services/websocket.py
3. Increase timeout: WS_TIMEOUT=30000 in .env
```

## 📊 PERFORMANCE OPTIMIZATION CHECKLIST

```python
# Run this checklist for any performance issue
performance_checklist = {
    "frontend": [
        "Bundle size < 500KB?",
        "Lazy loading implemented?",
        "Images optimized with nuxt/image?",
        "API calls debounced?",
    ],
    "backend": [
        "Database queries using indexes?",
        "N+1 queries eliminated?",
        "Redis caching enabled?",
        "Connection pooling configured?",
    ],
    "graphrag": [
        "Graph queries optimized?",
        "Batch processing for large datasets?",
        "Embeddings cached?",
        "Vector search indexes updated?",
    ]
}
```

## 🔄 CONTEXT MANAGEMENT STRATEGY

### When Context Exceeds 10K Tokens

```python
def manage_large_context(context: str) -> str:
    """
    Use this when dealing with large contexts
    """
    if len(context) > 10000:
        # 1. Prioritize recent changes
        recent = extract_recent_changes(context, days=7)

        # 2. Keep critical architecture
        architecture = extract_architecture_decisions(context)

        # 3. Maintain active tasks
        active = extract_active_tasks(context)

        # 4. Compress historical data
        compressed = compress_historical(context)

        return merge_contexts([recent, architecture, active, compressed])
    return context
```

## 🎯 SUCCESS VALIDATION CRITERIA

### For Every Feature/Fix, Validate:

```yaml
validation_criteria:
  code_quality:
    - 'Types correctly defined?'
    - 'Tests passing with >90% coverage?'
    - 'Linting rules satisfied?'
    - 'No console.logs remaining?'

  performance:
    - 'API response < 200ms?'
    - 'Frontend bundle < 500KB?'
    - 'Database queries optimized?'
    - 'Caching implemented where needed?'

  graphrag:
    - 'Hallucination rate < 2%?'
    - 'Sources properly cited?'
    - 'Graph relationships valid?'
    - 'Validation pipeline passing?'

  user_experience:
    - 'Error messages helpful?'
    - 'Loading states implemented?'
    - 'Accessibility standards met?'
    - 'Mobile responsive?'
```

## 🔧 MCP INTEGRATION POINTS

```typescript
// MCP Tool Integration Examples
const mcpTools = {
  // For complex planning
  sequential_thinking: 'Use for multi-step reasoning',

  // For web research
  web_search: 'Validate external references',

  // For documentation
  project_knowledge_search: 'Find internal docs',

  // For development
  artifacts: 'Generate code artifacts',

  // For analysis
  repl: 'Test code snippets',
};

// Example Usage Pattern
async function enhancedPRDGeneration(requirement: string) {
  // 1. Search existing knowledge
  const context = await project_knowledge_search(requirement);

  // 2. Sequential planning
  const plan = await sequential_thinking({
    thought: 'Breaking down PRD requirements',
    total_thoughts: 5,
  });

  // 3. Generate artifact
  const prd = await artifacts({
    command: 'create',
    type: 'text/markdown',
    title: 'Product Requirements Document',
  });

  return prd;
}
```

## 📚 QUICK REFERENCES

| Task               | Command/File                             | Notes                 |
| ------------------ | ---------------------------------------- | --------------------- |
| Add new agent      | `.claude/agents/[name].md`               | Follow template       |
| Update GraphRAG    | `backend/services/graphrag_validator.py` | Test thoroughly       |
| New API endpoint   | `backend/api/routes/`                    | Include OpenAPI spec  |
| Frontend component | `frontend/components/`                   | Use Composition API   |
| Database migration | `alembic revision -m "message"`          | Never modify existing |
| Deploy to staging  | `./deploy.sh staging`                    | Requires approval     |

## 🆘 WHEN STUCK

1. **Check existing patterns**: `grep -r "similar_function" .`
2. **Review agent definitions**: `.claude/agents/README.md`
3. **Consult test files**: Often contain usage examples
4. **Check GraphRAG logs**: `tail -f logs/graphrag.log`
5. **Use sequential thinking**: Break complex problems into steps

---

**Remember**: You're building an enterprise platform. Every decision impacts:

- Performance at scale (100+ concurrent users)
- Hallucination rates (must stay <2%)
- Developer experience (other team members will use this)
- Maintainability (code lives for years)

**Your North Star**: Ship fast, but ship quality. Use agents wisely.

```

## Summary of Key Enhancements

1. **Added Interactive Decision Trees** - Clear paths for common scenarios
2. **Included Copy-Paste Ready Code** - Practical patterns developers can use immediately
3. **Created Error Recovery Playbook** - Specific solutions for common problems
4. **Enhanced MCP Integration** - Shows how to leverage available tools
5. **Added Success Validation Criteria** - Clear checklist for quality assurance
6. **Improved Quick Reference Tables** - Faster access to common operations
7. **Better Context Management** - Strategies for handling large contexts
8. **More Specific Anti-Patterns** - Clear "never do this" examples

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md
```
