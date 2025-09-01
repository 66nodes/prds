# CLAUDE.md - AI Agent Context for Strategic Planning Platform

## 🎯 Project Context & Objectives

**Primary Mission**: Build an enterprise-grade AI-powered strategic planning platform that reduces planning cycles from weeks to hours through conversational AI workflows while maintaining <2% hallucination rate.

**Current Status**: MVP Development Phase  
**Target Delivery**: Q2 2025  
**Architecture**: Nuxt.js 4 + FastAPI + Neo4j GraphRAG

## 📁 Document Access Patterns

### Critical Path Documents
```yaml
Priority_1_Master_Documents:
  - docs/PRD.md                 # Single source of truth - 1073 lines
  - docs/005-TRD.md            # Technical architecture foundation  
  - docs/006-Rag-Stratety.md   # GraphRAG hallucination prevention

Priority_2_Implementation:
  - docs/008-ui-ux-requirements.md     # UI/UX specifications
  - docs/010-FrontEnd-to-Orchestration.md  # API integration patterns
  - docs/git-instructions.md          # Development workflows

Priority_3_Business_Context:
  - docs/001_project_charter.md       # Project scope and objectives
  - docs/Strategic-Planning-document.md  # Business strategy
  - docs/review_findings.md           # Known architectural gaps
```

### Document Hierarchy Navigation
```
📋 PRD.md (Master)
├── 🏗️ Technical Specifications
│   ├── 005-TRD.md (Infrastructure)
│   ├── 006-Rag-Stratety.md (GraphRAG)
│   └── 007-neo4j-content-pipeline.md (Data)
├── 💻 Development Guides  
│   ├── git-instructions.md (CI/CD)
│   ├── 008-ui-ux-requirements.md (Frontend)
│   └── 010-FrontEnd-to-Orchestration.md (Integration)
├── 📊 Business Documentation
│   ├── 001_project_charter.md (Charter)
│   └── Strategic-Planning-document.md (Strategy)
└── 🔍 Quality Assurance
    └── review_findings.md (Gaps & Issues)
```

## 🧠 AI Agent Optimization Context

### GraphRAG Integration Requirements
```typescript
// Critical constraint for all AI operations
interface HallucinationPrevention {
  validationLayers: ['entity', 'community', 'global']
  confidenceThreshold: 0.8
  maxHallucinationRate: 0.02  // <2% requirement
  validationInterval: 30      // seconds
}
```

### Performance Optimization Patterns

#### Frontend (Nuxt.js 4)
- **Bundle Size**: <500KB initial load
- **Time to Interactive**: <3s on 3G
- **Component Library**: Nuxt UI + Reka UI (50+ components)
- **State Management**: Pinia with real-time WebSocket updates

#### Backend (FastAPI + Neo4j)
- **API Response**: <200ms P95 for complex queries
- **Concurrent Users**: 100+ with auto-scaling
- **GraphRAG Validation**: <500ms for comprehensive validation
- **Database Queries**: <50ms for indexed operations

### AI Workflow Orchestration
```python
# 4-Phase Conversational Pipeline
workflow_phases = {
    'phase_0': 'project_invitation',      # Concept capture
    'phase_1': 'objective_clarification', # AI-generated questions  
    'phase_2': 'objective_drafting',      # SMART objectives
    'phase_3': 'section_co_creation',     # Collaborative building
    'phase_4': 'synthesis_finalization'   # Document generation
}
```

## 🚀 Integration Points for AI Agents

### 1. Context Manager Integration
```yaml
primary_context_sources:
  - project_requirements: docs/PRD.md
  - technical_architecture: docs/005-TRD.md
  - known_issues: docs/review_findings.md
  
context_refresh_triggers:
  - document_updates: immediate
  - architecture_changes: high_priority
  - performance_issues: critical_priority
```

### 2. Document Generation Agents
```yaml
draft_agent_config:
  model: claude-3-sonnet
  context_limit: 32k_tokens
  validation_required: true
  graphrag_integration: mandatory
  
judge_agent_config:
  model: opus
  critique_mode: comprehensive
  quality_thresholds:
    technical_accuracy: 0.95
    completeness: 0.90
    consistency: 0.95
```

### 3. Code Generation Optimization
```typescript
// Frontend component generation context
interface ComponentContext {
  framework: 'nuxt-4' | 'vue-3'
  ui_library: 'nuxt-ui' | 'reka-ui'
  styling: 'tailwind-css'
  theme: 'ink-indigo'
  typescript: true
  composition_api: true
}

// Backend service generation context  
interface ServiceContext {
  framework: 'fastapi'
  database: 'neo4j' | 'postgresql'
  auth: 'supabase-jwt'
  validation: 'pydantic-v2'
  async_patterns: true
}
```

## 📊 Performance Monitoring for AI Operations

### Response Time Targets
```yaml
ai_operation_slas:
  content_generation: <10s
  graphrag_validation: <2s
  document_synthesis: <30s
  ui_component_generation: <5s
  api_endpoint_creation: <15s

quality_metrics:
  hallucination_rate: <2%
  stakeholder_satisfaction: >90%
  technical_accuracy: >95%
  implementation_success: >85%
```

### Error Recovery Patterns
```python
class AIOperationRecovery:
    """Recovery patterns for AI agent failures"""
    
    def handle_graphrag_timeout(self):
        # Fallback to cached validation + human review flag
        return {'validation': 'cached', 'requires_review': True}
    
    def handle_generation_failure(self):
        # Retry with simpler prompt + context reduction
        return {'retry_mode': 'simplified', 'context_reduction': 0.3}
    
    def handle_integration_error(self):
        # Generate basic implementation + enhancement suggestions
        return {'implementation': 'basic', 'enhancement_tasks': list}
```

## 🔧 Development Workflow Integration

### Git Workflow Context
- **Primary Branch**: `main` (production-ready)
- **Feature Branches**: `feature/ai-{component-name}`
- **Review Process**: Automated AI code review + human validation
- **CI/CD**: GitHub Actions with comprehensive test suite

### Quality Gates for AI-Generated Code
```yaml
pre_commit_checks:
  - lint: eslint, pylint, bandit
  - type_check: typescript, mypy
  - test: vitest, pytest (>85% coverage)
  - security: dependency scan, SAST

pre_merge_validation:
  - graphrag_validation: entity + community + global
  - performance_testing: load testing, benchmarks
  - integration_testing: e2e with playwright
  - accessibility: wcag 2.1 aa compliance
```

### Code Pattern Recognition
```typescript
// Frontend patterns to follow
const nuxt4Patterns = {
  composables: 'use{FeatureName}' // useApiClient, usePrdWorkflow
  components: 'PascalCase'        // PrdGenerator, WorkflowStepper  
  pages: 'kebab-case'            // prd-creation, dashboard
  stores: '{feature}Store'        // prdStore, authStore
}

// Backend patterns to follow  
const fastapiPatterns = {
  routers: '/api/v1/{resource}'   // /api/v1/prds, /api/v1/users
  services: '{Resource}Service'   // PrdService, GraphRagService
  models: '{Resource}Model'       // PrdModel, UserModel  
  dependencies: 'get_{resource}' // get_db, get_current_user
}
```

## 🎯 Specialized Agent Contexts

### Frontend Development Agents
```yaml
vue_expert_context:
  framework: nuxt-4
  patterns: composition-api
  state: pinia-stores
  routing: file-based
  styling: tailwind-ink-indigo
  
ui_designer_context:
  component_library: nuxt-ui + reka-ui
  design_tokens: custom-black-scale
  accessibility: wcag-2.1-aa
  responsive: mobile-first
  theme_support: dark-light-toggle
```

### Backend Development Agents
```yaml
backend_architect_context:
  architecture: microservices-fastapi
  database: neo4j-primary + postgresql-auth
  caching: redis-multi-tier
  auth: supabase-jwt-rbac
  monitoring: prometheus-grafana
  
ai_engineer_context:
  llm_integration: openrouter-multi-model
  graphrag: microsoft-graphrag + llamaindex
  validation: three-tier-pipeline
  fallback: local-ollama-models
```

### Quality Assurance Agents
```yaml
test_automator_context:
  frontend_testing: vitest + playwright
  backend_testing: pytest + testcontainers
  coverage_target: 85%_minimum
  e2e_scenarios: 4-phase-workflow
  
security_auditor_context:
  threat_model: owasp-top-10
  compliance: gdpr + ccpa + sox
  encryption: aes-256 + tls-1.3
  access_control: rbac + zero-trust
```

## 🚨 Critical Architectural Constraints

### Known Issues (from review_findings.md)
```yaml
architectural_gaps:
  graphrag_orchestration:
    issue: "Lack of technical detail for three-tier validation"
    impact: "May not achieve <2% hallucination rate"
    priority: critical
    
  enterprise_scaling:
    issue: "Missing multi-region and horizontal GraphRAG patterns"
    impact: "Cannot support 500+ concurrent users"
    priority: high
    
  monitoring_observability:
    issue: "Insufficient GraphRAG quality metrics specification"
    impact: "Reactive vs proactive hallucination detection"
    priority: high
```

### Implementation Priority Matrix
```yaml
p0_critical:
  - graphrag_validation_pipeline
  - authentication_infrastructure  
  - core_prd_workflow_phases

p1_high:
  - real_time_collaboration
  - document_generation_engine
  - performance_optimization

p2_medium:
  - advanced_ui_components
  - monitoring_dashboards
  - integration_apis
```

## 💡 AI Agent Coordination Patterns

### Multi-Agent Orchestration
```python
class AgentOrchestration:
    """Coordinate multiple AI agents for complex tasks"""
    
    async def prd_generation_workflow(self, user_input: str):
        # Phase 1: Context analysis
        context = await self.context_manager.analyze(user_input)
        
        # Phase 2: Draft generation  
        draft = await self.draft_agent.generate(context)
        
        # Phase 3: GraphRAG validation
        validation = await self.graphrag_validator.validate(draft)
        
        # Phase 4: Quality assessment
        quality_score = await self.judge_agent.evaluate(draft, validation)
        
        # Phase 5: Refinement loop
        if quality_score < 0.9:
            return await self.refinement_loop(draft, validation, quality_score)
            
        return {'document': draft, 'validation': validation, 'score': quality_score}
```

### Context Sharing Protocol
```yaml
shared_context_keys:
  project_domain: "ai-strategic-planning-platform"
  tech_stack: ["nuxt-4", "fastapi", "neo4j", "graphrag"]
  performance_targets: {"api_response": "200ms", "hallucination": "2%"}
  current_phase: "mvp-development"
  
context_update_triggers:
  - document_changes: broadcast_to_all_agents
  - architecture_updates: priority_notification
  - performance_issues: immediate_alert
```

---

## 🔄 Continuous Optimization

### Performance Monitoring
- Monitor AI agent response times and quality scores
- Track GraphRAG validation effectiveness and accuracy
- Analyze user workflow completion rates and satisfaction
- Measure system performance against SLA targets

### Learning Loop Integration
- Capture successful patterns for reuse across agents
- Document failure modes and recovery strategies
- Refine prompts based on quality assessment results
- Optimize context usage for better token efficiency

**Last Updated**: January 2025  
**Context Version**: v2.2.0  
**AI Agent Compatibility**: Claude Code + Task Master AI