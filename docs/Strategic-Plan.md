Strategic Implementation Guide

## **What:** Core Architecture Overview

PlanExe is an **AI-powered strategic planning engine** that transforms vague descriptions into
comprehensive, actionable project plans. Think of it as automated management consulting that
generates 50+ page strategic documents complete with work breakdown structures, risk assessments,
and implementation timelines.

**Key Value Proposition:** Input a two-sentence idea, output enterprise-grade project documentation
that typically requires weeks of consulting work.

## **Core Components:**

### 1. **Planning Pipeline (`planexe/plan/`)**

- **Multi-stage AI orchestration** using structured LLM calls
- Work Breakdown Structure (WBS) generation at 3 hierarchical levels
- Project timeline and resource allocation automation
- Risk assessment and mitigation planning

### 2. **LLM Factory (`llm_factory.py`)**

- **Provider-agnostic AI integration** supporting:
  - OpenRouter (cloud-based, recommended for production)
  - Ollama (local deployment)
  - LM Studio, OpenAI, Groq, MistralAI
- **Smart fallback mechanisms** with priority-based model selection

### 3. **Expert Modules (`planexe/expert/`, `planexe/governance/`)**

- Domain-specific planning logic for governance, compliance, and stakeholder management
- **Pre-built frameworks** for common business scenarios

### 4. **Document Generation Engine**

- Outputs multiple formats: HTML reports, CSV data, JSON structures, ZIP archives
- **Gantt charts, executive summaries, and detailed implementation guides**

## **Why:** Strategic Business Rationale

This architecture addresses **three critical enterprise pain points:**

1. **Planning Bottlenecks:** Reduces strategic planning cycles from weeks to hours
2. **Consistency Gaps:** Standardizes planning methodology across projects
3. **Resource Optimization:** Automates high-value but repetitive analytical work

**Market Positioning:** Positioned as an internal productivity multiplier rather than client-facing
solution—similar to how your team uses AI for proposal generation or technical documentation.

## **How:** Implementation Strategy

### **Phase 1: Proof of Concept (Week 1-2)**

bash

```bash
# Local deployment using OpenRouter (recommended path)
git clone https://github.com/neoneye/PlanExe.git
cd PlanExe
python3 -m venv venv && source venv/bin/activate
pip install '.[gradio-ui]'

# Configure OpenRouter API key in .env
# Launch: python -m planexe.plan.app_text2plan
```

**Why OpenRouter:** Cost-effective, reliable, no infrastructure overhead. Perfect for validating use
cases before deeper investment.

### **Phase 2: Enterprise Integration (Week 3-4)**

**Customization Priorities:**

1. **Industry-Specific Prompts** → Modify `planexe/prompt/` for your domain expertise
2. **Client Template Integration** → Extend document generation for your standard deliverable
   formats
3. **CRM/Project Management Hooks** → API integration points already exist in the FastAPI structure

### **Phase 3: Scale & Operationalize (Month 2)**

**Production Considerations:**

- **Multi-user deployment** → Already supports HuggingFace Spaces architecture
- **Model Selection Strategy** → Priority-based fallback system handles availability/cost
  optimization
- **Output Standardization** → Leverage existing WBS framework for consistent deliverable quality

## **Go-to-Market Application for Onix**

### **Internal Operations:**

- **Proposal Development:** Generate comprehensive project scopes from client briefs
- **Solution Architecture Planning:** Structure complex cloud/AI implementations
- **Risk Assessment Automation:** Standardize due diligence across opportunities

### **Client Engagement:**

- **Discovery Workshop Enhancement:** Generate structured implementation roadmaps during client
  sessions
- **Proof-of-Concept Planning:** Rapid prototyping of project frameworks for client presentations
- **Change Management Planning:** Systematic approach to organizational transformation initiatives

## **Next Steps**

1. **Install and test** with 2-3 real project scenarios from your current pipeline
2. **Evaluate output quality** against your existing planning standards
3. **Identify highest-impact use cases** within your current sales process
4. **Pilot with one client engagement** to validate commercial viability

**Bottom Line:** This is a force multiplier for strategic planning workflows. The 5-10 minute
generation time transforms what used to be days of analytical work into rapid, consistent output
that enhances rather than replaces your strategic expertise.
