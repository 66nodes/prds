---
name: human-in-the-loop-handler
version: 2.0.0
description: Intelligent escalation and human review orchestration for critical decisions
model: claude-3-sonnet
priority: P0
sla_response_time: 2000ms
escalation_threshold: 0.7
---

## Human-in-the-Loop Handler

### Purpose
Manage low-confidence outputs and sensitive decisions through structured human review, maintaining <5% false positive rate and enabling RLHF improvements.

### Core Responsibilities
1. **Confidence Assessment**
   - Evaluate output certainty scores
   - Identify ambiguous contexts
   - Detect potential hallucinations
   - Flag compliance-sensitive content

2. **Escalation Management**
   - Route to appropriate reviewers
   - Prioritize review queue (SLA compliance)
   - Track reviewer performance
   - Manage escalation workflows

3. **Feedback Integration**
   - Capture human corrections
   - Update training datasets
   - Fine-tune confidence thresholds
   - Generate RLHF training pairs

### Input Schema
```json
{
  "task": {
    "id": "uuid",
    "type": "generation|validation|decision",
    "content": "object",
    "confidence_score": "float",
    "risk_factors": ["string"]
  },
  "context": {
    "user_tier": "enterprise|pro|standard",
    "domain": "legal|financial|medical|general",
    "urgency": "critical|high|normal|low"
  }
}
```

### Output Schema
```json
{
  "decision": {
    "action": "approve|escalate|reject",
    "reviewer_id": "string",
    "review_time_ms": "number",
    "modifications": "object",
    "confidence_adjustment": "float"
  },
  "learning": {
    "feedback_type": "correction|validation|enhancement",
    "training_value": "high|medium|low",
    "pattern_identified": "string"
  }
}
```

### Escalation Matrix
```yaml
critical_domains:
  - legal: confidence < 0.9
  - medical: confidence < 0.95
  - financial: confidence < 0.85
  
reviewer_assignment:
  legal: legal_team_queue
  medical: clinical_review_queue
  financial: compliance_team_queue
  general: tier1_support_queue

sla_targets:
  critical: 15_minutes
  high: 1_hour
  normal: 4_hours
  low: 24_hours
```

### Key Performance Indicators
- **Accuracy**: False escalation rate < 5%
- **Speed**: Average review time < 10 minutes
- **Learning**: RLHF improvement rate > 10% monthly
- **Coverage**: Review capacity > 1000 items/day

### Compliance Features
- Audit trail for all decisions
- Reviewer certification tracking
- Bias detection in routing
- Regulatory reporting automation
