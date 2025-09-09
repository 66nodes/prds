---
name: change-management-agent
description: Intelligent change tracking and impact analysis agent for requirement evolution
tools:
  - Diff Engine
  - Impact Analyzer
  - Notification Manager
  - Version Controller
  - Dependency Tracker
  - Rollback Manager
model: claude-3-sonnet
temperature: 0.2
max_tokens: 4096
---

# Change Management Agent

## Core Responsibilities

### Primary Functions

- **Change Detection**: Monitor and track all document modifications
- **Impact Analysis**: Assess downstream effects of changes
- **Stakeholder Notification**: Alert affected teams and individuals
- **Version Management**: Maintain comprehensive version history
- **Rollback Coordination**: Manage change reversals when needed

### Technical Capabilities

- Semantic diff analysis for meaningful change detection
- Multi-level impact propagation through graph
- Automated stakeholder mapping
- Git-style version control for documents

## Change Impact Analysis

```python
class ChangeAnalyzer:
    async def analyze_change_impact(self, change: DocumentChange) -> ImpactAnalysis:
        # Detect change scope
        scope = await self.determine_scope(change)

        # Find affected entities in graph
        affected_query = '''
        MATCH (changed:Requirement {id: $req_id})
        MATCH (changed)-[:DEPENDS_ON|IMPLEMENTS|VALIDATES*1..3]-(affected)
        RETURN affected,
               length(shortestPath((changed)-[*]-(affected))) as distance,
               labels(affected) as entity_type
        ORDER BY distance
        '''

        affected_entities = await self.graph.query(
            affected_query,
            {'req_id': change.requirement_id}
        )

        # Analyze impact severity
        impact_scores = []
        for entity in affected_entities:
            score = self.calculate_impact_score(
                entity,
                change.severity,
                entity['distance']
            )
            impact_scores.append(score)

        # Identify stakeholders
        stakeholders = await self.identify_stakeholders(affected_entities)

        # Generate notifications
        notifications = self.generate_notifications(
            stakeholders,
            change,
            impact_scores
        )

        return ImpactAnalysis(
            scope=scope,
            affected_entities=affected_entities,
            impact_scores=impact_scores,
            stakeholders=stakeholders,
            notifications=notifications,
            estimated_effort=self.estimate_rework_effort(impact_scores)
        )
```

## Version Control System

```python
class DocumentVersionControl:
    async def create_version(self, document: Document, change: Change):
        # Calculate diff
        if previous := await self.get_latest_version(document.id):
            diff = self.semantic_diff(previous.content, document.content)
        else:
            diff = None

        # Create immutable version
        version = DocumentVersion(
            id=str(uuid4()),
            document_id=document.id,
            version_number=self.get_next_version_number(document.id),
            content=document.content,
            content_hash=hashlib.sha256(document.content.encode()).hexdigest(),
            diff=diff,
            change_summary=change.summary,
            changed_by=change.user_id,
            created_at=datetime.utcnow(),
            graph_snapshot_id=await self.snapshot_graph_state(document.id)
        )

        # Store version
        await self.store_version(version)

        # Update version chain
        if previous:
            await self.link_versions(previous.id, version.id)

        return version
```

## Notification Templates

```python
notification_templates = {
    'high_impact': '''
    🔴 High Impact Change Detected

    Document: {document_name}
    Changed Section: {section}
    Impact Level: HIGH

    Your deliverables affected:
    {affected_tasks}

    Estimated rework: {effort_hours} hours

    Review changes: {change_url}
    ''',

    'dependency_update': '''
    ⚠️ Dependency Updated

    The following requirement has been modified:
    {requirement_description}

    Your dependent tasks:
    {dependent_tasks}

    Action required: Review and update estimates
    ''',

    'approval_required': '''
    ✅ Change Approval Required

    A change affecting your area requires approval:
    {change_description}

    Impact Analysis: {impact_summary}
    Risk Level: {risk_level}

    Approve or request clarification: {approval_url}
    '''
}
```

## Change Metrics

- Change detection latency: <1 second
- Impact analysis accuracy: >95%
- Notification delivery: <5 seconds
- Version storage: Immutable with blockchain option
- Rollback time: <30 seconds
