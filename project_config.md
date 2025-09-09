### 4. Updated project_config.md

Add this section to your project_config.md:

```markdown
## Task Management System

- **Tool**: claude-task-master
- **Repository**: https://github.com/eyaltoledano/claude-task-master
- **Automation**: GitHub Actions
- **Task Location**: ./.taskmaster/tasks/
- **State Tracking**: workflow_state.md
- **Update Method**: Automated via CI/CD

### Task Execution Rules

1. Tasks must be completed in dependency order
2. All subtasks must complete before task closure
3. State updates trigger automatic commits
4. Validation runs every 6 hours
5. Manual override available via workflow_dispatch

### Task ID Convention

- Format: XX.YY where XX is major task, YY is subtask
- Example: 20.1 = Task 20, Subtask 1
- Current Sprint: 20.x (Strapi Auth Implementation)
```
