#!/usr/bin/env python3
import re
import json
from datetime import datetime
from pathlib import Path

def update_workflow_state():
    """Update workflow_state.md with current progress"""
    state_file = Path('workflow_state.md')
    
    # Load current state
    with open('.task_state/state.json', 'r') as f:
        state = json.load(f)
    
    # Read existing content
    content = state_file.read_text()
    
    # Update timestamp
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    content = re.sub(
        r'Last Updated: \[.*?\]',
        f'Last Updated: [{timestamp}]',
        content
    )
    
    # Update current task
    content = re.sub(
        r'Current Task: \[.*?\]',
        f"Current Task: [{state.get('current_task', '')}]",
        content
    )
    
    # Update current subtask
    content = re.sub(
        r'Current Subtask: \[.*?\]',
        f"Current Subtask: [{state.get('current_subtask', '')}]",
        content
    )
    
    # Add to activity log
    log_entry = f"\n- {timestamp}: Task {state['current_task']} - {state.get('last_action', 'Updated')}"
    
    # Find Activity Log section and append
    log_pattern = r'(## Activity Log\n)(.*?)(\n##|\Z)'
    match = re.search(log_pattern, content, re.DOTALL)
    if match:
        current_log = match.group(2)
        new_log = current_log + log_entry
        content = content[:match.start(2)] + new_log + content[match.end(2):]
    
    # Write updated content
    state_file.write_text(content)
    
    print(f"Updated workflow_state.md at {timestamp}")

if __name__ == '__main__':
    update_workflow_state()
