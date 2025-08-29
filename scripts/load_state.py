#!/usr/bin/env python3
import os
import re
import json
from pathlib import Path

def load_workflow_state():
    """Load current state from workflow_state.md"""
    state_file = Path('workflow_state.md')
    state = {
        'current_task': None,
        'current_subtask': None,
        'completed_tasks': [],
        'completed_subtasks': [],
        'blocked_tasks': []
    }
    
    if state_file.exists():
        content = state_file.read_text()
        
        # Extract current task
        task_match = re.search(r'Current Task:\s*(\d+\.\d+)', content)
        if task_match:
            state['current_task'] = task_match.group(1)
            
        # Extract current subtask
        subtask_match = re.search(r'Current Subtask:\s*(\d+)/(\d+)', content)
        if subtask_match:
            state['current_subtask'] = f"{subtask_match.group(1)}/{subtask_match.group(2)}"
            
        # Extract completed tasks from table
        table_pattern = r'\|\s*(\d+\.\d+)\s*\|.*?\|\s*COMPLETED\s*\|'
        state['completed_tasks'] = re.findall(table_pattern, content)
    
    # Save state for other steps
    os.makedirs('.task_state', exist_ok=True)
    
    with open('.task_state/current_task.txt', 'w') as f:
        f.write(state['current_task'] or '20.1')
        
    with open('.task_state/current_subtask.txt', 'w') as f:
        f.write(state['current_subtask'] or '1/1')
        
    with open('.task_state/state.json', 'w') as f:
        json.dump(state, f, indent=2)
    
    return state

if __name__ == '__main__':
    load_workflow_state()
