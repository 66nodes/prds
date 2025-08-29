#!/usr/bin/env python3
import sys
import json
import subprocess
from pathlib import Path

def validate_dependencies(task_id):
    """Check if all dependencies for a task are met"""
    # Get task info from task-master
    result = subprocess.run(
        ['task-master', 'show', '--id', task_id],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error loading task {task_id}")
        sys.exit(1)
    
    # Parse task info
    task_info = json.loads(result.stdout)
    dependencies = task_info.get('dependencies', [])
    
    # Load completed tasks
    with open('.task_state/state.json', 'r') as f:
        state = json.load(f)
    
    unmet_deps = []
    for dep in dependencies:
        if dep not in state['completed_tasks']:
            unmet_deps.append(dep)
    
    if unmet_deps:
        print(f"Unmet dependencies: {', '.join(unmet_deps)}")
        # Mark task as blocked
        subprocess.run([
            'task-master', 'set-status',
            '--id', task_id,
            '--status', 'blocked'
        ])
        sys.exit(1)
    
    print(f"All dependencies met for task {task_id}")
    return True

if __name__ == '__main__':
    if len(sys.argv) > 1:
        validate_dependencies(sys.argv[1])
