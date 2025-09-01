#!/usr/bin/env python3
"""
Claude Execution Orchestrator
Ensures no aspect of the project is left untouched
"""

import yaml
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
import asyncio
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"

@dataclass
class ExecutionTask:
    name: str
    prompt: str
    validation: str
    artifacts: List[str]
    status: TaskStatus
    dependencies: List[str] = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None

class ProjectExecutor:
    """
    Orchestrates the complete project execution
    Ensures nothing is missed
    """
    
    def __init__(self, manifest_path: str = ".claude/agents/PROJECT_EXECUTION_MANIFEST.yaml"):
        self.manifest_path = Path(manifest_path)
        self.manifest = self._load_manifest()
        self.execution_log = []
        self.start_time = datetime.now()
        
    def _load_manifest(self) -> Dict:
        """Load the execution manifest"""
        with open(self.manifest_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _save_manifest(self):
        """Save updated manifest"""
        with open(self.manifest_path, 'w') as f:
            yaml.dump(self.manifest, f, default_flow_style=False)
    
    def _update_tracking(self):
        """Update tracking metrics"""
        total = 0
        completed = 0
        in_progress = 0
        blocked = 0
        
        for phase_name, phase_data in self.manifest['execution_phases'].items():
            for task in phase_data.get('checklist', []):
                total += 1
                status = task.get('status', 'pending')
                if status == 'completed':
                    completed += 1
                elif status == 'in_progress':
                    in_progress += 1
                elif status == 'blocked':
                    blocked += 1
        
        self.manifest['tracking'] = {
            'total_tasks': total,
            'completed': completed,
            'in_progress': in_progress,
            'blocked': blocked,
            'completion_percentage': round((completed / total * 100) if total > 0 else 0, 2)
        }
        
        self._save_manifest()
    
    def _check_dependencies(self, phase_name: str) -> bool:
        """Check if phase dependencies are met"""
        phase = self.manifest['execution_phases'][phase_name]
        dependencies = phase.get('dependencies', [])
        
        for dep in dependencies:
            dep_phase = self.manifest['execution_phases'].get(dep)
            if dep_phase and dep_phase.get('status') != 'completed':
                logger.warning(f"Dependency not met: {dep} for phase {phase_name}")
                return False
        
        return True
    
    def _validate_task(self, task: Dict) -> bool:
        """Validate a task using its validation command"""
        validation_cmd = task.get('validation')
        if not validation_cmd:
            return True
        
        try:
            result = subprocess.run(
                validation_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Validation passed: {validation_cmd}")
                return True
            else:
                logger.error(f"❌ Validation failed: {validation_cmd}")
                logger.error(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏱️ Validation timeout: {validation_cmd}")
            return False
        except Exception as e:
            logger.error(f"🔥 Validation error: {e}")
            return False
    
    def _check_artifacts(self, artifacts: List[str]) -> Dict[str, bool]:
        """Check if required artifacts exist"""
        artifact_status = {}
        
        for artifact in artifacts:
            path = Path(artifact)
            exists = path.exists()
            artifact_status[artifact] = exists
            
            if exists:
                logger.info(f"✅ Artifact exists: {artifact}")
            else:
                logger.warning(f"⚠️ Missing artifact: {artifact}")
        
        return artifact_status
    
    async def execute_phase(self, phase_name: str) -> bool:
        """Execute a complete phase"""
        logger.info(f"🚀 Starting phase: {phase_name}")
        
        # Check dependencies
        if not self._check_dependencies(phase_name):
            logger.error(f"❌ Dependencies not met for phase: {phase_name}")
            return False
        
        phase = self.manifest['execution_phases'][phase_name]
        phase['status'] = 'in_progress'
        self._save_manifest()
        
        # Execute each task in the phase
        all_tasks_completed = True
        for task_idx, task in enumerate(phase.get('checklist', [])):
            logger.info(f"📋 Task {task_idx + 1}: {task['task']}")
            
            # Update task status
            task['status'] = 'in_progress'
            self._save_manifest()
            
            # Show prompt for Claude
            print("\n" + "="*80)
            print(f"PROMPT FOR CLAUDE:")
            print("-"*80)
            print(task['prompt'])
            print("="*80 + "\n")
            
            # Wait for user confirmation (in production, this would be automated)
            input("Press Enter after executing the prompt...")
            
            # Validate task completion
            if self._validate_task(task):
                # Check artifacts
                artifact_status = self._check_artifacts(task.get('artifacts', []))
                
                if all(artifact_status.values()):
                    task['status'] = 'completed'
                    logger.info(f"✅ Task completed: {task['task']}")
                else:
                    task['status'] = 'blocked'
                    logger.error(f"❌ Task blocked due to missing artifacts: {task['task']}")
                    all_tasks_completed = False
            else:
                task['status'] = 'failed'
                logger.error(f"❌ Task failed validation: {task['task']}")
                all_tasks_completed = False
            
            self._save_manifest()
            self._update_tracking()
        
        # Update phase status
        if all_tasks_completed:
            phase['status'] = 'completed'
            logger.info(f"✅ Phase completed: {phase_name}")
        else:
            phase['status'] = 'blocked'
            logger.error(f"⚠️ Phase blocked: {phase_name}")
        
        self._save_manifest()
        return all_tasks_completed
    
    async def execute_all(self):
        """Execute all phases in order"""
        logger.info("🎯 Starting complete project execution")
        
        # Get phases sorted by priority
        phases = sorted(
            self.manifest['execution_phases'].items(),
            key=lambda x: x[1].get('priority', 999)
        )
        
        for phase_name, phase_data in phases:
            success = await self.execute_phase(phase_name)
            
            if not success:
                logger.error(f"🛑 Execution stopped at phase: {phase_name}")
                break
        
        # Final report
        self._generate_report()
    
    def _generate_report(self):
        """Generate execution report"""
        duration = datetime.now() - self.start_time
        tracking = self.manifest.get('tracking', {})
        
        report = f"""
========================================
    PROJECT EXECUTION REPORT
========================================
Project: {self.manifest['project']['name']}
Version: {self.manifest['project']['version']}
Duration: {duration}

COMPLETION STATUS:
- Total Tasks: {tracking.get('total_tasks', 0)}
- Completed: {tracking.get('completed', 0)}
- In Progress: {tracking.get('in_progress', 0)}
- Blocked: {tracking.get('blocked', 0)}
- Completion: {tracking.get('completion_percentage', 0)}%

PHASE STATUS:
"""
        
        for phase_name, phase_data in self.manifest['execution_phases'].items():
            status = phase_data.get('status', 'pending')
            emoji = {
                'completed': '✅',
                'in_progress': '🔄',
                'blocked': '⚠️',
                'pending': '⏳',
                'failed': '❌'
            }.get(status, '❓')
            
            report += f"  {emoji} {phase_name}: {status}\n"
        
        report += """
========================================
        """
        
        print(report)
        
        # Save report to file
        with open('execution_report.txt', 'w') as f:
            f.write(report)
        
        logger.info("📊 Report saved to execution_report.txt")
    
    def validate_completeness(self) -> bool:
        """Validate that nothing was missed"""
        logger.info("🔍 Validating project completeness...")
        
        issues = []
        
        # Check all tasks
        for phase_name, phase_data in self.manifest['execution_phases'].items():
            for task in phase_data.get('checklist', []):
                if task.get('status') != 'completed':
                    issues.append(f"Incomplete task in {phase_name}: {task['task']}")
                
                # Check artifacts
                for artifact in task.get('artifacts', []):
                    if not Path(artifact).exists():
                        issues.append(f"Missing artifact: {artifact}")
        
        # Check validation gates
        for gate in self.manifest.get('validation_gates', []):
            logger.info(f"Checking gate: {gate['name']}")
            # Add gate validation logic here
        
        if issues:
            logger.error("❌ Completeness validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        else:
            logger.info("✅ All aspects of the project have been completed!")
            return True

# CLI Interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Claude Execution Orchestrator")
    parser.add_argument('--phase', help='Execute specific phase')
    parser.add_argument('--validate', action='store_true', help='Validate completeness')
    parser.add_argument('--report', action='store_true', help='Generate report')
    parser.add_argument('--reset', action='store_true', help='Reset all task statuses')
    
    args = parser.parse_args()
    
    executor = ProjectExecutor()
    
    if args.reset:
        # Reset all statuses to pending
        for phase in executor.manifest['execution_phases'].values():
            phase['status'] = 'pending'
            for task in phase.get('checklist', []):
                task['status'] = 'pending'
        executor._save_manifest()
        executor._update_tracking()
        print("✅ All task statuses reset to pending")
        
    elif args.validate:
        # Validate completeness
        is_complete = executor.validate_completeness()
        exit(0 if is_complete else 1)
        
    elif args.report:
        # Generate report
        executor._generate_report()
        
    elif args.phase:
        # Execute specific phase
        asyncio.run(executor.execute_phase(args.phase))
        
    else:
        # Execute all phases
        asyncio.run(executor.execute_all())
