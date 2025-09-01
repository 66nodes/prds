#!/usr/bin/env python3
"""
Enhanced Claude Orchestrator with Agent Assignment
Manages task distribution and agent coordination
"""

import asyncio
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
import redis
import json

logger = logging.getLogger(__name__)

@dataclass
class AgentTask:
    """Enhanced task with agent assignment"""
    task_id: str
    task_name: str
    agent: str
    fallback_agent: str
    prompt: str
    validation: str
    estimated_hours: float
    retry_policy: Dict[str, Any]
    artifacts: List[str]
    status: str = "pending"
    attempts: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None

class AgentOrchestrator:
    """
    Orchestrates agent task execution with fallback and retry logic
    """
    
    def __init__(self, manifest_path: str):
        self.manifest = self._load_manifest(manifest_path)
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.agent_tasks = {}
        self.agent_workload = {}
        self._initialize_agents()
    
    def _load_manifest(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_agents(self):
        """Initialize agent workload tracking"""
        for agent in self.manifest['orchestration']['agent_pool']:
            self.agent_workload[agent] = {
                'current_tasks': [],
                'completed_tasks': 0,
                'failed_tasks': 0,
                'total_hours': 0
            }
    
    async def assign_task(self, task: AgentTask) -> str:
        """
        Assign task to appropriate agent
        """
        assigned_agent = task.agent
        
        # Check if primary agent is available
        if self._is_agent_available(assigned_agent):
            logger.info(f"Assigning task {task.task_id} to {assigned_agent}")
        else:
            # Try fallback agent
            if self._is_agent_available(task.fallback_agent):
                assigned_agent = task.fallback_agent
                logger.info(f"Primary agent busy, assigning {task.task_id} to fallback: {assigned_agent}")
            else:
                # Queue task for later
                await self._queue_task(task)
                logger.warning(f"Both agents busy, queuing task {task.task_id}")
                return "queued"
        
        # Update agent workload
        self.agent_workload[assigned_agent]['current_tasks'].append(task.task_id)
        task.start_time = datetime.now()
        
        # Publish task assignment
        await self._publish_task_assignment(assigned_agent, task)
        
        return assigned_agent
    
    def _is_agent_available(self, agent: str) -> bool:
        """Check if agent can take new tasks"""
        current_load = len(self.agent_workload[agent]['current_tasks'])
        # Each agent can handle up to 3 concurrent tasks
        return current_load < 3
    
    async def _queue_task(self, task: AgentTask):
        """Queue task for later execution"""
        task_json = json.dumps({
            'task_id': task.task_id,
            'task_name': task.task_name,
            'agent': task.agent,
            'priority': 1  # Can be enhanced with priority logic
        })
        self.redis_client.lpush('task_queue', task_json)
    
    async def _publish_task_assignment(self, agent: str, task: AgentTask):
        """Publish task to agent via Redis pub/sub"""
        message = {
            'task_id': task.task_id,
            'agent': agent,
            'prompt': task.prompt,
            'validation': task.validation,
            'artifacts': task.artifacts,
            'timestamp': datetime.now().isoformat()
        }
        
        channel = f"agent:{agent}:tasks"
        self.redis_client.publish(channel, json.dumps(message))
        
        # Store task details
        self.redis_client.hset(
            f"task:{task.task_id}",
            mapping={
                'agent': agent,
                'status': 'assigned',
                'start_time': task.start_time.isoformat()
            }
        )
    
    async def handle_task_completion(self, task_id: str, success: bool, result: Dict):
        """Handle task completion or failure"""
        task = self.agent_tasks.get(task_id)
        if not task:
            logger.error(f"Unknown task: {task_id}")
            return
        
        task.end_time = datetime.now()
        assigned_agent = self.redis_client.hget(f"task:{task_id}", "agent")
        
        # Update agent workload
        if assigned_agent in self.agent_workload:
            workload = self.agent_workload[assigned_agent]
            if task_id in workload['current_tasks']:
                workload['current_tasks'].remove(task_id)
            
            if success:
                workload['completed_tasks'] += 1
                task.status = 'completed'
                logger.info(f"Task {task_id} completed by {assigned_agent}")
            else:
                workload['failed_tasks'] += 1
                task.status = 'failed'
                task.error_message = result.get('error')
                
                # Handle retry logic
                if task.attempts < task.retry_policy['max_retries']:
                    task.attempts += 1
                    delay = task.retry_policy['delay_seconds']
                    logger.info(f"Retrying task {task_id} after {delay} seconds (attempt {task.attempts})")
                    await asyncio.sleep(delay)
                    await self.assign_task(task)
                else:
                    logger.error(f"Task {task_id} failed after {task.attempts} attempts")
                    # Escalate to human
                    await self._escalate_to_human(task)
            
            # Calculate hours spent
            duration = (task.end_time - task.start_time).total_seconds() / 3600
            workload['total_hours'] += duration
        
        # Check for queued tasks
        await self._process_queue()
    
    async def _process_queue(self):
        """Process queued tasks when agents become available"""
        while True:
            # Check if any agent is available
            available_agents = [
                agent for agent, workload in self.agent_workload.items()
                if len(workload['current_tasks']) < 3
            ]
            
            if not available_agents:
                break
            
            # Get next task from queue
            task_json = self.redis_client.rpop('task_queue')
            if not task_json:
                break
            
            task_data = json.loads(task_json)
            # Reconstruct and assign task
            # (Implementation depends on task storage strategy)
    
    async def _escalate_to_human(self, task: AgentTask):
        """Escalate failed task to human operator"""
        escalation = {
            'task_id': task.task_id,
            'task_name': task.task_name,
            'assigned_agents': [task.agent, task.fallback_agent],
            'attempts': task.attempts,
            'error': task.error_message,
            'prompt': task.prompt,
            'timestamp': datetime.now().isoformat()
        }
        
        # Send notification (webhook, email, etc.)
        self.redis_client.lpush('human_escalations', json.dumps(escalation))
        logger.critical(f"Task {task.task_id} escalated to human review")
    
    def generate_agent_report(self) -> str:
        """Generate agent performance report"""
        report = "AGENT PERFORMANCE REPORT\n"
        report += "=" * 50 + "\n\n"
        
        for agent, workload in self.agent_workload.items():
            efficiency = 0
            if workload['completed_tasks'] > 0:
                total_tasks = workload['completed_tasks'] + workload['failed_tasks']
                efficiency = (workload['completed_tasks'] / total_tasks) * 100
            
            report += f"{agent}:\n"
            report += f"  Current Tasks: {len(workload['current_tasks'])}\n"
            report += f"  Completed: {workload['completed_tasks']}\n"
            report += f"  Failed: {workload['failed_tasks']}\n"
            report += f"  Total Hours: {workload['total_hours']:.2f}\n"
            report += f"  Efficiency: {efficiency:.1f}%\n\n"
        
        return report
    
    async def execute_phase(self, phase_name: str):
        """Execute all tasks in a phase with agent orchestration"""
        phase = self.manifest['execution_phases'][phase_name]
        tasks = []
        
        for task_data in phase['checklist']:
            task = AgentTask(
                task_id=task_data['task_id'],
                task_name=task_data['task'],
                agent=task_data['agent'],
                fallback_agent=task_data['fallback_agent'],
                prompt=task_data['prompt'],
                validation=task_data['validation'],
                estimated_hours=task_data['estimated_hours'],
                retry_policy=task_data['retry_policy'],
                artifacts=task_data['artifacts']
            )
            
            self.agent_tasks[task.task_id] = task
            tasks.append(task)
        
        # Assign tasks to agents
        assignments = []
        for task in tasks:
            agent = await self.assign_task(task)
            assignments.append((task.task_id, agent))
        
        logger.info(f"Phase {phase_name} tasks assigned to agents")
        
        # Monitor phase completion
        await self._monitor_phase_completion(phase_name, [t.task_id for t in tasks])
    
    async def _monitor_phase_completion(self, phase_name: str, task_ids: List[str]):
        """Monitor phase completion status"""
        while True:
            completed = 0
            failed = 0
            in_progress = 0
            
            for task_id in task_ids:
                task = self.agent_tasks[task_id]
                if task.status == 'completed':
                    completed += 1
                elif task.status == 'failed':
                    failed += 1
                else:
                    in_progress += 1
            
            total = len(task_ids)
            completion_rate = (completed / total) * 100 if total > 0 else 0
            
            logger.info(f"Phase {phase_name}: {completed}/{total} completed ({completion_rate:.1f}%)")
            
            if completed + failed == total:
                if failed == 0:
                    logger.info(f"✅ Phase {phase_name} completed successfully")
                else:
                    logger.warning(f"⚠️ Phase {phase_name} completed with {failed} failures")
                break
            
            await asyncio.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    orchestrator = AgentOrchestrator("PROJECT_EXECUTION_MANIFEST_v2.yaml")
    
    # Example: Execute infrastructure phase
    asyncio.run(orchestrator.execute_phase("infrastructure"))
    
    # Generate report
    print(orchestrator.generate_agent_report())
