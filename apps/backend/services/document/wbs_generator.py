"""
Work Breakdown Structure (WBS) generation service.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum

from services.llm.llm_service import LLMService

logger = logging.getLogger(__name__)


class TaskComplexity(str, Enum):
    """Task complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class WBSTask(BaseModel):
    """Individual task in Work Breakdown Structure."""
    
    id: str = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Task name")
    description: str = Field(..., description="Detailed task description")
    complexity: TaskComplexity = Field(default=TaskComplexity.MODERATE, description="Task complexity")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Task priority")
    estimated_hours: float = Field(..., description="Estimated hours to complete")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")
    skills_required: List[str] = Field(default_factory=list, description="Required skills")
    deliverables: List[str] = Field(default_factory=list, description="Task deliverables")
    phase: str = Field(..., description="Project phase")
    milestone: Optional[str] = Field(None, description="Associated milestone")
    risk_level: str = Field(default="low", description="Risk assessment")


class WBSPhase(BaseModel):
    """Project phase in Work Breakdown Structure."""
    
    id: str = Field(..., description="Phase identifier")
    name: str = Field(..., description="Phase name")
    description: str = Field(..., description="Phase description")
    order: int = Field(..., description="Phase order")
    estimated_duration_days: int = Field(..., description="Estimated duration in days")
    tasks: List[WBSTask] = Field(default_factory=list, description="Tasks in this phase")
    dependencies: List[str] = Field(default_factory=list, description="Phase dependencies")
    success_criteria: List[str] = Field(default_factory=list, description="Success criteria")


class WBSStructure(BaseModel):
    """Complete Work Breakdown Structure."""
    
    project_name: str = Field(..., description="Project name")
    total_estimated_hours: float = Field(..., description="Total estimated hours")
    total_estimated_days: int = Field(..., description="Total estimated days")
    phases: List[WBSPhase] = Field(..., description="Project phases")
    critical_path: List[str] = Field(default_factory=list, description="Critical path task IDs")
    milestones: List[Dict[str, Any]] = Field(default_factory=list, description="Project milestones")
    resource_summary: Dict[str, Any] = Field(default_factory=dict, description="Resource requirements summary")
    risk_assessment: Dict[str, Any] = Field(default_factory=dict, description="Overall risk assessment")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class WBSGenerator:
    """Service for generating Work Breakdown Structures."""
    
    def __init__(self):
        self.llm_service = LLMService()
        
        # Default phases for different project types
        self.default_phases = {
            "software": [
                {"name": "Planning & Analysis", "order": 1},
                {"name": "Design & Architecture", "order": 2},
                {"name": "Development", "order": 3},
                {"name": "Testing & QA", "order": 4},
                {"name": "Deployment & Launch", "order": 5}
            ],
            "prd": [
                {"name": "Research & Discovery", "order": 1},
                {"name": "Requirements Definition", "order": 2},
                {"name": "Design & Specification", "order": 3},
                {"name": "Implementation Planning", "order": 4},
                {"name": "Validation & Review", "order": 5}
            ],
            "default": [
                {"name": "Initiation", "order": 1},
                {"name": "Planning", "order": 2},
                {"name": "Execution", "order": 3},
                {"name": "Monitoring", "order": 4},
                {"name": "Closure", "order": 5}
            ]
        }
        
        # Complexity multipliers for estimation
        self.complexity_multipliers = {
            TaskComplexity.SIMPLE: 0.8,
            TaskComplexity.MODERATE: 1.0,
            TaskComplexity.COMPLEX: 1.5,
            TaskComplexity.EXPERT: 2.0
        }

    async def generate_wbs(
        self,
        title: str,
        requirements: List[str],
        context: Optional[str] = None,
        project_type: str = "software"
    ) -> WBSStructure:
        """Generate a complete Work Breakdown Structure."""
        try:
            logger.info(f"Generating WBS for project: {title}")
            
            # Generate project phases
            phases = await self._generate_phases(title, requirements, context, project_type)
            
            # Generate tasks for each phase
            for phase in phases:
                phase.tasks = await self._generate_phase_tasks(phase, requirements, context)
            
            # Calculate totals and dependencies
            total_hours, total_days = self._calculate_totals(phases)
            critical_path = self._identify_critical_path(phases)
            milestones = self._generate_milestones(phases)
            resource_summary = self._generate_resource_summary(phases)
            risk_assessment = self._assess_risks(phases)
            
            wbs = WBSStructure(
                project_name=title,
                total_estimated_hours=total_hours,
                total_estimated_days=total_days,
                phases=phases,
                critical_path=critical_path,
                milestones=milestones,
                resource_summary=resource_summary,
                risk_assessment=risk_assessment
            )
            
            logger.info(f"WBS generated: {len(phases)} phases, {sum(len(p.tasks) for p in phases)} tasks, {total_hours} hours")
            return wbs
            
        except Exception as e:
            logger.error(f"WBS generation failed: {e}")
            raise

    async def _generate_phases(
        self,
        title: str,
        requirements: List[str],
        context: Optional[str],
        project_type: str
    ) -> List[WBSPhase]:
        """Generate project phases based on requirements."""
        try:
            # Use default phases as starting point
            default_phases = self.default_phases.get(project_type, self.default_phases["default"])
            
            # Generate AI-enhanced phase descriptions
            prompt = self._build_phase_prompt(title, requirements, context, default_phases)
            
            response = await self.llm_service.generate_structured_content(
                prompt=prompt,
                context=context,
                max_tokens=2000
            )
            
            # Parse and structure phases
            phases = await self._parse_phases_response(response, default_phases)
            
            return phases
            
        except Exception as e:
            logger.error(f"Phase generation failed: {e}")
            # Fallback to default phases
            return self._create_default_phases(project_type)

    def _build_phase_prompt(
        self,
        title: str,
        requirements: List[str],
        context: Optional[str],
        default_phases: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for phase generation."""
        requirements_text = "\n".join([f"- {req}" for req in requirements[:10]])
        phases_text = "\n".join([f"{p['order']}. {p['name']}" for p in default_phases])
        
        prompt = f"""
Generate detailed project phases for: "{title}"

Requirements:
{requirements_text}

Context: {context or 'No additional context'}

Default phases structure:
{phases_text}

For each phase, provide:
1. Enhanced name and description
2. Estimated duration in days
3. Key success criteria
4. Dependencies on other phases

Format as JSON array:
[
  {{
    "id": "phase_1",
    "name": "Enhanced Phase Name",
    "description": "Detailed phase description",
    "order": 1,
    "estimated_duration_days": 10,
    "success_criteria": ["Criteria 1", "Criteria 2"],
    "dependencies": []
  }}
]

Focus on creating realistic, actionable phases that align with the project requirements.
"""
        return prompt

    async def _parse_phases_response(
        self,
        response: str,
        default_phases: List[Dict[str, Any]]
    ) -> List[WBSPhase]:
        """Parse AI response into WBSPhase objects."""
        try:
            # Try to parse JSON response
            phases_data = json.loads(response)
            if isinstance(phases_data, list):
                phases = []
                for i, phase_data in enumerate(phases_data):
                    phase = WBSPhase(
                        id=phase_data.get("id", f"phase_{i+1}"),
                        name=phase_data.get("name", default_phases[i]["name"]),
                        description=phase_data.get("description", f"Phase {i+1} activities"),
                        order=phase_data.get("order", i + 1),
                        estimated_duration_days=phase_data.get("estimated_duration_days", 14),
                        dependencies=phase_data.get("dependencies", []),
                        success_criteria=phase_data.get("success_criteria", [])
                    )
                    phases.append(phase)
                return phases
        except (json.JSONDecodeError, KeyError, IndexError):
            logger.warning("Failed to parse phases response, using defaults")
        
        # Fallback to default phases
        return self._create_default_phases("default")

    def _create_default_phases(self, project_type: str) -> List[WBSPhase]:
        """Create default phases for project type."""
        default_phases = self.default_phases.get(project_type, self.default_phases["default"])
        phases = []
        
        for phase_data in default_phases:
            phase = WBSPhase(
                id=f"phase_{phase_data['order']}",
                name=phase_data["name"],
                description=f"{phase_data['name']} phase activities",
                order=phase_data["order"],
                estimated_duration_days=14,
                success_criteria=[f"Complete {phase_data['name'].lower()} deliverables"]
            )
            phases.append(phase)
        
        return phases

    async def _generate_phase_tasks(
        self,
        phase: WBSPhase,
        requirements: List[str],
        context: Optional[str]
    ) -> List[WBSTask]:
        """Generate tasks for a specific phase."""
        try:
            prompt = self._build_task_prompt(phase, requirements, context)
            
            response = await self.llm_service.generate_structured_content(
                prompt=prompt,
                context=context,
                max_tokens=3000
            )
            
            tasks = await self._parse_tasks_response(response, phase)
            return tasks
            
        except Exception as e:
            logger.error(f"Task generation failed for phase {phase.name}: {e}")
            # Return default tasks for phase
            return self._create_default_tasks(phase)

    def _build_task_prompt(
        self,
        phase: WBSPhase,
        requirements: List[str],
        context: Optional[str]
    ) -> str:
        """Build prompt for task generation."""
        requirements_text = "\n".join([f"- {req}" for req in requirements[:10]])
        
        prompt = f"""
Generate detailed tasks for phase: "{phase.name}"

Phase Description: {phase.description}
Phase Duration: {phase.estimated_duration_days} days

Project Requirements:
{requirements_text}

Context: {context or 'No additional context'}

For each task, provide:
1. Clear name and description
2. Complexity level (simple/moderate/complex/expert)
3. Priority (low/medium/high/critical)
4. Estimated hours
5. Required skills
6. Deliverables
7. Risk level assessment

Format as JSON array:
[
  {{
    "id": "task_1",
    "name": "Task Name",
    "description": "Detailed task description",
    "complexity": "moderate",
    "priority": "high",
    "estimated_hours": 16,
    "skills_required": ["Skill 1", "Skill 2"],
    "deliverables": ["Deliverable 1"],
    "risk_level": "medium"
  }}
]

Generate 3-6 realistic, actionable tasks that accomplish the phase objectives.
"""
        return prompt

    async def _parse_tasks_response(
        self,
        response: str,
        phase: WBSPhase
    ) -> List[WBSTask]:
        """Parse AI response into WBSTask objects."""
        try:
            tasks_data = json.loads(response)
            if isinstance(tasks_data, list):
                tasks = []
                for i, task_data in enumerate(tasks_data):
                    task = WBSTask(
                        id=task_data.get("id", f"{phase.id}_task_{i+1}"),
                        name=task_data.get("name", f"Task {i+1}"),
                        description=task_data.get("description", "Task description"),
                        complexity=TaskComplexity(task_data.get("complexity", "moderate")),
                        priority=TaskPriority(task_data.get("priority", "medium")),
                        estimated_hours=float(task_data.get("estimated_hours", 16)),
                        skills_required=task_data.get("skills_required", []),
                        deliverables=task_data.get("deliverables", []),
                        phase=phase.id,
                        risk_level=task_data.get("risk_level", "medium")
                    )
                    tasks.append(task)
                return tasks
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse tasks response: {e}")
        
        # Fallback to default tasks
        return self._create_default_tasks(phase)

    def _create_default_tasks(self, phase: WBSPhase) -> List[WBSTask]:
        """Create default tasks for a phase."""
        tasks = [
            WBSTask(
                id=f"{phase.id}_task_1",
                name=f"{phase.name} Planning",
                description=f"Plan and organize {phase.name.lower()} activities",
                complexity=TaskComplexity.MODERATE,
                priority=TaskPriority.HIGH,
                estimated_hours=8,
                phase=phase.id,
                skills_required=["Planning", "Project Management"],
                deliverables=[f"{phase.name} plan document"]
            ),
            WBSTask(
                id=f"{phase.id}_task_2",
                name=f"{phase.name} Execution",
                description=f"Execute main {phase.name.lower()} activities",
                complexity=TaskComplexity.COMPLEX,
                priority=TaskPriority.HIGH,
                estimated_hours=32,
                phase=phase.id,
                skills_required=["Technical Skills", "Domain Expertise"],
                deliverables=[f"{phase.name} deliverables"]
            ),
            WBSTask(
                id=f"{phase.id}_task_3",
                name=f"{phase.name} Review",
                description=f"Review and validate {phase.name.lower()} outputs",
                complexity=TaskComplexity.MODERATE,
                priority=TaskPriority.MEDIUM,
                estimated_hours=8,
                phase=phase.id,
                skills_required=["Quality Assurance", "Review"],
                deliverables=[f"{phase.name} review report"]
            )
        ]
        return tasks

    def _calculate_totals(self, phases: List[WBSPhase]) -> Tuple[float, int]:
        """Calculate total hours and days for project."""
        total_hours = 0
        for phase in phases:
            for task in phase.tasks:
                # Apply complexity multiplier
                multiplier = self.complexity_multipliers[task.complexity]
                total_hours += task.estimated_hours * multiplier
        
        # Assume 8 hours per work day
        total_days = int(total_hours / 8) + (1 if total_hours % 8 > 0 else 0)
        
        return total_hours, total_days

    def _identify_critical_path(self, phases: List[WBSPhase]) -> List[str]:
        """Identify critical path tasks (simplified algorithm)."""
        critical_tasks = []
        
        for phase in phases:
            # Find the longest task in each phase (simplified critical path)
            if phase.tasks:
                longest_task = max(phase.tasks, key=lambda t: t.estimated_hours * self.complexity_multipliers[t.complexity])
                critical_tasks.append(longest_task.id)
        
        return critical_tasks

    def _generate_milestones(self, phases: List[WBSPhase]) -> List[Dict[str, Any]]:
        """Generate project milestones based on phases."""
        milestones = []
        cumulative_days = 0
        
        for phase in phases:
            cumulative_days += phase.estimated_duration_days
            
            milestone = {
                "id": f"milestone_{phase.order}",
                "name": f"{phase.name} Complete",
                "description": f"Completion of {phase.name.lower()} phase",
                "target_date": f"Day {cumulative_days}",
                "phase_id": phase.id,
                "deliverables": [task.deliverables for task in phase.tasks if task.deliverables]
            }
            milestones.append(milestone)
        
        return milestones

    def _generate_resource_summary(self, phases: List[WBSPhase]) -> Dict[str, Any]:
        """Generate resource requirements summary."""
        skills_needed = set()
        total_tasks = 0
        complexity_distribution = {c.value: 0 for c in TaskComplexity}
        
        for phase in phases:
            for task in phase.tasks:
                total_tasks += 1
                skills_needed.update(task.skills_required)
                complexity_distribution[task.complexity.value] += 1
        
        return {
            "total_tasks": total_tasks,
            "unique_skills_required": list(skills_needed),
            "complexity_distribution": complexity_distribution,
            "recommended_team_size": min(max(total_tasks // 10, 2), 8),  # 2-8 team members
            "estimated_full_time_resources": max(1, total_tasks // 20)  # Rough estimate
        }

    def _assess_risks(self, phases: List[WBSPhase]) -> Dict[str, Any]:
        """Assess project risks based on WBS."""
        total_tasks = sum(len(phase.tasks) for phase in phases)
        high_risk_tasks = 0
        complex_tasks = 0
        
        for phase in phases:
            for task in phase.tasks:
                if task.risk_level in ["high", "critical"]:
                    high_risk_tasks += 1
                if task.complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT]:
                    complex_tasks += 1
        
        risk_score = (high_risk_tasks + complex_tasks) / total_tasks if total_tasks > 0 else 0
        
        risk_level = "low"
        if risk_score > 0.6:
            risk_level = "high"
        elif risk_score > 0.3:
            risk_level = "medium"
        
        return {
            "overall_risk_level": risk_level,
            "risk_score": round(risk_score, 2),
            "high_risk_tasks": high_risk_tasks,
            "complex_tasks": complex_tasks,
            "total_tasks": total_tasks,
            "risk_mitigation_needed": risk_score > 0.4,
            "recommended_actions": self._get_risk_recommendations(risk_level, risk_score)
        }

    def _get_risk_recommendations(self, risk_level: str, risk_score: float) -> List[str]:
        """Get risk mitigation recommendations."""
        recommendations = []
        
        if risk_level == "high":
            recommendations.extend([
                "Conduct detailed risk assessment for complex tasks",
                "Consider adding buffer time to schedule",
                "Implement regular progress reviews",
                "Assign experienced team members to high-risk tasks"
            ])
        elif risk_level == "medium":
            recommendations.extend([
                "Monitor high-risk tasks closely",
                "Consider prototype development for complex features",
                "Regular team check-ins and progress reviews"
            ])
        else:
            recommendations.append("Continue with planned approach, monitor for emerging risks")
        
        if risk_score > 0.5:
            recommendations.append("Consider breaking down complex tasks into smaller components")
        
        return recommendations