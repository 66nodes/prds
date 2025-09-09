"""
Resource estimation service for project planning.
"""

import asyncio
import logging
import math
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum

from services.llm.llm_service import LLMService
from .wbs_generator import WBSStructure, TaskComplexity, TaskPriority

logger = logging.getLogger(__name__)


class ResourceType(str, Enum):
    """Types of project resources."""
    HUMAN = "human"
    INFRASTRUCTURE = "infrastructure"
    SOFTWARE = "software"
    HARDWARE = "hardware"
    EXTERNAL = "external"


class SkillLevel(str, Enum):
    """Skill levels for human resources."""
    JUNIOR = "junior"
    INTERMEDIATE = "intermediate"
    SENIOR = "senior"
    EXPERT = "expert"


class ResourceRequirement(BaseModel):
    """Individual resource requirement."""
    
    id: str = Field(..., description="Resource requirement ID")
    name: str = Field(..., description="Resource name")
    type: ResourceType = Field(..., description="Resource type")
    skill_level: Optional[SkillLevel] = Field(None, description="Required skill level for human resources")
    quantity: float = Field(..., description="Required quantity")
    unit: str = Field(..., description="Unit of measurement")
    estimated_cost: float = Field(..., description="Estimated cost per unit")
    total_cost: float = Field(..., description="Total estimated cost")
    duration_days: int = Field(..., description="Duration needed in days")
    description: str = Field(..., description="Resource description")
    alternatives: List[str] = Field(default_factory=list, description="Alternative resources")
    risk_factor: float = Field(default=1.0, description="Risk multiplier (1.0 = no risk)")
    utilization_rate: float = Field(default=0.8, description="Expected utilization rate")


class TeamComposition(BaseModel):
    """Recommended team composition."""
    
    total_team_size: int = Field(..., description="Total team size")
    roles: Dict[str, int] = Field(..., description="Role distribution")
    skill_distribution: Dict[str, int] = Field(..., description="Skill level distribution")
    estimated_monthly_cost: float = Field(..., description="Monthly team cost")
    recommended_duration_months: int = Field(..., description="Recommended project duration")
    scaling_options: List[Dict[str, Any]] = Field(default_factory=list, description="Team scaling options")


class CostEstimate(BaseModel):
    """Comprehensive cost estimation."""
    
    human_resources: float = Field(..., description="Human resource costs")
    infrastructure: float = Field(..., description="Infrastructure costs")
    software_licenses: float = Field(..., description="Software licensing costs")
    hardware: float = Field(..., description="Hardware costs")
    external_services: float = Field(..., description="External service costs")
    contingency: float = Field(..., description="Contingency buffer")
    total_cost: float = Field(..., description="Total estimated cost")
    currency: str = Field(default="USD", description="Cost currency")
    confidence_level: float = Field(..., description="Estimation confidence (0-1)")
    cost_breakdown_by_phase: Dict[str, float] = Field(default_factory=dict, description="Costs by project phase")


class TimelineEstimate(BaseModel):
    """Project timeline estimation."""
    
    total_duration_days: int = Field(..., description="Total project duration in days")
    total_duration_months: float = Field(..., description="Total project duration in months")
    parallel_execution_savings: int = Field(default=0, description="Days saved through parallel execution")
    critical_path_duration: int = Field(..., description="Critical path duration in days")
    buffer_days: int = Field(..., description="Recommended buffer days")
    phase_durations: Dict[str, int] = Field(default_factory=dict, description="Duration by phase")
    milestone_dates: List[Dict[str, Any]] = Field(default_factory=list, description="Milestone timeline")
    risk_adjusted_duration: int = Field(..., description="Risk-adjusted duration")


class ResourceEstimation(BaseModel):
    """Complete resource estimation."""
    
    project_name: str = Field(..., description="Project name")
    estimation_date: datetime = Field(default_factory=datetime.utcnow)
    team_composition: TeamComposition = Field(..., description="Recommended team structure")
    resource_requirements: List[ResourceRequirement] = Field(..., description="Detailed resource needs")
    cost_estimate: CostEstimate = Field(..., description="Cost breakdown")
    timeline_estimate: TimelineEstimate = Field(..., description="Timeline projection")
    assumptions: List[str] = Field(default_factory=list, description="Key assumptions")
    risks: List[Dict[str, Any]] = Field(default_factory=list, description="Resource-related risks")
    recommendations: List[str] = Field(default_factory=list, description="Resource recommendations")


class ResourceEstimator:
    """Service for estimating project resources and costs."""
    
    def __init__(self):
        self.llm_service = LLMService()
        
        # Standard hourly rates by skill level (USD)
        self.hourly_rates = {
            SkillLevel.JUNIOR: 50,
            SkillLevel.INTERMEDIATE: 75,
            SkillLevel.SENIOR: 120,
            SkillLevel.EXPERT: 180
        }
        
        # Standard role mappings
        self.role_skill_mapping = {
            "Project Manager": SkillLevel.SENIOR,
            "Tech Lead": SkillLevel.EXPERT,
            "Senior Developer": SkillLevel.SENIOR,
            "Developer": SkillLevel.INTERMEDIATE,
            "Junior Developer": SkillLevel.JUNIOR,
            "DevOps Engineer": SkillLevel.SENIOR,
            "QA Engineer": SkillLevel.INTERMEDIATE,
            "UX Designer": SkillLevel.SENIOR,
            "Business Analyst": SkillLevel.INTERMEDIATE,
            "Data Analyst": SkillLevel.INTERMEDIATE
        }
        
        # Infrastructure cost estimates (monthly USD)
        self.infrastructure_costs = {
            "small_project": 500,
            "medium_project": 1500,
            "large_project": 5000,
            "enterprise_project": 15000
        }

    async def estimate_resources(
        self,
        content: Dict[str, Any],
        wbs: Optional[WBSStructure] = None,
        project_context: Optional[Dict[str, Any]] = None
    ) -> ResourceEstimation:
        """Generate comprehensive resource estimation."""
        try:
            logger.info("Starting resource estimation")
            
            project_name = content.get("overview", "Project")[:50]
            
            # Estimate team composition
            team_composition = await self._estimate_team_composition(content, wbs, project_context)
            
            # Estimate detailed resource requirements
            resource_requirements = await self._estimate_detailed_resources(content, wbs, team_composition)
            
            # Calculate cost estimates
            cost_estimate = self._calculate_cost_estimates(resource_requirements, team_composition, wbs)
            
            # Calculate timeline estimates
            timeline_estimate = self._calculate_timeline_estimates(wbs, team_composition)
            
            # Generate assumptions and recommendations
            assumptions = self._generate_assumptions(content, project_context)
            risks = self._identify_resource_risks(content, wbs, team_composition)
            recommendations = self._generate_recommendations(team_composition, cost_estimate, risks)
            
            estimation = ResourceEstimation(
                project_name=project_name,
                team_composition=team_composition,
                resource_requirements=resource_requirements,
                cost_estimate=cost_estimate,
                timeline_estimate=timeline_estimate,
                assumptions=assumptions,
                risks=risks,
                recommendations=recommendations
            )
            
            logger.info(f"Resource estimation complete: {team_composition.total_team_size} team members, ${cost_estimate.total_cost:,.0f} total cost")
            return estimation
            
        except Exception as e:
            logger.error(f"Resource estimation failed: {e}")
            raise

    async def _estimate_team_composition(
        self,
        content: Dict[str, Any],
        wbs: Optional[WBSStructure],
        project_context: Optional[Dict[str, Any]]
    ) -> TeamComposition:
        """Estimate optimal team composition."""
        try:
            # Analyze project complexity
            complexity_score = self._assess_project_complexity(content, wbs)
            
            # Determine project scale
            project_scale = self._determine_project_scale(complexity_score, wbs)
            
            # Generate team recommendations using AI
            team_data = await self._generate_team_recommendations(content, complexity_score, project_scale)
            
            # Calculate costs
            monthly_cost = self._calculate_team_cost(team_data["roles"])
            duration_months = math.ceil((wbs.total_estimated_days if wbs else 90) / 22)  # ~22 working days/month
            
            team_composition = TeamComposition(
                total_team_size=sum(team_data["roles"].values()),
                roles=team_data["roles"],
                skill_distribution=team_data.get("skill_distribution", {}),
                estimated_monthly_cost=monthly_cost,
                recommended_duration_months=duration_months,
                scaling_options=team_data.get("scaling_options", [])
            )
            
            return team_composition
            
        except Exception as e:
            logger.error(f"Team composition estimation failed: {e}")
            # Return default team composition
            return self._get_default_team_composition()

    def _assess_project_complexity(
        self,
        content: Dict[str, Any],
        wbs: Optional[WBSStructure]
    ) -> float:
        """Assess overall project complexity (0-1 scale)."""
        complexity_indicators = 0
        total_indicators = 10
        
        # Requirements complexity
        requirements = content.get("requirements", [])
        if len(requirements) > 20:
            complexity_indicators += 1
        elif len(requirements) > 10:
            complexity_indicators += 0.5
        
        # Technology stack complexity
        implementation = content.get("implementation", {})
        technologies = implementation.get("technologies", [])
        if len(technologies) > 5:
            complexity_indicators += 1
        elif len(technologies) > 3:
            complexity_indicators += 0.5
        
        # Integration requirements
        if any(keyword in str(content).lower() for keyword in ["api", "integration", "microservice", "database"]):
            complexity_indicators += 1
        
        # WBS complexity
        if wbs:
            if wbs.total_estimated_hours > 1000:
                complexity_indicators += 1
            elif wbs.total_estimated_hours > 500:
                complexity_indicators += 0.5
            
            # Task complexity distribution
            complex_tasks = sum(1 for phase in wbs.phases for task in phase.tasks 
                              if task.complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT])
            total_tasks = sum(len(phase.tasks) for phase in wbs.phases)
            if total_tasks > 0 and (complex_tasks / total_tasks) > 0.3:
                complexity_indicators += 1
        
        # Domain complexity indicators
        domain_keywords = ["ai", "machine learning", "blockchain", "real-time", "high-performance", "security", "compliance"]
        if any(keyword in str(content).lower() for keyword in domain_keywords):
            complexity_indicators += 1
        
        # Scale indicators
        if any(keyword in str(content).lower() for keyword in ["enterprise", "scalable", "distributed", "cloud"]):
            complexity_indicators += 1
        
        # Performance requirements
        if any(keyword in str(content).lower() for keyword in ["performance", "optimization", "latency", "throughput"]):
            complexity_indicators += 0.5
        
        # User base size
        if any(keyword in str(content).lower() for keyword in ["million users", "high traffic", "concurrent"]):
            complexity_indicators += 1
        
        # Compliance requirements
        if any(keyword in str(content).lower() for keyword in ["gdpr", "compliance", "audit", "regulation"]):
            complexity_indicators += 0.5
        
        return min(complexity_indicators / total_indicators, 1.0)

    def _determine_project_scale(self, complexity_score: float, wbs: Optional[WBSStructure]) -> str:
        """Determine project scale category."""
        if complexity_score > 0.8 or (wbs and wbs.total_estimated_hours > 2000):
            return "enterprise_project"
        elif complexity_score > 0.6 or (wbs and wbs.total_estimated_hours > 800):
            return "large_project"
        elif complexity_score > 0.3 or (wbs and wbs.total_estimated_hours > 200):
            return "medium_project"
        else:
            return "small_project"

    async def _generate_team_recommendations(
        self,
        content: Dict[str, Any],
        complexity_score: float,
        project_scale: str
    ) -> Dict[str, Any]:
        """Generate AI-based team recommendations."""
        try:
            prompt = f"""
Recommend optimal team composition for this project:

Project Overview: {content.get('overview', 'Not specified')}
Complexity Score: {complexity_score:.2f} (0=simple, 1=very complex)
Project Scale: {project_scale}
Technologies: {content.get('implementation', {}).get('technologies', [])}
Requirements Count: {len(content.get('requirements', []))}

Provide team composition as JSON:
{{
  "roles": {{
    "Project Manager": 1,
    "Tech Lead": 1,
    "Senior Developer": 2,
    "Developer": 3,
    "DevOps Engineer": 1,
    "QA Engineer": 1
  }},
  "skill_distribution": {{
    "junior": 2,
    "intermediate": 3,
    "senior": 3,
    "expert": 1
  }},
  "scaling_options": [
    {{
      "phase": "MVP",
      "team_size": 5,
      "duration_months": 3,
      "description": "Core team for initial development"
    }}
  ]
}}

Consider:
- Project complexity and scale
- Technology requirements
- Typical industry standards
- Cost optimization
- Risk mitigation
"""
            
            response = await self.llm_service.generate_structured_content(
                prompt=prompt,
                max_tokens=1500
            )
            
            import json
            return json.loads(response)
            
        except Exception as e:
            logger.warning(f"AI team recommendation failed: {e}")
            return self._get_default_team_data(project_scale)

    def _get_default_team_data(self, project_scale: str) -> Dict[str, Any]:
        """Get default team composition based on project scale."""
        team_configs = {
            "small_project": {
                "roles": {
                    "Project Manager": 1,
                    "Tech Lead": 1,
                    "Developer": 2,
                    "QA Engineer": 1
                },
                "skill_distribution": {"intermediate": 3, "senior": 2}
            },
            "medium_project": {
                "roles": {
                    "Project Manager": 1,
                    "Tech Lead": 1,
                    "Senior Developer": 2,
                    "Developer": 3,
                    "DevOps Engineer": 1,
                    "QA Engineer": 1
                },
                "skill_distribution": {"junior": 1, "intermediate": 4, "senior": 4}
            },
            "large_project": {
                "roles": {
                    "Project Manager": 1,
                    "Tech Lead": 2,
                    "Senior Developer": 3,
                    "Developer": 5,
                    "DevOps Engineer": 2,
                    "QA Engineer": 2,
                    "UX Designer": 1
                },
                "skill_distribution": {"junior": 2, "intermediate": 6, "senior": 7, "expert": 1}
            },
            "enterprise_project": {
                "roles": {
                    "Project Manager": 2,
                    "Tech Lead": 2,
                    "Senior Developer": 5,
                    "Developer": 8,
                    "DevOps Engineer": 3,
                    "QA Engineer": 3,
                    "UX Designer": 2,
                    "Business Analyst": 1
                },
                "skill_distribution": {"junior": 3, "intermediate": 10, "senior": 10, "expert": 3}
            }
        }
        
        return team_configs.get(project_scale, team_configs["medium_project"])

    def _calculate_team_cost(self, roles: Dict[str, int]) -> float:
        """Calculate monthly team cost based on roles."""
        monthly_cost = 0
        hours_per_month = 160  # ~40 hours/week * 4 weeks
        
        for role, count in roles.items():
            skill_level = self.role_skill_mapping.get(role, SkillLevel.INTERMEDIATE)
            hourly_rate = self.hourly_rates[skill_level]
            monthly_cost += count * hourly_rate * hours_per_month
        
        return monthly_cost

    def _get_default_team_composition(self) -> TeamComposition:
        """Get default team composition for fallback."""
        return TeamComposition(
            total_team_size=6,
            roles={
                "Project Manager": 1,
                "Tech Lead": 1,
                "Senior Developer": 2,
                "Developer": 2,
                "QA Engineer": 1
            },
            skill_distribution={
                "intermediate": 3,
                "senior": 3
            },
            estimated_monthly_cost=72000,  # $450/hour * 160 hours * 6 people
            recommended_duration_months=4,
            scaling_options=[]
        )

    async def _estimate_detailed_resources(
        self,
        content: Dict[str, Any],
        wbs: Optional[WBSStructure],
        team_composition: TeamComposition
    ) -> List[ResourceRequirement]:
        """Estimate detailed resource requirements."""
        requirements = []
        
        # Human resources
        for role, count in team_composition.roles.items():
            skill_level = self.role_skill_mapping.get(role, SkillLevel.INTERMEDIATE)
            hourly_rate = self.hourly_rates[skill_level]
            
            duration_days = wbs.total_estimated_days if wbs else 90
            total_hours = count * 8 * duration_days  # 8 hours per day
            
            req = ResourceRequirement(
                id=f"human_{role.lower().replace(' ', '_')}",
                name=role,
                type=ResourceType.HUMAN,
                skill_level=skill_level,
                quantity=count,
                unit="person",
                estimated_cost=hourly_rate,
                total_cost=total_hours * hourly_rate,
                duration_days=duration_days,
                description=f"{count} {role}(s) for project duration",
                utilization_rate=0.8,
                risk_factor=1.1 if skill_level == SkillLevel.EXPERT else 1.0
            )
            requirements.append(req)
        
        # Infrastructure resources
        project_scale = self._determine_project_scale(0.5, wbs)  # Use default complexity
        infrastructure_cost = self.infrastructure_costs[project_scale]
        duration_months = math.ceil((wbs.total_estimated_days if wbs else 90) / 22)
        
        req = ResourceRequirement(
            id="infrastructure_cloud",
            name="Cloud Infrastructure",
            type=ResourceType.INFRASTRUCTURE,
            quantity=duration_months,
            unit="month",
            estimated_cost=infrastructure_cost,
            total_cost=infrastructure_cost * duration_months,
            duration_days=wbs.total_estimated_days if wbs else 90,
            description="Cloud hosting and infrastructure services",
            alternatives=["On-premise servers", "Hybrid cloud"],
            risk_factor=1.2
        )
        requirements.append(req)
        
        # Software licenses
        software_cost_per_month = team_composition.total_team_size * 100  # $100 per person per month
        req = ResourceRequirement(
            id="software_licenses",
            name="Software Licenses",
            type=ResourceType.SOFTWARE,
            quantity=duration_months,
            unit="month",
            estimated_cost=software_cost_per_month,
            total_cost=software_cost_per_month * duration_months,
            duration_days=wbs.total_estimated_days if wbs else 90,
            description="Development tools and software licenses"
        )
        requirements.append(req)
        
        return requirements

    def _calculate_cost_estimates(
        self,
        resource_requirements: List[ResourceRequirement],
        team_composition: TeamComposition,
        wbs: Optional[WBSStructure]
    ) -> CostEstimate:
        """Calculate comprehensive cost estimates."""
        human_resources = sum(req.total_cost for req in resource_requirements if req.type == ResourceType.HUMAN)
        infrastructure = sum(req.total_cost for req in resource_requirements if req.type == ResourceType.INFRASTRUCTURE)
        software_licenses = sum(req.total_cost for req in resource_requirements if req.type == ResourceType.SOFTWARE)
        hardware = sum(req.total_cost for req in resource_requirements if req.type == ResourceType.HARDWARE)
        external_services = sum(req.total_cost for req in resource_requirements if req.type == ResourceType.EXTERNAL)
        
        subtotal = human_resources + infrastructure + software_licenses + hardware + external_services
        contingency = subtotal * 0.15  # 15% contingency
        total_cost = subtotal + contingency
        
        # Cost breakdown by phase
        cost_breakdown_by_phase = {}
        if wbs:
            for phase in wbs.phases:
                phase_hours = sum(task.estimated_hours for task in phase.tasks)
                phase_percentage = phase_hours / wbs.total_estimated_hours if wbs.total_estimated_hours > 0 else 0
                cost_breakdown_by_phase[phase.name] = total_cost * phase_percentage
        
        return CostEstimate(
            human_resources=human_resources,
            infrastructure=infrastructure,
            software_licenses=software_licenses,
            hardware=hardware,
            external_services=external_services,
            contingency=contingency,
            total_cost=total_cost,
            confidence_level=0.75,  # 75% confidence
            cost_breakdown_by_phase=cost_breakdown_by_phase
        )

    def _calculate_timeline_estimates(
        self,
        wbs: Optional[WBSStructure],
        team_composition: TeamComposition
    ) -> TimelineEstimate:
        """Calculate timeline estimates."""
        if wbs:
            total_duration_days = wbs.total_estimated_days
            critical_path_duration = total_duration_days  # Simplified
        else:
            total_duration_days = 90  # Default 90 days
            critical_path_duration = 75
        
        # Calculate parallel execution savings
        parallel_savings = min(total_duration_days * 0.2, 30)  # Up to 20% savings or 30 days max
        
        # Add buffer based on project risk
        buffer_days = max(int(total_duration_days * 0.1), 5)  # At least 10% buffer or 5 days
        
        # Risk adjustment
        risk_adjusted_duration = total_duration_days + buffer_days
        
        total_months = total_duration_days / 22  # ~22 working days per month
        
        # Phase durations
        phase_durations = {}
        if wbs:
            for phase in wbs.phases:
                phase_durations[phase.name] = phase.estimated_duration_days
        
        # Milestone dates
        milestone_dates = []
        if wbs:
            cumulative_days = 0
            for milestone in wbs.milestones:
                milestone_dates.append({
                    "name": milestone["name"],
                    "day": cumulative_days,
                    "date": f"Day {cumulative_days}",
                    "description": milestone.get("description", "")
                })
        
        return TimelineEstimate(
            total_duration_days=total_duration_days,
            total_duration_months=total_months,
            parallel_execution_savings=int(parallel_savings),
            critical_path_duration=critical_path_duration,
            buffer_days=buffer_days,
            phase_durations=phase_durations,
            milestone_dates=milestone_dates,
            risk_adjusted_duration=risk_adjusted_duration
        )

    def _generate_assumptions(
        self,
        content: Dict[str, Any],
        project_context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate key estimation assumptions."""
        assumptions = [
            "Team members work 8 hours per day with 80% utilization",
            "Standard industry hourly rates applied",
            "15% contingency buffer included in cost estimates",
            "Cloud infrastructure costs based on current market rates",
            "No major scope changes during project execution",
            "Team availability as planned throughout project duration"
        ]
        
        # Add context-specific assumptions
        if project_context:
            if project_context.get("remote_team"):
                assumptions.append("Remote team collaboration tools and overhead accounted for")
            if project_context.get("tight_deadline"):
                assumptions.append("Accelerated timeline may require additional resources")
        
        return assumptions

    def _identify_resource_risks(
        self,
        content: Dict[str, Any],
        wbs: Optional[WBSStructure],
        team_composition: TeamComposition
    ) -> List[Dict[str, Any]]:
        """Identify resource-related risks."""
        risks = []
        
        # Team size risks
        if team_composition.total_team_size > 12:
            risks.append({
                "type": "team_size",
                "level": "medium",
                "description": "Large team may face coordination challenges",
                "mitigation": "Implement strong project management and communication protocols"
            })
        
        # Skill dependency risks
        expert_count = team_composition.skill_distribution.get("expert", 0)
        if expert_count < 2:
            risks.append({
                "type": "skill_dependency",
                "level": "high",
                "description": "Limited expert-level resources may create bottlenecks",
                "mitigation": "Cross-training and knowledge sharing initiatives"
            })
        
        # Complex task risks
        if wbs:
            complex_task_ratio = sum(1 for phase in wbs.phases for task in phase.tasks 
                                   if task.complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT])
            total_tasks = sum(len(phase.tasks) for phase in wbs.phases)
            
            if total_tasks > 0 and (complex_task_ratio / total_tasks) > 0.4:
                risks.append({
                    "type": "task_complexity",
                    "level": "high",
                    "description": "High percentage of complex tasks increases delivery risk",
                    "mitigation": "Break down complex tasks and add experienced resources"
                })
        
        # Timeline risks
        if wbs and wbs.total_estimated_days > 180:
            risks.append({
                "type": "timeline",
                "level": "medium",
                "description": "Extended project duration increases scope creep and team turnover risk",
                "mitigation": "Regular milestone reviews and scope management"
            })
        
        return risks

    def _generate_recommendations(
        self,
        team_composition: TeamComposition,
        cost_estimate: CostEstimate,
        risks: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate resource optimization recommendations."""
        recommendations = []
        
        # Team composition recommendations
        if team_composition.total_team_size > 10:
            recommendations.append("Consider splitting into smaller sub-teams with clear ownership areas")
        
        # Cost optimization
        if cost_estimate.total_cost > 500000:
            recommendations.append("Explore cost optimization through offshore resources or phased delivery")
        
        # Risk-based recommendations
        high_risk_count = sum(1 for risk in risks if risk["level"] == "high")
        if high_risk_count > 1:
            recommendations.append("Implement risk mitigation strategies before project start")
        
        # Generic best practices
        recommendations.extend([
            "Regular sprint planning and retrospectives for agile delivery",
            "Implement continuous integration and deployment practices",
            "Establish clear communication channels and documentation standards",
            "Plan for knowledge transfer and team onboarding",
            "Monitor progress against estimates and adjust as needed"
        ])
        
        return recommendations