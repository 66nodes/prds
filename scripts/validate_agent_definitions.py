#!/usr/bin/env python3
"""
Agent Definition Validation Script.

Comprehensive validation system for 100+ agent definitions with syntax checking,
capability validation, integration testing, and performance benchmarking.
"""

import asyncio
import json
import os
import sys
import re
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import structlog

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.services.agent_orchestrator import AgentType
from backend.services.enhanced_context_manager import EnhancedContextManager
from backend.services.prompt_engineering_system import PromptEngineeringSystem
from backend.services.agent_state_manager import AgentState
from backend.services.agent_communication_protocols import MessageType

logger = structlog.get_logger(__name__)


class ValidationSeverity(str, Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(str, Enum):
    """Categories of validation checks."""
    SYNTAX = "syntax"
    STRUCTURE = "structure"
    CONTENT = "content"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    COMPLIANCE = "compliance"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    category: ValidationCategory
    severity: ValidationSeverity
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    agent_id: Optional[str] = None
    suggestion: Optional[str] = None
    auto_fixable: bool = False
    

@dataclass
class ValidationResult:
    """Result of validation process."""
    agent_id: str
    file_path: str
    passed: bool = True
    issues: List[ValidationIssue] = None
    performance_score: float = 0.0
    compliance_score: float = 0.0
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
    
    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)
        if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.passed = False
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues filtered by severity."""
        return [issue for issue in self.issues if issue.severity == severity]


class AgentDefinitionValidator:
    """Comprehensive validator for agent definitions."""
    
    def __init__(self):
        self.agents_dir = project_root / ".claude" / "agents"
        self.validation_results: List[ValidationResult] = []
        self.required_fields = {
            "name", "description", "model", "temperature", "max_tokens"
        }
        self.optional_fields = {
            "tools", "color", "dependencies", "capabilities", "examples"
        }
        self.valid_models = {
            "haiku", "sonnet", "opus", "claude-3-haiku", "claude-3-sonnet", "claude-3-opus"
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            "max_tokens": {"haiku": 4000, "sonnet": 8000, "opus": 12000},
            "temperature": {"min": 0.0, "max": 1.0},
            "response_time_ms": 30000,  # 30 seconds max
        }
        
        # Compliance requirements
        self.compliance_requirements = {
            "security_review": False,  # Security review required
            "accessibility": True,     # Must be accessible
            "documentation": True,     # Must have documentation
            "testing": False,         # Testing not yet required
        }
    
    async def validate_all_agents(self, fix_issues: bool = False) -> Dict[str, Any]:
        """Validate all agent definitions."""
        logger.info("Starting comprehensive agent validation")
        
        if not self.agents_dir.exists():
            logger.error(f"Agents directory not found: {self.agents_dir}")
            return {"error": "Agents directory not found"}
        
        # Find all agent definition files
        agent_files = list(self.agents_dir.glob("*.md"))
        logger.info(f"Found {len(agent_files)} agent definition files")
        
        # Validate each agent
        for agent_file in agent_files:
            try:
                result = await self.validate_agent_file(agent_file, fix_issues)
                self.validation_results.append(result)
            except Exception as e:
                logger.error(f"Failed to validate {agent_file}: {str(e)}")
                result = ValidationResult(
                    agent_id=agent_file.stem,
                    file_path=str(agent_file),
                    passed=False
                )
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.SYNTAX,
                    severity=ValidationSeverity.CRITICAL,
                    title="Validation Failed",
                    description=f"Exception during validation: {str(e)}",
                    file_path=str(agent_file)
                ))
                self.validation_results.append(result)
        
        # Generate comprehensive report
        return await self.generate_validation_report()
    
    async def validate_agent_file(self, file_path: Path, fix_issues: bool = False) -> ValidationResult:
        """Validate a single agent definition file."""
        agent_id = file_path.stem
        logger.debug(f"Validating agent: {agent_id}")
        
        result = ValidationResult(
            agent_id=agent_id,
            file_path=str(file_path)
        )
        
        try:
            # Read file content
            content = file_path.read_text(encoding='utf-8')
            
            # Parse YAML frontmatter and markdown content
            frontmatter, markdown_content = self.parse_agent_file(content)
            
            # Run validation checks
            await self.validate_syntax(result, frontmatter, markdown_content)
            await self.validate_structure(result, frontmatter, markdown_content)
            await self.validate_content(result, frontmatter, markdown_content)
            await self.validate_integration(result, frontmatter)
            await self.validate_performance(result, frontmatter)
            await self.validate_security(result, frontmatter, markdown_content)
            await self.validate_compliance(result, frontmatter, markdown_content)
            
            # Auto-fix issues if requested
            if fix_issues:
                await self.auto_fix_issues(result, file_path, frontmatter, markdown_content)
            
        except Exception as e:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.SYNTAX,
                severity=ValidationSeverity.CRITICAL,
                title="File Processing Error",
                description=f"Failed to process agent file: {str(e)}",
                file_path=str(file_path)
            ))
        
        return result
    
    def parse_agent_file(self, content: str) -> Tuple[Dict[str, Any], str]:
        """Parse agent file frontmatter and content."""
        # Split YAML frontmatter from markdown content
        if content.startswith('---\\n'):\n            parts = content.split('---\\n', 2)\n            if len(parts) >= 3:\n                try:\n                    frontmatter = yaml.safe_load(parts[1])\n                    markdown_content = parts[2].strip()\n                    return frontmatter or {}, markdown_content\n                except yaml.YAMLError as e:\n                    raise ValueError(f\"Invalid YAML frontmatter: {str(e)}\")\n        \n        # No frontmatter found\n        return {}, content\n    \n    async def validate_syntax(self, result: ValidationResult, frontmatter: Dict[str, Any], content: str) -> None:\n        \"\"\"Validate syntax and basic structure.\"\"\"\n        # Check required frontmatter fields\n        for field in self.required_fields:\n            if field not in frontmatter:\n                result.add_issue(ValidationIssue(\n                    category=ValidationCategory.SYNTAX,\n                    severity=ValidationSeverity.ERROR,\n                    title=f\"Missing Required Field: {field}\",\n                    description=f\"Agent definition is missing required field '{field}'\",\n                    agent_id=result.agent_id,\n                    suggestion=f\"Add '{field}' to the YAML frontmatter\",\n                    auto_fixable=True\n                ))\n        \n        # Check for unknown fields\n        valid_fields = self.required_fields | self.optional_fields\n        for field in frontmatter.keys():\n            if field not in valid_fields:\n                result.add_issue(ValidationIssue(\n                    category=ValidationCategory.SYNTAX,\n                    severity=ValidationSeverity.WARNING,\n                    title=f\"Unknown Field: {field}\",\n                    description=f\"Field '{field}' is not recognized\",\n                    agent_id=result.agent_id,\n                    suggestion=f\"Remove '{field}' or check field name spelling\"\n                ))\n        \n        # Validate model field\n        if \"model\" in frontmatter:\n            model = frontmatter[\"model\"].lower()\n            if model not in self.valid_models:\n                result.add_issue(ValidationIssue(\n                    category=ValidationCategory.SYNTAX,\n                    severity=ValidationSeverity.ERROR,\n                    title=\"Invalid Model\",\n                    description=f\"Model '{model}' is not valid. Valid models: {', '.join(self.valid_models)}\",\n                    agent_id=result.agent_id,\n                    suggestion=f\"Use one of: {', '.join(self.valid_models)}\",\n                    auto_fixable=True\n                ))\n        \n        # Validate temperature range\n        if \"temperature\" in frontmatter:\n            temp = frontmatter[\"temperature\"]\n            if not isinstance(temp, (int, float)) or not (0.0 <= temp <= 1.0):\n                result.add_issue(ValidationIssue(\n                    category=ValidationCategory.SYNTAX,\n                    severity=ValidationSeverity.ERROR,\n                    title=\"Invalid Temperature\",\n                    description=f\"Temperature must be between 0.0 and 1.0, got: {temp}\",\n                    agent_id=result.agent_id,\n                    suggestion=\"Set temperature to a value between 0.0 and 1.0\",\n                    auto_fixable=True\n                ))\n    \n    async def validate_structure(self, result: ValidationResult, frontmatter: Dict[str, Any], content: str) -> None:\n        \"\"\"Validate document structure and organization.\"\"\"\n        # Check for proper markdown structure\n        if not content.strip():\n            result.add_issue(ValidationIssue(\n                category=ValidationCategory.STRUCTURE,\n                severity=ValidationSeverity.WARNING,\n                title=\"Empty Content\",\n                description=\"Agent definition has no content beyond frontmatter\",\n                agent_id=result.agent_id,\n                suggestion=\"Add description and usage instructions\"\n            ))\n        \n        # Check for required sections\n        required_sections = [\"Core Responsibilities\", \"Capabilities\", \"Usage\"]\n        content_lower = content.lower()\n        \n        for section in required_sections:\n            if section.lower() not in content_lower:\n                result.add_issue(ValidationIssue(\n                    category=ValidationCategory.STRUCTURE,\n                    severity=ValidationSeverity.WARNING,\n                    title=f\"Missing Section: {section}\",\n                    description=f\"Recommended section '{section}' not found\",\n                    agent_id=result.agent_id,\n                    suggestion=f\"Add a '{section}' section to document the agent\"\n                ))\n        \n        # Check for proper heading structure\n        headings = re.findall(r'^#{1,6}\\s+(.+)$', content, re.MULTILINE)\n        if len(headings) < 2:\n            result.add_issue(ValidationIssue(\n                category=ValidationCategory.STRUCTURE,\n                severity=ValidationSeverity.INFO,\n                title=\"Limited Structure\",\n                description=\"Consider adding more headings to organize content\",\n                agent_id=result.agent_id,\n                suggestion=\"Use markdown headings to organize content into sections\"\n            ))\n    \n    async def validate_content(self, result: ValidationResult, frontmatter: Dict[str, Any], content: str) -> None:\n        \"\"\"Validate content quality and completeness.\"\"\"\n        # Check description length and quality\n        if \"description\" in frontmatter:\n            desc = frontmatter[\"description\"]\n            if len(desc) < 20:\n                result.add_issue(ValidationIssue(\n                    category=ValidationCategory.CONTENT,\n                    severity=ValidationSeverity.WARNING,\n                    title=\"Short Description\",\n                    description=\"Agent description is very short and may not be descriptive enough\",\n                    agent_id=result.agent_id,\n                    suggestion=\"Expand the description to better explain the agent's purpose\"\n                ))\n        \n        # Check for placeholder text\n        placeholders = [\"TODO\", \"FIXME\", \"PLACEHOLDER\", \"TBD\", \"...\", \"xxx\"]\n        full_text = (frontmatter.get(\"description\", \"\") + \" \" + content).lower()\n        \n        for placeholder in placeholders:\n            if placeholder.lower() in full_text:\n                result.add_issue(ValidationIssue(\n                    category=ValidationCategory.CONTENT,\n                    severity=ValidationSeverity.WARNING,\n                    title=\"Placeholder Text Found\",\n                    description=f\"Found placeholder text: '{placeholder}'\",\n                    agent_id=result.agent_id,\n                    suggestion=\"Replace placeholder text with actual content\"\n                ))\n        \n        # Check for examples and usage instructions\n        if \"example\" not in content.lower() and \"usage\" not in content.lower():\n            result.add_issue(ValidationIssue(\n                category=ValidationCategory.CONTENT,\n                severity=ValidationSeverity.INFO,\n                title=\"Missing Examples\",\n                description=\"No examples or usage instructions found\",\n                agent_id=result.agent_id,\n                suggestion=\"Add examples or usage instructions to help users understand the agent\"\n            ))\n    \n    async def validate_integration(self, result: ValidationResult, frontmatter: Dict[str, Any]) -> None:\n        \"\"\"Validate integration with the agent system.\"\"\"\n        # Check if agent type is recognized\n        agent_name = frontmatter.get(\"name\", result.agent_id)\n        \n        try:\n            # Try to find matching AgentType\n            agent_type_found = False\n            for agent_type in AgentType:\n                if (agent_type.value.replace(\"_\", \"-\") == result.agent_id or\n                    agent_type.value.replace(\"_\", \" \").lower() in agent_name.lower()):\n                    agent_type_found = True\n                    break\n            \n            if not agent_type_found:\n                result.add_issue(ValidationIssue(\n                    category=ValidationCategory.INTEGRATION,\n                    severity=ValidationSeverity.WARNING,\n                    title=\"Agent Type Not Found\",\n                    description=f\"No matching AgentType found for '{result.agent_id}'\",\n                    agent_id=result.agent_id,\n                    suggestion=\"Ensure the agent is registered in AgentType enum\"\n                ))\n            \n        except Exception as e:\n            result.add_issue(ValidationIssue(\n                category=ValidationCategory.INTEGRATION,\n                severity=ValidationSeverity.ERROR,\n                title=\"Integration Check Failed\",\n                description=f\"Failed to validate integration: {str(e)}\",\n                agent_id=result.agent_id\n            ))\n        \n        # Check tools specification\n        if \"tools\" in frontmatter:\n            tools = frontmatter[\"tools\"]\n            if isinstance(tools, str):\n                # Parse tools list\n                tool_list = [tool.strip() for tool in tools.split(\",\")]\n                \n                # Check for valid tools\n                valid_tools = {\n                    \"Read\", \"Write\", \"MultiEdit\", \"Bash\", \"Docker\", \"database\", \n                    \"redis\", \"postgresql\", \"magic\", \"context7\", \"playwright\"\n                }\n                \n                for tool in tool_list:\n                    if tool not in valid_tools and tool != \"*\":\n                        result.add_issue(ValidationIssue(\n                            category=ValidationCategory.INTEGRATION,\n                            severity=ValidationSeverity.WARNING,\n                            title=\"Unknown Tool\",\n                            description=f\"Tool '{tool}' is not recognized\",\n                            agent_id=result.agent_id,\n                            suggestion=f\"Check tool name or add to valid tools list\"\n                        ))\n    \n    async def validate_performance(self, result: ValidationResult, frontmatter: Dict[str, Any]) -> None:\n        \"\"\"Validate performance-related settings.\"\"\"\n        score = 100.0\n        \n        # Check max_tokens setting\n        if \"max_tokens\" in frontmatter:\n            max_tokens = frontmatter[\"max_tokens\"]\n            model = frontmatter.get(\"model\", \"sonnet\").lower()\n            \n            if model in self.performance_thresholds[\"max_tokens\"]:\n                recommended_max = self.performance_thresholds[\"max_tokens\"][model]\n                if max_tokens > recommended_max:\n                    score -= 10\n                    result.add_issue(ValidationIssue(\n                        category=ValidationCategory.PERFORMANCE,\n                        severity=ValidationSeverity.WARNING,\n                        title=\"High Token Limit\",\n                        description=f\"max_tokens ({max_tokens}) exceeds recommended limit ({recommended_max}) for {model}\",\n                        agent_id=result.agent_id,\n                        suggestion=f\"Consider reducing max_tokens to {recommended_max} or lower\"\n                    ))\n        \n        # Check temperature for performance impact\n        if \"temperature\" in frontmatter:\n            temp = frontmatter[\"temperature\"]\n            if isinstance(temp, (int, float)) and temp > 0.7:\n                score -= 5\n                result.add_issue(ValidationIssue(\n                    category=ValidationCategory.PERFORMANCE,\n                    severity=ValidationSeverity.INFO,\n                    title=\"High Temperature\",\n                    description=f\"High temperature ({temp}) may impact response consistency\",\n                    agent_id=result.agent_id,\n                    suggestion=\"Consider using lower temperature for more consistent results\"\n                ))\n        \n        result.performance_score = max(0, score)\n    \n    async def validate_security(self, result: ValidationResult, frontmatter: Dict[str, Any], content: str) -> None:\n        \"\"\"Validate security-related aspects.\"\"\"\n        # Check for sensitive information in content\n        sensitive_patterns = [\n            (r'api[_-]?key', \"API Key\"),\n            (r'secret[_-]?key', \"Secret Key\"),\n            (r'password', \"Password\"),\n            (r'token', \"Token\"),\n            (r'private[_-]?key', \"Private Key\")\n        ]\n        \n        full_text = (str(frontmatter) + \" \" + content).lower()\n        \n        for pattern, name in sensitive_patterns:\n            if re.search(pattern, full_text):\n                result.add_issue(ValidationIssue(\n                    category=ValidationCategory.SECURITY,\n                    severity=ValidationSeverity.WARNING,\n                    title=f\"Potential Sensitive Data: {name}\",\n                    description=f\"Found potential {name.lower()} reference in agent definition\",\n                    agent_id=result.agent_id,\n                    suggestion=f\"Ensure no actual {name.lower()} values are exposed\"\n                ))\n        \n        # Check for proper tool restrictions\n        if \"tools\" in frontmatter and frontmatter[\"tools\"] == \"*\":\n            result.add_issue(ValidationIssue(\n                category=ValidationCategory.SECURITY,\n                severity=ValidationSeverity.INFO,\n                title=\"All Tools Access\",\n                description=\"Agent has access to all tools (*)\",\n                agent_id=result.agent_id,\n                suggestion=\"Consider limiting tools to only what's needed for security\"\n            ))\n    \n    async def validate_compliance(self, result: ValidationResult, frontmatter: Dict[str, Any], content: str) -> None:\n        \"\"\"Validate compliance with standards and policies.\"\"\"\n        score = 100.0\n        \n        # Check documentation completeness\n        if len(content.strip()) < 100:\n            score -= 20\n            result.add_issue(ValidationIssue(\n                category=ValidationCategory.COMPLIANCE,\n                severity=ValidationSeverity.WARNING,\n                title=\"Insufficient Documentation\",\n                description=\"Agent documentation is too brief\",\n                agent_id=result.agent_id,\n                suggestion=\"Add more detailed documentation about the agent's capabilities and usage\"\n            ))\n        \n        # Check for accessibility considerations\n        accessibility_keywords = [\"accessible\", \"accessibility\", \"screen reader\", \"wcag\"]\n        if any(keyword in content.lower() for keyword in accessibility_keywords):\n            score += 5\n        else:\n            result.add_issue(ValidationIssue(\n                category=ValidationCategory.COMPLIANCE,\n                severity=ValidationSeverity.INFO,\n                title=\"No Accessibility Mentions\",\n                description=\"No explicit accessibility considerations mentioned\",\n                agent_id=result.agent_id,\n                suggestion=\"Consider adding accessibility guidelines if the agent handles UI/content\"\n            ))\n        \n        # Check for proper agent naming conventions\n        if \"name\" in frontmatter:\n            name = frontmatter[\"name\"]\n            if not re.match(r'^[a-zA-Z][a-zA-Z0-9\\s\\-_]*[a-zA-Z0-9]$', name):\n                score -= 10\n                result.add_issue(ValidationIssue(\n                    category=ValidationCategory.COMPLIANCE,\n                    severity=ValidationSeverity.WARNING,\n                    title=\"Non-Standard Agent Name\",\n                    description=\"Agent name doesn't follow naming conventions\",\n                    agent_id=result.agent_id,\n                    suggestion=\"Use alphanumeric characters, spaces, hyphens, and underscores only\"\n                ))\n        \n        result.compliance_score = max(0, score)\n    \n    async def auto_fix_issues(self, result: ValidationResult, file_path: Path, frontmatter: Dict[str, Any], content: str) -> None:\n        \"\"\"Automatically fix issues that can be resolved.\"\"\"\n        fixed_issues = []\n        modified_frontmatter = frontmatter.copy()\n        \n        for issue in result.issues:\n            if not issue.auto_fixable:\n                continue\n            \n            try:\n                if \"Missing Required Field\" in issue.title:\n                    field_name = issue.title.split(\": \")[-1]\n                    if field_name == \"model\":\n                        modified_frontmatter[\"model\"] = \"sonnet\"\n                    elif field_name == \"temperature\":\n                        modified_frontmatter[\"temperature\"] = 0.3\n                    elif field_name == \"max_tokens\":\n                        modified_frontmatter[\"max_tokens\"] = 4000\n                    elif field_name == \"description\":\n                        modified_frontmatter[\"description\"] = f\"Specialized AI agent: {result.agent_id}\"\n                    elif field_name == \"name\":\n                        modified_frontmatter[\"name\"] = result.agent_id.replace(\"-\", \" \").title()\n                    \n                    fixed_issues.append(issue)\n                \n                elif \"Invalid Temperature\" in issue.title:\n                    modified_frontmatter[\"temperature\"] = 0.3\n                    fixed_issues.append(issue)\n                \n                elif \"Invalid Model\" in issue.title:\n                    modified_frontmatter[\"model\"] = \"sonnet\"\n                    fixed_issues.append(issue)\n                    \n            except Exception as e:\n                logger.warning(f\"Failed to auto-fix issue '{issue.title}': {str(e)}\")\n        \n        # Write fixed content back to file if changes were made\n        if fixed_issues and modified_frontmatter != frontmatter:\n            try:\n                # Reconstruct file content\n                yaml_content = yaml.dump(modified_frontmatter, default_flow_style=False)\n                new_content = f\"---\\n{yaml_content}---\\n\\n{content}\"\n                \n                # Write to file\n                file_path.write_text(new_content, encoding='utf-8')\n                \n                logger.info(f\"Auto-fixed {len(fixed_issues)} issues in {file_path}\")\n                \n                # Remove fixed issues from result\n                result.issues = [issue for issue in result.issues if issue not in fixed_issues]\n                \n            except Exception as e:\n                logger.error(f\"Failed to write auto-fixed content to {file_path}: {str(e)}\")\n    \n    async def generate_validation_report(self) -> Dict[str, Any]:\n        \"\"\"Generate comprehensive validation report.\"\"\"\n        if not self.validation_results:\n            return {\"error\": \"No validation results available\"}\n        \n        # Calculate summary statistics\n        total_agents = len(self.validation_results)\n        passed_agents = len([r for r in self.validation_results if r.passed])\n        failed_agents = total_agents - passed_agents\n        \n        # Count issues by severity\n        issue_counts = {\n            \"critical\": 0,\n            \"error\": 0,\n            \"warning\": 0,\n            \"info\": 0\n        }\n        \n        # Count issues by category\n        category_counts = {category.value: 0 for category in ValidationCategory}\n        \n        all_issues = []\n        for result in self.validation_results:\n            all_issues.extend(result.issues)\n        \n        for issue in all_issues:\n            issue_counts[issue.severity.value] += 1\n            category_counts[issue.category.value] += 1\n        \n        # Calculate average scores\n        avg_performance_score = sum(r.performance_score for r in self.validation_results) / total_agents\n        avg_compliance_score = sum(r.compliance_score for r in self.validation_results) / total_agents\n        \n        # Find most problematic agents\n        problematic_agents = sorted(\n            [r for r in self.validation_results if not r.passed],\n            key=lambda r: len([i for i in r.issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]),\n            reverse=True\n        )[:5]\n        \n        # Find best agents\n        best_agents = sorted(\n            [r for r in self.validation_results if r.passed],\n            key=lambda r: (r.performance_score + r.compliance_score) / 2,\n            reverse=True\n        )[:5]\n        \n        # Generate recommendations\n        recommendations = await self.generate_recommendations(all_issues)\n        \n        return {\n            \"summary\": {\n                \"total_agents\": total_agents,\n                \"passed_agents\": passed_agents,\n                \"failed_agents\": failed_agents,\n                \"success_rate\": (passed_agents / total_agents) * 100,\n                \"total_issues\": len(all_issues),\n                \"average_performance_score\": round(avg_performance_score, 2),\n                \"average_compliance_score\": round(avg_compliance_score, 2)\n            },\n            \"issue_breakdown\": {\n                \"by_severity\": issue_counts,\n                \"by_category\": category_counts\n            },\n            \"problematic_agents\": [\n                {\n                    \"agent_id\": r.agent_id,\n                    \"file_path\": r.file_path,\n                    \"critical_issues\": len(r.get_issues_by_severity(ValidationSeverity.CRITICAL)),\n                    \"error_issues\": len(r.get_issues_by_severity(ValidationSeverity.ERROR)),\n                    \"total_issues\": len(r.issues)\n                }\n                for r in problematic_agents\n            ],\n            \"best_agents\": [\n                {\n                    \"agent_id\": r.agent_id,\n                    \"file_path\": r.file_path,\n                    \"performance_score\": r.performance_score,\n                    \"compliance_score\": r.compliance_score,\n                    \"total_issues\": len(r.issues)\n                }\n                for r in best_agents\n            ],\n            \"recommendations\": recommendations,\n            \"validation_timestamp\": datetime.utcnow().isoformat(),\n            \"detailed_results\": [\n                {\n                    \"agent_id\": r.agent_id,\n                    \"file_path\": r.file_path,\n                    \"passed\": r.passed,\n                    \"performance_score\": r.performance_score,\n                    \"compliance_score\": r.compliance_score,\n                    \"issues\": [\n                        {\n                            \"category\": i.category.value,\n                            \"severity\": i.severity.value,\n                            \"title\": i.title,\n                            \"description\": i.description,\n                            \"suggestion\": i.suggestion,\n                            \"auto_fixable\": i.auto_fixable\n                        }\n                        for i in r.issues\n                    ]\n                }\n                for r in self.validation_results\n            ]\n        }\n    \n    async def generate_recommendations(self, issues: List[ValidationIssue]) -> List[str]:\n        \"\"\"Generate actionable recommendations based on validation issues.\"\"\"\n        recommendations = []\n        \n        # Count issues by type to identify patterns\n        issue_types = {}\n        for issue in issues:\n            key = f\"{issue.category.value}_{issue.severity.value}\"\n            issue_types[key] = issue_types.get(key, 0) + 1\n        \n        # Generate recommendations based on common issues\n        if issue_types.get(\"syntax_error\", 0) > 5:\n            recommendations.append(\n                \"Consider implementing a pre-commit hook to validate YAML syntax before committing agent definitions\"\n            )\n        \n        if issue_types.get(\"content_warning\", 0) > 10:\n            recommendations.append(\n                \"Many agents have content quality issues. Consider creating a content style guide for agent documentation\"\n            )\n        \n        if issue_types.get(\"performance_warning\", 0) > 5:\n            recommendations.append(\n                \"Several agents have performance concerns. Review token limits and temperature settings for optimal performance\"\n            )\n        \n        if issue_types.get(\"security_warning\", 0) > 3:\n            recommendations.append(\n                \"Security issues detected. Implement a security review process for agent definitions\"\n            )\n        \n        if issue_types.get(\"integration_warning\", 0) > 5:\n            recommendations.append(\n                \"Integration issues found. Ensure all agents are properly registered in the AgentType enum\"\n            )\n        \n        # Default recommendations\n        if len(recommendations) == 0:\n            recommendations.extend([\n                \"All agents passed basic validation. Consider implementing automated testing for agent definitions\",\n                \"Set up continuous integration to validate agents on every commit\",\n                \"Create documentation templates to ensure consistency across agent definitions\"\n            ])\n        \n        return recommendations\n    \n    def print_summary(self, report: Dict[str, Any]) -> None:\n        \"\"\"Print a formatted summary of validation results.\"\"\"\n        print(\"\\n\" + \"=\" * 80)\n        print(\"🤖 AGENT DEFINITION VALIDATION REPORT\")\n        print(\"=\" * 80)\n        \n        summary = report[\"summary\"]\n        print(f\"\\n📊 SUMMARY:\")\n        print(f\"   Total Agents: {summary['total_agents']}\")\n        print(f\"   ✅ Passed: {summary['passed_agents']} ({summary['success_rate']:.1f}%)\")\n        print(f\"   ❌ Failed: {summary['failed_agents']}\")\n        print(f\"   🔍 Total Issues: {summary['total_issues']}\")\n        print(f\"   ⚡ Avg Performance Score: {summary['average_performance_score']}/100\")\n        print(f\"   📋 Avg Compliance Score: {summary['average_compliance_score']}/100\")\n        \n        # Issue breakdown\n        issues = report[\"issue_breakdown\"]\n        print(f\"\\n🚨 ISSUES BY SEVERITY:\")\n        for severity, count in issues[\"by_severity\"].items():\n            if count > 0:\n                emoji = {\"critical\": \"🔴\", \"error\": \"🟠\", \"warning\": \"🟡\", \"info\": \"🔵\"}[severity]\n                print(f\"   {emoji} {severity.upper()}: {count}\")\n        \n        print(f\"\\n📂 ISSUES BY CATEGORY:\")\n        for category, count in issues[\"by_category\"].items():\n            if count > 0:\n                print(f\"   • {category.upper()}: {count}\")\n        \n        # Problematic agents\n        if report[\"problematic_agents\"]:\n            print(f\"\\n⚠️  MOST PROBLEMATIC AGENTS:\")\n            for i, agent in enumerate(report[\"problematic_agents\"][:3], 1):\n                print(f\"   {i}. {agent['agent_id']} - {agent['total_issues']} issues\")\n        \n        # Best agents\n        if report[\"best_agents\"]:\n            print(f\"\\n🏆 TOP PERFORMING AGENTS:\")\n            for i, agent in enumerate(report[\"best_agents\"][:3], 1):\n                avg_score = (agent['performance_score'] + agent['compliance_score']) / 2\n                print(f\"   {i}. {agent['agent_id']} - {avg_score:.1f}/100 avg score\")\n        \n        # Recommendations\n        if report[\"recommendations\"]:\n            print(f\"\\n💡 RECOMMENDATIONS:\")\n            for i, rec in enumerate(report[\"recommendations\"][:3], 1):\n                print(f\"   {i}. {rec}\")\n        \n        print(\"\\n\" + \"=\" * 80)\n        print(f\"Validation completed at: {report['validation_timestamp']}\")\n        print(\"=\" * 80)\n\n\nasync def main():\n    \"\"\"Main validation script.\"\"\"\n    import argparse\n    \n    parser = argparse.ArgumentParser(description=\"Validate agent definitions\")\n    parser.add_argument(\"--fix\", action=\"store_true\", help=\"Auto-fix issues where possible\")\n    parser.add_argument(\"--output\", help=\"Output file for detailed report (JSON)\")\n    parser.add_argument(\"--verbose\", \"-v\", action=\"store_true\", help=\"Verbose output\")\n    \n    args = parser.parse_args()\n    \n    # Configure logging\n    if args.verbose:\n        structlog.configure(level=\"DEBUG\")\n    else:\n        structlog.configure(level=\"INFO\")\n    \n    # Run validation\n    validator = AgentDefinitionValidator()\n    \n    print(\"🚀 Starting agent definition validation...\")\n    report = await validator.validate_all_agents(fix_issues=args.fix)\n    \n    if \"error\" in report:\n        print(f\"❌ Validation failed: {report['error']}\")\n        sys.exit(1)\n    \n    # Print summary\n    validator.print_summary(report)\n    \n    # Save detailed report if requested\n    if args.output:\n        with open(args.output, 'w') as f:\n            json.dump(report, f, indent=2)\n        print(f\"\\n📄 Detailed report saved to: {args.output}\")\n    \n    # Exit with error code if validation failed\n    if report[\"summary\"][\"failed_agents\"] > 0:\n        print(\"\\n⚠️  Some agents failed validation. Please review and fix issues.\")\n        sys.exit(1)\n    else:\n        print(\"\\n✅ All agents passed validation!\")\n        sys.exit(0)\n\n\nif __name__ == \"__main__\":\n    asyncio.run(main())