"""
PydanticAI agents for intelligent PRD processing and strategic planning.
"""

from .base_agent import BaseAgent, AgentResult
from .context_manager import ContextManagerAgent  
from .prd_generator import PRDGeneratorAgent
from .draft_agent import DraftAgent
from .judge_agent import JudgeAgent
from .task_executor import TaskExecutorAgent
from .documentation_librarian import DocumentationLibrarianAgent

# Legacy agents (if they exist)
try:
    from .prd_agent import PRDCreationAgent
    from .research_agent import ResearchAgent
    from .task_agent import TaskGenerationAgent
    from .validation_agent import ValidationAgent
    LEGACY_AGENTS = [
        "PRDCreationAgent",
        "ResearchAgent", 
        "TaskGenerationAgent",
        "ValidationAgent"
    ]
except ImportError:
    LEGACY_AGENTS = []

__all__ = [
    "BaseAgent",
    "AgentResult",
    "ContextManagerAgent",
    "PRDGeneratorAgent", 
    "DraftAgent",
    "JudgeAgent",
    "TaskExecutorAgent",
    "DocumentationLibrarianAgent",
] + LEGACY_AGENTS