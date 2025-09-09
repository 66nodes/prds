"""
Base agent interface for PydanticAI agents in the Strategic Planning Platform.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from datetime import datetime
import structlog

from pydantic import BaseModel, Field
from pydantic_ai import Agent as PydanticAIAgent

logger = structlog.get_logger(__name__)


class AgentResult(BaseModel):
    """Standard result structure for agent operations."""
    success: bool = Field(..., description="Whether the operation succeeded")
    result: Any = Field(..., description="Operation result")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence in result")
    validation_results: List[Dict[str, Any]] = Field(default_factory=list, description="Validation results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    update_context: Optional[Dict[str, Any]] = Field(None, description="Context updates to apply")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    error: Optional[str] = Field(None, description="Error message if failed")


class BaseAgent(ABC):
    """
    Abstract base class for all PydanticAI agents.
    Provides common functionality and interface standardization.
    """
    
    def __init__(self, hybrid_rag_service, agent_name: str):
        self.hybrid_rag = hybrid_rag_service
        self.agent_name = agent_name
        self.pydantic_agent: Optional[PydanticAIAgent] = None
        self._initialize_agent()
    
    @abstractmethod
    def _initialize_agent(self) -> None:
        """Initialize the PydanticAI agent instance."""
        pass
    
    @abstractmethod
    async def execute(self, operation: str, context: Dict[str, Any]) -> AgentResult:
        """
        Execute an operation with the given context.
        
        Args:
            operation: The operation to perform
            context: Execution context including parameters and shared data
            
        Returns:
            AgentResult with operation outcome
        """
        pass
    
    async def validate_with_graphrag(
        self, 
        content: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate content using the HybridRAG service.
        
        Args:
            content: Content to validate
            context: Additional context for validation
            
        Returns:
            Validation results from GraphRAG
        """
        try:
            return await self.hybrid_rag.validate_content(content, context)
        except Exception as e:
            logger.error(
                f"GraphRAG validation failed in {self.agent_name}",
                error=str(e),
                exc_info=True
            )
            return {
                'validation_id': 'failed',
                'confidence': 0.5,
                'passes_threshold': False,
                'error': str(e)
            }
    
    def _create_success_result(
        self,
        result: Any,
        confidence_score: Optional[float] = None,
        validation_results: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        update_context: Optional[Dict[str, Any]] = None,
        processing_time_ms: Optional[int] = None
    ) -> AgentResult:
        """Create a successful AgentResult."""
        return AgentResult(
            success=True,
            result=result,
            confidence_score=confidence_score,
            validation_results=validation_results or [],
            metadata=metadata or {},
            update_context=update_context,
            processing_time_ms=processing_time_ms
        )
    
    def _create_error_result(
        self,
        error: str,
        result: Any = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """Create a failed AgentResult."""
        return AgentResult(
            success=False,
            result=result,
            error=error,
            metadata=metadata or {}
        )
    
    def _extract_context_parameter(
        self,
        context: Dict[str, Any],
        parameter_name: str,
        required: bool = True,
        default: Any = None
    ) -> Any:
        """
        Extract a parameter from the execution context.
        
        Args:
            context: Execution context
            parameter_name: Name of parameter to extract
            required: Whether the parameter is required
            default: Default value if not found
            
        Returns:
            Parameter value
            
        Raises:
            ValueError: If required parameter is missing
        """
        if parameter_name in context.get("task_parameters", {}):
            return context["task_parameters"][parameter_name]
        elif parameter_name in context:
            return context[parameter_name]
        elif not required:
            return default
        else:
            raise ValueError(f"Required parameter '{parameter_name}' missing from context")
    
    def _log_operation_start(self, operation: str, context: Dict[str, Any]) -> datetime:
        """Log the start of an operation."""
        start_time = datetime.utcnow()
        logger.info(
            f"Starting {operation} operation",
            agent_name=self.agent_name,
            operation=operation,
            workflow_id=context.get("workflow_id"),
            user_id=context.get("user_id")
        )
        return start_time
    
    def _log_operation_complete(
        self, 
        operation: str, 
        start_time: datetime,
        success: bool,
        error: Optional[str] = None
    ) -> int:
        """Log the completion of an operation."""
        processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        if success:
            logger.info(
                f"Completed {operation} operation",
                agent_name=self.agent_name,
                operation=operation,
                processing_time_ms=processing_time_ms
            )
        else:
            logger.error(
                f"Failed {operation} operation",
                agent_name=self.agent_name,
                operation=operation,
                processing_time_ms=processing_time_ms,
                error=error
            )
        
        return processing_time_ms