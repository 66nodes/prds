"""
GraphRAG Validation Middleware for Enhanced Parallel Execution

Integrates comprehensive GraphRAG validation with the enhanced parallel agent execution
system to ensure <2% hallucination rate and high-quality agent outputs.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid

import structlog
from services.enhanced_parallel_executor import (
    ExecutionMetrics, ExecutionStatus, PriorityLevel
)
from services.agent_orchestrator import AgentTask, WorkflowContext, AgentType
from services.graphrag.validation_pipeline import (
    ValidationPipeline, ValidationLevel, ValidationResult, ValidationStatus, ValidationConfig
)

logger = structlog.get_logger(__name__)


class ValidationStage(Enum):
    """Validation stages in the pipeline."""
    PRE_EXECUTION = "pre_execution"      # Before agent execution
    POST_EXECUTION = "post_execution"    # After agent execution
    ITERATIVE = "iterative"              # During iterative improvement
    FINAL = "final"                      # Final validation before completion


@dataclass
class ValidationCheckpoint:
    """Configuration for validation checkpoints."""
    stage: ValidationStage
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    enable_corrections: bool = True
    enable_iterative_improvement: bool = True
    max_iterations: int = 3
    confidence_threshold: float = 0.8
    hallucination_threshold: float = 0.02  # 2% max hallucination rate
    timeout_seconds: int = 30


@dataclass
class AgentValidationResult:
    """Result of agent output validation."""
    validation_id: str
    task_id: str
    agent_type: AgentType
    stage: ValidationStage
    
    # Validation results
    validation_result: Optional[ValidationResult] = None
    passes_validation: bool = False
    confidence_score: float = 0.0
    hallucination_rate: float = 0.0
    
    # Processing metrics
    processing_time_ms: float = 0.0
    iterations_performed: int = 0
    
    # Content transformation
    original_content: Optional[str] = None
    validated_content: Optional[str] = None
    corrections_applied: List[str] = field(default_factory=list)
    
    # Quality metrics
    quality_improvements: Dict[str, float] = field(default_factory=dict)
    validation_feedback: Dict[str, Any] = field(default_factory=dict)


class GraphRAGValidationMiddleware:
    """
    Validation middleware integrating GraphRAG pipeline with enhanced parallel execution.
    
    Provides configurable validation checkpoints with iterative improvement,
    ensuring high-quality agent outputs with <2% hallucination rate.
    """
    
    def __init__(self):
        self.validation_pipeline = ValidationPipeline()
        self.is_initialized = False
        
        # Default validation checkpoints for different agent types
        self.agent_validation_configs = {
            # Strategic agents require strict validation
            AgentType.DRAFT_AGENT: ValidationCheckpoint(
                stage=ValidationStage.POST_EXECUTION,
                validation_level=ValidationLevel.STRICT,
                confidence_threshold=0.9,
                hallucination_threshold=0.01  # 1% for strategic content
            ),
            AgentType.JUDGE_AGENT: ValidationCheckpoint(
                stage=ValidationStage.POST_EXECUTION,
                validation_level=ValidationLevel.STRICT,
                confidence_threshold=0.95,
                hallucination_threshold=0.005  # 0.5% for judgment
            ),
            
            # Business agents need standard validation
            AgentType.BUSINESS_ANALYST: ValidationCheckpoint(
                stage=ValidationStage.POST_EXECUTION,
                validation_level=ValidationLevel.STANDARD,
                confidence_threshold=0.8,
                hallucination_threshold=0.02
            ),
            AgentType.PROJECT_ARCHITECT: ValidationCheckpoint(
                stage=ValidationStage.POST_EXECUTION,
                validation_level=ValidationLevel.STANDARD,
                confidence_threshold=0.85,
                hallucination_threshold=0.015
            ),
            
            # Context manager can use basic validation for speed
            AgentType.CONTEXT_MANAGER: ValidationCheckpoint(
                stage=ValidationStage.POST_EXECUTION,
                validation_level=ValidationLevel.BASIC,
                confidence_threshold=0.75,
                hallucination_threshold=0.03
            )
        }
        
        # Default checkpoint for unknown agent types
        self.default_checkpoint = ValidationCheckpoint(
            stage=ValidationStage.POST_EXECUTION,
            validation_level=ValidationLevel.STANDARD
        )
        
        # Performance tracking
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'corrections_applied': 0,
            'avg_processing_time_ms': 0.0,
            'avg_iterations': 0.0,
            'hallucination_prevention_count': 0,
            'quality_improvement_rate': 0.0
        }
        
        # Active validation tasks
        self.active_validations: Dict[str, AgentValidationResult] = {}
    
    async def initialize(self) -> None:
        """Initialize the validation middleware."""
        try:
            logger.info("Initializing GraphRAG validation middleware...")
            start_time = datetime.now()
            
            # Initialize validation pipeline
            await self.validation_pipeline.initialize()
            
            init_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"GraphRAG validation middleware initialized in {init_time:.2f}ms")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize validation middleware: {str(e)}")
            raise
    
    async def validate_agent_output(
        self,
        task: AgentTask,
        agent_output: Dict[str, Any],
        workflow: WorkflowContext,
        execution_metrics: ExecutionMetrics,
        custom_checkpoint: Optional[ValidationCheckpoint] = None
    ) -> AgentValidationResult:
        """
        Validate agent output with comprehensive GraphRAG validation.
        
        Args:
            task: The agent task that was executed
            agent_output: The output from the agent
            workflow: Workflow context
            execution_metrics: Execution metrics from parallel executor
            custom_checkpoint: Optional custom validation configuration
            
        Returns:
            Comprehensive validation result with corrections if needed
        """
        if not self.is_initialized:
            raise RuntimeError("Validation middleware not initialized")
        
        validation_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Get validation checkpoint configuration
        checkpoint = custom_checkpoint or self.agent_validation_configs.get(
            task.agent_type, self.default_checkpoint
        )
        
        # Initialize validation result
        validation_result = AgentValidationResult(
            validation_id=validation_id,
            task_id=task.task_id,
            agent_type=task.agent_type,
            stage=checkpoint.stage
        )
        
        self.active_validations[validation_id] = validation_result
        
        try:
            # Extract content from agent output
            content = self._extract_content_from_output(agent_output)
            validation_result.original_content = content
            
            if not content:
                # If no content to validate, mark as passed
                validation_result.passes_validation = True
                validation_result.confidence_score = 1.0
                validation_result.validated_content = content
                return validation_result
            
            # Prepare validation context
            validation_context = {
                'task_id': task.task_id,
                'agent_type': task.agent_type.value,
                'workflow_id': workflow.workflow_id,
                'execution_metrics': {
                    'duration_ms': execution_metrics.duration_ms,
                    'queue_wait_time_ms': execution_metrics.queue_wait_time_ms,
                    'retry_count': execution_metrics.execution_attempts
                },
                'task_metadata': task.metadata or {},
                'input_data': task.input_data
            }
            
            # Perform validation with iterative improvement
            final_content = content
            iterations = 0
            best_validation = None
            
            while iterations < checkpoint.max_iterations:
                iterations += 1
                
                logger.debug(
                    f"Performing validation iteration {iterations}",
                    task_id=task.task_id,
                    validation_id=validation_id
                )
                
                # Run GraphRAG validation
                graphrag_result = await self.validation_pipeline.validate_content(
                    final_content,
                    checkpoint.validation_level,
                    context=validation_context
                )
                
                # Check if validation passes thresholds
                passes_confidence = graphrag_result.overall_confidence >= checkpoint.confidence_threshold
                passes_hallucination = True
                
                if graphrag_result.hallucination_detection_result:
                    hallucination_rate = graphrag_result.hallucination_detection_result.get('hallucination_rate', 0)
                    passes_hallucination = hallucination_rate <= checkpoint.hallucination_threshold
                    validation_result.hallucination_rate = hallucination_rate
                
                # Update validation result
                validation_result.validation_result = graphrag_result
                validation_result.confidence_score = graphrag_result.overall_confidence
                validation_result.passes_validation = (
                    passes_confidence and 
                    passes_hallucination and 
                    graphrag_result.status in [ValidationStatus.PASSED, ValidationStatus.WARNING]
                )
                
                # Store best result so far
                if best_validation is None or graphrag_result.overall_confidence > best_validation.overall_confidence:
                    best_validation = graphrag_result
                
                # If validation passes, we're done
                if validation_result.passes_validation:
                    break
                
                # Apply corrections if enabled and available
                if checkpoint.enable_corrections and graphrag_result.corrections:
                    corrected_content = await self._apply_corrections(
                        final_content, graphrag_result.corrections
                    )
                    
                    if corrected_content != final_content:
                        validation_result.corrections_applied.extend([
                            f"Iteration {iterations}: Applied {len(graphrag_result.corrections)} corrections"
                        ])
                        final_content = corrected_content
                        continue
                
                # If no corrections available or not enabled, break
                break
            
            # Update final results
            validation_result.iterations_performed = iterations
            validation_result.validated_content = final_content
            validation_result.validation_result = best_validation or graphrag_result
            
            # Calculate quality improvements
            if final_content != content:
                validation_result.quality_improvements = {
                    'content_length_change': len(final_content) - len(content),
                    'corrections_applied': len(validation_result.corrections_applied),
                    'confidence_improvement': validation_result.confidence_score - (best_validation.overall_confidence if best_validation else 0)
                }
            
            # Update processing time
            validation_result.processing_time_ms = (time.time() - start_time) * 1000
            
            # Update statistics
            self._update_validation_stats(validation_result, checkpoint)
            
            logger.info(
                "Agent output validation completed",
                task_id=task.task_id,
                validation_id=validation_id,
                passes_validation=validation_result.passes_validation,
                confidence=validation_result.confidence_score,
                hallucination_rate=validation_result.hallucination_rate,
                iterations=iterations,
                processing_time_ms=validation_result.processing_time_ms
            )
            
            return validation_result
            
        except Exception as e:
            validation_result.passes_validation = False
            validation_result.processing_time_ms = (time.time() - start_time) * 1000
            validation_result.validation_feedback = {
                'error': str(e),
                'error_type': type(e).__name__
            }
            
            logger.error(
                f"Agent output validation failed: {str(e)}",
                task_id=task.task_id,
                validation_id=validation_id
            )
            
            return validation_result
        
        finally:
            # Clean up active validation
            if validation_id in self.active_validations:
                del self.active_validations[validation_id]
    
    async def validate_batch_outputs(
        self,
        task_outputs: List[Tuple[AgentTask, Dict[str, Any], WorkflowContext, ExecutionMetrics]],
        custom_checkpoint: Optional[ValidationCheckpoint] = None
    ) -> List[AgentValidationResult]:
        """
        Validate multiple agent outputs in parallel for efficiency.
        
        Args:
            task_outputs: List of (task, output, workflow, metrics) tuples
            custom_checkpoint: Optional custom validation configuration
            
        Returns:
            List of validation results corresponding to input order
        """
        logger.info(f"Starting batch validation of {len(task_outputs)} agent outputs")
        
        # Create validation tasks
        validation_tasks = [
            self.validate_agent_output(task, output, workflow, metrics, custom_checkpoint)
            for task, output, workflow, metrics in task_outputs
        ]
        
        # Execute validations in parallel
        results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        validation_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task, output, workflow, metrics = task_outputs[i]
                error_result = AgentValidationResult(
                    validation_id=str(uuid.uuid4()),
                    task_id=task.task_id,
                    agent_type=task.agent_type,
                    stage=ValidationStage.POST_EXECUTION
                )
                error_result.passes_validation = False
                error_result.validation_feedback = {
                    'error': str(result),
                    'error_type': type(result).__name__
                }
                validation_results.append(error_result)
            else:
                validation_results.append(result)
        
        return validation_results
    
    def _extract_content_from_output(self, agent_output: Dict[str, Any]) -> Optional[str]:
        """Extract content from agent output for validation."""
        # Try common content fields
        content_fields = ['content', 'output', 'result', 'text', 'generated_text', 'response']
        
        for field in content_fields:
            if field in agent_output:
                content = agent_output[field]
                if isinstance(content, str) and content.strip():
                    return content.strip()
        
        # Try nested structures
        if 'data' in agent_output and isinstance(agent_output['data'], dict):
            for field in content_fields:
                if field in agent_output['data']:
                    content = agent_output['data'][field]
                    if isinstance(content, str) and content.strip():
                        return content.strip()
        
        # Fallback to stringifying the entire output
        if agent_output:
            return str(agent_output)
        
        return None
    
    async def _apply_corrections(
        self,
        content: str,
        corrections: List[Dict[str, Any]]
    ) -> str:
        """Apply GraphRAG corrections to content."""
        corrected_content = content
        
        for correction in corrections:
            correction_type = correction.get('type', 'unknown')
            
            if correction_type == 'factual_correction':
                correction_data = correction.get('correction', {})
                original_text = correction_data.get('original', '')
                corrected_text = correction_data.get('corrected', '')
                
                if original_text and corrected_text and original_text in corrected_content:
                    corrected_content = corrected_content.replace(original_text, corrected_text)
            
            elif correction_type == 'content_correction':
                # Apply more sophisticated content corrections
                correction_data = correction.get('correction', {})
                if 'suggestions' in correction_data:
                    # Apply suggestions based on their confidence scores
                    for suggestion in correction_data['suggestions']:
                        if suggestion.get('confidence', 0) > 0.8:
                            original = suggestion.get('original', '')
                            replacement = suggestion.get('replacement', '')
                            if original and replacement and original in corrected_content:
                                corrected_content = corrected_content.replace(original, replacement)
        
        return corrected_content
    
    def _update_validation_stats(
        self,
        validation_result: AgentValidationResult,
        checkpoint: ValidationCheckpoint
    ) -> None:
        """Update validation statistics."""
        self.validation_stats['total_validations'] += 1
        
        if validation_result.passes_validation:
            self.validation_stats['successful_validations'] += 1
        else:
            self.validation_stats['failed_validations'] += 1
        
        if validation_result.corrections_applied:
            self.validation_stats['corrections_applied'] += len(validation_result.corrections_applied)
        
        # Update averages
        total = self.validation_stats['total_validations']
        
        # Processing time average
        self.validation_stats['avg_processing_time_ms'] = (
            (self.validation_stats['avg_processing_time_ms'] * (total - 1) + 
             validation_result.processing_time_ms) / total
        )
        
        # Iterations average
        self.validation_stats['avg_iterations'] = (
            (self.validation_stats['avg_iterations'] * (total - 1) + 
             validation_result.iterations_performed) / total
        )
        
        # Hallucination prevention
        if validation_result.hallucination_rate > checkpoint.hallucination_threshold:
            self.validation_stats['hallucination_prevention_count'] += 1
        
        # Quality improvement rate
        if validation_result.quality_improvements:
            self.validation_stats['quality_improvement_rate'] = (
                (self.validation_stats['quality_improvement_rate'] * (total - 1) + 1) / total
            )
    
    async def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics."""
        stats = dict(self.validation_stats)
        
        # Calculate success rate
        if stats['total_validations'] > 0:
            stats['success_rate'] = (
                stats['successful_validations'] / stats['total_validations'] * 100
            )
        else:
            stats['success_rate'] = 0.0
        
        # Add pipeline statistics
        pipeline_stats = await self.validation_pipeline.get_validation_statistics()
        stats['pipeline_stats'] = pipeline_stats
        
        # Active validations
        stats['active_validations'] = len(self.active_validations)
        
        # Agent-specific statistics
        agent_stats = {}
        for agent_type, checkpoint in self.agent_validation_configs.items():
            agent_stats[agent_type.value] = {
                'validation_level': checkpoint.validation_level.value,
                'confidence_threshold': checkpoint.confidence_threshold,
                'hallucination_threshold': checkpoint.hallucination_threshold,
                'max_iterations': checkpoint.max_iterations
            }
        
        stats['agent_configurations'] = agent_stats
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Check validation middleware health."""
        try:
            if not self.is_initialized:
                return {
                    'status': 'unhealthy',
                    'error': 'Validation middleware not initialized'
                }
            
            # Check pipeline health
            pipeline_health = await self.validation_pipeline.health_check()
            
            # Calculate health metrics
            total_validations = self.validation_stats['total_validations']
            success_rate = (
                self.validation_stats['successful_validations'] / total_validations * 100
                if total_validations > 0 else 100
            )
            
            avg_processing_time = self.validation_stats['avg_processing_time_ms']
            
            # Determine overall health
            is_healthy = (
                pipeline_health.get('status') in ['healthy', 'degraded'] and
                success_rate >= 80 and  # At least 80% success rate
                avg_processing_time < 10000  # Less than 10 seconds average
            )
            
            return {
                'status': 'healthy' if is_healthy else 'degraded',
                'middleware_metrics': {
                    'total_validations': total_validations,
                    'success_rate': round(success_rate, 1),
                    'avg_processing_time_ms': round(avg_processing_time, 2),
                    'active_validations': len(self.active_validations)
                },
                'pipeline_health': pipeline_health,
                'agent_configurations_loaded': len(self.agent_validation_configs)
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def configure_agent_validation(
        self,
        agent_type: AgentType,
        checkpoint: ValidationCheckpoint
    ) -> None:
        """Configure validation checkpoint for specific agent type."""
        self.agent_validation_configs[agent_type] = checkpoint
        
        logger.info(
            f"Configured validation for {agent_type.value}",
            validation_level=checkpoint.validation_level.value,
            confidence_threshold=checkpoint.confidence_threshold,
            hallucination_threshold=checkpoint.hallucination_threshold
        )
    
    async def close(self) -> None:
        """Close validation middleware and cleanup resources."""
        if self.validation_pipeline:
            await self.validation_pipeline.close()
        
        self.is_initialized = False
        logger.info("GraphRAG validation middleware closed")


# Singleton instance
_validation_middleware_instance: Optional[GraphRAGValidationMiddleware] = None


async def get_validation_middleware() -> GraphRAGValidationMiddleware:
    """Get the global validation middleware instance."""
    global _validation_middleware_instance
    
    if not _validation_middleware_instance:
        _validation_middleware_instance = GraphRAGValidationMiddleware()
        await _validation_middleware_instance.initialize()
    
    return _validation_middleware_instance


async def validate_agent_output(
    task: AgentTask,
    agent_output: Dict[str, Any],
    workflow: WorkflowContext,
    execution_metrics: ExecutionMetrics,
    custom_checkpoint: Optional[ValidationCheckpoint] = None
) -> AgentValidationResult:
    """Convenience function for validating single agent output."""
    middleware = await get_validation_middleware()
    return await middleware.validate_agent_output(
        task, agent_output, workflow, execution_metrics, custom_checkpoint
    )


async def validate_batch_outputs(
    task_outputs: List[Tuple[AgentTask, Dict[str, Any], WorkflowContext, ExecutionMetrics]],
    custom_checkpoint: Optional[ValidationCheckpoint] = None
) -> List[AgentValidationResult]:
    """Convenience function for batch validation."""
    middleware = await get_validation_middleware()
    return await middleware.validate_batch_outputs(task_outputs, custom_checkpoint)