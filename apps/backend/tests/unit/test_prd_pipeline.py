"""
Unit tests for PRD Pipeline service.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import json

from services.prd_pipeline import PRDPipeline, PRDRequest, PRDResponse, PRDStatus
from services.pydantic_agents.draft_agent import DraftAgent
from services.pydantic_agents.judge_agent import JudgeAgent
from services.graphrag.hallucination_detector import HallucinationDetector


class TestPRDPipeline:
    """Test suite for PRD Pipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create a PRDPipeline instance for testing."""
        with patch('services.prd_pipeline.DraftAgent'), \
             patch('services.prd_pipeline.JudgeAgent'), \
             patch('services.prd_pipeline.HallucinationDetector'):
            return PRDPipeline()

    @pytest.fixture
    def sample_prd_request(self):
        """Sample PRD request for testing."""
        return PRDRequest(
            title="AI-Powered Task Management System",
            description="A comprehensive task management system that uses AI to prioritize and organize tasks",
            project_id="project-123",
            requirements=[
                "User authentication and authorization",
                "AI-powered task prioritization", 
                "Real-time collaboration features",
                "Mobile app support"
            ],
            constraints=[
                "Must comply with GDPR regulations",
                "Response time under 200ms",
                "Support 10,000 concurrent users"
            ],
            target_audience="Small to medium-sized businesses",
            success_metrics=[
                "User engagement increased by 30%",
                "Task completion rate improved by 25%", 
                "User satisfaction score > 4.5/5"
            ]
        )

    @pytest.fixture
    def mock_generated_prd(self):
        """Mock generated PRD content."""
        return """
        # AI-Powered Task Management System

        ## Overview
        This document outlines the requirements for an AI-powered task management system 
        designed to help small to medium-sized businesses improve productivity and task organization.

        ## Features
        - User authentication and role-based access control
        - AI-powered task prioritization algorithm
        - Real-time collaboration workspace
        - Cross-platform mobile applications
        - Advanced analytics and reporting

        ## Technical Requirements
        - RESTful API architecture
        - PostgreSQL database
        - Redis for caching
        - WebSocket for real-time features
        - JWT for authentication

        ## Success Metrics
        - User engagement increased by 30%
        - Task completion rate improved by 25%
        - User satisfaction score > 4.5/5
        """

    @pytest.fixture
    def mock_validation_result(self):
        """Mock validation result."""
        return {
            "hallucination_rate": 0.015,
            "validation_score": 0.985,
            "is_valid": True,
            "graph_evidence": [
                {"node_id": "concept_1", "confidence": 0.95},
                {"node_id": "concept_2", "confidence": 0.88}
            ],
            "issues": []
        }

    @pytest.mark.asyncio
    async def test_generate_prd_success(self, pipeline, sample_prd_request, mock_generated_prd, mock_validation_result):
        """Test successful PRD generation."""
        with patch.object(pipeline.draft_agent, 'generate_prd', new_callable=AsyncMock) as mock_draft, \
             patch.object(pipeline.judge_agent, 'evaluate_prd', new_callable=AsyncMock) as mock_judge, \
             patch.object(pipeline.hallucination_detector, 'validate_content', new_callable=AsyncMock) as mock_validate, \
             patch.object(pipeline, '_save_prd', new_callable=AsyncMock) as mock_save:
            
            # Mock draft generation
            mock_draft.return_value = mock_generated_prd
            
            # Mock judge evaluation
            mock_judge.return_value = {
                "quality_score": 0.92,
                "completeness_score": 0.88,
                "clarity_score": 0.94,
                "overall_score": 0.91,
                "feedback": "Well-structured PRD with clear requirements",
                "suggestions": ["Add more technical details", "Include risk assessment"]
            }
            
            # Mock hallucination validation
            mock_validate.return_value = type('ValidationResult', (), mock_validation_result)()
            
            # Mock PRD saving
            prd_id = "prd-456"
            mock_save.return_value = prd_id
            
            result = await pipeline.generate_prd(sample_prd_request)
            
            assert isinstance(result, PRDResponse)
            assert result.id == prd_id
            assert result.title == sample_prd_request.title
            assert result.content == mock_generated_prd
            assert result.hallucination_rate == 0.015
            assert result.validation_score == 0.985
            assert result.metadata.status == PRDStatus.DRAFT
            
            # Verify all components were called
            mock_draft.assert_called_once_with(sample_prd_request)
            mock_judge.assert_called_once_with(mock_generated_prd, sample_prd_request)
            mock_validate.assert_called_once_with(mock_generated_prd, sample_prd_request.project_id)
            mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_prd_draft_failure(self, pipeline, sample_prd_request):
        """Test PRD generation when draft agent fails."""
        with patch.object(pipeline.draft_agent, 'generate_prd', new_callable=AsyncMock) as mock_draft:
            mock_draft.side_effect = Exception("Draft generation failed")
            
            with pytest.raises(Exception, match="Draft generation failed"):
                await pipeline.generate_prd(sample_prd_request)

    @pytest.mark.asyncio
    async def test_generate_prd_high_hallucination(self, pipeline, sample_prd_request, mock_generated_prd):
        """Test PRD generation with high hallucination rate."""
        with patch.object(pipeline.draft_agent, 'generate_prd', new_callable=AsyncMock) as mock_draft, \
             patch.object(pipeline.judge_agent, 'evaluate_prd', new_callable=AsyncMock) as mock_judge, \
             patch.object(pipeline.hallucination_detector, 'validate_content', new_callable=AsyncMock) as mock_validate:
            
            mock_draft.return_value = mock_generated_prd
            mock_judge.return_value = {"overall_score": 0.85}
            
            # High hallucination rate
            high_hallucination_result = {
                "hallucination_rate": 0.045,  # 4.5%
                "validation_score": 0.75,
                "is_valid": False,
                "graph_evidence": [],
                "issues": ["High hallucination detected", "Insufficient evidence"]
            }
            mock_validate.return_value = type('ValidationResult', (), high_hallucination_result)()
            
            with pytest.raises(ValueError, match="hallucination rate exceeds threshold"):
                await pipeline.generate_prd(sample_prd_request)

    @pytest.mark.asyncio
    async def test_regenerate_prd_with_feedback(self, pipeline, sample_prd_request):
        """Test PRD regeneration with feedback."""
        original_prd = "Original PRD content"
        feedback = "Add more technical details and risk assessment"
        max_iterations = 3
        
        with patch.object(pipeline.draft_agent, 'regenerate_prd', new_callable=AsyncMock) as mock_regenerate, \
             patch.object(pipeline.judge_agent, 'evaluate_prd', new_callable=AsyncMock) as mock_judge, \
             patch.object(pipeline.hallucination_detector, 'validate_content', new_callable=AsyncMock) as mock_validate:
            
            # Mock improved PRD after feedback
            improved_prd = "Improved PRD with technical details and risk assessment"
            mock_regenerate.return_value = improved_prd
            
            mock_judge.return_value = {"overall_score": 0.95}
            mock_validate.return_value = type('ValidationResult', (), {
                "hallucination_rate": 0.01,
                "validation_score": 0.99,
                "is_valid": True,
                "issues": []
            })()
            
            result = await pipeline.regenerate_prd_with_feedback(
                original_prd, sample_prd_request, feedback, max_iterations
            )
            
            assert result == improved_prd
            mock_regenerate.assert_called_once_with(original_prd, sample_prd_request, feedback)
            mock_judge.assert_called_once()
            mock_validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_prd_content(self, pipeline, mock_generated_prd, mock_validation_result):
        """Test PRD content validation."""
        project_id = "project-123"
        
        with patch.object(pipeline.hallucination_detector, 'validate_content', new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = type('ValidationResult', (), mock_validation_result)()
            
            result = await pipeline.validate_prd_content(mock_generated_prd, project_id)
            
            assert result.hallucination_rate == 0.015
            assert result.validation_score == 0.985
            assert result.is_valid
            mock_validate.assert_called_once_with(mock_generated_prd, project_id)

    @pytest.mark.asyncio
    async def test_evaluate_prd_quality(self, pipeline, mock_generated_prd, sample_prd_request):
        """Test PRD quality evaluation."""
        with patch.object(pipeline.judge_agent, 'evaluate_prd', new_callable=AsyncMock) as mock_evaluate:
            mock_evaluation = {
                "quality_score": 0.88,
                "completeness_score": 0.92,
                "clarity_score": 0.85,
                "technical_accuracy": 0.90,
                "overall_score": 0.89,
                "feedback": "Good overall structure",
                "suggestions": ["Add more examples", "Clarify technical requirements"]
            }
            mock_evaluate.return_value = mock_evaluation
            
            result = await pipeline.evaluate_prd_quality(mock_generated_prd, sample_prd_request)
            
            assert result == mock_evaluation
            assert result["overall_score"] == 0.89
            assert len(result["suggestions"]) == 2
            mock_evaluate.assert_called_once_with(mock_generated_prd, sample_prd_request)

    @pytest.mark.asyncio
    async def test_extract_requirements_from_description(self, pipeline):
        """Test requirements extraction from description."""
        description = """
        We need a system that allows users to create and manage tasks.
        The system should have user authentication and real-time notifications.
        It must support mobile devices and have offline capabilities.
        Performance should be optimized for large datasets.
        """
        
        with patch.object(pipeline, '_analyze_text_for_requirements', new_callable=AsyncMock) as mock_analyze:
            expected_requirements = [
                "Task creation and management functionality",
                "User authentication system",
                "Real-time notification system", 
                "Mobile device support",
                "Offline functionality",
                "Performance optimization for large datasets"
            ]
            mock_analyze.return_value = expected_requirements
            
            requirements = await pipeline.extract_requirements_from_description(description)
            
            assert isinstance(requirements, list)
            assert len(requirements) == 6
            assert "Task creation and management functionality" in requirements
            mock_analyze.assert_called_once_with(description)

    @pytest.mark.asyncio
    async def test_suggest_success_metrics(self, pipeline, sample_prd_request):
        """Test success metrics suggestion."""
        with patch.object(pipeline, '_generate_metrics_suggestions', new_callable=AsyncMock) as mock_suggest:
            suggested_metrics = [
                "User adoption rate > 80% within 6 months",
                "Average task completion time reduced by 30%",
                "System uptime > 99.9%",
                "Mobile app rating > 4.0 stars"
            ]
            mock_suggest.return_value = suggested_metrics
            
            metrics = await pipeline.suggest_success_metrics(sample_prd_request)
            
            assert isinstance(metrics, list)
            assert len(metrics) == 4
            assert all(">" in metric or "%" in metric for metric in metrics)  # Quantifiable metrics
            mock_suggest.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_prd_versions(self, pipeline, sample_prd_request):
        """Test generating multiple PRD versions."""
        versions_count = 3
        
        with patch.object(pipeline, 'generate_prd', new_callable=AsyncMock) as mock_generate:
            # Mock different PRD versions
            mock_versions = [
                PRDResponse(
                    id=f"prd-v{i}",
                    title=sample_prd_request.title,
                    content=f"PRD Version {i} content",
                    hallucination_rate=0.01 + i * 0.005,
                    validation_score=0.95 - i * 0.02,
                    metadata=type('Metadata', (), {"status": PRDStatus.DRAFT})(),
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                for i in range(1, versions_count + 1)
            ]
            mock_generate.side_effect = mock_versions
            
            versions = await pipeline.generate_prd_versions(sample_prd_request, versions_count)
            
            assert isinstance(versions, list)
            assert len(versions) == versions_count
            assert all(isinstance(v, PRDResponse) for v in versions)
            assert mock_generate.call_count == versions_count

    def test_calculate_complexity_score(self, pipeline, sample_prd_request):
        """Test complexity score calculation."""
        score = pipeline.calculate_complexity_score(sample_prd_request)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score > 0.3  # Should be moderately complex

    def test_estimate_generation_time(self, pipeline, sample_prd_request):
        """Test generation time estimation."""
        estimated_time = pipeline.estimate_generation_time(sample_prd_request)
        
        assert isinstance(estimated_time, int)
        assert estimated_time > 0
        assert estimated_time < 300  # Should be under 5 minutes

    @pytest.mark.asyncio
    async def test_get_prd_by_id(self, pipeline):
        """Test retrieving PRD by ID."""
        prd_id = "prd-123"
        
        with patch.object(pipeline, '_load_prd_from_storage', new_callable=AsyncMock) as mock_load:
            mock_prd = PRDResponse(
                id=prd_id,
                title="Test PRD",
                content="Test content",
                hallucination_rate=0.02,
                validation_score=0.95,
                metadata=type('Metadata', (), {"status": PRDStatus.PUBLISHED})(),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            mock_load.return_value = mock_prd
            
            result = await pipeline.get_prd_by_id(prd_id)
            
            assert result == mock_prd
            assert result.id == prd_id
            mock_load.assert_called_once_with(prd_id)

    @pytest.mark.asyncio
    async def test_update_prd_status(self, pipeline):
        """Test updating PRD status."""
        prd_id = "prd-123"
        new_status = PRDStatus.PUBLISHED
        
        with patch.object(pipeline, '_update_prd_in_storage', new_callable=AsyncMock) as mock_update:
            mock_update.return_value = True
            
            success = await pipeline.update_prd_status(prd_id, new_status)
            
            assert success is True
            mock_update.assert_called_once_with(prd_id, {"status": new_status})

    @pytest.mark.asyncio
    async def test_delete_prd(self, pipeline):
        """Test PRD deletion."""
        prd_id = "prd-123"
        
        with patch.object(pipeline, '_delete_prd_from_storage', new_callable=AsyncMock) as mock_delete:
            mock_delete.return_value = True
            
            success = await pipeline.delete_prd(prd_id)
            
            assert success is True
            mock_delete.assert_called_once_with(prd_id)

    @pytest.mark.asyncio
    async def test_export_prd_to_format(self, pipeline, mock_generated_prd):
        """Test exporting PRD to different formats."""
        formats = ["pdf", "docx", "markdown", "html"]
        
        for format_type in formats:
            with patch.object(pipeline, f'_export_to_{format_type}', new_callable=AsyncMock) as mock_export:
                mock_export.return_value = b"exported_content"
                
                result = await pipeline.export_prd_to_format(mock_generated_prd, format_type)
                
                assert result == b"exported_content"
                mock_export.assert_called_once_with(mock_generated_prd)

    def test_validate_prd_request(self, pipeline, sample_prd_request):
        """Test PRD request validation."""
        # Test valid request
        is_valid, errors = pipeline.validate_prd_request(sample_prd_request)
        assert is_valid
        assert len(errors) == 0
        
        # Test invalid request (missing title)
        invalid_request = sample_prd_request.copy()
        invalid_request.title = ""
        
        is_valid, errors = pipeline.validate_prd_request(invalid_request)
        assert not is_valid
        assert len(errors) > 0
        assert any("title" in error.lower() for error in errors)

    @pytest.mark.asyncio
    async def test_analyze_prd_quality_trends(self, pipeline):
        """Test analyzing PRD quality trends."""
        project_id = "project-123"
        
        with patch.object(pipeline, '_get_project_prds', new_callable=AsyncMock) as mock_get_prds:
            mock_prds = [
                {"created_at": datetime.utcnow(), "validation_score": 0.95, "hallucination_rate": 0.01},
                {"created_at": datetime.utcnow(), "validation_score": 0.88, "hallucination_rate": 0.02},
                {"created_at": datetime.utcnow(), "validation_score": 0.92, "hallucination_rate": 0.015}
            ]
            mock_get_prds.return_value = mock_prds
            
            trends = await pipeline.analyze_prd_quality_trends(project_id)
            
            assert isinstance(trends, dict)
            assert "average_validation_score" in trends
            assert "average_hallucination_rate" in trends
            assert "quality_trend" in trends
            assert "total_prds" in trends
            
            assert trends["total_prds"] == 3
            assert 0.85 <= trends["average_validation_score"] <= 0.95

    @pytest.mark.benchmark
    def test_prd_generation_performance(self, benchmark, pipeline, sample_prd_request):
        """Benchmark PRD generation performance."""
        async def generate_prd_sync():
            # Mock async calls for benchmarking
            with patch.object(pipeline.draft_agent, 'generate_prd', new_callable=AsyncMock) as mock_draft:
                mock_draft.return_value = "Mock PRD content"
                return await pipeline.generate_prd(sample_prd_request)
        
        import asyncio
        result = benchmark(lambda: asyncio.run(generate_prd_sync()))
        assert result is not None

    @pytest.mark.parametrize("complexity_level", ["simple", "moderate", "complex"])
    def test_complexity_based_generation(self, pipeline, complexity_level):
        """Test PRD generation with different complexity levels."""
        requirements_count = {"simple": 3, "moderate": 7, "complex": 15}[complexity_level]
        
        request = PRDRequest(
            title=f"Test {complexity_level} system",
            description=f"A {complexity_level} system description",
            project_id="test-project",
            requirements=[f"Requirement {i}" for i in range(requirements_count)]
        )
        
        complexity_score = pipeline.calculate_complexity_score(request)
        expected_ranges = {
            "simple": (0.0, 0.4),
            "moderate": (0.3, 0.7), 
            "complex": (0.6, 1.0)
        }
        
        min_score, max_score = expected_ranges[complexity_level]
        assert min_score <= complexity_score <= max_score