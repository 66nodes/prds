"""
Unit tests for document generation services.
"""

import pytest
import asyncio
from unittest.mock import patch, Mock, AsyncMock
from datetime import datetime
import json

from services.document.document_generator import DocumentGenerator, DocumentRequest, DocumentResponse
from services.document.wbs_generator import WBSGenerator, TaskComplexity, TaskPriority, WBSTask, WBSPhase
from services.document.resource_estimator import ResourceEstimator, SkillLevel, ResourceType
from services.document.export_service import ExportService, ExportFormat, ExportOptions


class TestDocumentGenerator:
    """Test document generator functionality."""
    
    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service."""
        with patch('services.document.document_generator.LLMService') as mock:
            mock_instance = AsyncMock()
            mock.return_value = mock_instance
            mock_instance.generate_structured_content.return_value = json.dumps({
                "overview": "Test project overview",
                "requirements": ["Requirement 1", "Requirement 2"],
                "implementation": {
                    "approach": "Agile development",
                    "phases": ["Planning", "Development", "Testing"],
                    "technologies": ["Python", "FastAPI", "React"]
                },
                "timeline": {
                    "total_duration": "12 weeks",
                    "milestones": [
                        {"name": "MVP", "date": "Week 6", "description": "Minimum viable product"}
                    ]
                }
            })
            yield mock_instance
    
    @pytest.fixture
    def mock_wbs_generator(self):
        """Mock WBS generator."""
        mock = Mock()
        mock.generate_wbs = AsyncMock()
        mock.generate_wbs.return_value = Mock(
            project_name="Test Project",
            total_estimated_hours=240.0,
            total_estimated_days=30,
            phases=[],
            critical_path=[],
            milestones=[],
            resource_summary={},
            risk_assessment={}
        )
        return mock
    
    @pytest.fixture
    def mock_resource_estimator(self):
        """Mock resource estimator."""
        mock = Mock()
        mock.estimate_resources = AsyncMock()
        mock.estimate_resources.return_value = Mock(
            project_name="Test Project",
            team_composition=Mock(
                total_team_size=5,
                roles={"Developer": 3, "QA": 1, "Manager": 1},
                estimated_monthly_cost=50000
            ),
            cost_estimate=Mock(total_cost=200000),
            timeline_estimate=Mock(total_duration_days=90)
        )
        return mock
    
    @pytest.fixture
    def mock_export_service(self):
        """Mock export service."""
        mock = Mock()
        mock.export_document = AsyncMock()
        mock.export_document.return_value = "/tmp/test_document.json"
        return mock
    
    @pytest.mark.asyncio
    async def test_document_generator_initialization(self, mock_llm_service):
        """Test document generator initialization."""
        generator = DocumentGenerator()
        
        # Test initialization
        await generator.initialize()
        
        assert generator.llm_service is not None
        assert generator.wbs_generator is not None
        assert generator.resource_estimator is not None
        assert generator.export_service is not None
    
    @pytest.mark.asyncio
    async def test_generate_document_basic(self, mock_llm_service, mock_wbs_generator, mock_resource_estimator, mock_export_service):
        """Test basic document generation."""
        generator = DocumentGenerator()
        generator.wbs_generator = mock_wbs_generator
        generator.resource_estimator = mock_resource_estimator
        generator.export_service = mock_export_service
        
        # Initialize mock query engine
        generator._query_engine = Mock()
        
        request = DocumentRequest(
            title="Test Document",
            document_type="prd",
            context="Test context",
            sections=["overview", "requirements"],
            export_formats=[ExportFormat.JSON],
            include_wbs=True,
            include_estimates=True
        )
        
        # Generate document
        result = await generator.generate_document(request)
        
        # Verify result structure
        assert isinstance(result, DocumentResponse)
        assert result.title == "Test Document"
        assert result.document_type == "prd"
        assert "overview" in result.content
        assert "requirements" in result.content
        assert result.generation_time_ms > 0
        
        # Verify WBS and estimates were generated
        mock_wbs_generator.generate_wbs.assert_called_once()
        mock_resource_estimator.estimate_resources.assert_called_once()
        mock_export_service.export_document.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_document_without_wbs_estimates(self, mock_llm_service):
        """Test document generation without WBS and estimates."""
        generator = DocumentGenerator()
        generator._query_engine = Mock()
        
        request = DocumentRequest(
            title="Simple Document",
            document_type="prd",
            include_wbs=False,
            include_estimates=False,
            export_formats=[ExportFormat.JSON]
        )
        
        result = await generator.generate_document(request)
        
        assert result.wbs is None
        assert result.estimates is None
    
    @pytest.mark.asyncio
    async def test_content_generation_with_structured_response(self, mock_llm_service):
        """Test content generation with properly structured LLM response."""
        generator = DocumentGenerator()
        generator._query_engine = Mock()
        
        request = DocumentRequest(
            title="Structured Test",
            document_type="technical_spec",
            sections=["overview", "architecture"]
        )
        
        content = await generator._generate_content(request)
        
        assert isinstance(content, dict)
        assert "overview" in content
        assert "requirements" in content
        assert "implementation" in content
    
    @pytest.mark.asyncio
    async def test_content_parsing_fallback(self, mock_llm_service):
        """Test content parsing when JSON parsing fails."""
        generator = DocumentGenerator()
        generator._query_engine = Mock()
        
        # Mock LLM to return non-JSON content
        mock_llm_service.generate_structured_content.return_value = "This is plain text content"
        
        request = DocumentRequest(
            title="Fallback Test",
            document_type="prd"
        )
        
        content = await generator._generate_content(request)
        
        # Should fall back to structured format
        assert isinstance(content, dict)
        assert "overview" in content
        assert "raw_content" in content
        assert content["raw_content"] == "This is plain text content"
    
    @pytest.mark.asyncio
    async def test_get_document_templates(self):
        """Test getting document templates."""
        generator = DocumentGenerator()
        
        templates = await generator.get_document_templates()
        
        assert isinstance(templates, list)
        assert len(templates) > 0
        
        # Check template structure
        template = templates[0]
        assert "id" in template
        assert "name" in template
        assert "type" in template
        assert "sections" in template


class TestWBSGenerator:
    """Test WBS generator functionality."""
    
    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service for WBS generation."""
        with patch('services.document.wbs_generator.LLMService') as mock:
            mock_instance = AsyncMock()
            mock.return_value = mock_instance
            mock_instance.generate_structured_content.return_value = json.dumps([
                {
                    "id": "phase_1",
                    "name": "Planning Phase",
                    "description": "Project planning and setup",
                    "order": 1,
                    "estimated_duration_days": 14,
                    "success_criteria": ["Requirements documented", "Team assembled"]
                }
            ])
            yield mock_instance
    
    @pytest.mark.asyncio
    async def test_wbs_generation(self, mock_llm_service):
        """Test WBS generation with AI assistance."""
        generator = WBSGenerator()
        
        wbs = await generator.generate_wbs(
            title="Test Project",
            requirements=["Requirement 1", "Requirement 2"],
            context="Test context",
            project_type="software"
        )
        
        assert wbs.project_name == "Test Project"
        assert wbs.total_estimated_hours > 0
        assert wbs.total_estimated_days > 0
        assert len(wbs.phases) > 0
        assert isinstance(wbs.critical_path, list)
        assert isinstance(wbs.milestones, list)
        assert isinstance(wbs.resource_summary, dict)
        assert isinstance(wbs.risk_assessment, dict)
    
    @pytest.mark.asyncio
    async def test_phase_generation_fallback(self, mock_llm_service):
        """Test phase generation with fallback to defaults."""
        generator = WBSGenerator()
        
        # Mock LLM to return invalid JSON
        mock_llm_service.generate_structured_content.return_value = "Invalid JSON"
        
        wbs = await generator.generate_wbs(
            title="Fallback Test",
            requirements=["Req 1"],
            project_type="software"
        )
        
        # Should use default phases
        assert len(wbs.phases) == 5  # Default software phases
        phase_names = [phase.name for phase in wbs.phases]
        assert "Planning & Analysis" in phase_names
        assert "Development" in phase_names
    
    @pytest.mark.asyncio
    async def test_task_generation_for_phase(self, mock_llm_service):
        """Test task generation for individual phases."""
        generator = WBSGenerator()
        
        # Mock task generation response
        mock_llm_service.generate_structured_content.return_value = json.dumps([
            {
                "id": "task_1",
                "name": "Setup Development Environment",
                "description": "Configure development tools and environment",
                "complexity": "moderate",
                "priority": "high",
                "estimated_hours": 8,
                "skills_required": ["DevOps", "Configuration"],
                "deliverables": ["Dev environment setup"],
                "risk_level": "low"
            }
        ])
        
        # Create test phase
        phase = WBSPhase(
            id="phase_1",
            name="Setup Phase",
            description="Project setup",
            order=1,
            estimated_duration_days=5
        )
        
        tasks = await generator._generate_phase_tasks(
            phase=phase,
            requirements=["Setup requirement"],
            context="Test context"
        )
        
        assert len(tasks) > 0
        task = tasks[0]
        assert isinstance(task, WBSTask)
        assert task.name == "Setup Development Environment"
        assert task.complexity == TaskComplexity.MODERATE
        assert task.priority == TaskPriority.HIGH
    
    def test_complexity_multipliers(self):
        """Test complexity multipliers for estimation."""
        generator = WBSGenerator()
        
        assert generator.complexity_multipliers[TaskComplexity.SIMPLE] == 0.8
        assert generator.complexity_multipliers[TaskComplexity.MODERATE] == 1.0
        assert generator.complexity_multipliers[TaskComplexity.COMPLEX] == 1.5
        assert generator.complexity_multipliers[TaskComplexity.EXPERT] == 2.0
    
    def test_critical_path_identification(self):
        """Test critical path identification algorithm."""
        generator = WBSGenerator()
        
        # Create test phases with tasks
        tasks1 = [
            WBSTask(id="t1", name="Task 1", description="", estimated_hours=8, phase="p1"),
            WBSTask(id="t2", name="Task 2", description="", estimated_hours=16, phase="p1")
        ]
        tasks2 = [
            WBSTask(id="t3", name="Task 3", description="", estimated_hours=12, phase="p2")
        ]
        
        phases = [
            WBSPhase(id="p1", name="Phase 1", description="", order=1, estimated_duration_days=5, tasks=tasks1),
            WBSPhase(id="p2", name="Phase 2", description="", order=2, estimated_duration_days=3, tasks=tasks2)
        ]
        
        critical_path = generator._identify_critical_path(phases)
        
        # Should identify longest task in each phase
        assert len(critical_path) == 2
        assert "t2" in critical_path  # Longest task in phase 1 (16 hours)
        assert "t3" in critical_path  # Only task in phase 2


class TestResourceEstimator:
    """Test resource estimator functionality."""
    
    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service for resource estimation."""
        with patch('services.document.resource_estimator.LLMService') as mock:
            mock_instance = AsyncMock()
            mock.return_value = mock_instance
            mock_instance.generate_structured_content.return_value = json.dumps({
                "roles": {
                    "Project Manager": 1,
                    "Senior Developer": 2,
                    "Developer": 3,
                    "QA Engineer": 1
                },
                "skill_distribution": {
                    "intermediate": 3,
                    "senior": 4
                }
            })
            yield mock_instance
    
    @pytest.mark.asyncio
    async def test_resource_estimation(self, mock_llm_service):
        """Test comprehensive resource estimation."""
        estimator = ResourceEstimator()
        
        content = {
            "overview": "Test project",
            "requirements": ["Req 1", "Req 2"],
            "implementation": {
                "technologies": ["Python", "React", "PostgreSQL"]
            }
        }
        
        estimation = await estimator.estimate_resources(
            content=content,
            wbs=None,
            project_context={"remote_team": True}
        )
        
        assert estimation.project_name == "Test project"
        assert estimation.team_composition.total_team_size > 0
        assert estimation.cost_estimate.total_cost > 0
        assert estimation.timeline_estimate.total_duration_days > 0
        assert len(estimation.resource_requirements) > 0
        assert len(estimation.assumptions) > 0
        assert len(estimation.recommendations) > 0
    
    def test_project_complexity_assessment(self):
        """Test project complexity scoring."""
        estimator = ResourceEstimator()
        
        # Simple project
        simple_content = {
            "requirements": ["Simple req"],
            "implementation": {"technologies": ["Python"]}
        }
        simple_score = estimator._assess_project_complexity(simple_content, None)
        assert 0 <= simple_score <= 1
        
        # Complex project
        complex_content = {
            "requirements": [f"Requirement {i}" for i in range(25)],  # Many requirements
            "implementation": {
                "technologies": ["Python", "React", "PostgreSQL", "Redis", "Docker", "Kubernetes"]  # Many technologies
            },
            "overview": "enterprise scalable distributed real-time machine learning system"  # Complex keywords
        }
        complex_score = estimator._assess_project_complexity(complex_content, None)
        assert complex_score > simple_score
    
    def test_project_scale_determination(self):
        """Test project scale categorization."""
        estimator = ResourceEstimator()
        
        # Test different complexity levels
        assert estimator._determine_project_scale(0.2, None) == "small_project"
        assert estimator._determine_project_scale(0.5, None) == "medium_project"
        assert estimator._determine_project_scale(0.7, None) == "large_project"
        assert estimator._determine_project_scale(0.9, None) == "enterprise_project"
    
    def test_team_cost_calculation(self):
        """Test team cost calculation."""
        estimator = ResourceEstimator()
        
        roles = {
            "Junior Developer": 2,
            "Senior Developer": 1,
            "Project Manager": 1
        }
        
        monthly_cost = estimator._calculate_team_cost(roles)
        
        # Should calculate based on hourly rates and hours per month
        expected_cost = (
            2 * estimator.hourly_rates[SkillLevel.JUNIOR] * 160 +  # 2 junior devs
            1 * estimator.hourly_rates[SkillLevel.SENIOR] * 160 +  # 1 senior dev  
            1 * estimator.hourly_rates[SkillLevel.SENIOR] * 160    # 1 PM (maps to senior)
        )
        
        assert monthly_cost == expected_cost
    
    def test_risk_identification(self):
        """Test resource risk identification."""
        estimator = ResourceEstimator()
        
        # Create test data with risk factors
        team_composition = Mock()
        team_composition.total_team_size = 15  # Large team
        team_composition.skill_distribution = {"expert": 1}  # Limited experts
        
        wbs = Mock()
        wbs.phases = []
        wbs.total_estimated_days = 200  # Long project
        
        risks = estimator._identify_resource_risks({}, wbs, team_composition)
        
        assert isinstance(risks, list)
        risk_types = [risk["type"] for risk in risks]
        assert "team_size" in risk_types
        assert "skill_dependency" in risk_types
        assert "timeline" in risk_types


class TestExportService:
    """Test document export functionality."""
    
    @pytest.fixture
    def mock_document(self):
        """Mock document response for export testing."""
        return Mock(
            id="test_doc_123",
            title="Test Document",
            document_type="prd",
            content={
                "overview": "Test overview",
                "requirements": ["Req 1", "Req 2"],
                "implementation": {
                    "approach": "Agile",
                    "technologies": ["Python", "React"]
                }
            },
            metadata={"version": "1.0"},
            wbs=None,
            estimates=None,
            created_at=datetime.now(),
            generation_time_ms=1500
        )
    
    @pytest.mark.asyncio
    async def test_json_export(self, mock_document, tmp_path):
        """Test JSON export functionality."""
        export_service = ExportService()
        export_service.export_dir = tmp_path
        
        file_path = await export_service.export_document(
            document=mock_document,
            format=ExportFormat.JSON
        )
        
        # Verify file was created
        assert file_path
        exported_file = tmp_path / file_path.split('/')[-1]
        assert exported_file.exists()
        
        # Verify content
        with open(exported_file, 'r') as f:
            data = json.load(f)
        
        assert data["id"] == "test_doc_123"
        assert data["title"] == "Test Document"
        assert "content" in data
    
    @pytest.mark.asyncio
    async def test_markdown_export(self, mock_document, tmp_path):
        """Test Markdown export functionality."""
        export_service = ExportService()
        export_service.export_dir = tmp_path
        
        file_path = await export_service.export_document(
            document=mock_document,
            format=ExportFormat.MARKDOWN
        )
        
        # Verify file was created
        assert file_path
        exported_file = tmp_path / file_path.split('/')[-1]
        assert exported_file.exists()
        
        # Verify content structure
        with open(exported_file, 'r') as f:
            content = f.read()
        
        assert "# Test Document" in content
        assert "## Overview" in content
        assert "## Requirements" in content
    
    @pytest.mark.asyncio 
    async def test_html_export(self, mock_document, tmp_path):
        """Test HTML export functionality."""
        export_service = ExportService()
        export_service.export_dir = tmp_path
        
        file_path = await export_service.export_document(
            document=mock_document,
            format=ExportFormat.HTML,
            options=ExportOptions(
                format=ExportFormat.HTML,
                template_style="professional"
            )
        )
        
        # Verify file was created
        assert file_path
        exported_file = tmp_path / file_path.split('/')[-1]
        assert exported_file.exists()
        
        # Verify HTML structure
        with open(exported_file, 'r') as f:
            content = f.read()
        
        assert "<!DOCTYPE html>" in content
        assert "<title>Test Document</title>" in content
        assert "Test overview" in content
    
    @pytest.mark.asyncio
    async def test_export_with_options(self, mock_document, tmp_path):
        """Test export with custom options."""
        export_service = ExportService()
        export_service.export_dir = tmp_path
        
        options = ExportOptions(
            format=ExportFormat.JSON,
            include_metadata=True,
            include_wbs=False,
            include_estimates=False
        )
        
        file_path = await export_service.export_document(
            document=mock_document,
            format=ExportFormat.JSON,
            options=options
        )
        
        # Verify options were applied
        exported_file = tmp_path / file_path.split('/')[-1]
        with open(exported_file, 'r') as f:
            data = json.load(f)
        
        assert "metadata" in data  # Metadata included
        assert "wbs" not in data    # WBS excluded
        assert "estimates" not in data  # Estimates excluded
    
    def test_markdown_content_generation(self, mock_document):
        """Test Markdown content generation."""
        export_service = ExportService()
        
        options = ExportOptions(
            format=ExportFormat.MARKDOWN,
            include_metadata=True
        )
        
        markdown_content = export_service._generate_markdown_content(mock_document, options)
        
        # Verify structure
        lines = markdown_content.split('\n')
        assert "# Test Document" in lines
        assert any("## Overview" in line for line in lines)
        assert any("## Requirements" in line for line in lines)
        assert any("Test overview" in line for line in lines)
    
    def test_wbs_serialization(self):
        """Test WBS serialization for export."""
        export_service = ExportService()
        
        # Create mock WBS
        mock_wbs = Mock()
        mock_wbs.project_name = "Test Project"
        mock_wbs.total_estimated_hours = 240.0
        mock_wbs.total_estimated_days = 30
        mock_wbs.phases = []
        mock_wbs.critical_path = ["task_1", "task_2"]
        mock_wbs.milestones = []
        mock_wbs.resource_summary = {"total_tasks": 10}
        mock_wbs.risk_assessment = {"risk_level": "medium"}
        
        serialized = export_service._serialize_wbs(mock_wbs)
        
        assert isinstance(serialized, dict)
        assert serialized["project_name"] == "Test Project"
        assert serialized["total_estimated_hours"] == 240.0
        assert serialized["critical_path"] == ["task_1", "task_2"]