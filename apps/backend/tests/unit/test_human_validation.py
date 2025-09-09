"""
Unit tests for Human-in-the-Loop Validation System
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from services.human_in_the_loop import (
    HumanInTheLoopService,
    HumanValidationPrompt,
    HumanValidationType,
    HumanValidationOption,
    ValidationEvent,
    ValidationEventType,
    ValidationEventStatus
)


class TestHumanValidationPrompt:
    """Test HumanValidationPrompt model"""
    
    def test_create_approval_prompt(self):
        """Test creating an approval validation prompt"""
        prompt = HumanValidationPrompt(
            type=HumanValidationType.APPROVAL,
            question="Should we proceed with this approach?",
            context="We're about to implement a new feature that requires significant changes."
        )
        
        assert prompt.type == HumanValidationType.APPROVAL
        assert prompt.question == "Should we proceed with this approach?"
        assert prompt.context == "We're about to implement a new feature that requires significant changes."
        assert prompt.required is True
        assert prompt.timeout is None
        assert prompt.options is None
        assert len(prompt.id) > 0
    
    def test_create_choice_prompt(self):
        """Test creating a choice validation prompt"""
        options = [
            HumanValidationOption(label="Option A", value="a", description="First option"),
            HumanValidationOption(label="Option B", value="b", description="Second option")
        ]
        
        prompt = HumanValidationPrompt(
            type=HumanValidationType.CHOICE,
            question="Which approach should we use?",
            context="We have multiple implementation options",
            options=options,
            timeout=30000
        )
        
        assert prompt.type == HumanValidationType.CHOICE
        assert len(prompt.options) == 2
        assert prompt.options[0].label == "Option A"
        assert prompt.timeout == 30000
    
    def test_create_input_prompt(self):
        """Test creating an input validation prompt"""
        prompt = HumanValidationPrompt(
            type=HumanValidationType.INPUT,
            question="Please provide additional requirements",
            context="We need more details about the user interface",
            required=False,
            metadata={"category": "ui_requirements"}
        )
        
        assert prompt.type == HumanValidationType.INPUT
        assert prompt.required is False
        assert prompt.metadata["category"] == "ui_requirements"


class TestHumanInTheLoopService:
    """Test HumanInTheLoopService"""
    
    @pytest.fixture
    def mock_websocket_manager(self):
        """Mock WebSocket manager"""
        mock = Mock()
        mock.send_to_user = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_validation_pipeline(self):
        """Mock validation pipeline"""
        mock = Mock()
        mock.record_human_approval = AsyncMock()
        mock.record_human_rejection = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        mock = Mock()
        mock.add = Mock()
        mock.commit = Mock()
        mock.query = Mock()
        return mock
    
    @pytest.fixture
    def validation_service(self, mock_websocket_manager, mock_validation_pipeline, mock_db_session):
        """Create HumanInTheLoopService instance with mocks"""
        return HumanInTheLoopService(
            websocket_manager=mock_websocket_manager,
            validation_pipeline=mock_validation_pipeline,
            db_session=mock_db_session
        )
    
    @pytest.mark.asyncio
    async def test_request_human_validation(self, validation_service, mock_websocket_manager, mock_db_session):
        """Test requesting human validation"""
        prompt = HumanValidationPrompt(
            type=HumanValidationType.APPROVAL,
            question="Test validation?",
            context="Test context"
        )
        
        validation_id = await validation_service.request_human_validation(
            conversation_id="test_conversation",
            user_id="test_user",
            prompt=prompt
        )
        
        # Verify validation request was stored
        assert validation_id == prompt.id
        assert prompt.id in validation_service.active_requests
        
        # Verify database was called
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        
        # Verify WebSocket message was sent
        mock_websocket_manager.send_to_user.assert_called_once_with(
            "test_user",
            {
                "type": "human_validation_request",
                "payload": {
                    "id": prompt.id,
                    "conversation_id": "test_conversation",
                    "prompt": prompt.dict(),
                    "context": None
                }
            }
        )
    
    @pytest.mark.asyncio
    async def test_request_validation_with_timeout(self, validation_service):
        """Test requesting validation with timeout"""
        prompt = HumanValidationPrompt(
            type=HumanValidationType.CHOICE,
            question="Choose an option",
            context="Test context",
            timeout=5000  # 5 seconds
        )
        
        validation_id = await validation_service.request_human_validation(
            conversation_id="test_conversation",
            user_id="test_user",
            prompt=prompt
        )
        
        # Verify timeout task was created
        assert validation_id in validation_service.timeout_tasks
    
    @pytest.mark.asyncio
    async def test_submit_validation_response_approved(self, validation_service, mock_websocket_manager, mock_db_session, mock_validation_pipeline):
        """Test submitting approved validation response"""
        # First, create a validation request
        prompt = HumanValidationPrompt(
            type=HumanValidationType.APPROVAL,
            question="Test validation?",
            context="Test context"
        )
        
        validation_id = await validation_service.request_human_validation(
            conversation_id="test_conversation",
            user_id="test_user",
            prompt=prompt
        )
        
        # Mock database query
        mock_event = Mock()
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_event
        
        # Submit response
        success = await validation_service.submit_validation_response(
            validation_id=validation_id,
            user_id="test_user",
            response={"feedback": "Looks good"},
            approved=True
        )
        
        assert success is True
        assert validation_id not in validation_service.active_requests
        
        # Verify database was updated
        assert mock_event.status == ValidationEventStatus.COMPLETED
        assert mock_event.type == ValidationEventType.APPROVED
        mock_db_session.commit.assert_called()
        
        # Verify GraphRAG integration was called
        mock_validation_pipeline.record_human_approval.assert_called_once()
        
        # Verify WebSocket confirmation was sent
        assert mock_websocket_manager.send_to_user.call_count == 2  # Initial request + confirmation
    
    @pytest.mark.asyncio
    async def test_submit_validation_response_rejected(self, validation_service, mock_websocket_manager, mock_db_session, mock_validation_pipeline):
        """Test submitting rejected validation response"""
        # First, create a validation request
        prompt = HumanValidationPrompt(
            type=HumanValidationType.APPROVAL,
            question="Test validation?",
            context="Test context"
        )
        
        validation_id = await validation_service.request_human_validation(
            conversation_id="test_conversation",
            user_id="test_user",
            prompt=prompt
        )
        
        # Mock database query
        mock_event = Mock()
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_event
        
        # Submit response
        success = await validation_service.submit_validation_response(
            validation_id=validation_id,
            user_id="test_user",
            response={"feedback": "Needs changes"},
            approved=False,
            feedback="Please revise the approach"
        )
        
        assert success is True
        
        # Verify database was updated
        assert mock_event.status == ValidationEventStatus.COMPLETED
        assert mock_event.type == ValidationEventType.REJECTED
        
        # Verify GraphRAG rejection was recorded
        mock_validation_pipeline.record_human_rejection.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_submit_validation_response_unauthorized(self, validation_service):
        """Test submitting validation response with wrong user"""
        # Create validation request
        prompt = HumanValidationPrompt(
            type=HumanValidationType.APPROVAL,
            question="Test validation?",
            context="Test context"
        )
        
        validation_id = await validation_service.request_human_validation(
            conversation_id="test_conversation",
            user_id="correct_user",
            prompt=prompt
        )
        
        # Try to submit with different user
        success = await validation_service.submit_validation_response(
            validation_id=validation_id,
            user_id="wrong_user",
            response={"feedback": "test"},
            approved=True
        )
        
        assert success is False
        assert validation_id in validation_service.active_requests  # Request should still be active
    
    @pytest.mark.asyncio
    async def test_validation_timeout(self, validation_service, mock_websocket_manager, mock_db_session):
        """Test validation request timeout"""
        prompt = HumanValidationPrompt(
            type=HumanValidationType.APPROVAL,
            question="Test validation?",
            context="Test context",
            timeout=100  # 100ms for quick test
        )
        
        validation_id = await validation_service.request_human_validation(
            conversation_id="test_conversation",
            user_id="test_user",
            prompt=prompt
        )
        
        # Mock database query for timeout handler
        mock_event = Mock()
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_event
        
        # Wait for timeout
        await asyncio.sleep(0.2)  # Wait longer than timeout
        
        # Verify timeout was handled
        assert validation_id not in validation_service.active_requests
        assert mock_event.status == ValidationEventStatus.EXPIRED
        assert mock_event.type == ValidationEventType.TIMEOUT
    
    @pytest.mark.asyncio
    async def test_cancel_validation_request(self, validation_service, mock_db_session):
        """Test cancelling validation request"""
        # Create validation request
        prompt = HumanValidationPrompt(
            type=HumanValidationType.APPROVAL,
            question="Test validation?",
            context="Test context"
        )
        
        validation_id = await validation_service.request_human_validation(
            conversation_id="test_conversation",
            user_id="test_user",
            prompt=prompt
        )
        
        # Mock database query
        mock_event = Mock()
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_event
        
        # Cancel the request
        success = await validation_service.cancel_validation_request(
            validation_id=validation_id,
            user_id="test_user"
        )
        
        assert success is True
        assert validation_id not in validation_service.active_requests
        assert mock_event.status == ValidationEventStatus.CANCELLED
        assert mock_event.type == ValidationEventType.CANCELLED
    
    @pytest.mark.asyncio
    async def test_get_validation_history(self, validation_service, mock_db_session):
        """Test getting validation history"""
        # Mock database query results
        mock_event1 = Mock()
        mock_event1.id = "event1"
        mock_event1.type = ValidationEventType.APPROVED
        mock_event1.conversation_id = "conv1"
        mock_event1.user_id = "user1"
        mock_event1.request_data = {"test": "data"}
        mock_event1.response_data = {"approved": True}
        mock_event1.status = ValidationEventStatus.COMPLETED
        mock_event1.created_at = datetime.now()
        mock_event1.updated_at = datetime.now()
        mock_event1.expires_at = None
        
        mock_db_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [mock_event1]
        
        history = await validation_service.get_validation_history(
            conversation_id="conv1",
            user_id="user1",
            limit=10
        )
        
        assert len(history) == 1
        assert history[0]["id"] == "event1"
        assert history[0]["type"] == ValidationEventType.APPROVED
        assert history[0]["status"] == ValidationEventStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_get_active_validations(self, validation_service):
        """Test getting active validations"""
        # Create multiple validation requests
        prompt1 = HumanValidationPrompt(
            type=HumanValidationType.APPROVAL,
            question="First validation?",
            context="First context"
        )
        
        prompt2 = HumanValidationPrompt(
            type=HumanValidationType.CHOICE,
            question="Second validation?",
            context="Second context"
        )
        
        validation_id1 = await validation_service.request_human_validation(
            conversation_id="conv1",
            user_id="user1",
            prompt=prompt1
        )
        
        validation_id2 = await validation_service.request_human_validation(
            conversation_id="conv2",
            user_id="user1",
            prompt=prompt2
        )
        
        # Get active validations for user1
        active_validations = await validation_service.get_active_validations("user1")
        
        assert len(active_validations) == 2
        validation_ids = [v["id"] for v in active_validations]
        assert validation_id1 in validation_ids
        assert validation_id2 in validation_ids
        
        # Get active validations for different user
        active_validations_other = await validation_service.get_active_validations("other_user")
        assert len(active_validations_other) == 0


class TestValidationUtilityFunctions:
    """Test utility functions for validation"""
    
    @pytest.mark.asyncio
    async def test_request_approval_utility(self):
        """Test request_approval utility function"""
        from ...services.human_in_the_loop import request_approval
        
        # Mock service
        mock_service = Mock()
        mock_service.request_human_validation = AsyncMock(return_value="test_validation_id")
        mock_service.active_requests = {}  # Empty active requests to simulate completion
        
        # Call utility function (note: this would normally wait for user response)
        # For testing, we simulate immediate completion by having empty active_requests
        result = await request_approval(
            service=mock_service,
            conversation_id="test_conv",
            user_id="test_user",
            question="Test approval?",
            context="Test context",
            timeout_ms=1000  # Short timeout for test
        )
        
        # Verify service was called
        mock_service.request_human_validation.assert_called_once()
        
        # Since we mocked empty active_requests, result should be False (default)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_request_choice_utility(self):
        """Test request_choice utility function"""
        from ...services.human_in_the_loop import request_choice, HumanValidationOption
        
        # Mock service
        mock_service = Mock()
        mock_service.request_human_validation = AsyncMock(return_value="test_validation_id")
        mock_service.active_requests = {}  # Empty active requests to simulate completion
        
        options = [
            HumanValidationOption(label="Option A", value="a"),
            HumanValidationOption(label="Option B", value="b")
        ]
        
        # Call utility function
        result = await request_choice(
            service=mock_service,
            conversation_id="test_conv",
            user_id="test_user",
            question="Choose an option",
            options=options,
            context="Test context",
            timeout_ms=1000
        )
        
        # Verify service was called
        mock_service.request_human_validation.assert_called_once()
        
        # Since we mocked empty active_requests, result should be None (no choice made)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])