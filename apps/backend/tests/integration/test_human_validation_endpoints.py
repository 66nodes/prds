"""
Integration tests for Human Validation API endpoints
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from main import app
from services.human_in_the_loop import HumanValidationType


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_auth_user():
    """Mock authenticated user"""
    return Mock(id="test_user_123", email="testuser@company.com")


@pytest.fixture
def mock_validation_service():
    """Mock validation service"""
    service = Mock()
    service.request_human_validation = AsyncMock(return_value="test_validation_id")
    service.submit_validation_response = AsyncMock(return_value=True)
    service.cancel_validation_request = AsyncMock(return_value=True)
    service.get_validation_history = AsyncMock(return_value=[])
    service.get_active_validations = AsyncMock(return_value=[])
    service.cleanup_expired_requests = Mock()
    return service


class TestHumanValidationEndpoints:
    """Test human validation API endpoints"""
    
    @patch('apps.backend.api.endpoints.human_validation.get_current_user')
    @patch('apps.backend.api.endpoints.human_validation.get_validation_service')
    def test_request_validation_approval(self, mock_get_service, mock_get_user, client, mock_auth_user, mock_validation_service):
        """Test requesting approval validation"""
        mock_get_user.return_value = mock_auth_user
        mock_get_service.return_value = mock_validation_service
        
        request_data = {
            "conversation_id": "test_conversation",
            "validation_type": "approval",
            "question": "Should we proceed with this approach?",
            "context": "We're implementing a new feature",
            "required": True,
            "timeout_ms": 30000
        }
        
        response = client.post("/human-validation/request", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["validation_id"] == "test_validation_id"
        assert data["status"] == "requested"
        
        # Verify service was called correctly
        mock_validation_service.request_human_validation.assert_called_once()
        call_args = mock_validation_service.request_human_validation.call_args
        assert call_args[1]["conversation_id"] == "test_conversation"
        assert call_args[1]["user_id"] == "test_user_123"
    
    @patch('apps.backend.api.endpoints.human_validation.get_current_user')
    @patch('apps.backend.api.endpoints.human_validation.get_validation_service')
    def test_request_validation_choice(self, mock_get_service, mock_get_user, client, mock_auth_user, mock_validation_service):
        """Test requesting choice validation"""
        mock_get_user.return_value = mock_auth_user
        mock_get_service.return_value = mock_validation_service
        
        request_data = {
            "conversation_id": "test_conversation",
            "validation_type": "choice",
            "question": "Which approach should we use?",
            "context": "We have multiple options",
            "options": [
                {"label": "Option A", "value": "a", "description": "First option"},
                {"label": "Option B", "value": "b", "description": "Second option"}
            ],
            "required": True,
            "timeout_ms": 60000
        }
        
        response = client.post("/human-validation/request", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["validation_id"] == "test_validation_id"
        
        # Verify prompt was created with options
        call_args = mock_validation_service.request_human_validation.call_args
        prompt = call_args[1]["prompt"]
        assert prompt.type == HumanValidationType.CHOICE
        assert len(prompt.options) == 2
        assert prompt.options[0].label == "Option A"
    
    @patch('apps.backend.api.endpoints.human_validation.get_current_user')
    @patch('apps.backend.api.endpoints.human_validation.get_validation_service')
    def test_submit_validation_response(self, mock_get_service, mock_get_user, client, mock_auth_user, mock_validation_service):
        """Test submitting validation response"""
        mock_get_user.return_value = mock_auth_user
        mock_get_service.return_value = mock_validation_service
        
        response_data = {
            "validation_id": "test_validation_id",
            "response": {"feedback": "Looks good to me"},
            "approved": True,
            "feedback": "I approve this approach"
        }
        
        response = client.post("/human-validation/respond", json=response_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["validation_id"] == "test_validation_id"
        assert data["status"] == "completed"
        
        # Verify service was called
        mock_validation_service.submit_validation_response.assert_called_once_with(
            validation_id="test_validation_id",
            user_id="test_user_123",
            response={"feedback": "Looks good to me"},
            approved=True,
            feedback="I approve this approach"
        )
    
    @patch('apps.backend.api.endpoints.human_validation.get_current_user')
    @patch('apps.backend.api.endpoints.human_validation.get_validation_service')
    def test_submit_validation_response_failure(self, mock_get_service, mock_get_user, client, mock_auth_user, mock_validation_service):
        """Test submitting validation response when service returns False"""
        mock_get_user.return_value = mock_auth_user
        mock_get_service.return_value = mock_validation_service
        mock_validation_service.submit_validation_response.return_value = False
        
        response_data = {
            "validation_id": "invalid_validation_id",
            "response": {"feedback": "test"},
            "approved": False
        }
        
        response = client.post("/human-validation/respond", json=response_data)
        
        assert response.status_code == 400
        assert "Failed to submit validation response" in response.json()["detail"]
    
    @patch('apps.backend.api.endpoints.human_validation.get_current_user')
    @patch('apps.backend.api.endpoints.human_validation.get_validation_service')
    def test_cancel_validation(self, mock_get_service, mock_get_user, client, mock_auth_user, mock_validation_service):
        """Test cancelling validation request"""
        mock_get_user.return_value = mock_auth_user
        mock_get_service.return_value = mock_validation_service
        
        response = client.delete("/human-validation/test_validation_id")
        
        assert response.status_code == 200
        data = response.json()
        assert data["validation_id"] == "test_validation_id"
        assert data["status"] == "cancelled"
        
        # Verify service was called
        mock_validation_service.cancel_validation_request.assert_called_once_with(
            validation_id="test_validation_id",
            user_id="test_user_123"
        )
    
    @patch('apps.backend.api.endpoints.human_validation.get_current_user')
    @patch('apps.backend.api.endpoints.human_validation.get_validation_service')
    def test_get_validation_history(self, mock_get_service, mock_get_user, client, mock_auth_user, mock_validation_service):
        """Test getting validation history"""
        mock_get_user.return_value = mock_auth_user
        mock_get_service.return_value = mock_validation_service
        
        # Mock history data
        mock_validation_service.get_validation_history.return_value = [
            {
                "id": "validation_1",
                "type": "approval",
                "conversation_id": "conv_1",
                "user_id": "test_user_123",
                "request": {"question": "Test question"},
                "response": {"approved": True},
                "status": "completed",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:01:00",
                "expires_at": None
            }
        ]
        
        response = client.get("/human-validation/history")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "validation_1"
        assert data[0]["type"] == "approval"
        
        # Test with filters
        response = client.get("/human-validation/history?conversation_id=conv_1&limit=10")
        assert response.status_code == 200
        
        # Verify service was called with filters
        mock_validation_service.get_validation_history.assert_called_with(
            conversation_id="conv_1",
            user_id="test_user_123",
            limit=10
        )
    
    @patch('apps.backend.api.endpoints.human_validation.get_current_user')
    @patch('apps.backend.api.endpoints.human_validation.get_validation_service')
    def test_get_active_validations(self, mock_get_service, mock_get_user, client, mock_auth_user, mock_validation_service):
        """Test getting active validations"""
        mock_get_user.return_value = mock_auth_user
        mock_get_service.return_value = mock_validation_service
        
        # Mock active validations data
        mock_validation_service.get_active_validations.return_value = [
            {
                "id": "active_validation_1",
                "prompt": {
                    "type": "approval",
                    "question": "Active validation question",
                    "context": "Active context"
                },
                "conversation_id": "conv_1",
                "created_at": "2024-01-01T00:00:00"
            }
        ]
        
        response = client.get("/human-validation/active")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "active_validation_1"
        assert data[0]["conversation_id"] == "conv_1"
        
        # Verify service was called
        mock_validation_service.get_active_validations.assert_called_once_with(
            user_id="test_user_123"
        )
    
    @patch('apps.backend.api.endpoints.human_validation.get_current_user')
    @patch('apps.backend.api.endpoints.human_validation.get_validation_service')
    def test_cleanup_expired_validations(self, mock_get_service, mock_get_user, client, mock_auth_user, mock_validation_service):
        """Test cleanup endpoint"""
        mock_get_user.return_value = mock_auth_user
        mock_get_service.return_value = mock_validation_service
        
        response = client.post("/human-validation/cleanup")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "scheduled"
        assert "cleanup" in data["message"].lower()
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/human-validation/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "human-in-the-loop-validation"
        assert "timestamp" in data
    
    @patch('apps.backend.api.endpoints.human_validation.get_current_user')
    @patch('apps.backend.api.endpoints.human_validation.get_validation_service')
    def test_test_approval_endpoint(self, mock_get_service, mock_get_user, client, mock_auth_user, mock_validation_service):
        """Test the test approval endpoint"""
        mock_get_user.return_value = mock_auth_user
        mock_get_service.return_value = mock_validation_service
        
        with patch('apps.backend.api.endpoints.human_validation.request_approval') as mock_request_approval:
            mock_request_approval.return_value = True
            
            response = client.post(
                "/human-validation/test/approval",
                params={
                    "conversation_id": "test_conv",
                    "question": "Test approval?",
                    "context": "Test context"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["approved"] is True
            assert "Approved" in data["message"]
    
    @patch('apps.backend.api.endpoints.human_validation.get_current_user')
    @patch('apps.backend.api.endpoints.human_validation.get_validation_service')
    def test_test_choice_endpoint(self, mock_get_service, mock_get_user, client, mock_auth_user, mock_validation_service):
        """Test the test choice endpoint"""
        mock_get_user.return_value = mock_auth_user
        mock_get_service.return_value = mock_validation_service
        
        with patch('apps.backend.api.endpoints.human_validation.request_choice') as mock_request_choice:
            mock_request_choice.return_value = "option_a"
            
            request_data = {
                "conversation_id": "test_conv",
                "question": "Choose option?",
                "context": "Test context",
                "options": [
                    {"label": "Option A", "value": "option_a"},
                    {"label": "Option B", "value": "option_b"}
                ]
            }
            
            response = client.post("/human-validation/test/choice", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["choice"] == "option_a"
            assert "option_a" in data["message"]
    
    @patch('apps.backend.api.endpoints.human_validation.get_current_user')
    @patch('apps.backend.api.endpoints.human_validation.get_db_session')
    def test_get_validation_details(self, mock_get_db, mock_get_user, client, mock_auth_user):
        """Test getting validation details"""
        mock_get_user.return_value = mock_auth_user
        
        # Mock database session and query
        mock_db = Mock()
        mock_get_db.return_value = mock_db
        
        mock_validation_event = Mock()
        mock_validation_event.id = "test_validation_id"
        mock_validation_event.type = "approval"
        mock_validation_event.conversation_id = "conv_1"
        mock_validation_event.user_id = "test_user_123"
        mock_validation_event.request_data = {"question": "Test question"}
        mock_validation_event.response_data = {"approved": True}
        mock_validation_event.status = "completed"
        mock_validation_event.created_at = Mock()
        mock_validation_event.created_at.isoformat.return_value = "2024-01-01T00:00:00"
        mock_validation_event.updated_at = Mock()
        mock_validation_event.updated_at.isoformat.return_value = "2024-01-01T00:01:00"
        mock_validation_event.expires_at = None
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_validation_event
        
        response = client.get("/human-validation/test_validation_id")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test_validation_id"
        assert data["type"] == "approval"
        assert data["status"] == "completed"
    
    @patch('apps.backend.api.endpoints.human_validation.get_current_user')
    @patch('apps.backend.api.endpoints.human_validation.get_db_session')
    def test_get_validation_details_not_found(self, mock_get_db, mock_get_user, client, mock_auth_user):
        """Test getting validation details when not found"""
        mock_get_user.return_value = mock_auth_user
        
        # Mock database session to return None
        mock_db = Mock()
        mock_get_db.return_value = mock_db
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        response = client.get("/human-validation/nonexistent_validation_id")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestValidationRequestValidation:
    """Test request validation"""
    
    @patch('apps.backend.api.endpoints.human_validation.get_current_user')
    @patch('apps.backend.api.endpoints.human_validation.get_validation_service')
    def test_invalid_validation_type(self, mock_get_service, mock_get_user, client, mock_auth_user, mock_validation_service):
        """Test request with invalid validation type"""
        mock_get_user.return_value = mock_auth_user
        mock_get_service.return_value = mock_validation_service
        
        request_data = {
            "conversation_id": "test_conversation",
            "validation_type": "invalid_type",  # Invalid type
            "question": "Test question?",
            "context": "Test context"
        }
        
        response = client.post("/human-validation/request", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    @patch('apps.backend.api.endpoints.human_validation.get_current_user')
    @patch('apps.backend.api.endpoints.human_validation.get_validation_service')
    def test_missing_required_fields(self, mock_get_service, mock_get_user, client, mock_auth_user, mock_validation_service):
        """Test request with missing required fields"""
        mock_get_user.return_value = mock_auth_user
        mock_get_service.return_value = mock_validation_service
        
        request_data = {
            "conversation_id": "test_conversation",
            # Missing validation_type, question, context
        }
        
        response = client.post("/human-validation/request", json=request_data)
        
        assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])