#!/usr/bin/env python3
"""
Simple test script to verify Human-in-the-Loop validation system works
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, AsyncMock

# Add the backend path
import sys
import os
sys.path.append('/home/jgrewal/projects/website/prds/apps/backend')

# Import our validation classes
from services.human_in_the_loop import (
    HumanValidationPrompt,
    HumanValidationType,
    HumanValidationOption,
    HumanInTheLoopService
)

async def test_validation_system():
    """Test the human validation system"""
    print("🚀 Testing Human-in-the-Loop Validation System")
    print("=" * 50)
    
    # Test 1: Create validation prompts
    print("\n1. Testing validation prompt creation...")
    
    approval_prompt = HumanValidationPrompt(
        type=HumanValidationType.APPROVAL,
        question="Do you approve this AI-generated PRD approach?",
        context="The AI has proposed a microservices architecture with GraphRAG validation. This will require significant development effort.",
        required=True,
        timeout=30000
    )
    
    choice_prompt = HumanValidationPrompt(
        type=HumanValidationType.CHOICE,
        question="Which database architecture should we use?",
        context="We need to choose the optimal database solution for our scale and requirements.",
        options=[
            HumanValidationOption(
                label="PostgreSQL + Redis",
                value="postgres_redis",
                description="Traditional relational database with caching"
            ),
            HumanValidationOption(
                label="Neo4j + Milvus",
                value="neo4j_milvus", 
                description="Graph database with vector search for GraphRAG"
            )
        ],
        required=True
    )
    
    input_prompt = HumanValidationPrompt(
        type=HumanValidationType.INPUT,
        question="Please provide additional requirements for the user authentication system",
        context="We need more details about security requirements, SSO integration, and user role management.",
        required=True
    )
    
    print(f"✅ Created approval prompt: {approval_prompt.id}")
    print(f"✅ Created choice prompt: {choice_prompt.id}")
    print(f"✅ Created input prompt: {input_prompt.id}")
    
    # Test 2: Mock service components
    print("\n2. Testing service initialization...")
    
    mock_websocket = Mock()
    mock_websocket.send_to_user = AsyncMock()
    
    mock_validation_pipeline = Mock()
    mock_validation_pipeline.record_human_approval = AsyncMock()
    mock_validation_pipeline.record_human_rejection = AsyncMock()
    
    mock_db = Mock()
    mock_db.add = Mock()
    mock_db.commit = Mock()
    mock_db.query = Mock()
    
    service = HumanInTheLoopService(
        websocket_manager=mock_websocket,
        validation_pipeline=mock_validation_pipeline,
        db_session=mock_db
    )
    
    print("✅ Service initialized successfully")
    
    # Test 3: Request validation
    print("\n3. Testing validation request...")
    
    validation_id = await service.request_human_validation(
        conversation_id="test_conversation_123",
        user_id="test_user_456",
        prompt=approval_prompt,
        step_context={"step": "architecture_approval", "priority": "high"}
    )
    
    print(f"✅ Validation requested: {validation_id}")
    print(f"   - Conversation: test_conversation_123")
    print(f"   - User: test_user_456")
    print(f"   - Type: {approval_prompt.type}")
    
    # Verify WebSocket was called
    assert mock_websocket.send_to_user.called
    call_args = mock_websocket.send_to_user.call_args
    assert call_args[0][0] == "test_user_456"  # user_id
    assert call_args[0][1]["type"] == "human_validation_request"
    
    print("✅ WebSocket notification sent")
    
    # Test 4: Submit validation response (approval)
    print("\n4. Testing approval response...")
    
    # Mock database query for response submission
    mock_event = Mock()
    mock_event.status = None
    mock_event.type = None
    mock_event.updated_at = None
    mock_db.query.return_value.filter.return_value.first.return_value = mock_event
    
    success = await service.submit_validation_response(
        validation_id=validation_id,
        user_id="test_user_456",
        response={"feedback": "This approach looks excellent! I approve the GraphRAG integration."},
        approved=True,
        feedback="Great work on the architecture design!"
    )
    
    print(f"✅ Approval response submitted: {success}")
    assert success is True
    
    # Verify GraphRAG integration was called
    assert mock_validation_pipeline.record_human_approval.called
    print("✅ GraphRAG approval recorded")
    
    # Test 5: Test choice validation
    print("\n5. Testing choice validation...")
    
    choice_validation_id = await service.request_human_validation(
        conversation_id="test_conversation_123",
        user_id="test_user_456", 
        prompt=choice_prompt
    )
    
    choice_success = await service.submit_validation_response(
        validation_id=choice_validation_id,
        user_id="test_user_456",
        response={"choice": "neo4j_milvus"},
        approved=True
    )
    
    print(f"✅ Choice validation completed: {choice_success}")
    print("   - Selected: Neo4j + Milvus")
    
    # Test 6: Test rejection
    print("\n6. Testing rejection response...")
    
    reject_prompt = HumanValidationPrompt(
        type=HumanValidationType.APPROVAL,
        question="Should we implement this complex AI feature?",
        context="The AI suggests implementing real-time sentiment analysis with 99% accuracy.",
        required=True
    )
    
    reject_validation_id = await service.request_human_validation(
        conversation_id="test_conversation_123",
        user_id="test_user_456",
        prompt=reject_prompt
    )
    
    reject_success = await service.submit_validation_response(
        validation_id=reject_validation_id,
        user_id="test_user_456",
        response={"feedback": "This is too complex for our MVP. Let's start with basic sentiment analysis."},
        approved=False,
        feedback="Needs to be simpler for the first release."
    )
    
    print(f"✅ Rejection response submitted: {reject_success}")
    
    # Verify GraphRAG rejection was recorded
    assert mock_validation_pipeline.record_human_rejection.called
    print("✅ GraphRAG rejection recorded")
    
    # Test 7: Test active validations
    print("\n7. Testing active validations...")
    
    active_validations = await service.get_active_validations("test_user_456")
    print(f"✅ Active validations retrieved: {len(active_validations)} items")
    
    # Test 8: Test unauthorized response
    print("\n8. Testing unauthorized response...")
    
    unauth_success = await service.submit_validation_response(
        validation_id=validation_id,
        user_id="wrong_user_123",  # Wrong user
        response={"test": "data"},
        approved=True
    )
    
    print(f"✅ Unauthorized response correctly rejected: {not unauth_success}")
    assert unauth_success is False
    
    print("\n" + "=" * 50)
    print("🎉 All Human Validation Tests Passed!")
    print("=" * 50)
    
    # Print summary
    print("\n📋 Test Summary:")
    print("✅ Validation prompt creation")
    print("✅ Service initialization") 
    print("✅ Validation request handling")
    print("✅ WebSocket notifications")
    print("✅ Approval responses")
    print("✅ Choice validation")
    print("✅ Rejection handling")
    print("✅ GraphRAG integration")
    print("✅ Active validation retrieval")
    print("✅ Authorization checks")
    
    print("\n🚀 Human-in-the-Loop Validation System is working correctly!")

if __name__ == "__main__":
    asyncio.run(test_validation_system())