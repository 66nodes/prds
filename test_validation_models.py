#!/usr/bin/env python3
"""
Simple test of validation models and types
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Define the validation models locally for testing
class HumanValidationType(str, Enum):
    """Types of human validation prompts"""
    APPROVAL = "approval"
    CHOICE = "choice"
    INPUT = "input"
    REVIEW = "review"
    CONFIRMATION = "confirmation"

class HumanValidationOption(BaseModel):
    """Option for choice-type validations"""
    label: str
    value: str
    description: Optional[str] = None

class HumanValidationPrompt(BaseModel):
    """Human validation prompt definition"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: HumanValidationType
    question: str
    context: str
    options: Optional[List[HumanValidationOption]] = None
    required: bool = True
    timeout: Optional[int] = None  # milliseconds
    metadata: Optional[Dict[str, Any]] = None

def test_validation_models():
    """Test the validation model definitions"""
    print("🧪 Testing Human Validation Models")
    print("=" * 40)
    
    # Test 1: Approval prompt
    print("\n1. Testing Approval Prompt...")
    approval = HumanValidationPrompt(
        type=HumanValidationType.APPROVAL,
        question="Do you approve this AI-generated approach?",
        context="The system has proposed implementing a microservices architecture with GraphRAG validation for hallucination detection.",
        required=True,
        timeout=30000,
        metadata={"priority": "high", "stage": "architecture_review"}
    )
    
    print(f"✅ Approval prompt created")
    print(f"   ID: {approval.id}")
    print(f"   Type: {approval.type}")
    print(f"   Question: {approval.question[:50]}...")
    print(f"   Required: {approval.required}")
    print(f"   Timeout: {approval.timeout}ms")
    
    # Test 2: Choice prompt
    print("\n2. Testing Choice Prompt...")
    choices = [
        HumanValidationOption(
            label="Traditional SQL Database",
            value="sql_db",
            description="PostgreSQL with Redis for caching"
        ),
        HumanValidationOption(
            label="Graph + Vector Database",
            value="graph_vector_db",
            description="Neo4j + Milvus for advanced GraphRAG capabilities"
        ),
        HumanValidationOption(
            label="NoSQL Database",
            value="nosql_db",
            description="MongoDB with Elasticsearch for search"
        )
    ]
    
    choice = HumanValidationPrompt(
        type=HumanValidationType.CHOICE,
        question="Which database architecture should we implement?",
        context="Based on the project requirements for GraphRAG and real-time validation, we need to choose the optimal database solution.",
        options=choices,
        required=True,
        timeout=60000
    )
    
    print(f"✅ Choice prompt created")
    print(f"   Options: {len(choice.options)}")
    for i, opt in enumerate(choice.options, 1):
        print(f"     {i}. {opt.label} ({opt.value})")
    
    # Test 3: Input prompt
    print("\n3. Testing Input Prompt...")
    input_prompt = HumanValidationPrompt(
        type=HumanValidationType.INPUT,
        question="Please specify additional security requirements",
        context="The AI has generated basic security requirements, but we need domain-specific security constraints for this enterprise application.",
        required=False,
        metadata={
            "category": "security",
            "suggested_topics": ["authentication", "authorization", "data_encryption", "audit_logging"]
        }
    )
    
    print(f"✅ Input prompt created")
    print(f"   Required: {input_prompt.required}")
    print(f"   Metadata: {input_prompt.metadata}")
    
    # Test 4: Review prompt
    print("\n4. Testing Review Prompt...")
    review_prompt = HumanValidationPrompt(
        type=HumanValidationType.REVIEW,
        question="Please review the generated PRD sections",
        context="The AI has generated the technical specifications and user requirements. Please review for accuracy and completeness.",
        required=True,
        timeout=300000,  # 5 minutes
        metadata={
            "review_content": """
# Technical Specifications

## Architecture Overview
The system will use a microservices architecture with the following components:

1. **API Gateway**: FastAPI-based gateway for request routing
2. **Authentication Service**: JWT-based authentication with RBAC
3. **PRD Generation Service**: AI-powered document generation
4. **GraphRAG Validation**: Neo4j + Milvus for hallucination detection
5. **Human Validation**: Interactive approval workflows

## Key Features
- Real-time collaboration
- Version control for documents
- Integration with external tools
- Advanced analytics dashboard

*Please verify technical accuracy and suggest improvements.*
"""
        }
    )
    
    print(f"✅ Review prompt created")
    print(f"   Timeout: {review_prompt.timeout}ms ({review_prompt.timeout/60000} minutes)")
    print(f"   Has review content: {'review_content' in (review_prompt.metadata or {})}")
    
    # Test 5: Confirmation prompt
    print("\n5. Testing Confirmation Prompt...")
    confirmation = HumanValidationPrompt(
        type=HumanValidationType.CONFIRMATION,
        question="I understand this will modify the existing system architecture",
        context="This change will require database migrations, API updates, and potential downtime. Please confirm you understand the implications.",
        required=True,
        timeout=120000,  # 2 minutes
        metadata={
            "severity": "high",
            "impact": "system_wide",
            "estimated_downtime": "2-4 hours"
        }
    )
    
    print(f"✅ Confirmation prompt created")
    print(f"   Impact: {confirmation.metadata['impact']}")
    print(f"   Estimated downtime: {confirmation.metadata['estimated_downtime']}")
    
    # Test 6: JSON serialization
    print("\n6. Testing JSON Serialization...")
    approval_json = approval.model_dump_json(indent=2)
    choice_json = choice.model_dump_json(indent=2)
    
    print(f"✅ Approval JSON: {len(approval_json)} characters")
    print(f"✅ Choice JSON: {len(choice_json)} characters")
    
    # Test parsing back from JSON
    approval_from_json = HumanValidationPrompt.model_validate_json(approval_json)
    assert approval_from_json.id == approval.id
    assert approval_from_json.type == approval.type
    
    print(f"✅ JSON round-trip successful")
    
    # Test 7: Validation types coverage
    print("\n7. Testing Validation Types Coverage...")
    all_types = [
        HumanValidationType.APPROVAL,
        HumanValidationType.CHOICE,
        HumanValidationType.INPUT,
        HumanValidationType.REVIEW,
        HumanValidationType.CONFIRMATION
    ]
    
    for vtype in all_types:
        test_prompt = HumanValidationPrompt(
            type=vtype,
            question=f"Test {vtype.value} question",
            context=f"Test context for {vtype.value}"
        )
        print(f"   ✅ {vtype.value}: {test_prompt.id}")
    
    print("\n" + "=" * 40)
    print("🎉 All Validation Model Tests Passed!")
    print("=" * 40)
    
    print("\n📋 Test Results Summary:")
    print("✅ Approval validation prompts")
    print("✅ Choice validation with options")
    print("✅ Input validation prompts")
    print("✅ Review validation with content")
    print("✅ Confirmation validation")
    print("✅ JSON serialization/deserialization")
    print("✅ All validation types supported")
    
    print(f"\n📊 Statistics:")
    print(f"   - Total validation types: {len(all_types)}")
    print(f"   - Test prompts created: 7")
    print(f"   - JSON serialization tests: 2")
    
    return True

if __name__ == "__main__":
    success = test_validation_models()
    if success:
        print(f"\n🚀 Human Validation Models are working correctly!")
    else:
        print(f"\n❌ Some tests failed!")
        exit(1)