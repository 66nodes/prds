#!/usr/bin/env python3
"""
Comment System Validation Script.

This script validates that the comment and annotation system components
are properly implemented and can work together.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from uuid import uuid4


async def validate_comment_models():
    """Validate comment data models."""
    print("🔍 Validating comment models...")
    
    try:
        from models.comments import (
            Comment, CommentCreate, CommentUpdate, CommentType, 
            CommentStatus, CommentPriority, DocumentType, 
            SelectionRange, CommentPosition, CommentThread
        )
        
        # Test comment creation model
        comment_data = CommentCreate(
            document_id=str(uuid4()),
            document_type=DocumentType.PRD,
            content="Test comment validation",
            comment_type=CommentType.COMMENT,
            priority=CommentPriority.MEDIUM,
            tags=["validation", "test"],
            mentions=[],
            assignees=[]
        )
        
        print("✅ CommentCreate model validation passed")
        
        # Test comment with text selection
        annotation_data = CommentCreate(
            document_id=str(uuid4()),
            document_type=DocumentType.PRD,
            content="This is an annotation",
            comment_type=CommentType.ANNOTATION,
            selection_range=SelectionRange(
                start_offset=10,
                end_offset=25,
                selected_text="test text",
                container_element="p"
            ),
            position=CommentPosition(x=100.0, y=200.0)
        )
        
        print("✅ Annotation with text selection validation passed")
        
        # Test complete comment model
        full_comment = Comment(
            id=str(uuid4()),
            document_id=str(uuid4()),
            document_type=DocumentType.PRD,
            author_id=str(uuid4()),
            author_name=f"User {uuid4()}",
            thread_id=str(uuid4()),
            content=f"Validation comment {uuid4()}",
            comment_type=CommentType.COMMENT,
            status=CommentStatus.OPEN,
            depth=0,
            reply_count=0,
            tags=["validation"],
            mentions=[],
            assignees=[]
        )
        
        print("✅ Complete Comment model validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Comment model validation failed: {e}")
        return False


async def validate_websocket_handler():
    """Validate WebSocket handler functionality."""
    print("\n🔍 Validating WebSocket handler...")
    
    try:
        # Mock WebSocket manager for testing
        class MockWebSocketManager:
            async def _send_to_connection(self, connection_id, message):
                return True
        
        from services.comment_websocket_handler import CommentWebSocketHandler
        
        # Create handler instance
        websocket_manager = MockWebSocketManager()
        handler = CommentWebSocketHandler(websocket_manager)
        
        # Test initialization
        await handler.initialize()
        print("✅ WebSocket handler initialization passed")
        
        # Test subscription management
        connection_id = f"connection-{uuid4()}"
        document_id = str(uuid4())
        
        await handler.subscribe_to_document(connection_id, document_id)
        await handler.unsubscribe_from_document(connection_id, document_id)
        print("✅ WebSocket subscription management passed")
        
        # Test cleanup
        await handler.shutdown()
        print("✅ WebSocket handler shutdown passed")
        
        return True
        
    except Exception as e:
        print(f"❌ WebSocket handler validation failed: {e}")
        return False


async def validate_api_structure():
    """Validate API endpoint structure."""
    print("\n🔍 Validating API endpoint structure...")
    
    try:
        # Check if API endpoints file exists and has correct structure
        api_file = Path(__file__).parent / "api" / "endpoints" / "comments.py"
        
        if not api_file.exists():
            print(f"❌ API file not found: {api_file}")
            return False
        
        # Read and validate basic structure
        with open(api_file, 'r') as f:
            content = f.read()
        
        # Check for essential endpoints
        required_endpoints = [
            "POST /comments/",  # create_comment
            "GET /comments/{comment_id}",  # get_comment
            "PUT /comments/{comment_id}",  # update_comment
            "DELETE /comments/{comment_id}",  # delete_comment
            "GET /comments/document/{document_id}",  # list_document_comments
            "GET /comments/thread/{thread_id}",  # get_comment_thread
            "POST /comments/search",  # search_comments
            "GET /comments/analytics/{document_id}"  # get_comment_analytics
        ]
        
        missing_endpoints = []
        for endpoint in required_endpoints:
            # Simple check for endpoint presence
            method, path = endpoint.split(" ", 1)
            if f'@router.{method.lower()}("{path.split("{")[0]}' not in content:
                if f'@router.{method.lower()}("' not in content or path.split("/")[-1] not in content:
                    missing_endpoints.append(endpoint)
        
        if missing_endpoints:
            print(f"❌ Missing API endpoints: {missing_endpoints}")
            return False
        
        print("✅ API endpoint structure validation passed")
        
        # Check for proper error handling
        if "HTTPException" not in content:
            print("❌ No error handling found in API endpoints")
            return False
        
        print("✅ API error handling validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ API structure validation failed: {e}")
        return False


async def validate_integration_readiness():
    """Validate that all components can work together."""
    print("\n🔍 Validating integration readiness...")
    
    try:
        # Check if all required files exist
        required_files = [
            "models/comments.py",
            "api/endpoints/comments.py", 
            "services/comment_websocket_handler.py",
            "services/websocket_manager.py"
        ]
        
        base_path = Path(__file__).parent
        missing_files = []
        
        for file_path in required_files:
            full_path = base_path / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            print(f"❌ Missing required files: {missing_files}")
            return False
        
        print("✅ All required files present")
        
        # Check if test files exist
        test_files = [
            "tests/conftest.py",
            "tests/integration/test_comment_system_integration.py"
        ]
        
        missing_test_files = []
        for file_path in test_files:
            full_path = base_path / file_path
            if not full_path.exists():
                missing_test_files.append(file_path)
        
        if missing_test_files:
            print(f"❌ Missing test files: {missing_test_files}")
            return False
        
        print("✅ All test files present")
        print("✅ Integration readiness validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration readiness validation failed: {e}")
        return False


async def validate_test_configuration():
    """Validate test configuration and setup."""
    print("\n🔍 Validating test configuration...")
    
    try:
        # Check pytest configuration
        pytest_file = Path(__file__).parent / "pytest.ini"
        if pytest_file.exists():
            print("✅ pytest.ini configuration found")
        else:
            print("⚠️ pytest.ini not found (tests will use defaults)")
        
        # Check test runner
        test_runner = Path(__file__).parent / "test_runner.py"
        if test_runner.exists():
            print("✅ Test runner script found")
        else:
            print("❌ Test runner script not found")
            return False
        
        # Verify test runner is executable
        if test_runner.is_file() and test_runner.stat().st_mode & 0o111:
            print("✅ Test runner is executable")
        else:
            print("⚠️ Test runner may not be executable")
        
        print("✅ Test configuration validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Test configuration validation failed: {e}")
        return False


def generate_validation_report(results: Dict[str, bool]) -> Dict[str, Any]:
    """Generate a comprehensive validation report."""
    
    total_checks = len(results)
    passed_checks = sum(1 for result in results.values() if result)
    
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "success_rate": (passed_checks / total_checks) * 100,
        "overall_status": "PASS" if passed_checks == total_checks else "FAIL",
        "detailed_results": results,
        "recommendations": []
    }
    
    # Add recommendations based on failed checks
    if not results.get("models", False):
        report["recommendations"].append("Fix comment model validation errors")
    
    if not results.get("websocket", False):
        report["recommendations"].append("Review WebSocket handler implementation")
    
    if not results.get("api", False):
        report["recommendations"].append("Complete API endpoint implementation")
    
    if not results.get("integration", False):
        report["recommendations"].append("Ensure all required files are present")
    
    if not results.get("tests", False):
        report["recommendations"].append("Set up proper test configuration")
    
    return report


async def main():
    """Main validation function."""
    
    print("🧪 Comment System Validation")
    print("=" * 50)
    print(f"Validation started at: {datetime.utcnow().isoformat()}")
    print("")
    
    # Run all validation checks
    validation_results = {
        "models": await validate_comment_models(),
        "websocket": await validate_websocket_handler(),
        "api": await validate_api_structure(),
        "integration": await validate_integration_readiness(),
        "tests": await validate_test_configuration()
    }
    
    # Generate comprehensive report
    report = generate_validation_report(validation_results)
    
    print("\n" + "=" * 50)
    print("📊 VALIDATION REPORT")
    print("=" * 50)
    print(f"Overall Status: {report['overall_status']}")
    print(f"Success Rate: {report['success_rate']:.1f}%")
    print(f"Passed: {report['passed_checks']}/{report['total_checks']} checks")
    
    if report["recommendations"]:
        print("\n📋 RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"{i}. {rec}")
    
    print(f"\nValidation completed at: {datetime.utcnow().isoformat()}")
    
    # Return appropriate exit code
    return 0 if report['overall_status'] == 'PASS' else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))