#!/usr/bin/env python3
"""
Test runner for comment system integration tests.

This script runs the integration tests for the comment and annotation system
to validate that all components work together properly.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_comment_integration_tests():
    """Run comment system integration tests."""
    
    # Set up test environment
    test_env = os.environ.copy()
    test_env.update({
        "TESTING": "true",
        "SECRET_KEY": "test-secret-key-for-integration-testing",
        "NEO4J_PASSWORD": "development",  # Use development password for testing
        "OPENROUTER_API_KEY": "test-key",
        "REDIS_URL": "redis://localhost:6379/1",  # Use test database
        "DATABASE_URL": "postgresql://test_user:test_password@localhost/test_db"
    })
    
    # Test commands to run
    test_commands = [
        # Run comment system integration tests
        [
            "python", "-m", "pytest", 
            "tests/integration/test_comment_system_integration.py",
            "-v", "--tb=short", 
            "-m", "integration"
        ],
        
        # Run specific test classes
        [
            "python", "-m", "pytest", 
            "tests/integration/test_comment_system_integration.py::TestCommentCRUDIntegration",
            "-v"
        ],
        
        # Run WebSocket integration tests
        [
            "python", "-m", "pytest", 
            "tests/integration/test_comment_system_integration.py::TestCommentWebSocketIntegration",
            "-v", "-s"  # -s to see print output
        ]
    ]
    
    print("🧪 Running Comment System Integration Tests")
    print("=" * 50)
    
    for i, cmd in enumerate(test_commands, 1):
        print(f"\n📋 Test Suite {i}/{len(test_commands)}: {' '.join(cmd)}")
        print("-" * 40)
        
        try:
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent,
                env=test_env,
                capture_output=False,  # Show output in real-time
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print(f"✅ Test Suite {i} passed")
            else:
                print(f"❌ Test Suite {i} failed with exit code {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Test Suite {i} timed out")
            return False
        except Exception as e:
            print(f"💥 Test Suite {i} crashed: {e}")
            return False
    
    print("\n🎉 All comment integration tests completed successfully!")
    return True


def run_unit_tests():
    """Run unit tests for comment system."""
    
    print("\n🔬 Running Comment System Unit Tests")
    print("=" * 50)
    
    test_env = os.environ.copy()
    test_env.update({
        "TESTING": "true",
        "SECRET_KEY": "test-secret-key",
        "NEO4J_PASSWORD": "development"
    })
    
    # Unit test commands
    unit_commands = [
        # Test comment models
        [
            "python", "-m", "pytest", 
            "tests/unit/",
            "-v", "--tb=short",
            "-m", "unit"
        ]
    ]
    
    for cmd in unit_commands:
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent,
                env=test_env,
                text=True,
                timeout=120  # 2 minute timeout for unit tests
            )
            
            if result.returncode != 0:
                print(f"❌ Unit tests failed with exit code {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⏰ Unit tests timed out")
            return False
        except Exception as e:
            print(f"💥 Unit tests crashed: {e}")
            return False
    
    print("✅ Unit tests completed successfully!")
    return True


def check_test_environment():
    """Check if test environment is properly set up."""
    
    print("🔍 Checking test environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check required packages
    required_packages = [
        "pytest", "fastapi", "httpx", "pydantic", "structlog"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} is not installed")
            return False
    
    # Check test files exist
    test_files = [
        "tests/conftest.py",
        "tests/integration/test_comment_system_integration.py"
    ]
    
    for test_file in test_files:
        test_path = Path(__file__).parent / test_file
        if test_path.exists():
            print(f"✅ {test_file} exists")
        else:
            print(f"❌ {test_file} not found")
            return False
    
    print("✅ Test environment check passed!")
    return True


def main():
    """Main test runner function."""
    
    print("🚀 Comment System Integration Test Runner")
    print("=" * 60)
    
    # Check environment first
    if not check_test_environment():
        print("\n💥 Environment check failed. Please install missing dependencies.")
        return 1
    
    # Run tests based on command line arguments
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == "unit":
            success = run_unit_tests()
        elif test_type == "integration":
            success = run_comment_integration_tests()
        elif test_type == "all":
            success = run_unit_tests() and run_comment_integration_tests()
        else:
            print(f"Unknown test type: {test_type}")
            print("Usage: python test_runner.py [unit|integration|all]")
            return 1
    else:
        # Default: run integration tests
        success = run_comment_integration_tests()
    
    if success:
        print("\n🎊 All tests completed successfully!")
        return 0
    else:
        print("\n💥 Some tests failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    exit(main())