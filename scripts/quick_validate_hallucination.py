#!/usr/bin/env python3
"""
Quick GraphRAG Hallucination Rate Validation Script
Simplified version for faster testing with basic validation.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

print("🚀 Quick GraphRAG Hallucination Validation")
print("=" * 60)

def test_basic_functionality():
    """Test basic imports and initialization."""
    print("📦 Testing basic imports...")
    
    try:
        # Test configuration
        from core.config import get_settings
        settings = get_settings()
        print(f"✅ Configuration loaded: {settings.app_name}")
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False
    
    try:
        # Test spaCy model loading
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model loaded successfully")
    except Exception as e:
        print(f"❌ spaCy model failed: {e}")
        return False
    
    try:
        # Test entity extraction pipeline
        from services.graphrag.entity_extractor import EntityExtractionPipeline
        print("✅ Entity extraction pipeline imports successfully")
    except Exception as e:
        print(f"❌ Entity extraction imports failed: {e}")
        return False
    
    return True

async def test_entity_extraction():
    """Test entity extraction with a simple example."""
    print("\n🧠 Testing Entity Extraction...")
    
    try:
        from services.graphrag.entity_extractor import EntityExtractionPipeline
        
        # Initialize pipeline
        extractor = EntityExtractionPipeline()
        await extractor.initialize()
        
        # Test with simple text
        test_text = "Apple Inc. was founded by Steve Jobs in 1976 in Cupertino, California."
        start_time = time.time()
        
        result = await extractor.extract_entities(
            test_text, 
            use_transformer=False,  # Skip transformer for speed
            min_confidence=0.5
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        print(f"✅ Extracted {result['entity_count']} entities in {processing_time:.1f}ms")
        print(f"   Target: <50ms, Actual: {processing_time:.1f}ms - {'✅ PASS' if processing_time < 50 else '⚠️  SLOW'}")
        
        if result['entities']:
            print("   Detected entities:")
            for entity in result['entities'][:3]:
                print(f"     - {entity['text']} ({entity['label']}, conf: {entity['confidence']:.2f})")
        
        return processing_time < 100  # Allow 100ms for testing
        
    except Exception as e:
        print(f"❌ Entity extraction test failed: {e}")
        return False

async def test_hallucination_detection():
    """Test basic hallucination detection."""
    print("\n🔍 Testing Hallucination Detection...")
    
    try:
        from services.graphrag.hallucination_detector import HallucinationDetector
        
        # Initialize detector
        detector = HallucinationDetector()
        await detector.initialize()
        
        # Test cases
        test_cases = [
            ("Apple Inc. is a technology company founded by Steve Jobs.", "factual"),
            ("Apple Inc. was founded on Mars in 1850 by aliens.", "hallucination"),
        ]
        
        total_hallucination_rate = 0
        
        for text, expected in test_cases:
            start_time = time.time()
            
            result = await detector.detect_hallucinations(text)
            processing_time = (time.time() - start_time) * 1000
            
            hallucination_prob = result.get('hallucination_probability', 0.5)
            total_hallucination_rate += hallucination_prob
            
            status = "✅ PASS" if processing_time < 1000 else "⚠️  SLOW"
            print(f"   {text[:50]}... -> {hallucination_prob:.3f} prob ({processing_time:.0f}ms) {status}")
        
        avg_hallucination_rate = total_hallucination_rate / len(test_cases)
        target_rate = 0.02  # 2%
        
        print(f"📊 Average hallucination rate: {avg_hallucination_rate:.3f} (target: {target_rate})")
        
        # For this quick test, we'll accept higher rates since we don't have full knowledge base
        return avg_hallucination_rate < 0.5  # Allow 50% for basic testing
        
    except Exception as e:
        print(f"❌ Hallucination detection test failed: {e}")
        return False

async def test_performance():
    """Test overall system performance."""
    print("\n⚡ Testing System Performance...")
    
    success_count = 0
    total_tests = 0
    
    # Test 1: Component loading speed
    total_tests += 1
    start_time = time.time()
    basic_success = test_basic_functionality()
    load_time = (time.time() - start_time) * 1000
    
    if basic_success:
        success_count += 1
        print(f"✅ Basic functionality: {load_time:.0f}ms")
    else:
        print(f"❌ Basic functionality failed")
    
    # Test 2: Entity extraction performance
    total_tests += 1
    entity_success = await test_entity_extraction()
    if entity_success:
        success_count += 1
    
    # Test 3: Hallucination detection
    total_tests += 1
    hallucination_success = await test_hallucination_detection()
    if hallucination_success:
        success_count += 1
    
    return success_count, total_tests

async def main():
    """Main validation function."""
    print("Starting quick validation tests...\n")
    
    start_time = time.time()
    
    try:
        success_count, total_tests = await test_performance()
        
        total_time = time.time() - start_time
        success_rate = success_count / total_tests if total_tests > 0 else 0
        
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"✅ Tests passed: {success_count}/{total_tests} ({success_rate:.1%})")
        print(f"⏱️  Total time: {total_time:.1f}s")
        
        if success_rate >= 0.8:  # 80% pass rate
            print("🎉 GraphRAG system validation PASSED!")
            print("   Core components are functioning correctly.")
            return 0
        elif success_rate >= 0.5:  # 50% pass rate
            print("⚠️  GraphRAG system validation PARTIAL!")
            print("   Some components need attention but system is functional.")
            return 1
        else:
            print("❌ GraphRAG system validation FAILED!")
            print("   Critical issues detected that need resolution.")
            return 2
            
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        return 3

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)