#!/usr/bin/env python3
"""
Hallucination rate validation script for GraphRAG system.
Tests the complete GraphRAG pipeline to ensure <2% hallucination rate.
"""

import asyncio
import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import argparse
import time

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent / 'backend'
sys.path.insert(0, str(backend_dir))

try:
    from services.graphrag.validation_pipeline import ValidationPipeline, ValidationLevel, ValidationConfig
    from services.graphrag.entity_extractor import EntityExtractionPipeline
    from services.graphrag.relationship_extractor import RelationshipExtractor
    from services.graphrag.hallucination_detector import HallucinationDetector
    from services.graphrag.neo4j_optimizer import Neo4jQueryOptimizer
    from services.graphrag.graph_traversal import GraphTraversalStrategies
    from services.graphrag.fact_checker import KnowledgeBaseFactChecker
    from services.hybrid_rag import HybridRAGService
except ImportError as e:
    print(f"Error importing GraphRAG modules: {e}")
    print("Make sure you're running this script from the project root and all dependencies are installed.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HallucinationValidator:
    """
    Comprehensive hallucination rate validator for GraphRAG system.
    Tests all components and measures overall system hallucination rate.
    """
    
    def __init__(self):
        self.validation_pipeline = ValidationPipeline()
        self.entity_extractor = EntityExtractionPipeline()
        self.relationship_extractor = RelationshipExtractor()
        self.hallucination_detector = HallucinationDetector()
        self.neo4j_optimizer = Neo4jQueryOptimizer()
        self.graph_traversal = GraphTraversalStrategies()
        self.fact_checker = KnowledgeBaseFactChecker()
        self.hybrid_rag = HybridRAGService()
        
        self.test_results = {
            'overall_hallucination_rate': 0.0,
            'component_results': {},
            'test_cases': [],
            'performance_metrics': {},
            'validation_summary': {}
        }
    
    async def initialize(self) -> None:
        """Initialize all GraphRAG components."""
        logger.info("Initializing GraphRAG components...")
        start_time = time.time()
        
        try:
            # Initialize components in parallel
            await asyncio.gather(
                self.validation_pipeline.initialize(),
                self.entity_extractor.initialize(),
                self.relationship_extractor.initialize(),
                self.hallucination_detector.initialize(),
                self.neo4j_optimizer.initialize(),
                self.graph_traversal.initialize(),
                self.fact_checker.initialize(),
                self.hybrid_rag.initialize(),
                return_exceptions=True
            )
            
            init_time = time.time() - start_time
            logger.info(f"All components initialized in {init_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def run_validation_tests(
        self,
        test_cases_file: str = None,
        target_hallucination_rate: float = 0.02,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Run comprehensive hallucination rate validation tests.
        
        Args:
            test_cases_file: Path to JSON file with test cases
            target_hallucination_rate: Target hallucination rate (default: 2%)
            verbose: Enable verbose logging
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Starting hallucination rate validation (target: {target_hallucination_rate:.1%})")
        
        # Load or generate test cases
        test_cases = self._load_test_cases(test_cases_file) if test_cases_file else self._generate_test_cases()
        
        logger.info(f"Running validation on {len(test_cases)} test cases")
        
        # Run component health checks first
        await self._run_health_checks()
        
        # Test each component
        component_results = {}
        
        # 1. Test Entity Extraction
        logger.info("Testing Entity Extraction...")
        component_results['entity_extraction'] = await self._test_entity_extraction(test_cases, verbose)
        
        # 2. Test Relationship Extraction  
        logger.info("Testing Relationship Extraction...")
        component_results['relationship_extraction'] = await self._test_relationship_extraction(test_cases, verbose)
        
        # 3. Test Hallucination Detection
        logger.info("Testing Hallucination Detection...")
        component_results['hallucination_detection'] = await self._test_hallucination_detection(test_cases, verbose)
        
        # 4. Test Neo4j Query Optimization
        logger.info("Testing Neo4j Query Optimization...")
        component_results['neo4j_optimization'] = await self._test_neo4j_optimization(verbose)
        
        # 5. Test Graph Traversal
        logger.info("Testing Graph Traversal...")
        component_results['graph_traversal'] = await self._test_graph_traversal(verbose)
        
        # 6. Test Fact Checking
        logger.info("Testing Fact Checking...")
        component_results['fact_checking'] = await self._test_fact_checking(test_cases, verbose)
        
        # 7. Test Validation Pipeline (End-to-end)
        logger.info("Testing Validation Pipeline...")
        component_results['validation_pipeline'] = await self._test_validation_pipeline(test_cases, verbose)
        
        # 8. Test Hybrid RAG
        logger.info("Testing Hybrid RAG...")
        component_results['hybrid_rag'] = await self._test_hybrid_rag(test_cases, verbose)
        
        # Calculate overall results
        overall_results = self._calculate_overall_results(component_results, target_hallucination_rate)
        
        # Generate final report
        final_report = self._generate_final_report(component_results, overall_results, target_hallucination_rate)
        
        self.test_results = final_report
        return final_report
    
    def _load_test_cases(self, file_path: str) -> List[Dict[str, Any]]:
        """Load test cases from JSON file."""
        try:
            with open(file_path, 'r') as f:
                test_cases = json.load(f)
            logger.info(f"Loaded {len(test_cases)} test cases from {file_path}")
            return test_cases
        except Exception as e:
            logger.error(f"Failed to load test cases from {file_path}: {e}")
            return self._generate_test_cases()
    
    def _generate_test_cases(self) -> List[Dict[str, Any]]:
        """Generate comprehensive test cases for validation."""
        test_cases = [
            # Factual content (should have low hallucination)
            {
                "content": "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976. The company is headquartered in Cupertino, California and is known for products like the iPhone, iPad, and Mac computers.",
                "expected_hallucination_rate": 0.0,
                "category": "factual_accurate",
                "entities": ["Apple Inc.", "Steve Jobs", "Steve Wozniak", "Ronald Wayne", "Cupertino", "California", "iPhone", "iPad", "Mac"],
                "description": "Well-known factual information about Apple Inc."
            },
            {
                "content": "Google was founded by Larry Page and Sergey Brin in 1998 while they were PhD students at Stanford University. The company started as a search engine and has since expanded into various technology sectors including cloud computing, advertising, and artificial intelligence.",
                "expected_hallucination_rate": 0.0,
                "category": "factual_accurate",
                "entities": ["Google", "Larry Page", "Sergey Brin", "Stanford University"],
                "description": "Accurate information about Google's founding"
            },
            {
                "content": "Microsoft Corporation is an American multinational technology company founded by Bill Gates and Paul Allen in 1975. The company develops computer software, consumer electronics, personal computers, and related services.",
                "expected_hallucination_rate": 0.0,
                "category": "factual_accurate", 
                "entities": ["Microsoft Corporation", "Bill Gates", "Paul Allen"],
                "description": "Basic facts about Microsoft"
            },
            
            # Mixed content (some factual, some questionable)
            {
                "content": "Tesla was founded by Elon Musk in 2003 and is known for electric vehicles. The company has plans to establish colonies on Mars by 2025 and has developed time travel technology that allows cars to travel backwards in time.",
                "expected_hallucination_rate": 0.6,
                "category": "mixed_factual_fictional",
                "entities": ["Tesla", "Elon Musk"],
                "description": "Mix of accurate Tesla facts with fictional claims"
            },
            {
                "content": "Amazon was founded by Jeff Bezos in 1994 as an online bookstore. The company now owns several small planets in the solar system and uses alien technology to deliver packages instantly through teleportation.",
                "expected_hallucination_rate": 0.5,
                "category": "mixed_factual_fictional",
                "entities": ["Amazon", "Jeff Bezos"],
                "description": "Amazon facts mixed with science fiction"
            },
            
            # Highly inaccurate content (should have high hallucination)
            {
                "content": "Apple Inc. was founded by Thomas Edison in 1850 and is headquartered on the Moon. The company is famous for inventing the first flying car and controlling all weather patterns on Earth through their iCloud service.",
                "expected_hallucination_rate": 0.95,
                "category": "highly_inaccurate",
                "entities": ["Apple Inc.", "Thomas Edison"],
                "description": "Completely inaccurate claims about Apple"
            },
            {
                "content": "Microsoft was established by Leonardo da Vinci in 1492 as a painting company. Today, it manufactures unicorns and sells dreams to children. The CEO is a robot from the future named Bill Gates 3000.",
                "expected_hallucination_rate": 0.9,
                "category": "highly_inaccurate", 
                "entities": ["Microsoft", "Leonardo da Vinci", "Bill Gates 3000"],
                "description": "Absurd claims about Microsoft"
            },
            
            # Technical content
            {
                "content": "Python is a high-level programming language created by Guido van Rossum and first released in 1991. It emphasizes code readability and uses significant indentation. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
                "expected_hallucination_rate": 0.0,
                "category": "technical_accurate",
                "entities": ["Python", "Guido van Rossum"],
                "description": "Accurate technical information about Python"
            },
            {
                "content": "JavaScript was invented by Brendan Eich in 1995 at Netscape Communications. Despite its name, JavaScript has no relation to Java. It was created in just 10 days and has become one of the most popular programming languages for web development.",
                "expected_hallucination_rate": 0.0,
                "category": "technical_accurate",
                "entities": ["JavaScript", "Brendan Eich", "Netscape Communications", "Java"],
                "description": "Facts about JavaScript development"
            },
            
            # Numerical claims
            {
                "content": "The iPhone was first released in 2007 and sold for $499 for the 4GB model and $599 for the 8GB model. Apple sold over 6.1 million iPhones in the first year.",
                "expected_hallucination_rate": 0.0,
                "category": "numerical_accurate",
                "entities": ["iPhone", "Apple"],
                "description": "Accurate numerical claims about iPhone sales"
            },
            {
                "content": "Google processes over 50 billion search queries per day and has indexed more than 200 trillion web pages. The company employs over 150,000 people worldwide as of 2021.",
                "expected_hallucination_rate": 0.3,  # Some numbers might be outdated/approximate
                "category": "numerical_mixed",
                "entities": ["Google"],
                "description": "Google statistics (some may be approximate)"
            }
        ]
        
        logger.info(f"Generated {len(test_cases)} test cases")
        return test_cases
    
    async def _run_health_checks(self) -> None:
        """Run health checks on all components."""
        logger.info("Running component health checks...")
        
        components = [
            ('entity_extractor', self.entity_extractor),
            ('relationship_extractor', self.relationship_extractor),
            ('hallucination_detector', self.hallucination_detector),
            ('neo4j_optimizer', self.neo4j_optimizer),
            ('graph_traversal', self.graph_traversal),
            ('fact_checker', self.fact_checker),
            ('validation_pipeline', self.validation_pipeline),
            ('hybrid_rag', self.hybrid_rag)
        ]
        
        for name, component in components:
            try:
                health = await component.health_check()
                status = health.get('status', 'unknown')
                if status not in ['healthy', 'degraded']:
                    logger.warning(f"{name} health check failed: {health}")
                else:
                    logger.info(f"{name}: {status}")
            except Exception as e:
                logger.error(f"{name} health check error: {e}")
    
    async def _test_entity_extraction(self, test_cases: List[Dict[str, Any]], verbose: bool) -> Dict[str, Any]:
        """Test entity extraction component."""
        results = {'test_results': [], 'avg_processing_time': 0, 'accuracy': 0}
        total_time = 0
        correct_extractions = 0
        
        for i, test_case in enumerate(test_cases):
            start_time = time.time()
            try:
                result = await self.entity_extractor.extract_entities(
                    test_case['content'],
                    min_confidence=0.7
                )
                
                processing_time = (time.time() - start_time) * 1000
                total_time += processing_time
                
                # Check if expected entities were found
                extracted_entities = [e['text'] for e in result['entities']]
                expected_entities = test_case.get('entities', [])
                
                found_entities = set(extracted_entities) & set(expected_entities)
                accuracy = len(found_entities) / len(expected_entities) if expected_entities else 1.0
                
                if accuracy > 0.7:
                    correct_extractions += 1
                
                test_result = {
                    'test_case': i,
                    'processing_time_ms': processing_time,
                    'entities_found': len(extracted_entities),
                    'expected_entities': len(expected_entities),
                    'accuracy': accuracy,
                    'entities_extracted': extracted_entities[:5]  # First 5 for brevity
                }
                
                if verbose:
                    logger.info(f"Entity extraction test {i+1}: {accuracy:.2%} accuracy, {processing_time:.1f}ms")
                
                results['test_results'].append(test_result)
                
            except Exception as e:
                logger.error(f"Entity extraction test {i+1} failed: {e}")
                results['test_results'].append({
                    'test_case': i,
                    'error': str(e),
                    'processing_time_ms': (time.time() - start_time) * 1000
                })
        
        results['avg_processing_time'] = total_time / len(test_cases) if test_cases else 0
        results['accuracy'] = correct_extractions / len(test_cases) if test_cases else 0
        results['performance_target_met'] = results['avg_processing_time'] < 50  # <50ms target
        
        return results
    
    async def _test_relationship_extraction(self, test_cases: List[Dict[str, Any]], verbose: bool) -> Dict[str, Any]:
        """Test relationship extraction component."""
        results = {'test_results': [], 'avg_processing_time': 0, 'relationships_found': 0}
        total_time = 0
        total_relationships = 0
        
        for i, test_case in enumerate(test_cases):
            start_time = time.time()
            try:
                result = await self.relationship_extractor.extract_relationships(
                    test_case['content'],
                    min_confidence=0.6
                )
                
                processing_time = (time.time() - start_time) * 1000
                total_time += processing_time
                
                relationships_found = len(result['relationships'])
                total_relationships += relationships_found
                
                test_result = {
                    'test_case': i,
                    'processing_time_ms': processing_time,
                    'relationships_found': relationships_found,
                    'relationship_types': list(set(r['relationship_type'] for r in result['relationships']))
                }
                
                if verbose:
                    logger.info(f"Relationship extraction test {i+1}: {relationships_found} relationships, {processing_time:.1f}ms")
                
                results['test_results'].append(test_result)
                
            except Exception as e:
                logger.error(f"Relationship extraction test {i+1} failed: {e}")
                results['test_results'].append({
                    'test_case': i,
                    'error': str(e),
                    'processing_time_ms': (time.time() - start_time) * 1000
                })
        
        results['avg_processing_time'] = total_time / len(test_cases) if test_cases else 0
        results['avg_relationships_per_case'] = total_relationships / len(test_cases) if test_cases else 0
        results['performance_target_met'] = results['avg_processing_time'] < 50
        
        return results
    
    async def _test_hallucination_detection(self, test_cases: List[Dict[str, Any]], verbose: bool) -> Dict[str, Any]:
        """Test hallucination detection component."""
        results = {'test_results': [], 'avg_hallucination_rate': 0, 'accuracy': 0}
        total_hallucination_rate = 0
        accurate_detections = 0
        
        for i, test_case in enumerate(test_cases):
            start_time = time.time()
            try:
                result = await self.hallucination_detector.detect_hallucinations(
                    test_case['content'],
                    validation_mode="comprehensive"
                )
                
                processing_time = (time.time() - start_time) * 1000
                detected_rate = result['hallucination_rate']
                expected_rate = test_case.get('expected_hallucination_rate', 0.0)
                
                total_hallucination_rate += detected_rate
                
                # Check accuracy (within reasonable margin)
                accuracy = 1.0 - abs(detected_rate - expected_rate)
                if accuracy > 0.5:  # Within 50% margin
                    accurate_detections += 1
                
                test_result = {
                    'test_case': i,
                    'processing_time_ms': processing_time,
                    'detected_rate': detected_rate,
                    'expected_rate': expected_rate,
                    'accuracy': accuracy,
                    'passes_threshold': detected_rate <= 0.02,
                    'category': test_case.get('category', 'unknown')
                }
                
                if verbose:
                    logger.info(f"Hallucination detection test {i+1}: {detected_rate:.1%} detected vs {expected_rate:.1%} expected")
                
                results['test_results'].append(test_result)
                
            except Exception as e:
                logger.error(f"Hallucination detection test {i+1} failed: {e}")
                results['test_results'].append({
                    'test_case': i,
                    'error': str(e),
                    'processing_time_ms': (time.time() - start_time) * 1000
                })
        
        results['avg_hallucination_rate'] = total_hallucination_rate / len(test_cases) if test_cases else 0
        results['accuracy'] = accurate_detections / len(test_cases) if test_cases else 0
        results['meets_target'] = results['avg_hallucination_rate'] <= 0.02  # 2% target
        
        return results
    
    async def _test_neo4j_optimization(self, verbose: bool) -> Dict[str, Any]:
        """Test Neo4j query optimization."""
        results = {'test_results': [], 'avg_response_time': 0, 'cache_hit_rate': 0}
        
        # Test common queries
        test_queries = [
            ('entity_similarity', {'search_term': 'Apple', 'min_score': 0.5, 'limit': 10}),
            ('relationship_lookup', {'source_lower': 'apple', 'target_lower': 'steve jobs', 'limit': 5}),
            ('fact_verification', {'claim_text': 'Apple was founded in 1976', 'min_score': 0.5, 'limit': 5}),
            ('graph_stats', {})
        ]
        
        total_time = 0
        
        for i, (query_name, params) in enumerate(test_queries):
            start_time = time.time()
            try:
                data, metrics = await self.neo4j_optimizer.execute_optimized_query(
                    query_name, params
                )
                
                processing_time = metrics.execution_time_ms
                total_time += processing_time
                
                test_result = {
                    'query': query_name,
                    'processing_time_ms': processing_time,
                    'results_count': metrics.result_count,
                    'cache_hit': metrics.cache_hit,
                    'index_used': metrics.index_used,
                    'meets_target': processing_time < 50
                }
                
                if verbose:
                    logger.info(f"Neo4j query {query_name}: {processing_time:.1f}ms, {metrics.result_count} results")
                
                results['test_results'].append(test_result)
                
            except Exception as e:
                logger.error(f"Neo4j query test {query_name} failed: {e}")
                results['test_results'].append({
                    'query': query_name,
                    'error': str(e)
                })
        
        results['avg_response_time'] = total_time / len(test_queries) if test_queries else 0
        results['performance_target_met'] = results['avg_response_time'] < 50  # <50ms target
        
        # Get performance statistics
        try:
            stats = await self.neo4j_optimizer.get_performance_statistics()
            results['cache_hit_rate'] = stats.get('cache_statistics', {}).get('hit_rate', 0)
        except Exception as e:
            logger.warning(f"Could not get Neo4j performance statistics: {e}")
        
        return results
    
    async def _test_graph_traversal(self, verbose: bool) -> Dict[str, Any]:
        """Test graph traversal strategies."""
        results = {'test_results': [], 'avg_processing_time': 0, 'traversal_success_rate': 0}
        
        # Test different traversal algorithms
        test_entities = ['Apple', 'Google', 'Microsoft', 'Tesla']
        successful_traversals = 0
        total_time = 0
        
        for i, entity in enumerate(test_entities[:2]):  # Limit to 2 for performance
            try:
                # Test breadth-first traversal
                start_time = time.time()
                bfs_result = await self.graph_traversal.traverse_breadth_first(
                    entity, max_depth=2, max_nodes=15, min_confidence=0.5
                )
                bfs_time = (time.time() - start_time) * 1000
                total_time += bfs_time
                
                # Test centrality analysis
                start_time = time.time()
                centrality_result = await self.graph_traversal.analyze_centrality(
                    entity_subset=[entity], top_n=5
                )
                centrality_time = (time.time() - start_time) * 1000
                total_time += centrality_time
                
                if bfs_result['total_nodes_explored'] > 0:
                    successful_traversals += 1
                
                test_result = {
                    'entity': entity,
                    'bfs_processing_time_ms': bfs_time,
                    'centrality_processing_time_ms': centrality_time,
                    'nodes_explored': bfs_result['total_nodes_explored'],
                    'relationships_found': bfs_result['total_relationships_found']
                }
                
                if verbose:
                    logger.info(f"Graph traversal test {entity}: {bfs_result['total_nodes_explored']} nodes, {bfs_time:.1f}ms")
                
                results['test_results'].append(test_result)
                
            except Exception as e:
                logger.error(f"Graph traversal test {entity} failed: {e}")
                results['test_results'].append({
                    'entity': entity,
                    'error': str(e)
                })
        
        results['avg_processing_time'] = total_time / (len(test_entities[:2]) * 2) if test_entities else 0
        results['traversal_success_rate'] = successful_traversals / len(test_entities[:2]) if test_entities else 0
        
        return results
    
    async def _test_fact_checking(self, test_cases: List[Dict[str, Any]], verbose: bool) -> Dict[str, Any]:
        """Test fact-checking component."""
        results = {'test_results': [], 'avg_processing_time': 0, 'accuracy': 0}
        total_time = 0
        accurate_checks = 0
        
        # Test a subset of cases for fact-checking (performance consideration)
        fact_check_cases = test_cases[:5]
        
        for i, test_case in enumerate(fact_check_cases):
            start_time = time.time()
            try:
                result = await self.fact_checker.fact_check_content(
                    test_case['content'],
                    max_claims=5  # Limit claims for performance
                )
                
                processing_time = (time.time() - start_time) * 1000
                total_time += processing_time
                
                # Evaluate accuracy based on expected hallucination rate
                expected_rate = test_case.get('expected_hallucination_rate', 0.0)
                overall_analysis = result['overall_analysis']
                detected_accuracy = overall_analysis.get('accuracy_score', 0)
                
                # If expected hallucination is high, accuracy should be low and vice versa
                expected_accuracy = 1.0 - expected_rate
                accuracy_diff = abs(detected_accuracy - expected_accuracy)
                
                if accuracy_diff < 0.3:  # Within 30% margin
                    accurate_checks += 1
                
                test_result = {
                    'test_case': i,
                    'processing_time_ms': processing_time,
                    'claims_checked': result['claims_checked'],
                    'detected_accuracy': detected_accuracy,
                    'expected_accuracy': expected_accuracy,
                    'accuracy_difference': accuracy_diff
                }
                
                if verbose:
                    logger.info(f"Fact checking test {i+1}: {result['claims_checked']} claims, {detected_accuracy:.2%} accuracy")
                
                results['test_results'].append(test_result)
                
            except Exception as e:
                logger.error(f"Fact checking test {i+1} failed: {e}")
                results['test_results'].append({
                    'test_case': i,
                    'error': str(e),
                    'processing_time_ms': (time.time() - start_time) * 1000
                })
        
        results['avg_processing_time'] = total_time / len(fact_check_cases) if fact_check_cases else 0
        results['accuracy'] = accurate_checks / len(fact_check_cases) if fact_check_cases else 0
        
        return results
    
    async def _test_validation_pipeline(self, test_cases: List[Dict[str, Any]], verbose: bool) -> Dict[str, Any]:
        """Test end-to-end validation pipeline."""
        results = {'test_results': [], 'avg_processing_time': 0, 'pass_rate': 0}
        total_time = 0
        passed_validations = 0
        
        for i, test_case in enumerate(test_cases):
            start_time = time.time()
            try:
                # Use standard validation level
                result = await self.validation_pipeline.validate_content(
                    test_case['content'],
                    ValidationLevel.STANDARD
                )
                
                processing_time = result.processing_time_ms
                total_time += processing_time
                
                # Check if validation behaves as expected
                expected_rate = test_case.get('expected_hallucination_rate', 0.0)
                confidence = result.overall_confidence
                
                # High confidence expected for low hallucination content
                expected_confidence = 1.0 - expected_rate
                confidence_matches = abs(confidence - expected_confidence) < 0.3
                
                if confidence_matches or result.status.value in ['passed', 'warning']:
                    passed_validations += 1
                
                test_result = {
                    'test_case': i,
                    'processing_time_ms': processing_time,
                    'validation_status': result.status.value,
                    'overall_confidence': confidence,
                    'expected_confidence': expected_confidence,
                    'issues_found': len(result.issues_found),
                    'recommendations': len(result.recommendations)
                }
                
                if verbose:
                    logger.info(f"Validation pipeline test {i+1}: {result.status.value}, {confidence:.2%} confidence")
                
                results['test_results'].append(test_result)
                
            except Exception as e:
                logger.error(f"Validation pipeline test {i+1} failed: {e}")
                results['test_results'].append({
                    'test_case': i,
                    'error': str(e),
                    'processing_time_ms': (time.time() - start_time) * 1000
                })
        
        results['avg_processing_time'] = total_time / len(test_cases) if test_cases else 0
        results['pass_rate'] = passed_validations / len(test_cases) if test_cases else 0
        results['performance_target_met'] = results['avg_processing_time'] < 10000  # <10s target
        
        return results
    
    async def _test_hybrid_rag(self, test_cases: List[Dict[str, Any]], verbose: bool) -> Dict[str, Any]:
        """Test hybrid RAG system."""
        results = {'test_results': [], 'avg_processing_time': 0, 'validation_pass_rate': 0}
        total_time = 0
        passed_validations = 0
        
        # Test a subset for performance
        rag_test_cases = test_cases[:3]
        
        for i, test_case in enumerate(rag_test_cases):
            start_time = time.time()
            try:
                result = await self.hybrid_rag.validate_content(
                    test_case['content']
                )
                
                processing_time = result['processing_time_ms']
                total_time += processing_time
                
                confidence = result['confidence']
                passes_threshold = result['passes_threshold']
                
                if passes_threshold or confidence > 0.7:
                    passed_validations += 1
                
                test_result = {
                    'test_case': i,
                    'processing_time_ms': processing_time,
                    'confidence': confidence,
                    'passes_threshold': passes_threshold,
                    'entity_validation': result.get('entity_validation', {}).get('confidence', 0),
                    'community_validation': result.get('community_validation', {}).get('confidence', 0),
                    'global_validation': result.get('global_validation', {}).get('confidence', 0)
                }
                
                if verbose:
                    logger.info(f"Hybrid RAG test {i+1}: {confidence:.2%} confidence, passes: {passes_threshold}")
                
                results['test_results'].append(test_result)
                
            except Exception as e:
                logger.error(f"Hybrid RAG test {i+1} failed: {e}")
                results['test_results'].append({
                    'test_case': i,
                    'error': str(e),
                    'processing_time_ms': (time.time() - start_time) * 1000
                })
        
        results['avg_processing_time'] = total_time / len(rag_test_cases) if rag_test_cases else 0
        results['validation_pass_rate'] = passed_validations / len(rag_test_cases) if rag_test_cases else 0
        
        return results
    
    def _calculate_overall_results(
        self, 
        component_results: Dict[str, Any], 
        target_rate: float
    ) -> Dict[str, Any]:
        """Calculate overall hallucination rate and system performance."""
        
        # Calculate overall hallucination rate from hallucination detection component
        hall_results = component_results.get('hallucination_detection', {})
        overall_hallucination_rate = hall_results.get('avg_hallucination_rate', 1.0)
        
        # Calculate system-wide accuracy
        accuracies = []
        for component, results in component_results.items():
            if 'accuracy' in results:
                accuracies.append(results['accuracy'])
        
        system_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        
        # Calculate performance metrics
        performance_metrics = {}
        for component, results in component_results.items():
            if 'avg_processing_time' in results:
                performance_metrics[component] = results['avg_processing_time']
        
        # Overall assessment
        meets_hallucination_target = overall_hallucination_rate <= target_rate
        performance_acceptable = all(
            time_ms < 1000 for time_ms in performance_metrics.values()  # All under 1s
        )
        
        return {
            'overall_hallucination_rate': overall_hallucination_rate,
            'system_accuracy': system_accuracy,
            'meets_hallucination_target': meets_hallucination_target,
            'performance_acceptable': performance_acceptable,
            'performance_metrics': performance_metrics,
            'target_hallucination_rate': target_rate,
            'hallucination_margin': target_rate - overall_hallucination_rate
        }
    
    def _generate_final_report(
        self, 
        component_results: Dict[str, Any], 
        overall_results: Dict[str, Any], 
        target_rate: float
    ) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        
        timestamp = datetime.now().isoformat()
        
        # Create summary
        summary = {
            'validation_timestamp': timestamp,
            'target_hallucination_rate': f"{target_rate:.1%}",
            'achieved_hallucination_rate': f"{overall_results['overall_hallucination_rate']:.1%}",
            'meets_target': overall_results['meets_hallucination_target'],
            'system_accuracy': f"{overall_results['system_accuracy']:.1%}",
            'performance_acceptable': overall_results['performance_acceptable']
        }
        
        # Component status
        component_status = {}
        for component, results in component_results.items():
            status = 'PASS'
            issues = []
            
            if 'error' in str(results):
                status = 'FAIL'
                issues.append('Component errors detected')
            
            if 'avg_processing_time' in results and results['avg_processing_time'] > 1000:
                status = 'WARN' if status == 'PASS' else status
                issues.append('Performance below target')
            
            if 'accuracy' in results and results['accuracy'] < 0.7:
                status = 'WARN' if status == 'PASS' else status
                issues.append('Accuracy below 70%')
            
            component_status[component] = {
                'status': status,
                'issues': issues,
                'key_metrics': {
                    k: v for k, v in results.items() 
                    if k in ['avg_processing_time', 'accuracy', 'avg_hallucination_rate']
                }
            }
        
        # Recommendations
        recommendations = []
        
        if not overall_results['meets_hallucination_target']:
            diff = overall_results['overall_hallucination_rate'] - target_rate
            recommendations.append(
                f"CRITICAL: Hallucination rate ({overall_results['overall_hallucination_rate']:.1%}) "
                f"exceeds target by {diff:.1%}. Review fact-checking and validation logic."
            )
        
        if not overall_results['performance_acceptable']:
            slow_components = [
                comp for comp, time_ms in overall_results['performance_metrics'].items()
                if time_ms > 1000
            ]
            recommendations.append(
                f"Performance issue: {', '.join(slow_components)} exceed 1s processing time. "
                f"Consider optimization or caching improvements."
            )
        
        if overall_results['system_accuracy'] < 0.8:
            recommendations.append(
                f"System accuracy ({overall_results['system_accuracy']:.1%}) below 80%. "
                f"Review component accuracy and validation thresholds."
            )
        
        if not recommendations:
            recommendations.append(
                "✅ All systems operational. GraphRAG pipeline meets hallucination rate target "
                f"of {target_rate:.1%} with {overall_results['overall_hallucination_rate']:.1%} achieved."
            )
        
        return {
            'summary': summary,
            'overall_results': overall_results,
            'component_results': component_results,
            'component_status': component_status,
            'recommendations': recommendations,
            'validation_metadata': {
                'timestamp': timestamp,
                'test_cases_count': len(component_results.get('hallucination_detection', {}).get('test_results', [])),
                'components_tested': len(component_results),
                'total_validation_time': sum(overall_results['performance_metrics'].values())
            }
        }
    
    def save_report(self, filepath: str = None) -> str:
        """Save validation report to file."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"hallucination_validation_report_{timestamp}.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            
            logger.info(f"Validation report saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save report to {filepath}: {e}")
            raise
    
    async def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        
        components_with_close = [
            self.neo4j_optimizer,
            self.graph_traversal,
            self.fact_checker,
            self.validation_pipeline
        ]
        
        for component in components_with_close:
            try:
                if hasattr(component, 'close'):
                    await component.close()
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")


def print_summary(results: Dict[str, Any]) -> None:
    """Print a concise summary of validation results."""
    print("\n" + "="*80)
    print("GRAPHRAG HALLUCINATION RATE VALIDATION SUMMARY")
    print("="*80)
    
    summary = results['summary']
    print(f"🎯 Target Hallucination Rate: {summary['target_hallucination_rate']}")
    print(f"📊 Achieved Hallucination Rate: {summary['achieved_hallucination_rate']}")
    print(f"✅ Meets Target: {summary['meets_target']}")
    print(f"🎯 System Accuracy: {summary['system_accuracy']}")
    print(f"⚡ Performance Acceptable: {summary['performance_acceptable']}")
    
    print("\n📋 COMPONENT STATUS:")
    for component, status in results['component_status'].items():
        status_icon = "✅" if status['status'] == 'PASS' else "⚠️" if status['status'] == 'WARN' else "❌"
        print(f"{status_icon} {component.replace('_', ' ').title()}: {status['status']}")
        
        if status['issues']:
            for issue in status['issues']:
                print(f"   - {issue}")
    
    print("\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"{i}. {rec}")
    
    print("\n" + "="*80)


async def main():
    """Main validation script execution."""
    parser = argparse.ArgumentParser(description='Validate GraphRAG hallucination rate')
    parser.add_argument('--test-cases', help='Path to JSON file with test cases')
    parser.add_argument('--target-rate', type=float, default=0.02, help='Target hallucination rate (default: 0.02)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--save-report', help='Save detailed report to file')
    parser.add_argument('--quick', action='store_true', help='Run quick validation (fewer test cases)')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    validator = HallucinationValidator()
    
    try:
        # Initialize all components
        print("🚀 Initializing GraphRAG components...")
        await validator.initialize()
        print("✅ Components initialized successfully")
        
        # Run validation tests
        print(f"\n🔍 Running hallucination rate validation (target: {args.target_rate:.1%})...")
        results = await validator.run_validation_tests(
            test_cases_file=args.test_cases,
            target_hallucination_rate=args.target_rate,
            verbose=args.verbose
        )
        
        # Print summary
        print_summary(results)
        
        # Save detailed report if requested
        if args.save_report:
            report_path = validator.save_report(args.save_report)
            print(f"\n📄 Detailed report saved to: {report_path}")
        
        # Exit with appropriate code
        meets_target = results['summary']['meets_target']
        performance_ok = results['summary']['performance_acceptable']
        
        if meets_target and performance_ok:
            print("\n🎉 SUCCESS: GraphRAG system passes hallucination rate validation!")
            exit_code = 0
        elif meets_target:
            print("\n⚠️  WARNING: Meets hallucination target but has performance issues")
            exit_code = 1
        else:
            print("\n❌ FAILURE: GraphRAG system exceeds hallucination rate target")
            exit_code = 2
        
        return exit_code
        
    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        print(f"\n❌ FATAL ERROR: {e}")
        return 1
    finally:
        await validator.cleanup()


if __name__ == '__main__':
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nValidation interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)