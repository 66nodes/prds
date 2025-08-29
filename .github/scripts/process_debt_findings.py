#!/usr/bin/env python3
"""
Process technical debt findings with comprehensive error handling and validation
"""
import json
import yaml
import argparse
import sys
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DebtProcessor:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.validation_errors = []
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.validate_config(config)
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in config file: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
            
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure"""
        required_sections = ['debt-categories', 'sprint-config', 'analysis']
        for section in required_sections:
            if section not in config:
                self.validation_errors.append(f"Missing required section: {section}")
        
        # Validate debt categories
        if 'debt-categories' in config:
            for category_name, category_config in config['debt-categories'].items():
                required_fields = ['labels', 'priority', 'sla-days']
                for field in required_fields:
                    if field not in category_config:
                        self.validation_errors.append(
                            f"Missing required field '{field}' in category '{category_name}'"
                        )
                        
        if self.validation_errors:
            raise ValueError(f"Configuration validation failed: {self.validation_errors}")
    
    def process_results(self, results_json: str, output_path: str) -> None:
        """Process analysis results with comprehensive error handling"""
        try:
            results = json.loads(results_json)
            logger.info(f"Processing {len(results)} analysis results")
            
            debt_items = self.categorize_findings(results)
            report = self.generate_report(debt_items)
            sprint_allocation = self.calculate_sprint_allocation(debt_items)
            
            # Add sprint allocation to report
            report['sprint_allocation'] = sprint_allocation
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Successfully processed {len(debt_items)} debt items")
            logger.info(f"Report saved to: {output_path}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in results: {e}")
            self.generate_fallback_report(output_path, f"JSON decode error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to process results: {e}")
            self.generate_fallback_report(output_path, str(e))
            raise
    
    def categorize_findings(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Categorize findings based on configuration"""
        debt_items = []
        categories = self.config['debt-categories']
        
        try:
            # Process different types of analysis results
            if isinstance(results, list):
                # Handle list of items
                for item in results:
                    categorized_item = self.categorize_single_item(item, categories)
                    if categorized_item:
                        debt_items.append(categorized_item)
            elif isinstance(results, dict):
                # Handle structured results
                for analysis_type, items in results.items():
                    if isinstance(items, list):
                        for item in items:
                            item['analysis_type'] = analysis_type
                            categorized_item = self.categorize_single_item(item, categories)
                            if categorized_item:
                                debt_items.append(categorized_item)
                                
        except Exception as e:
            logger.error(f"Error categorizing findings: {e}")
            
        return debt_items
    
    def categorize_single_item(self, item: Dict[str, Any], categories: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Categorize a single debt item"""
        try:
            # Determine category based on item properties
            item_type = item.get('type', 'unknown')
            severity = item.get('severity', 'low')
            message = item.get('message', '').lower()
            
            # Map to debt categories
            category = self.determine_category(item_type, severity, message, categories)
            
            # Calculate estimated effort
            estimated_effort = self.calculate_effort(item, category, categories)
            
            # Calculate SLA date
            sla_days = categories[category]['sla-days']
            sla_date = datetime.now() + timedelta(days=sla_days)
            
            categorized_item = {
                'id': self.generate_item_id(item),
                'title': self.generate_title(item),
                'description': item.get('message', 'No description provided'),
                'type': item_type,
                'severity': severity,
                'category': category,
                'priority': categories[category]['priority'],
                'estimated_effort': estimated_effort,
                'file_path': item.get('file', item.get('file_path', 'unknown')),
                'line_number': item.get('line', item.get('line_number', 0)),
                'rule_id': item.get('rule', item.get('rule_id', 'unknown')),
                'created_date': datetime.now().isoformat(),
                'sla_date': sla_date.isoformat(),
                'labels': self.generate_labels(item, category, categories),
                'metadata': {
                    'analysis_type': item.get('analysis_type', 'unknown'),
                    'tool': item.get('tool', 'unknown'),
                    'confidence': item.get('confidence', 1.0)
                }
            }
            
            return categorized_item
            
        except Exception as e:
            logger.warning(f"Failed to categorize item: {e}, item: {item}")
            return None
    
    def determine_category(self, item_type: str, severity: str, message: str, categories: Dict[str, Any]) -> str:
        """Determine the appropriate category for a debt item"""
        
        # Security issues are always critical or high
        if item_type in ['security', 'vulnerability'] or 'security' in message:
            return 'critical' if severity in ['critical', 'high'] else 'high'
        
        # Performance issues
        if item_type in ['performance', 'memory-leak'] or any(word in message for word in ['performance', 'slow', 'memory', 'leak']):
            return 'high' if severity in ['critical', 'high'] else 'medium'
        
        # Accessibility issues
        if 'accessibility' in message or item_type == 'accessibility':
            return 'high'
        
        # Code quality issues
        if item_type in ['complexity', 'duplication', 'code-smell']:
            severity_mapping = {
                'critical': 'high',
                'high': 'medium',
                'medium': 'medium',
                'low': 'low'
            }
            return severity_mapping.get(severity, 'medium')
        
        # Documentation and style issues
        if item_type in ['documentation', 'style', 'formatting']:
            return 'low'
        
        # Default mapping based on severity
        severity_mapping = {
            'critical': 'critical',
            'high': 'high',  
            'medium': 'medium',
            'low': 'low',
            'error': 'high',
            'warning': 'medium'
        }
        
        return severity_mapping.get(severity, 'medium')
    
    def calculate_effort(self, item: Dict[str, Any], category: str, categories: Dict[str, Any]) -> float:
        """Calculate estimated effort for fixing the debt item"""
        base_effort = item.get('estimated_effort', 1)
        
        # Adjust based on category
        category_multipliers = {
            'critical': 3.0,
            'high': 2.0,
            'medium': 1.0,
            'low': 0.5
        }
        
        effort = base_effort * category_multipliers.get(category, 1.0)
        
        # Cap at max debt points from sprint config
        max_points = self.config['sprint-config']['max-debt-points']
        return min(effort, max_points)
    
    def generate_item_id(self, item: Dict[str, Any]) -> str:
        """Generate a unique ID for the debt item"""
        import hashlib
        
        content = f"{item.get('file', '')}{item.get('line', 0)}{item.get('message', '')}{item.get('rule', '')}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def generate_title(self, item: Dict[str, Any]) -> str:
        """Generate a descriptive title for the debt item"""
        item_type = item.get('type', 'Issue').title()
        file_path = item.get('file', item.get('file_path', ''))
        
        if file_path:
            file_name = os.path.basename(file_path)
            return f"{item_type} in {file_name}"
        else:
            rule = item.get('rule', item.get('rule_id', ''))
            if rule:
                return f"{item_type}: {rule}"
            else:
                return f"{item_type} detected"
    
    def generate_labels(self, item: Dict[str, Any], category: str, categories: Dict[str, Any]) -> List[str]:
        """Generate appropriate labels for the debt item"""
        labels = ['technical-debt', category]
        
        # Add category-specific labels
        category_labels = categories[category].get('labels', [])
        labels.extend(category_labels)
        
        # Add type-specific labels
        item_type = item.get('type')
        if item_type:
            labels.append(item_type)
        
        # Add component labels
        file_path = item.get('file', item.get('file_path', ''))
        if file_path:
            if 'frontend' in file_path or any(ext in file_path for ext in ['.js', '.ts', '.vue', '.jsx', '.tsx']):
                labels.append('frontend')
            elif 'backend' in file_path or any(ext in file_path for ext in ['.py', '.java', '.go', '.rs']):
                labels.append('backend')
        
        return list(set(labels))  # Remove duplicates
    
    def calculate_sprint_allocation(self, debt_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate sprint allocation recommendations"""
        sprint_config = self.config['sprint-config']
        total_capacity = sprint_config['default-capacity']
        debt_allocation_pct = sprint_config['debt-allocation-percentage'] / 100
        available_debt_capacity = total_capacity * debt_allocation_pct
        
        # Sort items by priority and effort
        sorted_items = sorted(debt_items, key=lambda x: (x['priority'], -x['estimated_effort']))
        
        # Allocate items to current sprint
        current_sprint = []
        current_effort = 0
        backlog = []
        
        for item in sorted_items:
            if current_effort + item['estimated_effort'] <= available_debt_capacity:
                current_sprint.append(item)
                current_effort += item['estimated_effort']
            else:
                backlog.append(item)
        
        return {
            'total_capacity': total_capacity,
            'debt_capacity': available_debt_capacity,
            'allocated_effort': current_effort,
            'utilization_percentage': (current_effort / available_debt_capacity) * 100 if available_debt_capacity > 0 else 0,
            'current_sprint_items': len(current_sprint),
            'backlog_items': len(backlog),
            'current_sprint': current_sprint[:10],  # Limit for output size
            'next_sprint_candidates': backlog[:5]
        }
    
    def generate_report(self, debt_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive debt report"""
        categories = self.config['debt-categories']
        
        # Calculate summary statistics
        category_counts = {}
        total_effort = 0
        
        for category in categories.keys():
            category_items = [item for item in debt_items if item['category'] == category]
            category_counts[category] = {
                'count': len(category_items),
                'effort': sum(item['estimated_effort'] for item in category_items)
            }
            total_effort += category_counts[category]['effort']
        
        # Generate trends (simplified - in real implementation, compare with historical data)
        trends = self.calculate_trends(debt_items)
        
        return {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'repository': os.environ.get('GITHUB_REPOSITORY', 'unknown'),
                'analysis_version': '1.0',
                'config_version': '1.0',
                'total_items': len(debt_items)
            },
            'summary': {
                'total_items': len(debt_items),
                'total_estimated_effort': total_effort,
                'category_breakdown': category_counts,
                'priority_distribution': self.calculate_priority_distribution(debt_items),
                'component_breakdown': self.calculate_component_breakdown(debt_items)
            },
            'trends': trends,
            'recommendations': self.generate_recommendations(debt_items, category_counts),
            'debt_items': debt_items
        }
    
    def calculate_trends(self, debt_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate debt trends (simplified version)"""
        # In a real implementation, this would compare with historical data
        return {
            'trend_direction': 'stable',  # Would be 'increasing', 'decreasing', or 'stable'
            'week_over_week_change': 0,   # Percentage change
            'new_items_this_period': len(debt_items),
            'resolved_items_this_period': 0,
            'net_change': len(debt_items)
        }
    
    def calculate_priority_distribution(self, debt_items: List[Dict[str, Any]]) -> Dict[int, int]:
        """Calculate distribution of items by priority"""
        distribution = {}
        for item in debt_items:
            priority = item['priority']
            distribution[priority] = distribution.get(priority, 0) + 1
        return distribution
    
    def calculate_component_breakdown(self, debt_items: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate breakdown by component/area"""
        breakdown = {}
        for item in debt_items:
            file_path = item.get('file_path', 'unknown')
            if 'frontend' in file_path or any(ext in file_path for ext in ['.js', '.ts', '.vue']):
                component = 'frontend'
            elif 'backend' in file_path or any(ext in file_path for ext in ['.py', '.java', '.go']):
                component = 'backend'
            else:
                component = 'other'
            
            breakdown[component] = breakdown.get(component, 0) + 1
        return breakdown
    
    def generate_recommendations(self, debt_items: List[Dict[str, Any]], category_counts: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Critical items
        critical_count = category_counts.get('critical', {}).get('count', 0)
        if critical_count > 0:
            recommendations.append(f"🚨 Address {critical_count} critical debt items immediately")
        
        # High priority items
        high_count = category_counts.get('high', {}).get('count', 0)
        if high_count > 5:
            recommendations.append(f"⚠️ High debt load: {high_count} high-priority items need attention")
        
        # Effort distribution
        total_effort = sum(category['effort'] for category in category_counts.values())
        if total_effort > self.config['sprint-config']['default-capacity']:
            recommendations.append("📊 Consider increasing debt allocation percentage in sprint planning")
        
        # Component-specific recommendations
        component_breakdown = self.calculate_component_breakdown(debt_items)
        max_component = max(component_breakdown.items(), key=lambda x: x[1]) if component_breakdown else None
        if max_component and max_component[1] > len(debt_items) * 0.6:
            recommendations.append(f"🎯 Focus debt reduction efforts on {max_component[0]} component")
        
        return recommendations
    
    def generate_fallback_report(self, output_path: str, error_message: str) -> None:
        """Generate fallback report when processing fails"""
        fallback_report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': error_message,
                'repository': os.environ.get('GITHUB_REPOSITORY', 'unknown')
            },
            'summary': {
                'total_items': 0,
                'total_estimated_effort': 0,
                'category_breakdown': {},
                'error_details': error_message
            },
            'debt_items': [],
            'recommendations': [
                "⚠️ Debt analysis failed - manual review required",
                "🔧 Check analysis tool configurations",
                "📋 Verify input data format and structure"
            ]
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(fallback_report, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write fallback report: {e}")

    def get_repository_info(self) -> Dict[str, str]:
        """Get repository information from environment"""
        return {
            'name': os.environ.get('GITHUB_REPOSITORY', 'unknown'),
            'run_id': os.environ.get('GITHUB_RUN_ID', 'unknown'),
            'run_number': os.environ.get('GITHUB_RUN_NUMBER', 'unknown'),
            'actor': os.environ.get('GITHUB_ACTOR', 'unknown'),
            'ref': os.environ.get('GITHUB_REF', 'unknown')
        }

def main():
    parser = argparse.ArgumentParser(description='Process technical debt findings')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--results', required=True, help='Analysis results JSON')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        processor = DebtProcessor(args.config)
        processor.process_results(args.results, args.output)
        
        logger.info("✅ Technical debt processing completed successfully")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"❌ Script execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()