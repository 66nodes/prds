#!/usr/bin/env python3
"""
AI-powered code review script using DeepSeek or OpenAI API
"""
import os
import sys
import json
import requests
import logging
from typing import Dict, List, Any, Optional
from github import Github
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AICodeReviewer:
    def __init__(self):
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.pr_number = int(os.getenv('PR_NUMBER', 0))
        self.review_focus = os.getenv('REVIEW_FOCUS', 'comprehensive')
        self.changed_files = os.getenv('CHANGED_FILES', '').split(',')
        
        # API configuration
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # Choose API based on availability
        self.api_provider = self._select_api_provider()
        
        # GitHub client
        self.github = Github(self.github_token)
        self.repo = self.github.get_repo(os.getenv('GITHUB_REPOSITORY'))
        self.pr = self.repo.get_pull(self.pr_number) if self.pr_number else None
        
    def _select_api_provider(self) -> str:
        """Select API provider based on available keys"""
        if self.deepseek_api_key:
            return 'deepseek'
        elif self.openai_api_key:
            return 'openai'
        else:
            logger.error("No AI API key provided")
            sys.exit(1)
            
    def get_system_prompt(self) -> str:
        """Load and return the system prompt for code review"""
        prompt_file = '.github/config/code-reviewer-prompt.md'
        default_prompt = """You are an expert code reviewer. Analyze code for:

1. **Code Quality**: 
   - SOLID principles adherence
   - Clean code practices
   - Readability and maintainability

2. **Security**:
   - SQL injection risks
   - XSS vulnerabilities
   - Authentication/Authorization issues
   - Data exposure risks

3. **Performance**:
   - N+1 query problems
   - Inefficient algorithms
   - Memory management issues
   - Proper caching implementation

4. **Best Practices**:
   - Error handling
   - Logging
   - Testing coverage
   - Documentation

Provide specific, actionable feedback with code examples when appropriate. Rate severity:
- 🔴 Critical: Must fix immediately
- 🟡 Warning: Should address soon
- 🔵 Suggestion: Optional improvement

Focus on the most important issues first. Be constructive and professional."""
        
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r') as f:
                return f.read()
        return default_prompt
        
    def get_pr_diff(self) -> List[Dict[str, Any]]:
        """Get the diff for changed files in the PR"""
        if not self.pr:
            return []
            
        files_data = []
        for file in self.pr.get_files():
            if file.filename in self.changed_files or not self.changed_files[0]:
                files_data.append({
                    'filename': file.filename,
                    'patch': file.patch if file.patch else '',
                    'additions': file.additions,
                    'deletions': file.deletions,
                    'changes': file.changes,
                    'status': file.status
                })
        return files_data
        
    def analyze_with_deepseek(self, code_diff: str) -> Dict[str, Any]:
        """Analyze code using DeepSeek API"""
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-coder",
            "messages": [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": f"Review focus: {self.review_focus}\n\nCode diff:\n{code_diff}"}
            ],
            "temperature": 0.1,
            "max_tokens": 4000
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            return {"error": str(e)}
            
    def analyze_with_openai(self, code_diff: str) -> Dict[str, Any]:
        """Analyze code using OpenAI API"""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4-turbo-preview",
            "messages": [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": f"Review focus: {self.review_focus}\n\nCode diff:\n{code_diff}"}
            ],
            "temperature": 0.1,
            "max_tokens": 4000
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {"error": str(e)}
            
    def analyze_code(self, code_diff: str) -> str:
        """Analyze code using selected API provider"""
        if self.api_provider == 'deepseek':
            result = self.analyze_with_deepseek(code_diff)
        else:
            result = self.analyze_with_openai(code_diff)
            
        if 'error' in result:
            return f"Analysis error: {result['error']}"
            
        try:
            return result['choices'][0]['message']['content']
        except (KeyError, IndexError):
            return "Unable to extract review content from API response"
            
    def batch_analyze_files(self, files_data: List[Dict[str, Any]], batch_size: int = 5) -> List[Dict[str, str]]:
        """Analyze files in batches to avoid API limits"""
        reviews = []
        
        for i in range(0, len(files_data), batch_size):
            batch = files_data[i:i + batch_size]
            
            for file_data in batch:
                if not file_data['patch']:
                    continue
                    
                logger.info(f"Analyzing {file_data['filename']}...")
                
                # Truncate very large diffs
                diff = file_data['patch']
                if len(diff) > 10000:
                    diff = diff[:10000] + "\n... (truncated)"
                    
                review = self.analyze_code(diff)
                reviews.append({
                    'filename': file_data['filename'],
                    'review': review,
                    'changes': file_data['changes']
                })
                
                # Rate limiting
                time.sleep(1)
                
        return reviews
        
    def categorize_issues(self, reviews: List[Dict[str, str]]) -> Dict[str, Any]:
        """Categorize issues from reviews"""
        categories = {
            'critical': [],
            'warnings': [],
            'suggestions': [],
            'labels': set()
        }
        
        for review_item in reviews:
            review_text = review_item['review'].lower()
            
            # Detect severity levels
            if '🔴' in review_item['review'] or 'critical' in review_text:
                categories['critical'].append(review_item['filename'])
                categories['labels'].add('needs-immediate-fix')
                
            if '🟡' in review_item['review'] or 'warning' in review_text:
                categories['warnings'].append(review_item['filename'])
                categories['labels'].add('needs-attention')
                
            # Detect issue types for labeling
            if 'security' in review_text:
                categories['labels'].add('security-review')
            if 'performance' in review_text:
                categories['labels'].add('performance-review')
            if 'test' in review_text:
                categories['labels'].add('needs-tests')
                
        categories['labels'] = list(categories['labels'])
        return categories
        
    def format_review_summary(self, reviews: List[Dict[str, str]], categories: Dict[str, Any]) -> str:
        """Format the review summary for posting"""
        summary = "## 🤖 AI Code Review Summary\n\n"
        summary += f"**Review Focus**: {self.review_focus.title()}\n"
        summary += f"**Files Reviewed**: {len(reviews)}\n"
        summary += f"**Timestamp**: {datetime.utcnow().isoformat()}Z\n\n"
        
        # Add severity summary
        if categories['critical']:
            summary += f"### 🔴 Critical Issues Found ({len(categories['critical'])})\n"
            for file in categories['critical'][:5]:
                summary += f"- {file}\n"
            summary += "\n"
            
        if categories['warnings']:
            summary += f"### 🟡 Warnings ({len(categories['warnings'])})\n"
            for file in categories['warnings'][:5]:
                summary += f"- {file}\n"
            summary += "\n"
            
        # Add individual file reviews
        summary += "### 📄 Detailed Reviews\n\n"
        
        for review_item in reviews[:10]:  # Limit to prevent comment size issues
            summary += f"<details>\n<summary><b>{review_item['filename']}</b> ({review_item['changes']} changes)</summary>\n\n"
            summary += review_item['review']
            summary += "\n\n</details>\n\n"
            
        # Add footer
        summary += "---\n"
        summary += "*This review was generated by AI and should be validated by human reviewers.*\n"
        summary += f"*API Provider: {self.api_provider.title()}*"
        
        return summary
        
    def save_results(self, reviews: List[Dict[str, str]], categories: Dict[str, Any], summary: str):
        """Save review results to files"""
        # Save JSON output
        output = {
            'timestamp': datetime.utcnow().isoformat(),
            'pr_number': self.pr_number,
            'review_focus': self.review_focus,
            'files_reviewed': len(reviews),
            'categories': categories,
            'labels': categories['labels'],
            'summary': summary
        }
        
        with open('review-output.json', 'w') as f:
            json.dump(output, f, indent=2)
            
        # Save detailed markdown
        with open('review-details.md', 'w') as f:
            f.write(summary)
            f.write("\n\n## Full Reviews\n\n")
            for review_item in reviews:
                f.write(f"### {review_item['filename']}\n\n")
                f.write(review_item['review'])
                f.write("\n\n---\n\n")
                
    def run(self):
        """Main execution function"""
        try:
            logger.info(f"Starting AI code review for PR #{self.pr_number}")
            logger.info(f"Review focus: {self.review_focus}")
            logger.info(f"API provider: {self.api_provider}")
            
            # Get PR diff
            files_data = self.get_pr_diff()
            if not files_data:
                logger.warning("No files to review")
                return
                
            logger.info(f"Found {len(files_data)} files to review")
            
            # Analyze files
            reviews = self.batch_analyze_files(files_data)
            
            # Categorize issues
            categories = self.categorize_issues(reviews)
            
            # Format summary
            summary = self.format_review_summary(reviews, categories)
            
            # Save results
            self.save_results(reviews, categories, summary)
            
            logger.info("Code review completed successfully")
            
        except Exception as e:
            logger.error(f"Code review failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    reviewer = AICodeReviewer()
    reviewer.run()