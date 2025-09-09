# GitHub Actions Workflows Documentation

This directory contains comprehensive GitHub Actions workflows that provide AI-powered development
automation, quality assurance, and technical debt management for the Strategic Planning Platform.

## 📁 Directory Structure

```
.github/
├── workflows/
│   ├── ai-code-review.yml           # AI-powered code reviews with DeepSeek/OpenAI
│   ├── dependency-updates.yml       # Automated dependency management
│   ├── workflow-debugging.yml       # Intelligent workflow failure analysis
│   ├── tech-debt-analysis.yml       # Technical debt identification and tracking
│   └── debt-monitoring.yml          # Technical debt dashboard and monitoring
├── scripts/
│   ├── ai_code_review.py           # AI code review implementation
│   ├── process_debt_findings.py    # Technical debt processing logic
│   └── error_handler.sh            # Comprehensive error handling utilities
├── config/
│   ├── tech-debt-config.yml        # Technical debt management configuration
│   └── code-reviewer-prompt.md     # AI code reviewer system prompt
└── README.md                       # This documentation
```

## 🤖 AI-Powered Code Review

### Features

- **Multi-Provider Support**: DeepSeek API (primary) with OpenAI fallback
- **Comprehensive Analysis**: Security, performance, code quality, and best practices
- **Intelligent Categorization**: Automatic issue severity classification and labeling
- **Batch Processing**: Handles large PRs with rate limiting and optimization
- **Contextual Reviews**: Custom prompts based on review focus and file types

### Usage

```yaml
# Automatic trigger
on:
  pull_request:
    types: [opened, synchronize, reopened]

# Manual trigger with custom focus
workflow_dispatch:
  inputs:
    review-focus:
      description: 'Review focus (security, performance, quality)'
      default: 'comprehensive'
```

### Configuration

Set up required secrets in your repository:

- `DEEPSEEK_API_KEY` (recommended)
- `OPENAI_API_KEY` (fallback)

### Output

- Detailed code review comments on PRs
- Automatic labeling based on findings
- Severity classification (🔴 Critical, 🟡 Warning, 🔵 Suggestion)
- Downloadable analysis reports

## 📦 Dependency Updates

### Features

- **Automated Updates**: Weekly security updates with manual override options
- **Multi-Language Support**: Node.js (frontend) and Python (backend) dependencies
- **Security-First Approach**: Prioritizes security patches and vulnerability fixes
- **Intelligent Batching**: Limits PRs to prevent overwhelming reviewers
- **Auto-Merge Capability**: Optional auto-merge for passing tests

### Update Types

- **Security**: Critical vulnerability fixes (default)
- **Patch**: Bug fixes and patch version updates
- **Minor**: Feature additions and minor version updates
- **Major**: Breaking changes and major version updates

### Configuration

```yaml
# Weekly schedule
schedule:
  - cron: '0 10 * * 1' # Monday 10 AM UTC

# Manual execution
workflow_dispatch:
  inputs:
    update-type: 'security|patch|minor|major'
    auto-merge: boolean
```

## 🔧 Workflow Debugging

### Features

- **Intelligent Failure Detection**: Automatically identifies critical workflow failures
- **AI-Powered Analysis**: Uses DeepSeek/OpenAI to analyze logs and provide solutions
- **Pattern Recognition**: Detects common error patterns and suggests fixes
- **Automatic Issue Creation**: Creates GitHub issues with detailed analysis and recommendations

### Trigger Conditions

- 2+ consecutive failures in any workflow
- Single failure in critical workflows (deploy, release, security)
- Manual execution for specific workflow run analysis

### Analysis Depth

- **Basic**: Standard error pattern recognition
- **Standard**: AI-powered root cause analysis (default)
- **Deep**: Comprehensive system analysis with historical context

## 📊 Technical Debt Management

### Architecture

The technical debt management system consists of three main workflows:

1. **Analysis** (`tech-debt-analysis.yml`): Identifies and categorizes debt
2. **Monitoring** (`debt-monitoring.yml`): Tracks metrics and generates dashboards
3. **Processing** (Python scripts): Processes findings and manages sprint allocation

### Debt Categories

```yaml
critical: # Security, crashes, data loss (3-day SLA)
high: # Performance, accessibility (7-day SLA)
medium: # Maintainability, complexity (14-day SLA)
low: # Style, documentation (30-day SLA)
```

### Analysis Tools

- **Frontend**: ESLint, TypeScript compiler, code duplication detection
- **Backend**: Bandit (security), Pylint (quality), MyPy (types), Radon (complexity)
- **Universal**: Custom pattern analysis, historical trend tracking

### Sprint Integration

- Automatic calculation of debt allocation (default: 20% of sprint capacity)
- Smart prioritization based on severity and business impact
- GitHub issue creation with proper labeling and assignment
- Sprint milestone integration

## 📈 Monitoring & Dashboards

### Metrics Collected

- **Issue Metrics**: Total, open, closed, resolution time
- **Category Distribution**: By severity and type
- **Age Analysis**: Issue aging patterns and SLA compliance
- **Trend Analysis**: Historical data and directional indicators
- **Health Score**: Composite score (0-100) based on multiple factors

### Dashboard Features

- **Automated Generation**: Weekly updates with version history
- **Multiple Formats**: Markdown, JSON, CSV exports
- **Historical Tracking**: 12-month data retention with trend analysis
- **Alert System**: Issues created for health scores below thresholds

### Health Score Calculation

```
Base Score: 100
- High open issue ratio: -30 points max
- Old issues (>3 months): -20 points max
- Increasing trend: -15 points max
- Slow resolution (>30 days): -10 points max
```

## 🛠️ Configuration

### Required Secrets

```yaml
# AI Services (at least one required)
DEEPSEEK_API_KEY: 'your-deepseek-api-key'
OPENAI_API_KEY: 'your-openai-api-key'

# GitHub (automatically provided)
GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### Configuration Files

#### Technical Debt Config (`config/tech-debt-config.yml`)

```yaml
debt-categories:
  critical:
    labels: ['security', 'crash', 'data-loss']
    priority: 1
    sla-days: 3

sprint-config:
  default-capacity: 40
  debt-allocation-percentage: 20

analysis:
  sonarqube: { enabled: true }
  eslint: { enabled: true }
  python: { enabled: true }

rate-limiting:
  api-calls-per-hour: 1000
  batch-size: 30
```

#### Code Reviewer Prompt (`config/code-reviewer-prompt.md`)

Comprehensive system prompt for AI code reviewers covering:

- Code quality and architecture principles
- Security analysis guidelines
- Performance optimization criteria
- Best practices and standards
- Framework-specific guidance

## 📋 Best Practices

### Workflow Optimization

1. **Resource Management**: Use caching for dependencies and build artifacts
2. **Parallel Execution**: Run independent jobs concurrently
3. **Smart Triggering**: Avoid unnecessary workflow runs with path filters
4. **Error Handling**: Use the provided error handler for consistent failure management

### Security Considerations

1. **Secret Management**: Never hardcode API keys or tokens
2. **Permissions**: Use minimal required permissions for each job
3. **Validation**: Validate all external inputs and API responses
4. **Audit Logging**: Comprehensive logging for security and compliance

### Performance Guidelines

1. **Caching Strategy**: Implement multi-level caching (dependencies, analysis results, build
   artifacts)
2. **Batch Processing**: Process large datasets in configurable batches
3. **Rate Limiting**: Respect API rate limits with intelligent backoff
4. **Resource Monitoring**: Track job duration and resource usage

### Quality Assurance

1. **Testing**: All scripts include comprehensive error handling and validation
2. **Documentation**: Keep documentation updated with configuration changes
3. **Monitoring**: Regular health checks and performance monitoring
4. **Feedback Loops**: Use workflow results to continuously improve processes

## 🚀 Getting Started

### 1. Initial Setup

```bash
# Copy workflows to your repository
cp -r .github/ /path/to/your/repository/

# Set up required secrets
gh secret set DEEPSEEK_API_KEY --body "your-api-key"
gh secret set OPENAI_API_KEY --body "your-backup-api-key"
```

### 2. Customize Configuration

Edit the configuration files to match your project needs:

- Update debt categories and SLA requirements
- Adjust sprint capacity and allocation percentages
- Enable/disable analysis tools based on your tech stack
- Customize the AI reviewer prompt for your coding standards

### 3. Enable Workflows

Workflows will automatically trigger based on their configured events:

- **Code Reviews**: Trigger on every pull request
- **Dependencies**: Weekly automated updates
- **Debt Analysis**: Weekly technical debt assessment
- **Debugging**: Automatic on workflow failures
- **Monitoring**: Weekly dashboard updates

### 4. Monitor and Iterate

- Review the technical debt dashboard weekly
- Monitor workflow performance and adjust configurations
- Update AI prompts based on review quality feedback
- Adjust debt categories and SLAs based on team capacity

## 📞 Support & Troubleshooting

### Common Issues

#### API Rate Limits

- **Symptom**: 429 errors from AI APIs
- **Solution**: Reduce batch sizes or increase delays in configuration
- **Prevention**: Monitor usage and consider multiple API keys

#### Large PR Analysis

- **Symptom**: Timeouts on very large pull requests
- **Solution**: Files are automatically filtered and truncated
- **Configuration**: Adjust `MAX_FILES_PER_REVIEW` and `MAX_LINES_PER_REVIEW`

#### Disk Space Issues

- **Symptom**: Workflows failing with disk space errors
- **Solution**: Automatic cleanup is implemented in all workflows
- **Monitoring**: Disk usage is checked in error handler

### Debugging Workflows

Use the workflow debugging assistant for automatic analysis:

```yaml
# Trigger manual analysis
workflow_dispatch:
  inputs:
    workflow-run-id: '123456789'
    failure-analysis-depth: 'deep'
```

### Performance Monitoring

All workflows include built-in performance monitoring:

- Execution time tracking
- Resource usage monitoring
- Success/failure rate tracking
- Historical performance trends

## 🔄 Version History

### v1.0.0 (Current)

- Initial implementation of all core workflows
- AI-powered code reviews with multi-provider support
- Comprehensive technical debt management system
- Automated dependency updates with security focus
- Intelligent workflow debugging and monitoring

### Roadmap

- [ ] Integration with external quality tools (SonarQube, CodeClimate)
- [ ] Advanced metrics and predictive analytics
- [ ] Slack/Teams notification integration
- [ ] Custom webhook support for external integrations
- [ ] Multi-repository debt tracking and aggregation

## 📄 License

This GitHub Actions implementation is part of the Strategic Planning Platform project and follows
the same license terms.

---

For questions, issues, or contributions, please refer to the main project documentation or create an
issue in the repository.
