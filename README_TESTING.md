# Testing Documentation for Strategic Planning Platform

## Overview

This document provides comprehensive information about the testing infrastructure, test execution,
and quality assurance processes for the Strategic Planning Platform.

## 🧪 Test Suite Architecture

### Test Categories

1. **Unit Tests** (`backend/tests/unit/`)
   - Test individual functions and classes in isolation
   - Target: >90% code coverage
   - Fast execution (<30 seconds)

2. **Integration Tests** (`backend/tests/integration/`)
   - Test API endpoints and service interactions
   - Database and external service integration
   - Medium execution time (1-3 minutes)

3. **End-to-End Tests** (`tests/e2e/`)
   - Complete user journey testing with Playwright
   - Cross-browser compatibility testing
   - Longer execution time (5-10 minutes)

4. **Performance Tests** (`tests/performance/`)
   - Load testing with Locust
   - API response time validation
   - Throughput and scalability testing

5. **Security Tests** (`backend/tests/security/`)
   - Authentication and authorization testing
   - Input validation and injection prevention
   - Data protection and compliance testing

6. **Hallucination Validation Tests** (`backend/tests/validation/`)
   - GraphRAG system validation
   - Content quality and accuracy testing
   - AI model output verification

## 🚀 Quick Start

### Running All Tests

```bash
# Execute the comprehensive test suite
./scripts/run_all_tests.sh
```

### Running Specific Test Categories

```bash
# Backend unit tests
cd backend
source venv/bin/activate
pytest tests/unit/ --cov=backend --cov-report=html

# Integration tests
pytest tests/integration/ -v

# Security tests
pytest tests/security/ -v

# Hallucination validation tests
pytest tests/validation/ -v

# E2E tests
cd frontend
npx playwright test

# Performance tests
cd backend
locust -f tests/performance/locustfile.py --host=http://localhost:8000
```

## 📊 Coverage Reports

### Generating Coverage Reports

```bash
# Generate comprehensive coverage report
./scripts/generate_test_coverage_report.py --type all

# Generate unit test coverage only
./scripts/generate_test_coverage_report.py --type unit

# Generate integration test coverage only
./scripts/generate_test_coverage_report.py --type integration
```

### Coverage Targets

- **Minimum Coverage**: 75%
- **Target Coverage**: 90%
- **Security Components**: 95%
- **Critical Business Logic**: 95%

### Viewing Coverage Reports

Coverage reports are generated in multiple formats:

- **HTML Report**: `test-results/coverage_report_latest.html`
- **JSON Report**: `test-results/coverage_report_latest.json`
- **Terminal Output**: Displayed during test execution

## ⚡ Performance Benchmarking

### Running Performance Analysis

```bash
# Generate performance benchmark report
./scripts/test_performance_benchmark.py
```

### Performance Targets

#### Test Execution Times

- **Unit Tests**: <30 seconds (excellent), <60 seconds (good)
- **Integration Tests**: <60 seconds (excellent), <180 seconds (good)
- **E2E Tests**: <180 seconds (excellent), <300 seconds (good)
- **Performance Tests**: <120 seconds (excellent), <180 seconds (good)

#### API Performance

- **Response Time**: <200ms (95th percentile)
- **Throughput**: >100 requests/second
- **Error Rate**: <1%
- **Concurrent Users**: Support 1,000+ concurrent users

## 🛡️ Security Testing

### Security Test Categories

1. **Authentication Security**
   - Password strength validation
   - JWT token security
   - Session management
   - Brute force protection

2. **API Security**
   - Input validation (SQL injection, XSS, etc.)
   - Rate limiting
   - Authorization controls
   - Data sanitization

3. **Data Protection**
   - Encryption at rest and in transit
   - PII detection and masking
   - GDPR compliance
   - Audit logging

### Running Security Tests

```bash
# Run all security tests
cd backend
pytest tests/security/ -v

# Run specific security test categories
pytest tests/security/test_authentication_security.py -v
pytest tests/security/test_api_security.py -v
pytest tests/security/test_data_protection.py -v
```

## 🔍 Hallucination Validation Testing

### Purpose

The hallucination validation tests ensure that the GraphRAG system:

- Accurately detects hallucinated content
- Maintains quality thresholds (<2% hallucination rate)
- Properly validates generated PRDs
- Provides reliable fact-checking

### Test Categories

1. **Hallucination Detection Tests**
   - Content validation against knowledge graph
   - Entity extraction and verification
   - Fact-checking accuracy

2. **GraphRAG Validation Tests**
   - Knowledge graph integration
   - Semantic similarity validation
   - Cross-domain validation

3. **Quality Metrics Tests**
   - Content quality assessment
   - Validation scoring algorithms
   - Performance benchmarking

### Running Validation Tests

```bash
cd backend
pytest tests/validation/ -v --tb=short
```

## 🎭 End-to-End Testing

### Test Structure

E2E tests simulate complete user workflows:

1. **Authentication Flow**
   - User registration and login
   - Password reset
   - Session management

2. **PRD Generation Workflow**
   - Complete PRD creation process
   - Validation and regeneration
   - Export functionality

3. **Collaborative Features**
   - Review and approval workflow
   - Comment and feedback system
   - Real-time collaboration

### Running E2E Tests

```bash
cd frontend

# Run all E2E tests
npx playwright test

# Run tests in headed mode (for debugging)
npx playwright test --headed

# Run specific test file
npx playwright test tests/e2e/test_user_journey_complete_prd_workflow.js

# Generate test report
npx playwright test --reporter=html
```

### E2E Test Configuration

- **Browsers**: Chromium, Firefox, WebKit
- **Viewports**: Desktop (1920x1080), Mobile (375x667)
- **Network**: Fast 3G simulation for performance testing
- **Screenshots**: Captured on test failure
- **Videos**: Recorded for failed tests

## 📈 Test Data Management

### Test Database Setup

```bash
# Set up test database
export DATABASE_URL="postgresql://test_user:test_pass@localhost:5432/test_prds_db"
export TESTING=true

# Run database migrations for testing
alembic upgrade head
```

### Test Data Fixtures

Test data is managed through fixtures:

- **User Fixtures**: Test users with different roles
- **Project Fixtures**: Sample projects and PRDs
- **Authentication Fixtures**: Valid and invalid tokens
- **GraphRAG Fixtures**: Knowledge graph test data

### Cleaning Test Data

```bash
# Clean up test database after tests
pytest --tb=no --disable-warnings tests/cleanup.py
```

## 🚨 Quality Gates

### Automated Quality Checks

The test suite enforces the following quality gates:

1. **Code Coverage**: Minimum 75%, target 90%
2. **Test Success Rate**: 100% (all tests must pass)
3. **Performance**: API response times <200ms
4. **Security**: No high-severity security issues
5. **Hallucination Rate**: <2% for generated content

### Quality Gate Enforcement

Quality gates are enforced in:

- Pre-commit hooks
- CI/CD pipelines
- Pull request validation
- Release deployment

## 🔧 Test Configuration

### Backend Test Configuration

```python
# pytest.ini configuration
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --strict-markers
    --disable-socket
    --allow-unix-socket
    --cov=backend
    --cov-report=term-missing
    --cov-report=html
    --cov-fail-under=75
```

### Frontend Test Configuration

```javascript
// playwright.config.js
module.exports = {
  testDir: './tests/e2e',
  timeout: 30000,
  expect: {
    timeout: 5000,
  },
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'html',
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
  ],
};
```

## 🐛 Debugging Tests

### Backend Test Debugging

```bash
# Run tests with verbose output
pytest tests/unit/ -v -s

# Run specific test with debugging
pytest tests/unit/test_auth_service.py::TestAuthService::test_hash_password -v -s

# Debug with pdb
pytest tests/unit/test_auth_service.py --pdb

# Run tests with coverage and show missing lines
pytest tests/unit/ --cov=backend --cov-report=term-missing
```

### Frontend Test Debugging

```bash
# Run Playwright tests in debug mode
npx playwright test --debug

# Run tests with browser UI visible
npx playwright test --headed

# Generate and open test report
npx playwright test --reporter=html
npx playwright show-report

# Debug specific test
npx playwright test tests/e2e/test_user_journey_complete_prd_workflow.js --debug
```

## 📊 Continuous Integration

### CI/CD Test Pipeline

The CI/CD pipeline runs tests in the following order:

1. **Linting and Code Quality**
   - ESLint (frontend)
   - Black, isort, flake8 (backend)
   - Type checking (mypy)

2. **Unit Tests**
   - Backend unit tests with coverage
   - Fast feedback for code changes

3. **Integration Tests**
   - API endpoint testing
   - Database integration testing

4. **Security Tests**
   - Authentication and authorization
   - Input validation testing
   - Dependency vulnerability scanning

5. **E2E Tests**
   - Cross-browser testing
   - User workflow validation

6. **Performance Tests**
   - Load testing and benchmarking
   - Response time validation

### Test Environment Setup

```yaml
# GitHub Actions workflow example
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test_pass
          POSTGRES_DB: test_prds_db
        options:
          --health-cmd pg_isready --health-interval 10s --health-timeout 5s --health-retries 5
      redis:
        image: redis:7
        options:
          --health-cmd "redis-cli ping" --health-interval 10s --health-timeout 5s --health-retries 5

    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r backend/requirements-test.txt
      - name: Run tests
        run: ./scripts/run_all_tests.sh
```

## 📝 Test Reporting

### Report Generation

Test reports are automatically generated in multiple formats:

1. **HTML Coverage Report**: Interactive coverage visualization
2. **JSON Coverage Data**: Machine-readable coverage metrics
3. **JUnit XML**: CI/CD integration format
4. **Performance Benchmarks**: Response time and throughput analysis
5. **Security Audit Reports**: Vulnerability and compliance status

### Report Locations

- `test-results/coverage/`: Coverage reports
- `test-results/performance/`: Performance test results
- `test-results/e2e/`: End-to-end test reports
- `test-results/security/`: Security test results
- `test-results/validation/`: Hallucination validation reports

## 🔄 Maintenance

### Regular Maintenance Tasks

1. **Update Test Dependencies**

   ```bash
   # Update Python test dependencies
   pip-compile backend/requirements-test.in

   # Update Node.js test dependencies
   npm update --dev
   ```

2. **Review and Update Test Data**
   - Update test fixtures with realistic data
   - Refresh knowledge graph test data
   - Update security test payloads

3. **Performance Baseline Updates**
   - Review and update performance thresholds
   - Update benchmark comparisons
   - Analyze performance trends

4. **Test Suite Optimization**
   - Identify and optimize slow tests
   - Remove obsolete tests
   - Improve test isolation

### Test Metrics Monitoring

Monitor the following metrics regularly:

- Test execution time trends
- Code coverage trends
- Test failure rates
- Performance benchmark results
- Security test results

## 📚 Additional Resources

- [PyTest Documentation](https://docs.pytest.org/)
- [Playwright Documentation](https://playwright.dev/)
- [Locust Documentation](https://locust.io/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

## 🤝 Contributing

### Writing New Tests

When adding new features, include:

1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test API endpoints and services
3. **Security Tests**: Validate security controls
4. **E2E Tests**: Test critical user workflows
5. **Documentation**: Update test documentation

### Test Review Checklist

- [ ] Tests are independent and isolated
- [ ] Tests have descriptive names and docstrings
- [ ] Test data is properly managed with fixtures
- [ ] Tests cover both success and failure scenarios
- [ ] Performance impact is considered
- [ ] Security implications are tested
- [ ] Tests are maintainable and readable

---

For questions or issues with the testing infrastructure, please refer to the main project
documentation or contact the development team.
