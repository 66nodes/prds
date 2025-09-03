#!/bin/bash

# Comprehensive Test Suite Runner for Strategic Planning Platform
# This script runs all test categories and generates comprehensive reports

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT=$(pwd)
BACKEND_DIR="backend"
FRONTEND_DIR="frontend"
REPORTS_DIR="test-results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Test configuration
MIN_COVERAGE=90
PERFORMANCE_TIMEOUT=300  # 5 minutes
E2E_TIMEOUT=600         # 10 minutes

echo -e "${BLUE}🚀 Starting Comprehensive Test Suite - $(date)${NC}"
echo "======================================================================"

# Create reports directory
mkdir -p ${REPORTS_DIR}/{coverage,performance,security,e2e,screenshots}

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}$1${NC}"
    echo "----------------------------------------------------------------------"
}

# Function to check if command exists
check_dependency() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}❌ $1 is required but not installed${NC}"
        exit 1
    fi
}

# Function to run command with timeout and logging
run_with_timeout() {
    local timeout=$1
    local description=$2
    shift 2
    
    echo -e "${YELLOW}⏳ Running: $description${NC}"
    
    if timeout $timeout "$@"; then
        echo -e "${GREEN}✅ Success: $description${NC}"
        return 0
    else
        local exit_code=$?
        echo -e "${RED}❌ Failed: $description (exit code: $exit_code)${NC}"
        return $exit_code
    fi
}

# Function to calculate test duration
calculate_duration() {
    local start_time=$1
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "${duration}s"
}

print_section "📋 Pre-flight Checks"

# Check dependencies
echo "Checking dependencies..."
check_dependency "python3"
check_dependency "npm"
check_dependency "docker"

# Check if backend dependencies are installed
if [ ! -f "${BACKEND_DIR}/requirements.txt" ]; then
    echo -e "${RED}❌ Backend requirements.txt not found${NC}"
    exit 1
fi

if [ ! -f "${BACKEND_DIR}/requirements-test.txt" ]; then
    echo -e "${RED}❌ Backend test requirements not found${NC}"
    exit 1
fi

# Check if frontend dependencies are installed
if [ ! -f "${FRONTEND_DIR}/package.json" ]; then
    echo -e "${RED}❌ Frontend package.json not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All dependencies check passed${NC}"

print_section "🔧 Environment Setup"

# Setup backend test environment
echo "Setting up backend test environment..."
cd $BACKEND_DIR

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install test dependencies
echo "Installing backend test dependencies..."
pip install -r requirements-test.txt

# Setup test database
echo "Setting up test database..."
export DATABASE_URL="postgresql://test_user:test_pass@localhost:5432/test_prds_db"
export TESTING=true
export SECRET_KEY="test-secret-key-for-testing-only"

# Start test services if needed (Redis, PostgreSQL)
echo "Starting test services..."
if ! pgrep -f redis-server > /dev/null; then
    echo "Starting Redis for testing..."
    redis-server --daemonize yes --port 6380
fi

cd $PROJECT_ROOT

# Setup frontend test environment
echo "Setting up frontend test environment..."
cd $FRONTEND_DIR

echo "Installing frontend dependencies..."
npm ci

# Install Playwright browsers if not already installed
if [ ! -d "node_modules/.bin" ] || ! npx playwright --version > /dev/null 2>&1; then
    echo "Installing Playwright browsers..."
    npx playwright install
fi

cd $PROJECT_ROOT

echo -e "${GREEN}✅ Environment setup completed${NC}"

# Initialize test results summary
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
TEST_START_TIME=$(date +%s)

print_section "🧪 Backend Unit Tests"

cd $BACKEND_DIR
source venv/bin/activate

UNIT_START_TIME=$(date +%s)

# Run unit tests with coverage
echo "Running backend unit tests with coverage..."
run_with_timeout 300 "Backend Unit Tests" \
    python -m pytest tests/unit/ \
    --cov=backend \
    --cov-report=html:../${REPORTS_DIR}/coverage/backend-unit \
    --cov-report=xml:../${REPORTS_DIR}/coverage/backend-unit.xml \
    --cov-report=term \
    --cov-fail-under=${MIN_COVERAGE} \
    --junit-xml=../${REPORTS_DIR}/backend-unit-results.xml \
    -v

UNIT_EXIT_CODE=$?
UNIT_DURATION=$(calculate_duration $UNIT_START_TIME)

if [ $UNIT_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Backend unit tests passed in $UNIT_DURATION${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${RED}❌ Backend unit tests failed in $UNIT_DURATION${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

TOTAL_TESTS=$((TOTAL_TESTS + 1))
cd $PROJECT_ROOT

print_section "🔗 Backend Integration Tests"

cd $BACKEND_DIR
source venv/bin/activate

INTEGRATION_START_TIME=$(date +%s)

# Run integration tests
echo "Running backend integration tests..."
run_with_timeout 300 "Backend Integration Tests" \
    python -m pytest tests/integration/ \
    --cov=backend \
    --cov-report=html:../${REPORTS_DIR}/coverage/backend-integration \
    --cov-report=xml:../${REPORTS_DIR}/coverage/backend-integration.xml \
    --junit-xml=../${REPORTS_DIR}/backend-integration-results.xml \
    -v

INTEGRATION_EXIT_CODE=$?
INTEGRATION_DURATION=$(calculate_duration $INTEGRATION_START_TIME)

if [ $INTEGRATION_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Backend integration tests passed in $INTEGRATION_DURATION${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${RED}❌ Backend integration tests failed in $INTEGRATION_DURATION${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

TOTAL_TESTS=$((TOTAL_TESTS + 1))
cd $PROJECT_ROOT

print_section "🛡️ Security Tests"

cd $BACKEND_DIR
source venv/bin/activate

SECURITY_START_TIME=$(date +%s)

# Run security tests
echo "Running security tests..."
run_with_timeout 300 "Security Tests" \
    python -m pytest tests/security/ \
    --junit-xml=../${REPORTS_DIR}/security-results.xml \
    -v

SECURITY_EXIT_CODE=$?
SECURITY_DURATION=$(calculate_duration $SECURITY_START_TIME)

if [ $SECURITY_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Security tests passed in $SECURITY_DURATION${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${RED}❌ Security tests failed in $SECURITY_DURATION${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

TOTAL_TESTS=$((TOTAL_TESTS + 1))
cd $PROJECT_ROOT

print_section "🔍 Hallucination Validation Tests"

cd $BACKEND_DIR
source venv/bin/activate

VALIDATION_START_TIME=$(date +%s)

# Run hallucination validation tests
echo "Running hallucination validation tests..."
run_with_timeout 300 "Hallucination Validation Tests" \
    python -m pytest tests/validation/ \
    --junit-xml=../${REPORTS_DIR}/validation-results.xml \
    -v

VALIDATION_EXIT_CODE=$?
VALIDATION_DURATION=$(calculate_duration $VALIDATION_START_TIME)

if [ $VALIDATION_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Hallucination validation tests passed in $VALIDATION_DURATION${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${RED}❌ Hallucination validation tests failed in $VALIDATION_DURATION${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

TOTAL_TESTS=$((TOTAL_TESTS + 1))
cd $PROJECT_ROOT

print_section "🎭 End-to-End Tests"

cd $FRONTEND_DIR

E2E_START_TIME=$(date +%s)

# Set environment variables for E2E tests
export E2E_BASE_URL="http://localhost:3000"
export API_BASE_URL="http://localhost:8000"
export HEADLESS=true
export RECORD_VIDEO=true

# Start backend server for E2E tests (in background)
cd $PROJECT_ROOT/$BACKEND_DIR
source venv/bin/activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "Started backend server (PID: $BACKEND_PID)"

# Wait for backend to start
sleep 10

# Start frontend server (in background)
cd $PROJECT_ROOT/$FRONTEND_DIR
npm run dev &
FRONTEND_PID=$!
echo "Started frontend server (PID: $FRONTEND_PID)"

# Wait for frontend to start
sleep 15

# Run E2E tests
echo "Running E2E tests..."
run_with_timeout $E2E_TIMEOUT "End-to-End Tests" \
    npx playwright test \
    --reporter=html \
    --output-dir=../${REPORTS_DIR}/e2e

E2E_EXIT_CODE=$?
E2E_DURATION=$(calculate_duration $E2E_START_TIME)

# Stop servers
echo "Stopping test servers..."
kill $BACKEND_PID 2>/dev/null || true
kill $FRONTEND_PID 2>/dev/null || true

if [ $E2E_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ E2E tests passed in $E2E_DURATION${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${RED}❌ E2E tests failed in $E2E_DURATION${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

TOTAL_TESTS=$((TOTAL_TESTS + 1))
cd $PROJECT_ROOT

print_section "⚡ Performance Tests"

cd $BACKEND_DIR
source venv/bin/activate

PERF_START_TIME=$(date +%s)

# Start backend for performance testing
python -m uvicorn main:app --host 0.0.0.0 --port 8000 &
PERF_BACKEND_PID=$!
echo "Started backend server for performance testing (PID: $PERF_BACKEND_PID)"

# Wait for server to start
sleep 10

# Run performance tests with Locust
echo "Running performance tests..."
run_with_timeout $PERFORMANCE_TIMEOUT "Performance Tests" \
    locust -f tests/performance/locustfile.py \
    --host=http://localhost:8000 \
    --users=50 \
    --spawn-rate=5 \
    --run-time=2m \
    --headless \
    --html=../${REPORTS_DIR}/performance/performance-report.html \
    --csv=../${REPORTS_DIR}/performance/performance-results

PERF_EXIT_CODE=$?
PERF_DURATION=$(calculate_duration $PERF_START_TIME)

# Stop performance test server
kill $PERF_BACKEND_PID 2>/dev/null || true

if [ $PERF_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Performance tests passed in $PERF_DURATION${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${RED}❌ Performance tests failed in $PERF_DURATION${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

TOTAL_TESTS=$((TOTAL_TESTS + 1))
cd $PROJECT_ROOT

print_section "🧹 Cleanup"

# Stop any remaining services
echo "Cleaning up test services..."
pkill -f redis-server || true
pkill -f uvicorn || true
pkill -f npm || true

# Deactivate virtual environment
deactivate 2>/dev/null || true

print_section "📊 Test Results Summary"

TOTAL_DURATION=$(calculate_duration $TEST_START_TIME)

echo "======================================================================"
echo -e "${BLUE}📋 COMPREHENSIVE TEST RESULTS${NC}"
echo "======================================================================"
echo "Total Test Suites: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
echo "Total Duration: $TOTAL_DURATION"
echo ""

# Individual test suite results
echo "📝 Individual Results:"
echo "- Backend Unit Tests: $([ $UNIT_EXIT_CODE -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED") ($UNIT_DURATION)"
echo "- Backend Integration Tests: $([ $INTEGRATION_EXIT_CODE -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED") ($INTEGRATION_DURATION)"
echo "- Security Tests: $([ $SECURITY_EXIT_CODE -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED") ($SECURITY_DURATION)"
echo "- Hallucination Validation: $([ $VALIDATION_EXIT_CODE -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED") ($VALIDATION_DURATION)"
echo "- E2E Tests: $([ $E2E_EXIT_CODE -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED") ($E2E_DURATION)"
echo "- Performance Tests: $([ $PERF_EXIT_CODE -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED") ($PERF_DURATION)"
echo ""

print_section "📈 Coverage Reports"

echo "Coverage reports generated:"
echo "- Backend Unit Coverage: ${REPORTS_DIR}/coverage/backend-unit/index.html"
echo "- Backend Integration Coverage: ${REPORTS_DIR}/coverage/backend-integration/index.html"
echo "- Performance Report: ${REPORTS_DIR}/performance/performance-report.html"
echo "- E2E Test Report: ${REPORTS_DIR}/playwright-report/index.html"
echo ""

# Generate combined coverage report if possible
if command -v coverage &> /dev/null; then
    echo "Generating combined coverage report..."
    cd $BACKEND_DIR
    source venv/bin/activate
    coverage combine
    coverage html -d ../${REPORTS_DIR}/coverage/combined
    coverage report --show-missing
    cd $PROJECT_ROOT
    echo "- Combined Coverage: ${REPORTS_DIR}/coverage/combined/index.html"
fi

print_section "🚨 Quality Gates"

QUALITY_GATES_PASSED=true

# Check minimum coverage
if [ -f "${REPORTS_DIR}/coverage/backend-unit.xml" ]; then
    # Extract coverage percentage from XML (simplified)
    COVERAGE=$(grep -oP 'line-rate="\K[0-9.]+' ${REPORTS_DIR}/coverage/backend-unit.xml | head -1)
    COVERAGE_PERCENT=$(echo "$COVERAGE * 100" | bc -l | cut -d. -f1)
    
    if [ "$COVERAGE_PERCENT" -lt "$MIN_COVERAGE" ]; then
        echo -e "${RED}❌ Coverage quality gate failed: ${COVERAGE_PERCENT}% < ${MIN_COVERAGE}%${NC}"
        QUALITY_GATES_PASSED=false
    else
        echo -e "${GREEN}✅ Coverage quality gate passed: ${COVERAGE_PERCENT}% >= ${MIN_COVERAGE}%${NC}"
    fi
fi

# Check if any critical tests failed
if [ $FAILED_TESTS -gt 0 ]; then
    echo -e "${RED}❌ Test quality gate failed: $FAILED_TESTS test suite(s) failed${NC}"
    QUALITY_GATES_PASSED=false
else
    echo -e "${GREEN}✅ Test quality gate passed: All test suites passed${NC}"
fi

print_section "🎯 Final Status"

if [ $QUALITY_GATES_PASSED = true ] && [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}"
    echo "🎉 ALL TESTS PASSED! 🎉"
    echo "✅ Quality gates satisfied"
    echo "✅ Ready for deployment"
    echo -e "${NC}"
    exit 0
else
    echo -e "${RED}"
    echo "💥 TESTS FAILED! 💥"
    echo "❌ Quality gates not satisfied"
    echo "❌ Please fix failing tests before deployment"
    echo -e "${NC}"
    exit 1
fi