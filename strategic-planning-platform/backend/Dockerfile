# Multi-stage build for FastAPI with Python 3.11+
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1001 app && \
    useradd --uid 1001 --gid 1001 --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Development stage
FROM base AS development

# Install development dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

# Copy source code
COPY . .
COPY ../shared ./shared

# Change ownership
RUN chown -R app:app /app
USER app

# Create necessary directories
RUN mkdir -p logs uploads

# Expose ports
EXPOSE 8000 8001

# Start development server with hot reload
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "info"]

# Builder stage
FROM base AS builder

# Copy requirements
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
COPY ../shared ./shared

# Compile Python files
RUN python -m compileall -b .

# Remove source files, keep only bytecode
RUN find . -name "*.py" -not -path "./tests/*" -delete && \
    find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Production stage
FROM base AS production

# Install only runtime dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Copy built application from builder
COPY --from=builder --chown=app:app /app /app

# Copy shared types
COPY --chown=app:app ../shared /app/shared

# Install production monitoring tools
RUN pip install --no-cache-dir \
    uvicorn[standard] \
    gunicorn \
    prometheus-client

# Create necessary directories and files
RUN mkdir -p /app/logs /app/uploads /app/temp && \
    touch /app/logs/app.log && \
    chown -R app:app /app

# Switch to non-root user
USER app

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8001

# Production startup script
COPY --chown=app:app ./scripts/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Start the application
CMD ["/app/start.sh"]

# Testing stage
FROM development AS test

# Copy test files
COPY tests/ ./tests/
COPY pytest.ini ./

# Install additional test dependencies
RUN pip install --no-cache-dir \
    pytest-cov \
    pytest-mock \
    httpx \
    pytest-asyncio

# Run tests
CMD ["pytest", "--cov=.", "--cov-report=html", "--cov-report=term"]

# Security scanning stage
FROM production AS security

USER root

# Install security scanning tools
RUN pip install --no-cache-dir \
    safety \
    bandit

# Run security scans
RUN safety check --json --output /app/safety-report.json || true
RUN bandit -r /app -f json -o /app/bandit-report.json || true

USER app

CMD ["python", "-c", "print('Security scan completed')"]