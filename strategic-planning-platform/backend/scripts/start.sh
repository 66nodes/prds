#!/bin/bash
set -e

# Strategic Planning Platform - Production Startup Script

echo "Starting Strategic Planning Platform Backend..."

# Wait for dependencies
echo "Waiting for database connections..."

# Wait for PostgreSQL
echo "Checking PostgreSQL connection..."
while ! nc -z postgres 5432; do
  echo "Waiting for PostgreSQL to be ready..."
  sleep 2
done
echo "PostgreSQL is ready!"

# Wait for Redis
echo "Checking Redis connection..."
while ! nc -z redis 6379; do
  echo "Waiting for Redis to be ready..."
  sleep 2
done
echo "Redis is ready!"

# Wait for Neo4j
echo "Checking Neo4j connection..."
while ! nc -z neo4j 7687; do
  echo "Waiting for Neo4j to be ready..."
  sleep 2
done
echo "Neo4j is ready!"

# Run database migrations
echo "Running database migrations..."
python -c "
import asyncio
from app.core.database import init_databases
asyncio.run(init_databases())
print('Database initialization completed')
"

# Initialize Neo4j schema if needed
echo "Initializing Neo4j schema..."
python -c "
import asyncio
from app.services.graphrag.neo4j_service import Neo4jService
async def init_schema():
    service = Neo4jService()
    await service.init_schema()
    await service.close()
    print('Neo4j schema initialization completed')
asyncio.run(init_schema())
"

# Set up monitoring
echo "Setting up monitoring endpoints..."
export PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc_dir
mkdir -p $PROMETHEUS_MULTIPROC_DIR

# Production server configuration
WORKERS=${WORKERS:-4}
WORKER_CLASS=${WORKER_CLASS:-uvicorn.workers.UvicornWorker}
WORKER_CONNECTIONS=${WORKER_CONNECTIONS:-1000}
MAX_REQUESTS=${MAX_REQUESTS:-1000}
MAX_REQUESTS_JITTER=${MAX_REQUESTS_JITTER:-100}
TIMEOUT=${TIMEOUT:-120}
KEEPALIVE=${KEEPALIVE:-5}

echo "Starting with configuration:"
echo "  Workers: $WORKERS"
echo "  Worker class: $WORKER_CLASS"
echo "  Worker connections: $WORKER_CONNECTIONS"
echo "  Max requests: $MAX_REQUESTS"
echo "  Timeout: $TIMEOUT"

# Start the application with Gunicorn for production
if [ "$FASTAPI_ENV" = "development" ]; then
    echo "Starting in development mode..."
    exec uvicorn main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload \
        --log-level info \
        --access-log \
        --use-colors
else
    echo "Starting in production mode with Gunicorn..."
    exec gunicorn main:app \
        --bind 0.0.0.0:8000 \
        --workers $WORKERS \
        --worker-class $WORKER_CLASS \
        --worker-connections $WORKER_CONNECTIONS \
        --max-requests $MAX_REQUESTS \
        --max-requests-jitter $MAX_REQUESTS_JITTER \
        --timeout $TIMEOUT \
        --keepalive $KEEPALIVE \
        --preload \
        --log-level info \
        --access-logfile - \
        --error-logfile - \
        --log-config logging.conf \
        --capture-output \
        --enable-stdio-inheritance
fi