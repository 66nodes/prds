#!/bin/bash

# AI Agent Platform - Infrastructure Deployment Script
# Deploys the Hybrid RAG infrastructure stack with Milvus + Neo4j

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_PROJECT_NAME="ai-agent-platform"
STACK_NAME="ai-platform"
ENVIRONMENT="${1:-development}"

echo -e "${BLUE}🚀 AI Agent Platform Infrastructure Deployment${NC}"
echo "Environment: ${ENVIRONMENT}"
echo "Stack Name: ${STACK_NAME}"
echo ""

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
echo -e "${BLUE}📋 Checking Prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi
print_status "Docker: $(docker --version)"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed or not in PATH"
    exit 1
fi
print_status "Docker Compose: $(docker-compose --version)"

# Check if running in swarm mode (optional)
if docker info --format '{{.Swarm.LocalNodeState}}' | grep -q 'active'; then
    print_status "Docker Swarm mode active"
    SWARM_MODE=true
else
    print_warning "Docker Swarm mode not active - using docker-compose instead"
    SWARM_MODE=false
fi

# Check environment file
if [ ! -f ".env.${ENVIRONMENT}" ]; then
    print_warning "Environment file .env.${ENVIRONMENT} not found"
    if [ ! -f ".env.template" ]; then
        print_error ".env.template not found. Please create environment configuration first."
        exit 1
    fi
    
    print_status "Copying .env.template to .env.${ENVIRONMENT}"
    cp .env.template ".env.${ENVIRONMENT}"
    print_warning "Please edit .env.${ENVIRONMENT} with your actual configuration values"
fi

# Create necessary directories
echo -e "\n${BLUE}📁 Creating Data Directories...${NC}"
mkdir -p {logs,data,backups}
mkdir -p data/{neo4j,postgres,redis,milvus,minio}
print_status "Data directories created"

# Create Docker secrets (for swarm mode)
if [ "$SWARM_MODE" = true ]; then
    echo -e "\n${BLUE}🔐 Setting up Docker Secrets...${NC}"
    
    # Create secrets if they don't exist
    echo "development_password" | docker secret create neo4j_password - 2>/dev/null || print_warning "neo4j_password secret already exists"
    echo "development_password" | docker secret create postgres_password - 2>/dev/null || print_warning "postgres_password secret already exists"
    echo "jwt-secret-key-for-development" | docker secret create jwt_secret - 2>/dev/null || print_warning "jwt_secret secret already exists"
    echo "your-openrouter-api-key-here" | docker secret create openrouter_api_key - 2>/dev/null || print_warning "openrouter_api_key secret already exists"
    
    print_status "Docker secrets configured"
fi

# Deploy the stack
echo -e "\n${BLUE}🚢 Deploying Infrastructure Stack...${NC}"

if [ "$SWARM_MODE" = true ]; then
    # Deploy using Docker Swarm
    print_status "Deploying with Docker Swarm..."
    docker stack deploy -c docker-stack.yml "$STACK_NAME"
else
    # Deploy using Docker Compose
    print_status "Deploying with Docker Compose..."
    
    # Create a docker-compose.yml from the docker-stack.yml for local development
    cat > docker-compose.yml << 'EOF'
version: '3.8'

networks:
  ai-platform-network:
    driver: bridge

volumes:
  neo4j_data:
  postgres_data:
  redis_data:
  milvus_etcd:
  milvus_minio:
  milvus_data:

services:
  # Core Infrastructure Services
  etcd:
    image: quay.io/coreos/etcd:v3.5.0
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - milvus_etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    networks:
      - ai-platform-network

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9001:9001"
      - "9000:9000"
    volumes:
      - milvus_minio:/minio_data
    command: minio server /minio_data --console-address ":9001"
    networks:
      - ai-platform-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  milvus-standalone:
    image: milvusdb/milvus:v2.3.3
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - milvus_data:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"
    networks:
      - ai-platform-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      timeout: 20s
      retries: 5

  neo4j:
    image: neo4j:5.15-community
    environment:
      - NEO4J_AUTH=neo4j/development
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*
      - NEO4J_apoc_export_file_enabled=true
      - NEO4J_apoc_import_file_enabled=true
      - NEO4J_dbms_memory_heap_initial__size=1G
      - NEO4J_dbms_memory_heap_max__size=2G
      - NEO4J_dbms_memory_pagecache_size=1G
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
    networks:
      - ai-platform-network
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "development", "RETURN 1"]
      interval: 30s
      timeout: 10s
      retries: 5

  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_DB=aiplatform
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=development
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./infrastructure/postgres/init:/docker-entrypoint-initdb.d
    networks:
      - ai-platform-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d aiplatform"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7.2-alpine
    command: redis-server --appendonly yes
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - ai-platform-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
EOF

    # Start services
    docker-compose --env-file ".env.${ENVIRONMENT}" up -d
fi

# Wait for services to be ready
echo -e "\n${BLUE}⏳ Waiting for Services to Start...${NC}"
sleep 30

# Health check
echo -e "\n${BLUE}🩺 Running Health Checks...${NC}"

check_service() {
    local service_name=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -sf "$url" >/dev/null 2>&1; then
            print_status "$service_name is healthy"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_warning "$service_name health check failed"
    return 1
}

# Check services
check_service "Milvus" "http://localhost:9091/healthz"
check_service "MinIO" "http://localhost:9000/minio/health/live"
check_service "Neo4j" "http://localhost:7474"

# Check database connections
if command -v psql &> /dev/null; then
    if PGPASSWORD=development psql -h localhost -U postgres -d aiplatform -c "SELECT 1;" >/dev/null 2>&1; then
        print_status "PostgreSQL is accessible"
    else
        print_warning "PostgreSQL connection failed"
    fi
fi

if command -v redis-cli &> /dev/null; then
    if redis-cli -h localhost ping >/dev/null 2>&1; then
        print_status "Redis is accessible"
    else
        print_warning "Redis connection failed"
    fi
fi

# Display connection information
echo -e "\n${GREEN}🎉 Infrastructure Deployment Complete!${NC}"
echo ""
echo -e "${BLUE}📊 Service Endpoints:${NC}"
echo "• Neo4j Browser: http://localhost:7474 (neo4j/development)"
echo "• Milvus: localhost:19530"
echo "• PostgreSQL: localhost:5432 (postgres/development)"
echo "• Redis: localhost:6379"
echo "• MinIO Console: http://localhost:9001 (minioadmin/minioadmin)"
echo "• Milvus WebUI: http://localhost:9091"
echo ""
echo -e "${BLUE}🔧 Next Steps:${NC}"
echo "1. Verify all services are running: docker ps"
echo "2. Check logs if any service is failing: docker-compose logs [service_name]"
echo "3. Update .env.${ENVIRONMENT} with your actual API keys"
echo "4. Initialize the backend application"
echo "5. Deploy the frontend application"
echo ""
echo -e "${YELLOW}💡 Useful Commands:${NC}"
echo "• View logs: docker-compose logs -f"
echo "• Stop services: docker-compose down"
echo "• Restart services: docker-compose restart"
echo "• Remove all data: docker-compose down -v"