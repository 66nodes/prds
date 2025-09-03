#!/bin/bash
# AI Agent Platform - Restore Script
# Restores databases and application data from backups

set -e

# Configuration
BACKUP_DIR="/backups"
S3_BUCKET="${BACKUP_S3_BUCKET:-aiplatform-backups}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

usage() {
    echo "Usage: $0 [OPTIONS] <backup-date>"
    echo ""
    echo "Options:"
    echo "  -s, --from-s3         Download backup from S3 first"
    echo "  -p, --postgres        Restore PostgreSQL only"
    echo "  -n, --neo4j          Restore Neo4j only"
    echo "  -r, --redis          Restore Redis only"
    echo "  -m, --milvus         Restore Milvus only"
    echo "  -a, --app-data       Restore application data only"
    echo "  -f, --force          Skip confirmation prompts"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 2024-01-15                    # Restore from local backup"
    echo "  $0 -s 2024-01-15                # Download from S3 and restore"
    echo "  $0 -p -n 2024-01-15             # Restore only PostgreSQL and Neo4j"
    echo ""
}

# Parse command line arguments
RESTORE_ALL=true
RESTORE_POSTGRES=false
RESTORE_NEO4J=false
RESTORE_REDIS=false
RESTORE_MILVUS=false
RESTORE_APP_DATA=false
FROM_S3=false
FORCE=false
BACKUP_DATE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--from-s3)
            FROM_S3=true
            shift
            ;;
        -p|--postgres)
            RESTORE_ALL=false
            RESTORE_POSTGRES=true
            shift
            ;;
        -n|--neo4j)
            RESTORE_ALL=false
            RESTORE_NEO4J=true
            shift
            ;;
        -r|--redis)
            RESTORE_ALL=false
            RESTORE_REDIS=true
            shift
            ;;
        -m|--milvus)
            RESTORE_ALL=false
            RESTORE_MILVUS=true
            shift
            ;;
        -a|--app-data)
            RESTORE_ALL=false
            RESTORE_APP_DATA=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [ -z "$BACKUP_DATE" ]; then
                BACKUP_DATE="$1"
            else
                error "Unknown option: $1"
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$BACKUP_DATE" ]; then
    error "Backup date is required"
    usage
    exit 1
fi

# Set restore flags if restoring all
if [ "$RESTORE_ALL" = true ]; then
    RESTORE_POSTGRES=true
    RESTORE_NEO4J=true
    RESTORE_REDIS=true
    RESTORE_MILVUS=true
    RESTORE_APP_DATA=true
fi

BACKUP_PATH="${BACKUP_DIR}/${BACKUP_DATE}"

# Download from S3 if requested
download_from_s3() {
    log "☁️ Downloading backup from S3..."
    
    if ! command -v aws >/dev/null 2>&1; then
        error "AWS CLI not found"
        exit 1
    fi
    
    local s3_path="s3://${S3_BUCKET}/${BACKUP_DATE//-/\/}"
    
    mkdir -p "${BACKUP_PATH}"
    
    if aws s3 sync "${s3_path}" "${BACKUP_PATH}"; then
        log "✅ Backup downloaded from S3"
    else
        error "❌ Failed to download backup from S3"
        exit 1
    fi
}

# Confirmation prompt
confirm_restore() {
    if [ "$FORCE" = true ]; then
        return 0
    fi
    
    echo ""
    warning "🚨 WARNING: This will overwrite existing data!"
    echo "Backup date: ${BACKUP_DATE}"
    echo "Backup path: ${BACKUP_PATH}"
    echo ""
    echo "Services to restore:"
    [ "$RESTORE_POSTGRES" = true ] && echo "  - PostgreSQL"
    [ "$RESTORE_NEO4J" = true ] && echo "  - Neo4j"
    [ "$RESTORE_REDIS" = true ] && echo "  - Redis"
    [ "$RESTORE_MILVUS" = true ] && echo "  - Milvus"
    [ "$RESTORE_APP_DATA" = true ] && echo "  - Application Data"
    echo ""
    
    read -p "Are you sure you want to continue? (yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log "Restore cancelled"
        exit 0
    fi
}

# Check if backup exists
check_backup_exists() {
    if [ ! -d "${BACKUP_PATH}" ]; then
        error "Backup directory not found: ${BACKUP_PATH}"
        exit 1
    fi
    
    log "✅ Backup directory found: ${BACKUP_PATH}"
}

# Stop services before restore
stop_services() {
    log "⏸️ Stopping services for restore..."
    
    local services=()
    [ "$RESTORE_POSTGRES" = true ] && services+=("postgres")
    [ "$RESTORE_NEO4J" = true ] && services+=("neo4j")
    [ "$RESTORE_REDIS" = true ] && services+=("redis")
    [ "$RESTORE_MILVUS" = true ] && services+=("milvus-standalone etcd minio")
    
    for service in "${services[@]}"; do
        docker service scale "aiplatform_${service}=0" || warning "Failed to stop ${service}"
        sleep 5
    done
    
    log "✅ Services stopped"
}

# Start services after restore
start_services() {
    log "▶️ Starting services after restore..."
    
    local services=()
    [ "$RESTORE_MILVUS" = true ] && services+=("etcd=1 minio=1")
    [ "$RESTORE_POSTGRES" = true ] && services+=("postgres=1")
    [ "$RESTORE_NEO4J" = true ] && services+=("neo4j=1") 
    [ "$RESTORE_REDIS" = true ] && services+=("redis=1")
    [ "$RESTORE_MILVUS" = true ] && services+=("milvus-standalone=1")
    
    sleep 10  # Wait for dependencies to start
    
    for service in "${services[@]}"; do
        docker service scale "aiplatform_${service}" || warning "Failed to start ${service}"
        sleep 5
    done
    
    log "✅ Services started"
}

# Restore PostgreSQL
restore_postgres() {
    log "📊 Restoring PostgreSQL database..."
    
    local backup_file=$(find "${BACKUP_PATH}" -name "postgres_*.sql.gz" | head -1)
    
    if [ -z "$backup_file" ]; then
        error "PostgreSQL backup file not found"
        return 1
    fi
    
    log "Using backup file: $(basename "$backup_file")"
    
    # Wait for PostgreSQL to be ready
    sleep 30
    
    # Drop existing database and recreate
    docker exec -i $(docker ps -q -f name=aiplatform_postgres) psql -U postgres -c "DROP DATABASE IF EXISTS aiplatform;" || true
    docker exec -i $(docker ps -q -f name=aiplatform_postgres) psql -U postgres -c "CREATE DATABASE aiplatform;"
    
    # Restore from backup
    if gunzip -c "$backup_file" | docker exec -i $(docker ps -q -f name=aiplatform_postgres) psql -U postgres aiplatform; then
        log "✅ PostgreSQL restore completed"
    else
        error "❌ PostgreSQL restore failed"
        return 1
    fi
}

# Restore Neo4j
restore_neo4j() {
    log "🕸️ Restoring Neo4j graph database..."
    
    local backup_file=$(find "${BACKUP_PATH}" -name "neo4j_*.dump" | head -1)
    
    if [ -z "$backup_file" ]; then
        error "Neo4j backup file not found"
        return 1
    fi
    
    log "Using backup file: $(basename "$backup_file")"
    
    # Copy backup file to container
    docker cp "$backup_file" $(docker ps -q -f name=aiplatform_neo4j):/tmp/neo4j.dump
    
    # Stop Neo4j and restore
    docker exec -i $(docker ps -q -f name=aiplatform_neo4j) neo4j stop || true
    docker exec -i $(docker ps -q -f name=aiplatform_neo4j) neo4j-admin database load neo4j --from-path=/tmp --overwrite-destination=true
    docker exec -i $(docker ps -q -f name=aiplatform_neo4j) neo4j start
    
    # Wait for Neo4j to be ready
    sleep 30
    
    if docker exec -i $(docker ps -q -f name=aiplatform_neo4j) cypher-shell -u neo4j -p development "RETURN 1" >/dev/null 2>&1; then
        log "✅ Neo4j restore completed"
    else
        error "❌ Neo4j restore failed"
        return 1
    fi
}

# Restore Redis
restore_redis() {
    log "💾 Restoring Redis data..."
    
    local backup_file=$(find "${BACKUP_PATH}" -name "redis_*.rdb" | head -1)
    
    if [ -z "$backup_file" ]; then
        error "Redis backup file not found"
        return 1
    fi
    
    log "Using backup file: $(basename "$backup_file")"
    
    # Copy RDB file to container
    docker cp "$backup_file" $(docker ps -q -f name=aiplatform_redis):/data/dump.rdb
    
    # Restart Redis to load the new RDB file
    docker service update --force aiplatform_redis
    
    sleep 10
    
    if docker exec -i $(docker ps -q -f name=aiplatform_redis) redis-cli ping | grep -q PONG; then
        log "✅ Redis restore completed"
    else
        error "❌ Redis restore failed"
        return 1
    fi
}

# Restore Milvus
restore_milvus() {
    log "🔍 Restoring Milvus vector database..."
    
    local backup_dir=$(find "${BACKUP_PATH}" -name "milvus_*" -type d | head -1)
    
    if [ -z "$backup_dir" ]; then
        error "Milvus backup directory not found"
        return 1
    fi
    
    log "Using backup directory: $(basename "$backup_dir")"
    
    # Stop Milvus and dependencies
    docker service scale aiplatform_milvus-standalone=0
    sleep 10
    
    # Restore Milvus data
    if docker run --rm -v aiplatform_milvus_data:/data -v "${backup_dir}:/backup" \
       alpine sh -c 'rm -rf /data/* && tar xzf /backup/milvus_data.tar.gz -C /data'; then
        log "✅ Milvus restore completed"
    else
        error "❌ Milvus restore failed"
        return 1
    fi
}

# Restore application data
restore_app_data() {
    log "📁 Restoring application data..."
    
    local backup_dir=$(find "${BACKUP_PATH}" -name "app_data_*" -type d | head -1)
    
    if [ -z "$backup_dir" ]; then
        warning "Application data backup directory not found, skipping..."
        return 0
    fi
    
    log "Using backup directory: $(basename "$backup_dir")"
    
    # Restore GraphRAG cache
    if [ -f "${backup_dir}/graphrag_cache.tar.gz" ]; then
        docker run --rm -v aiplatform_graphrag_cache:/data -v "${backup_dir}:/backup" \
            alpine sh -c 'rm -rf /data/* && tar xzf /backup/graphrag_cache.tar.gz -C /data' || \
            warning "Failed to restore GraphRAG cache"
    fi
    
    # Restore logs (optional)
    if [ -f "${backup_dir}/agent_logs.tar.gz" ]; then
        docker run --rm -v aiplatform_agent_logs:/data -v "${backup_dir}:/backup" \
            alpine sh -c 'tar xzf /backup/agent_logs.tar.gz -C /data' || \
            warning "Failed to restore agent logs"
    fi
    
    log "✅ Application data restore completed"
}

# Verify restore
verify_restore() {
    log "🔍 Verifying restore..."
    
    local errors=0
    
    # Check PostgreSQL
    if [ "$RESTORE_POSTGRES" = true ]; then
        if docker exec -i $(docker ps -q -f name=aiplatform_postgres) psql -U postgres -d aiplatform -c "SELECT 1;" >/dev/null 2>&1; then
            log "✅ PostgreSQL verification passed"
        else
            error "❌ PostgreSQL verification failed"
            ((errors++))
        fi
    fi
    
    # Check Neo4j
    if [ "$RESTORE_NEO4J" = true ]; then
        if docker exec -i $(docker ps -q -f name=aiplatform_neo4j) cypher-shell -u neo4j -p development "RETURN 1;" >/dev/null 2>&1; then
            log "✅ Neo4j verification passed"
        else
            error "❌ Neo4j verification failed"
            ((errors++))
        fi
    fi
    
    # Check Redis
    if [ "$RESTORE_REDIS" = true ]; then
        if docker exec -i $(docker ps -q -f name=aiplatform_redis) redis-cli ping | grep -q PONG; then
            log "✅ Redis verification passed"
        else
            error "❌ Redis verification failed"
            ((errors++))
        fi
    fi
    
    # Check Milvus
    if [ "$RESTORE_MILVUS" = true ]; then
        sleep 30  # Give Milvus more time to start
        if docker exec -i $(docker ps -q -f name=aiplatform_milvus) curl -f http://localhost:9091/healthz >/dev/null 2>&1; then
            log "✅ Milvus verification passed"
        else
            error "❌ Milvus verification failed"
            ((errors++))
        fi
    fi
    
    if [ $errors -eq 0 ]; then
        log "🎉 All verifications passed"
        return 0
    else
        error "❌ ${errors} verification(s) failed"
        return 1
    fi
}

# Main restore process
main() {
    local start_time=$(date +%s)
    
    log "🎯 Starting AI Platform restore from ${BACKUP_DATE}"
    
    # Download from S3 if requested
    if [ "$FROM_S3" = true ]; then
        download_from_s3
    fi
    
    # Check if backup exists
    check_backup_exists
    
    # Confirm restore
    confirm_restore
    
    # Stop services
    stop_services
    
    # Perform restore operations
    [ "$RESTORE_POSTGRES" = true ] && restore_postgres
    [ "$RESTORE_NEO4J" = true ] && restore_neo4j
    [ "$RESTORE_REDIS" = true ] && restore_redis
    [ "$RESTORE_MILVUS" = true ] && restore_milvus
    [ "$RESTORE_APP_DATA" = true ] && restore_app_data
    
    # Start services
    start_services
    
    # Wait for services to be ready
    sleep 60
    
    # Verify restore
    if verify_restore; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        log "🎉 Restore completed successfully in ${duration} seconds"
        
        # Send notification (if configured)
        if [ -n "${NOTIFICATION_WEBHOOK}" ]; then
            curl -X POST "${NOTIFICATION_WEBHOOK}" \
                 -H "Content-Type: application/json" \
                 -d "{\"text\": \"✅ AI Platform restore completed successfully in ${duration}s\"}" \
                 2>/dev/null || true
        fi
    else
        error "❌ Restore verification failed"
        exit 1
    fi
}

# Signal handlers
trap 'error "❌ Restore interrupted"; exit 1' INT TERM

# Run main function
main "$@"