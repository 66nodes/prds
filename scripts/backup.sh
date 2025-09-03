#!/bin/bash
# AI Agent Platform - Backup Script
# Creates backups of all databases and critical data

set -e

# Configuration
BACKUP_DIR="/backups/$(date +%Y-%m-%d)"
S3_BUCKET="${BACKUP_S3_BUCKET:-aiplatform-backups}"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

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

# Create backup directory
mkdir -p "${BACKUP_DIR}"
cd "${BACKUP_DIR}"

log "🚀 Starting backup process..."

# Check if services are running
check_service() {
    local service=$1
    if ! docker service ls --filter name="${service}" --format "{{.Replicas}}" | grep -q "1/1"; then
        error "Service ${service} is not running or not healthy"
        return 1
    fi
}

# PostgreSQL Backup
backup_postgres() {
    log "📊 Backing up PostgreSQL database..."
    
    local backup_file="postgres_${TIMESTAMP}.sql.gz"
    
    if docker exec -i $(docker ps -q -f name=aiplatform_postgres) pg_dump -U postgres aiplatform | gzip > "${backup_file}"; then
        log "✅ PostgreSQL backup completed: ${backup_file}"
        
        # Test backup integrity
        if gunzip -t "${backup_file}"; then
            log "✅ PostgreSQL backup integrity verified"
        else
            error "❌ PostgreSQL backup integrity check failed"
            return 1
        fi
    else
        error "❌ PostgreSQL backup failed"
        return 1
    fi
}

# Neo4j Backup
backup_neo4j() {
    log "🕸️  Backing up Neo4j graph database..."
    
    local backup_file="neo4j_${TIMESTAMP}.dump"
    
    if docker exec -i $(docker ps -q -f name=aiplatform_neo4j) neo4j-admin database dump neo4j --to-path=/tmp && \
       docker cp $(docker ps -q -f name=aiplatform_neo4j):/tmp/neo4j.dump "${backup_file}"; then
        log "✅ Neo4j backup completed: ${backup_file}"
    else
        error "❌ Neo4j backup failed"
        return 1
    fi
}

# Redis Backup
backup_redis() {
    log "💾 Backing up Redis data..."
    
    local backup_file="redis_${TIMESTAMP}.rdb"
    
    # Trigger Redis BGSAVE
    if docker exec -i $(docker ps -q -f name=aiplatform_redis) redis-cli BGSAVE; then
        # Wait for background save to complete
        sleep 5
        
        # Copy the RDB file
        if docker cp $(docker ps -q -f name=aiplatform_redis):/data/dump.rdb "${backup_file}"; then
            log "✅ Redis backup completed: ${backup_file}"
        else
            error "❌ Redis backup copy failed"
            return 1
        fi
    else
        error "❌ Redis backup failed"
        return 1
    fi
}

# Milvus Backup
backup_milvus() {
    log "🔍 Backing up Milvus vector database..."
    
    local backup_dir="milvus_${TIMESTAMP}"
    mkdir -p "${backup_dir}"
    
    # Copy Milvus data volumes
    if docker run --rm -v aiplatform_milvus_data:/data -v "${BACKUP_DIR}/${backup_dir}:/backup" \
       alpine tar czf "/backup/milvus_data.tar.gz" -C /data .; then
        log "✅ Milvus backup completed: ${backup_dir}/milvus_data.tar.gz"
    else
        error "❌ Milvus backup failed"
        return 1
    fi
}

# Application Data Backup
backup_app_data() {
    log "📁 Backing up application data..."
    
    local backup_dir="app_data_${TIMESTAMP}"
    mkdir -p "${backup_dir}"
    
    # Backup GraphRAG cache
    if docker run --rm -v aiplatform_graphrag_cache:/data -v "${BACKUP_DIR}/${backup_dir}:/backup" \
       alpine tar czf "/backup/graphrag_cache.tar.gz" -C /data .; then
        log "✅ GraphRAG cache backup completed"
    else
        warning "⚠️ GraphRAG cache backup failed (non-critical)"
    fi
    
    # Backup logs
    if docker run --rm -v aiplatform_agent_logs:/data -v "${BACKUP_DIR}/${backup_dir}:/backup" \
       alpine tar czf "/backup/agent_logs.tar.gz" -C /data .; then
        log "✅ Agent logs backup completed"
    else
        warning "⚠️ Agent logs backup failed (non-critical)"
    fi
}

# Configuration Backup
backup_config() {
    log "⚙️ Backing up configuration files..."
    
    local config_dir="config_${TIMESTAMP}"
    mkdir -p "${config_dir}"
    
    # Copy important configuration files
    cp -r ../monitoring "${config_dir}/"
    cp -r ../infrastructure "${config_dir}/"
    cp ../docker-stack.yml "${config_dir}/"
    cp ../.env.production.example "${config_dir}/"
    
    log "✅ Configuration backup completed"
}

# Upload to S3
upload_to_s3() {
    log "☁️ Uploading backups to S3..."
    
    if command -v aws >/dev/null 2>&1; then
        local s3_path="s3://${S3_BUCKET}/$(date +%Y/%m/%d)"
        
        if aws s3 sync "${BACKUP_DIR}" "${s3_path}" --storage-class STANDARD_IA; then
            log "✅ Backup uploaded to S3: ${s3_path}"
        else
            error "❌ S3 upload failed"
            return 1
        fi
    else
        warning "⚠️ AWS CLI not found, skipping S3 upload"
    fi
}

# Cleanup old backups
cleanup_old_backups() {
    log "🧹 Cleaning up old backups..."
    
    # Local cleanup
    find /backups -type d -name "*-*-*" -mtime +${RETENTION_DAYS} -exec rm -rf {} + || true
    
    # S3 cleanup (if AWS CLI is available)
    if command -v aws >/dev/null 2>&1; then
        aws s3api list-objects-v2 --bucket "${S3_BUCKET}" --query "Contents[?LastModified<=\`$(date -d "${RETENTION_DAYS} days ago" --iso-8601)\`].Key" --output text | \
        xargs -r -n1 aws s3api delete-object --bucket "${S3_BUCKET}" --key || true
    fi
    
    log "✅ Cleanup completed"
}

# Health check before backup
health_check() {
    log "🏥 Performing health check..."
    
    local services=("aiplatform_postgres" "aiplatform_neo4j" "aiplatform_redis" "aiplatform_milvus-standalone")
    
    for service in "${services[@]}"; do
        if ! docker ps --filter name="${service}" --filter status=running -q | grep -q .; then
            error "Service ${service} is not running"
            return 1
        fi
    done
    
    log "✅ All services are healthy"
}

# Create backup manifest
create_manifest() {
    log "📋 Creating backup manifest..."
    
    cat > "backup_manifest_${TIMESTAMP}.json" <<EOF
{
    "timestamp": "${TIMESTAMP}",
    "date": "$(date --iso-8601)",
    "version": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "services": {
        "postgres": "$(docker exec $(docker ps -q -f name=aiplatform_postgres) psql -U postgres -t -c 'SELECT version();' 2>/dev/null | head -1 | xargs || echo 'unknown')",
        "neo4j": "$(docker exec $(docker ps -q -f name=aiplatform_neo4j) cypher-shell -u neo4j -p development 'CALL dbms.components() YIELD name, versions RETURN name, versions[0]' --format plain 2>/dev/null | grep Neo4j | awk '{print $2}' || echo 'unknown')",
        "redis": "$(docker exec $(docker ps -q -f name=aiplatform_redis) redis-cli INFO SERVER | grep redis_version | cut -d: -f2 | tr -d '\r' || echo 'unknown')",
        "milvus": "$(docker exec $(docker ps -q -f name=aiplatform_milvus) milvus --version 2>/dev/null || echo 'unknown')"
    },
    "backup_size_mb": $(du -sm "${BACKUP_DIR}" | cut -f1),
    "files": [
$(find "${BACKUP_DIR}" -type f -printf '        "%p",\n' | sed '$ s/,$//')
    ]
}
EOF
    
    log "✅ Backup manifest created"
}

# Main backup process
main() {
    local start_time=$(date +%s)
    
    log "🎯 Starting AI Platform backup at $(date)"
    
    # Perform health check
    if ! health_check; then
        error "❌ Health check failed, aborting backup"
        exit 1
    fi
    
    # Perform all backups
    backup_postgres || error "PostgreSQL backup failed"
    backup_neo4j || error "Neo4j backup failed" 
    backup_redis || error "Redis backup failed"
    backup_milvus || error "Milvus backup failed"
    backup_app_data
    backup_config
    
    # Create manifest
    create_manifest
    
    # Upload to S3
    upload_to_s3
    
    # Cleanup old backups
    cleanup_old_backups
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "🎉 Backup completed successfully in ${duration} seconds"
    log "📁 Backup location: ${BACKUP_DIR}"
    log "💾 Total size: $(du -sh "${BACKUP_DIR}" | cut -f1)"
    
    # Send notification (if configured)
    if [ -n "${NOTIFICATION_WEBHOOK}" ]; then
        curl -X POST "${NOTIFICATION_WEBHOOK}" \
             -H "Content-Type: application/json" \
             -d "{\"text\": \"✅ AI Platform backup completed successfully in ${duration}s\"}" \
             2>/dev/null || true
    fi
}

# Signal handlers
trap 'error "❌ Backup interrupted"; exit 1' INT TERM

# Run main function
main "$@"