#!/bin/bash
# AI Agent Platform - Production Readiness Check
# Comprehensive validation of production environment

set -e

# Configuration
DOMAIN_NAME="${DOMAIN_NAME:-aiplatform.example.com}"
API_DOMAIN="api.${DOMAIN_NAME}"
MONITORING_DOMAIN="monitoring.${DOMAIN_NAME}"
TRAEFIK_DOMAIN="traefik.${DOMAIN_NAME}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
    ((FAILED_CHECKS++))
}

warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
    ((WARNING_CHECKS++))
}

success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
    ((PASSED_CHECKS++))
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

check() {
    ((TOTAL_CHECKS++))
    info "🔍 Checking: $1"
}

# Header
print_header() {
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                AI Agent Platform                                ║"
    echo "║              Production Readiness Check                        ║"
    echo "║                                                                ║"
    echo "║  Domain: ${DOMAIN_NAME}                           ║"
    echo "║  Date: $(date '+%Y-%m-%d %H:%M:%S')                                     ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo
}

# 1. Infrastructure Checks
check_infrastructure() {
    echo "🏗️  Infrastructure Checks"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Docker Swarm
    check "Docker Swarm Status"
    if docker info --format '{{.Swarm.LocalNodeState}}' | grep -q "active"; then
        success "Docker Swarm is active"
        
        # Node health
        local manager_count=$(docker node ls --filter role=manager --format "{{.Status}}" | grep -c "Ready" || echo 0)
        local worker_count=$(docker node ls --filter role=worker --format "{{.Status}}" | grep -c "Ready" || echo 0)
        
        if [ "$manager_count" -ge 1 ] && [ "$worker_count" -ge 2 ]; then
            success "Swarm has sufficient nodes (${manager_count} managers, ${worker_count} workers)"
        else
            warning "Insufficient nodes for HA (${manager_count} managers, ${worker_count} workers)"
        fi
    else
        error "Docker Swarm is not active"
    fi
    
    # Network
    check "Docker Networks"
    if docker network ls --format "{{.Name}}" | grep -q "traefik-public"; then
        success "External network 'traefik-public' exists"
    else
        error "External network 'traefik-public' not found"
    fi
    
    # Volumes
    check "Docker Volumes"
    local required_volumes=("neo4j_data" "postgres_data" "redis_data" "milvus_data")
    for volume in "${required_volumes[@]}"; do
        if docker volume ls --format "{{.Name}}" | grep -q "aiplatform_${volume}"; then
            success "Volume 'aiplatform_${volume}' exists"
        else
            warning "Volume 'aiplatform_${volume}' not found"
        fi
    done
    
    echo
}

# 2. Secrets and Configuration
check_secrets() {
    echo "🔐 Secrets and Configuration"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Required secrets
    local required_secrets=("postgres_password" "redis_password" "neo4j_password" "jwt_secret" "openai_api_key")
    
    for secret in "${required_secrets[@]}"; do
        check "Secret: ${secret}"
        if docker secret ls --format "{{.Name}}" | grep -q "^${secret}$"; then
            success "Secret '${secret}' exists"
        else
            error "Secret '${secret}' not found"
        fi
    done
    
    # Configuration files
    check "Configuration Files"
    if [ -f ".env.production" ]; then
        success "Environment configuration exists"
    else
        warning "No .env.production file found"
    fi
    
    if docker config ls --format "{{.Name}}" | grep -q "prometheus_config"; then
        success "Prometheus configuration exists"
    else
        warning "Prometheus configuration not found"
    fi
    
    echo
}

# 3. Service Health
check_services() {
    echo "🚀 Service Health"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Core services
    local core_services=("traefik" "backend" "frontend" "postgres" "redis" "neo4j" "milvus-standalone")
    
    for service in "${core_services[@]}"; do
        check "Service: aiplatform_${service}"
        
        local replicas=$(docker service ls --filter name="aiplatform_${service}" --format "{{.Replicas}}" 2>/dev/null || echo "0/0")
        local running=$(echo "$replicas" | cut -d'/' -f1)
        local desired=$(echo "$replicas" | cut -d'/' -f2)
        
        if [ "$running" = "$desired" ] && [ "$running" -gt "0" ]; then
            success "Service 'aiplatform_${service}' is healthy (${replicas})"
        elif [ "$desired" -gt "0" ]; then
            error "Service 'aiplatform_${service}' is unhealthy (${replicas})"
        else
            warning "Service 'aiplatform_${service}' not found"
        fi
    done
    
    echo
}

# 4. Database Connectivity
check_databases() {
    echo "🗄️  Database Connectivity"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # PostgreSQL
    check "PostgreSQL Connection"
    local pg_container=$(docker ps -q -f name=aiplatform_postgres)
    if [ -n "$pg_container" ]; then
        if docker exec "$pg_container" pg_isready -U postgres -d aiplatform >/dev/null 2>&1; then
            success "PostgreSQL is ready and accepting connections"
        else
            error "PostgreSQL connection failed"
        fi
    else
        error "PostgreSQL container not found"
    fi
    
    # Neo4j
    check "Neo4j Connection"
    local neo4j_container=$(docker ps -q -f name=aiplatform_neo4j)
    if [ -n "$neo4j_container" ]; then
        if timeout 10 docker exec "$neo4j_container" cypher-shell -u neo4j -p development "RETURN 1" >/dev/null 2>&1; then
            success "Neo4j is ready and accepting connections"
        else
            error "Neo4j connection failed"
        fi
    else
        error "Neo4j container not found"
    fi
    
    # Redis
    check "Redis Connection"
    local redis_container=$(docker ps -q -f name=aiplatform_redis)
    if [ -n "$redis_container" ]; then
        if docker exec "$redis_container" redis-cli ping | grep -q PONG; then
            success "Redis is ready and accepting connections"
        else
            error "Redis connection failed"
        fi
    else
        error "Redis container not found"
    fi
    
    # Milvus
    check "Milvus Connection"
    local milvus_container=$(docker ps -q -f name=aiplatform_milvus)
    if [ -n "$milvus_container" ]; then
        if timeout 10 docker exec "$milvus_container" curl -s -f http://localhost:9091/healthz >/dev/null 2>&1; then
            success "Milvus is ready and accepting connections"
        else
            error "Milvus health check failed"
        fi
    else
        error "Milvus container not found"
    fi
    
    echo
}

# 5. API Endpoints
check_api_endpoints() {
    echo "🌐 API Endpoints"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Health endpoints
    local endpoints=(
        "https://${API_DOMAIN}/health|Backend Health"
        "https://${DOMAIN_NAME}/|Frontend"
        "https://${MONITORING_DOMAIN}/api/health|Grafana"
        "https://${TRAEFIK_DOMAIN}/api/rawdata|Traefik API"
    )
    
    for endpoint_info in "${endpoints[@]}"; do
        local url=$(echo "$endpoint_info" | cut -d'|' -f1)
        local name=$(echo "$endpoint_info" | cut -d'|' -f2)
        
        check "Endpoint: ${name}"
        local response_code=$(curl -s -o /dev/null -w "%{http_code}" -k --connect-timeout 10 "$url" 2>/dev/null || echo "000")
        
        if [ "$response_code" -eq 200 ]; then
            success "${name} endpoint is accessible (${response_code})"
        elif [ "$response_code" -eq 302 ] || [ "$response_code" -eq 301 ]; then
            success "${name} endpoint is accessible with redirect (${response_code})"
        else
            error "${name} endpoint failed (${response_code})"
        fi
    done
    
    echo
}

# 6. SSL/TLS Configuration
check_ssl() {
    echo "🔒 SSL/TLS Configuration"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    local domains=("${DOMAIN_NAME}" "${API_DOMAIN}" "${MONITORING_DOMAIN}")
    
    for domain in "${domains[@]}"; do
        check "SSL Certificate: ${domain}"
        
        local cert_info=$(echo | timeout 10 openssl s_client -servername "$domain" -connect "${domain}:443" 2>/dev/null | openssl x509 -noout -dates 2>/dev/null)
        
        if [ $? -eq 0 ]; then
            local expiry=$(echo "$cert_info" | grep notAfter | cut -d= -f2)
            local expiry_epoch=$(date -d "$expiry" +%s 2>/dev/null || echo 0)
            local current_epoch=$(date +%s)
            local days_until_expiry=$(( (expiry_epoch - current_epoch) / 86400 ))
            
            if [ "$days_until_expiry" -gt 30 ]; then
                success "SSL certificate for ${domain} is valid (expires in ${days_until_expiry} days)"
            elif [ "$days_until_expiry" -gt 0 ]; then
                warning "SSL certificate for ${domain} expires soon (${days_until_expiry} days)"
            else
                error "SSL certificate for ${domain} has expired"
            fi
        else
            error "Could not retrieve SSL certificate for ${domain}"
        fi
    done
    
    echo
}

# 7. Performance Metrics
check_performance() {
    echo "⚡ Performance Metrics"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # API Response Time
    check "API Response Time"
    local start_time=$(date +%s%3N)
    local response_code=$(curl -s -o /dev/null -w "%{http_code}" "https://${API_DOMAIN}/health" --connect-timeout 10 2>/dev/null || echo "000")
    local end_time=$(date +%s%3N)
    local response_time=$((end_time - start_time))
    
    if [ "$response_code" -eq 200 ]; then
        if [ "$response_time" -lt 2000 ]; then
            success "API response time is acceptable (${response_time}ms)"
        else
            warning "API response time is slow (${response_time}ms)"
        fi
    else
        error "API health check failed (${response_code})"
    fi
    
    # Database Performance
    check "Database Query Performance"
    local pg_container=$(docker ps -q -f name=aiplatform_postgres)
    if [ -n "$pg_container" ]; then
        local query_time=$(docker exec "$pg_container" psql -U postgres -d aiplatform -c "SELECT 1;" -t 2>/dev/null | wc -l)
        if [ "$query_time" -gt 0 ]; then
            success "PostgreSQL queries are responsive"
        else
            error "PostgreSQL query test failed"
        fi
    fi
    
    echo
}

# 8. Resource Utilization
check_resources() {
    echo "💾 Resource Utilization"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Memory usage
    check "Memory Usage"
    local memory_info=$(free | grep MemAvailable | awk '{print int($2/1024/1024)}')
    if [ "$memory_info" -gt 2 ]; then
        success "Available memory: ${memory_info}GB"
    else
        warning "Low available memory: ${memory_info}GB"
    fi
    
    # Disk usage
    check "Disk Usage"
    local disk_usage=$(df /var/lib/docker | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$disk_usage" -lt 80 ]; then
        success "Disk usage is acceptable (${disk_usage}%)"
    elif [ "$disk_usage" -lt 90 ]; then
        warning "Disk usage is high (${disk_usage}%)"
    else
        error "Disk usage is critical (${disk_usage}%)"
    fi
    
    # Docker system usage
    check "Docker System Usage"
    local docker_space=$(docker system df --format "table {{.Type}}\t{{.Size}}" | tail -n +2 | awk '{sum+=$2} END {print sum}' || echo 0)
    info "Docker system space usage: ${docker_space}MB"
    
    echo
}

# 9. Security Checks
check_security() {
    echo "🛡️  Security Configuration"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Firewall status
    check "Firewall Status"
    if command -v ufw >/dev/null 2>&1; then
        if ufw status | grep -q "Status: active"; then
            success "UFW firewall is active"
        else
            warning "UFW firewall is not active"
        fi
    elif command -v firewall-cmd >/dev/null 2>&1; then
        if systemctl is-active --quiet firewalld; then
            success "Firewalld is active"
        else
            warning "Firewalld is not active"
        fi
    else
        warning "No firewall detected"
    fi
    
    # Docker daemon security
    check "Docker Daemon Security"
    if docker version --format '{{.Server.Version}}' >/dev/null 2>&1; then
        success "Docker daemon is accessible"
        
        # Check for rootless mode
        if docker info 2>/dev/null | grep -q "rootless"; then
            success "Docker is running in rootless mode"
        else
            warning "Docker is running with root privileges"
        fi
    else
        error "Docker daemon is not accessible"
    fi
    
    # Container security
    check "Container Security"
    local privileged_containers=$(docker ps --format "{{.Names}}" --filter "label=privileged=true" | wc -l)
    if [ "$privileged_containers" -eq 0 ]; then
        success "No privileged containers detected"
    else
        warning "${privileged_containers} privileged containers found"
    fi
    
    echo
}

# 10. Monitoring and Alerting
check_monitoring() {
    echo "📊 Monitoring and Alerting"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Prometheus
    check "Prometheus Metrics"
    if docker ps --filter name=aiplatform_prometheus --format "{{.Status}}" | grep -q "Up"; then
        success "Prometheus is running"
        
        # Test metrics endpoint
        local prom_container=$(docker ps -q -f name=aiplatform_prometheus)
        if [ -n "$prom_container" ]; then
            if docker exec "$prom_container" curl -s http://localhost:9090/-/healthy >/dev/null 2>&1; then
                success "Prometheus health check passed"
            else
                error "Prometheus health check failed"
            fi
        fi
    else
        error "Prometheus is not running"
    fi
    
    # Grafana
    check "Grafana Dashboard"
    local grafana_response=$(curl -s -o /dev/null -w "%{http_code}" "https://${MONITORING_DOMAIN}/api/health" 2>/dev/null || echo "000")
    if [ "$grafana_response" -eq 200 ]; then
        success "Grafana dashboard is accessible"
    else
        error "Grafana dashboard is not accessible (${grafana_response})"
    fi
    
    # Alerting
    check "Alert Rules"
    local prom_container=$(docker ps -q -f name=aiplatform_prometheus)
    if [ -n "$prom_container" ]; then
        local rule_count=$(docker exec "$prom_container" curl -s http://localhost:9090/api/v1/rules 2>/dev/null | jq '.data.groups[].rules | length' 2>/dev/null | head -1 || echo 0)
        if [ "$rule_count" -gt 0 ]; then
            success "Alert rules are loaded (${rule_count} rules)"
        else
            warning "No alert rules found"
        fi
    fi
    
    echo
}

# 11. Backup System
check_backup() {
    echo "💾 Backup System"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Backup script
    check "Backup Scripts"
    if [ -x "./scripts/backup.sh" ]; then
        success "Backup script is executable"
    else
        error "Backup script not found or not executable"
    fi
    
    if [ -x "./scripts/restore.sh" ]; then
        success "Restore script is executable"
    else
        error "Restore script not found or not executable"
    fi
    
    # Backup directory
    check "Backup Storage"
    if [ -d "/backups" ]; then
        local backup_space=$(df /backups | tail -1 | awk '{print int($2/1024/1024)}')
        success "Backup directory exists (${backup_space}GB available)"
    else
        warning "Backup directory not found"
    fi
    
    # S3 configuration (if available)
    check "S3 Backup Configuration"
    if command -v aws >/dev/null 2>&1; then
        if aws sts get-caller-identity >/dev/null 2>&1; then
            success "AWS CLI is configured"
        else
            warning "AWS CLI is not configured"
        fi
    else
        warning "AWS CLI not found"
    fi
    
    echo
}

# Generate summary report
generate_summary() {
    echo "📋 Summary Report"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo
    echo "Total Checks: ${TOTAL_CHECKS}"
    echo -e "${GREEN}Passed: ${PASSED_CHECKS}${NC}"
    echo -e "${YELLOW}Warnings: ${WARNING_CHECKS}${NC}"
    echo -e "${RED}Failed: ${FAILED_CHECKS}${NC}"
    echo
    
    local success_rate=$(( (PASSED_CHECKS * 100) / TOTAL_CHECKS ))
    
    if [ "$FAILED_CHECKS" -eq 0 ]; then
        if [ "$WARNING_CHECKS" -eq 0 ]; then
            echo -e "${GREEN}🎉 PRODUCTION READY: All checks passed successfully!${NC}"
            return 0
        else
            echo -e "${YELLOW}⚠️  PRODUCTION READY WITH WARNINGS: ${WARNING_CHECKS} warnings detected${NC}"
            return 0
        fi
    else
        echo -e "${RED}❌ NOT PRODUCTION READY: ${FAILED_CHECKS} critical issues detected${NC}"
        echo
        echo "Please resolve the failed checks before deploying to production."
        return 1
    fi
}

# Main execution
main() {
    print_header
    
    # Run all checks
    check_infrastructure
    check_secrets
    check_services
    check_databases
    check_api_endpoints
    check_ssl
    check_performance
    check_resources
    check_security
    check_monitoring
    check_backup
    
    # Generate summary
    echo
    generate_summary
}

# Signal handlers
trap 'echo -e "\n${RED}Production check interrupted${NC}"; exit 1' INT TERM

# Run main function
main "$@"