# AI Agent Platform - Production Deployment Runbook

## 🎯 Overview

This runbook provides comprehensive instructions for deploying and maintaining the AI Agent Platform in production using Docker Swarm. The platform includes:

- **Frontend**: Nuxt.js application
- **Backend**: FastAPI with PydanticAI agents
- **Databases**: PostgreSQL, Neo4j, Redis, Milvus
- **Monitoring**: Prometheus, Grafana, Alertmanager
- **Reverse Proxy**: Traefik with SSL/TLS

## 📋 Prerequisites

### Infrastructure Requirements

| Component | CPU | RAM | Storage | Notes |
|-----------|-----|-----|---------|-------|
| Manager Node | 4 vCPU | 8 GB | 100 GB SSD | Docker Swarm manager |
| Worker Nodes | 8 vCPU | 16 GB | 200 GB SSD | Application workloads |
| Database Node | 4 vCPU | 32 GB | 500 GB SSD | Dedicated for databases |
| **Total Min** | **16 vCPU** | **56 GB** | **800 GB** | 3-node cluster |

### Software Requirements

- **Docker Engine**: 24.0+
- **Docker Compose**: v2.20+
- **Git**: Latest
- **OpenSSL**: For certificate generation
- **AWS CLI**: For S3 backups (optional)

### Network Requirements

- **Ports 80/443**: HTTP/HTTPS traffic
- **Port 2377**: Docker Swarm management
- **Ports 7946/4789**: Docker Swarm communication
- **Internal networking**: Service discovery and communication

## 🚀 Initial Setup

### 1. Docker Swarm Initialization

```bash
# On the manager node
docker swarm init --advertise-addr <MANAGER_IP>

# Join worker nodes (run on each worker)
docker swarm join --token <WORKER_TOKEN> <MANAGER_IP>:2377

# Label nodes for service placement
docker node update --label-add postgres=true <NODE_ID>
docker node update --label-add neo4j=true <NODE_ID>
docker node update --label-add redis=true <NODE_ID>
docker node update --label-add milvus=true <NODE_ID>
docker node update --label-add monitoring=true <NODE_ID>
```

### 2. Network Setup

```bash
# Create external network for Traefik
docker network create --driver=overlay traefik-public
```

### 3. Secrets Management

```bash
# Run the secrets creation script
./scripts/create-secrets.sh

# Verify secrets are created
docker secret ls
```

### 4. Configuration Files

```bash
# Copy and customize environment file
cp .env.production .env.production.local

# Edit with your specific values
nano .env.production.local

# Create Prometheus config
docker config create prometheus_config monitoring/prometheus/prometheus.yml

# Create Grafana dashboard config
docker config create grafana_dashboard_config monitoring/grafana/provisioning/dashboards/dashboard.yml
```

## 🏗️ Deployment Process

### Step 1: Pre-deployment Checks

```bash
# Verify all nodes are ready
docker node ls

# Check available resources
docker system df

# Validate configuration files
docker-compose -f docker-stack.yml config
```

### Step 2: Deploy Infrastructure Services

```bash
# Deploy the complete stack
docker stack deploy -c docker-stack.yml aiplatform

# Monitor deployment progress
watch docker service ls
```

### Step 3: Verify Deployment

```bash
# Check service health
docker service ps aiplatform_backend
docker service ps aiplatform_frontend
docker service ps aiplatform_postgres
docker service ps aiplatform_neo4j

# View service logs
docker service logs aiplatform_backend
docker service logs aiplatform_frontend
```

### Step 4: Initialize Databases

```bash
# PostgreSQL: Create initial schema
docker exec -i $(docker ps -q -f name=aiplatform_postgres) psql -U postgres aiplatform < database/schema.sql

# Neo4j: Load initial data
docker exec -i $(docker ps -q -f name=aiplatform_neo4j) cypher-shell -u neo4j -p ${NEO4J_PASSWORD} < database/neo4j/initial_data.cypher

# Milvus: Create collections
python scripts/setup_milvus_collections.py
```

### Step 5: SSL Certificate Setup

```bash
# Traefik will automatically request Let's Encrypt certificates
# Monitor certificate acquisition
docker service logs aiplatform_traefik

# Verify HTTPS is working
curl -I https://your-domain.com
```

## 📊 Monitoring Setup

### Access Monitoring Dashboards

- **Grafana**: https://monitoring.your-domain.com
- **Traefik Dashboard**: https://traefik.your-domain.com
- **Prometheus**: Internal access only

### Key Metrics to Monitor

| Metric | Threshold | Action |
|--------|-----------|---------|
| API Response Time | > 2s | Scale backend |
| Error Rate | > 5% | Investigate logs |
| Memory Usage | > 85% | Add resources |
| Disk Space | < 15% | Clean up/expand |
| Hallucination Rate | > 2% | Review AI models |

### Alert Configuration

Alerts are configured in `monitoring/prometheus/alerts.yml`:

- **Critical**: Immediate notification (5-15 min response)
- **Warning**: Standard notification (30-60 min response)
- **Info**: Logged only

## 🔄 Maintenance Operations

### Rolling Updates

```bash
# Update backend service
docker service update --image ghcr.io/your-org/backend:v1.2.0 aiplatform_backend

# Update frontend service  
docker service update --image ghcr.io/your-org/frontend:v1.2.0 aiplatform_frontend

# Monitor rollout
docker service ps aiplatform_backend
```

### Scaling Services

```bash
# Scale backend horizontally
docker service scale aiplatform_backend=6

# Scale frontend
docker service scale aiplatform_frontend=4

# Monitor resource usage
docker stats
```

### Database Maintenance

```bash
# PostgreSQL vacuum and analyze
docker exec -i $(docker ps -q -f name=aiplatform_postgres) psql -U postgres aiplatform -c "VACUUM ANALYZE;"

# Neo4j consistency check
docker exec -i $(docker ps -q -f name=aiplatform_neo4j) neo4j-admin check-consistency

# Redis memory optimization
docker exec -i $(docker ps -q -f name=aiplatform_redis) redis-cli MEMORY PURGE
```

## 💾 Backup and Recovery

### Automated Backups

```bash
# Setup automated backups (daily at 2 AM)
echo "0 2 * * * /path/to/scripts/backup.sh" | crontab -

# Manual backup
./scripts/backup.sh

# Verify backup
ls -la /backups/$(date +%Y-%m-%d)
```

### Recovery Procedures

```bash
# Full system restore
./scripts/restore.sh 2024-01-15

# Selective restore (PostgreSQL only)
./scripts/restore.sh -p 2024-01-15

# Restore from S3
./scripts/restore.sh -s 2024-01-15
```

### Disaster Recovery Plan

1. **RTO**: 4 hours (Recovery Time Objective)
2. **RPO**: 24 hours (Recovery Point Objective)
3. **Backup Retention**: 30 days local, 90 days S3
4. **Testing**: Monthly DR drills

## 🛠️ Troubleshooting Guide

### Common Issues

#### Service Won't Start

```bash
# Check service status
docker service ps aiplatform_backend --no-trunc

# View detailed logs
docker service logs aiplatform_backend --details

# Check resource constraints
docker node ls
docker service inspect aiplatform_backend
```

#### Database Connection Issues

```bash
# Test PostgreSQL connectivity
docker exec -i $(docker ps -q -f name=aiplatform_postgres) pg_isready

# Test Neo4j connectivity
docker exec -i $(docker ps -q -f name=aiplatform_neo4j) cypher-shell -u neo4j -p ${NEO4J_PASSWORD} "RETURN 1"

# Test Redis connectivity
docker exec -i $(docker ps -q -f name=aiplatform_redis) redis-cli ping
```

#### Performance Issues

```bash
# Check resource usage
docker stats --no-stream

# Analyze slow queries (PostgreSQL)
docker exec -i $(docker ps -q -f name=aiplatform_postgres) psql -U postgres aiplatform -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"

# Monitor API performance
curl -w "@curl-format.txt" -s -o /dev/null https://api.your-domain.com/health
```

#### High Memory Usage

```bash
# Identify memory-intensive containers
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

# Restart services with memory leaks
docker service update --force aiplatform_backend

# Scale down temporarily
docker service scale aiplatform_backend=2
```

### Emergency Procedures

#### Complete System Failure

1. **Assess Impact**: Check monitoring dashboards
2. **Isolate Issue**: Identify failing components
3. **Restore Service**: Use backup/DR procedures
4. **Post-Incident**: Conduct root cause analysis

#### Security Breach

1. **Immediate**: Isolate affected systems
2. **Assess**: Determine scope of compromise
3. **Rotate**: Update all secrets and keys
4. **Restore**: From clean backup if needed
5. **Report**: Document and report incident

## 📈 Performance Optimization

### Database Optimization

```sql
-- PostgreSQL index optimization
CREATE INDEX CONCURRENTLY idx_prd_created_at ON prds(created_at);
CREATE INDEX CONCURRENTLY idx_users_email ON users(email);

-- Neo4j index optimization
CREATE INDEX FOR (n:Entity) ON (n.name);
CREATE INDEX FOR (n:Project) ON (n.id);
```

### Application Optimization

```bash
# Enable query caching
export REDIS_CACHE_TTL=3600

# Optimize connection pools
export DB_POOL_SIZE=20
export DB_MAX_OVERFLOW=30

# Enable compression
export GZIP_COMPRESSION=true
```

### Infrastructure Optimization

```bash
# SSD optimization
echo mq-deadline > /sys/block/sda/queue/scheduler

# Network optimization
echo 'net.core.rmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' >> /etc/sysctl.conf
```

## 🔒 Security Checklist

### Pre-Production Security

- [ ] All secrets stored in Docker Secrets
- [ ] SSL/TLS certificates configured
- [ ] Firewall rules implemented
- [ ] Database access restricted
- [ ] API rate limiting enabled
- [ ] Monitoring and alerting active
- [ ] Backup encryption enabled
- [ ] Access logs configured

### Regular Security Tasks

- [ ] Monthly security updates
- [ ] Quarterly penetration testing
- [ ] Annual security audit
- [ ] Key rotation (every 90 days)
- [ ] Certificate renewal monitoring
- [ ] Vulnerability scanning

## 📞 Support Contacts

| Role | Contact | Availability |
|------|---------|--------------|
| Platform Team | platform@company.com | 24/7 |
| DevOps Engineer | devops@company.com | Business hours |
| Database Admin | dba@company.com | On-call rotation |
| Security Team | security@company.com | 24/7 |

## 📚 Additional Resources

- [Docker Swarm Documentation](https://docs.docker.com/engine/swarm/)
- [Traefik Configuration](https://doc.traefik.io/traefik/)
- [Prometheus Alerting](https://prometheus.io/docs/alerting/latest/)
- [Grafana Dashboards](https://grafana.com/docs/)
- [PostgreSQL Administration](https://www.postgresql.org/docs/)
- [Neo4j Operations](https://neo4j.com/docs/operations-manual/)

---

**Last Updated**: $(date +"%Y-%m-%d")  
**Version**: 1.0  
**Maintained By**: Platform Team