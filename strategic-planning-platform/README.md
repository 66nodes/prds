# Strategic Planning Platform

An AI-powered strategic planning platform that transforms high-level project ideas into comprehensive strategic planning documents through AI-driven conversational workflows.

## 🏗️ Architecture

- **Frontend**: Nuxt.js 4 with TypeScript, Vue 3, Tailwind CSS
- **Backend**: FastAPI with Python 3.11+, AsyncIO
- **Graph Database**: Neo4j with GraphRAG for hallucination prevention
- **User Database**: PostgreSQL for authentication and user data
- **Cache**: Redis for sessions, caching, and message queuing
- **Real-time**: WebSocket for live collaboration

## 📋 Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- Node.js 18+ and pnpm (for local development)
- Python 3.11+ (for local development)
- Git

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd strategic-planning-platform

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
vim .env  # or your preferred editor
```

### 2. Configure Environment

Update the `.env` file with your settings:

```bash
# Required: Database passwords
POSTGRES_PASSWORD=your_secure_postgres_password
NEO4J_PASSWORD=your_secure_neo4j_password
REDIS_PASSWORD=your_secure_redis_password

# Required: JWT secrets
JWT_SECRET=your_super_secret_jwt_key_at_least_32_chars_long
JWT_REFRESH_SECRET=your_different_refresh_secret_key

# Required: LLM API keys
OPENROUTER_API_KEY=your_openrouter_api_key
OPENAI_API_KEY=your_openai_api_key  # fallback
```

### 3. Start Development Environment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Start with development tools (optional)
COMPOSE_PROFILES=dev-tools docker-compose up -d
```

### 4. Initialize Database

```bash
# Initialize PostgreSQL and Neo4j schemas
docker-compose exec backend python scripts/init_db.py

# Create admin user
docker-compose exec backend python scripts/create_admin_user.py
```

### 5. Access the Application

- **Frontend**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Neo4j Browser**: http://localhost:7474 (neo4j / your_neo4j_password)
- **Grafana**: http://localhost:3001 (admin / admin)
- **Mailhog** (dev): http://localhost:8025
- **PgAdmin** (dev): http://localhost:5050

## 🛠️ Development

### Frontend Development

```bash
cd frontend

# Install dependencies
pnpm install

# Start development server
pnpm dev

# Type checking
pnpm type-check

# Run tests
pnpm test
pnpm test:e2e

# Build for production
pnpm build
```

### Backend Development

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Start development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest
pytest --cov=. --cov-report=html

# Format code
black .
isort .

# Lint code
flake8
mypy .
```

### Shared Types

The shared types are located in `/shared/types/` and are used by both frontend and backend:

```typescript
// Example usage in frontend
import { User, PRD, ValidationResponse } from '@/shared/types'

// Example usage in backend
from shared.types.api import ApiResponse, ApiError
```

## 🐳 Docker Configuration

### Production Deployment

```bash
# Build and start production environment
docker-compose -f docker-compose.yml up -d

# Start with monitoring
COMPOSE_PROFILES=monitoring docker-compose up -d

# Start with logging
COMPOSE_PROFILES=logging docker-compose up -d

# Start with all profiles
COMPOSE_PROFILES=monitoring,logging docker-compose up -d
```

### Health Checks

```bash
# Check service health
docker-compose ps

# Detailed health check
curl http://localhost/health
curl http://localhost:8000/health
```

### Scaling Services

```bash
# Scale backend instances
docker-compose up -d --scale backend=3

# Scale with load balancer
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## 📊 Monitoring

### Grafana Dashboards

1. Open http://localhost:3001
2. Login with admin/admin
3. Import dashboards from `/monitoring/grafana/dashboards/`

### Prometheus Metrics

- Application metrics: http://localhost:9090
- Custom metrics available at `/metrics` endpoint
- Neo4j metrics: http://localhost:7474/browser

### Logging

With logging profile enabled:

- **Elasticsearch**: http://localhost:9200
- **Kibana**: http://localhost:5601
- **Logstash**: Configured for log processing

## 🔒 Security

### Environment Security

```bash
# Generate secure secrets
openssl rand -hex 32  # For JWT_SECRET
openssl rand -hex 32  # For JWT_REFRESH_SECRET

# Set up SSL certificates (production)
mkdir -p nginx/ssl
# Copy your certificates to nginx/ssl/
```

### Database Security

- All databases use encrypted connections in production
- Row-level security enabled in PostgreSQL
- Neo4j auth configured with strong passwords
- Redis auth enabled with password protection

### API Security

- JWT-based authentication with refresh tokens
- Rate limiting on all endpoints
- CORS configuration for allowed origins
- Input validation on all endpoints
- SQL injection prevention with parameterized queries

## 🧪 Testing

### Running All Tests

```bash
# Frontend tests
cd frontend && pnpm test

# Backend tests
cd backend && pytest

# E2E tests
cd frontend && pnpm test:e2e

# Load testing
cd backend && locust -f tests/load_test.py
```

### Test Coverage

- Target: >90% code coverage
- Frontend: Jest + Vue Test Utils + Playwright
- Backend: Pytest + Pytest-cov + AsyncIO testing

## 📖 API Documentation

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### GraphQL Playground

GraphRAG queries are available through the REST API with GraphQL-like syntax for Neo4j queries.

## 🔧 Configuration

### Feature Flags

Control features via environment variables:

```bash
ENABLE_GRAPHRAG_VALIDATION=true
ENABLE_REAL_TIME_COLLABORATION=true
ENABLE_EMAIL_NOTIFICATIONS=true
ENABLE_ANALYTICS=true
```

### Performance Tuning

```bash
# Backend workers
WORKERS=4
WORKER_CONNECTIONS=1000

# Database connections
DB_POOL_SIZE=20
NEO4J_MAX_CONNECTION_POOL_SIZE=50
REDIS_MAX_CONNECTIONS=100

# Caching
CACHE_DEFAULT_TTL=3600
CACHE_MAX_SIZE=1000
```

## 🚨 Troubleshooting

### Common Issues

**Services won't start:**
```bash
# Check Docker logs
docker-compose logs [service-name]

# Restart specific service
docker-compose restart [service-name]

# Rebuild service
docker-compose up -d --build [service-name]
```

**Database connection issues:**
```bash
# Check database status
docker-compose exec postgres pg_isready
docker-compose exec neo4j cypher-shell -u neo4j -p [password] "RETURN 1"
docker-compose exec redis redis-cli ping
```

**Frontend build issues:**
```bash
# Clear node_modules and reinstall
docker-compose exec frontend rm -rf node_modules
docker-compose exec frontend pnpm install

# Clear Nuxt cache
docker-compose exec frontend rm -rf .nuxt
```

**Backend import issues:**
```bash
# Check Python path
docker-compose exec backend python -c "import sys; print(sys.path)"

# Reinstall dependencies
docker-compose exec backend pip install -r requirements.txt
```

### Performance Issues

```bash
# Check resource usage
docker stats

# Check database performance
docker-compose exec postgres psql -U strategic_user -d strategic_planning -c "SELECT * FROM pg_stat_activity;"

# Check Redis memory usage
docker-compose exec redis redis-cli info memory
```

### Logging and Debugging

```bash
# Enable debug logging
export LOG_LEVEL=debug

# Follow logs in real-time
docker-compose logs -f --tail=100

# Debug specific service
docker-compose exec backend python -m pdb app/main.py
```

## 📚 Documentation

- [API Documentation](./docs/api.md)
- [Frontend Development Guide](./frontend/README.md)
- [Backend Development Guide](./backend/README.md)
- [Deployment Guide](./docs/deployment.md)
- [GraphRAG Integration](./docs/graphrag.md)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For support and questions:

- Create an issue in GitHub
- Check the troubleshooting guide above
- Review logs for error details
- Consult the API documentation

## 🗺️ Roadmap

- [ ] Advanced GraphRAG validation algorithms
- [ ] Multi-tenant support
- [ ] Mobile application
- [ ] Advanced analytics dashboard
- [ ] Third-party integrations (Slack, Teams, Jira)
- [ ] AI-powered requirement suggestions
- [ ] Automated testing generation
- [ ] Advanced export formats