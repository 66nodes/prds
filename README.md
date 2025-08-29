# AI-Powered Strategic Planning Platform

An enterprise-grade web application that transforms high-level project ideas into comprehensive strategic planning documents through AI-driven conversational workflows.

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm 9+
- Python 3.11+
- Docker and Docker Compose
- Neo4j instance (existing)

### Development Setup

1. **Clone and setup environment**:
```bash
git clone https://github.com/66nodes/prds.git
cd prds
cp .env.example .env
# Edit .env with your configuration
```

2. **Start services**:
```bash
docker-compose up -d
```

3. **Access applications**:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/api/docs

## 🏗️ Architecture

### Technology Stack
- **Frontend**: Nuxt.js 4, Vue 3, TypeScript, Tailwind CSS, Pinia
- **Backend**: FastAPI, Python 3.11+, PostgreSQL, Neo4j, Redis
- **AI/ML**: GraphRAG, OpenRouter, Multi-LLM support
- **Infrastructure**: Docker, Kubernetes, Prometheus, Grafana

### Project Structure
```
strategic-planning-platform/
├── frontend/          # Nuxt.js 4 application
├── backend/           # FastAPI services
├── shared/            # Shared TypeScript types
└── docker/            # Docker configurations

docs/                  # Documentation
├── PRD.md            # Project Requirements Document
├── implementation_runbook.md
├── ui-ux-requirements.md
└── RAG_strategy.md
```

## 🎯 Key Features

### AI-Powered PRD Creation
- **Phase 0**: Project concept capture with similarity matching
- **Phase 1**: Objective clarification through dynamic questioning
- **Phase 2**: SMART objective generation and refinement
- **Phase 3**: Section-by-section collaborative creation
- **Phase 4**: Document synthesis and export

### GraphRAG Integration
- **<2% Hallucination Rate**: Multi-level validation pipeline
- **Entity Validation**: Real-time fact checking against knowledge graph
- **Community Validation**: Pattern verification within requirement clusters
- **Global Validation**: Strategic alignment with organizational objectives

### Enterprise Features
- **Role-Based Access Control**: Admin, Project Manager, Contributor, Viewer roles
- **Real-time Collaboration**: WebSocket-based live editing and presence
- **Multi-format Export**: PDF, Word, Markdown with professional branding
- **Work Breakdown Structure**: Automatic WBS generation with dependencies

## 📊 Performance Targets

- **Page Load**: <2 seconds initial load
- **API Response**: <200ms for GraphRAG validation
- **Concurrent Users**: 100+ simultaneous sessions
- **Uptime**: 99.9% availability SLA
- **Quality Score**: >90% stakeholder satisfaction

## 🔧 Development

### Frontend Development
```bash
cd strategic-planning-platform/frontend
npm install
npm run dev
```

### Backend Development
```bash
cd strategic-planning-platform/backend
pip install -r requirements.txt
python -m app.main
```

### Testing
```bash
# Frontend tests
npm run test
npm run test:coverage

# Backend tests
pytest --cov=app tests/
```

## 🐳 Docker Deployment

### Development
```bash
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

### Production
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## 📚 Documentation

- **[PRD](docs/PRD.md)**: Comprehensive project requirements
- **[Implementation Guide](docs/implementation_runbook.md)**: Step-by-step setup
- **[UI/UX Requirements](docs/ui-ux-requirements.md)**: Design specifications
- **[RAG Strategy](docs/RAG_strategy.md)**: GraphRAG implementation details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Microsoft GraphRAG for hallucination prevention framework
- OpenRouter for multi-LLM integration
- Neo4j for graph database capabilities
- The open-source community for the excellent tools and libraries

---

**Built with ❤️ by [66nodes](https://github.com/66nodes)**