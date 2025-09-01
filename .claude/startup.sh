#!/bin/bash
# Agent System Startup Script

set -e

echo "🚀 Starting Agent Ecosystem..."

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    command -v kubectl >/dev/null 2>&1 || { echo "kubectl required but not installed."; exit 1; }
    command -v helm >/dev/null 2>&1 || { echo "helm required but not installed."; exit 1; }
    command -v docker >/dev/null 2>&1 || { echo "docker required but not installed."; exit 1; }
}

# Create namespace
create_namespace() {
    echo "Creating namespace..."
    kubectl create namespace ai-agents --dry-run=client -o yaml | kubectl apply -f -
}

# Deploy infrastructure components
deploy_infrastructure() {
    echo "Deploying infrastructure..."
    kubectl apply -f deployments/global-deployment-config.yaml
    kubectl apply -f config/resource-allocation.yaml
    kubectl apply -f monitoring/monitoring-stack.yaml
}

# Deploy agents by category
deploy_agents() {
    echo "Deploying agents..."
    
    # Core agents first
    kubectl apply -f deployments/deployment-development-agents.yaml
    kubectl apply -f deployments/deployment-infrastructure-agents.yaml
    kubectl apply -f deployments/deployment-ai-ml-agents.yaml
    
    # Wait for core agents
    kubectl wait --for=condition=ready pod -l tier=core -n ai-agents --timeout=300s
    
    # Deploy remaining agents
    for deployment in deployments/deployment-*.yaml; do
        kubectl apply -f "$deployment"
    done
}

# Setup monitoring
setup_monitoring() {
    echo "Setting up monitoring..."
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    helm install prometheus prometheus-community/kube-prometheus-stack \
        --namespace ai-agents \
        --values monitoring/prometheus-values.yaml
        
    helm install grafana grafana/grafana \
        --namespace ai-agents \
        --values monitoring/grafana-values.yaml
}

# Verify deployment
verify_deployment() {
    echo "Verifying deployment..."
    
    # Check pod status
    kubectl get pods -n ai-agents
    
    # Check services
    kubectl get services -n ai-agents
    
    # Run health checks
    ./health-check.sh
}

# Main execution
main() {
    check_prerequisites
    create_namespace
    deploy_infrastructure
    deploy_agents
    setup_monitoring
    verify_deployment
    
    echo "✅ Agent ecosystem successfully deployed!"
    echo "📊 Access monitoring at: http://localhost:3000 (port-forward required)"
    echo "📝 View logs: kubectl logs -n ai-agents -l category=<category>"
}

main "$@"
