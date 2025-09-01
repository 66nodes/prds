#!/bin/bash
# Agent System Health Check

set -e

NAMESPACE="ai-agents"
FAILURES=0

echo "🏥 Running Agent System Health Checks..."

# Check namespace exists
check_namespace() {
    if kubectl get namespace $NAMESPACE >/dev/null 2>&1; then
        echo "✅ Namespace exists"
    else
        echo "❌ Namespace missing"
        ((FAILURES++))
    fi
}

# Check agent deployments
check_deployments() {
    echo "Checking deployments..."
    
    EXPECTED_DEPLOYMENTS=(
        "development-agents"
        "infrastructure-agents"
        "ai-ml-agents"
    )
    
    for deployment in "${EXPECTED_DEPLOYMENTS[@]}"; do
        if kubectl get deployment $deployment -n $NAMESPACE >/dev/null 2>&1; then
            READY=$(kubectl get deployment $deployment -n $NAMESPACE -o jsonpath='{.status.readyReplicas}')
            DESIRED=$(kubectl get deployment $deployment -n $NAMESPACE -o jsonpath='{.spec.replicas}')
            if [ "$READY" == "$DESIRED" ]; then
                echo "✅ $deployment: $READY/$DESIRED replicas ready"
            else
                echo "⚠️ $deployment: $READY/$DESIRED replicas ready"
                ((FAILURES++))
            fi
        else
            echo "❌ $deployment: not found"
            ((FAILURES++))
        fi
    done
}

# Check services
check_services() {
    echo "Checking services..."
    
    SERVICES=$(kubectl get services -n $NAMESPACE -o name | wc -l)
    if [ "$SERVICES" -gt 0 ]; then
        echo "✅ $SERVICES services running"
    else
        echo "❌ No services found"
        ((FAILURES++))
    fi
}

# Check GraphRAG connection
check_graphrag() {
    echo "Checking GraphRAG connection..."
    
    # This would need actual endpoint testing
    echo "⚠️ GraphRAG check requires manual verification"
}

# Check resource usage
check_resources() {
    echo "Checking resource usage..."
    
    kubectl top nodes
    kubectl top pods -n $NAMESPACE | head -20
}

# Generate report
generate_report() {
    echo ""
    echo "════════════════════════════════════════"
    echo "Health Check Summary"
    echo "════════════════════════════════════════"
    
    if [ $FAILURES -eq 0 ]; then
        echo "✅ All checks passed!"
    else
        echo "⚠️ $FAILURES checks failed"
        echo "Run 'kubectl describe pod -n $NAMESPACE' for details"
    fi
    
    echo ""
    echo "Quick Commands:"
    echo "- View logs: kubectl logs -n $NAMESPACE <pod-name>"
    echo "- Get events: kubectl get events -n $NAMESPACE"
    echo "- Port forward: kubectl port-forward -n $NAMESPACE svc/grafana 3000:80"
}

# Main execution
main() {
    check_namespace
    check_deployments
    check_services
    check_graphrag
    check_resources
    generate_report
}

main "$@"
