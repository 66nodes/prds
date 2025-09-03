#!/bin/bash
# AI Agent Platform - Docker Swarm Secrets Creation Script

set -e

echo "🔐 Creating Docker Swarm secrets for AI Agent Platform"

# Check if swarm is initialized
if ! docker info --format '{{.Swarm.LocalNodeState}}' | grep -q "active"; then
    echo "❌ Docker Swarm is not initialized. Please run 'docker swarm init' first."
    exit 1
fi

# Function to generate secure random passwords
generate_password() {
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-32
}

# Function to generate JWT secret
generate_jwt_secret() {
    openssl rand -base64 64 | tr -d "=+/" | cut -c1-64
}

# Function to create or update a secret
create_secret() {
    local secret_name=$1
    local secret_value=$2
    
    if docker secret ls --format "{{.Name}}" | grep -q "^${secret_name}$"; then
        echo "⚠️  Secret '${secret_name}' already exists. Skipping..."
    else
        echo "${secret_value}" | docker secret create "${secret_name}" -
        echo "✅ Created secret: ${secret_name}"
    fi
}

# Function to create secret from environment variable or prompt
create_secret_from_env_or_prompt() {
    local secret_name=$1
    local env_var_name=$2
    local prompt_message=$3
    local secret_value=""
    
    # Check if environment variable exists
    if [ -n "${!env_var_name}" ]; then
        secret_value="${!env_var_name}"
        echo "📝 Using ${env_var_name} from environment"
    else
        echo -n "${prompt_message}: "
        read -s secret_value
        echo
    fi
    
    create_secret "${secret_name}" "${secret_value}"
}

echo "🎯 Creating application secrets..."

# Database passwords
echo "🗄️  Database credentials:"
create_secret "postgres_password" "$(generate_password)"
create_secret "redis_password" "$(generate_password)"
create_secret "neo4j_password" "$(generate_password)"

# Application secrets
echo "🔑 Application secrets:"
create_secret "jwt_secret" "$(generate_jwt_secret)"

# API Keys (these need to be provided)
echo "🤖 AI Service API Keys:"
create_secret_from_env_or_prompt "openai_api_key" "OPENAI_API_KEY" "Enter OpenAI API Key"

# Monitoring credentials
echo "📊 Monitoring credentials:"
create_secret "grafana_admin_password" "$(generate_password)"

# Email configuration (optional)
if [ -n "${SMTP_PASSWORD}" ]; then
    create_secret "smtp_password" "${SMTP_PASSWORD}"
    echo "✅ Created SMTP password secret"
fi

# AWS credentials for backups (optional)
if [ -n "${AWS_SECRET_ACCESS_KEY}" ]; then
    create_secret "aws_secret_access_key" "${AWS_SECRET_ACCESS_KEY}"
    echo "✅ Created AWS secret access key"
fi

echo ""
echo "🎉 Secrets creation completed!"
echo ""
echo "📋 Created secrets:"
docker secret ls --format "table {{.Name}}\t{{.CreatedAt}}"

echo ""
echo "⚠️  IMPORTANT SECURITY NOTES:"
echo "1. Store the generated passwords in a secure password manager"
echo "2. The secrets are stored encrypted in Docker Swarm"
echo "3. Only services with explicit access can read these secrets"
echo "4. Consider rotating these secrets periodically"

echo ""
echo "🚀 Next steps:"
echo "1. Update your .env.production file with the actual values"
echo "2. Deploy the stack: docker stack deploy -c docker-stack.yml aiplatform"