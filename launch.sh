#!/bin/bash

# launch.sh - Project Bootstrapper for Claude Code
# Author: Claude Orchestration Framework
# Description: Automates context initialization, hallucination validation, and project orchestration.

WORKFLOWS_DIR="./.claude/workflows"
AGENTS_DIR="./.claude/agents"

echo "🧠 Launching Claude Code Orchestration..."

# Step 1: Validate Required Files
REQUIRED_FILES=(
  "$WORKFLOWS_DIR/project-startup.yaml"
  "$AGENTS_DIR/context-manager.yaml"
  "$AGENTS_DIR/hallucination-trace-agent.yaml"
)

echo "🔍 Validating required files..."
for file in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "$file" ]; then
    echo "❌ Missing required file: $file"
    exit 1
  fi
done
echo "✅ All required files found."

# Step 2: Trigger hallucination validation
echo "🧪 Running hallucination-trace-agent for ./docs..."
claude agent run "$AGENTS_DIR/hallucination-trace-agent.yaml" --inputs "folder=./docs" "threshold=0.02"
if [ $? -ne 0 ]; then
  echo "❗ Hallucination trace failed. Please resolve issues before continuing."
  exit 2
fi
echo "✅ Hallucination scan passed."

# Step 3: Execute context manager orchestration
echo "🧬 Executing project-startup workflow..."
claude plan run "$WORKFLOWS_DIR/project-startup.yaml"

# Step 4: Start monitoring logs
echo "📈 Launching orchestration log reporter..."
bash .claude/tools/orchestration-reporter.sh

echo "🚀 Project orchestration complete."
