#!/bin/bash

# orchestration-reporter.sh - Real-time orchestration monitor

LOG_FILE="./logs/orchestration-status.log"

echo "📡 Monitoring task orchestrator logs..."

if [ ! -f "$LOG_FILE" ]; then
  echo "📝 Log file not found. Creating new one..."
  mkdir -p ./logs
  touch "$LOG_FILE"
fi

tail -n 20 -f "$LOG_FILE" | while read line; do
  echo "📌 $(date +"%H:%M:%S") | $line"
done
