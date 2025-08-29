#!/bin/bash
# Comprehensive error handling wrapper for GitHub Actions
# Usage: source this script at the beginning of your workflow steps

set -eEuo pipefail

# Color codes for output
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly GREEN='\033[0;32m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Global variables
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_FILE="${GITHUB_WORKSPACE:-/tmp}/error_context.log"
readonly ERROR_REPORT="${GITHUB_WORKSPACE:-/tmp}/error_report.md"

# Initialize logging
init_logging() {
    echo "$(date -u): Error handler initialized" > "$LOG_FILE"
    echo "Script: ${BASH_SOURCE[1]:-unknown}" >> "$LOG_FILE"
    echo "Working directory: $(pwd)" >> "$LOG_FILE"
    echo "User: $(whoami)" >> "$LOG_FILE"
    echo "Environment: ${GITHUB_ACTIONS:-local}" >> "$LOG_FILE"
    echo "---" >> "$LOG_FILE"
}

# Enhanced error handling function
handle_error() {
    local exit_code=$?
    local line_number=$1
    local command="$2"
    local function_name="${FUNCNAME[2]:-main}"
    local script_name="${BASH_SOURCE[2]:-unknown}"
    
    # Log error details
    {
        echo "ERROR OCCURRED:"
        echo "Timestamp: $(date -u)"
        echo "Script: $script_name"
        echo "Function: $function_name"
        echo "Line: $line_number"
        echo "Command: $command"
        echo "Exit Code: $exit_code"
        echo "Working Directory: $(pwd)"
        echo "---"
    } >> "$LOG_FILE"
    
    # Print to console with colors
    echo -e "${RED}❌ ERROR occurred at line $line_number: $command${NC}" >&2
    echo -e "${RED}Exit code: $exit_code${NC}" >&2
    echo -e "${BLUE}Function: $function_name in $script_name${NC}" >&2
    
    # Add context information
    collect_error_context "$exit_code" "$line_number" "$command"
    
    # Generate error report for GitHub
    generate_error_report "$exit_code" "$line_number" "$command" "$function_name" "$script_name"
    
    # Try to provide helpful suggestions
    suggest_fix "$exit_code" "$command"
    
    exit $exit_code
}

# Collect additional context about the error
collect_error_context() {
    local exit_code=$1
    local line_number=$2
    local command="$3"
    
    {
        echo "CONTEXT INFORMATION:"
        echo "Environment Variables:"
        env | grep -E '^(GITHUB_|CI_|NODE_|PYTHON_|HOME|PATH)' | head -20
        echo ""
        
        echo "System Information:"
        uname -a 2>/dev/null || echo "uname not available"
        
        echo "Disk Space:"
        df -h 2>/dev/null || echo "df not available"
        
        echo "Memory Usage:"
        free -h 2>/dev/null || echo "free not available"
        
        echo "Process List (top 10):"
        ps aux 2>/dev/null | head -10 || echo "ps not available"
        
        echo "Recent Log Entries:"
        tail -10 "$LOG_FILE" 2>/dev/null || echo "No recent logs"
        
        echo "---"
    } >> "$LOG_FILE"
}

# Generate structured error report for GitHub Actions
generate_error_report() {
    local exit_code=$1
    local line_number=$2
    local command="$3"
    local function_name="$4"
    local script_name="$5"
    
    cat > "$ERROR_REPORT" << EOF
# ❌ Workflow Error Report

## Error Summary
- **Timestamp**: $(date -u)
- **Script**: $(basename "$script_name")
- **Function**: $function_name
- **Line Number**: $line_number
- **Exit Code**: $exit_code
- **Failed Command**: \`$command\`

## Environment
- **Runner**: \${RUNNER_OS:-unknown}
- **Workflow**: \${GITHUB_WORKFLOW:-unknown}
- **Job**: \${GITHUB_JOB:-unknown}
- **Step**: \${GITHUB_ACTION:-unknown}

## Context
\`\`\`bash
$command
\`\`\`

## Suggested Actions
$(suggest_fix_markdown "$exit_code" "$command")

## Debug Information
<details>
<summary>Click to expand debug details</summary>

\`\`\`
$(tail -20 "$LOG_FILE" 2>/dev/null || echo "No debug information available")
\`\`\`
</details>

---
*This report was generated automatically by the error handler*
EOF

    # Add to GitHub step summary if in GitHub Actions
    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
        cat "$ERROR_REPORT" >> "$GITHUB_STEP_SUMMARY"
    fi
}

# Provide intelligent fix suggestions based on exit codes and commands
suggest_fix() {
    local exit_code=$1
    local command="$2"
    
    case $exit_code in
        1)
            echo -e "${YELLOW}💡 Suggestion: General error - check command syntax and arguments${NC}" >&2
            ;;
        2)
            echo -e "${YELLOW}💡 Suggestion: Command not found or misuse of shell builtins${NC}" >&2
            if [[ "$command" =~ ^[a-zA-Z] ]]; then
                echo -e "${YELLOW}   Try: which ${command%% *}${NC}" >&2
            fi
            ;;
        126)
            echo -e "${YELLOW}💡 Suggestion: Permission denied - check file permissions${NC}" >&2
            echo -e "${YELLOW}   Try: chmod +x script_name${NC}" >&2
            ;;
        127)
            echo -e "${YELLOW}💡 Suggestion: Command not found${NC}" >&2
            echo -e "${YELLOW}   Check if the command is installed and in PATH${NC}" >&2
            ;;
        128)
            echo -e "${YELLOW}💡 Suggestion: Invalid exit argument${NC}" >&2
            ;;
        130)
            echo -e "${YELLOW}💡 Suggestion: Script terminated by Ctrl+C${NC}" >&2
            ;;
        *)
            if [[ "$command" =~ npm|yarn ]]; then
                echo -e "${YELLOW}💡 Suggestion: Node.js issue - check node_modules and package.json${NC}" >&2
            elif [[ "$command" =~ pip|python ]]; then
                echo -e "${YELLOW}💡 Suggestion: Python issue - check virtual environment and requirements${NC}" >&2
            elif [[ "$command" =~ docker ]]; then
                echo -e "${YELLOW}💡 Suggestion: Docker issue - check Docker daemon and permissions${NC}" >&2
            elif [[ "$command" =~ git ]]; then
                echo -e "${YELLOW}💡 Suggestion: Git issue - check repository state and permissions${NC}" >&2
            else
                echo -e "${YELLOW}💡 Suggestion: Check logs above for specific error details${NC}" >&2
            fi
            ;;
    esac
}

# Markdown version of fix suggestions
suggest_fix_markdown() {
    local exit_code=$1
    local command="$2"
    
    case $exit_code in
        1)
            echo "- Check command syntax and arguments"
            echo "- Verify all required parameters are provided"
            ;;
        2)
            echo "- Verify the command exists and is in PATH"
            echo "- Check for typos in the command name"
            if [[ "$command" =~ ^[a-zA-Z] ]]; then
                echo "- Run: \`which ${command%% *}\` to check availability"
            fi
            ;;
        126)
            echo "- Check file permissions: \`ls -la script_name\`"
            echo "- Make executable: \`chmod +x script_name\`"
            ;;
        127)
            echo "- Install the missing command"
            echo "- Add to PATH if installed in non-standard location"
            ;;
        *)
            if [[ "$command" =~ npm|yarn ]]; then
                echo "- Run \`npm install\` or \`yarn install\`"
                echo "- Check Node.js version compatibility"
                echo "- Clear cache: \`npm cache clean --force\`"
            elif [[ "$command" =~ pip|python ]]; then
                echo "- Activate virtual environment"
                echo "- Install requirements: \`pip install -r requirements.txt\`"
                echo "- Check Python version compatibility"
            elif [[ "$command" =~ docker ]]; then
                echo "- Check if Docker daemon is running"
                echo "- Verify Docker permissions for current user"
                echo "- Check available disk space"
            elif [[ "$command" =~ git ]]; then
                echo "- Check git configuration: \`git config --list\`"
                echo "- Verify repository permissions"
                echo "- Check if branch exists"
            else
                echo "- Review error logs for specific details"
                echo "- Check system resources (disk space, memory)"
                echo "- Verify all dependencies are installed"
            fi
            ;;
    esac
}

# Graceful cleanup function
cleanup_on_exit() {
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ Script completed successfully${NC}"
    else
        echo -e "${RED}❌ Script failed with exit code $exit_code${NC}" >&2
    fi
    
    # Archive logs if in GitHub Actions
    if [ -n "${GITHUB_ACTIONS:-}" ] && [ -f "$LOG_FILE" ]; then
        echo "Archiving error logs..."
        cp "$LOG_FILE" "${GITHUB_WORKSPACE}/error_logs_$(date +%Y%m%d_%H%M%S).log" 2>/dev/null || true
    fi
}

# Timeout handler
timeout_handler() {
    echo -e "${YELLOW}⏰ Operation timed out${NC}" >&2
    echo "TIMEOUT: $(date -u)" >> "$LOG_FILE"
    exit 124
}

# Network error handler
check_network() {
    if ! ping -c 1 8.8.8.8 &>/dev/null; then
        echo -e "${YELLOW}⚠️ Network connectivity issues detected${NC}" >&2
        echo "Network check failed: $(date -u)" >> "$LOG_FILE"
        return 1
    fi
}

# Disk space checker
check_disk_space() {
    local min_space_mb=${1:-1000}  # Default 1GB minimum
    local available_mb
    
    available_mb=$(df . | awk 'NR==2 {print int($4/1024)}')
    
    if [ "$available_mb" -lt "$min_space_mb" ]; then
        echo -e "${YELLOW}⚠️ Low disk space: ${available_mb}MB available${NC}" >&2
        echo "Low disk space warning: ${available_mb}MB available" >> "$LOG_FILE"
        return 1
    fi
}

# Memory checker
check_memory() {
    if command -v free >/dev/null; then
        local mem_usage
        mem_usage=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
        
        if [ "$mem_usage" -gt 90 ]; then
            echo -e "${YELLOW}⚠️ High memory usage: ${mem_usage}%${NC}" >&2
            echo "High memory usage warning: ${mem_usage}%" >> "$LOG_FILE"
        fi
    fi
}

# Retry mechanism with exponential backoff
retry_with_backoff() {
    local max_attempts=${1:-3}
    local delay=${2:-1}
    local command_to_retry="$3"
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo -e "${BLUE}🔄 Attempt $attempt of $max_attempts: $command_to_retry${NC}"
        
        if eval "$command_to_retry"; then
            echo -e "${GREEN}✅ Command succeeded on attempt $attempt${NC}"
            return 0
        else
            local exit_code=$?
            echo -e "${YELLOW}⚠️ Command failed on attempt $attempt (exit code: $exit_code)${NC}" >&2
            
            if [ $attempt -lt $max_attempts ]; then
                echo -e "${BLUE}⏳ Waiting ${delay}s before retry...${NC}"
                sleep $delay
                delay=$((delay * 2))  # Exponential backoff
            fi
        fi
        
        ((attempt++))
    done
    
    echo -e "${RED}❌ Command failed after $max_attempts attempts${NC}" >&2
    return 1
}

# Safe command execution with logging
safe_run() {
    local command="$1"
    local description="${2:-Running command}"
    
    echo -e "${BLUE}▶️ $description${NC}"
    echo "$(date -u): $description - $command" >> "$LOG_FILE"
    
    if eval "$command"; then
        echo -e "${GREEN}✅ $description completed successfully${NC}"
        echo "$(date -u): SUCCESS - $description" >> "$LOG_FILE"
        return 0
    else
        local exit_code=$?
        echo -e "${RED}❌ $description failed (exit code: $exit_code)${NC}" >&2
        echo "$(date -u): FAILED - $description (exit code: $exit_code)" >> "$LOG_FILE"
        return $exit_code
    fi
}

# Initialize on script load
init_logging

# Set up trap for error handling
trap 'handle_error ${LINENO} "${BASH_COMMAND}"' ERR

# Set up cleanup trap
trap cleanup_on_exit EXIT

# Set up timeout trap
trap timeout_handler ALRM

# Export functions for use in other scripts
export -f handle_error collect_error_context suggest_fix cleanup_on_exit
export -f check_network check_disk_space check_memory retry_with_backoff safe_run

# Print initialization message
echo -e "${GREEN}✅ Error handler initialized${NC}"
echo -e "${BLUE}📝 Logs will be written to: $LOG_FILE${NC}"