#!/bin/bash

# Quick vLLM-omni Test Script
# Fast test for basic functionality after changes
# Usage: ./scripts/quick_test.sh [port]

set -e

PORT=${1:-8000}
HOST="localhost"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Cleanup
cleanup() {
    log_info "Cleaning up..."
    pkill -f "vllm serve" 2>/dev/null || true
}

trap cleanup EXIT

# Quick test
main() {
    log_info "Quick vLLM-omni test on port $PORT"
    
    # Test imports
    log_info "Testing imports..."
    python -c "import vllm_omni; from vllm_omni.entrypoints.omni_llm import OmniLLM, AsyncOmniLLM; print('✅ Imports OK')" || {
        log_error "Import failed"
        exit 1
    }
    
    # Start server
    log_info "Starting server..."
    vllm serve ./models/Qwen3-0.6B --omni --port $PORT > /dev/null 2>&1 &
    SERVER_PID=$!
    
    # Wait for server
    sleep 15
    
    # Test health with retry
    log_info "Testing health endpoint..."
    for i in {1..5}; do
        if curl -s http://$HOST:$PORT/health | grep -q "healthy"; then
            log_success "Health check passed"
            break
        else
            if [ $i -eq 5 ]; then
                log_error "Health check failed after 5 attempts"
                exit 1
            fi
            log_info "Health check attempt $i failed, retrying in 3 seconds..."
            sleep 3
        fi
    done
    
    # Test generation
    log_info "Testing text generation..."
    response=$(curl -s -X POST http://$HOST:$PORT/generate \
        -H "Content-Type: application/json" \
        -d '{"prompts": ["Quick test"], "max_tokens": 10, "temperature": 0.7}')
    
    if echo "$response" | python -c "import sys, json; data=json.load(sys.stdin); exit(0 if 'outputs' in data else 1)" 2>/dev/null; then
        log_success "Generation test passed"
    else
        log_error "Generation test failed"
        exit 1
    fi
    
    log_success "Quick test completed successfully!"
}

cd "$(dirname "$0")/.."
main "$@"
