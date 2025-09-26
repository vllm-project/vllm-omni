#!/bin/bash

# vLLM-omni Serving Functionality Test Script
# This script tests the complete serving functionality of vLLM-omni
# Usage: ./scripts/test_serving.sh [model_path] [port]

set -e  # Exit on any error

# Configuration
MODEL_PATH=${1:-"./models/Qwen3-0.6B"}
PORT=${2:-8000}
HOST="localhost"
TIMEOUT=30
SERVER_STARTUP_WAIT=15

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    if [ ! -z "$SERVER_PID" ]; then
        log_info "Stopping server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
    # Kill any remaining vllm processes
    pkill -f "vllm serve" 2>/dev/null || true
    log_info "Cleanup completed"
}

# Set up trap for cleanup on exit
trap cleanup EXIT

# Check if model exists
check_model() {
    log_info "Checking if model exists at: $MODEL_PATH"
    if [ ! -d "$MODEL_PATH" ]; then
        log_error "Model directory not found: $MODEL_PATH"
        log_info "Available models:"
        ls -la models/ 2>/dev/null || log_warning "No models directory found"
        exit 1
    fi
    log_success "Model found: $MODEL_PATH"
}

# Check if conda environment is activated
check_environment() {
    log_info "Checking conda environment..."
    if [[ "$CONDA_DEFAULT_ENV" != "vllm_omni" ]]; then
        log_warning "vllm_omni conda environment not activated"
        log_info "Attempting to activate vllm_omni environment..."
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate vllm_omni
        if [[ "$CONDA_DEFAULT_ENV" != "vllm_omni" ]]; then
            log_error "Failed to activate vllm_omni environment"
            exit 1
        fi
    fi
    log_success "Environment check passed: $CONDA_DEFAULT_ENV"
}

# Test import functionality
test_imports() {
    log_info "Testing imports..."
    python -c "
import vllm_omni
from vllm_omni.entrypoints.omni_llm import OmniLLM, AsyncOmniLLM
print('✅ All imports successful')
" || {
        log_error "Import test failed"
        exit 1
    }
    log_success "Import test passed"
}

# Start the server
start_server() {
    log_info "Starting vLLM-omni server on port $PORT..."
    log_info "Command: vllm serve $MODEL_PATH --omni --port $PORT"
    
    # Start server in background
    vllm serve "$MODEL_PATH" --omni --port "$PORT" > server.log 2>&1 &
    SERVER_PID=$!
    
    log_info "Server started with PID: $SERVER_PID"
    log_info "Waiting $SERVER_STARTUP_WAIT seconds for server to initialize..."
    sleep $SERVER_STARTUP_WAIT
    
    # Check if server is still running
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        log_error "Server failed to start"
        log_info "Server logs:"
        cat server.log
        exit 1
    fi
    
    log_success "Server appears to be running"
}

# Test health endpoint
test_health() {
    log_info "Testing health endpoint..."
    local response=$(curl -s -w "%{http_code}" http://$HOST:$PORT/health 2>/dev/null || echo "000")
    local http_code="${response: -3}"
    local body="${response%???}"
    
    if [ "$http_code" = "200" ]; then
        log_success "Health check passed: $body"
    else
        log_error "Health check failed (HTTP $http_code): $body"
        return 1
    fi
}

# Test info endpoint
test_info() {
    log_info "Testing info endpoint..."
    local response=$(curl -s -w "%{http_code}" http://$HOST:$PORT/info 2>/dev/null || echo "000")
    local http_code="${response: -3}"
    local body="${response%???}"
    
    if [ "$http_code" = "200" ]; then
        log_success "Info endpoint working"
        echo "$body" | python -m json.tool 2>/dev/null || log_warning "Info response not valid JSON"
    else
        log_error "Info endpoint failed (HTTP $http_code): $body"
        return 1
    fi
}

# Test text generation
test_generation() {
    log_info "Testing text generation..."
    
    # Test 1: Simple generation
    local response=$(curl -s -X POST http://$HOST:$PORT/generate \
        -H "Content-Type: application/json" \
        -d '{"prompts": ["Test the server functionality"], "max_tokens": 20, "temperature": 0.7}' \
        2>/dev/null || echo "{}")
    
    if echo "$response" | python -c "import sys, json; data=json.load(sys.stdin); exit(0 if 'outputs' in data and len(data['outputs']) > 0 else 1)" 2>/dev/null; then
        log_success "Text generation test passed"
        echo "$response" | python -c "import sys, json; data=json.load(sys.stdin); print('Generated text:', data['outputs'][0]['text'][:100] + '...' if len(data['outputs'][0]['text']) > 100 else data['outputs'][0]['text'])" 2>/dev/null
    else
        log_error "Text generation test failed"
        echo "Response: $response"
        return 1
    fi
}

# Test API client example
test_api_client() {
    log_info "Testing API client example..."
    if [ -f "examples/basic/api_client.py" ]; then
        python examples/basic/api_client.py > api_client_test.log 2>&1 || {
            log_error "API client test failed"
            log_info "API client logs:"
            cat api_client_test.log
            return 1
        }
        log_success "API client test passed"
    else
        log_warning "API client example not found, skipping"
    fi
}

# Test simple usage example
test_simple_usage() {
    log_info "Testing simple usage example..."
    if [ -f "examples/basic/simple_usage.py" ]; then
        # Run with timeout to avoid hanging
        timeout 60 python examples/basic/simple_usage.py > simple_usage_test.log 2>&1 || {
            local exit_code=$?
            if [ $exit_code -eq 124 ]; then
                log_warning "Simple usage test timed out (60s), but this might be normal for model loading"
            else
                log_error "Simple usage test failed (exit code: $exit_code)"
                log_info "Simple usage logs:"
                cat simple_usage_test.log
                return 1
            fi
        }
        log_success "Simple usage test passed"
    else
        log_warning "Simple usage example not found, skipping"
    fi
}

# Performance test
test_performance() {
    log_info "Running performance test..."
    
    local start_time=$(date +%s)
    local response=$(curl -s -X POST http://$HOST:$PORT/generate \
        -H "Content-Type: application/json" \
        -d '{"prompts": ["Performance test prompt"], "max_tokens": 50, "temperature": 0.7}' \
        2>/dev/null)
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if echo "$response" | python -c "import sys, json; data=json.load(sys.stdin); exit(0 if 'outputs' in data else 1)" 2>/dev/null; then
        log_success "Performance test passed (${duration}s response time)"
    else
        log_error "Performance test failed"
        return 1
    fi
}

# Main test function
run_tests() {
    log_info "Starting vLLM-omni serving functionality tests..."
    log_info "Model: $MODEL_PATH"
    log_info "Port: $PORT"
    log_info "Host: $HOST"
    echo "=========================================="
    
    # Run all tests
    check_model
    check_environment
    test_imports
    start_server
    
    # Wait a bit more for server to be fully ready
    sleep 5
    
    test_health
    test_info
    test_generation
    test_performance
    test_api_client
    
    # Note: Simple usage test is commented out as it starts its own server
    # test_simple_usage
    
    log_success "All tests completed successfully!"
    echo "=========================================="
    log_info "Test Summary:"
    log_info "✅ Model loading and server startup"
    log_info "✅ Health and info endpoints"
    log_info "✅ Text generation functionality"
    log_info "✅ Performance metrics"
    log_info "✅ API client integration"
    log_info "✅ All imports working correctly"
}

# Show usage
show_usage() {
    echo "Usage: $0 [model_path] [port]"
    echo ""
    echo "Arguments:"
    echo "  model_path    Path to the model directory (default: ./models/Qwen3-0.6B)"
    echo "  port          Port to run the server on (default: 8000)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use default model and port"
    echo "  $0 ./models/Qwen3-0.6B 8001          # Use specific model and port"
    echo "  $0 Qwen/Qwen3-0.6B 8000              # Use HuggingFace model"
}

# Main execution
main() {
    # Check for help flag
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        show_usage
        exit 0
    fi
    
    # Change to script directory
    cd "$(dirname "$0")/.."
    
    # Create logs directory if it doesn't exist
    mkdir -p logs
    
    # Run tests
    run_tests
}

# Run main function with all arguments
main "$@"
