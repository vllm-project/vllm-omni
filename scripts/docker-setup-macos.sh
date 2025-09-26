#!/bin/bash

# vLLM-omni Docker Setup Script for macOS
# This script helps set up Docker for vLLM-omni on macOS

set -e

echo "🐳 vLLM-omni Docker Setup for macOS"
echo "=================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker Desktop for Mac:"
    echo "   https://www.docker.com/products/docker-desktop/"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

echo "✅ Docker is installed and running"

# Check if NVIDIA Docker runtime is available (optional for macOS)
if docker info | grep -q nvidia; then
    echo "✅ NVIDIA Docker runtime detected"
    GPU_SUPPORT=true
else
    echo "⚠️  NVIDIA Docker runtime not detected (normal on macOS)"
    GPU_SUPPORT=false
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models logs notebooks

# Build the Docker image
echo "🔨 Building vLLM-omni Docker image..."
docker build -f docker/Dockerfile.cpu -t vllm-omni-cpu:latest .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Failed to build Docker image"
    exit 1
fi

# Test the image
echo "🧪 Testing Docker image..."
docker run --rm vllm-omni-cpu:latest python -c "import vllm_omni; print('vLLM-omni imported successfully')"

if [ $? -eq 0 ]; then
    echo "✅ Docker image test passed"
else
    echo "❌ Docker image test failed"
    exit 1
fi

echo ""
echo "🎉 Setup complete! You can now use:"
echo ""
echo "  # Start the server:"
echo "  docker-compose -f docker/docker-compose.cpu.yml up vllm-omni-cpu"
echo ""
echo "  # Start with Jupyter:"
echo "  docker-compose -f docker/docker-compose.cpu.yml --profile jupyter up"
echo ""
echo "  # Run a single command:"
echo "  docker run --rm -p 8000:8000 vllm-omni-cpu:latest vllm serve Qwen/Qwen3-0.6B --omni --port 8000"
echo ""
echo "  # Interactive shell:"
echo "  docker run -it --rm vllm-omni-cpu:latest bash"
echo ""

# Optional: Start the server
read -p "🚀 Start the vLLM-omni server now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting vLLM-omni server..."
    docker-compose -f docker/docker-compose.cpu.yml up vllm-omni-cpu
fi
