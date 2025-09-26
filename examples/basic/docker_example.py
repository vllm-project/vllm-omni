#!/usr/bin/env python3
"""
Docker example for vLLM-omni.

This example shows how to test the vLLM-omni API server running in Docker.
"""

import requests
import json
import time
import subprocess
import sys


def check_docker_running():
    """Check if Docker is running."""
    try:
        result = subprocess.run(['docker', 'info'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def start_docker_server():
    """Start the vLLM-omni server in Docker."""
    print("🐳 Starting vLLM-omni server in Docker...")
    
    # Check if container is already running
    result = subprocess.run(['docker', 'ps', '--filter', 'name=vllm-omni-cpu', '--format', '{{.Names}}'],
                          capture_output=True, text=True)
    
    if 'vllm-omni-cpu' in result.stdout:
        print("✅ Server already running")
        return True
    
    # Start the server
    cmd = [
        'docker', 'run', '-d', '--name', 'vllm-omni-cpu',
        '-p', '8000:8000',
        'vllm-omni-cpu:latest',
        'vllm', 'serve', 'Qwen/Qwen3-0.6B', '--omni', '--port', '8000'
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Server started successfully")
        
        # Wait for server to be ready
        print("⏳ Waiting for server to be ready...")
        for i in range(30):  # Wait up to 30 seconds
            if test_server_health():
                print("✅ Server is ready!")
                return True
            time.sleep(1)
        
        print("❌ Server failed to start within 30 seconds")
        return False
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start server: {e}")
        return False


def test_server_health():
    """Test if the server is healthy."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def test_generation():
    """Test text generation via API."""
    print("\n🧪 Testing text generation...")
    
    test_cases = [
        {
            "prompt": "Hello, how are you?",
            "max_tokens": 50,
            "temperature": 0.7
        },
        {
            "prompt": "Tell me a joke",
            "max_tokens": 100,
            "temperature": 0.9
        },
        {
            "prompt": "What is artificial intelligence?",
            "max_tokens": 150,
            "temperature": 0.5
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['prompt']}")
        
        try:
            response = requests.post(
                "http://localhost:8000/generate",
                json={
                    "prompts": [test_case["prompt"]],
                    "max_tokens": test_case["max_tokens"],
                    "temperature": test_case["temperature"]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result["outputs"][0]["text"]
                print(f"✅ Response: {text}")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")


def cleanup():
    """Clean up Docker container."""
    print("\n🧹 Cleaning up...")
    try:
        subprocess.run(['docker', 'stop', 'vllm-omni-cpu'], 
                      capture_output=True, text=True)
        subprocess.run(['docker', 'rm', 'vllm-omni-cpu'], 
                      capture_output=True, text=True)
        print("✅ Cleanup completed")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")


def main():
    """Main example function."""
    print("🐳 vLLM-omni Docker Example")
    print("=" * 40)
    
    # Check Docker
    if not check_docker_running():
        print("❌ Docker is not running. Please start Docker Desktop.")
        sys.exit(1)
    
    print("✅ Docker is running")
    
    # Check if image exists
    result = subprocess.run(['docker', 'images', 'vllm-omni-cpu:latest'], 
                          capture_output=True, text=True)
    if 'vllm-omni-cpu' not in result.stdout:
        print("❌ Docker image not found. Please build it first:")
        print("   docker build -f docker/Dockerfile.cpu -t vllm-omni-cpu .")
        sys.exit(1)
    
    print("✅ Docker image found")
    
    try:
        # Start server
        if not start_docker_server():
            sys.exit(1)
        
        # Test generation
        test_generation()
        
        print("\n🎉 Docker example completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Example interrupted by user")
    finally:
        # Ask if user wants to keep server running
        keep_running = input("\n🤔 Keep the server running? (y/N): ").lower().strip()
        if keep_running != 'y':
            cleanup()
        else:
            print("✅ Server kept running. Access it at http://localhost:8000")


if __name__ == "__main__":
    main()
