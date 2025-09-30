#!/usr/bin/env python3
"""
Simple API client example for vLLM-omni server.

This example shows how to interact with the vLLM-omni API server
using HTTP requests.
"""

import requests
import json
import time


def test_health(host="localhost", port=8000):
    """Test if the server is healthy."""
    url = f"http://{host}:{port}/health"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print("✅ Server is healthy")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        return False


def generate_text(prompt, host="localhost", port=8000, **kwargs):
    """Generate text using the API server."""
    url = f"http://{host}:{port}/generate"
    
    # Default parameters
    params = {
        "prompts": [prompt],
        "max_tokens": kwargs.get("max_tokens", 100),
        "temperature": kwargs.get("temperature", 0.7),
        "top_p": kwargs.get("top_p", 1.0),
        "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
        "presence_penalty": kwargs.get("presence_penalty", 0.0),
    }
    
    # Add optional parameters
    if "stop" in kwargs:
        params["stop"] = kwargs["stop"]
    
    try:
        response = requests.post(url, json=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return None


def main():
    """Main example function."""
    print("vLLM-omni API Client Example")
    print("=" * 40)
    
    # Test server health
    if not test_health():
        print("Please start the server first:")
        print("  vllm serve Qwen/Qwen3-0.6B --omni --port 8000")
        return
    
    # Example 1: Simple generation
    print("\n1. Simple text generation:")
    result = generate_text("Hello, how are you?", max_tokens=50)
    if result:
        text = result["outputs"][0]["text"]
        print(f"   Input: Hello, how are you?")
        print(f"   Output: {text}")
    
    # Example 2: Creative writing
    print("\n2. Creative writing:")
    result = generate_text(
        "Write a short story about a robot learning to paint",
        max_tokens=150,
        temperature=0.9
    )
    if result:
        text = result["outputs"][0]["text"]
        print(f"   Input: Write a short story about a robot learning to paint")
        print(f"   Output: {text}")
    
    # Example 3: Question answering
    print("\n3. Question answering:")
    result = generate_text(
        "What is the capital of France?",
        max_tokens=30,
        temperature=0.3
    )
    if result:
        text = result["outputs"][0]["text"]
        print(f"   Input: What is the capital of France?")
        print(f"   Output: {text}")
    
    # Example 4: Multiple prompts
    print("\n4. Multiple prompts:")
    prompts = [
        "Tell me a joke",
        "What's 2+2?",
        "Describe the color blue"
    ]
    
    for prompt in prompts:
        result = generate_text(prompt, max_tokens=30)
        if result:
            text = result["outputs"][0]["text"]
            print(f"   Q: {prompt}")
            print(f"   A: {text}")
    
    # Example 5: Show full response structure
    print("\n5. Full response structure:")
    result = generate_text("Hello world", max_tokens=20)
    if result:
        print("   Full API response:")
        print(json.dumps(result, indent=2))
    
    print("\n" + "=" * 40)
    print("API client examples completed!")


if __name__ == "__main__":
    main()

