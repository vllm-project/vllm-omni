# Basic vLLM-omni Examples

This directory contains simple examples demonstrating how to use vLLM-omni for text generation.

## Examples

### 1. `simple_usage.py` - Direct Library Usage

Demonstrates how to use vLLM-omni directly in your Python code:

```bash
# Run the example
python simple_usage.py
```

**Features:**
- Synchronous text generation
- Asynchronous text generation  
- Multi-prompt processing
- Shows both text output and token information

### 2. `api_client.py` - API Server Usage

Demonstrates how to interact with the vLLM-omni API server:

```bash
# First, start the server
vllm serve Qwen/Qwen3-0.6B --omni --port 8000

# Then run the client (in another terminal)
python api_client.py
```

**Features:**
- Health check
- Text generation via HTTP API
- Multiple prompt types
- Full response structure inspection

## Quick Start

1. **Install vLLM-omni:**
   ```bash
   pip install -e .
   ```

2. **Run direct usage example:**
   ```bash
   cd examples/basic
   python simple_usage.py
   ```

3. **Run API server example:**
   ```bash
   # Terminal 1: Start server
   vllm serve Qwen/Qwen3-0.6B --omni --port 8000
   
   # Terminal 2: Run client
   cd examples/basic
   python api_client.py
   ```

## Expected Output

The examples will show:
- ✅ Generated text responses
- ✅ Token counts and completion status
- ✅ Stage processing information
- ✅ Error handling

## Requirements

- Python 3.8+
- vLLM-omni installed
- For API examples: HTTP server running

