# vLLM-omni Examples

This directory contains examples demonstrating how to use vLLM-omni for multi-modality model inference and serving.

## Directory Structure

- `basic/` - Simple usage examples and API client
- `advanced/` - Advanced features and configurations  
- `multimodal/` - Multi-modality specific examples

## Quick Start

1. **Basic Usage:**
   ```bash
   cd examples/basic
   python simple_usage.py
   ```

2. **API Server:**
   ```bash
   # Start server
   vllm serve Qwen/Qwen3-0.6B --omni --port 8000
   
   # Test with client
   python api_client.py
   ```

## Examples Overview

### Basic Examples (`basic/`)
- `simple_usage.py` - Direct library usage (sync/async)
- `api_client.py` - HTTP API client examples
- `README.md` - Basic examples documentation

### Advanced Examples (`advanced/`)
- Multi-stage processing
- Custom configurations
- Performance optimization

### Multi-modality Examples (`multimodal/`)
- Text-to-image generation
- Image-to-text processing
- Multi-modal pipelines

## Requirements

- Python 3.8+
- vLLM-omni installed
- For API examples: HTTP server running