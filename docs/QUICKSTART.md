# Quick Start Guide

Get started with vLLM-omni in 5 minutes!

## Prerequisites

Before starting, ensure you have:
- Python 3.8+ (3.12 recommended)
- CUDA-compatible GPU
- 10+ GB free disk space

## Installation

### 1. Set up environment

```bash
# Create virtual environment
python -m venv vllm-omni-env
source vllm-omni-env/bin/activate  # On Windows: vllm-omni-env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 2. Install vLLM-omni

```bash
# Clone repository
git clone https://github.com/hsliuustc0106/vllm-omni.git
cd vllm-omni

# Install vLLM dependency (specific commit)
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 808a7b69df479b6b3a16181711cac7ca28a9b941
VLLM_USE_PRECOMPILED=1 pip install --editable .
cd ..

# Install vLLM-omni
pip install -e .
```

### 3. Verify installation

```bash
python -c "import vllm_omni; print(f'✓ vLLM-omni {vllm_omni.__version__} installed!')"
```

## Your First Model

### Option 1: Run Qwen2.5-omni Example

The fastest way to see vLLM-omni in action:

```bash
# Navigate to example directory
cd examples/offline_inference/qwen_2_5_omni

# Update PYTHONPATH in run.sh to point to your vllm-omni directory
# Then run the example
bash run.sh
```

This will:
- Load the Qwen2.5-omni model
- Process a sample input
- Generate audio output in `./output_audio/`

### Option 2: Simple Python Script

Create `my_first_model.py`:

```python
"""
Simple vLLM-omni example.
Note: This requires a compatible model to be available.
"""

# Basic import test
import vllm_omni

print(f"vLLM-omni version: {vllm_omni.__version__}")
print("vLLM-omni is ready!")

# Import main components
from vllm_omni import OmniLLM
from vllm_omni.config import create_ar_stage_config

print("Successfully imported OmniLLM!")

# Configuration example
config = create_ar_stage_config(
    model="facebook/opt-125m",  # Small test model
    max_model_len=512,
)

print(f"Created config: {config}")
print("\n✓ Setup complete! Ready for inference.")
```

Run it:
```bash
python my_first_model.py
```

## Understanding the Basics

### Core Concepts

1. **Multi-Engine Architecture**: vLLM-omni supports multiple processing engines
   - Autoregressive (AR) for text generation
   - Diffusion (DiT) for iterative generation
   - Hybrid for combined workflows

2. **Multi-Modal Support**: Process various data types
   - Text
   - Images
   - Audio
   - Combined inputs

3. **Stage-Based Processing**: Models can have multiple processing stages
   - Thinker: Initial reasoning
   - Talker: Response generation
   - Codec: Output encoding

### Basic Usage Pattern

```python
from vllm_omni import OmniLLM

# 1. Initialize model
llm = OmniLLM(
    model="path/to/model",
    # Optional configuration
)

# 2. Generate output
outputs = llm.generate(
    prompt="Your input here",
    # Additional parameters
)

# 3. Process results
for output in outputs:
    print(output.text)
```

## Common Tasks

### Task 1: Text Generation

```python
from vllm_omni import OmniLLM

llm = OmniLLM(model="path/to/text/model")
outputs = llm.generate("Hello, world!")

for output in outputs:
    print(f"Generated: {output.text}")
```

### Task 2: Multi-Modal Input

```python
from vllm_omni import OmniLLM
from PIL import Image

# Load model with multi-modal support
llm = OmniLLM(model="path/to/multimodal/model")

# Load image
image = Image.open("image.jpg")

# Generate with text and image
outputs = llm.generate(
    prompt="Describe this image",
    image=image
)

print(outputs[0].text)
```

### Task 3: Batch Processing

```python
from vllm_omni import OmniLLM

llm = OmniLLM(model="path/to/model")

# Multiple prompts
prompts = [
    "What is AI?",
    "Explain machine learning",
    "What are neural networks?"
]

# Generate for all prompts
outputs = llm.generate(prompts)

for i, output in enumerate(outputs):
    print(f"Prompt {i+1}: {output.text}")
```

## Configuration Tips

### Memory Management

```python
from vllm_omni import OmniLLM

llm = OmniLLM(
    model="path/to/model",
    gpu_memory_utilization=0.8,  # Use 80% of GPU memory
    max_model_len=2048,          # Limit context length
)
```

### Performance Tuning

```python
from vllm_omni import OmniLLM

llm = OmniLLM(
    model="path/to/model",
    tensor_parallel_size=2,      # Use 2 GPUs
    dtype="float16",             # Use FP16 for speed
)
```

## Next Steps

Now that you have vLLM-omni running:

1. **Explore Examples**
   - [Basic Examples](../examples/basic/) - Simple usage patterns
   - [Offline Inference](../examples/offline_inference/) - Batch processing
   - [Omni Examples](../examples/omni/) - Multi-modal models

2. **Read Documentation**
   - [Architecture](architecture/) - System design
   - [API Reference](api/) - Detailed API docs
   - [Installation Verification](INSTALLATION_VERIFICATION.md) - Troubleshooting

3. **Try Advanced Features**
   - Custom model integration
   - Multi-stage processing
   - Diffusion models
   - Custom sampling strategies

4. **Join the Community**
   - Star the [GitHub repo](https://github.com/hsliuustc0106/vllm-omni)
   - Report issues or ask questions
   - Contribute improvements

## Getting Help

### Common Issues

**Import Error**: Missing dependencies
```bash
pip install -r requirements.txt
```

**CUDA Error**: GPU not detected
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**Model Loading Error**: Check model path
```bash
ls -la /path/to/model  # Verify files exist
```

### Resources

- **Documentation**: [docs/](../)
- **Examples**: [examples/](../examples/)
- **Issues**: [GitHub Issues](https://github.com/hsliuustc0106/vllm-omni/issues)
- **Contributing**: [CONTRIBUTING.md](../CONTRIBUTING.md)

### Support Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community help
- **Email**: hsliuustc@gmail.com

## Best Practices

1. **Start Small**: Test with small models before large ones
2. **Monitor Resources**: Watch GPU memory usage with `nvidia-smi`
3. **Use Virtual Environments**: Keep dependencies isolated
4. **Read Examples**: Learn from working code
5. **Check Logs**: Enable verbose logging for debugging

## Benchmarking Your Setup

Test performance:

```python
import time
from vllm_omni import OmniLLM

llm = OmniLLM(model="path/to/model")

# Warm up
_ = llm.generate("Hello")

# Benchmark
start = time.time()
outputs = llm.generate("Test prompt" * 10)
elapsed = time.time() - start

print(f"Generation time: {elapsed:.2f}s")
```

---

**Congratulations!** You're now ready to build with vLLM-omni. Happy coding! 🚀
