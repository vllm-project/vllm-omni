# vLLM-omni: Multi-modal Extension for vLLM

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

vLLM-omni is designed to extend vLLM capabilities to support multi-modality model inference and serving, particularly focusing on non-autoregressive structures and non-textual outputs.

## 🎯 Overview

Traditional vLLM systems are limited to text-based, autoregressive generation. vLLM-omni addresses this limitation by enabling support for:

- **Multi-modal Models**: Text, image, video, audio, and sensor data processing
- **Non-autoregressive Architectures**: Diffusion Transformers (DiT) and other parallel generation models
- **Heterogeneous Outputs**: Beyond traditional text generation to multimodal outputs

## 🏗️ Architecture

vLLM-omni is built on a modular architecture that extends vLLM's core functionality:


## 🚀 Key Features

### Multi-Engine Support

- **Autoregressive Engine**: Traditional text generation with enhanced KV-caching
- **Diffusion Engine**: Support for DiT models and iterative generation
- **Hybrid Engine**: Combined AR+DiT processing pipelines

### Modality Processing

- **Text**: Advanced tokenization and embedding generation
- **Image**: Vision encoder integration (CLIP, etc.)
- **Audio**: Speech processing and audio embedding

## 📋 Supported Models

### AR + Diffusion Transformer (DiT) Models
- Qwen-omni (Thinker-Talker-Codec structure)
- HunyunaImage 3.0 (Ongoing)
- Qwen-Image (Ongoing)

## 🛠️ Installation

### Prerequisites

- **Python**: 3.8 or higher (3.12 recommended)
- **CUDA**: Compatible GPU with CUDA support
- **Operating System**: Linux (tested on Ubuntu 20.04+)

### Quick Install (Coming Soon)

Once released on PyPI:
```bash
pip install vllm-omni
```

### Development Installation

#### Step 1: Set up Python environment

Using `uv` (recommended):
```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
```

Or using standard `venv`:
```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 2: Install vLLM dependency

Install specific version of vLLM (commit: 808a7b69df479b6b3a16181711cac7ca28a9b941):

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 808a7b69df479b6b3a16181711cac7ca28a9b941
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
cd ..
```

#### Step 3: Install vLLM-omni

For development:
```bash
git clone https://github.com/hsliuustc0106/vllm-omni.git
cd vllm-omni
pip install -e ".[dev]"  # Includes development dependencies
```

Or for production:
```bash
git clone https://github.com/hsliuustc0106/vllm-omni.git
cd vllm-omni
pip install -e .
```

#### Step 4: Verify Installation

```bash
python -c "import vllm_omni; print(f'vLLM-omni {vllm_omni.__version__} installed successfully!')"
```

## 🚀 Quick Start

### Running Qwen2.5-omni Example

Navigate to the example directory:
```bash
cd examples/offline_inference/qwen_2_5_omni
```

Modify `PYTHONPATH` in `run.sh` to point to your vllm_omni installation path, then run:
```bash
bash run.sh
```

The output audio will be saved in `./output_audio`

For more examples, see the [examples](examples/) directory.

## 💡 Usage Examples

### Basic Text Generation

```python
from vllm_omni import OmniLLM

# Initialize the model
llm = OmniLLM(model="path/to/model")

# Generate text
outputs = llm.generate("Hello, how are you?")
print(outputs[0].text)
```

### Multi-modal Processing

```python
from vllm_omni import OmniLLM
from PIL import Image

# Initialize with multi-modal model
llm = OmniLLM(model="Qwen2.5-omni")

# Process text and image
image = Image.open("image.jpg")
outputs = llm.generate(
    prompt="Describe this image",
    image=image
)
```

For comprehensive examples, see:
- [Basic Examples](examples/basic/) - Simple text generation and API usage
- [Omni Examples](examples/omni/) - Multi-modal model examples
- [Offline Inference](examples/offline_inference/) - Batch processing examples

## ⚙️ Configuration

vLLM-omni can be configured through:

- **Python API**: Direct configuration when initializing models
- **YAML configs**: Stage-specific configurations in `vllm_omni/model_executor/stage_configs/`
- **Environment variables**: For system-level settings

Example configuration:
```python
from vllm_omni import OmniLLM
from vllm_omni.config import DiTConfig, DiTCacheConfig

# Configure DiT engine
dit_config = DiTConfig(
    max_iterations=50,
    guidance_scale=7.5
)

llm = OmniLLM(
    model="path/to/model",
    dit_config=dit_config
)
```

## 📍 Roadmap

- [x] Offline inference example for Qwen2.5-omni with single request
- [ ] Adaptation from current vllm branch to stable vllm v0.11.0
- [ ] Offline inference example for Qwen2.5-omni with streaming multiple requests
- [ ] Online inference support
- [ ] Support for other models

## 🔧 Troubleshooting

### Common Issues

#### ImportError: No module named 'cloudpickle'

Install missing dependencies:
```bash
pip install cloudpickle
```

#### CUDA Out of Memory

Reduce batch size or model size:
```python
llm = OmniLLM(
    model="path/to/model",
    max_model_len=2048,  # Reduce context length
    gpu_memory_utilization=0.8  # Reduce GPU memory usage
)
```

#### vLLM Version Mismatch

Ensure you're using the correct vLLM commit:
```bash
cd vllm
git fetch
git checkout 808a7b69df479b6b3a16181711cac7ca28a9b941
pip install --editable . --force-reinstall
```

#### Model Loading Errors

Verify model path and format:
```bash
ls -la /path/to/model  # Should contain config.json, model files, etc.
```

For more issues, check:
- [GitHub Issues](https://github.com/hsliuustc0106/vllm-omni/issues)
- [vLLM Documentation](https://docs.vllm.ai/)

### Getting Help

If you encounter issues:
1. Check existing [issues](https://github.com/hsliuustc0106/vllm-omni/issues)
2. Review [documentation](docs/)
3. Open a new issue with:
   - Error messages and stack traces
   - Python and CUDA versions
   - Minimal reproduction code
   - System information (OS, GPU model)

## 📚 Documentation

For more detailed information, see:
- [Architecture Design](docs/architecture/high_level_arch_design.md) - High-level system architecture
- [Implementation Details](docs/architecture/implementation_architecture.md) - Detailed implementation
- [API Documentation](docs/api/README.md) - API usage and design

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:
- Setting up your development environment
- Code style and standards
- Submitting pull requests
- Reporting issues

Please also read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## 🔒 Security

For security issues, please see our [Security Policy](SECURITY.md) for responsible disclosure guidelines.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📝 Citation

If you use vLLM-omni in your research, please cite:

```bibtex
@software{vllm_omni_2024,
  title = {vLLM-omni: Multi-modal Extension for vLLM},
  author = {vLLM-omni Team},
  year = {2024},
  url = {https://github.com/hsliuustc0106/vllm-omni}
}
```

## 🙏 Acknowledgments

vLLM-omni is built on top of [vLLM](https://github.com/vllm-project/vllm), an excellent fast and easy-to-use library for LLM inference and serving.

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/hsliuustc0106/vllm-omni/issues)
- **Email**: hsliuustc@gmail.com

---

**Star ⭐ this repository if you find it helpful!**