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

Set up basic environments
```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
```
Install certain version of vllm with commitid: 808a7b69df479b6b3a16181711cac7ca28a9b941

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 808a7b69df479b6b3a16181711cac7ca28a9b941
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```

## Run examples (Qwen2.5-omni)

Get into the example folder
```bash
cd vllm_omni
cd examples/offline_inference/qwen2_5_omni
```
Modify PYTHONPATH in run.sh as your path of vllm_omni. Then run.
```bash
bash run.sh
```
The output audio is saved in ./output_audio

## 📍 Roadmap

- [x] Offline inference example for Qwen2.5-omni with single request
- [ ] Adaptation from current vllm branch to stable vllm v0.11.0
- [ ] Offline inference example for Qwen2.5-omni with streaming multiple requests
- [ ] Online inference support
- [ ] Support for other models

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