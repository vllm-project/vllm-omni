# vLLM-omni: Multi-modal Extension for vLLM

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

## To-do list
- [x] Offline inference example for Qwen2.5-omni with single request
- [ ] Adaptation from current vllm branch to stable vllm v0.11.0
- [ ] Offline inference example for Qwen2.5-omni with streaming multiple requests
- [ ] Online inference support
- [ ] Support for other models

For detailed model management, see [vllm_omni_design.md](docs/architecture/vllm_omni_design.md) and [high_level_arch_design.md](docs/architecture/high_level_arch_design.md).