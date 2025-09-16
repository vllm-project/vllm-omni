# vLLM-omni: Multi-modal Extension for vLLM

vLLM-omni is designed to extend vLLM capabilities to support multi-modality model inference and serving, particularly focusing on non-autoregressive structures and non-textual outputs.

## 🎯 Overview

Traditional vLLM systems are limited to text-based, autoregressive generation. vLLM-omni addresses this limitation by enabling support for:

- **Multi-modal Models**: Text, image, video, audio, and sensor data processing
- **Non-autoregressive Architectures**: Diffusion Transformers (DiT) and other parallel generation models
- **Heterogeneous Outputs**: Beyond traditional text generation to structured, binary, and streaming outputs

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
- **Video**: Frame-by-frame and temporal processing
- **Sensor**: IoT and sensor data interpretation

### Output Formats

- **Structured Data**: JSON, XML, and custom formats
- **Binary Outputs**: Images, audio, and video generation
- **Streaming**: Real-time progressive generation
- **Multipart**: Combined multi-modal responses

## 📋 Supported Models

### AR + Diffusion Transformer (DiT) Models
- Qwen-Image (Image generation and editing)
- Qwen-omni (Thinker-Talker-Codec structure)
- Custom DiT and hiybrid architectures

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

### Install from Source

```bash
git clone https://github.com/your-org/vllm-omni.git
pip install -r requirements.txt
cd vllm-omni
pip install -e .
```
