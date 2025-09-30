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

### Quick Start

#### Option 1: Docker (Recommended for macOS)

```bash
# Clone the repository
git clone https://github.com/hsliuustc0106/vllm-omni.git
cd vllm-omni

# Run the automated Docker setup
./scripts/docker-setup-macos.sh
```

#### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/hsliuustc0106/vllm-omni.git
cd vllm-omni

# Run the installation script
./install.sh
```

### Prerequisites

- Python 3.11+ (recommended)
- Conda or Miniconda
- Git
- CUDA 11.8+ (for GPU acceleration) or CPU-only installation

### Installation Methods

#### Method 1: Automated Installation (Recommended)
```bash
# Using shell script
./install.sh

# Or using Python script
python install.py
```

#### Method 2: Manual Installation
```bash
# Create conda environment
conda create -n vllm_omni python=3.11 -y
conda activate vllm_omni

# Install PyTorch (CPU or GPU)
pip install torch>=2.7 --index-url https://download.pytorch.org/whl/cpu  # CPU
# pip install torch>=2.7 --index-url https://download.pytorch.org/whl/cu121  # GPU

# Install dependencies
pip install -r requirements.txt
pip install "vllm>=0.10.2"

# Install vLLM-omni
pip install -e .
```

### Verify Installation

```bash
# Test the installation
python test_installation.py

# Test basic functionality
python -c "import vllm_omni; print('Ready!')"

# Test CLI
vllm --help
```

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## 📥 Model Download

Models are automatically downloaded when first used, or you can pre-download them:

```bash
# Check downloaded models
python scripts/download_models.py --check-cache

# Download all default models
python scripts/download_models.py --all

# Download specific models
python scripts/download_models.py --ar-models Qwen/Qwen3-0.6B
python scripts/download_models.py --dit-models stabilityai/stable-diffusion-2-1
```

**Model Storage Location:**
- Default: `~/.cache/huggingface/hub/`
- AR models: 100MB - 1GB each
- DiT models: 2GB - 7GB each

For detailed model management, see [MODEL_DOWNLOAD_GUIDE.md](docs/MODEL_DOWNLOAD_GUIDE.md).
