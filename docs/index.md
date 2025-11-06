# vLLM-omni Documentation

**vLLM-omni** is a multi-modality extension for vLLM that supports non-autoregressive structures and non-textual outputs. It enables support for:

- **Multi-modal Models**: Text, image, video, audio, and sensor data processing
- **Non-autoregressive Architectures**: Diffusion Transformers (DiT) and other parallel generation models
- **Heterogeneous Outputs**: Beyond traditional text generation to multimodal outputs

## <span class="twemoji">🎯</span> Key Features

### Multi-Engine Support

- **Autoregressive Engine**: Traditional text generation with enhanced KV-caching
- **Diffusion Engine**: Support for DiT models and iterative generation
- **Hybrid Engine**: Combined AR+DiT processing pipelines

### Modality Processing

- **Text**: Advanced tokenization and embedding generation
- **Image**: Vision encoder integration (CLIP, etc.)
- **Audio**: Speech processing and audio embedding

## <span class="twemoji">🚀</span> Quick Start

### Installation

```bash
pip install vllm-omni
```

### Basic Usage

```python
from vllm_omni import OmniLLM

# Initialize the model
llm = OmniLLM(model="Qwen/Qwen2.5-Omni")

# Generate outputs
outputs = llm.generate(
    prompts="Your prompt here",
    sampling_params_list=[...]
)
```

## <span class="twemoji">🤖</span> Supported Models

- Qwen-omni (Thinker-Talker-Codec structure)
- HunyunaImage 3.0 (Ongoing)
- Qwen-Image (Ongoing)

## <span class="twemoji">📚</span> Documentation

- [:material-play-circle: Getting Started](getting_started/quickstart.md) - Installation and quick start guide
- [:material-sitemap: Architecture](contributing/design_documents/vllm_omni_design.md) - System architecture and design
- [:material-code-tags: API Reference](api/index.md) - Complete API documentation
- [:material-code-braces: Examples](examples/index.md) - Code examples and tutorials
- [:material-tools: Developer Guide](contributing/README.md) - Contributing and development guide

