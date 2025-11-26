# Welcome to vLLM-omni

**vLLM-omni** extends vLLM capabilities to support multi-modality model inference and serving, particularly focusing on non-autoregressive structures and non-textual outputs. It aims to support:

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

## <span class="twemoji">🤖</span> Supported Models

- Qwen-omni (Thinker-Talker-Codec structure)
- HunyunaImage 3.0 (Ongoing)
- Qwen-Image (Ongoing)

## <span class="twemoji">📚</span> Documentation Navigation

- To run open-source models on vLLM-Omni, we recommend starting with the [:material-code-tags: User Quide](user_guide/getting_started/quickstart.md)
- To develop and contribute to vLLM-Omni, we recommend starting with the [:material-tools: Developer Guide](contributing/README.md)
