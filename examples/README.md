# vLLM-omni Examples

This directory contains examples demonstrating how to use vLLM-omni for various tasks.

## Basic Examples

- [Text Generation](basic/text_generation.py) - Basic text generation using AR models
- [Image Generation](basic/image_generation.py) - Image generation using diffusion models
- [Multimodal Processing](basic/multimodal_processing.py) - Processing text and images together

## Advanced Examples

- [Custom Model Integration](advanced/custom_model.py) - Integrating custom models
- [Batch Processing](advanced/batch_processing.py) - Efficient batch processing
- [Streaming Output](advanced/streaming.py) - Real-time streaming output

## Multimodal Examples

- [Image-to-Text](multimodal/image_to_text.py) - Image captioning and description
- [Text-to-Image](multimodal/text_to_image.py) - Text-to-image generation
- [Audio Processing](multimodal/audio_processing.py) - Audio generation and processing
- [Video Generation](multimodal/video_generation.py) - Video generation workflows

## API Examples

- [REST API](api/rest_api.py) - Using the REST API interface
- [Gradio Interface](api/gradio_interface.py) - Creating Gradio web interfaces
- [ComfyUI Integration](api/comfyui_integration.py) - ComfyUI workflow integration

## Configuration Examples

- [Custom Configuration](config/custom_config.py) - Custom configuration setup
- [Multi-GPU Setup](config/multi_gpu.py) - Multi-GPU configuration
- [Distributed Processing](config/distributed.py) - Distributed processing setup

## Getting Started

1. Install vLLM-omni:
   ```bash
   pip install vllm-omni
   ```

2. Run a basic example:
   ```bash
   python examples/basic/text_generation.py
   ```

3. Explore the examples in each subdirectory for more advanced usage.

## Requirements

Most examples require additional dependencies. Install them with:

```bash
pip install -r requirements.txt
```

For development examples, install the development dependencies:

```bash
pip install -r requirements-dev.txt
```
