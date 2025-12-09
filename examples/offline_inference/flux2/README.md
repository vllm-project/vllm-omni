# FLUX 2 Offline Inference

This folder provides two simple entrypoints for experimenting with FLUX 2 using vLLM-Omni:

- `text_to_image.py`: command-line script for single image generation.
- `gradio_demo.py`: lightweight Gradio UI for interactive prompt/seed/guidance exploration.

## About FLUX 2

FLUX 2 is a state-of-the-art text-to-image diffusion model from Black Forest Labs featuring:
- **Mistral3-Small-24B** text encoder for rich, detailed prompt understanding
- Advanced transformer architecture with dual-stream and single-stream attention blocks
- 128-channel latent space with BatchNorm for high-quality generation
- Flow matching scheduler for efficient sampling

## Local CLI Usage

```bash
python text_to_image.py \
  --prompt "a serene mountain landscape at sunset" \
  --seed 42 \
  --guidance_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --output outputs/landscape.png
```

Key arguments:

- `--prompt`: text description (string).
- `--seed`: integer seed for deterministic sampling.
- `--guidance_scale`: guidance strength for classifier-free guidance (typically 3.5-5.0 for FLUX 2, default: 4.0).
- `--num_images_per_prompt`: number of images to generate per prompt (saves as `output`, `output_1`, ...).
- `--num_inference_steps`: diffusion sampling steps (50+ recommended for best quality, default: 50).
- `--height/--width`: output resolution (defaults 1024x1024).
- `--output`: path to save the generated PNG.

### Recommended Resolutions

FLUX 2 supports various aspect ratios. All dimensions must be divisible by 16. Official aspect ratio buckets:

- **1:1** (1024×1024) - Square
- **16:9** (1360×768) - Landscape
- **9:16** (768×1360) - Portrait
- **4:3** (1168×880) - Landscape
- **3:4** (880×1168) - Portrait
- **3:2** (1248×832) - Landscape
- **2:3** (832×1248) - Portrait

### Tips for Best Results

- **Prompt quality**: FLUX 2 uses Mistral3-Small-24B, which understands detailed, nuanced prompts better than simpler encoders.
- **Guidance scale**: Values between 3.5-5.0 typically work best. Lower values (2.5-3.5) for more creative/artistic results, higher (4.5-5.5) for stronger prompt adherence.
- **Inference steps**: 50 steps provide excellent quality. Use 30-40 for faster generation with slight quality trade-off, or 60+ for maximum detail.
- **Resolution**: Higher resolutions require more VRAM. Start with 1024×1024 and adjust based on your hardware.

## Web UI Demo

Launch the gradio demo:

```bash
python gradio_demo.py --port 7863
```

Then open `http://localhost:7863/` on your local browser to interact with the web UI.

### Gradio Options

```bash
python gradio_demo.py \
  --model black-forest-labs/FLUX.2-dev \
  --ip 0.0.0.0 \
  --port 7863 \
  --share
```

Arguments:

- `--model`: Model name or local path (default: `black-forest-labs/FLUX.2-dev`)
- `--ip`: Host/IP for Gradio server (default: 127.0.0.1)
- `--port`: Port for Gradio server (default: 7863)
- `--share`: Create a public shareable link via Gradio
- `--default-prompt`: Initial prompt shown in UI
- `--default-seed`: Initial seed value
- `--default-guidance-scale`: Initial guidance scale
- `--num_inference_steps`: Default number of steps

## Model Requirements

FLUX 2 models require the following structure:

- `model_index.json` with `"_class_name": "Flux2Pipeline"`
- `scheduler/` - Flow matching scheduler config
- `text_encoder/` - Mistral3-Small-24B weights
- `tokenizer/` - Mistral3 tokenizer
- `vae/` - AutoencoderKLFlux2 with BatchNorm
- `transformer/` - Flux2Transformer2DModel weights

The model is automatically detected as a diffusion model by the `Omni` entrypoint through the `model_index.json` file.

## Environment Setup

If using a custom diffusers installation with FLUX 2 components:

```bash
export VLLM_OMNI_DIFFUSERS_PATH=/path/to/custom/diffusers/src
```

This ensures FLUX 2-specific components (AutoencoderKLFlux2, Flux2ImageProcessor) are properly imported.

## Hardware Requirements

FLUX 2 is a large model. Recommended hardware:

- **Minimum**: 24GB VRAM (e.g., RTX 3090, RTX 4090, A5000)
- **Recommended**: 40GB+ VRAM (e.g., A100, H100) for higher resolutions and batch sizes
- **CPU**: Multi-core processor for efficient VAE decoding
- **RAM**: 32GB+ system RAM

For NPU devices, VAE memory optimizations (slicing and tiling) are automatically enabled.

