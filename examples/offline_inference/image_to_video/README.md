# Image-To-Video

This shared example generates videos from images with VACE, Wan2.2,
LTX-2/LTX-2.3, HunyuanVideo-1.5, Cosmos3, and other compatible pipelines.

- `image_to_video.py`: command-line script for single video generation with advanced options.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Key Arguments](#key-arguments)
- [More CLI Examples](#more-cli-examples)
- [Advanced Features](#advanced-features)
- [FAQ](#faq)

## Overview

This folder provides a unified CLI script for image-to-video generation using vLLM-Omni diffusion/video pipelines. The script selects practical defaults for supported model families while still exposing common sampling, memory, and parallelism options.

### Supported Models

| Model | Default Resolution | Default Frames | Default Steps | Guidance | VRAM Notes |
| ----- | ------------------ | -------------- | ------------- | -------- | ---------- |
| `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | 480 x 832 | 81 | 50 | 5.0 | Around 60 GiB BF16 for basic single-card usage |
| `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | 480 x 832 | 81 | 50 | 4.0 | Around 20–25 GiB BF16, smallest I2V model |
| `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v` | 480 x 832 | 121 | 50 | 6.0 | Around 100 GiB at default settings; the example enables `--enable-cpu-offload` + VAE tiling/slicing to fit an 80 GiB card |
| LTX2 (local path + `--model-class-name LTX2ImageToVideoPipeline`) | 512 x 768 | 121 | 40 | 4.0 | Memory use depends on frame count and tensor parallelism |

!!! info
    Peak VRAM: based on basic single-card usage, batch size = 1, without any acceleration/optimization features. Some model weights cannot fit into one card with 80 GiB VRAM, which may need to use CPU offloading.

Default model: `Wan-AI/Wan2.2-I2V-A14B-Diffusers`.

## Prerequisites

Download the example image used in the snippets below:

```bash
curl -L -o cherry_blossom.jpg https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg
```

## Quick Start

### Python API

Single-prompt generation using TI2V-5B (lightweight):

```python
import PIL.Image
import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

if __name__ == "__main__":
    image = PIL.Image.open("cherry_blossom.jpg").convert("RGB")
    image = image.resize((576, 320))

    omni = Omni(
        model="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        flow_shift=12.0,
    )
    outputs = omni.generate(
        {
            "prompt": "Cherry blossoms swaying gently in the breeze, petals falling",
            "multi_modal_data": {"image": image},
        },
        OmniDiffusionSamplingParams(
            height=320,
            width=576,
            num_frames=17,
            num_inference_steps=20,
            guidance_scale=4.0,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ),
    )
    from diffusers.utils import export_to_video
    frames = outputs[0].request_output.images
    export_to_video(frames, "quick_test_i2v.mp4", fps=16)
```

### Local CLI Usage

Quick test using TI2V-5B with a small resolution and few frames:

```bash
python image_to_video.py \
  --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --image cherry_blossom.jpg \
  --prompt "Cherry blossoms swaying gently in the breeze, petals falling, smooth motion" \
  --height 320 \
  --width 576 \
  --num-frames 17 \
  --guidance-scale 4.0 \
  --num-inference-steps 20 \
  --flow-shift 12.0 \
  --fps 16 \
  --output quick_test_i2v.mp4
```

## Key Arguments

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `--model` | str | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | Diffusers I2V model ID or local path |
| `--model-class-name` | str | `None` | Override model class name (e.g., `LTX2ImageToVideoPipeline`) |
| `--image` | str | (required) | Path to input image |
| `--prompt` | str | `""` | Text description of desired motion/animation |
| `--negative-prompt` | str | `""` | Optional list of artifacts to suppress |
| `--seed` | int | `42` | Random seed for deterministic sampling |
| `--guidance-scale` | float | `5.0` | CFG scale |
| `--guidance-scale-high` | float | `None` | Separate CFG for high-noise stage (MoE only) |
| `--height` | int | auto | Video height (auto-calculated from image if not set). Multiples of 16 |
| `--width` | int | auto | Video width (auto-calculated from image if not set). Multiples of 16 |
| `--num-frames` | int | `81` | Number of frames |
| `--num-inference-steps` | int | `50` | Number of denoising steps |
| `--boundary-ratio` | float | `0.875` | Boundary split ratio for two-stage MoE models |
| `--flow-shift` | float | `5.0` | Scheduler flow shift (5.0 for 720p, 12.0 for 480p) |
| `--sample-solver` | str | `unipc` | Wan2.2 sampling solver (`unipc` or `euler` for Lightning/Distill) |
| `--fps` | int | `None` | Frames per second for the saved MP4 |
| `--frame-rate` | float | `None` | Generation frame rate for pipelines that require it (e.g., LTX2) |
| `--output` | str | `i2v_output.mp4` | Path to save the generated video |
| `--vae-use-slicing` | flag | off | Enable VAE slicing for memory optimization |
| `--vae-use-tiling` | flag | off | Enable VAE tiling for memory optimization |
| `--enable-cpu-offload` | flag | off | Enable CPU offloading for diffusion models |
| `--enable-layerwise-offload` | flag | off | Enable layerwise offloading on DiT modules |
| `--cfg-parallel-size` | int | `1` | Set to `2` to enable CFG Parallel |
| `--tensor-parallel-size` | int | `1` | Tensor parallel size (effective for models that support TP, e.g. LTX2) |
| `--ulysses-degree` | int | `1` | Ulysses sequence parallel degree |
| `--ring-degree` | int | `1` | Ring sequence parallel degree |
| `--cache-backend` | str | `None` | Cache backend: `cache_dit` or `tea_cache` |
| `--use-hsdp` | flag | off | Enable Hybrid Sharded Data Parallel |
| `--hsdp-shard-size` | int | `-1` | GPUs per shard group (-1 auto-calculates) |
| `--hsdp-replicate-size` | int | `1` | Number of replica groups for HSDP |

## More CLI Examples

### Wan2.2-I2V-A14B-Diffusers (MoE)

```bash
python image_to_video.py \
  --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --image cherry_blossom.jpg \
  --prompt "Cherry blossoms swaying gently in the breeze, petals falling, smooth motion" \
  --negative-prompt "low quality, blurry" \
  --height 480 \
  --width 832 \
  --num-frames 48 \
  --guidance-scale 5.0 \
  --guidance-scale-high 6.0 \
  --num-inference-steps 40 \
  --boundary-ratio 0.875 \
  --flow-shift 12.0 \
  --fps 16 \
  --output i2v_wan_moe.mp4
```

### Wan2.2-TI2V-5B-Diffusers (Unified)

```bash
python image_to_video.py \
  --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --image cherry_blossom.jpg \
  --prompt "Cherry blossoms swaying gently in the breeze, petals falling, smooth motion" \
  --negative-prompt "low quality, blurry" \
  --height 480 \
  --width 832 \
  --num-frames 48 \
  --guidance-scale 4.0 \
  --num-inference-steps 40 \
  --flow-shift 12.0 \
  --fps 16 \
  --output i2v_wan_ti2v.mp4
```

### HunyuanVideo-1.5 I2V (480p)

```bash
python image_to_video.py \
  --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v \
  --image cherry_blossom.jpg \
  --prompt "Cherry blossoms swaying gently in the breeze, petals falling, smooth motion" \
  --height 480 \
  --width 832 \
  --num-frames 121 \
  --guidance-scale 6.0 \
  --flow-shift 5.0 \
  --num-inference-steps 50 \
  --fps 24 \
  --enable-cpu-offload \
  --vae-use-tiling \
  --vae-use-slicing \
  --output hunyuan_i2v.mp4
```

### LTX2 Image-to-Video

```bash
python image_to_video.py \
  --model /path/to/LTX-2 \
  --model-class-name LTX2ImageToVideoPipeline \
  --image cherry_blossom.jpg \
  --prompt "A cinematic dolly shot of cherry blossoms" \
  --height 512 \
  --width 768 \
  --num-frames 121 \
  --num-inference-steps 40 \
  --guidance-scale 4.0 \
  --frame-rate 24 \
  --fps 24 \
  --output ltx2_i2v.mp4
```

## Advanced Features

### CFG Parallel

Set `--cfg-parallel-size 2` to enable CFG Parallel for faster inference on multi-GPU setups.
See more examples in the [cfg_parallel user guide](../../../docs/user_guide/diffusion/parallelism/cfg_parallel.md).

### Cache Acceleration

Use `--cache-backend cache_dit` for Cache-DiT acceleration or `--cache-backend tea_cache` for Timestep Embedding Aware Cache:

```bash
python image_to_video.py \
  --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --image cherry_blossom.jpg \
  --prompt "Cherry blossoms swaying gently in the breeze" \
  --cache-backend cache_dit \
  --output i2v_cached.mp4
```

### LTX-2.3

```bash
python image_to_video.py \
  --model dg845/LTX-2.3-Diffusers \
  --model-class-name LTX23ImageToVideoPipeline \
  --image cherry_blossom.jpg \
  --prompt "Cherry blossoms swaying gently in the breeze with synchronized ambient sound" \
  --negative-prompt "worst quality, inconsistent motion, blurry, jittery, distorted" \
  --height 384 \
  --width 512 \
  --num-frames 25 \
  --guidance-scale 4.0 \
  --num-inference-steps 20 \
  --frame-rate 24 \
  --fps 24 \
  --output ltx23_i2v_output.mp4
```

Use a Diffusers-format checkpoint such as `dg845/LTX-2.3-Diffusers`; the
upstream `Lightricks/LTX-2.3` raw safetensors repo is not directly loadable by
this pipeline. Pass `--model-class-name LTX23ImageToVideoPipeline` to select
the LTX-2.3 image-to-video pipeline.

### Cosmos3

```bash
# Cosmos3 bundles example frames under assets/ (any RGB image works too):
python image_to_video.py \
  --model nvidia/Cosmos3-Nano \
  --image /path/to/Cosmos3-Nano/assets/example_i2v_input.jpg \
  --prompt "The scene comes to life with smooth, natural motion." \
  --negative-prompt "blurry, distorted, low quality" \
  --height 720 --width 1280 --num-frames 189 --fps 24 \
  --num-inference-steps 35 --guidance-scale 6.0 \
  --extra-body '{"flow_shift": 10.0, "max_sequence_length": 4096, "guardrails": false}' \
  --output cosmos3_i2v.mp4
```

Key arguments:

- `--model`: Model ID (I2V-A14B for MoE, TI2V-5B for unified T2V+I2V,
  LTX-2/LTX-2.3, Cosmos3, or VACE).
- `--image`: Path to the first-frame or source image.
- `--last-image`: Optional last-frame condition for models such as VACE.
- `--mask-image`: Optional inpainting mask. White pixels are regenerated and black pixels are preserved.
- `--reference-image`: Optional reference image; repeat it to provide multiple references.
- `--extra-body`: JSON object of model-specific generation params, filtered against the model's declared `extra_body_params` (see [`vllm_omni/model_extras`](../../../vllm_omni/model_extras)). Used by Cosmos3.
- `--prompt`: Text description of desired motion/animation.
- `--height/--width`: Output resolution (auto-calculated from image if not set).
  Wan dimensions should be multiples of 16; LTX dimensions should be multiples
  of 32.
- `--num-frames`: Number of frames (model-specific default; LTX-style models
  work best with `8k + 1`; Cosmos3 defaults to 189).
- `--guidance-scale` and `--guidance-scale-high`: CFG scale (applied to low/high-noise stages for MoE).
- `--negative-prompt`: Optional list of artifacts to suppress.
- `--boundary-ratio`: Boundary split ratio for two-stage MoE models.
- `--flow-shift`: Scheduler flow shift (default: model-specific — Wan/LTX2 5.0, Cosmos3 10.0).
- `--sample-solver`: Wan2.2 sampling solver. Use `unipc` for the default multistep solver, or `euler` for Lightning/Distill checkpoints.
- `--num-inference-steps`: Number of denoising steps (default: model-specific — Wan 50, LTX2 40, Cosmos3 35).
- `--fps`: Frames per second for the saved MP4 (requires `diffusers` export_to_video).
- `--audio-sample-rate`: fallback audio sample rate for embedded audio.
- `--output`: Path to save the generated video.
- `--vae-use-slicing`: Enable VAE slicing for memory optimization.
- `--vae-use-tiling`: Enable VAE tiling for memory optimization.
- `--cfg-parallel-size`: set it to 2 to enable CFG Parallel. See more examples in [`user_guide`](https://github.com/vllm-project/vllm-omni/tree/main/docs/user_guide/diffusion/parallelism/cfg_parallel.md).
- `--tensor-parallel-size`: tensor parallel size (effective for models that support TP, e.g. LTX2).
- `--enable-cpu-offload`: enable CPU offloading for diffusion models.
- `--use-hsdp`: Enable Hybrid Sharded Data Parallel to shard model weights across GPUs.
- `--hsdp-shard-size`: Number of GPUs to shard model weights across within each replica group. -1 (default) auto-calculates as world_size / replicate_size.
- `--hsdp-replicate-size`: Number of replica groups for HSDP. Each replica holds a full sharded copy. Default 1 means pure sharding (no replication).



> ℹ️ If you encounter OOM errors, try using `--vae-use-slicing` and `--vae-use-tiling` to reduce memory usage.

## Wan2.1 VACE Conditional Tasks

The shared script selects the VACE conditioning structure from the media inputs.
No explicit mode parameter is required: the script constructs the source video,
mask, or reference images consumed by the VACE pipeline.
Download the same Hugging Face assets used by the original VACE example:

```bash
wget -O astronaut.jpg https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg
wget -O vace_first_frame.png https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png
wget -O vace_last_frame.png https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_last_frame.png
```

### Image-to-Video (I2V)

```bash
python image_to_video.py \
  --model Wan-AI/Wan2.1-VACE-1.3B-diffusers \
  --image astronaut.jpg \
  --prompt "An astronaut emerging from a cracked, otherworldly egg on the surface of the moon" \
  --seed 42 --height 480 --width 832 --num-frames 81 \
  --num-inference-steps 30 --guidance-scale 5.0 --flow-shift 5.0 \
  --vae-use-tiling --output vace_i2v_output.mp4
```

### Video-to-Last-Frame (V2LF)

```bash
python image_to_video.py \
  --model Wan-AI/Wan2.1-VACE-1.3B-diffusers \
  --last-image astronaut.jpg \
  --prompt "An astronaut emerging from a cracked, otherworldly egg on the surface of the moon" \
  --seed 42 --height 480 --width 832 --num-frames 81 \
  --num-inference-steps 30 --guidance-scale 5.0 --flow-shift 5.0 \
  --vae-use-tiling --output vace_v2lf_output.mp4
```

### First-Last-Frame-to-Video (FLF2V)

```bash
python image_to_video.py \
  --model Wan-AI/Wan2.1-VACE-1.3B-diffusers \
  --image vace_first_frame.png \
  --last-image vace_last_frame.png \
  --prompt "CG animation style, a small blue bird takes off from a branch and lands on another branch" \
  --seed 42 --height 512 --width 512 --num-frames 81 \
  --num-inference-steps 30 --guidance-scale 5.0 --flow-shift 5.0 \
  --vae-use-tiling --output vace_flf2v_output.mp4
```

### Inpainting

Create a mask matching the original VACE example: a 160-pixel-wide white
vertical stripe marks the region to regenerate.

```bash
python - <<'PY'
from PIL import Image

mask = Image.new("L", (832, 480), 0)
mask.paste(255, (336, 0, 496, 480))
mask.save("vace_center_mask.png")
PY
```

```bash
python image_to_video.py \
  --model Wan-AI/Wan2.1-VACE-1.3B-diffusers \
  --image astronaut.jpg \
  --mask-image vace_center_mask.png \
  --prompt "Shrek, the ogre, walks out of a building in a happy mood" \
  --seed 42 --height 480 --width 832 --num-frames 81 \
  --num-inference-steps 30 --guidance-scale 5.0 --flow-shift 5.0 \
  --vae-use-tiling --output vace_inpaint_output.mp4
```

### Reference-to-Video (R2V)

Repeat `--reference-image` to provide more than one reference image.

```bash
python image_to_video.py \
  --model Wan-AI/Wan2.1-VACE-1.3B-diffusers \
  --reference-image astronaut.jpg \
  --prompt "Camera slowly zooms out from the character walking in a garden" \
  --seed 42 --height 480 --width 832 --num-frames 81 \
  --num-inference-steps 30 --guidance-scale 5.0 --flow-shift 5.0 \
  --vae-use-tiling --output vace_r2v_output.mp4
```

The VACE T2V command is documented in the shared
[`text_to_video.py`](../text_to_video/text_to_video.md#wan21-vace-t2v) example.

For Wan2.2 LightX2V-converted local Diffusers directories and related LoRA
assets, see the [LoRA guide](../../../docs/user_guide/diffusion/lora.md#wan22-lightx2v-offline-assembly).

## FAQ

**OOM errors**: Try using `--vae-use-slicing` and `--vae-use-tiling` to reduce memory usage. For very large models, add `--enable-cpu-offload` or `--enable-layerwise-offload`.

**Auto-calculated resolution**: If `--height` and `--width` are not provided, the script calculates output dimensions from the input image while maintaining aspect ratio and targeting 480 x 832 area (or 512 x 768 for LTX2).

**Wan2.2 MoE vs unified**: `I2V-A14B` is a larger Mixture-of-Experts model with separate low/high-noise DiT stages (use `--guidance-scale-high` and `--boundary-ratio`). `TI2V-5B` is a smaller unified T2V+I2V model that does not need these extra arguments.
