# Text-To-Video

Generate videos from text prompts using vLLM-Omni's diffusion and video pipeline entrypoints.

- `text_to_video.py`: command-line script for single video generation with model-aware defaults and advanced options.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Key Arguments](#key-arguments)
- [More CLI Examples](#more-cli-examples)
- [FAQ](#faq)

## Overview

This folder provides a unified CLI script for text-to-video generation using vLLM-Omni diffusion/video pipelines. The script selects practical defaults for supported model families while still exposing common sampling, memory, and parallelism options.

### Supported Models

| Model | Default Resolution | Default Frames | Default Steps | Guidance | VRAM Notes |
| ----- | ------------------ | -------------- | ------------- | -------- | ---------- |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | 720 x 1280 | 81 | 40 | 4.0 | Around 60 GiB BF16 for basic single-card usage |
| `Lightricks/LTX-2` | 512 x 768 | 121 | 40 | 4.0 | Memory use depends on frame count, tensor parallelism, and audio export |
| `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v` | 480 x 832 | 121 | 50 | 6.0 | Plan for an 80 GiB GPU for conservative single-card usage |
| `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` | 720 x 1280 | 121 | 50 | 6.0 | Use FP8 and VAE tiling for 720p single-card runs |

!!! info

    VRAM notes are conservative estimates for basic generation and can vary with driver, dependency versions, frame count, resolution, and memory optimization flags.

Default model: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`

## Quick Start

### Local CLI Usage

```bash
python text_to_video.py \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --negative-prompt "worst quality, inconsistent motion, blurry, jittery, distorted" \
  --height 480 \
  --width 832 \
  --num-frames 33 \
  --guidance-scale 4.0 \
  --guidance-scale-high 3.0 \
  --flow-shift 12.0 \
  --num-inference-steps 40 \
  --fps 16 \
  --output t2v_out.mp4
```

## Key Arguments

**Common arguments:**

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `--model` | str | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | Diffusers model ID or local path |
| `--prompt` | str | `"A serene lakeside sunrise with mist over the water."` | Text description for video generation |
| `--seed` | int | `42` | Integer seed for deterministic sampling |
| `--height` / `--width` | int | model-specific | Output video resolution in pixels |
| `--num-frames` | int | model-specific | Number of generated frames |
| `--num-inference-steps` | int | model-specific | Diffusion sampling steps |
| `--guidance-scale` | float | model-specific | Classifier-free guidance scale |
| `--fps` | int | model-specific | Frames per second for the saved MP4 |
| `--output` | str | model-specific | Path to save the generated video |
| `--vae-use-slicing` | flag | off | Enable VAE slicing for memory optimization |
| `--vae-use-tiling` | flag | off | Enable VAE tiling for memory optimization |
| `--cfg-parallel-size` | int | `1` | Set to `2` to enable CFG Parallel |
| `--tensor-parallel-size` | int | `1` | Tensor parallel size for models that support TP, such as LTX2 |
| `--ulysses-degree` | int | `1` | Ulysses sequence parallel degree |
| `--ring-degree` | int | `1` | Ring sequence parallel degree |
| `--enable-cpu-offload` | flag | off | Enable CPU offloading for diffusion models |
| `--enable-layerwise-offload` | flag | off | Enable layerwise offloading on DiT modules |
| `--frame-rate` | float | `None` | Optional generation frame rate for pipelines that require it, such as LTX2 |
| `--audio-sample-rate` | int | `24000` | Audio sample rate when the pipeline returns audio |
| `--quantization` | str | `None` | Quantization method: `fp8` or `gguf` |
| `--flow-shift` | float | model-specific | Scheduler `flow_shift` parameter |

**Wan2.2-specific arguments:**

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `--negative-prompt` | str | `""` | Artifacts or visual qualities to suppress |
| `--guidance-scale-high` | float | `None` | Separate CFG scale for the high-noise stage |
| `--boundary-ratio` | float | `None` | Boundary split ratio for low/high DiT; Wan2.2 default is `0.875` |
| `--flow-shift` | float | model-specific | Recommended values: `5.0` for 720p, `12.0` for 480p |
| `--cache-backend` | str | `None` | Use `cache_dit` to enable the Cache-DiT acceleration backend |
| `--enable-cache-dit-summary` | flag | off | Print Cache-DiT summary logging after diffusion forward passes |

**HunyuanVideo-1.5 optimal configs:**

| Variant | `--flow-shift` | `--guidance-scale` | `--num-inference-steps` |
| ------- | -------------- | ------------------ | ----------------------- |
| 480p T2V | `5.0` | `6.0` | `50` |
| 720p T2V | `9.0` | `6.0` | `50` |
| 480p I2V | `5.0` | `6.0` | `50` |
| 720p I2V | `7.0` | `6.0` | `50` |
| CFG-distilled | same as variant | `1.0` | `50` |

## More CLI Examples

### LTX2

```bash
python text_to_video.py \
  --model Lightricks/LTX-2 \
  --prompt "A cinematic close-up of ocean waves at golden hour." \
  --negative-prompt "worst quality, inconsistent motion, blurry, jittery, distorted" \
  --height 512 \
  --width 768 \
  --num-frames 121 \
  --num-inference-steps 40 \
  --guidance-scale 4.0 \
  --frame-rate 24 \
  --output ltx2_out.mp4
```

### HunyuanVideo-1.5 480p

```bash
python text_to_video.py \
  --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \
  --prompt "A cat walks through a sunlit garden, flowers swaying gently in the breeze." \
  --height 480 \
  --width 832 \
  --num-frames 121 \
  --guidance-scale 6.0 \
  --flow-shift 5.0 \
  --num-inference-steps 50 \
  --fps 24 \
  --output hunyuan_video_15_output.mp4
```

### HunyuanVideo-1.5 720p

```bash
python text_to_video.py \
  --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v \
  --prompt "A serene lakeside sunrise with mist over the water." \
  --height 720 \
  --width 1280 \
  --num-frames 121 \
  --guidance-scale 6.0 \
  --flow-shift 9.0 \
  --num-inference-steps 50 \
  --fps 24 \
  --output hunyuan_720p.mp4
```

### HunyuanVideo-1.5 with FP8 Quantization

```bash
python text_to_video.py \
  --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \
  --prompt "A dog running across a field of golden wheat." \
  --quantization fp8 \
  --height 480 \
  --width 832 \
  --num-frames 121 \
  --guidance-scale 6.0 \
  --flow-shift 5.0 \
  --output hunyuan_fp8.mp4
```

### Quick Test

```bash
python text_to_video.py \
  --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \
  --prompt "A serene lakeside sunrise with mist over the water." \
  --height 320 \
  --width 576 \
  --num-frames 17 \
  --num-inference-steps 30 \
  --flow-shift 5.0 \
  --output quick_test.mp4
```

## FAQ

**What should I try if generation runs out of memory?**

Try one or more memory-saving options: `--vae-use-slicing`, `--vae-use-tiling`, `--enable-cpu-offload`, or `--quantization fp8`. For quick smoke tests, use a smaller resolution and fewer frames, such as the Quick Test example above.
