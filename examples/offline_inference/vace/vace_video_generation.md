# VACE Video Generation

Generate videos from text prompts, images, or video conditions using vLLM-Omni's VACE diffusion pipeline.

- `vace_video_generation.py`: command-line script for multi-mode video generation with advanced options.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Key Arguments](#key-arguments)
- [More CLI Examples](#more-cli-examples)

## Overview

[VACE](https://github.com/ali-vilab/VACE) (Video All-in-one Creation Engine) supports multiple video tasks through a single unified model, including text-to-video, image-to-video, first-last-frame interpolation, inpainting, and reference image-guided generation.

### Supported Models

| Model | Architecture | Peak VRAM (GiB) * | Model Weights (GiB) | HuggingFace |
|-------|-------------|-------------------|---------------------|-------------|
| Wan2.1-VACE (1.3B) | Wan2.1 | TBD | ~10 | [Wan-AI/Wan2.1-VACE-1.3B-diffusers](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers) |
| Wan2.1-VACE (14B) | Wan2.1 | TBD | ~38 | [Wan-AI/Wan2.1-VACE-14B-diffusers](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B-diffusers) |

!!! info
*Peak VRAM: based on basic single-card usage, 480x832 resolution, 81 frames, 30 inference steps, without any acceleration/optimization features.

Default model: `Wan-AI/Wan2.1-VACE-14B-diffusers`

## Quick Start

### Python API

Text-to-video generation:

```python
import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

if __name__ == "__main__":
    omni = Omni(model="Wan-AI/Wan2.1-VACE-1.3B-diffusers")
    prompt = "A sleek robot stands in a vast warehouse filled with boxes"
    outputs = omni.generate(
        prompt,
        OmniDiffusionSamplingParams(
            height=480,
            width=832,
            num_frames=81,
            num_inference_steps=30,
            guidance_scale=5.0,
        ),
    )
    video = outputs[0].images

    from diffusers.utils import export_to_video
    export_to_video(list(video[0]), "t2v_output.mp4", fps=16)
    omni.close()
```

### Local CLI Usage

```bash
python vace_video_generation.py \
  --mode t2v \
  --prompt "A sleek robot stands in a vast warehouse filled with boxes" \
  --height 480 --width 832 --num-frames 81 \
  --output t2v_output.mp4
```

## Key Arguments

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `--mode` | str | `"t2v"` | VACE task mode: `t2v`, `i2v`, `v2lf`, `flf2v`, `inpaint`, `r2v` |
| `--model` | str | `"Wan-AI/Wan2.1-VACE-14B-diffusers"` | Model ID or local path |
| `--prompt` | str | `"A cat walking in a garden"` | Text description of desired video |
| `--negative-prompt` | str | `""` | Negative prompt for classifier-free guidance |
| `--image` | str | `None` | Input image path (for I2V, R2V, FLF2V, inpaint modes) |
| `--last-image` | str | `None` | Last frame image path (for FLF2V mode) |
| `--height` | int | `480` | Output video height in pixels (should be a multiple of 16) |
| `--width` | int | `832` | Output video width in pixels (should be a multiple of 16) |
| `--num-frames` | int | `81` | Number of video frames to generate |
| `--num-inference-steps` | int | `30` | Number of denoising steps (more steps = higher quality, slower) |
| `--guidance-scale` | float | `5.0` | Classifier-free guidance scale |
| `--flow-shift` | float | `5.0` | Scheduler flow shift parameter |
| `--seed` | int | `42` | Random seed for deterministic sampling |
| `--fps` | int | `16` | Frames per second for the saved MP4 |
| `--output` | str | `"vace_output.mp4"` | Path to save the generated video |
| `--vae-use-tiling` | flag | on | Enable VAE tiling for memory optimization |
| `--cfg-parallel-size` | int | `1` | Set to `2` to enable CFG Parallel |
| `--ulysses-degree` | int | `1` | Ulysses sequence parallel degree for multi-GPU inference |
| `--ring-degree` | int | `1` | Ring sequence parallel degree for hybrid Ulysses + Ring inference |
| `--tensor-parallel-size` | int | — | Tensor parallel size |
| `--enforce-eager` | flag | off | Disable torch.compile |

> If you encounter OOM errors, try `--vae-use-tiling` or multi-GPU parallelism options (`--ulysses-degree`, `--cfg-parallel-size`).

## More CLI Examples

### Image-to-Video (I2V)

First frame is kept, remaining frames are generated:

```bash
python vace_video_generation.py \
  --mode i2v \
  --image astronaut.jpg \
  --prompt "An astronaut emerging from a cracked egg on the moon" \
  --height 480 --width 832 --num-frames 81 \
  --output i2v_output.mp4
```

### First-Last-Frame Interpolation (FLF2V)

```bash
python vace_video_generation.py \
  --mode flf2v \
  --image first_frame.jpg --last-image last_frame.jpg \
  --prompt "A bird takes off from a branch and lands on another" \
  --height 512 --width 512 --num-frames 81 \
  --output flf2v_output.mp4
```

### Inpainting

Center vertical stripe is masked and regenerated:

```bash
python vace_video_generation.py \
  --mode inpaint \
  --image scene.jpg \
  --prompt "Shrek walks out of a building" \
  --height 480 --width 832 --num-frames 81 \
  --output inpaint_output.mp4
```

### Reference Image-guided (R2V)

```bash
python vace_video_generation.py \
  --mode r2v \
  --image reference.jpg \
  --prompt "Camera slowly zooms out from the character" \
  --height 480 --width 832 --num-frames 81 \
  --output r2v_output.mp4
```
