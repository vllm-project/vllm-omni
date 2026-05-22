# PipeFusion Guide

## Table of Content

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Example Script](#example-script)
- [Configuration Parameters](#configuration-parameters)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Summary](#summary)

---

## Overview

PipeFusion is a latency-optimized parallelism method that enhances Pipeline Parallelism (PP).
It reduces pipeline bubbles by splitting the denoising sequence into patches, allowing GPU stages to
overlap communication for one patch with computation for the next. This approach improves
throughput and reduces end-to-end latency for large-scale generation tasks.

The first warmup step(s) run with normal Pipeline Parallelism. After warmup, PipeFusion processes
spatial or temporal patches through the same PP stages with asynchronous communication. This
reduces PP bubble time for large generations where each denoising step has enough work to overlap.
PipeFusion is a lossy acceleration method: it is designed to preserve output quality in practice, but it
is not mathematically equivalent to plain Pipeline Parallelism.

---

## Quick Start

### Basic Usage

Simplest working example:

```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    parallel_config=DiffusionParallelConfig(
        pipeline_parallel_size=4,
        enable_pipefusion=True,
        vae_patch_parallel_size=4,
    ),
)

outputs = omni.generate(
    {"prompt": "A cinematic drone shot over snowy mountains"},
    OmniDiffusionSamplingParams(
        num_inference_steps=40,
        num_frames=81,
        height=704,
        width=1280,
        pipefusion_warmup_steps=1,
        pipefusion_split_dim="height",
    ),
)
```

PipeFusion requires `pipeline_parallel_size > 1`.

---

## Example Script

### Offline Inference

Use python scripts under:

- `examples/offline_inference/text_to_video/text_to_video.py`
- `examples/offline_inference/image_to_video/image_to_video.py`

Text-to-video example:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
--model=Wan-AI/Wan2.2-TI2V-5B-Diffusers \
--width=1280 \
--height=704 \
--guidance-scale=5.0 \
--prompt="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" \
--output=t2v_5B_pp4_pf.mp4 \
--pipeline-parallel-size=4 \
--enable-pipefusion \
--pipefusion-warmup-steps=5 \
--pipefusion-split-dim=temporal \
--vae-patch-parallel-size=4
```

PipeFusion can be combined with CFG-Parallel when the model supports both:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
--model=Wan-AI/Wan2.2-TI2V-5B-Diffusers \
--width=1280 \
--height=704 \
--guidance-scale=5.0 \
--prompt="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" \
--output=t2v_5B_pp2_cfg2_pf.mp4 \
--pipeline-parallel-size=2 \
--cfg-parallel-size=2 \
--enable-pipefusion \
--pipefusion-warmup-steps=5 \
--pipefusion-split-dim=temporal \
--vae-patch-parallel-size=4
```

### Online Serving

Enable PipeFusion in online serving:

```bash
vllm serve Wan-AI/Wan2.2-TI2V-5B-Diffusers --omni --port 8091 \
  --pipeline-parallel-size 4 \
  --enable-pipefusion \
  --vae-patch-parallel-size 4
```

Request-level parameters can be passed through the video generation request:

```bash
curl http://localhost:8091/v1/videos/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "prompt": "A cinematic drone shot over snowy mountains",
    "height": 704,
    "width": 1280,
    "num_frames": 81,
    "pipefusion_warmup_steps": 1,
    "pipefusion_split_dim": "height"
  }'
```

---

## Configuration Parameters

In `DiffusionParallelConfig`

| Parameter                | Type | Default | Description                                                                  |
|--------------------------|------|---------|------------------------------------------------------------------------------|
| `pipeline_parallel_size` | int  | 1       | Number of pipeline-parallel stages. Must be greater than 1 for PipeFusion    |
| `enable_pipefusion`      | bool | `False` | Enables PipeFusion patch-wise async execution on top of Pipeline Parallelism |

In `OmniDiffusionSamplingParams` or video generation requests

| Parameter                 | Type                                | Default | Description                                                                                                       |
|---------------------------|-------------------------------------|---------|-------------------------------------------------------------------------------------------------------------------|
| `pipefusion_warmup_steps` | int or `None`                       | `None`  | Number of initial denoising steps to run with standard PP before patch mode. `None` uses the runtime default of 1 |
| `pipefusion_split_dim`    | `"height"`, `"temporal"`, or `None` | `None`  | Latent dimension to split during patch mode. `None` uses the runtime default of `"height"`                        |

> [!NOTE]
> Total GPU count is the product of all enabled distributed dimensions, for example
> `pipeline_parallel_size * cfg_parallel_size * tensor_parallel_size * ulysses_degree * ring_degree`.

### Split Dimension

Use `height` for the default spatial split. Use `temporal` when you want patches to cover groups of frames instead of
height bands. The best split depends on the model and whether it is a video or image generation task.

### Warmup Steps

PipeFusion requires at least one warmup step. Warmup fills full-sequence attention and scheduler caches before async patch
execution begins. Increasing warmup steps can improve stability for some workloads but reduces the portion of denoising
that benefits from PipeFusion overlap.

---

## Best Practices

### When to Use

**Good for:**

- Supported large video diffusion pipelines with Pipeline Parallelism enabled
- Multi-GPU setups where plain PP has noticeable pipeline bubbles
- Images and shorter videos, where repeated KV-cache reads are less likely to become a memory bottleneck
- PP combined with CFG-Parallel on supported models

**Not for:**

- Single GPU setups
- Models that do not support PipeFusion
- Large resolutions or long videos where the overhead of repeated KV-cache reads may outweigh the overlap benefits
- Workloads where the primary goal is only to reduce model memory; plain Pipeline Parallelism may be enough

### PipeFusion vs Pipeline Parallelism

Pipeline Parallelism is a lossless memory-scaling technique: each rank owns only a slice of the transformer. PipeFusion
keeps that memory benefit and tries to improve utilization by feeding patches through the PP stages during later denoising
steps, but this patch-wise async schedule is lossy and should be validated against a plain PP baseline for your workload.

Start with plain PP when you only need to fit the model. Add PipeFusion when PP works correctly and you want to reduce
idle time during denoising.

---

## Troubleshooting

### Common Issue 1: `enable_pipefusion=True` fails with one PP stage

**Symptoms**: Configuration validation raises an error about `pipeline_parallel_size`.

**Solutions**:

1. Set `pipeline_parallel_size` to a value greater than 1.

```python
parallel_config = DiffusionParallelConfig(
    pipeline_parallel_size=2,
    enable_pipefusion=True,
)
```

2. If you are on one GPU, disable PipeFusion.

### Common Issue 2: PipeFusion run hangs

**Symptoms**: The process stalls during denoising.

**Solutions**:

- Try plain Pipeline Parallelism first to isolate PP setup issues, as PipeFusion relies on PP for communication.

### Common Issue 3: Temporal split fails or gives unexpected quality

**Symptoms**: `pipefusion_split_dim="temporal"` behaves worse than `height`.

**Solutions**:

1. Verify the frame count is compatible with the model patch size and PP size.
2. Use the default `height` split unless temporal splitting is known to help your workload.

---

## Summary

1. Enable Pipeline Parallelism with `pipeline_parallel_size > 1`.
2. Enable PipeFusion with `enable_pipefusion=True`.
3. Tune `pipefusion_warmup_steps` and `pipefusion_split_dim` per workload.
4. Use PipeFusion when you want to reduce PP idle time, not just model memory.
