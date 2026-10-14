# Pipeline Parallelism Guide

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

Pipeline Parallelism splits the denoising transformer block-wise into sequential stages across GPUs. Each rank owns only
part of the transformer, which reduces per-GPU model memory and enables larger diffusion models to run across multiple
devices.

It can also be combined with other distributed methods such as CFG-Parallel, Tensor Parallelism, and Sequence
Parallelism.

See supported models list in [Supported Models](../../diffusion_features.md#supported-models).

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
        pipeline_parallel_size=2,
    ),
)

outputs = omni.generate(
    {"prompt": "A cinematic drone shot over snowy mountains"},
    OmniDiffusionSamplingParams(
        num_inference_steps=40,
        num_frames=81,
        height=704,
        width=1280,
    ),
)
```

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
--output=t2v_5B_pp2.mp4 \
--pipeline-parallel-size=2
```

Pipeline Parallelism can also be combined with CFG-Parallel:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
--model=Wan-AI/Wan2.2-TI2V-5B-Diffusers \
--width=1280 \
--height=704 \
--guidance-scale=5.0 \
--prompt="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" \
--output=t2v_5B_pp2_cfg2.mp4 \
--pipeline-parallel-size=2 \
--cfg-parallel-size=2
```

### Online Serving

Enable Pipeline Parallelism in online serving:

```bash
# Default PP configuration
vllm serve Wan-AI/Wan2.2-TI2V-5B-Diffusers --omni --port 8091 --pipeline-parallel-size 2

# PP + CFG-Parallel
vllm serve Wan-AI/Wan2.2-TI2V-5B-Diffusers --omni --port 8091 \
  --pipeline-parallel-size 2 \
  --cfg-parallel-size 2
```

---

## Configuration Parameters

In `DiffusionParallelConfig`

| Parameter                | Type | Default | Description                                                                                                      |
|--------------------------|------|---------|------------------------------------------------------------------------------------------------------------------|
| `pipeline_parallel_size` | int  | 1       | Number of pipeline-parallel stages. Set to a value greater than 1 to split the denoising transformer across GPUs |

> [!NOTE]
> Total GPU count is the product of all enabled distributed dimensions, for example
`pipeline_parallel_size * cfg_parallel_size * tensor_parallel_size * ulysses_degree * ring_degree`.

### Manual Layer Partitioning

By default, transformer layers are distributed evenly across PP ranks. You can override this with the
`VLLM_PP_LAYER_PARTITION` environment variable to assign a specific number of layers to each rank:

```bash
# Example: 40 layers across 4 PP ranks, assigning 8 / 12 / 12 / 8 layers
export VLLM_PP_LAYER_PARTITION=8,12,12,8
```

The value must be a comma-separated list of integers whose length equals `pipeline_parallel_size` and whose sum equals
the total number of transformer layers. This is useful when you want to balance memory or compute asymmetrically across
ranks.


---

## Best Practices

### When to Use

**Good for:**

- Large diffusion transformers that do not fit comfortably on one GPU
- Multi-GPU setups where reducing per-GPU model memory is more important than minimizing communication
- Combining with CFG-Parallel or other distributed methods on supported models

**Not for:**

- Single GPU setups
- Models that do not support Pipeline Parallelism
  (check [supported models](../../diffusion_features.md#supported-models))
- Very small models where inter-stage communication overhead may outweigh the benefit

### Expected Behavior

Pipeline Parallelism primarily reduces per-GPU memory usage by splitting the transformer into stage-local blocks.
Depending on the model, topology, and resolution, it may also help execution fit into available hardware, but it is not
primarily a latency optimization.

---

## Troubleshooting

### Common Issue 1: No benefit from Pipeline Parallelism

**Symptoms**: PP is enabled, but latency does not improve or becomes slightly worse.

**Solutions**:

1. **Check your goal:**

```python
# PP is mainly for memory scaling, not guaranteed latency speedup
parallel_config = DiffusionParallelConfig(pipeline_parallel_size=2)
```

2. **Check model support:**
    - Verify your model in [supported models](../../diffusion_features.md#supported-models)
    - PP is currently validated only on selected pipelines

3. **Combine with other methods when appropriate:**
    - PP can be combined with CFG-Parallel, Tensor Parallelism, or Sequence Parallelism on supported models

### Common Issue 2: PP pipeline fails at import

**Symptoms**: Importing a custom pipeline raises a `TypeError` about `CFGParallelMixin` or mixin order.

**Solutions**:

1. Inherit both mixins.
2. Put `PipelineParallelMixin` before `CFGParallelMixin` in the class base list.
3. Use `class YourPipeline(nn.Module, PipelineParallelMixin, CFGParallelMixin): ...` as the reference pattern.

---

## Summary

1. ✅ **Enable Pipeline Parallelism** - Set `pipeline_parallel_size > 1` in `DiffusionParallelConfig`
2. ✅ **Use Supported Models** - Verify your model supports PP in
   [supported models](../../diffusion_features.md#supported-models)
3. ✅ **Combine When Needed** - PP can be combined with CFG-Parallel and other distributed methods on supported pipelines
4. ✅ **Scale for Memory** - Use PP primarily to reduce per-GPU model memory and fit larger transformers
