# HSDP Guide


## Table of Content

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Example Script](#example-script)
- [Configuration Parameters](#configuration-parameters)
- [Best Practices](#best-practices)
- [Summary](#summary)

---

## Overview

HSDP (Hybrid Sharded Data Parallel) shards model weights across GPUs to reduce per-GPU memory usage. This enables inference of large models (e.g., Wan2.2 14B) on GPUs with limited memory.

Unlike Tensor Parallelism which splits computation, HSDP uses PyTorch's FSDP2 to shard and redistribute weights at runtime. Each GPU only holds a fraction of the model weights, and weights are gathered on-demand during forward passes.

See supported models list in [Supported Models](../../diffusion_features.md#supported-models).

**Operating Modes:**

- **Standalone Mode**: HSDP alone without other parallelism. Must specify `hsdp_shard_size` explicitly.
- **Combined Mode**: HSDP overlays on top of other parallelism (Ulysses-SP, CFG-Parallel). HSDP dimensions must match world_size.

---

## Quick Start

### Basic Usage

Simplest working example (standalone HSDP, shard across 4 GPUs):

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.diffusion.data import DiffusionParallelConfig

omni = Omni(
    model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    parallel_config=DiffusionParallelConfig(
        use_hsdp=True,
        hsdp_shard_size=4,  # Shard across 4 GPUs
    ),
)

outputs = omni.generate(
    "A cat playing piano",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

### Combined with Sequence Parallel

```python
omni = Omni(
    model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    parallel_config=DiffusionParallelConfig(
        ulysses_degree=4,  # Sequence parallel
        use_hsdp=True,     # HSDP overlays on SP
    ),
)
```

---

## Example Script

### Offline Inference

Use Python script under `examples/offline_inference/image_to_video/`:

```bash
# Standalone HSDP: shard across 4 GPUs
python examples/offline_inference/image_to_video/image_to_video.py \
    --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --use-hsdp \
    --hsdp-shard-size 4

# Combined HSDP + Sequence Parallel
python examples/offline_inference/image_to_video/image_to_video.py \
    --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --ulysses-degree 4 \
    --use-hsdp
```

### Online Serving

**Standalone HSDP** (shard model across 4 GPUs):

```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni --port 8091 \
    --use-hsdp --hsdp-shard-size 4
```

**Combined with Sequence Parallel**:

```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni --port 8091 \
    --use-hsdp --usp 4
```

---

## Configuration Parameters

In `DiffusionParallelConfig`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_hsdp` | bool | False | Enable HSDP |
| `hsdp_shard_size` | int | -1 | Number of GPUs to shard weights across. `-1` = auto (requires other parallelism > 1) |
| `hsdp_replicate_size` | int | 1 | Number of replica groups. Each group holds a full sharded copy |

**Constraints:**

- `hsdp_replicate_size × hsdp_shard_size == world_size`
- HSDP cannot be used with Tensor Parallelism (`tensor_parallel_size` must be 1)

---

## Best Practices

### When to Use

**Good for:**

- Very large models (e.g., Wan2.2 14B)
- Multi-GPU setups where memory reduction is the primary goal
- Combining with Sequence Parallelism for large video models

**Not for:**

- Models that fit comfortably in single-GPU memory
- Use cases requiring Tensor Parallelism (HSDP and TP are mutually exclusive)

### Adding HSDP Support to New Models

For detailed instructions on adding HSDP support to new models, see the [HSDP Contributing Guide](../../../design/feature/hsdp.md).

---

## Summary

1. ✅ **Enable HSDP** - Set `use_hsdp=True` and `hsdp_shard_size` to reduce per-GPU memory for large models
2. ✅ **Combine with SP** - Use together with `ulysses_degree` for video models requiring both memory reduction and sequence parallelism
3. ⚠️ **Incompatible with TP** - `tensor_parallel_size` must be 1 when HSDP is enabled
