# Tensor Parallelism Guide


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

Tensor Parallelism (TP) shards some model weights across multiple GPUs, usually the Linear layers. This enables running large models that don't fit on a single GPU. It's essential for memory-constrained setups or very large models.

See supported models list in [Supported Models](../../diffusion_features.md#supported-models).

!!! note "TP Limitations for Diffusion Models"
    We currently implement Tensor Parallelism (TP) only for the DiT (Diffusion Transformer) blocks. This is because the `text_encoder` component in vLLM-Omni uses the original Transformers implementation, which does not yet support TP.

    - Good news: The text_encoder typically has minimal impact on overall inference performance.
    - Bad news: When TP is enabled, every TP process retains a full copy of the text_encoder weights, leading to significant GPU memory waste.

    We are actively refactoring this design to address this. For details and progress, please refer to [Issue #771](https://github.com/vllm-project/vllm-omni/issues/771).

---

## Quick Start


### Basic Usage

Simplest working example:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.diffusion.data import DiffusionParallelConfig

omni = Omni(
    model="Tongyi-MAI/Z-Image-Turbo",
    parallel_config=DiffusionParallelConfig(tensor_parallel_size=2),  # Enable TP
)

outputs = omni.generate(
    "a cat reading a book",
    OmniDiffusionSamplingParams(num_inference_steps=9),
)
```

---

## Example Script

### Offline Inference

Use Python script under `examples/offline_inference`, and enable TP:

```bash
# Text-to-Image with Qwen-Image
python examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --tensor-parallel-size 2

# Image Editing with Qwen-Image-Edit
python examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Edit \
    --image input.png \
    --prompt "Edit description" \
    --tensor-parallel-size 2
```

### Online Serving

You can enable tensor parallelism in online serving via `--tensor-parallel-size`:

```bash
# Text-to-Image with Qwen-Image on 2 GPUs
vllm serve Qwen/Qwen-Image --omni --port 8000 \
    --tensor-parallel-size 2

# Text-to-Image with Z-Image (TP=2 only)
vllm serve Tongyi-MAI/Z-Image-Turbo --omni --port 8000 \
    --tensor-parallel-size 2
```

---

## Configuration Parameters

In `DiffusionParallelConfig`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor_parallel_size` | int | 1 | Number of GPUs to shard model weights across. Must divide number of heads. |


---

## Best Practices

### When to Use

**Good for:**

- Large models that don't fit on a single GPU, especially for models with large DiT blocks (transformer layers)
- Memory-constrained environments

**Not for:**

- When maximum throughput is needed and memory is sufficient
- Models with incompatible dimensions (e.g., Z-Image `num_heads=30`, which now supports `tensor_parallel_size=2`)


## Troubleshooting

### Common Issue 1: Out of Memory (OOM)

**Symptoms**: CUDA OOM errors during model loading or inference, process crashes with memory errors

**Solution**:
```python
# Step 1: Enable TP with smallest degree
parallel_config=DiffusionParallelConfig(tensor_parallel_size=2)

# Step 2: If still OOM, increase TP degree
parallel_config=DiffusionParallelConfig(tensor_parallel_size=4)

```

### Common Issue 2: Divisibility Error

**Symptoms**: Error like "Model dimension X not divisible by tensor_parallel_size Y"

**Solutions**:
1. Check model-specific constraints (e.g., Z-Image only supports TP=2)
2. Use a smaller TP size that divides model dimensions
3. Consult [Supported Models](../../diffusion_features.md#supported-models) for compatible TP sizes


---

## Summary

1. ✅ **Enable TP** - Set `--tensor-parallel-size` to reduce per-GPU memory
2. ✅ **Increase TP size** - Only increase if OOM persists
3. ⚠️ **Text encoder not sharded** - Known limitation
