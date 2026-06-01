# Sequence Parallelism Guide


## Table of Content

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Example Script](#example-script)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Summary](#summary)

---

## Overview

Sequence parallelism splits the input along the sequence dimension across multiple GPUs, allowing each device to process only a portion of the sequence. vLLM-Omni provides 1.5x-3.6x speedup for large images and videos using DeepSpeed Ulysses, Ring-Attention, or hybrid approaches. Use sequence parallelism when generating high-resolution images/videos that don't fit on a single GPU or require faster inference.

See supported models list in [Diffusion Features - Supported Models](../../diffusion_features.md#supported-models).

**Supported Methods:**

- **DeepSpeed Ulysses Sequence Parallel (Ulysses-SP)** ([paper](https://arxiv.org/pdf/2309.14509)): Uses all-to-all communication for subset of attention heads per device
- **Ring-Attention** ([paper](https://arxiv.org/abs/2310.01889)): Uses ring-based P2P communication with sharded sequence dimension throughout
- **Hybrid Ulysses + Ring**: Combines both for larger scale parallelism (`ulysses_degree × ring_degree`)

---

## Quick Start

### Basic Usage - Ulysses-SP

Simplest working example with Ulysses Sequence Parallel:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.diffusion.data import DiffusionParallelConfig

omni = Omni(
    model="Qwen/Qwen-Image",
    parallel_config=DiffusionParallelConfig(ulysses_degree=2)  # Enable Ulysses-SP
)

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50, width=1024, height=1024),
)
```

!!! note "Experimental UAA mode"
    `ulysses_mode="advanced_uaa"` is an experimental extension to Ulysses-SP. It lets Ulysses attention handle arbitrary sequence lengths and arbitrary attention head counts without relying on `attention_mask`-based token padding.

    In hybrid Ulysses + Ring mode, Ring still requires every rank in the same ring group to observe the same post-Ulysses sequence length. If that condition is not met, vLLM-Omni raises a validation error instead of entering the ring kernel with inconsistent shapes.

To enable the experimental UAA mode, use a model/configuration that requires it. For example, `Tongyi-MAI/Z-Image-Turbo` has 30 attention heads, so `ulysses_degree=4` requires UAA because 30 is not divisible by 4:

```python
omni = Omni(
    model="Tongyi-MAI/Z-Image-Turbo",
    parallel_config=DiffusionParallelConfig(
        ulysses_degree=4,
        ulysses_mode="advanced_uaa",
    ),
)
```

### Alternative Methods

**Ring-Attention** (better for very long sequences):

```python
omni = Omni(
    model="Qwen/Qwen-Image",
    parallel_config=DiffusionParallelConfig(ring_degree=2)  # Enable Ring-Attention
)
```

**Hybrid Ulysses + Ring** (for larger scale):

```python
omni = Omni(
    model="Qwen/Qwen-Image",
    parallel_config=DiffusionParallelConfig(ulysses_degree=2, ring_degree=2)  # 4 GPUs total
)
```

---

## Example Script

### Offline Inference

Use Python script under `examples/offline_inference/text_to_image/text_to_image.py`:

**Ulysses-SP:**

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "A cat sitting on a windowsill" \
    --ulysses-degree 2 \
    --width 1024 --height 1024
```

**Ring-Attention:**

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "A cat sitting on a windowsill" \
    --ring-degree 2 \
    --width 1024 --height 1024
```

**Hybrid Ulysses + Ring:**

```bash
# Hybrid: 2 Ulysses × 2 Ring = 4 GPUs total
python examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "A cat sitting on a windowsill" \
    --ulysses-degree 2 --ring-degree 2 \
    --width 1024 --height 1024
```

### Online Serving

**Ulysses-SP:**

```bash
# Text-to-image (requires >= 2 GPUs)
vllm serve Qwen/Qwen-Image --omni --port 8000 --usp 2
```

**Ulysses-SP with UAA mode** (for models with non-divisible head counts):

```bash
vllm serve Tongyi-MAI/Z-Image-Turbo --omni --port 8000 --usp 4 --ulysses-mode advanced_uaa
```

**Ring-Attention:**

```bash
# Text-to-image (requires >= 2 GPUs)
vllm serve Qwen/Qwen-Image --omni --port 8000 --ring 2
```

**Hybrid Ulysses + Ring:**

```bash
# Text-to-image (requires >= 4 GPUs)
vllm serve Qwen/Qwen-Image --omni --port 8000 --usp 2 --ring 2
```

---

## Configuration Parameters

In `DiffusionParallelConfig`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ulysses_degree` | int | 1 | Number of GPUs for Ulysses-SP. Uses all-to-all communication. |
| `ring_degree` | int | 1 | Number of GPUs for Ring-Attention. Uses P2P ring communication. |
| `ulysses_mode` | str | `"default"` | Ulysses attention mode. Set to `"advanced_uaa"` to handle arbitrary sequence lengths and head counts without padding. |

**Notes:**
- Total sequence parallel size equals to `ulysses_degree × ring_degree`
- Degrees must evenly divide the sequence length for optimal performance (or use `ulysses_mode="advanced_uaa"` for Ulysses-SP)


## Best Practices

### When to Use

**Good for:**

- Large images (1024x1024 or higher) or videos
- Fast inter-GPU communication, larger bandwidth (e.g., NVLink)

**Not for:**

- Small images (<1024px) - overhead exceeds benefit, use single GPU with cache instead


---

## Troubleshooting

### Common Issue 1: Performance Not Scaling

**Symptoms**: Adding GPUs doesn't improve speed proportionally, or higher parallelism degree is slower

**Diagnosis:**
```bash
# Check GPU topology
nvidia-smi topo -m

```

**Solutions:**

1. Check inter-GPU communication - NVLink is better than PCIe
2. Reduce parallelism degree if over-parallelized:
```python
# If 4 GPUs is slower than 2
parallel_config=DiffusionParallelConfig(ulysses_degree=2)
```
3. Try to switch between Ring-Attention and Ulysses-SP

- Ring-Attention has advantages, like communication-computation overlap, but the block-wise loop overhead is relatively higher, especially for short sequences
- Ulysses-SP: can benefit from larger bandwidth (such as NVLink), with two major constraints, the sequence length should be divisible by usp size, and the number of heads should be divisible by usp size (or use `ulysses_mode="advanced_uaa"`)


### Common Issue 2: Out of Memory (OOM)

**Symptoms**: CUDA OOM errors or process crashes with memory errors

**Solutions:**

1. Increase parallelism degree to split sequence more:
```python
parallel_config=DiffusionParallelConfig(ulysses_degree=4)  # From 2
```
2. Combine with other parallelism method, e.g., tensor parallel, and memory optimization methods, e.g., cpu offloading.


## Summary

1. ✅ **Enable Sequence Parallelism** - Set `ulysses_degree` or `ring_degree` for long sequence generation
2. ✅ **UAA mode** - Use `ulysses_mode="advanced_uaa"` when head count is not divisible by `ulysses_degree`
3. ✅ **Troubleshooting** - Check GPU topology with `nvidia-smi topo -m`, reduce degree if performance doesn't scale
