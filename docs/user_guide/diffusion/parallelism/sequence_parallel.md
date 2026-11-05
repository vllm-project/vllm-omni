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
vllm serve Qwen/Qwen-Image --omni --port 8091 --usp 2
```

**Ulysses-SP with UAA mode** (for models with non-divisible head counts):

```bash
vllm serve Tongyi-MAI/Z-Image-Turbo --omni --port 8091 --usp 4 --ulysses-mode advanced_uaa
```

**Ring-Attention:**

```bash
# Text-to-image (requires >= 2 GPUs)
vllm serve Qwen/Qwen-Image --omni --port 8091 --ring 2
```

**Hybrid Ulysses + Ring:**

```bash
# Text-to-image (requires >= 4 GPUs)
vllm serve Qwen/Qwen-Image --omni --port 8091 --usp 2 --ring 2
```

---

## Configuration Parameters

In `DiffusionParallelConfig`:

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `ulysses_degree` | int | 1 | Number of GPUs for Ulysses-SP. Uses all-to-all communication. |
| `ring_degree` | int | 1 | Number of GPUs for Ring-Attention. Uses P2P ring communication. |
| `ulysses_mode` | str | `"strict"` | Ulysses attention mode. Set to `"advanced_uaa"` to handle arbitrary sequence lengths and head counts without padding. |
| `ulysses_a2a_backend` | str or None | `None` | Strict Ulysses exchange backend: `nccl`, `lsa` (SM-based fused transport), or `ce` (NVLink copy engines). `ce` requires the matching optional `fast-ulysses` wheel. |
| `mask_sp_padding` | bool | `False` | When the sequence length is not divisible by the SP size, tokens are auto-padded with zeros. Set to `True` to mask those padding tokens (strict, but uses the slower varlen attention path); the default `False` leaves them unmasked, keeping the fast path with negligible numerical impact. |

**Notes:**

- Total sequence parallel size equals to `ulysses_degree × ring_degree`
- Degrees must evenly divide the sequence length for optimal performance (or use `ulysses_mode="advanced_uaa"` for Ulysses-SP)
- `mask_sp_padding` is an experimental feature, currently only supported by `Wan2.2`, `Wan2.2 Vace`, `Qwen-Image`, `Flux 2`, and `HunyuanVideo 1.5`
- `ulysses_a2a_backend="ce"` is for one NVLink-connected node. Install the `fast-ulysses` wheel matching the installed PyTorch and CUDA versions before starting workers; for CUDA 13, use `pip install fast-ulysses --index-url https://triple-mu.github.io/fast-ulysses/whl/cu13/` (use the `cu12/` index for CUDA 12). It is selected in the Omni attention path and does not require a worker extension.

The explicit backend takes precedence over the legacy `ulysses_a2a_permute` flag.
Without an explicit backend, `ulysses_a2a_permute=True` selects LSA and the default
selects NCCL. Explicit LSA and CE require `ulysses_degree > 1` and
`ulysses_mode="strict"`. CE initialization checks that each Ulysses subgroup is
on one host; the `fast-ulysses` runtime then requires that host's GPUs to be
connected by the supported NVLink topology and raises an error if it cannot
create the copy-engine group.

Select CE for offline inference with:

```python
parallel_config = DiffusionParallelConfig(
    ulysses_degree=4,
    ulysses_a2a_backend="ce",
)
omni = Omni(model="Qwen/Qwen-Image", parallel_config=parallel_config)
```

The same option is available to the serving CLI and to stage deploy configs:

```bash
vllm serve Qwen/Qwen-Image --omni --ulysses-degree 4 --ulysses-a2a-backend ce
```

Workers initialize CE on their Ulysses process subgroup before loading the model,
reuse its symmetric-memory resources for Q/K/V and output exchanges, and release
them before destroying process groups at shutdown. In eager execution, CE uses
distinct reusable symmetric output buffers for Q, K, V, and the reverse exchange
to avoid copying each result out of the CE window. A cache miss during CUDA Graph
capture or `torch.compile` uses the functional API instead, preserving output
ownership and compiler alias semantics. End-to-end speedups depend on the model
and workload.

### Qwen-Image performance example

We measured the complete Qwen-Image generation path with a 1536×1536 image, 20
steps, BF16, FlashAttention, eager execution, Ulysses degree 2, and two
NVLink-connected L20X GPUs. The measurement included text encoding, denoising,
VAE decode, and CPU image return, while excluding model loading and warmup. LSA
took 10.294 s and CE took 10.097 s in the trace-validated run, a reduction of
about **1.9%**. The generated image pixels were bit-identical.

The trace explains where that gain comes from: the complete communication
window fell from 1219 ms with LSA to 976 ms with CE (about 20%), while the rest
of the GPU work stayed similar. This version of Ulysses still waits for Q/K/V
exchange before attention and for attention before the reverse exchange, so the
measurement does not include communication-compute overlap. A future sharded
attention implementation can use CE's independent copy engines with double
buffering or chunked execution to overlap transfers with independent attention
work; that is an expected follow-up opportunity rather than a claim about the
current measurement.

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
