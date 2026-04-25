# Diffusion Advanced Features

## Table of Contents

- [Overview](#overview)
- [Supported Features](#supported-features)
- [Supported Models](#supported-models)
- [Feature Compatibility](#feature-compatibility)
- [Learn More](#learn-more)

## Overview

vLLM-Omni supports various advanced features for diffusion models:

- Acceleration: **cache methods**, **parallelism methods**, **startup optimizations**
- Memory optimization: **cpu offloading**, **quantization**
- Extensions: **LoRA inference**, **frame interpolation**
- Execution modes: **step execution**

## Supported Features

### Acceleration

#### Lossy Acceleration

Cache methods trade minimal quality for significant speedup. Quality loss is typically imperceptible with proper tuning.

| Method | Description | Best For |
|--------|-------------|----------|
| **[TeaCache](diffusion/cache_acceleration/teacache.md)** | Adaptive caching using modulated inputs | Quick setup, balanced quality/speed on single GPU |
| **[Cache-DiT](diffusion/cache_acceleration/cache_dit.md)** | Multiple caching techniques: DBCache, TaylorSeer, SCM | Fine-grained control, tunable quality-speed tradeoff |


#### Lossless Acceleration

Parallelism methods distribute computation across GPUs without quality loss (mathematically equivalent to single-GPU).

| Method | Description | Best For |
|--------|-------------|----------|
| **[Ulysses-SP](diffusion/parallelism/sequence_parallel.md)** | Sequence parallelism via all-to-all communication | High-resolution images (>1536px) or long videos with 2-8 GPUs |
| **[Ring-Attention](diffusion/parallelism/sequence_parallel.md)** | Sequence parallelism via ring-based communication | Videos, very long sequences, memory-constrained, with 2-8 GPUs |
| **[CFG-Parallel](diffusion/parallelism/cfg_parallel.md)** | Splits CFG positive/negative branches across devices | Image editing with CFG guidance (true_cfg_scale > 1) on 2 GPUs |
| **[Tensor Parallelism](diffusion/parallelism/tensor_parallel.md)** | Shards model weights across devices | Large models that don't fit in single GPU, with 2+ GPUs |
| **[HSDP](diffusion/parallelism/hsdp.md)** | Weight sharding via FSDP2, redistributed on-demand at runtime | Very large models (14B+) on limited VRAM, combinable with SP |
| **[Expert Parallelism](diffusion/parallelism/expert_parallel.md)** | Shards MoE expert MLP blocks across devices | MoE diffusion models (e.g., HunyuanImage3.0) |

#### Startup Optimization

| Method | Description | Best For |
|--------|-------------|----------|
| **[Multi-Thread Weight Loading](#multi-thread-weight-loading)** | Loads safetensors shards in parallel using a thread pool | All diffusion models; reduces startup from minutes to seconds |

**Note:** Some acceleration methods can be combined together for optimized performance. See [Feature Compatibility Table](#feature-compatibility) and [Feature Compatibility Tutorial](feature_compatibility.md) for detailed configuration examples.

### Memory Optimization

Memory optimization methods help reduce GPU memory usage, enabling inference on resource-constrained hardware or larger models.

| Method | Description | Best For |
|--------|-------------|----------|
| **[CPU Offload](diffusion/cpu_offload_diffusion.md)** | Offloads model components to CPU memory | Limited VRAM, large models on consumer GPUs |
| **[Quantization](diffusion/quantization/overview.md)** | Reduces DiT layers from BF16 to FP8/INT8/etc. | Limited VRAM, minimal accuracy loss    |
| **[VAE Patch Parallelism](diffusion/parallelism/vae_patch_parallel.md)** | Distributes VAE decode tiling across GPUs | High-resolution generation with reduced VAE memory peak |

### Extensions

Extension methods add specialized capabilities to diffusion models beyond standard inference.

| Method | Description | Best For |
|--------|-------------|----------|
| **[LoRA Inference](diffusion/lora.md)** | Enables inference with Low-Rank Adaptation (LoRA) adapters weights | Reinforcement learning extensions |
| **[Frame Interpolation](diffusion/frame_interpolation.md)** | Inserts intermediate video frames after generation for smoother motion | Video generation pipelines that need higher temporal smoothness |


### Execution Modes

Execution modes control how the diffusion pipeline processes denoise steps.

| Method | Description | Best For |
|--------|-------------|----------|
| **[Step Execution](diffusion/step_execution.md)** | Per-step denoise execution with mid-request abort support | Request cancellation between denoise steps, fine-grained execution control |

**Note:** Step execution is currently supported by QwenImagePipeline only. See [Supported Models](#supported-models) for details.

### Quantization Methods

| Method | Configuration | Description | Best For |
|--------|--------------|-------------|----------|
| **[FP8](diffusion/quantization/fp8.md)** | `quantization="fp8"` | FP8 W8A8 on Ada/Hopper, weight-only on older GPUs | Memory reduction, inference speedup |
| **[INT8](diffusion/quantization/int8.md)** | `quantization="int8"` | INT8 weight-only, no calibration or pre-quantized checkpoint needed | Memory reduction, broad GPU compatibility |
| **[GGUF](diffusion/quantization/gguf.md)** | `quantization="gguf"` | Native GGUF transformer-only weights (Q4, Q8, etc.) | Memory reduction on consumer GPUs |

## Supported Models

The following tables show which models support each feature:

- **🔀SP (Ulysses & Ring)**: Includes both Ulysses-SP and Ring-Attention methods
- ✅ = Fully supported
- ❌ = Not supported

> Notes:

> 1. CPU Offload has two methods: Module-wise (default for models with DiT + text encoder) and Layerwise. The tables below show **Layerwise support** only.
> 2. The **💾Quantization** column is collapsed for readability. See [Quantization Overview](diffusion/quantization/overview.md) for per-method (FP8, GGUF, …) and per-model support details.

### ImageGen

| Model | ⚡TeaCache | ⚡Cache-DiT | 🔀SP (Ulysses & Ring) | 🔀CFG-Parallel | 🔀Tensor-Parallel | 🔀HSDP | 💾CPU Offload (Layerwise) | 💾VAE-Patch-Parallel | 💾Quantization | 🔄Step Execution |
|-------|:----------:|:-----------:|:---------------------:|:--------------:|:-----------------:|:------:|:------------------------:|:--------------------:|:--------------:|:----------------:|
| **Bagel** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅  | ❌ | ❌ | ❌ |
| **FLUX.1-dev** | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| **FLUX.1-schnell** | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| **FLUX.2-klein** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| **FLUX.1-Kontext-dev** | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **FLUX.2-dev** | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **GLM-Image** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **HunyuanImage3** | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **LongCat-Image** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **LongCat-Image-Edit** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **MagiHuman** | ❌ | ❌ | ❌ | ❓ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **MammothModa2(T2I)** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Nextstep_1(T2I)** | ❓ | ❓ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **OmniGen2** | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Ovis-Image** | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Qwen-Image** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (decode) | ✅ | ✅ |
| **Qwen-Image-2512** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (decode) | ✅ | ✅ |
| **Qwen-Image-Edit** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (decode) | ❌ | ❌ |
| **Qwen-Image-Edit-2509** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (decode) | ✅ | ❌ | ❌ |
| **Qwen-Image-Layered** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (decode) | ❌ | ❌ |
| **Stable-Diffusion3.5** | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ (decode) | ❌ | ❌ |
| **Z-Image** | ✅ | ✅ | ✅ | ❓ | ✅ (TP=2 only) | ✅ | ❌ | ✅ (decode) | ✅ | ❌ |

> Notes:
> 1. Nextstep_1(T2I) does not support cache acceleration methods such as TeaCache or Cache-DiT.
> 2. `Tongyi-MAI/Z-Image-Turbo` and `SII-GAIR/daVinci-MagiHuman-Base-1080p` are distilled models with minimal NFEs; CFG-Parallel is not necessary.

### VideoGen

| Model | ⚡TeaCache | ⚡Cache-DiT | 🔀SP (Ulysses & Ring) | 🔀CFG-Parallel | 🔀Tensor-Parallel | 🔀HSDP | 💾CPU Offload (Layerwise) | 💾VAE-Patch-Parallel | 💾Quantization | 🔄Step Execution |
|-------|:----------:|:-----------:|:---------------------:|:--------------:|:-----------------:|:------:|:------------------------:|:--------------------:|:--------------:|:----------------:|
| **Wan2.2** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (encode/decode) | ❌ | ❌ |
| **Wan2.1-VACE** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (decode) | ❌ | ❌ |
| **LTX-2** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **LTX-2.3** | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Helios** | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **HunyuanVideo-1.5 T2V I2V** | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ (decode) | ✅ | ❌ |
| **DreamID-Omni** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |

**Frame Interpolation Support**

- **Supported**: Wan2.2 text-to-video, image-to-video, and TI2V pipelines
- **Not supported**: Wan2.1-VACE, LTX-2, LTX-2.3, Helios, HunyuanVideo-1.5, DreamID-Omni

### AudioGen

| Model | ⚡TeaCache | ⚡Cache-DiT | 🔀SP (Ulysses & Ring) | 🔀CFG-Parallel | 🔀Tensor-Parallel | 🔀HSDP | 💾CPU Offload (Layerwise) | 💾VAE-Patch-Parallel | 💾Quantization | 🔄Step Execution |
|-------|:----------:|:-----------:|:---------------------:|:--------------:|:-----------------:|:------:|:------------------------:|:--------------------:|:--------------:|:----------------:|
| **Stable-Audio-Open** | ✅ | ❌ | ❓ | ❓ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ |


## Feature Compatibility

**Legend:**

- ✅: Functionality is supported
- ❌: No support plan
- ❓: Not verified yet and Not Recommended

|  | ⚡TeaCache | ⚡Cache-DiT | 🔀Ulysses-SP | 🔀Ring-Attn | 🔀CFG-Parallel | 🔀Tensor Parallel | 🔀HSDP | 🔀Expert Parallel | 💾CPU Offloading (Layerwise) | 💾CPU Offloading (Module-wise) | 💾VAE Patch Parallel | 💾FP8 Quant | 🔧LoRA Inference | 🔄Step Execution |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **⚡TeaCache** | | | | | | | | | | | | | | |
| **⚡Cache-DiT** | ❌ | | | | | | | | | | | | | |
| **🔀Ulysses-SP** | ✅ | ✅ | | | | | | | | | | | | |
| **🔀Ring-Attn** | ✅ | ✅ | ✅ | | | | | | | | | | | |
| **🔀CFG-Parallel** | ✅ | ✅ | ✅ | ✅ | | | | | | | | | | |
| **🔀Tensor Parallel** | ✅ | ✅ | ✅ | ✅ | ✅ | | | | | | | | | |
| **🔀HSDP** | ❓ | ❓ | ❓ | ❓ | ❓ | ❌ | | | | | | | | |
| **🔀Expert Parallel** | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | | | | | | | |
| **💾CPU Offloading (Layerwise)** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | | | | | | |
| **💾CPU Offloading (Module-wise)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | ❌ | | | | | |
| **💾VAE Patch Parallel** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | | | | |
| **💾FP8 Quant** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | ✅ | ✅ | ✅ | | | |
| **🔧LoRA Inference** | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | | |
| **🔄Step Execution** | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | ✅ | ❓ | ✅ | ✅ | ❌ | |

!!! info

    1. Tensor Parallel and HSDP are not compatible.
    2. TeaCache and Cache-DiT are not compatible.
    3. CPU Offloading (Layerwise) and CPU Offloading (Module-wise) are not compatible.
    4. CPU Offloading (Layerwise) supports single-card for now.
    5. Using FP8-Quant as an example of qunatization methods.
    6. Step Execution is not compatible with cache backends (TeaCache, Cache-DiT) or LoRA.


## Multi-Thread Weight Loading

Large diffusion models can take several minutes to load weights at startup (e.g., ~3 min for Qwen-Image, ~5 min for Wan2.2 I2V 14B). Multi-thread weight loading speeds up this process by loading safetensors shards in parallel using a thread pool instead of sequentially.

This optimization is **enabled by default** with 4 threads. No configuration is needed for the default behavior.

### Configuration

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| `enable_multithread_weight_load` | `--disable-multithread-weight-load` | `True` (enabled) | Pass the flag to disable multi-thread loading |
| `num_weight_load_threads` | `--num-weight-load-threads` | `4` | Number of threads for parallel weight loading |

!!! tip
    The default of 4 threads balances speed and disk I/O contention. On fast NVMe storage you may benefit from more threads (e.g., 8). On HDD or network storage, the default of 4 avoids saturating I/O bandwidth.

### Online Serving

```bash
# Default (multi-thread enabled, 4 threads)
vllm serve Qwen/Qwen-Image --omni --port 8091

# Custom thread count
vllm serve Wan-AI/Wan2.2-I2V-A14B-Diffusers --omni --num-weight-load-threads 8

# Disable multi-thread loading
vllm serve Qwen/Qwen-Image --omni --disable-multithread-weight-load
```

### Offline Inference

```python
from vllm_omni import Omni

# Default (multi-thread enabled, 4 threads)
omni = Omni(model="Qwen/Qwen-Image")

# Custom thread count
omni = Omni(
    model="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    num_weight_load_threads=8,
)
```

### Benchmarks

Measured on NVIDIA H800:

| Model | Before | After | Speedup |
|-------|--------|-------|---------|
| **Qwen/Qwen-Image** (53.7 GiB) | 168s | 27s | **6.2x** |
| **Wan-AI/Wan2.2-I2V-A14B-Diffusers** (64.5 GiB) | 283s | 56s | **5.1x** |

## Learn More

**Cache Acceleration:**

- **[TeaCache Configuration Guide](diffusion/cache_acceleration/teacache.md)** - Parameter tuning, performance tips, troubleshooting
- **[Cache-DiT Advanced Guide](diffusion/cache_acceleration/cache_dit.md)** - DBCache, TaylorSeer, SCM techniques and optimization

**Parallelism Methods:**

- **[Parallelism Overview](diffusion/parallelism/overview.md)** - Tensor Parallelism, Sequence Parallelism, CFG Parallelism, HSDP, and Expert Parallelism

**Memory Optimization:**

- **[CPU Offload Guide](diffusion/cpu_offload_diffusion.md)** - Offload model components to CPU, reduce GPU memory usage
- **[VAE Patch Parallelism Guide](diffusion/parallelism/vae_patch_parallel.md)** - Distribute VAE decode tiling across GPUs for high-resolution images
- **[Quantization Overview](diffusion/quantization/overview.md)** - Overview of quantization methods for diffusion models

**Extensions:**

- **[LoRA Inference Guide](diffusion/lora.md)** - Low-Rank Adaptation for style customization and fine-tuning
- **[Frame Interpolation Guide](diffusion/frame_interpolation.md)** - Worker-side post-generation video frame interpolation for smoother motion

**Execution Modes:**

- **[Step Execution Guide](diffusion/step_execution.md)** - Per-step denoise execution with mid-request abort support

**Startup Optimization:**

- **[Multi-Thread Weight Loading](#multi-thread-weight-loading)** - Speed up model startup by loading safetensors shards in parallel

**Advanced Topics:**

- **[Feature Compatibility](feature_compatibility.md)** - How to combine multiple features for maximum performance
