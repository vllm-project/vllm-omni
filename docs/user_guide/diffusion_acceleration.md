# Diffusion Acceleration Overview

vLLM-Omni supports various cache acceleration methods to speed up diffusion model inference with minimal quality degradation. These methods include **cache methods** that intelligently cache intermediate computations to avoid redundant work across diffusion timesteps, and **parallelism methods** that distribute the computation across multiple devices.

## Supported Acceleration Methods

vLLM-Omni currently supports two main cache acceleration backends:

1. **[TeaCache](diffusion/teacache.md)** - Hook-based adaptive caching that caches transformer computations when consecutive timesteps are similar
2. **[Cache-DiT](diffusion/cache_dit_acceleration.md)** - Library-based acceleration using multiple techniques:
    - **DBCache** (Dual Block Cache): Caches intermediate transformer block outputs based on residual differences
    - **TaylorSeer**: Uses Taylor expansion-based forecasting for faster inference
    - **SCM** (Step Computation Masking): Selectively computes steps based on adaptive masking

Both methods can provide significant speedups (typically **1.5x-2.0x**) while maintaining high output quality.

vLLM-Omni also supports parallelism methods for diffusion models, including:

1. [Ulysses-SP](diffusion/parallelism_acceleration.md#ulysses-sp) - splits the input along the sequence dimension and uses all-to-all communication to allow each device to compute only a subset of attention heads.

2. [Ring-Attention](diffusion/parallelism_acceleration.md#ring-attention) - splits the input along the sequence dimension and uses ring-based P2P communication to accumulate attention results, keeping the sequence dimension sharded.

3. [CFG-Parallel](diffusion/parallelism_acceleration.md#cfg-parallel) - runs the positive/negative prompts of classifier-free guidance (CFG) on different devices, then merges on a single device to perform the scheduler step.

## Quick Comparison

### Cache Methods

| Method | Configuration | Description | Best For |
|--------|--------------|-------------|----------|
| **TeaCache** | `cache_backend="tea_cache"` | Simple, adaptive caching with minimal configuration | Quick setup, balanced speed/quality |
| **Cache-DiT** | `cache_backend="cache_dit"` | Advanced caching with multiple techniques (DBCache, TaylorSeer, SCM) | Maximum acceleration, fine-grained control |

## Supported Models

The following table shows which models are currently supported by each acceleration method:

### ImageGen

| Model | Model Identifier | TeaCache | Cache-DiT | Ulysses-SP | Ring-Attention | CFG-Parallel |
|-------|------------------|:----------:|:-----------:|:-----------:|:----------------:|:----------------:|
| **LongCat-Image** | `meituan-longcat/LongCat-Image` | ❌ | ✅ | ❌ | ❌ | ❌ |
| **LongCat-Image-Edit** | `meituan-longcat/LongCat-Image-Edit` | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Ovis-Image** | `OvisAI/Ovis-Image` | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Qwen-Image** | `Qwen/Qwen-Image` | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Qwen-Image-2512** | `Qwen/Qwen-Image-2512` | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Qwen-Image-Edit** | `Qwen/Qwen-Image-Edit` | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Qwen-Image-Edit-2509** | `Qwen/Qwen-Image-Edit-2509` | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Qwen-Image-Layered** | `Qwen/Qwen-Image-Layered` | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Z-Image** | `Tongyi-MAI/Z-Image-Turbo` | ❌ | ✅ |❌ | ❌ | ❌ |
| **Stable-Diffusion3.5** | `stabilityai/stable-diffusion-3.5` | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Bagel** | `ByteDance-Seed/BAGEL-7B-MoT` | ❌ | ✅ | ❌ | ❌ | ❌ |

### VideoGen

| Model | Model Identifier | TeaCache | Cache-DiT | Ulysses-SP | Ring-Attention |
|-------|------------------|:--------:|:---------:|:----------:|:--------------:|
| **Wan2.2** | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | ❌ | ✅ | ❌ | ❌ |


## Performance Benchmarks

The following benchmarks were measured on **Qwen/Qwen-Image** and **Qwen/Qwen-Image-Edit** models generating 1024x1024 images with 50 inference steps:

!!! note "Benchmark Disclaimer"
    These benchmarks are provided for **general reference only**. The configurations shown use default or common parameter settings and have not been exhaustively optimized for maximum performance. Actual performance may vary based on:

    - Specific model and use case
    - Hardware configuration
    - Careful parameter tuning
    - Different inference settings (e.g., number of steps, image resolution)

    For optimal performance in your specific scenario, we recommend experimenting with different parameter configurations as described in the detailed guides below.

| Model | Cache Backend | Cache Config | Generation Time | Speedup | Notes |
|-------|---------------|--------------|----------------|---------|-------|
| **Qwen/Qwen-Image** | None  | None | 20.0s | 1.0x | Baseline (diffusers) |
| **Qwen/Qwen-Image** | TeaCache | `rel_l1_thresh=0.2` | 10.47s | **1.91x** | Recommended default setting |
| **Qwen/Qwen-Image** | Cache-DiT | DBCache + TaylorSeer (Fn=1, Bn=0, W=8, TaylorSeer order=1) | 10.8s | **1.85x** | - |
| **Qwen/Qwen-Image** | Cache-DiT | DBCache + TaylorSeer + SCM (Fn=8, Bn=0, W=4, TaylorSeer order=1, SCM fast) | 14.0s | **1.43x** | - |
| **Qwen/Qwen-Image-Edit** | None | No acceleration | 51.5s | 1.0x | Baseline (diffusers) |
| **Qwen/Qwen-Image-Edit** | Cache-DiT | Default (Fn=1, Bn=0, W=4, TaylorSeer disabled, SCM disabled) | 21.6s | **2.38x** | - |

To measure the parallelism methods, we run benchmarks with **Qwen/Qwen-Image** model generating images (**2048x2048** as long sequence input) with 50 inference steps. The hardware devices are NVIDIA H800 GPUs. `sdpa` is the attention backends.

| Configuration | Ulysses degree |Generation Time | Speedup |
|---------------|----------------|---------|---------|
| **Baseline (diffusers)** | - | 112.5s | 1.0x |
| Ulysses-SP  |  2  |  65.2s | 1.73x |
| Ulysses-SP  |  4  | 39.6s | 2.84x |
| Ulysses-SP  |  8  | 30.8s | 3.65x |

## Quick Start

### Using TeaCache

```python
from vllm_omni import Omni

omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2}  # Optional, defaults to 0.2
)

outputs = omni.generate(prompt="A cat sitting on a windowsill", num_inference_steps=50)
```

### Using Cache-DiT

```python
from vllm_omni import Omni

omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="cache_dit",
    cache_config={
        "Fn_compute_blocks": 1,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 8,
        "enable_taylorseer": True,
        "taylorseer_order": 1,
    }
)

outputs = omni.generate(prompt="A cat sitting on a windowsill", num_inference_steps=50)
```

### Using Ulysses-SP

Run text-to-image:
```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
ulysses_degree = 2

omni = Omni(
    model="Qwen/Qwen-Image",
    parallel_config=DiffusionParallelConfig(ulysses_degree=ulysses_degree)
)

outputs = omni.generate(prompt="A cat sitting on a windowsill", num_inference_steps=50, width=2048, height=2048)
```


Run image-to-image:
```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
ulysses_degree = 2

omni = Omni(
    model="Qwen/Qwen-Image-Edit",
    parallel_config=DiffusionParallelConfig(ulysses_degree=ulysses_degree)
)

outputs = omni.generate(prompt="turn this cat to a dog",
        pil_image=input_image, num_inference_steps=50)
```

### Using Ring-Attention

Run text-to-image:
```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
ring_degree = 2

omni = Omni(
    model="Qwen/Qwen-Image",
    parallel_config=DiffusionParallelConfig(ring_degree=2)
)

outputs = omni.generate(prompt="A cat sitting on a windowsill", num_inference_steps=50, width=2048, height=2048)
```

### Using CFG-Parallel

Run image-to-image:

CFG-Parallel splits the CFG positive/negative branches across GPUs. Use it when you set a non-trivial `true_cfg_scale`.

```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
cfg_parallel_size = 2

omni = Omni(
    model="Qwen/Qwen-Image-Edit",
    parallel_config=DiffusionParallelConfig(cfg_parallel_size=cfg_parallel_size)
)

outputs = omni.generate(prompt="turn this cat to a dog",
        pil_image=input_image, num_inference_steps=50, true_cfg_scale=4.0)
```

## Documentation

For detailed information on each acceleration method:

- **[TeaCache Guide](diffusion/teacache.md)** - Complete TeaCache documentation, configuration options, and best practices
- **[Cache-DiT Acceleration Guide](diffusion/cache_dit_acceleration.md)** - Comprehensive Cache-DiT guide covering DBCache, TaylorSeer, SCM, and configuration parameters
- **[Sequence Parallelism](diffusion/parallelism_acceleration.md#sequence-parallelism)** - Guidance on how to set sequence parallelism with configuration.
- **[CFG-Parallel](diffusion/parallelism_acceleration.md#cfg-parallel)** - Guidance on how to set CFG-Parallel to run positive/negative branches across ranks.
