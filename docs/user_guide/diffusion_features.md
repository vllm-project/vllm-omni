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

| Method                                                                 | Description                                                              | Best For                                                          |
|------------------------------------------------------------------------|--------------------------------------------------------------------------|-------------------------------------------------------------------|
| **[Ulysses-SP](diffusion/parallelism/sequence_parallel.md)**           | Sequence parallelism via all-to-all communication                        | High-resolution images (>1536px) or long videos with 2-8 GPUs     |
| **[Ring-Attention](diffusion/parallelism/sequence_parallel.md)**       | Sequence parallelism via ring-based communication                        | Videos, very long sequences, memory-constrained, with 2-8 GPUs    |
| **[CFG-Parallel](diffusion/parallelism/cfg_parallel.md)**              | Splits CFG positive/negative branches across devices                     | Image editing with CFG guidance (true_cfg_scale > 1) on 2 GPUs    |
| **[Tensor Parallelism](diffusion/parallelism/tensor_parallel.md)**     | Shards model weights across devices                                      | Large models that don't fit in single GPU, with 2+ GPUs           |
| **[Pipeline Parallelism](diffusion/parallelism/pipeline_parallel.md)** | Splits the denoising transformer block-wise across sequential GPU stages | Large diffusion transformers that need lower per-GPU model memory |
| **[HSDP](diffusion/parallelism/hsdp.md)**                              | Weight sharding via FSDP2, redistributed on-demand at runtime            | Very large models (14B+) on limited VRAM, combinable with SP      |
| **[Expert Parallelism](diffusion/parallelism/expert_parallel.md)**     | Shards MoE expert MLP blocks across devices                              | MoE diffusion models (e.g., HunyuanImage3.0)                      |

#### Startup Optimization

| Method | Description | Best For |
|--------|-------------|----------|
| **[Multi-Thread Weight Loading](diffusion/startup_and_loading.md)** | Loads safetensors shards in parallel using a thread pool | All diffusion models; reduces startup from minutes to seconds |

**Note:** Some acceleration methods can be combined together for optimized performance. See [Feature Compatibility Table](#feature-compatibility) and [Feature Compatibility Tutorial](feature_compatibility.md) for detailed configuration examples.

### Memory Optimization

Memory optimization methods help reduce GPU memory usage, enabling inference on resource-constrained hardware or larger models.

| Method | Description | Best For |
|--------|-------------|----------|
| **[CPU Offload](diffusion/cpu_offload.md)** | Offloads model components to CPU memory | Limited VRAM, large models on consumer GPUs |
| **[Quantization](quantization/overview.md)** | Reduces transformer stages from BF16 to FP8/INT8/etc. | Limited VRAM, minimal accuracy loss    |
| **[VAE Parallelism](diffusion/parallelism/vae_parallelism.md)** | Distributes VAE decode work across GPUs | High-resolution generation with reduced VAE memory peak |

### Extensions

Extension methods add specialized capabilities to diffusion models beyond standard inference.

| Method | Description | Best For |
|--------|-------------|----------|
| **[LoRA Inference](diffusion/lora.md)** | Enables inference with Low-Rank Adaptation (LoRA) adapters weights | Reinforcement learning extensions |
| **[Frame Interpolation](diffusion/frame_interpolation.md)** | Inserts intermediate video frames after generation for smoother motion | Video generation pipelines that need higher temporal smoothness |


### Execution Modes

Execution modes control how the diffusion pipeline processes requests and
denoise steps.

| Method | Description | Best For |
|--------|-------------|----------|
| **[Diffusion Execution Modes](diffusion/execution_modes.md)** | Configures serial requests, request batching, step execution, continuous batching, and streaming output | Matching latency, throughput, cancellation, and output-delivery requirements |

**Note:** Request-level batching and step execution are capability-based.
Consult the execution guide and the selected pipeline's documentation for
current support.

### Quantization Methods

| Method | Configuration | Description | Best For |
|--------|--------------|-------------|----------|
| **[FP8](quantization/fp8.md)** | `quantization="fp8"` | FP8 W8A8 on validated transformer stages | Memory reduction, inference speedup |
| **[INT8](quantization/int8.md)** | `quantization="int8"` | INT8 W8A8 on validated transformer stages | Memory reduction, broad GPU compatibility |
| **[GGUF](quantization/gguf.md)** | `quantization="gguf"` | Native GGUF transformer-only weights (Q4, Q8, etc.) | Memory reduction on consumer GPUs |

## Supported Models

The following tables show which models support each feature:

- **🔀SP (Ulysses & Ring)**: Includes both Ulysses-SP and Ring-Attention methods
- ✅ = Fully supported
- ✅* = Supported with the constraint listed below the table
- ❌ = Not supported
- ❓ = Not verified; not recommended

> Notes:

> 1. CPU Offload has three strategies: model-level, layerwise, and distributed
>    layerwise. The tables below show **layerwise support** only. Split models
>    like Cosmos3 swap their reasoner/generator components for model-level
>    offload; see the [CPU Offload Guide](diffusion/cpu_offload.md).
> 2. The **💾Quantization** column is collapsed for readability. See [Quantization](quantization/overview.md) for per-method and per-model support details.

### ImageGen

| Model                    | ⚡TeaCache | ⚡Cache-DiT | 🔀SP (Ulysses & Ring) | 🔀CFG-Parallel | 🔀Tensor-Parallel | 🔀Pipeline-Parallel | 🔀HSDP | 💾CPU Offload (Layerwise) | 💾VAE-Patch-Parallel | 💾Quantization | 🔄Step Execution |
|--------------------------|:---------:|:----------:|:---------------------:|:--------------:|:-----------------:|:-------------------:|:------:|:-------------------------:|:--------------------:|:--------------:|:----------------:|
| **Bagel**                |     ✅     |     ✅      |           ✅           |       ✅        |         ✅         |          ❌          |   ✅    |             ✅             |          ❌           |       ❌        |        ❌         |
| **FLUX.1-dev**           |     ✅     |     ✅      |           ❌           |       ✅        |         ✅         |          ❌          |   ✅    |             ❌             |          ❌           |       ✅        |        ❌         |
| **FLUX.1-schnell**       |     ❌     |     ✅      |           ❌           |       ✅        |         ✅         |          ❌          |   ✅    |             ❌             |          ❌           |       ✅        |        ❌         |
| **FLUX.2-klein**         |     ✅     |     ✅      |           ✅           |       ✅        |         ✅         |          ❌          |   ✅    |             ❌             |          ❌           |       ✅        |        ❌         |
| **FLUX.1-Kontext-dev**   |     ❌     |     ✅      |           ❌           |       ❌        |         ✅         |          ❌          |   ✅    |             ❌             |          ❌           |       ❌        |        ❌         |
| **FLUX.2-dev**           |     ✅     |     ✅      |           ✅           |       ✅        |         ✅         |          ❌          |   ✅    |             ✅             |          ✅           |       ❌        |        ❌         |
| **GLM-Image**            |     ❌     |     ❌      |           ❌           |       ✅        |         ✅         |          ❌          |   ✅    |             ❌             |          ❌           |       ❌        |        ❌         |
| **Hidream-I1-Full**        |     ❌     |     ❌      |           ❌           |       ❌        |         ✅         |          ❌          |   ❌    |             ❌             |          ❌           |       ❌        |        ❌         |
| **HiDream-O1-Image**     |     ❌     |     ✅      |           ❌           |       ❌        |         ✅         |          ❌          |   ❌    |             ❌             |          ❌           |       ❌        |        ❌         |
| **HunyuanImage3**        |     ❌     |     ✅      |           ❌           |       ❌        |         ✅         |          ❌          |   ❌    |             ❌             |          ❌           |       ✅        |        ✅*        |
| **Krea 2**               |     ❌     |     ✅      |           ❌           |       ❌        |         ❌         |          ❌          |   ✅    |             ✅             |      ✅ (decode)      |       ❌        |        ❌         |
| **LongCat-Image**        |     ✅     |     ✅      |           ✅           |       ✅        |         ✅         |          ❌          |   ❌    |             ✅             |          ❌           |       ❌        |        ❌         |
| **LongCat-Image-Edit**   |     ✅     |     ✅      |           ✅           |       ✅        |         ✅         |          ❌          |   ❌    |             ✅             |          ❌           |       ❌        |        ❌         |
| **MammothModa2(T2I)**    |     ❌     |     ❌      |           ❌           |       ❌        |         ❌         |          ❌          |   ❌    |             ❌             |          ❌           |       ❌        |        ❌         |
| **Nextstep_1(T2I)**      |     ❓     |     ❓      |           ❌           |       ✅        |         ✅         |          ❌          |   ❌    |             ✅             |          ❌           |       ❌        |        ❌         |
| **OmniGen2**             |     ❌     |     ✅      |           ✅           |       ❌        |         ✅         |          ❌          |   ❌    |             ❌             |          ❌           |       ❌        |        ❌         |
| **Ovis-Image**           |     ❌     |     ✅      |           ❌           |       ✅        |         ❌         |          ❌          |   ❌    |             ✅             |          ❌           |       ❌        |        ❌         |
| **Qwen-Image**           |     ✅     |     ✅      |           ✅           |       ✅        |         ✅         |          ❌          |   ✅    |             ✅             |      ✅ (decode)      |       ✅        |        ✅         |
| **Qwen-Image-2512**      |     ✅     |     ✅      |           ✅           |       ✅        |         ✅         |          ❌          |   ✅    |             ✅             |      ✅ (decode)      |       ✅        |        ✅         |
| **Qwen-Image-Edit**      |     ✅     |     ✅      |           ✅           |       ✅        |         ✅         |          ❌          |   ✅    |             ✅             |      ✅ (decode)      |       ❌        |        ❌         |
| **Qwen-Image-Edit-2509** |     ✅     |     ✅      |           ✅           |       ✅        |         ✅         |          ❌          |   ✅    |        ✅ (decode)         |          ✅           |       ❌        |        ❌         |
| **Qwen-Image-Layered**   |     ✅     |     ✅      |           ✅           |       ✅        |         ✅         |          ❌          |   ✅    |             ✅             |      ✅ (decode)      |       ❌        |        ❌         |
| **SenseNova-U1**         |     ❌     |     ✅      |           ❌           |       ✅        |         ✅         |          ❌          |   ❌    |             ✅             |          ❌           |       ❌        |        ❌         |
| **Stable-Diffusion-XL**  |     ❌     |     ❌      |           ✅           |       ✅        |         ✅         |          ❌          |   ✅    |             ✅             |      ✅ (decode)      |       ❌        |        ❌         |
| **Stable-Diffusion3.5**  |     ❌     |     ✅      |           ❌           |       ✅        |         ✅         |          ❌          |   ❌    |             ✅             |      ✅ (decode)      |       ❌        |        ❌         |
| **Z-Image**              |     ✅     |     ✅      |           ✅           |       ❓        |   ✅ (TP=2 only)   |          ❌          |   ✅    |             ❌             |      ✅ (decode)      |       ✅        |        ❌         |
| **ERNIE-Image**          |     ❌     |     ✅      |           ✅           |       ❓        |         ✅         |          ❌          |   ✅    |             ✅             |          ❌           |       ❌        |        ❌         |
| **Cosmos3**              |     ❌     |     ✅      |           ✅           |       ✅        |         ✅         |          ❌          |   ✅    |             ✅             |      ✅ (decode)      |       ✅        |        ❌         |

> Notes:
> 1. Nextstep_1(T2I) does not support cache acceleration methods such as TeaCache or Cache-DiT.
> 2. `Tongyi-MAI/Z-Image-Turbo` is a distilled model with minimal NFEs; CFG-Parallel is not necessary.
> 3. Cosmos3 T2I uses `Cosmos3OmniDiffusersPipeline` with `modalities=["image"]`. Model-level CPU offload swaps the nested UND reasoner and GEN generator pathways; layerwise offload remains available for blockwise GEN/UND offload.
> 4. Krea 2 currently supports single-GPU inference plus LoRA, Cache-DiT, HSDP, CPU/layerwise offload, and VAE-patch-parallel (decode). TP/SP/CFG-Parallel are not yet wired. The few-step distilled (Turbo) checkpoint uses `is_distilled=true` (fixed timestep shift `mu=1.15`); generate at 2048x2048 by default with `num_inference_steps≈8` and `guidance_scale=0`. The Raw checkpoint uses 1024x1024, `num_inference_steps=28`, and `guidance_scale=4.5`.
> 5. HunyuanImage3 supports step execution. Multi-request step execution requires `TORCH_SDPA`; see [Diffusion Execution Modes](diffusion/execution_modes.md#step-execution).

### VideoGen

| Model                        | ⚡TeaCache | ⚡Cache-DiT | 🔀SP (Ulysses & Ring) | 🔀CFG-Parallel | 🔀Tensor-Parallel | Pipeline-Parallel | 🔀HSDP | 💾CPU Offload (Layerwise) | 💾VAE-Patch-Parallel | 💾Quantization | 🔄Step Execution |
|------------------------------|:---------:|:----------:|:---------------------:|:--------------:|:-----------------:|:-----------------:|:------:|:-------------------------:|:--------------------:|:--------------:|:----------------:|
| **Wan2.2**                   |     ❌     |     ✅      |           ✅           |       ✅        |         ✅         |         ✅         |   ✅    |             ✅             |  ✅ (encode/decode)   |       ❌        |        ❌         |
| **Wan2.2-S2V**               |     ❌     |     ✅      |           ✅           |       ✅        |         ✅         |         ❌         |   ✅    |             ✅             |  ✅ (encode/decode)   |       ❌        |        ❌         |
| **Wan2.1-VACE**              |     ❌     |     ✅      |           ✅           |       ✅        |         ✅         |         ❌         |   ✅    |             ✅             |      ✅ (decode)      |       ❌        |        ❌         |
| **LTX-2**                    |     ❌     |     ✅      |           ✅           |       ✅        |         ✅         |         ❌         |   ✅    |             ✅             |      ✅ (decode)      |       ❌        |        ❌         |
| **LTX-2.3**                  |     ❌     |     ✅      |           ✅           |       ✅        |         ✅         |         ❌         |   ✅    |             ✅             |      ✅ (decode)      |       ❌        |        ❌         |
| **LTX-2.5**                  |     ❌     | ❓ (one-stage) | ❓ (Ulysses only) | ✅ (Full only) |         ❓         |         ❌         |   ✅    |             ✅             |      ✅ (decode)      |    ❓ (FP8)     |        ❌         |
| **SANA-Video-2B T2V I2V**   |     ❓     |     ❓      |           ❓           |       ❓        |         ❓         |         ❓         |   ❓    |             ❓             |          ❓           |       ❓        |        ❌         |
| **Helios**                   |     ❌     |     ✅      |           ✅           |       ✅        |         ✅         |         ❌         |   ✅    |             ✅             |          ❌           |       ❌        |        ✅*        |
| **HunyuanVideo-1.5 T2V I2V** |     ❌     |     ✅      |           ✅           |       ✅        |         ✅         |         ❌         |   ✅    |             ✅             |  ✅ (encode/decode)   |       ✅        |        ❌         |
| **Cosmos3**                  |     ❌     |     ✅      |           ✅           |       ✅        |         ✅         |         ❌         |   ✅    |             ✅             |  ✅ (encode/decode)   |       ✅        |        ❌         |
| **LongCat-Video-Avatar-1.5** |     ❌     |     ❌      |           ❌           |       ❌        |         ❌         |         ❌         |   ❌    |             ❌             |          ❌           |       ❌        |        ❌         |
| **MiniMax-H3**               | ✅ (FL2VA) |     ✅      |           ✅           |       ❌        |       ✅ (DiT/TE)  |         ❌         |   ✅    |             ✅             |       ✅ (tile)       |      ✅ (DiT)      |        ❌         |
| **SANA-WM**                  |     ❌     |     ❌      |          ❌<sup>5</sup> |       ✅        |         ✅         |         ❌         |   ❌    |             ❌             |          ❌           |       ❌        |        ❌         |

> Notes:
> 5. SANA-WM cannot support sequence parallelism: its bidirectional gated delta
>    recurrence carries state across frames, so a rank cannot denoise a slice of
>    the token sequence in isolation. Doing so would need a distributed scan or
>    an all-gather before every GDN block. The remaining ❌ columns are simply
>    unvalidated on this model, not known-broken.

> **Step execution note:** Helios supports single-request step execution only;
> use `max_num_seqs=1`.

**Frame Interpolation Support**

- **Supported**: Wan2.2 text-to-video, image-to-video, and TI2V pipelines; SANA-Video-2B native text-to-video and image-to-video pipelines
- **Not supported**: Wan2.1-VACE, LTX-2, LTX-2.3, LTX-2.5, Helios, HunyuanVideo-1.5, DreamID-Omni, SANA-WM; SANA-Video-2B Diffusers-adapter text-to-video and image-to-video pipelines

### AudioGen

| Model                 | ⚡TeaCache | ⚡Cache-DiT | 🔀SP (Ulysses & Ring) | 🔀CFG-Parallel | 🔀Tensor-Parallel | 🔀Pipeline-Parallel | 🔀HSDP | 💾CPU Offload (Layerwise) | 💾VAE-Patch-Parallel | 💾Quantization | 🔄Step Execution |
|-----------------------|:---------:|:----------:|:---------------------:|:--------------:|:-----------------:|:-------------------:|:------:|:-------------------------:|:--------------------:|:--------------:|:----------------:|
| **Stable-Audio-Open** |     ✅     |     ❌      |           ❓           |       ❓        |         ❌         |          ❌          |   ✅    |             ✅             |          ❌           |       ✅        |        ❌         |


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
| **🔀HSDP** | ❓ | ❓ | ✅ | ❓ | ✅ | ❌ | | | | | | | | |
| **🔀Expert Parallel** | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | | | | | | | |
| **💾CPU Offloading (Layerwise)** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | | | | | | |
| **💾CPU Offloading (Module-wise)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | ❌ | | | | | |
| **💾VAE Patch Parallel** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | | | | |
| **💾FP8 Quant** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | ✅ | ✅ | ✅ | | | |
| **🔧LoRA Inference** | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | | |
| **🔄Step Execution** | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | ✅ | ❓ | ✅ | ✅ | ✅ | |

!!! info

    1. HSDP can be combined with Ulysses-SP or CFG-Parallel. Tensor Parallel
       and HSDP are not compatible; other HSDP combinations in the table
       remain unverified.
    2. TeaCache and Cache-DiT are not compatible.
    3. CPU Offloading (Layerwise) and CPU Offloading (Module-wise) are not compatible.
    4. The CPU Offloading (Layerwise) row describes local layerwise offload.
       Multi-device Distributed Layerwise Offload has a separate topology and
       compatibility matrix in the [Distributed Layerwise Offloading guide](diffusion/offloader/distributed_layerwise_offload.md).
    5. The compatibility matrix uses FP8 as the representative quantization method.
    6. Step Execution is not compatible with any diffusion cache backend. LoRA is supported, but each scheduled batch must use a single adapter (requests with different `lora_request` or `lora_scale` are kept in separate batches).


## Multi-Thread Weight Loading

The loading guide now lives at [Diffusion Startup and
Loading](diffusion/startup_and_loading.md). This heading remains so existing
links to this section continue to work.

## Learn More

The Diffusion Acceleration navigation groups the remaining guides as follows:

| Area | Guide |
| --- | --- |
| Compatibility | [Feature Compatibility](feature_compatibility.md) |
| CPU offloading | [CPU Offloading](diffusion/cpu_offload.md) |
| Cache acceleration | [TeaCache](diffusion/cache_acceleration/teacache.md), [Cache-DiT](diffusion/cache_acceleration/cache_dit.md) |
| Parallelism | [Parallelism Overview](diffusion/parallelism/overview.md) |
| Attention | [Attention Backends](diffusion/attention_backends.md) |
| Compilation | [Regional Compilation](diffusion/regional_compilation.md) |
| Video extension | [Frame Interpolation](diffusion/frame_interpolation.md) |
| Startup | [Startup and Loading](diffusion/startup_and_loading.md) |
| Adapters | [LoRA](diffusion/lora.md) |

Related cross-model and runtime features are documented separately:

- [Quantization](quantization/overview.md) covers diffusion-only models,
  multi-stage omni/TTS models, and multi-stage diffusion models.
- [Execution Modes and Streaming](diffusion/execution_modes.md) covers the
  diffusion runtime, including batching, step execution, and streaming output.
