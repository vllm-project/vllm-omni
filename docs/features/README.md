# Features

Use this section to choose and configure vLLM-Omni runtime and optimization
features. The navigation follows the active
[Feature Design](../design/index.md#feature-design-documents) taxonomy whenever
an implementation contract has a user-facing workflow.

!!! note "User guides, design documents, and recipes"

    User guides explain how to enable a feature and where it is supported.
    Design documents define internal contracts and do not, by themselves,
    imply general or production support. For model-specific launch flags and
    hardware requirements, use the linked examples and validated recipes from
    [Supported Models](../models/supported_models.md).

## Runtime and Stage Execution

| Goal | User guide | Related design contract |
| --- | --- | --- |
| Choose serial, batched, step-wise, or streaming diffusion execution | [Execution Modes and Streaming](../user_guide/diffusion/execution_modes.md) | [Diffusion Continuous Batching](../design/feature/diffusion_continuous_batching.md), [Async Diffusion Output](../design/feature/async_diffusion_output.md) |
| Reclaim stage memory without restarting the server | [Sleep Mode](sleep_mode.md) | Runtime lifecycle behavior is documented in the user guide |

The [Prefill-Decode Disaggregation design reference](pd_disaggregation.md)
describes the experimental Qwen3-Omni runtime topology and its current
integration boundaries. It is not yet a supported launch recipe.

Some runtime designs are deliberately not promoted as standalone User Guide
features yet:

- Generic, OmniConnector-based
  [Disaggregated Inference](../design/feature/disaggregated_inference.md) is not
  yet a standalone User Guide feature. Its
  [OmniConnector implementations](../design/index.md#communication) describe
  transport contracts, while practical connector configuration remains under
  [Pipeline and deploy configurations](../configuration/stage_configs.md).
- [Async Chunk](../design/feature/async_chunk.md) and
  [Async Omni Output Materialization](../design/feature/omni_async_output_materialization.md)
  are model- and pipeline-dependent. Use the selected model's deploy
  configuration and recipe for the supported settings.
- [Automatic Prefix Caching](../design/feature/prefix_caching.md) remains a
  design-level contract until its user-facing configuration and compatibility
  surface is consolidated.

## Quantization

Quantization is a cross-model feature rather than a diffusion-only
optimization. The unified [`quantization_config` guide](../user_guide/quantization/overview.md)
covers diffusion-only models, multi-stage omni/TTS models, and multi-stage
diffusion models. Its [design contract](../design/feature/quantization.md)
defines the shared configuration and backend extension points.

## Diffusion Acceleration

| Goal | User guide | Related design contract |
| --- | --- | --- |
| Compare acceleration methods and supported combinations | [Overview](../user_guide/diffusion_features.md), [Feature Compatibility](../user_guide/feature_compatibility.md) | [Diffusion acceleration designs](../design/index.md#diffusion-acceleration) |
| Move weights between host and device memory | [CPU Offloading](../user_guide/diffusion/cpu_offload.md) | [CPU Offloading](../design/feature/offloader/README.md) |
| Reuse denoising computation | [Cache Acceleration](../user_guide/diffusion/cache_acceleration/cache_dit.md) | [Cache-DiT](../design/feature/cache_dit.md), [TeaCache](../design/feature/teacache.md) |
| Distribute diffusion work across devices | [Parallelism](../user_guide/diffusion/parallelism/overview.md) | [Parallelism designs](../design/index.md#parallelism) |
| Select dense, sparse, or quantized attention paths | [Attention Backends](../user_guide/diffusion/attention_backends.md) | [Attention Backend Selection](../design/feature/attention_backend_selection.md) |
| Compile repeated diffusion regions | [Regional Compilation](../user_guide/diffusion/regional_compilation.md) | User-facing optimization guide |
| Add generated video frames | [Frame Interpolation](../user_guide/diffusion/frame_interpolation.md) | User-facing extension guide |
| Reduce diffusion model startup time | [Startup and Loading](../user_guide/diffusion/startup_and_loading.md) | User-facing loading guide |
| Apply diffusion adapters | [LoRA](../user_guide/diffusion/lora.md) | User-facing extension guide |

The [Pipeline Parallelism guide](../user_guide/diffusion/parallelism/pipeline_parallel.md)
remains available by direct link, but it is not promoted in the primary User
Guide navigation until its user-facing support and placement are settled.

## Experimental

[Session State Manager](session_state_manager.md) is opt-in and experimental.
Its APIs and compatibility may change without notice.
