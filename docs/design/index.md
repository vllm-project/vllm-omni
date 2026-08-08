# Design Documents

This section contains design documents and architecture specifications for
vLLM-Omni. The sidebar groups documents by the system concern they describe;
retired module pages are preserved in the legacy archive instead of remaining
in the active navigation.

## Architecture Documents

- [Architecture Overview](architecture_overview.md)

## Feature Design Documents

### Runtime and stage execution

- [Disaggregated Inference](feature/disaggregated_inference.md)
- [Async Chunk](feature/async_chunk.md)
- [Async Diffusion Output](feature/async_diffusion_output.md)
- [Async Omni Output Materialization](feature/omni_async_output_materialization.md)
- [Automatic Prefix Caching in Omni Models](feature/prefix_caching.md)

### Communication

#### OmniConnector implementations

- [Mooncake Store Connector](feature/omni_connectors/mooncake_store_connector.md)
- [Mooncake Transfer Engine Connector](feature/omni_connectors/mooncake_transfer_engine_connector.md)
- [Mori Transfer Engine Connector](feature/omni_connectors/mori_transfer_engine_connector.md)
- [Shared Memory Connector](feature/omni_connectors/shared_memory_connector.md)
- [Yuanrong Store Connector](feature/omni_connectors/yuanrong_connector.md)
- [Yuanrong Transfer Engine Connector](feature/omni_connectors/yuanrong_transfer_engine_connector.md)

### Diffusion acceleration

#### Parallelism

- [CFG-Parallel](feature/cfg_parallel.md)
- [Expert Parallel](feature/expert_parallel.md)
- [Hybrid Sharded Data Parallel (HSDP)](feature/hsdp.md)
- [Pipeline Parallel](feature/pipeline_parallel.md)
- [Sequence Parallel](feature/sequence_parallel.md)
- [Tensor Parallel](feature/tensor_parallel.md)
- [VAE Patch Parallelism](feature/vae_parallel.md)

#### Attention Backends

- [Skip-Softmax](feature/skip_softmax.md)

#### Quantization

- [Quantization](feature/quantization.md)

- [Cache-DiT](feature/cache_dit.md)
- [TeaCache](feature/teacache.md)
- [Diffusion Continuous Batching](feature/diffusion_continuous_batching.md)
- [Distributed Layerwise Offload](feature/distributed_layerwise_offload.md)

## Infrastructure and Performance

- [Prometheus Metrics](metrics.md)
- [Speech Generation Performance Optimizations](qwen3_omni_tts_performance_optimization.md)

## Module Design Documents

- [Entrypoints and Serving Boundaries](module/entrypoints.md)
- [vLLM-Omni Configuration](module/vllm_omni_config.md)
- [Input, Output, and Modality Contracts](module/input_output_modality_contracts.md)
- [Error Classification, Propagation, and Rendering](module/error_contracts.md)
- [Engine Orchestration](module/engine_orchestration.md)
- [Stage Runtime and Replica Lifecycle](module/stage_runtime.md)
- [OmniConnector](module/omni_connector.md)
- [Model Integration](module/model_integration.md)
- [Autoregressive Runtime](module/ar_runtime.md)
- Diffusion
  - [Overview](module/diffusion/index.md)
  - [Runtime](module/diffusion/diffusion_runtime.md)
  - [Model Integration](module/diffusion/diffusion_model_integration.md)
  - [Continuous Batching](module/diffusion/continuous_batching.md)
  - [Parallelism](module/diffusion/parallelism.md)
  - [Offloader](module/diffusion/offloader.md)
- [Execution Platforms](module/execution_platforms.md)
- [Cache Management](module/cache_management.md)
- [Quantization](module/quantization.md)
- [Observability](module/observability.md)
- [Profiling](module/profiling.md)
- [Benchmarking](module/benchmarking.md)

The pre-#5137 pages are preserved in the
[legacy module archive](module/archive/README.md) for historical reference and
are not active design contracts.
