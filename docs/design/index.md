# Design Documents

This section contains design documents and architecture specifications for
vLLM-Omni. The sidebar groups documents by the system concern they describe,
while the source paths remain stable so existing links continue to work.

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

### Runtime Modules

- [AR Module](module/ar_module.md)
- [DIT Module](module/dit_module.md)

### Orchestration

- [Entrypoint Module](module/entrypoint_module.md)
- [AsyncOmni Architecture (Qwen3-Omni Example)](module/async_omni_architecture.md)
