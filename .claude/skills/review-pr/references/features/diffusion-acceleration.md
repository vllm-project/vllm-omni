# Diffusion Acceleration Features

Use with the exact diffusion submodule plus platform, cache, quantization, or
benchmarking contracts selected by the live path.

| Feature family | Designs |
| --- | --- |
| Parallelism | [CFG](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/cfg_parallel/), [Expert](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/expert_parallel/), [HSDP](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/hsdp/), [Pipeline](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/pipeline_parallel/), [Sequence](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/sequence_parallel/), [Tensor](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/tensor_parallel/), [VAE patch](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/vae_parallel/) |
| Attention | [Skip-Softmax](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/skip_softmax/) |
| Quantization | [Quantization](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/quantization/) |
| Cache | [Cache-DiT](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/cache_dit/), [TeaCache](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/teacache/) |
| Batching | [Diffusion Continuous Batching](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/diffusion_continuous_batching/) |
| Offload | [Distributed Layerwise Offload](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/distributed_layerwise_offload/) |

## Feature checks

- Verify feature selection, supported models/platforms/topologies, defaults,
  dependencies, and every touched feature combination.
- Preserve feature-off and single-rank correctness; reject unsupported
  combinations before collectives, loading, or execution begin.
- Trace shapes, shards, ranks, collectives, precision, cache validity, residency,
  and per-request state through the selected runtime hooks.
- Compare output quality or accuracy against the disabled/reference path before
  accepting latency or memory improvements.
- Require exact commands, topology, environment, warmup, repetitions, and
  measurements for performance claims; update feature compatibility docs.
