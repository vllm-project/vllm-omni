# Diffusion Module Family

Primary design: [Diffusion Modules](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/module/diffusion/).

Use for diffusion-stage behavior, then load the exact submodule page from the
reviewed head:

| Submodule | Signals | Design |
| --- | --- | --- |
| Runtime | Admission, scheduling, execution, progress, outputs, cancellation, hooks, cleanup | [Diffusion Runtime](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/module/diffusion/diffusion_runtime/) |
| Model Integration | Pipelines, registry, loader, adapters, shared layers, model-specific processing | [Diffusion Model Integration](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/module/diffusion/diffusion_model_integration/) |
| Continuous Batching | Compatibility, request/step batches, per-request progress, capacity | [Continuous Batching](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/module/diffusion/continuous_batching/) |
| Parallelism | Rank topology, process groups, sharding, collectives, distributed execution | [Parallelism](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/module/diffusion/parallelism/) |
| Offloader | Residency, transfers, memory accounting, prefetch, teardown | [Offloader](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/module/diffusion/offloader/) |

## Family-wide checks

- Keep one scheduler-owned lifecycle from admission through completion,
  cancellation, failure, and cleanup; optional features must use defined hooks.
- Select pipelines through registry and loader contracts instead of scattered
  model-name conditionals; keep genuine model differences in model directories.
- Trace latent, conditioning, timestep, generator, shape, layout, dtype, device,
  batch expansion, and output conversion through the actual consumer.
- Make batch compatibility and per-request isolation explicit; never associate
  output by unstable batch position.
- Derive topology from validated configuration, define every sharding boundary,
  keep collectives symmetric, and preserve a supported single-rank path.
- Track one residency owner per offloaded component, wait for transfer readiness,
  preserve model state, bound retained copies, and clean up deterministically.

Load matching feature designs for cache, quantization, parallelism, continuous
batching, or offload. Require feature-off correctness, representative inference,
terminal-path tests, and quality evidence for optimizations.
