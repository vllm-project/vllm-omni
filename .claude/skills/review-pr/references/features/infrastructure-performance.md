# Infrastructure and Performance Features

Load the matching design from the reviewed head:

- [Prometheus Metrics](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/metrics/)
  for logger wrapping, gating, lifecycle, stage transfer, and metric definitions.
- [Speech Generation Performance Optimizations](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/qwen3_omni_tts_performance_optimization/)
  for stage batching, CUDA graphs, async chunk, streaming, re-prefill, compile,
  and deployment evidence.

## Feature checks

- Route instrumentation changes through observability or profiling and result
  methodology through benchmarking and the performance evidence check.
- Preserve opt-in/default behavior, units, labels, aggregation, timing scope,
  lifecycle resets, and bounded overhead.
- For stacked optimizations, measure each increment against the same baseline
  and explain interactions, quality guards, regressions, and limitations.
- Require exact model, checkpoint, hardware, software, workload, topology,
  concurrency, warmup, repetitions, and production commands.
- Update the design page when metric meaning, supported optimization stack, or
  deployment guidance changes.
