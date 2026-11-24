# Benchmarking

Primary design: [Benchmarking](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/module/benchmarking/).

Use when benchmark code, workloads, metric calculations, CLI benchmark
entrypoints, or result metadata change. Use the performance check separately
when a PR uses benchmark results as evidence.

## Contract checks

- Record model and revision, code SHA, hardware, software, configuration,
  workload, precision, topology, concurrency, warmup, repetitions, and
  measurement window.
- Validate successful and semantically acceptable outputs before accepting
  timing, memory, or scaling results.
- Define every metric's unit, population, aggregation, synchronization, and
  timing scope.
- Keep workload parsing and metric computation deterministic and reusable
  across base and head.
- Reject comparisons that change material workload or runtime settings without
  isolating and explaining that difference.

Test workload parsing and metric calculations with deterministic fixtures and
publish exact commands plus variability for quantitative claims.
