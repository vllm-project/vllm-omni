# Performance and Accuracy Verification

Activate this reference when the PR adds a model or claims or intentionally
changes latency, throughput, memory, scaling, precision, or output quality.

Official docs: [profiling](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/profiling/),
[serving benchmarks](https://docs.vllm.ai/projects/vllm-omni/en/latest/cli/bench/serve/),
and [metrics](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/metrics/).

## Comparable A/B contract

Compare frozen base and head for an existing path. For a new model absent from
base, compare head with a pinned canonical reference implementation. Use the same:

- hardware, driver, software versions, model/checkpoint, and dependencies;
- input or dataset, seed, precision, batch/concurrency, topology, and feature
  flags;
- warmup, measured repetitions, synchronization, timing scope, and memory
  collection method.

Keep exact commands and report variability rather than a cherry-picked run.
Use the repository's benchmark when it covers the claim; otherwise use the
smallest reproducible workload that reaches the changed production path.

## Evidence table

| Dimension | Typical evidence |
| --- | --- |
| Latency | End-to-end latency and, when available, TTFT plus a defined per-stage and transfer-time split. |
| Throughput/scaling | Requests, tokens, frames, or audio duration per second across the claimed concurrency/topology. |
| Memory | Peak allocated/reserved device memory and OOM boundary if relevant. |
| Quality/accuracy | Repository metric or known-good output comparison with an explicit tolerance; paired samples when no metric exists. |

Every optimization needs correctness or quality evidence. A faster result on a
different workload, precision, seed, or topology is not a valid comparison.

## Reviewer verification ladder

1. Run base/head A/B on suitable hardware when affordable.
2. If only one side can run, verify the script and request a comparable pair.
3. If hardware or assets are unavailable, audit methodology, code path, and
   contributor evidence; name the exact unverified claim.

Classify discrepancies before reporting: implementation regression, benchmark
bug, environmental drift, noise, or unsupported claim. Explain material
regressions against the PR's stated goal or repository contract; do not invent
universal percentage thresholds.

Report claimed and measured values together, bind them to the frozen SHAs and
environment, and keep unavailable hardware as a validation gap rather than a
fabricated pass.
