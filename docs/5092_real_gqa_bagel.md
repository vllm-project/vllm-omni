# Issue 5092: real BAGEL GQA SP validation

This note records the real-model validation for the sequence-parallel strategy
selector in issue #5092. The run used a BAGEL checkpoint on 4x A800-SXM4 80 GB
GPUs, with two independent seeds per strategy. Timings are steady-state denoising
step latency; initialization is excluded from the comparison.

| strategy | mean step (ms) | peak memory (MiB) | DINOv2 mean drop | max frame drop |
| --- | ---: | ---: | ---: | ---: |
| Ulysses | 207.29 | 31,205 | 5.3% | 9.0% |
| Hybrid (Ulysses + Ring) | 231.69 | 31,119 | 5.7% | 9.0% |
| Ring (raw) | 244.77 | 30,645 | 14.6% | 17.9% |
| Ring (guard4) | 236.57 | 30,633 | 5.7% | 8.8% |
| AllGather-KV | 169.57 | 30,587 | 5.8% | 9.2% |

The quality gate is mean DINOv2 drop <= 8% and maximum single-frame drop <=
15%. The raw Ring path does not pass this gate; the calibrated `guard4` path
does. AllGather-KV is both the fastest and quality-compliant configuration in
this run. These results do not claim that one strategy is universally optimal;
they provide the measured calibration data consumed by `strategy: auto`.

## What changed

* Ring SDPA keeps compact GQA K/V tensors during communication and expands K/V
  heads only immediately before local SDPA (`repeat_kv`). This avoids inflating
  Ring traffic while making the local attention shape valid.
* A causal BAGEL cache-attention module can opt out of SP construction. BAGEL
  manually shards the non-causal VAE denoising path, so constructing a
  non-causal-only AllGather-KV strategy for the causal cache path is incorrect.
* Added regression coverage for the causal-cache opt-out and real GQA/MQA SP
  configurations.

## Reproduction

The benchmark requires the BAGEL checkpoint and four GPUs. For each strategy,
run the existing 5092 BAGEL E2E harness with a distinct `--strategy` value
(`ulysses`, `hybrid`, `ring`, `allgather_kv`, or `ring_guard4`) and two seeds.
Collect `result.json` from each run, then compute pixel metrics and DINOv2
quality against the single-rank cold baseline. The benchmark intentionally
does not commit model weights or generated frames to the source tree.
