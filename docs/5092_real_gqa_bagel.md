# Issue 5092: real BAGEL GQA strategy comparison

Measured on a real BAGEL GQA workload with 4x A800-SXM4 80 GB GPUs and two
seeds per strategy. Timings are steady-state denoising step latency.

| strategy | mean step (ms) | peak memory (MiB) | DINOv2 mean drop | max frame drop |
| --- | ---: | ---: | ---: | ---: |
| Ulysses | 207.29 | 31,205 | 5.3% | 9.0% |
| Hybrid (Ulysses + Ring) | 231.69 | 31,119 | 5.7% | 9.0% |
| Ring (raw) | 244.77 | 30,645 | 14.6% | 17.9% |
| Ring (guard4) | 236.57 | 30,633 | 5.7% | 8.8% |
| AllGather-KV | 169.57 | 30,587 | 5.8% | 9.2% |

The quality gate is mean DINOv2 drop <= 8% and maximum single-frame drop <=
15%. Raw Ring fails this gate; the guarded Ring path passes. AllGather-KV is
the fastest quality-compliant configuration in this run. These measurements
are calibration data for `strategy: auto`, not a claim that one strategy is
universally optimal.

The benchmark does not commit model weights or generated frames.
