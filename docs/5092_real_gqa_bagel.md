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

## Startup-time selection

Auto selection is opt-in and is resolved before distributed process groups are
initialized. A diffusion stage can declare a representative deployment shape
and an offline JSONL calibration profile:

```yaml
parallel_config:
  sp_strategy: auto
  sp_selector_profile: /path/to/sp_calibration.jsonl
  sp_selector_workload:
    seq_len: 4096
    sp_degree: 4
    num_heads: 32
    num_kv_heads: 4
    head_dim: 128
    batch_size: 1
    dtype_bytes: 2
    interconnect: nvlink
```

The calibration file contains one row per measured strategy and shape, using
`strategy`, `interconnect`, `sp_degree`, `kv_ratio`, `seq_len`, `batch_size`,
`head_dim`, `dtype_bytes`, and `latency_ms`. All shape fields participate in
calibration matching; missing or mismatched dimensions are rejected instead of
silently selecting from a different tensor shape. The earlier #5092 names
(`sp`, `f`, `seq`, `batch`, `dim`, `dtype`, and `p50_ms`) are also accepted;
known dtype names such as `bf16` are normalized to their byte width. The
selector first rejects unsupported strategies, then chooses the lowest
predicted latency and writes the corresponding degree before group
initialization. Manual degree configuration remains the default. AllGather-KV
is considered feasible only when the global sequence length is divisible by
the SP degree because its collective requires equal local shard sizes.

Ring is excluded from auto selection unless `sp_selector_allow_ring: true` is
set, because the BAGEL result above shows that latency alone is not a sufficient
quality gate. Enable it only after model-specific numerical and perceptual
validation.
