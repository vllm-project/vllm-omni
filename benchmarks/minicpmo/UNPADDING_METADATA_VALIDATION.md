# MiniCPM-o 4.5 Vision Unpadding Metadata Optimization

## Change

`SiglipEncoder.forward` now computes FlashAttention unpadding metadata once for
the current attention mask and passes the same forward-local tuple through all
27 encoder layers. `SiglipFlashAttention2` retains a fallback that computes the
metadata itself when called directly without precomputed metadata.

The implementation does not use a global cache, tensor data pointers, or state
that survives an encoder forward. A later request with a different mask computes
new metadata.

## Test Environment

- Date: 2026-07-17 (Asia/Shanghai).
- Local branch: `codex/minicpmo-unpad-metadata-reuse` based on `0a9a5c23`.
- Remote test host: 2 x NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120).
- Model: ModelScope `OpenBMB/MiniCPM-o-4_5`, 437 VPM tensors and 417,792,240 vision parameters.
- Stack: Python 3.12.13, PyTorch 2.11.0+cu130, upstream flash-attn 2.8.3.post1.
- Benchmark: BF16, 27 active layers, five warmups and 30 paired samples per variant.

The benchmark loads official VPM weights and official image/video assets. It is
a vision-encoder benchmark, not full serving latency.

## Regression Tests

- CPU contract tests: 4 passed.
- Physical GPU 0: 8 passed.
- Physical GPU 1: 8 passed.
- The production encoder output matches the legacy per-layer metadata
  recomputation path bitwise (`atol=0`, `rtol=0`).
- Baseline and optimized outputs for every real-media workload have maximum
  absolute difference 0.0.

Coverage includes:

1. Padded encoder forwards compute metadata exactly once.
2. Dense forwards do not compute unpadding metadata.
3. Consecutive forwards with different masks compute distinct metadata.
4. Direct FlashAttention calls retain the per-layer fallback.
5. Moderate- and high-padding outputs match per-sample unpadded references.
6. The existing Q/K/V storage-sharing layout contract remains valid.

## Real-Media A/B Results

The baseline explicitly runs the legacy encoder loop with `unpadding_metadata=None`
for every layer. The optimized variant calls the production `SiglipEncoder.forward`.
Execution order alternates baseline and optimized on every measured iteration.

### Physical GPU 0

| Workload | Attention path | Baseline mean | Optimized mean | Change | Exact parity |
|---|---|---:|---:|---:|---:|
| Single image | dense | 8.798 ms | 8.832 ms | +0.034 ms (+0.39%) | max diff 0.0 |
| Mixed image batch | varlen | 21.697 ms | 13.790 ms | -7.907 ms (-36.44%) | max diff 0.0 |
| Short video, 4 frames | dense | 19.640 ms | 19.648 ms | +0.008 ms (+0.04%) | max diff 0.0 |

### Physical GPU 1

| Workload | Attention path | Baseline mean | Optimized mean | Change | Exact parity |
|---|---|---:|---:|---:|---:|
| Single image | dense | 9.059 ms | 9.068 ms | +0.009 ms (+0.10%) | max diff 0.0 |
| Mixed image batch | varlen | 22.162 ms | 14.113 ms | -8.049 ms (-36.32%) | max diff 0.0 |
| Short video, 4 frames | dense | 19.646 ms | 19.646 ms | +0.000 ms (+0.00%) | max diff 0.0 |

The mixed image workload uses official `fossil.png` and `highway.png`, valid
lengths 1040 and 1025, and only 0.721% padding. The fixed varlen setup cost is
therefore material even though very little attention computation is skipped.

## Trace Attribution

For the mixed-image workload:

| Metric | GPU 0 baseline -> optimized | GPU 1 baseline -> optimized |
|---|---:|---:|
| `_get_unpad_data` profiler entries | 54 -> 2 | 54 -> 2 |
| `_get_unpad_data` self CUDA | 4.607 -> 0.247 ms | 4.685 -> 0.241 ms |
| `index_first_axis` self CUDA | 0.228 -> 0.226 ms | 0.232 -> 0.229 ms |
| `pad_input` self CUDA | 0.928 -> 0.856 ms | 0.939 -> 0.862 ms |
| Varlen FlashAttention self CUDA | 1.778 -> 1.777 ms | 1.795 -> 1.790 ms |
| Materialization entries | 199 -> 121 | 199 -> 121 |
| Kernel launches | 611 -> 377 | 611 -> 377 |

Profiler entries include the instrumentation events used for CPU/CUDA
attribution. The `54 -> 2` change corresponds to removing 27 per-layer metadata
computations and retaining one encoder-level computation. Q/K/V gather, output
padding, and the varlen attention kernel remain necessary and are not removed.

## Conclusion

The production implementation reproduces the prototype result with a slightly
larger measured improvement: approximately 36.4% lower vision-encoder latency
for the heterogeneous padded image batch on both GPUs. Dense image and uniform
video paths remain statistically unchanged.

The evidence supports a performance PR for forward-local unpadding metadata
reuse. It does not support describing the existing Q/K/V transpose/view round
trip as a material GPU layout conversion; that path remains zero-copy cleanup.

## Caveats

- The first CPU test run retained an old fake FlashAttention signature and failed
  on the new keyword argument. The test double was updated; the retained retry log
  records all four CPU contract tests passing.
- Installed vLLM 0.25.0 and vLLM-Omni 0.1.dev2211 emit a version-mismatch warning.
- Ruff is not installed on the remote host. Python compilation, `git diff --check`,
  and `pip check` pass.
- No Buildkite files were changed. The CPU tests use the existing `core_model and
  cpu` collection; CUDA tests use the existing hardware markers.
