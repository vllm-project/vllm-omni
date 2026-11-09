# LingBot FA preparation: NPU investigation

This note preserves the original local-operator investigation. For subsequent
whole-model measurements and the curated implementation scope, see
[NPU_OPTIMIZATIONS.md](NPU_OPTIMIZATIONS.md).

The small operators before FA are not VAE normalization. Self-attention does
QKV projection, across-head Q/K RMSNorm (5120 dimensions, not each 128-wide
head), adjacent-pair Wan RoPE, QKV stacking, sequence/head all-to-all, KV slot
writes, and history gathering/staging. Cross-attention normalizes its query and
the first text-key projection, but has no video RoPE/Ulysses exchange; text K/V
is cached. AdaLN modulation precedes the self QKV projection.

## Implemented: indexed row writes

`_paged_write_attn_impl` used advanced indexed assignment for video/action K/V.
NPU now uses `index_copy_(0, slots, values)`; other devices retain the original
assignment. Cache planner slots are distinct LongTensor physical row indices.
The dtype conversion, explicit pool mutation, write-before-attention ordering,
and caller stream remain unchanged. No cache layout or precision change.

Ascend950PR, noncontiguous source strides `(3840,128,1)`, BF16 `[tokens,10,128]`,
pool 32768 tokens, random distinct slots, five warmup iterations and three
30-call measurements:

| Tokens | Advanced write ms | Index-copy ms | Pool parity |
| --- | --- | --- | --- |
| 1508 | 0.0836 | 0.0282 | Exact |
| 4524 | 0.2400 | 0.0272 | Exact |

This is a local operator measurement, not an end-to-end speedup. CPU/NPU tests
cover nonmonotonic slots, noncontiguous inputs, dtype conversion, overwrite,
empty writes, and untouched rows. The four-card retained service was not patched
or restarted; experiments used its otherwise idle physical device6 with small
additional allocations. Concurrent inference was not requested.

## Evaluated, not integrated

- A fused QKV stack/all-to-all-input transpose copy was numerically exact but
  slower: tokens1131 native0.0544ms vs Triton1.373ms; tokens1508 native0.1150ms
  vs Triton1.863ms. Keep the existing native packing. A future fusion should
  emit the communication layout from norm/RoPE, rather than add a slower copy.
- Existing native Q/K RMSNorm at tokens1131 was 0.0255ms (BF16) /0.0334ms
  (FP32 input) vs decomposed0.0927ms. Both changed54 output elements out of
  5.79million (relative L2~1.58e-5). At4524 tokens: nativeBF160.0926ms,
  nativeFP320.4135ms, decomposed0.7115ms;178 elements differed. These are random
  QKV projections/weights, not model-quality acceptance. This capability already
  exists behind `lingbot_npu_fused_rmsnorm`; do not duplicate it or call it exact.

Raw scripts, JSON and independent write profiling were retained with the
2026-11-04 operator experiment artifacts; they are not deployment dependencies.
Preserve the 5120-wide norm, epsilon, FP32 gamma multiplication and BF16 boundary
when attempting a larger norm/RoPE/pack fusion. HCCL communication itself cannot
be removed by an elementwise kernel; specialize only the surrounding layouts.
