# 20260727 Stage-2 Profile and First Optimization Suggestions

- Decision: keep as the first validated profiling-chain record.
- Base Git SHA: `ecf965fceb657f1449adc43609680191adb22298` plus the profiling-tool worktree diff.
- Scope: local diagnostic profile, not a score run.
- Hardware: one physical Ascend 910C card; Stage 2 uses logical chip 1.
- Workload: text input, text+audio output, seed 42, 128 Thinker tokens, 128 Talker tokens.
- Capture: two unprofiled warmups followed by one Stage-2-only profiled request.
- Local artifact root: `artifacts/minicpmo_ascend/profiles/20260727-profile-v1-stage2/`.
- Analysis SHA256: `363d96b21df5f0932e5cf1730e8a80b7e7e03580208ccdcbd7f19e067a104386`.
- Capture SHA256: `bf4f7c5d05f69e0f8ae0771c9be98f410cef0da29ec771890656b8bc5b4ad94e`.
- Artifact manifest SHA256: `16776147c213c434b8cf0c65c173490181b15425032243b7c2777f30eaea011d`.

## Capture Gate

- Request: PASS.
- First text: 0.439 s.
- First audio: 1.343 s.
- E2E: 3.052 s.
- Audio: six ordered, non-duplicate 24 kHz PCM chunks; 10.08 seconds reconstructed.
- Profile export: PASS; CANN CSV, timeline, and database generated.
- Shutdown: PASS; no NPU process remained and both logical chips returned to idle HBM.

The profiled request is slower than the unprofiled warm request, as expected.
Profiler timing must not be compared with the score baseline.

## Stage-2 Evidence

| Signal | Value | Interpretation |
| --- | ---: | --- |
| Kernel calls | 97,957 | Very fragmented eager execution |
| Kernels <= 50 us | 95,418 (97.4%) | Launch/layout overhead is a first-order target |
| Aggregated kernel time | 1,005.1 ms | Streams and nested work make this non-additive with E2E |
| TransData | 222.2 ms / 8,313 calls | Largest operator family; layout conversion dominates |
| Transpose | 145.7 ms / 9,860 calls | Repeated layout changes are material |
| MatMulV2 | 108.8 ms / 8,714 calls | Main arithmetic family, but below layout total |
| LayerNormV3 | 96.3 ms / 6,951 calls | High-frequency small normalization kernels |
| ConcatD + Slice | 118.6 ms / 23,586 calls | Cache/chunk assembly is heavily fragmented |
| `aclopCompileAndExecute` | 182.7 ms / 2,771 calls | Eager/dynamic operator dispatch is material |
| `aclrtSynchronizeStream` | 17.5 ms / 94 calls | Explicit synchronization is not the primary Stage-2 cost |

`TransData + Transpose` account for 367.9 ms, or 36.6% of aggregated Stage-2
operator time. The trace matches code surfaces in `batched_token2wav.py` that
repeatedly transpose/contiguous tensors and concatenate CFG inputs, request
caches, attention caches, and mel/source/speech state.

## Optimization Suggestions v1

### 1. Reduce Layout Conversion in Batched Token2Wav

Priority: P1, first candidate.

- Keep Flow/CFM tensors in one Stage-2-native layout across `_encode_chunk`,
  `_decode_cfm`, and HiFT instead of repeatedly converting BxTxC and BxCxT.
- Audit the explicit `transpose(...).contiguous()` boundaries and the NPU HiFT
  linear-downsample patch before changing kernels.
- Cache or reuse shape-stable converted prompt features and windows.

Expected effect: fewer `TransData`/`Transpose` calls, lower first-audio and
chunk latency, and lower host launch work. Guardrails: byte-valid ordered audio,
effect gate, numeric/audio review, GPU behavior unchanged, and no request-state
sharing. Roll back if layout changes introduce copies elsewhere or fail NPU/GPU
parity.

### 2. Capture Exact Steady-State Stage-2 Shapes

Priority: P1, after the layout candidate is measured.

- Stage 2 currently runs eager while 2,771 `aclopCompileAndExecute` calls and a
  97.4% small-kernel ratio appear in one request.
- Evaluate ACL Graph capture only for proven exact-shape steady chunks; retain
  eager fallback for prompt/first/terminal or unsupported dynamic shapes.
- Bucket by the state-shape signature already used by Code2Wav batching.

Expected effect: reduce dispatch/compile and launch overhead. Guardrails:
warmup excluded, no graph reuse across incompatible request caches, terminal
flush correctness, cancellation safety, and HBM headroom.

### 3. Replace Repeated Cache Concatenation with Reusable Batch Buffers

Priority: P1, separable from graph capture.

- Profile evidence shows 7,041 ConcatD calls and 16,545 Slice calls.
- Measure whether `_stack_flow_cache`, CFG duplication, attention trimming, and
  HiFT cache assembly can use capacity-managed buffers or views.
- Keep one primary change per candidate; do not combine this with layout and
  graph experiments.

Expected effect: reduce small-kernel launches, temporary allocation, and the
concurrency-dependent HBM increase seen in the unprofiled baseline. Guardrails:
request isolation, exact cache epochs, abort cleanup, and fixed-shape stability.

## Deprioritized from This Capture

- Explicit stream synchronization: only 17.5 ms aggregated in Stage 2.
- Conv2DTranspose kernel tuning: six calls total 11.8 ms, too small for the
  first optimization.
- Quantization: higher risk and not justified before layout/dispatch work or
  publication of the official effect threshold.

## Next Experiment

Run `20260727-stage2-layout-boundaries` as an A/B/A candidate:

1. Add stage-local timing/counters around the explicit layout boundaries.
2. Change one boundary or cache representation.
3. Run focused CPU tests, Ascend smoke, correctness gate, and the unprofiled
   C1/C2/C4 text+audio benchmark.
4. Re-capture the identical Stage-2 profile only if the unprofiled candidate
   passes and exceeds baseline variance.
