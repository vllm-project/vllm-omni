# Attention diagnosis and backend recommendations

## Eligibility before speed

Read the selected revision's `vllm_omni/diffusion/attention/selector.py`, actual
backend and model processor; use the native attention benchmark where semantics
match. Record device architecture, dtype/accumulation, batch, Q/K lengths, Q/KV
heads, head dimension, mask/causal/varlen behavior, strides, scale and output layout.
Include CFG batching, graph/compile and TP/SP requirements.

Configured backend is not proof of executed backend. Verify dispatch and kernel
identity, including fallback. Already using a recent backend does not establish
near-optimality; a recent backend is not necessarily faster for this shape.

## Decompose the complete attention region

Separate QKV projection, norm/RoPE, packing/layout, attention core and output
projection. Existing fusion may cross these boundaries. State whether a measured
reference covers the core or the whole region. Include distributed exchanges in
L5 rather than attributing every long attention call to the local kernel.

For dense equal Q/K/V head dimension, useful matmul work is approximately
`4*B*Hq*Lq*Lk*D` for full attention, excluding softmax and other non-matmul work.
Causal/masked/unequal-value-dimension cases need the actual pair/work model.
Useful versus padded/executed work must remain distinct. Do not use a pure GEMM
MFU target as an attention stop threshold.

## Root-cause procedure

1. Compare matched current/reference shapes and semantics using normal timing.
2. Inspect grid/tiles and workload size: short sequences, few heads and tails can
   limit parallelism even with a well-implemented kernel.
3. Inspect matmul versus non-matmul instruction pressure, reduction/barrier waits,
   eligible work and data movement. An attention kernel is not uniformly GEMM.
4. Inspect memory footprint and whether the implementation materializes avoidable
   scores or introduces wrapper copies. Return structural findings to L3.
5. Check graph/eager selection, cache treatment and layout conversion before
   attributing a gap to the attention algorithm itself.
6. For SP, align ranks and consumer dependencies; local attention can be efficient
   while all-to-all, ring exchanges or readiness dominate the enclosing phase.

## Backend recommendation decision

Recommend a comparison only when an alternative is compatible with the exact
semantics and there is a concrete reason it might address the observed limiter.
Prefer available supported paths; absent dependencies become a proposed experiment,
not automatic installation or deployment changes.

Record alternatives considered and rejected: unsupported masks/dimensions, required
packing, incompatible graph/parallelism, different precision, or evidence of no
local advantage. If current FA3 is near its matched reference, investigate exposed
wrappers or work composition rather than recommending "switch to FA3" again.

Acceptance proposal: core numerical comparison -> complete attention-region timing
including transforms -> unprofiled E2E A/B -> full image/video quality gate. Local
numerical tolerance alone does not certify generated-media quality.

## Research interpretation

[FlashAttention](https://arxiv.org/abs/2205.14135) motivates examining materialized
traffic; [FA2](https://arxiv.org/abs/2307.08691) motivates work partitioning and
non-matmul cost; [FA3](https://arxiv.org/abs/2407.08608) motivates Hopper-specific
asynchronous pipelines. None makes a backend name or published efficiency a
completion criterion for this workload.
