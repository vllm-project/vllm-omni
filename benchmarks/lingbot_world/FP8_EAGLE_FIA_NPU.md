# Dense FP8 EagleFIA preparation

Experimental opt-in, default disabled. In the service process environment set:

```bash
export VLLM_OMNI_AR_DIFFUSION_FP8_FIA_MODE=C8V16_TILING512
```

This selects MindIE-SD `quant_attention(precision="fp8",
fp8_fa_mode=FP8FAMode.C8V16_TILING512)` after AR-Diffusion has gathered the valid
self-attention K/V window. The TND singleton inputs are adapted to BNSD and the
result restored to TND. Cross-attention is not changed. Keep the environment
unset for BF16 attention. It applies to the service's NPU AR-Diffusion paged
attention backend; validate only with the LingBot geometry before broader use.

MindIE-SD's coupled mode binds Q/K/V quant modes `7/7/12`, `inner_precise=4`,
Q128/K256 sequence blocks, and V512x64 blocks. Do not independently override
these settings or infer that all three tensors use512-token tiles.

The path calls dense FIA without any SparseBlockEstimate or block mask. This
curated branch does not include EQBSA. Causal attention is rejected.
Missing provider/API/mode fails explicitly rather than silently running BF16.
BF16 KV-cache storage and the separate Linear MXFP8 policy remain unchanged;
Q/K/V quantization and TND/BNSD conversions are incurred on each invocation.

Four CPU ABI tests cover rectangular/tail lengths, exact mode/scale propagation,
output layout and unsupported dtype. These tests mock MindIE and do not validate
native kernel availability, accuracy, or performance. Verified native and
four-card results, provider revision and precision limits are recorded in
[NPU_OPTIMIZATIONS.md](NPU_OPTIMIZATIONS.md). Previously deployed isolated
providers may predate the mode and require a version-matched rebuild.

Follow-up benchmark should compare BF16 FA, KV-index-copy-only, and dense FP8
EagleFIA with identical model settings, and report quantization/layout overhead
separately. Preserve short and long KV cases, three timing trials, independent
profiling, and output accuracy. Do not combine two changes in one attribution.
