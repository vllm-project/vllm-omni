# DreamZero Ulysses SP PR Handoff

Date: 2026-06-20

PR: https://github.com/vllm-project/vllm-omni/pull/4580

Base branch: `vllm-project/vllm-omni:main`

PR head branch: `TKONIY:draft/dreamzero-sp-ulysses`

Local branch used for the rebase: `draft/dreamzero-sp-ulysses-rebased`

Core implementation commit before this handoff doc: `4e3fce99 feat: add DreamZero Ulysses SP support`

## Goal

This PR adds Ulysses sequence parallel support for DreamZero while keeping the implementation small and compatible with the current `main` DreamZero code. The important constraint is that DreamZero already has a session-based KV cache, fused self-attention QKV projection, cross-attention cache, stepcache, and Cache-DiT related changes on `main`; this PR should layer SP support on top of those instead of reintroducing older DreamZero code.

## High-Level Design

DreamZero SP is implemented as video-token-only Ulysses sequence parallelism.

- Video tokens are sharded by sequence dimension through `sp_prepare`.
- Action and state tokens stay replicated and are appended after the local video shard.
- Self-attention participates in sequence parallelism.
- Cross-attention does not participate in SP and continues to run with replicated sequence inputs.
- KV cache is sharded by attention head under Ulysses, not by sequence.
- Ring SP with KV cache is intentionally not supported yet because its KV layout semantics are different.

## Key Files

### `vllm_omni/diffusion/models/dreamzero/causal_wan_model.py`

Main DreamZero model changes.

- Adds `_with_precise_bf16_matmul_reduction` around `_forward_blocks`.
  - This temporarily disables `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction` for bf16 block execution.
  - Reason: SP sharding changes GEMM shapes, and reduced precision bf16 reduction caused measurable numeric drift versus the single-rank baseline in tiny DreamZero precision tests.
- Adds `_dreamzero_sp_gather`.
  - Calls `sp_gather(..., validate=False)` and falls back to identity when no SP context is active.
- Keeps `WanT2VCrossAttention` and `WanI2VCrossAttention` with `skip_sequence_parallel=True`.
  - Do not flip these unless you also redesign cross-attn context/crossattn_cache sharding.
- Sets `CausalWanSelfAttention.attn` to `skip_sequence_parallel=False`.
  - This is the layer that should run through the active Ulysses strategy.
- Keeps main's fused `QKVParallelLinear` self-attention path.
  - Do not restore old separate `q`, `k`, `v` projections.
- Routes self-attention KV handling through `Attention.forward_with_kv_cache(...)`.
  - Action/state tokens are passed as `AttentionMetadata(joint_*, joint_strategy="rear")`.
  - Joint action/state KV is used for the live attention call but is not persisted into the video KV cache.
- Adds `DreamZeroSPPrepare` and `CausalWanModel._sp_plan`.
  - `sp_prepare` arg 0: `x_video`, split dim 1.
  - `sp_prepare` arg 1: `e_video`, split dim 1.
  - `sp_prepare` arg 2: `freqs`, split dim 0.
  - All use `expected_dims=3`, `split_output=True`, `auto_pad=False`.
- Adds `CausalWanModel.kv_cache_num_heads(ulysses_degree=...)`.
  - Returns local KV heads under Ulysses.
  - Raises if strict Ulysses degree does not divide local TP heads.
- In `_forward_blocks`:
  - Compute `e0 = self.time_projection(e)` before SP sharding.
  - Split only video `x/e0/freqs` through `self.sp_prepare`.
  - Append replicated action/state tokens after sharding.
  - Use local `seq_len` for block execution.
  - Gather only video tokens before `self.head`.
  - Use full unsharded `e[:, :video_seq_len]` for the output head.

### `vllm_omni/diffusion/models/dreamzero/pipeline_dreamzero.py`

Pipeline KV allocation changes.

- Adds `_get_runtime_ulysses_degree()`.
  - First tries `get_ulysses_parallel_world_size()`.
  - Falls back to the current diffusion config if parallel state is not initialized.
- `_prefill_kv_cache` now allocates DreamZero session KV cache with:
  - `num_heads = self.transformer.kv_cache_num_heads(ulysses_degree=ulysses_degree)`
  - This is required because under Ulysses the per-rank cache stores only local head shards.

### `vllm_omni/diffusion/attention/layer.py`

Generic attention changes used by DreamZero and covered by tests.

- Adds `append_attention_kv_cache(...)`.
  - Appends fresh K/V along sequence dimension and trims by `max_cache_len`.
  - Validates `[2, B, S, H, D]` layout.
- Adds `Attention.forward_with_kv_cache(...)`.
  - Runs the active parallel strategy first.
  - Removes live joint K/V before persisting the cache.
  - Appends only video K/V into the persistent KV cache.
  - Re-adds live joint K/V for the attention compute.
  - Runs local attention and then `strategy.post_attention(...)`.
- Explicitly rejects ring SP KV cache:
  - Raises `NotImplementedError("Ring sequence parallel KV cache layout is not implemented...")`.
  - This is intentional. Ring output may still work for non-cache attention paths, but DreamZero cached self-attn should not silently run with an invalid layout.

### `vllm_omni/diffusion/attention/parallel/ulysses.py`

Ulysses changes.

- Adds a packed Q/K/V all-to-all fast path for strict Ulysses when Q/K/V share shape, dtype, and device.
- Supports odd/non-divisible head counts in the UAA helper by padding heads before all-to-all and trimming after post attention.
- Carries joint-token metadata through the Ulysses context so `forward_with_kv_cache` can separate persistent video KV from live action/state KV.

### `vllm_omni/diffusion/attention/backends/sdpa.py`

SDPA test/debug controls.

- `DIFFUSION_SDPA_FORCE_FP32=1` casts Q/K/V to fp32 for SDPA and casts output back to the original dtype.
- `DIFFUSION_SDPA_FORCE_MATH=1` forces the math SDPA kernel.
- These are primarily for controlled precision/debug experiments, not a default performance path.

### Deploy Configs

Added:

- `vllm_omni/deploy/dreamzero_sp2.yaml`
- `vllm_omni/deploy/dreamzero_sp4.yaml`
- `vllm_omni/deploy/dreamzero_sp2_cfg2.yaml`

These are convenience configs for SP-only and SP+CFG experiments.

### Offline Export Example

`examples/offline_inference/dreamzero/export_prediction_video.py` now defaults `DIFFUSION_ATTENTION_BACKEND` to `TORCH_SDPA`, matching the DreamZero serving default and avoiding backend auto-selection surprises in offline export.

## Tests Added Or Updated

### Unit and contract tests

- `tests/diffusion/attention/test_sdpa_backend.py`
  - Covers SDPA fp32/math environment controls.
- `tests/diffusion/attention/test_ulysses_qkv_packing.py`
  - Covers packed Q/K/V all-to-all behavior.
- `tests/diffusion/models/dreamzero/test_causal_wan_sp_contract.py`
  - Covers DreamZero `_sp_plan`, `sp_prepare`, cross-attn/self-attn `skip_sequence_parallel`, fused QKV cached attention, joint metadata, KV append semantics, ring rejection, and local-head cache sizing.
- `tests/dreamzero/test_pipeline_state.py`
  - Adds regression coverage that `_prefill_kv_cache` allocates local Ulysses head count.
- Existing DreamZero tests:
  - `tests/dreamzero/test_qkv_fusion.py`
  - `tests/dreamzero/test_fused_qk_rms_norm.py`

### GPU precision tests

`tests/diffusion/models/dreamzero/test_causal_wan_sp_kv_cache.py`

Important tests:

- `test_dreamzero_cached_attention_ulysses_keeps_kv_cache_head_sharded`
- `test_dreamzero_cached_attention_ulysses_keeps_joint_tokens_out_of_kv_cache`
- `test_tiny_causal_wan_model_ulysses_matches_single_rank_with_actions`
- `test_tiny_causal_wan_model_ulysses_matches_single_rank_after_prefill`
- `test_tiny_causal_wan_model_ulysses_matches_single_rank_after_prefill_with_qk_norm`
- `test_tiny_causal_wan_model_ulysses_matches_single_rank_with_realistic_action_horizon`
- `test_tiny_causal_wan_i2v_ulysses_matches_single_rank_with_clip_and_y`

The tiny model tests compare SP outputs/caches against a single-rank baseline and are the most important correctness guard for future changes.

## Verification Already Run

After rebasing to `origin/main`, these commands passed locally:

```bash
pytest -q \
  tests/diffusion/attention/test_sdpa_backend.py \
  tests/diffusion/attention/test_ulysses_qkv_packing.py \
  tests/diffusion/models/dreamzero/test_causal_wan_sp_contract.py \
  tests/dreamzero/test_pipeline_state.py \
  tests/dreamzero/test_qkv_fusion.py \
  tests/dreamzero/test_fused_qk_rms_norm.py
```

Result: `29 passed`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 pytest -q \
  tests/diffusion/models/dreamzero/test_causal_wan_sp_kv_cache.py \
  -k 'tiny_causal_wan_model_ulysses_matches_single_rank_with_actions or tiny_causal_wan_model_ulysses_matches_single_rank_after_prefill or tiny_causal_wan_model_ulysses_matches_single_rank_after_prefill_with_qk_norm'
```

Result: `4 passed, 8 deselected`.

Environment note from those runs:

- 6 NVIDIA RTX PRO 6000 Blackwell-class GPUs were visible.
- There was a repository warning about vLLM and vLLM-Omni minor version mismatch:
  - vLLM-Omni: `0.21.0rc2.dev235+g32ca6e515.d20260601`
  - vLLM: `0.22.0`
  - The tested commands still passed.

## Known Risks And Non-Goals

### Ring SP

Ring SP is not implemented for DreamZero KV-cache attention in this PR. The code intentionally raises in `Attention.forward_with_kv_cache` when `self.use_ring` is true. Do not remove that guard unless you implement and test the correct ring KV cache layout.

Reason: Ulysses stores the persistent cache sharded by local heads after all-to-all. Ring uses different sequence/communication semantics, and silently sharing the Ulysses cache layout for ring would be wrong.

### Cross-Attention SP

Cross-attention stays `skip_sequence_parallel=True`.

Reason: DreamZero context/crossattn_cache is short and session-invariant. Sharding cross-attn would require rethinking cached text/image K/V layout and its interaction with replicated action/state tokens. The current PR only shards self-attn video sequence tokens.

### Action/State Tokens

Action/state tokens are intentionally not stored in the session KV cache. They are live joint tokens for the current step only. The persistent KV cache is video-only.

If a future change stores action/state KV persistently, it must update:

- `Attention.forward_with_kv_cache`
- DreamZero cache shape allocation
- tiny SP precision tests
- joint-token cache exclusion tests

### bf16 Precision

The `_with_precise_bf16_matmul_reduction` wrapper exists because SP changes GEMM partitioning, and bf16 reduced precision reduction caused SP-vs-single-rank drift in controlled tests. If you remove it for performance, first reproduce precision with the tiny GPU tests and the real DreamZero video comparison workflow.

### PR Shape

The PR was rebased to latest `origin/main` on 2026-06-20. DreamZero is already upstream on `main`, so the PR should stay focused on SP support and should not replay old DreamZero integration commits.

## Recommended Commands For The Next Agent

Start with:

```bash
git fetch origin main
git fetch tkoniy draft/dreamzero-sp-ulysses
git checkout -B draft/dreamzero-sp-ulysses tkoniy/draft/dreamzero-sp-ulysses
```

Then inspect:

```bash
git diff origin/main...HEAD --stat
git diff origin/main...HEAD -- vllm_omni/diffusion/models/dreamzero/causal_wan_model.py
git diff origin/main...HEAD -- vllm_omni/diffusion/attention/layer.py
```

Run fast tests:

```bash
pytest -q \
  tests/diffusion/attention/test_sdpa_backend.py \
  tests/diffusion/attention/test_ulysses_qkv_packing.py \
  tests/diffusion/models/dreamzero/test_causal_wan_sp_contract.py \
  tests/dreamzero/test_pipeline_state.py \
  tests/dreamzero/test_qkv_fusion.py \
  tests/dreamzero/test_fused_qk_rms_norm.py
```

Run GPU precision checks when GPUs are available:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 pytest -q \
  tests/diffusion/models/dreamzero/test_causal_wan_sp_kv_cache.py \
  -k 'dreamzero_cached_attention_ulysses or tiny_causal_wan_model_ulysses'
```

For a shorter GPU check:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 pytest -q \
  tests/diffusion/models/dreamzero/test_causal_wan_sp_kv_cache.py \
  -k 'tiny_causal_wan_model_ulysses_matches_single_rank_with_actions or tiny_causal_wan_model_ulysses_matches_single_rank_after_prefill or tiny_causal_wan_model_ulysses_matches_single_rank_after_prefill_with_qk_norm'
```

## What To Watch When Modifying The PR

1. Do not change DreamZero self-attn back to separate `q/k/v`; `main` uses fused `QKVParallelLinear`.
2. Do not make cross-attn SP-active unless you also implement crossattn_cache sharding and precision tests.
3. Do not persist joint action/state tokens into KV cache unless tests are rewritten for the new semantics.
4. Do not enable ring KV cache without a new layout design and tests.
5. If changing `_sp_plan` or `sp_prepare`, rerun both CPU contract tests and GPU tiny precision tests.
6. If changing SDPA fp32/math controls, keep them opt-in environment controls.
7. If changing bf16 precision handling, verify with GPU SP-vs-single-rank comparisons before claiming success.

## Local Workspace Notes

At the time this file was written, the local workspace also had unrelated untracked files that were intentionally left untouched:

- `docs/design/dreamzero_optimization_rfc.md`
- `docs/superpowers/`
- `vLLM-Omni Feature Design Doc Concise.md`

Do not accidentally add these to the PR unless the user explicitly asks for them.

## Performance Benchmarks (Parallelism Matrix)

Measured on this branch to compare TP / Ulysses-SP / CFG-parallel for DreamZero.

### Setup
- Model: `GEAR-Dreams/DreamZero-DROID` (14B, Wan2.1-I2V based), resolution 180×320, 4-frame AR chunks.
- Hardware: NVIDIA A100-SXM4-80GB. Software: vLLM 0.22.0 (+cu129) / torch 2.11+cu129, driver 555.
- Mode: **eager** (`enforce_eager=True`, forced by the offline export path), **step_cache disabled**,
  batch 1, `DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA`.
- Workload: 1 prefill + 12 AR chunks; metric = steady-state mean per-chunk latency over chunks[2:].
- Configs: `vllm_omni/deploy/dreamzero_{sp2,sp4,sp2_cfg2}.yaml` (SP/SP+CFG; the TP and TP×SP/TP×CFG
  runs used ad-hoc configs not shipped here); harness `examples/offline_inference/dreamzero/bench_dreamzero.py`.

### Results

| config   | GPUs | parallelism    | load (s) | steady mean (ms) | p90 (ms) | chunks/s | vs TP2 |
|----------|------|----------------|----------|------------------|----------|----------|--------|
| tp2      | 2    | TP2            | 73       | 8269             | 8777     | 0.121    | 1.00x  |
| sp2      | 2    | SP2 (ulysses2) | 75       | 8217             | 8648     | 0.122    | 1.01x  |
| tp4      | 4    | TP4            | 92       | 6347             | 6600     | 0.158    | 1.30x  |
| sp4      | 4    | SP4 (ulysses4) | 144      | 5527             | 5724     | 0.181    | 1.50x  |
| tp2_sp2  | 4    | TP2 x SP2      | 111      | 6122             | 6342     | 0.163    | 1.35x  |
| tp2_cfg2 | 4    | TP2 x CFG2     | 102      | 4998             | 5191     | 0.200    | 1.66x  |
| sp2_cfg2 | 4    | SP2 x CFG2     | 152      | 4981             | 5149     | 0.201    | 1.66x  |

### Takeaways
- **2-GPU: SP2 ≈ TP2** (8.22 vs 8.27 s/chunk) — Ulysses SP is a no-regression alternative to TP.
- **4-GPU single axis: SP4 (1.50x) > TP4 (1.30x)** — SP scales more efficiently (2→4 efficiency: SP 74%, TP 65%).
- **CFG-parallel is the most efficient axis** — TP2xCFG2 and SP2xCFG2 both 1.66x (fastest); SP composes with CFG as well as TP.
- TP2xSP2 (1.35x) sits between TP4 and SP4 → pure SP4 beats the TP/SP hybrid here.
- **Load time**: TP shards weights (TP4 92 s) vs SP/CFG replicate the full 14B per rank (SP2xCFG2 152 s).

### Notes on absolute latency
These are **conservative** numbers: eager + step_cache-off were chosen so the matrix isolates the
parallelism axis; they are not production speed.
- A100 raw bf16 GEMM measured 238–265 TFLOPS (~80% of peak); SM utilization during generation was
  bursty but low on average (mean ~18%) — eager-mode launch overhead starves the GPU, it is not compute-bound.
- Enabling **step_cache alone** (TP2 + `cache_backend: step_cache`) cut per-chunk latency
  **8269 → 3004 ms (2.75x)**. Production (step_cache + CUDA-graph where supported) is expected ~2.5–3x
  faster than the table above, with the same relative ordering between strategies.

### Environment caveat
GPU index 3 on the benchmark node is hardware-faulty (CUDA OOM on any allocation despite reporting free);
4-GPU runs were measured on physical GPUs 0,1,2,4. The committed configs use the standard `devices: "0,1,2,3"`.
