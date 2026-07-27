# Stage B — GPU validation TODO (Cosmos3 session-state port, RFC #4480)

These changes were authored on a **CPU-only box (no torch/vllm)**, so only
syntax (`py_compile`) and standalone adapter logic were checked. The following
**must run on a GPU box (H100 / L40S) with the full env** before opening the PR.

## What changed (all gated behind `enable_session_state_manager`, default off)

- `state_cosmos3_adapter.py` (new) — `Cosmos3StateAdapter`: per-(layer,branch)
  model-specific `Cosmos3UNDKV`, session-keyed; **no dense-KV generalization
  in the shared contract**.
- `tests/cosmos3/test_session_memory_equivalence.py` (new) — CPU adapter tests
  (equivalence + concurrency isolation).
- `pipeline_cosmos3.py` — opt-in in `__init__`; `_get_or_create_cosmos3_state` +
  `_kv_load_und` / `_kv_capture_und` / `_kv_reset_und`; `diffuse()` gains
  `session_id` and routes the CFG cond/uncond UND-K/V swap through the adapter;
  `forward` passes `session_id=req.request_id`.
- `transformer_cosmos3.py` — `freqs_gen` recomputed when absent (`need_kv or
  cached_freqs_gen is None`) so the session path can store K/V only. **Bespoke
  path byte-identical** (it sets cached_kv + cached_freqs_gen together).

## Must verify on GPU

1. **CPU unit tests pass:** `pytest tests/cosmos3/test_session_memory_equivalence.py -v`
   (needs torch installed; no GPU needed for these).
2. **Default-path regression = byte-identical** (flag OFF): same prompt/seed,
   T2V + I2V, current `main` vs this branch → output latents `max_abs_diff == 0`.
   This is the critical check that the `diffuse()`/transformer edits didn't change
   the default path.
3. **Flag ON bit-equivalence:** `enable_session_state_manager=true`, same
   prompt/seed → output identical to flag OFF (freqs recompute is lossless;
   per-layer dict store/rebuild is lossless).
4. **Concurrency:** two interleaved requests with different prompts, flag ON →
   each output matches its own single-request run (proves the no-keying clobber
   is fixed). Compare against flag OFF where the shared transformer-instance
   cache can corrupt under concurrency.
5. **Perf note:** flag ON recomputes `freqs_gen` each denoise step; measure the
   host-side delta. If significant, the documented fallback is a per-generation
   transient (still not stored in session memory).

## All 3 CFG paths are wired — verify each

`diffuse()` has three branches; session keying is now wired in all three (commit
e814d9c1). Verify bit-equivalence (flag on vs off) **for each**:
- **cfg-parallel** (`cfg_world_size > 1`, multi-GPU): each rank runs one branch
  (rank 0 cond / else uncond); load/capture that rank's branch. Needs ≥2 GPUs to
  exercise; check no cross-rank/branch mixing.
- **sequential CFG** (single GPU, `guidance_scale > 1`): the main Stage B path.
- **no-CFG** (`guidance_scale <= 1`): single cond branch.
- Also verify: with flag ON, no stale UND K/V leaks across two back-to-back
  generations in the cfg-parallel / no-CFG paths (the bug e814d9c1 fixed) —
  run gen A then gen B with different prompts, B must not inherit A's cache.

## Known stubs (Stage C/D, not this PR)

- `_kv_*` raise `NotImplementedError` when `self._bde_kv_state` is set (the BDE
  paged pool path); that requires the reference `dual_kv_cache` ported into main
  and the BDE branch confirmed with @tzhouam.
