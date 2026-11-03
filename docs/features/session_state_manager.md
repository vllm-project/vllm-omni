# Session State Manager (experimental)

!!! warning
    This feature is **experimental** (RFC [#4480](https://github.com/vllm-project/vllm-omni/issues/4480)).
    The code lives under `vllm_omni/experimental/world_models/` and its APIs may
    change without notice. It is **off by default**; when disabled, model
    behavior is byte-for-byte unchanged.

Autoregressive diffusion world models (DreamZero, Cosmos3, and similar) keep
per-session state across `forward()` calls: accumulated video latents, frame
stitching buffers, encode-once conditioning, counters, and reset heuristics.
Historically each model hand-rolled this state. The session state manager
extracts it into a shared, typed contract:

- `StateObject` — one typed unit of session state (e.g. `LatentBuffer` for
  accumulated latents and bounded rings), with a uniform lifecycle
  (`allocate` / `commit` / `view` / `reset` / `evict` / `nbytes_by_device`).
- `SessionStateManager` — maps `session_id -> {name: StateObject}`, caps the
  number of retained sessions by evicting the least recently used one first
  (matching the bespoke caches it replaces), and reports per-device byte usage
  via `stats()` for observability.

Autoregressive generation KV is *not* managed here: for DreamZero it is owned
by the AR-Diffusion engine's paged KV pool (PR
[#4534](https://github.com/vllm-project/vllm-omni/pull/4534)). Cosmos3's fixed,
encode-once UND conditioning K/V is different and is covered by the dense
Cosmos3 adapter described below. Integrating the engine's per-session paged KV
handle into the same byte accounting is RFC #4480 Phase 1.

## Enabling

Via config:

```yaml
# deploy config
enable_session_state_manager: true
```

or programmatically with `OmniDiffusionConfig(enable_session_state_manager=True)`.

Via environment variables (no config change needed):

```bash
export OMNI_DIFFUSION_SESSION_STATE_MANAGER=1            # 1/0/true/false/yes/no/on/off
export OMNI_DIFFUSION_SESSION_STATE_MANAGER_MAX_SESSIONS=64   # optional, positive int
```

**Precedence:** a *set* environment variable overrides the config value, in
both directions — `OMNI_DIFFUSION_SESSION_STATE_MANAGER=0` force-disables the
manager even if the config enables it. Unset or unparsable values fall back
to the config (an unparsable `MAX_SESSIONS` logs a warning and is ignored).

**Why both a config field and an environment variable?** The config field is
the API. The environment override exists for A/B equivalence validation: the
manager-backed path must be bit-identical to the bespoke path, and the way to
verify that is to run the *same* deploy config twice with only the flag
flipped per process — no config-file edits, so the two runs cannot drift
apart. (Same pattern as the `DIFFUSION_CACHE_BACKEND` fallback.)

**Why is `MAX_SESSIONS` not a config field?** The retained-session cap is not
a public tuning knob: it mirrors the bespoke `MAX_DREAMZERO_SESSIONS = 64`
constant (which is not configurable either), so the manager-backed path evicts
exactly like the path it replaces. The environment variable exists only to
stress eviction in tests and experiments. It is deliberately not promoted to
the config, because RFC #4480 Phase 1 replaces count-based capping with a byte
budget — that budget, not this cap, is the knob that deserves a config field.

## Choosing between an attribute and a `StateObject`

A session holds its values in two places. Named `StateObject`s live in the
session's object map; everything else lives in `SessionState.attrs`, declared
one line per value with the `SessionAttr` descriptor. The rule for choosing is:

> A value belongs in a `StateObject` when the **manager** must be able to act on
> it — release it under budget pressure, accumulate into it, stage and commit
> it, or rebuild it from a recorded source. It belongs in `attrs` when the
> manager only needs to **carry** it: the model assigns it wholesale, reads it
> back, and clears it, and no one but the model can decide its fate.

Byte accounting is independent of that choice. A session reports every byte it
holds in either bucket, because the number describes what the process is
holding, not what the manager may do about it.

The consequence is worth stating plainly: **a value left in `attrs` is a value
the byte-budget planner cannot free.** In Phase 0, which accounts but does not
enforce, that is legitimate — but it should be a recorded decision rather than
an accident, which is what the table below records.

Note what the rule does *not* say. "Is it big" is not the criterion, and
neither is "does it grow". The largest allocation a DreamZero session holds —
`vae_enc_feat_map`, 603 MiB — is an *attribute*, because nothing the manager
can do to it is useful: it is replaced wholesale, it has no recompute source,
and freeing it mid-session breaks the encoder stream. Ending the session is the
only release available, and that reclaims attributes anyway. Size alone argues
for counting a value, which the accounting does regardless of bucket; it does
not argue for a lifecycle.

What would change that answer is a release the manager can perform without
losing the value — offloading it to host memory and copying it back is
transparent, needs no recomputation, and suits a large contiguous cache. The
contract has no such operation: `evict()` frees storage, it does not park it.
So the bucket follows the operations that exist, and moves if they do.

### Byte accounting

`SessionStateManager.stats()` reports byte totals keyed by device
(`nbytes:cuda:0`, `nbytes:cpu`) alongside `total_nbytes`. The split is not
cosmetic: device memory is the contended resource and host memory generally is
not, so a single figure covering both answers no question. Which pool a budget
applies to is a question for whoever enforces it, and is not decided here.

Two details make the numbers honest. Bytes are counted per **storage**, not as
`numel() * element_size()`, so a narrow slice reports the whole allocation it
keeps alive rather than the part it exposes. And storages are **deduplicated**
across the whole session, so two views of one buffer — or an alias between an
attribute and a state object — are counted once.

## Scope and guarantees

- **Opt-in and equivalent.** With the flag off, models use their bespoke state
  paths unchanged. With the flag on, the manager-backed path is validated
  bit-identical to the bespoke path (CPU equivalence tests under
  `tests/dreamzero/test_session_state_equivalence.py`, plus GPU A/B runs).
- **Session eviction is count-based (LRU).** Evicting a session drops it from
  the lookup table only; an adapter still holding the session keeps its state
  (matching bespoke behavior, where the caller holds the state object).
- **Byte budget is recorded, not enforced.** `SessionStateManager.stats()`
  reports per-manager, per-device byte totals; budget *enforcement* (and
  eviction driven by it) is a later phase of RFC #4480.

## Supported models

### DreamZero

Sizes measured on one A100 at the shipped deploy config (`180x320` per camera,
three cameras stitched to `352x640`, bfloat16), over a 400-step session.

| Value | Bucket | Size | Notes |
|---|---|---|---|
| `vae_enc_feat_map` | `attrs` | **603 MiB** | Wan encoder causal-conv cache, 24 entries. Constant from ~step 50. The largest thing a session holds: 64 sessions is 37.7 GiB. Counted, not releasable — see the rule above. |
| `video_latents_across_time` | `LatentBuffer` | grows | AR video latent chunks, host memory (`.cpu()`), kept for decode. |
| `stitched_buffer` | `LatentBuffer` | bounded | Stitched pixel frames, host memory, ring of `FRAMES_PER_CHUNK`. |
| `vae_encoder_out` | `LatentBuffer` | 0.43 MiB | Bounded ring of `num_frame_per_block` latent frames — see below. |
| `clip_feas`, `ys`, `language`, `prompt_embeds` | `attrs` | small | Conditioning tensors, written once per session and reread. |
| `vae_pending_body_frames` | `attrs` | ≤2.6 MiB | Raw frames awaiting a 4-frame body chunk; bounded by construction. |
| `call_count`, `current_start_frame`, `vae_stream_initialized` | `attrs` | — | Counters and flags. |

Attention KV does not appear here: it is engine-owned (PR #4534).

**The bounded VAE encoder ring.** The bespoke state concatenates every encoded
chunk into one tensor and clears it only on a *session* reset — a window reset
deliberately keeps it — so it grows for as long as the session lives, at
0.215 MiB per latent frame (one per four steps). Its only reader,
`_vae_stream_get_observation_latents`, quantises the whole history and then
keeps the last `num_frame_per_block` latent frames. Since `quant_conv` is a
`WanCausalConv3d` of kernel size 1, pointwise in time, those frames do not
depend on the discarded ones. The manager-backed state therefore retains a ring
of `num_frame_per_block` frames instead: identical output, bounded memory, and
the per-step quantisation no longer scales with session length. Asking for more
frames than the ring retains raises rather than silently returning a shorter
window. The bespoke path is unchanged.

### Cosmos3

When `enable_session_state_manager` is enabled, the Cosmos3 adapter stores the
dense, encode-once UND text K/V for every transformer layer and active CFG
branch, together with that branch's GEN M-RoPE cosine/sine tensors. These are
GPU-resident tensors and can contribute to CUDA OOM. A useful description of
the retained storage on one rank is:

```text
sum_over_branches(
  sum_over_layers(storage_bytes(K) + storage_bytes(V))
  + storage_bytes(freqs_gen_cos) + storage_bytes(freqs_gen_sin)
)
```

This is not a peak-memory upper bound: construction of the next branch can
temporarily overlap the already retained branch with UND activations, the new
full K/V result, RoPE tensors, and generator activations. K/V slices also keep
their underlying allocation alive, so physical storage can follow the padded
text length rather than only the exposed trimmed length. The manager's byte
accounting deliberately measures storage for this reason.

Branch placement changes the per-rank cost:

- no CFG retains one conditional branch;
- sequential CFG can retain both conditional and unconditional branches on the
  same GPU;
- CFG-parallel retains one branch on each CFG rank;
- UND K/V is replicated across sequence-parallel ranks rather than divided by
  the SP degree.

Consequently, risk increases with text batch/sequence length, media token count
(which sizes `freqs_gen`), sequential CFG, dtype width, and concurrent live
requests. Phase 0 records `nbytes:cuda:*` but does not enforce `byte_budget`, so
the manager cannot prevent an allocation from exhausting the device. The
count-based session limit is not an OOM guarantee.

The adapter transfers ownership by tensor reference; it does not clone K/V or
RoPE tensors. For a normal serial request this should not materially increase
tensor storage relative to Cosmos3's existing bespoke cond/uncond cache, which
retains the same branch values. It does add small CPU-side session/object
metadata. This claim is based on the reference flow in the implementation;
there is not yet a current-head GPU peak-memory A/B result for this PR.

Request-finally cleanup drops the manager-owned session, so branches do not
accumulate with completed request IDs. The transformer handoff field still
references the last loaded branch until `reset_cache()` runs at the start of
the next generation (or the pipeline is destroyed). Thus this is bounded
post-request residency, not growth per completed request, but manager
`stats()` after `drop_session()` no longer includes that last transformer-held
alias. A profiler that only reads manager statistics can therefore under-report
live process memory.

For OOM validation, measure complete cache-off and true flag-on generations in
otherwise identical fresh processes. Synchronize CUDA and report process-level
peak memory plus PyTorch peak allocated and peak reserved memory. Include the
period between requests when checking cleanup; do not infer safety from the
manager byte total, a single denoising step, or an operator-only profile.
