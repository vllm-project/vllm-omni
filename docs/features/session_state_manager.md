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
  accumulated latents, `EncodeOnceKV` for write-once conditioning K/V), with a
  uniform lifecycle (`allocate` / `commit` / `view` / `reset` / `nbytes`).
- `SessionStateManager` — maps `session_id -> {name: StateObject}`, caps the
  number of retained sessions by evicting the least recently used one first
  (matching the bespoke caches it replaces), and reports byte usage via
  `stats()` for observability.

Attention KV is *not* managed here: for DreamZero it is owned by the
AR-Diffusion engine's paged KV pool (PR
[#4534](https://github.com/vllm-project/vllm-omni/pull/4534)). The manager
covers the model's non-KV session state; integrating the engine's per-session
KV handle into the same byte accounting is RFC #4480 Phase 1.

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

## Scope and guarantees

- **Opt-in and equivalent.** With the flag off, models use their bespoke state
  paths unchanged. With the flag on, the manager-backed path is validated
  bit-identical to the bespoke path (CPU equivalence tests under
  `tests/dreamzero/test_session_state_equivalence.py`, plus GPU A/B runs).
- **Session eviction is count-based (LRU).** Evicting a session drops it from
  the lookup table only; an adapter still holding the session keeps its state
  (matching bespoke behavior, where the caller holds the state object).
- **Byte budget is recorded, not enforced.** `SessionStateManager.stats()`
  reports per-manager byte totals; budget *enforcement* (and eviction driven
  by it) is a later phase of RFC #4480.

## Supported models

| Model | State routed through the manager |
|---|---|
| DreamZero | Frame stitching buffer, accumulated video latents (`LatentBuffer`), prompt-embed cache, incremental VAE encoder stream, counters and reset heuristics |
