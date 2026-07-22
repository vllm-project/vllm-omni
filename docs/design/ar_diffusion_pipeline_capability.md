# AR-Diffusion pipeline capability

> **Status:** experimental and non-normative. This note documents the current
> `experimental/ar_diffusion` contract. It follows the ownership and lifecycle
> direction in [#5137](https://github.com/vllm-project/vllm-omni/issues/5137),
> but it is not yet part of the stable module-design hierarchy.

`ARDiffusionEngine` is an opt-in diffusion backend for pipelines that keep
autoregressive attention KV across requests. The engine only selects
`ARDiffusionModelRunner`; the runner owns paged KV pools, session lookup, LRU
eviction, and failure cleanup.

A pipeline opts in by implementing `SupportsARDiffusionPipeline`. Its immutable
`ARDiffusionKVCacheSpec` reports worker-local KV geometry and policy:

- self-attention layers, TP-local KV heads, head size, and tokens per latent frame;
- latent frames committed by one model block, sliding window, and attention sink;
- logical KV branches mapped to worker-local storage slots;
- named fixed-length cross-attention KV, model-specific scratch geometry, and
  the pipeline's requested session-capacity upper bound.

The runner binds an `ARDiffusionKVState` only around one `forward()` call. The
pipeline may use it during that context but must not retain it. The session
survives subsequent requests with the same `session_id`. Request reset, explicit
close, LRU eviction, and forward exceptions all release runner-owned KV and call
the pipeline's reset/close lifecycle hook. Cross-attention KV is allocated lazily
per session and logical KV branch, and is released on the same path. A pipeline
publishes one named cache transactionally by yielding exactly one `(k, v)` pair
per layer:

```python
state.populate_cross_attention("main", "text", project_text_kv_by_layer())
```

The cache becomes readable only after the iterable completes successfully.
Logical branches remain independent even when they share a worker-local
`local_index`; that index only maps lazily executed self-attention and scratch
work to worker-local capacity slots. It does not define cross-attention cache
ownership.

Requests select the session with `extra_args["session_id"]`; setting `reset=True`
replaces it before the forward, while `close_session=True` releases it after a
successful forward. The runner also exposes `reset_session()` and
`close_session()` for an owning serving layer.

## Current execution limits

The experimental runner supports request-mode execution with
`max_num_seqs=1`. It rejects step execution and request-batch execution at load
time, and its inherited batch/step entry points also fail explicitly. Supporting
those paths requires a batch-aware state-binding contract rather than silently
bypassing `bind_ar_diffusion_state()`.

The current implementation retains one session per model instance. A request
for a different `session_id` closes the resident session through the same LRU
lifecycle used by explicit close and failure cleanup before allocating the new
one. Self-attention and cross-attention memory are therefore sized for one
resident session even if the pipeline declares a larger `session_capacity`.
This prioritizes rollout throughput and prevents block-pool exhaustion before a
count-based eviction can run. Multi-session residency can be added later with a
memory-backed capacity calculation or allocation-pressure eviction.

Scratch storage is also capability-driven. `frames_per_block` reserves space for
an uncommitted current video block, while `max_scratch_tokens_per_branch`
declares the maximum model-specific tokens, such as action/state registers, that
must coexist with it. `AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH` may increase
that derived minimum for deployment experiments but cannot reduce it.

## DreamZero adapter

DreamZero implements the capability directly on `DreamZeroPipeline`. Its adapter
reads DreamZero transformer geometry, describes the positive and negative CFG
branches, binds the state with a context manager, and translates the existing
boolean CFG calls to the explicit `"positive"` and `"negative"` KV branch names.
Its optional `ar_diffusion_warmup_requests()` provider owns the robot observation
schema and the one-frame/four-frame warmup convention; none of those details are
present in the generic runner.

Code migrating from the former experimental KV state API should use the
mechanical mapping below:

```python
# Before: state.pos, state.neg, state.get_kv_caches(is_negative, ...)
positive = state.adapter("positive")
negative = state.adapter("negative")
contexts = state.get_kv_caches("negative", seq_len=frame_tokens, commit_current=True)
```

Direct experimental `ARDiffusionKVCache` construction also migrates
mechanically: replace `local_branches` with explicit `kv_branches`,
`num_frame_per_block` with `frames_per_block`, and the fixed text/image length
arguments with a named `cross_attention_lengths` mapping plus
`session_capacity`. Normal engine users do not construct this object directly.

Cross-attention writers migrate from incremental publication:

```python
# Before
for layer_index, (key, value) in enumerate(project_text_kv_by_layer()):
    state.write_cross_attention_kv("text", layer_index, branch, key, value)
state.mark_cross_attention_populated(branch, "text")

# After
state.populate_cross_attention(branch, "text", project_text_kv_by_layer())
```

The iterable is consumed once and must yield exactly `num_layers` pairs. Failed
or incomplete projection leaves the previous complete cache unchanged, or
leaves the cache absent on its first population.

## Single-KV-branch pipeline sketch

A future single-branch causal DMD pipeline can opt in without changes to the
runner or KV core:

```python
class WorldPipeline:
    def ar_diffusion_kv_cache_spec(self):
        return ARDiffusionKVCacheSpec(
            num_layers=self.transformer.num_layers,
            num_kv_heads=self.transformer.tp_local_kv_heads,
            head_size=self.transformer.head_size,
            tokens_per_frame=self.transformer.tokens_per_latent_frame,
            frames_per_block=3,
            window_frames=self.transformer.sliding_window_frames,
            sink_frames=self.transformer.sink_frames,
            kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
            cross_attention=(ARDiffusionCrossAttentionKVSpec("text", self.text_length),),
            max_scratch_tokens_per_branch=self.transformer.max_register_tokens,
            session_capacity=1,
        )

    @contextmanager
    def bind_ar_diffusion_state(self, session_id, state):
        self.current_ar_state = state
        try:
            yield
        finally:
            self.current_ar_state = None

    def reset_ar_diffusion_session(self, session_id):
        self.model_states.pop(session_id, None)

    def close_ar_diffusion_session(self, session_id):
        self.model_states.pop(session_id, None)
```

The model writes/reads named cross-attention KV through the bound state and uses
the `"main"` KV branch for paged self-attention. Pipelines without a warmup
provider are loaded normally and skip AR rollout warmup.

## Relationship to the multi-stage KV manager RFC

[#5244](https://github.com/vllm-project/vllm-omni/issues/5244) describes a
future multi-stage KV transfer manager, connector integration, and cache-aware
scheduling. This capability has a narrower scope: it owns model-local paged KV
and session lifecycle inside one AR-Diffusion runner. It does not define
cross-stage transfer, connector behavior, or scheduler-visible cache affinity.

When the experimental runtime migrates toward #5244, the implementation should
reconcile its current full `KVCacheManager` use with the RFC's proposed
`BlockPool`-oriented DiT manager. The pipeline capability should remain the
model-facing geometry and state-binding boundary where possible, while the
experimental `ARDiffusionKVCache` class itself should not be treated as a stable
public API.
