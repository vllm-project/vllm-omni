# Automatic Prefix Caching in Omni Models

---

## Table of Contents

- [Overview](#overview)
- [High-Level Approach](#high-level-approach)
- [Example](#example)
- [What About Multimodal Inputs?](#what-about-multimodal-inputs)
- [Implementation](#implementation)
- [Related Files](#related-files)

---

### Overview

Prefix caching in the context of kv-cache management is a useful optimization for avoiding redundant computations. The main idea is that we store portions of the kv-cache from processed requests, so that we can reuse them if incoming requests have the same prefix as previous requests.

vLLM manages the kv-cache as blocks, which represent a span of tokens of a fixed length. Blocks are hashable by the content that they contain, which typically means the tokens within the span, but also could be influenced by other factors, e.g., LoRA and multimodal data.

vLLM implements automatic prefix caching for managing its kv-cache, which is best understood by reading the design document [here](https://docs.vllm.ai/en/latest/design/prefix_caching/). vLLM-Omni builds on top of the prefix caching mechanism in a noninvasive way to allow caching between stages in Omni pipelines. This typically means for a given stage we aim to support caching for the following:

- The last hidden states produced by the stage
- Model / stage specific multimodal data

!!! note "Note 1"
    This document describes vLLM-Omni's mechanism for caching tensor outputs that are meant to be passed between stages, when requests have common prefixes, similar to the way in which vLLM has prefix caching for the kv-cache. This works in conjunction with vLLM's multimodal encoder caching, but is distinct. See the final section for a concrete example for how they tie together in practice.

### High-Level Approach

!!! note "Note 2"
    Prior to reading this section, it's recommended to take a look at the design documents in vLLM for [Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/), which will make some of the concepts more clear.

The main focus of vLLM-Omni's approach to prefix caching stage outputs is to build on vLLM's prefix caching in the least invasive way possible while minimizing impact for cache misses, and consuming a minimal amount of GPU memory. To understand the implementation, there are a few important things to note:

- Between stages, device tensors are generally moved to CPU; this is important since we're just caching the outputs of stages, so it is okay to keep the entire cache on the CPU.

- For a tensor to be considered cacheable, the first dimension (currently) needs to be the same as the token count, as it allows us to reuse block/slot mappings for our externally maintained tensor caches. This allows us to dynamically discover the tensors to be marked as cacheable outputs in each Omni model without having to explicitly specify cacheable output field names in every model.

With this in mind, consider the set of blocks in a 2D layout, where the row represents the index of blocks being considered, and the columns represent the slots corresponding to tokens within each block. Since we know the `num_blocks` and `block_size` from our kv cache config, if we want to cache a tensor with feature size `D`, we can preallocate a CPU tensor of size `(num_blocks, block_size, D)`, and use the same block index and slot mapping to retrieve the corresponding feature vector.

### Example

!!! note "Note 3"
    Prefix caching in vLLM-Omni currently is only supported on AutoRegressive stages with one kv-cache group. Configure it with the pipeline-wide `enable_prefix_caching` field in the deploy config.

The way in which vLLM-Omni ties into vLLM's prefix caching is best understood by example. Say that we have the following:

- `num_blocks=8`
- `block_size=4`
- `hidden_size=2`
- A stage specific multimodal output tensor named `mm_feature` with feature dimension `16`

The prefix cache flow is then outlined below.

1. When the model is initialized, we can determine the `hidden_size` from the `ModelConfig`, and allocate a cache of size `(num_blocks, block_size, hidden_size)`.

2. Say we process the request `The quick brown fox was tired and slept beneath the shady tree`, which is 12 tokens and evenly divides into 3 blocks as shown below.

```text
         [  The quick brown fox  ] [  was tired and slept ] [beneath the shady tree ]
Block 1: |<--- block tokens ---->|
Block 2: |<------- prefix ------>| |<--- block tokens --->|
Block 3: |<------------------ prefix -------------------->| |<--- block tokens ---->|
```

When the request processes, we inspect the multimodal outputs and identify the `mm_feature` tensor, which will be of shape `(seq_len, feature_dim)`, i.e., `(12, 16)` in this example. We note that the first axis is dependent on the `seq_len` and add a new cache_tensor of shape `(num_blocks, block_size, feature_dim)` to our multimodal cache for tensors.

1. If we lay out the cache as a 2D tensor of shape (`num_blocks`, `block_size`), we'll have something like the following:

```text
0: [  The quick brown fox  ]
1: [  was tired and slept  ]
2: [beneath the shady tree ]
3: [EMPTY]
...
7: [EMPTY]
```

Or, if we flatten it down to 1D,

```text
0: The
1: quick
2: brown
3: fox
...
11: tree
12: [EMPTY]
...
```

which we can think of as row indices into the hidden states tensor if we view it as the 2D shape `(num_blocks x block_size, feature_dim)`. That is, the analogous flattened (from 3D -> 2D) mapping of the cache for hidden states becomes the following.

```text
0: <hidden states vector of len 2 corresponding to 'The'>
1: <hidden states vector of len 2 corresponding to 'quick'>
2: <hidden states vector of len 2 corresponding to 'brown'>
3: <hidden states vector of len 2 corresponding to 'fox'>
...
11: <hidden states vector of len 2 corresponding to 'tree'>
12: [EMPTY]
...
```

Similarly, for the multimodal outputs cache, the flattened coordinates are the same, but the `mm_feature` maps to vectors of length `16` instead of the hidden size of `2`. Note that in practice, we may have multiple  multimodal output tensors per forward pass, which may have different names and different feature dimensions.

1. Now, say that we receive a new request `The quick brown fox jumped over the dog`.

```text
         [  The quick brown fox  ] [  jumped over the dog ]
Block 1: |<--- block tokens ---->|
Block 2: |<------- prefix ------>| |<--- block tokens --->|
```

Here, we will have a cache hit for `Block 1` which will be detected by vLLM based on the hash of the first block when it's handling the prefix caching on the kv-cache. As a result, when we get the output from the scheduler, we will see that `num_computed_tokens=4` (corresponding to the cached first block), and we only need to process the remaining 4 new tokens in the new prefill.

Since we have the block indices / slot mappings from the kv cache manager, we can simply mirror the mappings and leverage the same indices for the cached hidden states and multimodal outputs. This allows us to look up the correct tensors from our externally maintained 3D caches.

```text
0: [  The quick brown fox  ] < already in the cache
1: [  was tired and slept  ]
2: [beneath the shady tree ]
3: [ jumped over the dog  ] < added on the second request
4: [EMPTY]
...
7: [EMPTY]
...
```

Finally, to pass the full hidden states and multimodal outputs to the next stage, we simply concatenate the cached contents with the corresponding new tensors computed from the current forward call.

### What About Multimodal Inputs?

It's also useful to consider the case about how Omni prefix caching is handled when we have multimodal inputs that don't cleanly end on block boundaries, as well as how this works with multimodal encoder caching in vLLM. For example:

```text
         [   Im0  Im1  Im2  Im3  ] [ Im4  Im5 foo <empty> ]
Block 1: |<--- block tokens ---->|
Block 2: |<------- prefix ------>| |<--- block tokens --->|
```

In this case, only `Block 1` will have outputs stored in the prefix tensor cache, because vLLM does not store partial blocks. This may appear to be a problem at first glance, because the multimodal input is fragmented across a new block that wasn't cached.

In reality, this isn't a big problem for correctness, because vLLM also maintains an encoder cache for multimodal inputs. In other words, after the first pass, we'll have the following:

- The Block 1 hash, which is used for prefix caching
- The hash describing the image data starting at position 0 and with length 6
- In vLLM's encoder cache, a mapping from the image hash above to the encoder output

To understand what happens, say we get the following input as a second request:

```text
         [   Im0  Im1  Im2  Im3  ] [  Im4  Im5 bar  baz  ]
Block 1: |<--- block tokens ---->|
Block 2: |<------- prefix ------>| |<--- block tokens --->|
```

First, the scheduler will check for a prefix cache hit, which we will see on `Block 1`. As a result, we will have 4 tokens marked as precomputed, and only see the remaining 4 tokens in the following prefill.

Because we have multimodal data in a scheduled span that isn't fully precomputed, we still need to call the visual encoder. However, since we have the image hash and encoder cache, we will retrieve the encoder outputs for `Im4` and `Im5` as we create the multimodal embeddings.

When we pass our multimodal tensors to the language model component in the same stage, we'll then expect the same outputs, because the prefix caching behaviors in vLLM-Omni / vLLM match, so the LLM will use vLLM's KV cache manager's prefix caching to correctly handle the attention information for `Block 1` while calculating the outputs for `Block 2`, giving us the correct results for processing `Block 2` with the context of `Block 1`.

Finally, we look up the output hidden states/multimodal tensors corresponding to the prefix cache hit `Block 1` and concatenate it with the forward pass result to get the final result, which is expected to be identical to the full hidden states when prefix caching is disabled.

### Implementation

The block/slot model is `vllm_omni/core/prefix_cache/`.
`OmniPrefixCacheManager` owns slot occupancy, the request-task table,
hit spans, and merge.
`OmniPrefixCacheController` moves data: a reusable `StagingBufferPool` for
this step's device→host copy, and writes into the durable `PrefixBlockPool`.
The state lock covers those tables only.

Miss is not an error (this step's forward slice only). A hit span whose
hidden rows are absent is fatal. For mm keys the rule is looser: a model may
emit a key only for some requests, so a hit span with no rows behind an mm key
reads zeros from the pool for those positions rather than raising. Abort still
writes: once a hash entered this step's batch it must land in the cache.

`enable_prefix_caching` is refused on KV-consumer stages (`kv_role` of
`kv_consumer` or `kv_both`). KV received from a producer is reported as
`num_computed_tokens` too, and the manager cannot tell it from a local hit.
Producer-only stages are unaffected. Pooling stages never save, so they get
no cache. This gate is one function, `stage_prefix_cache_config`, called by
both the GPU and the NPU model runner at kv-cache init.

Which stages may set `enable_prefix_caching: true`:

| Stage | `enable_prefix_caching` | Why |
| --- | --- | --- |
| AR stage with one full-attention kv group whose hidden states / per-token mm feed the next stage (Qwen3-Omni thinker and talker) | supported | The case the cache is built for: full-prompt hidden states are merged from the pool on a hit. |
| AR stage that sets `requires_full_prefix_cached_hidden_states = False`, optionally with `deferred_prefix_cache_mm_keys` (Qwen3-TTS talker, Higgs v3 talker) | supported | Hidden is not cached; deferred codec rows are written once on finish. |
| Pooling stage | ignored | Never saves; the gate returns no config. |
| `kv_role: kv_consumer` / `kv_both` | refused at kv-cache init (`OmniPrefixCacheUnmatchError`) | Producer KV shows up as `num_computed_tokens` and is indistinguishable from a local hit. |
| Hybrid / sliding-window / multi-group kv cache (e.g. a talker with `attention_type: sliding_recompute`) | refused at kv-cache init | The cache mirrors exactly one full-attention block table. |
| Codec decoder / Code2Wav stages (Qwen3-Omni stage 2, Qwen3-TTS stage 1) | keep `false` | Nothing downstream consumes their hidden states; the cache would only add device→host copies. Not validated. |
| Diffusion stages | n/a | No vLLM KV cache to mirror. |

Hit spans come from `scheduled_new_reqs` only, as in the pre-refactor cache:

- A new request with a (partial) prefix hit is the normal path: the hit
  blocks are read from the pool, the rest is this step's rows, and the
  divergent tail is written under its own blocks.
- `async_chunk` continuation: when the next upstream chunk arrives, the same
  request id re-enters `scheduled_new_reqs` with `num_computed_tokens` equal
  to what it already computed itself. Ids already in `live_reqs` are skipped
  for hit marking: those rows were delivered in earlier steps and re-emitting
  them would duplicate output. A `delivered_upto` span for this case is
  Phase 2.
- Preemption + reschedule: the resumed request arrives through
  `scheduled_cached_reqs` (`resumed_req_ids`), never `scheduled_new_reqs`, so
  a vLLM-side prefix hit on resume is not mirrored as an omni hit span; the
  resumed request gets only the rows it recomputes. Stages that need full
  prompt hidden states should be sized so preemption does not occur while
  prefix caching is on. Same as before this refactor; tracked for Phase 2.

Two write paths, split by `ModelCachePolicy.deferred_keys`:

- Immediate (`JOIN_NEXT_STEP`): save launches a whole-step device→host into
  a staging slot; the committer waits that event and writes the CPU pool.
  The next save waits `host_ready`.
- Deferred (`JOIN_ON_FINISH`): mm whose first dim is this step's token count
  stays on a per-request GPU clone; `_WriteChunk`s append across steps;
  finish/abort (or GPU-byte-budget pressure) forces the copy. One WriteTask
  per request — lifetime follows the request, not the step.

A `WriteTask` moves through `TaskState` only via `transition()`, one step at
a time along a strict chain:
`PENDING` (registered, not queued) → `QUEUED` (on a copy queue) →
`COPYING` (one thread owns the copy stage) → `HOST_READY` (host rows ready) →
`WRITTEN` (in the CPU pool); `FAILED` is reachable from any non-terminal
state. Skipping a step raises. The worker claims `COPYING` in the same `_wake`
critical section as the queue pop, so `escalate` never re-queues a task it can
see is already claimed. `host_ready` / `done` are wait primitives set by the
transitions into `HOST_READY` / `WRITTEN` / `FAILED`. GPU-clone bytes are
charged once per clone on a `_BudgetTicket` pinned by every task that views
it, and uncharged when the last holder releases (idempotent per tid).

Mm whose first dim equals the unpadded scheduled length *or* the CUDA-graph
padded length is registered on first sighting. Talker `codes.audio` is a
cat of scheduled rows and stays unpadded while hidden is padded; both must
open a pool key, whether immediate or deferred.
Leftover mm (lists, `codes.ref`, any tensor whose first dim is not this
step's token count) is copied to CPU at save without truncating that first
dim. That leftover copy is the async-builder read replica for this step; it
does not write the pool or carry abort/preempt occupancy.

```python
cache.register_policy(ModelCachePolicy.from_model(model))   # load_model
cache.new_step_starts(scheduler_output)   # before _update_states
sid = cache.save_outputs(hidden, mm_outputs, num_tokens_unpadded=n,
                         num_tokens_padded=n_pad)
outs = cache.materialize(sid, req_ids)    # or discard_step(sid)
```

Each step id is consumed exactly once. `req_ids` must be a subset of the save
snapshot. At most `staging_depth` unused step ids may exist at once: every
`save_outputs` claims one staging slot, including saves with only leftover
mm that copy no device→host page. A slot is also held by each immediate
write that views it, until the committer's pool write. A later save waits
for `materialize`/`discard_step` or that pool write to free a slot;
`staging_claim_timeout_s` then errors with the unused ids and the task count.
`materialize` may run on the async output builder after the engine has
entered the next step; leftover mm (not written to the pool) is copied
to CPU at `save_outputs` so the builder never reads live CUDA-graph
buffers. See
[Async Omni Output Materialization](omni_async_output_materialization.md).

The cache is constructed only on the last pipeline-parallel rank
(`_ensure_omni_prefix_cache`). Other ranks skip it: they never call
`save_outputs`, so a hit table there would resolve to absent slots.

Threads, locks, and what each may block on:

| Thread | Role | May block on | Must not hold while blocked |
| --- | --- | --- | --- |
| Engine | `new_step_starts`, `save_outputs` | previous-step `join_host_ready`; `reserve()` GPU-byte flush; staging-slot claim | `_state_lock` |
| Async output builder | `materialize` (may overlap the next engine step) | this step's `step_d2h_event`; `join` (`done`); deferred `fetch_host` | `_state_lock` |
| Committer | `_worker_loop`: wait device→host / deferred copy / pool write | `_wake.wait`; `step_d2h_event` or copy-stream sync | never takes `_state_lock` |
| Prefetch pool | hit-span gather during forward | `join` (`done`); deferred `fetch_host` | `_state_lock` |

| Lock | Covers | Does not cover |
| --- | --- | --- |
| manager `_state_lock` | occupancy tables, step contexts, hit spans | join, GPU-byte flush, copy, `step_d2h_event` wait |
| controller `_lock` / `_wake` | task registry, queues, GPU-clone byte budget | device→host / pool-write body (released before `synchronize`) |
| `WriteTask.lock` | `state`, `reassigned`, `append_chunk` | waiting on `host_ready` / `done` (those are events) |

### Related Files

- `vllm_omni/core/prefix_cache/`
- `vllm_omni/worker/gpu_model_runner.py` (`_ensure_omni_prefix_cache`)
- `tests/core/test_prefix_cache.py`
- [Async Omni Output Materialization](omni_async_output_materialization.md)
