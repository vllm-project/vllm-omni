# Realtime AR-Diffusion sessions

> **Status:** experimental. This document describes the internal execution
> contract implemented under `vllm_omni.experimental.ar_diffusion`. It does not
> define a public HTTP or WebSocket schema.

## Scope

Realtime AR-Diffusion serves world models that generate one autoregressive
latent block per request while retaining model state across requests. The
generic runtime owns identity, event ordering, session lifecycle, routing, and
KV allocation. A model adapter owns model-specific controls and conditioning.

LingBot World v2 is the first single-KV-branch integration. DreamZero uses the
same runner capability with two logical CFG branches.

## Identities

Four identifiers have distinct lifetimes:

| Identifier | Lifetime | Owner | Contract |
| --- | --- | --- | --- |
| `session_id` | One persistent world | Session manager | Selects runner KV and model-owned state. |
| `event_id` | One control/prompt update | Transport adapter | Unique and monotonically increasing within a session, including across reset. |
| `chunk_index` | One committed AR block position | Session | Contiguous from zero; reset starts again at zero. |
| `request_id` | One chunk execution | Session | Correlates the tick with its returned metadata. |

`AsyncOmni` keeps its normal UUID-suffixed engine routing ID. The tick
`request_id` remains inside the immutable AR-Diffusion snapshot and the
standard output metadata, while caller-visible `OmniRequestOutput.request_id`
continues to use AsyncOmni's existing external-ID contract. Protocol identity
therefore does not require a private entrypoint override.

## Request lifecycle

1. `ARDiffusionSession.accept_event()` validates and queues an immutable
   `ARDiffusionSessionEvent`. Prompt and opaque control tracks may be updated in
   the same event.
2. `ARDiffusionSession.next_chunk()` serializes chunk execution and calls
   `_build_tick_locked()`. The resulting `ARDiffusionTickRequest` is a snapshot
   of the current prompt, reduced controls, ordered event IDs, and chunk index.
3. `ARDiffusionOmniTickConsumer.execute_tick()` copies the stage sampling
   parameters, stores the typed tick in
   `sampling_params.extra_args["ar_diffusion_tick"]`, constructs the normal Omni
   prompt through its `prompt_provider`, and invokes `AsyncOmni.generate()`.
4. `AsyncOmni` allocates its normal unique engine routing ID and submits one
   `OmniDiffusionRequest` through the orchestrator.
5. `ARDiffusionModelRunner._request_session()` parses the typed tick.
   `_get_or_create_session()` obtains runner-owned paged KV, and
   `bind_ar_diffusion_state()` exposes it to the pipeline only for the duration
   of `execute_model()`.
6. The model pipeline validates the tick and generates exactly one latent
   block. LingBot interprets camera controls in
   `LingBotCameraControlReducer` and
   `LingBotWorldCausalDMDPipeline.forward()`; the generic runtime does not parse
   camera or action schemas.
7. The pipeline returns one `DiffusionOutput` with the generated payload and an
   `ARDiffusionChunkMetadata` mapping under `metadata["ar_diffusion"]`.
   `AsyncOmni` exposes it as
   `OmniRequestOutput.multimodal_output["metadata"]["ar_diffusion"]`.
8. The consumer parses the standard envelope. The session commits its reducer
   state, prompt, controls, event IDs, and next chunk index only when returned
   metadata exactly equals the submitted tick snapshot.

The metadata envelope is:

```json
{
  "metadata": {
    "ar_diffusion": {
      "session_id": "world-7",
      "request_id": "ar-world-7-3-...",
      "chunk_index": 3,
      "applied_event_ids": [10, 11]
    }
  }
}
```

## Transaction and failure semantics

Accepted events are not acknowledged as applied until the model output and
metadata are validated. Reducer `prepare()` is speculative; reducer `commit()`
and the session snapshot form one logical commit. A model, metadata, or reducer
failure leaves the chunk index and queued events uncommitted and transitions the
session to `FAILED`.

After a tick failure, worker state is closed because a forward may have
partially modified or already released KV. The same chunk is not retried in
place. A caller must explicitly reset the session, starting again at chunk
zero, or close it.

Explicit close follows this state machine:

```text
ACTIVE or FAILED
       |
       v
    CLOSING ---- worker cleanup succeeds ----> CLOSED
       |
       +------- worker cleanup fails --------> CLEANUP_FAILED
                                                   |
                                                   +-- close retry --> CLOSING
```

`CLEANUP_FAILED` is a local tombstone. The session manager retains it, rejects
creation of the same `session_id`, and rejects reset. Only an idempotent close
retry that succeeds removes the session. Transport disconnect uses the same
path.

## KV ownership and capacity

`ARDiffusionKVCacheSpec` is the pipeline-owned immutable geometry contract. It
declares:

- TP-local self-attention geometry, AR block size, sink, and recent window;
- logical KV branches and their worker-local slots;
- fixed-length cross-attention caches;
- scratch tokens required by an uncommitted forward;
- persistent model-owned CUDA bytes per resident session; and
- the requested resident-session capacity.

The runner computes:

```text
required(capacity) =
    scratch
  + managed_self_attention_pages(capacity)
  + capacity * cross_attention_bytes_per_session
  + capacity * model_owned_state_bytes_per_session
```

`gpu_memory_fraction` is a soft expansion budget:

```text
configured_budget = available_device_bytes * gpu_memory_fraction
effective_budget = max(configured_budget, required(1))
```

If `required(1)` exceeds actual available device memory, initialization fails.
Otherwise the runner always admits one viable session and uses the configured
fraction to select additional resident sessions up to the model-declared cap.
All model-owned reservations are deducted before the paged self-attention pool
is allocated.

LingBot advances a session-owned causal Wan VAE encoder for each condition
block: the opening pixel frame is followed by four zero pixel frames per later
latent frame. It retains one current condition block, with separate committed
and in-flight encoder histories so a failed block cannot overwrite committed
context. The state reservation includes both encoder histories, derived from
the encoder's convolution input grids, and the streaming decoder cache. These
caches are released through the same close/reset hooks. Condition encoding
currently requires the unpatched, non-tiled Wan encoder path; the existing
decoder's tiling fallback remains independent. Real-weight GPU parity with a
full-horizon encode still requires validation because the pointwise posterior
projection now runs per block.

DreamZero reports the measured 603 MiB per-session Wan VAE causal-convolution
state.

## Concurrency and routing limits

The current runner executes one request at a time:

- `max_num_seqs` must be one;
- request batching and diffusion step execution are rejected;
- one AR block is generated per engine request; and
- an AR-Diffusion stage must have exactly one replica.

The single-replica restriction is intentional. Multi-replica support requires
session-affine routing so every tick, reset, and close reaches the worker that
owns the session. Random or round-robin routing is not correct.

Multiple resident sessions may share one runner. Capacity selection accounts
for all persistent pools, and the runner performs LRU eviction through the same
model lifecycle hook used by explicit close.

## Model boundary

The generic runtime transports controls as opaque
`ARDiffusionControlInput(track, schema, data)` values. Model adapters validate
and reduce those values. This keeps target/velocity, key state, pose,
intrinsics, and other model-specific conventions out of the session protocol.

For LingBot World v2:

- `lingbot_world/actions.py` parses key-state and trajectory controls;
- `lingbot_world/camera.py` loads or constructs camera trajectories and Plücker embeddings;
- `lingbot_world/transformer.py` implements checkpoint-compatible causal attention; and
- `lingbot_world/pipeline.py` constructs conditioning, owns small non-KV session state, and
  produces one block plus the standard metadata envelope.

## Ulysses sequence parallelism

LingBot supports pure Ulysses sequence parallelism for both direct and
AR-Diffusion execution. With Ulysses degree greater than one, hidden tokens,
camera features, token-expanded timestep modulation, and RoPE tables are sharded
together. Without SP, timestep modulation retains the frame-broadcast path.
Self-attention performs the sequence-to-head all-to-all before reading or writing
paged KV, while static text K/V uses the same local head shard. Text K/V shards
own compact storage; cross-attention exchanges query/output layouts to use these
shards. This retains the shared cache geometry and reduces text K/V storage at
the cost of two all-to-all calls per layer.

The output head projects local tokens before gathering the flow values. For
the 14B model this reduces the gathered width from 5120 to 64; frame modulation
uses each shard's global token offset, including shards that split a frame. Only
`ulysses_mode="strict"` is supported; `advanced_uaa`, Ring, and AllGather-KV modes
remain unsupported for this model.

## Non-goals

This contract does not currently provide:

- a public HTTP or WebSocket request schema (the structured-interaction
  frontend is tracked separately in
  [#5527](https://github.com/vllm-project/vllm-omni/pull/5527));
- camera/action semantics shared by every world model;
- session migration or replication across workers;
- retry of an ambiguous partially executed chunk;
- cross-stage KV transfer; or
- stateful streaming VAE decode.

Those features can be layered on top without adding LingBot-specific identity
or lifecycle fields to the generic runtime.

## Required regression coverage

CPU contract tests cover:

- separation of tick correlation IDs from internal engine routing IDs;
- event ordering, metadata equality, and reducer fail-closed commit;
- close/disconnect failure tombstones and cleanup retry;
- capacity one under the default fraction for shipped LingBot geometry;
- capacity expansion under a tuned fraction;
- DreamZero model-owned state accounting;
- runner session reuse, reset, close, LRU eviction, and forward cleanup; and
- LingBot model registration, imports, camera/action reduction, and one-block
  output metadata.

GPU validation must additionally exercise real model loading, action input,
prompt switching, rolling-window replacement, reset, close, finite output
latents, and exact metadata/request identity.

## Experimental WorldKV memory for LingBot World

`model_config.worldkv` opts LingBot into training-free, pose-indexed KV memory.
Without it, the native sink and window paths are unchanged. Units are latent
frames; `window_frames` includes the 3-frame current chunk.

```yaml
model_config:
  worldkv:
    window_frames: 18
    sink_frames: 3            # multiple of the chunk size when retrieving
    retrieval_frames: 9       # budget after compression; 0 disables retrieval
    bank_bytes: 51539607552   # per-memory CPU bank, FIFO eviction
    world_keep_ratio: 0.25    # 1.0 disables compression; needs retrieval
    world_score: pairwise_mean  # or centroid, as in the released code
    reuse_attention_kv: false  # cache assembled windows across denoising forwards
    prefetch_kv: false         # CUDA host-bank retrieval on a dedicated stream
    prefetch_bank_allocation: false  # allocate the next host chunk during denoising
    # host_bank_limit_bytes: process-wide cap, defaults to physical memory
```

- **Retrieval:** evicted chunks go to the bank, keyed by their middle-frame
  C2W pose. Each chunk ranks bank chunks by translation plus rotation distance
  and fits whole chunks into the budget, in time order with original RoPE.
- **Compression:** only bank rows; the anchor frame is kept in full.
- **Assembly:** canonical GPU history contains an uncompressed sink and a
  contiguous recent suffix. Host-known frame boundaries select views of these
  regions before concatenation with retrieved and current rows, avoiding
  per-layer GPU index discovery and scalar synchronization. Bank compression
  does not change these history boundaries. With `reuse_attention_kv`, each layer
  assembles its window once per chunk and overwrites only the current K/V tail
  on later forwards. These borrowed attention views are consumed before the next
  forward on the same compute stream. Pose/geometry changes, commit, failure and
  close invalidate them. GPU admission includes one extra full window per layer.
- **Transfers:** CUDA commits copy K/V into pinned host memory on a separate
  stream. Host-bank retrieval waits for that copy; uncompressed chunks still
  resident in GPU history are read directly. At most one publication remains
  in flight. Uncompressed publication borrows the new history's tail; separately
  owned compressed source storage is included in the GPU reservation.
  Each publication packs K/V into one host allocation per dtype, with disjoint
  views for individual layers. Live chunks never share backing storage.
  Host-bank retrieval transfers only K/V; its metadata remains on the host. Uncompressed
  publication constructs contiguous frame/token metadata directly on CPU;
  compressed per-layer selections retain their original metadata.
  With `prefetch_kv`, the first layer queues selected K/V for all layers on a
  dedicated CUDA stream. Each consumer waits for its own layer event and records
  destination storage on the compute stream. Cancellation drains outstanding
  prefetches before releasing cached rows. CPU execution uses ordinary retrieval.
  These options default to false and can be enabled independently.
  With `prefetch_bank_allocation`, a worker prepares at most one upcoming
  publication's pinned allocation once layer geometry is known. Commit consumes
  it; forward failure and close wait for the worker and discard it. The worker
  owns allocation sizes and a device index, not request or GPU tensor references.
  This option does not preallocate the entire bank; its benefit depends on the
  host allocator and must be measured with the selected process-wide settings.
  Pinned allocations use PyTorch's caching allocator; initial bank filling
  still incurs allocation costs. `bank_bytes` bounds live entries, not allocator
  caches or the temporary replacement chunk needed for transactional publication.
- **Bank size:** about 1.79 GiB per chunk at 480x832, bf16, ratio 0.25. A
  revisit is recalled only while its chunk is still in the bank; the first
  eviction logs a warning.
- **Limits:** single rank, single sequence, eager, clean-x0 commits. Deep
  Sink, Participative Compression, RoPE rebasing and CausVid keys are rejected.
- **State:** history and bank publish only after an all-layer clean forward;
  failures, reset and close leave canonical state unchanged.

For large CUDA host banks, PyTorch's host-registration allocator can reduce
cold pinned-allocation cost. Before starting the process, add
`pinned_use_cuda_host_register:True,pinned_num_register_threads:8` to
`PYTORCH_ALLOC_CONF`, preserving any existing allocator options. This is a
process-wide PyTorch setting, not a WorldKV configuration field; benchmark it
with the actual workload and host CPU resources. It changes how pages are
allocated and registered rather than preallocating the entire bank. See
[PyTorch CUDA memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf).

For fixed-size large chunks, PyTorch versions supporting
`pinned_max_round_threshold_mb` can also use
`pinned_max_round_threshold_mb:128` in the same allocator configuration.
Allocations larger than that threshold use their requested size rather than
rounding up to a power of two. This can reduce physical pinned-memory overhead;
it does not change the logical `bank_bytes` limit or guarantee a latency gain.
Measure allocator memory and E2E latency separately for the actual chunk sizes.

This is a method transfer, not a reproduction of the paper's benchmark.
Quality and performance claims need E2E results for the exact revision.
