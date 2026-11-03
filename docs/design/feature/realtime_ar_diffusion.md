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

LingBot accepts an opt-in `model_config.worldkv` mapping for pose-ranked
history retrieval and CPU-bank KV compression. Omitting this mapping preserves
the native LingBot sink and paged/direct cache paths. All units below are latent
frames, and `window_frames` includes the current chunk.

The two mechanism controls are:

| Mechanism | Disabled by default | Enable explicitly |
| --- | --- | --- |
| Pose retrieval | `retrieval_frames: 0` | Set a positive frame budget and `bank_bytes` |
| Bank KV compression | `world_keep_ratio: 1.0` | Set a ratio in `[0,1)` with retrieval enabled |

Compression only reduces rows stored in the retrieval bank. It does not
compress the current chunk, native sink, or recent canonical history. Compression
requires retrieval; enabling compression alone is rejected rather than silently
activating retrieval. `world_score: centroid` uses the released WorldKV code's
normalized raw-anchor centroid; `pairwise_mean` (the default) uses mean cosine
similarity to the anchor tokens. These scores can select different tokens.
The anchor frame is always retained in full. For a three-frame chunk,
`world_keep_ratio: 0.5` therefore retains about two thirds of all chunk tokens.

The configuration below matches the paper's LingBot layout: an 18-frame window
holding 3 sink frames, 9 frames of retrieved memory, 3 recent frames and the
3-frame current chunk, with 25% of the non-anchor tokens kept per stored chunk.
Each stored chunk then costs 1.5 frames of tokens, so the 9-frame budget holds
six retrieved chunks.

```yaml
model_config:
  worldkv:
    window_frames: 18
    sink_frames: 3
    retrieval_frames: 9
    bank_bytes: 51539607552  # 48 GiB CPU bank; see bank sizing below
    world_keep_ratio: 0.25  # use 1.0 for the uncompressed retrieval control
    world_score: pairwise_mean  # paper Eq. 5; centroid follows the released code
```

`retrieval_frames` is a token budget measured in frames *after* compression.
The released code's `--retrieval_frames R` instead counts uncompressed frames
(`R / 3` chunks); the equivalent budget here is `R / 3 * (1 + 2 * ratio)`. For
example, its script default (`R = 18`, ratio 0.5) corresponds to
`retrieval_frames: 12`. Chunks are fitted whole, so a budget that is not a
multiple of one compressed chunk leaves the remainder of the region unused.

Omitted `window_frames` and `sink_frames` inherit the model's native window
(18 frames) and sink (9 frames), so an empty mapping attends to the same
sink and recent frames as native LingBot, without retrieval or compression,
through the transactional WorldKV cache instead of the native cache paths. The paper
layout above therefore sets `sink_frames: 3` explicitly. Configuration is
validated against the 3-frame chunk at model load: the window must fit sink,
retrieval and one complete chunk; with retrieval enabled, `sink_frames` must be
a multiple of the chunk size (otherwise the non-sink frames of a chunk
straddling the sink could never be retrieved) and `retrieval_frames` must fit
at least one compressed chunk.

Only single-rank, single-sequence eager attention with clean-x0 KV commits is
supported. First-block sharing and last-denoise-step KV reuse are rejected.
The model's session reservation includes policy history, pending commits,
retrieved device rows and layer-local scratch; a minimum legacy pool remains
allocated for runtime geometry.

### Bank sizing

The CPU bank is byte-bounded with FIFO eviction, and every chunk after the
sink is stored when it is committed. One stored chunk costs

```text
128 + layers * (T + ceil((chunk - 1) * T * ratio)) * (2 * heads * head_dim * itemsize + 16)
```

bytes, where `T` is the number of tokens per latent frame. For the 40-layer,
40-head LingBot geometry at 480x832 in bf16 (`T = 1560`), this is about
1.79 GiB at ratio 0.25, 2.38 GiB at 0.5 and 3.57 GiB at 1.0. A 48 GiB bank
therefore keeps about 26 chunks (78 latent frames, roughly 20 seconds of video)
at ratio 0.25. A revisit can only be recalled while its chunk is still in the
bank, so size `bank_bytes` for the longest revisit distance: holding the whole
trajectory needs `(latent_frames - sink_frames) / 3` chunks. The first
eviction in each memory logs a warning, and `WorldKV.evicted_chunks` counts
evicted chunks.

`bank_bytes` is a per-memory limit: every session branch and every direct
request owns a separate bank. A memory reserves its full `bank_bytes` when it
is created and releases it on reset, close or when a request-local cache is
dropped. A new memory is rejected when the reservations of all live memories
in the process would exceed `host_bank_limit_bytes`, which defaults to the
host's physical memory; set it lower to leave room for weights and other host
state.

### Retrieval

Retrieval ranks absolute C2W poses by normalized translation and rotation
distance. It excludes sink and still-visible recent chunks, fits complete
bank chunks into the retrieval token budget, and preserves chronological order
for attention. Retrieved rows
are transferred once per chunk/layer and reused during denoising. A new pose,
clean commit, failed forward, reset or close invalidates device staging.
Only a successful all-layer clean forward publishes history and bank together;
failed publication leaves canonical state unchanged. Session branches and
requests own separate memories and release them on reset or close.

This branch contains only the WorldKV-inspired retrieval/compression path.
Deep Sink, Participative Compression, RoPE rebasing and CausVid policy integration
are not included; their configuration keys are rejected. Stored and retrieved
keys retain their original temporal RoPE phase, matching the released code's
default (`--retrieval_rope_correction` off). Two further differences from the
released code are intentional. Each chunk is represented by the pose of its
middle latent frame rather than by the element-wise mean of its poses, which is
not a rotation. The released code also excludes the chunk immediately before the
visible recent frames from retrieval even though it is no longer attended; here
every chunk that has left the recent window is a candidate. This is a method transfer to
LingBot v2, not a claim to reproduce the original paper's full benchmark or
improve long-video quality. Prior three-minute results belong to the combined
branch and do not replace validation of this extraction.

CPU tests cover dense numerical compression oracles, raw/retrieved attention,
feature isolation, bank capacity and eviction, transaction rollback, and real
LingBot attention/transformer integration. GPU output and performance claims
require E2E results for the exact branch revision, backend and workload.
