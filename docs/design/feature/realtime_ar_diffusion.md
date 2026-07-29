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
| `request_id` | One chunk execution | Session | Exact engine routing key for one in-flight tick. |

`AsyncOmni.generate(..., _engine_request_id=request_id)` preserves the tick
request ID without adding a UUID. This private value is a routing key, not a
display label: it must be unique among active requests. `AsyncOmni` rejects a
collision before submitting the second request and never replaces the original
`ClientRequestState`.

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
4. `AsyncOmni` reserves the exact engine request ID and submits one normal
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

LingBot reports its persistent full-horizon image-condition tensor. DreamZero
reports the measured 603 MiB per-session Wan VAE causal-convolution state.

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

- `actions.py` parses key-state and trajectory controls;
- `camera.py` loads or constructs camera trajectories and Plücker embeddings;
- `transformer.py` implements checkpoint-compatible causal attention; and
- `pipeline.py` constructs conditioning, owns small non-KV session state, and
  produces one block plus the standard metadata envelope.

## Non-goals

This contract does not currently provide:

- a public HTTP or WebSocket request schema;
- camera/action semantics shared by every world model;
- session migration or replication across workers;
- retry of an ambiguous partially executed chunk;
- cross-stage KV transfer; or
- stateful streaming VAE decode.

Those features can be layered on top without adding LingBot-specific identity
or lifecycle fields to the generic runtime.

## Required regression coverage

CPU contract tests cover:

- exact engine request IDs and concurrent collision rejection;
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
