# MiniCPM-o 4.5 Full-Duplex Runtime Architecture

> Relocation note: after the H20 validation recorded below, the full-duplex
> implementation moved out of `vllm_omni/experimental/fullduplex/` into the
> stable tree (`vllm_omni/engine/duplex/`, `vllm_omni/entrypoints/duplex/`,
> `vllm_omni/model_executor/models/minicpmo_4_5/duplex/`, and companion
> modules). Module paths in this document reflect the new layout; the runtime
> evidence was produced on the pre-move tree, which is file-identical apart
> from module paths.

## Purpose

This document records the runtime architecture implemented by PR #3907, the
cleanup applied after published commit `e011d936`, and the remaining work that
must not be advertised as complete. Implemented contracts and future
architecture work are called out separately below.

The current 0.28 interface cleanup is specified in
[Transactional KV Append](feature/duplex_kv_append.md).
It unifies the shared OpenAI Stage0 path for MiniCPM-o, PersonaPlex, and
Nemotron while keeping model replay capability separate. The review snapshot
and dated validation sections below describe earlier trees, including their
then-current deployment limits. They are not acceptance evidence for this
cleanup or proof that the MiniCPM answer-ending risk has been resolved.

The checkpoint preserves the active runtime path already exercised on H20. It does
not introduce another reducer, controller, worker provider, or shadow runtime.

## Historical Review Snapshot

- PR: `vllm-project/vllm-omni#3907`
- Published head before this refactor: `e011d936`
- Published base snapshot: `62589203`
- Published diff: 93 files, approximately `+29.2k/-0.6k`
- Local runtime and pytest validation: used only for fast contract checks
- Required production evidence: a matching CUDA host and a vLLM build exposing
  the streaming-prompt scheduler API
- PR head reviewed for the current cleanup:
  `f4f78fa555af4d65e46b003022d352ca2ea9d703`
- Validated tree: the uncommitted refactor synchronized file-for-file to the
  isolated H20 worktree on 2026-07-20

The earlier runtime checkpoint received focused H20 E2E validation against
vLLM 0.25.0. The scheduler-native extension has now also received focused H200
E2E and fault-injection validation against vLLM 0.28.0 plus streaming-prompt
source commit `ed6e712edd77c041bab62836157b578a76799489`. The final historical
H20 affected matrix includes the pre-response continuation and
Stage1 CUDA Graph padding regressions. The historical H20 profile admitted two
concurrent sessions and rejected a third until capacity was returned. The
current checked-in MiniCPM profile admits four concurrent sessions through the
same engine-owned limit; on H200, two synchronized four-session/two-turn trials
completed all eight responses with independent request identity and no errors.
This is isolation, admission, and bounded fairness evidence for the tested
single-H200 four-session deployment, not an arbitrary-capacity SLO. The
validated paths include the experimental runtime
extension, typed resumable policy, extracted control plane and clients, request
preregistration, typed direct-output decision, engine-managed four-session
capacity in the checked-in deploy profile, separate public/runtime configuration
channels, private Session ledgers, ordered `session.update`, the MiniCPM runner
fast path, continuous
browser input, and the final Realtime input-lifecycle fixes. Exact evidence and
scope limits are recorded below.

## Scope

The implementation targets these contracts; the exact cleanup snapshot must
meet the acceptance criteria below before it is signed off:

- MiniCPM Stage0 conversation KV continuity;
- Stage1 TTS and Token2Wav continuity;
- model-owned listen/speak decisions on the normal auto-response path;
- continuous browser PCM upload during assistant playback, without browser VAD
  or browser-generated input commits;
- segment EOS and turn EOS as different boundaries;
- transcript/audio cursors scoped to a response and turn;
- playback acknowledgement and history commit;
- Stage0 request metadata routed per batch row rather than inherited from the
  first request in the batch;
- Stage0 token timing snapshots and client-observed audio chunk cadence;
- scheduler data-plane append over a resumable request;
- stale epoch/turn/response fencing;
- `/v1/duplex` and OpenAI Realtime projection;
- optional OpenAI `server_vad` turn detection backed by a process-shared Silero
  model and per-session VAD state;
- existing JoyVL behavior.

The current tree does not claim:

- raw KV-tensor append (the implemented primitive is prompt-token append with
  reuse of resident scheduler KV);
- an acoustic-onset-to-cancel latency target across arbitrary client chunk sizes;
- raw KV-tensor live migration after worker/replica loss (MiniCPM's separate
  replay capability can create a new physical request and rebuild from
  committed prompt units; PersonaPlex and Nemotron retain typed safe failure
  without automatic prompt replay);
- H20/H100 sign-off for the scheduler-native delta (the fresh evidence is H200);
- arbitrary-capacity fairness or starvation SLOs; configured session limits
  and earlier CUDA results require validation on the exact cleanup snapshot.
- audio/video synchronization beyond one camera frame per model unit (camera
  frames ride the audio appends as `video_frames`; the wire contract is in the
  [Realtime Duplex API](../serving/realtime_duplex_api.md) guide).

## Active Runtime Path

The actual path is:

```text
WebSocket
  -> OmniDuplexSessionHandler
  -> DuplexSession + MiniCPMO45ServingSessionState
  -> DuplexRequestClient
  -> AsyncOmni thin open/append/signal/close proxies
  -> DuplexControlClient over the engine-owned correlated RPC transport
  -> DuplexControlPlane + DuplexSessionRuntimeManager
  -> MiniCPM DuplexRuntimeExtension + engine session/stage bindings
  -> StagePool
  -> admit_duplex_request_async for the first unit
     / StageEngineCoreProc.append_streaming_prompt_unit for later units
  -> vLLM 0.28 retained request + StreamingUpdate
  -> MiniCPM Stage0
  -> Stage1 TTS / Token2Wav
  -> output processor
  -> Realtime protocol projection
  -> WebSocket writer
```

The removed typed reducer/facade and worker-provider runtimes were not in this
path. This refactor continues to modify only the path above.

## State Ownership

The architecture allows several state objects only when their facts are
orthogonal.

| Owner | Authoritative facts |
| --- | --- |
| `DuplexSession` | session state, epoch, turn, active/last response, pending input, overlap duration, playback cursor, history |
| `MiniCPMO45ServingSessionState` | MiniCPM PCM buffer, deferred overlap payload, data-plane task, continuation effect counter |
| `DuplexWebSocketActor` | inbound mailbox, outbound queue, writer, transport close state, effect task handles |
| `NativeRealtimeSessionProtocol` | wire-only response/item projection and input-buffer projection |
| `DuplexSessionRuntimeManager` | fence snapshot, append reservation, completed-operation cache, resource reservations, stage bindings |
| `DuplexControlPlane` | control-message validation, transactional open/append/update/close, extension invocation, typed segment/output decisions |

Important distinctions:

- Realtime response IDs are projection caches, not domain decision sources.
- Engine fences duplicate identity across a process boundary by design; they
  are immutable validation snapshots.
- Stage bindings are engine resources, not serving response state.
- `continuation_owner_id` scopes an effect counter either to a visible response
  or, before `response.created`, to an explicit model turn. A pre-response
  continuation also revalidates request, epoch, turn state, and model-turn
  identity immediately before append; it cannot survive turn completion.
- runtime open/close acknowledgements are handler-local effect bookkeeping,
  not Session lifecycle.

## Implemented Concurrency Contracts

### Single inbound mailbox

The Actor now uses one:

```python
asyncio.Queue[event]
```

This replaces competing control/input/event `Queue.get()` tasks and the
deferred-event list while preserving WebSocket arrival order. A later close,
cancel, or clear must not overtake earlier append, playback ACK, or
session-update events. Slow runtime appends remain background tasks. Cancelling
their asyncio task does not stop a synchronous callable already running in the
default executor, so cancellation correctness is provided by the engine fence,
not by `Task.cancel()`.

Properties covered by focused tests:

- every enqueued event is delivered exactly once;
- terminal control preserves wire order while cancelling background work;
- clear and cancel never scan or delete later mailbox events;
- one writer owns all WebSocket sends.

### Correlated RPC result router

`AsyncOmniEngine` now has one consumer for `rpc_output_queue`. Results route by:

```text
("duplex", control_id)
("collective", rpc_id)
```

The router replaces the global RPC lock. It supports out-of-order replies,
timeout unregister, late-result rejection, fatal-error broadcast, and close
unblocking.

`control_id` is only reply correlation. Logical append retry identity is the
separate `operation_id`, while stale epoch/turn rejection is owned by the
`DuplexFence` and session runtime. The RPC router does not own either append
idempotency or same-session ordering.

Additional lifecycle rules:

- uncorrelated non-fatal errors are not broadcast to unrelated waiters;
- a latched fatal rejects new calls before they enter the request queue;
- `EngineDeadError` is broadcast directly to the RPC queue even though the
  orchestrator consumes that exception during teardown;
- shutdown closes the router before joining the orchestrator;
- failure to enqueue shutdown closes the request queue and still attempts a
  bounded orchestrator join;
- `StageRuntime` cleanup runs only after the orchestrator has stopped, avoiding
  concurrent client teardown.

The router removes wait-side head-of-line blocking. The orchestrator request
handler remains ordered, but executor submission order is not itself a session
ordering guarantee.

### Extracted control ownership

The generic engine mechanism and the duplex plugin are split by
responsibility:

| Module | Responsibility |
| --- | --- |
| `engine.messages` | ordinary engine queue messages and collective RPC envelopes |
| `engine.rpc_result_router` | generic correlation routing through each result message's `rpc_correlation_key` |
| `engine.duplex.contracts` | immutable duplex DTOs, extension/stage protocols, typed decisions, request-ID codec |
| `engine.duplex.lease` | duplex lease activity, TTL/grace configuration, and expiry records |
| `engine.duplex.messages` | identity fence and duplex control message/result envelopes |
| `engine.duplex.runtime` | extension loading, validation, and compatibility exports |
| `engine.duplex.session` | session resources, fences, append reservations, idempotency cache, binding cleanup |
| `engine.duplex.control_plane` | control algorithms and the narrow `DuplexStagePort` used to submit and clean up stage requests |
| `engine.duplex.control_client` | typed control-message construction and calls through the engine-owned correlated RPC transport |
| `entrypoints.duplex_request_client` | deterministic request identity, client-state preregistration, acknowledgement validation, data-plane collection, rollback |

`Orchestrator`, `AsyncOmniEngine`, and `AsyncOmni` retain explicit wiring and
compatibility proxies. They no longer own the corresponding duplex algorithms.
The ControlClient does not own waiters or consume the result queue;
`RpcResultRouter` remains the single correlation owner for both duplex and
collective RPC. The ControlPlane receives a narrow stage port rather than the
Orchestrator object.

Control-plane enablement is independent from extension presence.
`PipelineConfig.duplex_control_enabled` determines whether an Orchestrator
constructs a ControlPlane. This preserves extension-free
`TURN_COMMIT_ONLY` deployments while allowing ordinary deployments to use a
single `None` fast-path check. The MiniCPM pipeline explicitly enables control
and separately selects its runtime extension.

### Transactional open and append

Opening a native session registers its engine state and reserves the Stage0
request resource as one externally atomic operation. If capability, sampling,
extension, or request-state initialization fails, the ControlPlane removes the
session, reserved request resources, and any preregistered request state before
returning the error. Cleanup distinguishes a reserved resource from a submitted
request, so it does not decrement the running counter for unrelated work.
Closing a session that never appended also removes the preregistered resource.

Expiry and request-triggered cleanup use a two-step close. The runtime manager
first marks the session closing but retains it in the admission set. Only a
successful stage cleanup finalizes and removes the session. A blocked or failed
cleanup therefore continues to consume its server-owned slot and is retried;
replacement admission cannot overlap request or KV resources that are still
owned by the closing session.

Append uses prepare/submit/commit ordering. `prepare_append()` computes the
next sequence values without mutating the session. Prompt planning and stage
submission use that reservation, and only a successful submit commits
`input_seq`, `input_turn_seq`, and the accepted fence. Serving uses a matching
PCM reservation and consumes bytes only after the append acknowledgement;
failure or a closed predecessor rolls the bytes back. An `operation_id` caches
completed append results so a retry cannot submit the same physical append
twice.

### Ordered append and configuration effects

Native append effects form a per-session tail. Each append awaits its
predecessor before entering the engine, and `session.update` awaits that same
tail before changing either the serving aggregate or engine configuration.
Consequently, an append received before an update uses the old immutable
sampling snapshot, while the first accepted append after the update uses
the new generation.

The engine increments `config_generation` when it accepts the replacement
configuration. At the next append boundary, the control plane rebuilds sampling
parameters from stage defaults through the model extension. It never mutates
the snapshot of a segment already in progress.

### Irreversible cancellation fence

Every cancel control carries the cancelled fence and the next session fence.
The `AsyncOmni` facade preserves both values across the serving,
entrypoint, engine-message, and orchestrator boundaries. The orchestrator then
atomically:

1. advances the runtime state to the next fence;
2. releases and aborts bindings owned by the cancelled fence;
3. permanently rejects an append that arrives later with the old fence.

Missing `next_fence` is rejected without releasing bindings; the engine never
advances independently from serving. A late cancel cannot move a session
backward when the runtime is already on a newer fence. Idle playback clear does
not send an engine cancel because no runtime resource is being cancelled.

`model_native_duplex` also validates its engine-client contract before opening
the session. The open, append, signal, and close methods must exist and declare
the required `fence` arguments; signal must additionally accept `next_fence`.
An older or custom client that cannot preserve this boundary fails with
`runtime_contract_invalid` instead of silently running an unfenced native
session. The generic serving-session adapter retains its compatibility behavior.

`DuplexFence` also carries a server-side session incarnation. Reusing a public
session ID after close increments that incarnation, and resource request IDs
include it after the first incarnation. A synchronous append from a closed
session therefore cannot be accepted by a newly opened session with the same
public ID. Stale close validates the full fence before removing the runtime
registry entry or releasing bindings.

## Implemented Serving Ownership

### Session-owned response and overlap state

The Actor no longer stores:

- a Session reference;
- lifecycle mirror state;
- active or last response ID;
- overlap duration;
- response-in-progress or playback business predicates;
- runtime open/close acknowledgements.

`DuplexSession.begin_response()` owns both active and last response identity.
The Session owns overlap accumulation/reset and playback state.

One `native_response_in_progress()` predicate now covers:

- active response ID;
- active engine request ID;
- uncommitted ACK-only playback;
- active generic response task;
- response-bound append tasks;
- MiniCPM data-plane task;
- assistant-generating turn state.

The write-only `MiniCPMO45ServingSessionState.response_emitted` field and a dead
data-plane task mirror were removed.

### Response-scoped options

`response.create` is represented by an immutable `ResponseCreateOptions` value.
The Session reserves it for one response, applies it to a response-local copy
when that response begins, and restores the base session configuration on
completion, listen, cancellation, or failure. Instructions, voice, modalities,
temperature, token limit, format, tools, and conversation metadata therefore do
not mutate the permanent session configuration or leak into a later response.

### Explicit event effects and pure writer

Business paths call `emit_event()`. It serializes domain effects and outbound
queue insertion with one session-local lock. Deferred append starts outside the
lock so recursive event emission cannot deadlock.

The writer only:

1. dequeues an event;
2. applies late stale filtering to streaming model output;
3. projects the event to the selected wire protocol;
4. sends JSON.

The writer does not promote overlap, clear input, advance Session state, call
the engine, or commit history.

### Single terminal acceptance point

Terminal events are accepted or rejected before domain effects run:

```text
response.done
response.listen
audio.cancelled
input.cancelled
session.closed
```

For a terminal carrying an epoch, that epoch must match the current Session.
Stale cancellation therefore cannot reset overlap belonging to a later turn.
Once a terminal is accepted and its effect is applied, the writer cannot revoke
it because close or epoch state changed while it waited in the output queue.

Streaming audio/text deltas still receive late stale filtering at send time,
which prevents queued old audio from leaking after a cancel or epoch bump.

## Runtime Type Boundary

The stable kernel is deliberately limited to generic engine and scheduler
mechanisms:

```text
vllm_omni.engine.messages                  # ordinary queue and collective RPC envelopes
vllm_omni.engine.rpc_result_router         # generic correlation routing
vllm_omni.engine.async_engine_utils        # ordinary engine lifecycle helpers
```

The full-duplex implementation lives in dedicated duplex modules:

```text
vllm_omni.engine.duplex.runtime
                                           # extension loading and validation
vllm_omni.engine.duplex.contracts
                                           # immutable DTOs and narrow protocols
vllm_omni.engine.duplex.lease
                                           # duplex lease primitives
vllm_omni.engine.duplex.messages
                                           # duplex identity and queue envelopes
vllm_omni.engine.duplex.session
                                           # session transaction implementation
vllm_omni.engine.duplex.control_plane
                                           # engine-side control algorithms
vllm_omni.engine.duplex.control_client
                                           # correlated control client
vllm_omni.entrypoints.duplex_request_client
                                           # entrypoint request/output lifecycle
```

Stable engine and entrypoint modules import the duplex kernel eagerly at
module load (top-of-file imports). Duplex behavior remains opt-in: the
ControlPlane, ControlClient, and request client are *constructed* only on
duplex-enabled deployments, and the model runtime extension still loads only
through the dotted-string plugin path. The client-side package
(`vllm_omni/clients/`, home of the public `DuplexClient`) is never imported
by runtime code.

`PipelineConfig.duplex_control_enabled` enables the generic mechanism, while
`PipelineConfig.duplex_runtime_extension` separately selects the model adapter.
The MiniCPM pipeline opts into both and installs
`MiniCPMO45DuplexRuntimeExtension`; ordinary pipelines construct neither a
ControlPlane nor a ControlClient. The model-neutral protocol exposes
sampling-parameter configuration, append planning, and a typed stage-output
decision. MiniCPM owns
stage-specific overrides, PCM-to-token budgeting, force-listen payload policy,
special-token interpretation, and `listen` response metadata. The generic
Orchestrator does not parse MiniCPM session keys, `listen_token_id`, or construct
MiniCPM metadata, and it does not import the adapter.

Extension loading is fail-fast. Startup validates every required callable and
the number and type of returned stage sampling parameters. Invalid custom
extensions fail before a session can open.

Direct listen/speak outcomes use `DuplexOutputDecision` on
`OmniRequestOutput`. That typed envelope is authoritative over inner completion
metadata for native projection, so a processed output cannot hide a listen
decision merely because it retains unrelated multimodal metadata.

Generic multimodal chunk accumulation lives in
`vllm_omni.outputs.multimodal_accumulation`. It owns snapshot replacement,
DELTA draining, and audio-finality interpretation; `OmniRequestState` only
invokes that policy while assembling request output. Model output processors no
longer need to add chunk-retention rules to the main output orchestration file.

The strict chunk-transfer schema still contains `hidden_states.tts` as the
generic AR-stage-to-TTS handoff and several MiniCPM streaming token IDs as
compatibility wire fields. The latter are not a preferred generic model
contract. Moving them behind a model-namespaced metadata envelope requires a
versioned producer/consumer migration because unknown msgspec fields are
rejected; silently deleting or renaming them in this checkpoint would break
existing Stage0-to-Stage1 transfer.

### Public and runtime configuration channels

Control messages carry two independent snapshots:

- `session_config` contains the public Realtime/session values used by serving
  and model prompt policy;
- `runtime_config` contains server-derived MiniCPM sampling, scheduler, context
  reserve, prefix-budget, and resolved reference-audio values.

The MiniCPM serving adapter derives `runtime_config` from deploy/model defaults
and validated public options. Clients cannot provide private runtime keys in
`session.create` or `session.update`; response-scoped `extra_body` drops those
keys rather than turning them into engine policy. The ControlPlane constructs
sampling parameters only from `runtime_config`,
while append planning receives both snapshots explicitly. Raw Realtime
`extra_body` is not used as the engine's sampling-policy channel.

`session.update` prepares both candidates, validates the runtime candidate in
the engine, and replaces the serving snapshots only after the engine ACK. A
rejected candidate therefore leaves both the public configuration and runtime
sampling generation unchanged.

`DeployConfig.duplex_session` is an immutable, server-owned
`DuplexSessionRuntimeConfig`. These limits are not read from client
`extra_body` or session payloads:

| Field | Default | Owner | Lifecycle or resource effect |
| --- | ---: | --- | --- |
| `idle_ttl_s` | `300.0` | Engine lease manager | Expires an attached or detached session after inactivity; `null` disables only idle expiry. |
| `disconnect_grace_s` | `30.0` | Engine lease manager and API attachment registry | Bounds how long a detached session may retain engine resources before cleanup. |
| `reaper_interval_s` | `5.0` | Orchestrator | Sets the cadence for lease expiry and pending cleanup retries. |
| `resume_replay_ttl_s` | `60.0` | API attachment registry | Expires replay events retained for resume or takeover. |
| `resume_replay_max_bytes_per_session` | `8388608` | API attachment registry | Bounds each session's replay buffer. |
| `attachment_io_timeout_s` | `5.0` | API attachment registry | Bounds the whole resume activation/replay and each terminal notification/close; must be finite and positive. |
| `max_pending_input_bytes_per_session` | `16777216` | API session input ledger | Applies per-session byte backpressure before input is admitted. |
| `max_pending_turns_per_session` | `4` | API session input ledger | Bounds queued turns that have not completed runtime processing. |
| `max_pending_appends_per_session` | `4` | Engine control plane | Bounds queued/in-flight appends per session; cancel and close bypass the limit and preempt pending append waits. |
| `max_sessions` | `1` | Engine session manager | Atomically enforces admission; closing sessions retain capacity until cleanup finalizes. |
| `completed_append_cache_size` | `256` | Engine session manager | Bounds completed append idempotency records per session. |

Deploy profiles may lower or raise these values to match scheduler capacity,
but clients cannot override them. The engine remains authoritative for
admission and cleanup ownership even when the API performs an early check.

The attachment journal owns immutable serialized event snapshots, not references
to mutable conversation items. Accounting, live delivery and resume replay use
the same snapshot. Attachments may expose a `send_text` callback to deliver the
encoded snapshot as a WebSocket text frame without JSON round-tripping; the
dictionary callback remains the compatibility path and handles control events.
Pending sends use a completion future that revocation can resolve directly,
without a separate Event.wait task for every message. On Python 3.12+, the
default event loop can finish nonblocking sends eagerly in an isolated task
context. Custom task factories are honored, older Python uses normal scheduling,
and every 32 inline completions yield to maintain cooperative progress.
Per-session attachment revocation releases a blocked writer
without cancelling its model-output producer. Resume serializes the engine
lease CAS with transport activation; cancellation preserves the last accepted
credential while detached, and an in-flight CAS records its result before a
retry can issue another CAS. Generation and incarnation fences still reject
old attachments and closed sessions.

Runtime terminal events have per-session ordered cleanup tasks. A slow task
cancellation in one session does not delay another's cleanup. Retired transport
notification/close is tracked, bounded best-effort work outside that cleanup.
Transport callbacks must remain asynchronous; cancelling a callback cannot undo
bytes already handed to its old socket, but its completion cannot change the
new attachment or its journal. A callback that delays cancellation remains
tracked until it completes.

The Realtime projector releases terminal response assembly state, retaining at
most 256 recent terminal IDs. After `response.created` establishes the serving
lifecycle, unknown deltas/terminals are dropped instead of creating response
state, including identities evicted from that cache. Conversation items retain
the completed content and playback truncation marks until explicit deletion or
session teardown. This is not automatic history eviction or model KV compaction.

### Client-side companion package

The public client surface ships under `vllm_omni/clients/` and stays on the
consumer side of the boundary above: `vllm_omni.clients.duplex` provides the
model-agnostic `DuplexClient` (typed events, per-response demux via
`ResponseHandle`, `session.resume` with acknowledgement-based replay dedup)
and the `EventCollector` helper used for timing and request-metric summaries.
Model-specific session defaults live in thin per-model preset modules —
`vllm_omni.clients.minicpmo_4_5.create_duplex_session_config()` and
`vllm_omni.clients.personaplex.create_duplex_session_config()` — so the core
client never imports model packages and the model duplex plugin packages
never import the client (both directions are enforced by tests:
`tests/engine/test_duplex_import_boundary.py`,
`tests/entrypoints/openai_api/test_duplex_capability.py`, and the subprocess
purity guards in `tests/clients/test_model_session_configs.py`).
`examples/online_serving/barge_in_client.py` (with its flow companion
`barge_in_client_flow.md`) is the runnable reference for the client library
against a live duplex deployment. The wire vocabulary the client speaks —
every `/v1/realtime?duplex=1` event with its OpenAI-compatibility tier and a
JSON example, plus the native-to-Realtime name map — is catalogued in the
[Realtime Duplex API](../serving/realtime_duplex_api.md) serving guide; that
page derives from the normative contract table above and must be updated
with it.

## Generic-Path Cleanup

Model-specific `MINICPMO45_PROFILE_LOGS` probes were removed from:

- the generic AR scheduler;
- orchestrator;
- AsyncOmni;
- the generic GPU AR runner.

The cleanup removes the temporary serving and generic-runtime probes. The
generic AR runner builds `DuplexSamplingRow` values only when the model exposes
the optional `prepare_duplex_sampling()` hook, then invokes that hook exactly
once before the normal sampler. Ordinary models do not scan request rows.
MiniCPM owns force-listen logits, turn-ended state, and its row-local sampling
policy behind that hook; the runner no longer reads `_minicpmo45_*` attributes
or retries sampling after matching `TypeError` text.

The scheduler no longer parses serving `session_config`, `extra_body`, or
`duplex_stage_sampling_params`. Stage-specific overrides are materialized as
ordinary `SamplingParams` by the model runtime extension before scheduler
admission. Existing vLLM stop-token, EOS, and max-token handling defines each
segment boundary; `resumable=True` parks the request for the next streaming
update. The scheduler does not inspect a duplex dictionary, Realtime
configuration, or MiniCPM sampling fields.

The generic runner's unused `_duplex_force_listen_applied_segments` cleanup and
the obsolete `streaming_accumulated_keys` accumulation hook were removed. The
latter had no model producer after its upstream Qwen removal; streaming input
still performs the ordinary replacement merge needed by active models.

Local `turn.signal` transitions such as `user_started` and
`assistant_started` no longer perform an engine RPC. Only cancellation and
`session.update` cross the engine boundary because those operations change
runtime resource identity or runtime configuration.

During a stage transition, `StreamingInputState.source_token_decoder` exposes a
short-lived, model-neutral decoder sourced from the upstream output processor.
The MiniCPM Stage1 input processor consumes that capability for token-faithful
delta transcripts. The generic orchestrator no longer installs or names a
MiniCPM-specific bridge-state key.

The stable API server imports the duplex serving package eagerly and inspects
the deployment `session_mode` before constructing the duplex handler. Ordinary
deployments do not create its registries and background state.
For an enabled deployment, the client still selects the Realtime duplex route
with `?duplex=1` (or an equivalent explicit true value). Model-name matching is
not used for routing or native-runtime activation. MiniCPM clients explicitly
set `extra_body.native_duplex=true`; the canonical way to do so is
the `vllm_omni.clients.minicpmo_4_5.create_duplex_session_config()` preset,
which the repository's barge-in example
(`examples/online_serving/barge_in_client.py`) uses in its session payload.

`DuplexRequestClient` derives the data-plane request identity from the accepted
fence and preregisters a resumable `ClientRequestState` before the append crosses
the engine boundary. The acknowledgement must return the same identity or the
operation fails. Unregistered outputs, including names that merely use the old
`duplex-` prefix, are dropped. Timeout, cancel, and close remove the
preregistered state.

## Scheduler-Native Prompt Append

`engine.kv_append` detects the single supported vLLM 0.28 retained-request
contract without modifying upstream classes. Session open checks that contract,
every live Stage0 client's Omni methods, and a positive context limit. The
shared OpenAI adapters for MiniCPM-o, PersonaPlex, and Nemotron all use this
transactional Stage0 path. Unsupported native submission is rejected; it no
longer falls back to `StagePool.submit_update`.

The default dependency remains vLLM 0.28.0. Its native implementation uses
`resumable` requests and `StreamingUpdate`; Omni's atomic append utility layers
operation identity and lifecycle control over that retained-KV mechanism. A
version pin alone does not supply Omni's sampling, retry, or recovery contracts.

The first unit uses the explicitly declared client
`admit_duplex_request_async`, which calls upstream `add_request_async` with a
resumable request. Initial acknowledgement waits for the complete prefill
(`num_computed_tokens >= num_prompt_tokens`), not a held last token. Subsequent
units use `StageEngineCoreProc.append_streaming_prompt_unit`, declared on the
Omni-owned Core subclass. There is no external finalize RPC.

The cleanup removes `_streaming_prompt_compat.py`, its experimental forwarding
module, import-time append/admission patches on upstream `EngineCore` and
`AsyncMPClient`, the synthetic status alias, dual-version scheduler branches,
and the old worker `new_prompt_token_ids` path. The Omni scheduler's private
`_append_duplex_tokens` helper delegates to its metadata-aware
`_update_request_as_session` override and then upstream's private method. That
upstream dependency remains an upgrade boundary; this is not a claim that
Omni uses only stable public vLLM APIs.

The scheduler keeps one request and its allocated KV blocks alive across model
unit boundaries:

```text
RUNNING
  -> segment stop
WAITING_FOR_STREAMING_REQ    (request and prior KV blocks retained)
  -> append_streaming_prompt_unit(operation_id)
WAITING                     (update and receipt commit finish in one utility)
  -> schedule
RUNNING
  -> session close
FINISHED                   (KV released)
```

Prompt-token append reuses retained KV. The update and `_commit_native_append`
execute synchronously in one EngineCore utility on the scheduler thread;
scheduling cannot interleave between them. The latter is internal receipt
bookkeeping, not another RPC or scheduling phase. The utility writes a
per-request pending journal after token mutation and before receipt commit.
If commit raises after token acceptance, only the same operation ID and full
digest may retry. A committed marker lets a retry publish its receipt without
committing twice, even if the model has already completed the new segment.
A different operation or fingerprint is rejected while the journal is pending,
so recovery cannot append the tokens
again. A bounded, scheduler-owned receipt cache deduplicates a lost-response
retry by `operation_id`; a full fingerprint binds the ID to the raw payload,
final bit, configuration generation, materialized prompt/model buffer, sampling
metadata, and token IDs. Reusing the same ID with any different input is
rejected, and StagePool rejects a native update that arrives without that full
fingerprint instead of degrading to token-only idempotency. The scheduler
utility repeats that validation before request lookup or mutation, so an
alternate caller cannot bypass the engine boundary and regain token-only
deduplication. Initial native admission likewise validates both fields before
replica selection or output registration. Evicted receipts leave bounded digest tombstones, so an old retry
fails explicitly instead of being physically appended again. Every native
append control, including the first request admission, must carry a non-empty
operation ID before request resources are reserved. Serving-generated silence
continuations receive one UUID when their task is created and reuse that stable
ID for any retry. A resident physical request remains pinned to its selected
Stage0 replica.

`supports_prompt_replay` is independent of atomic append. Only MiniCPM enables
it; PersonaPlex and Nemotron explicitly disable it. They cannot enter automatic
replay/rebuild or context rollover merely because their Stage0 uses append.
Replica loss instead returns typed safe failure such as
`native_kv_replica_lost`; reconstructing codec or recurrent model state requires
its own model contract.

For MiniCPM sessions with replay enabled, each successful materialized append
is also recorded as a complete unit in an engine-owned journal bounded by
token and byte limits. When membership reports that replica lost and a healthy
replica remains, the next
append creates a new physical request generation, replays only committed units,
then applies the live append. This is bounded prompt replay and KV rebuild, not
transfer of raw KV tensors. An uncertain append is never replayed across
replicas. A missing replacement, an incomplete/oversized journal, or a replay
failure returns a typed safe failure; when no recovery is possible the session
is terminal and the client must reopen/replay. Caller deadlines bound readiness
polling and the utility await, preventing a timed-out append from holding the
per-session control tail for the old fixed 120-second window. A cancel or close
preempts pending append or replay work and then aborts the physical request; it
does not cancel unrelated open/signal/touch/resume controls or re-cancel an
append task that is already publishing its correlated cancellation reply.
Terminal control waits for earlier session controls to finish, and normal
same-session controls remain FIFO.
After an uncertain utility timeout, StagePool retains one bounded operation
guard and output registration: only the same operation ID/fingerprint/tokens
may retry until its scheduler receipt resolves, so a later append cannot jump
past an operation that may already have committed. The serving bridge retries
that exact operation once on a transport timeout or typed `timeout` result,
reusing its payload and fence; native session open rejects engine clients that
cannot accept `operation_id`, so the retry cannot silently fall back to an
at-least-once append.

Sampling updates carry the new immutable `SamplingParams` through StagePool,
the explicit client/Core utility methods, and the scheduler/worker.
Utility arguments decoded as `Any` are explicitly reconstructed as
`SamplingParams`. The scheduler also updates its separate `max_tokens` field,
which upstream vLLM 0.28.0 does not update in `_update_request_as_session`.
Cached worker updates refresh InputBatch sampling tensors; a seeded generator
is retained unless its seed changes.

Partial prefills have no delivered sample. Their duplex sampling rows are
explicitly marked disabled, so the model cannot advance a turn fence,
terminator, repetition history, or RNG for a discarded token. This also applies
when a long prompt is recomputed in several batches after KV preemption.

Logical-epoch receipt ownership is separate from physical KV ownership. The
session retains SHA-256 tombstones for evicted operation IDs across rollover
and replica replay. At most 65,536 tombstones are retained; capacity is checked
before submission, and exhaustion requires cancel/reopen. Epoch advancement
clears this namespace only after old-fence operations become invalid.

MiniCPM also retains a physical-request-owned CPU history of actual prepared
audio/vision embeddings. It is bounded by 40,960 token positions and 512 MiB
per physical request and is released by terminal request cleanup. Preemption
recomputation overlays the historical spans at their original positions;
computed output-token gaps use ordinary token embeddings. This is local
recomputation support, not KV live migration. The engine publishes
`kv_append_start` for each unit so the worker can verify its exact prepared
span even after preemption resets the current computed offset. `final=True`
arms the turn-end fence but does not reserve an extra audio unit.

Failed input preparation is request-local: the runner latches the error,
masks that request's KV write slots with `PAD_SLOT_ID`, suppresses its generated
tokens and downstream payload, and returns `model_input_errors`. The scheduler
terminates it with `FINISHED_ERROR` rather than parking/resuming invalid KV.
The scheduler keeps a bounded recent-error cache (256 entries, messages capped
at 1024 characters) so a racing readiness RPC returns the original typed
`model_input_error` after physical request cleanup instead of `not_found`.
The same failure hooks are wired on CUDA and NPU; hardware evidence for one
platform must not be presented as validation of the other.

Stage0 rejects an append before mutation when projected retained context exceeds
`max_model_len`; the MiniCPM profile pins that hard ceiling to 40960. Before the
ceiling, MiniCPM's replay-enabled control plane plans a new physical generation at the configured
utilization fraction, retains the newest complete append units that fit the
rollover budget, and rebuilds KV before accepting the live append. It never
splits an append unit. This bounds resident context and replay storage, but it
does discard older prompt units: it is not automatic summarization, semantic
compression, or lossless arbitrary-context compaction. If even the rebased
candidate cannot fit, the append fails before old KV is destroyed. Successful
rebuilds distinguish historical replay from a live candidate rebased onto an
empty journal: only historical outputs are suppressed. The model publishes an
explicit per-step replay flag, including `false` on live output; accumulation
replaces this flag rather than concatenating it. Suppression reads both wire
dicts and structured completion payloads so delayed historical output cannot
escape merely because the request has already switched back to live mode.
Successful append replies expose current context, limit, utilization, receipt count, and
deduplication state. Prometheus exports bounded-label append outcomes/latency,
control-tail wait, active/closing sessions, KV/context growth, receipt growth,
replay-journal size, resource generation, rebuild outcomes, and replica-affinity
loss. No metric uses session, request, or operation IDs as labels. An
unrecoverable context failure remains non-retryable for that operation but does
not silently discard the session: `input.cancel`/barge-in rotates the fence and
aborts the exhausted resident request, `session.close` releases it immediately,
and idle/disconnect lease expiry is the final safety-net reclamation path.

The current interleave point is a completed model unit: append waits for
`WAITING_FOR_STREAMING_REQ`. It does not mutate an actively decoding prompt.
Scheduler responsibilities are request/KV state, runtime metadata,
stop/boundary handling, retry receipts, and final release. Response IDs,
playback, overlap, Realtime events, and model policy remain outside this owner.

The generic `StagePool.submit_update` retains active public-streaming,
downstream talker/code2wav, diffusion, and PD consumers. PersonaPlex's
independent browser backend also retains its frame-stepper/session lifecycle.
Neither is a retired KV shim. See the feature design's removed/protected
boundaries before proposing any further deletion.

### Earlier implementation evidence

The following records describe source snapshots before the single-contract
cleanup. They do not establish its correctness, capacity, or performance.
The current MiniCPM profile keeps Stage0 async scheduling disabled; enabling
it and validating complete answers require separate runs.

Native capability validation requires a positive Stage0 context limit. The
MiniCPM profile keeps `async_scheduling=false` on every platform. A 2026-09-06
follow-up added a worker-side unit-stop fence and consistent runner/CLI
scheduler selection. Aligned async-on now completes the H200
four-session gate; the failures described below are historical. The recorded
Seed-TTS performance samples below use chat fallback, not native KV append;
they cannot establish native speedup or its absence. Native performance needs
separate validation, so the default remains off. See
[the follow-up validation](../validation/kv_append_v028.md).
The scheduler contains an async fence that rolls back an unconsumed lookahead placeholder,
marks in-flight output stale, sets `drop_stale_output`, clears speculative
tokens, and rejects impossible placeholder/computed-token state before append.
It passed a real H200 single-session/two-turn run with
`OmniARAsyncScheduler`, stable request/replica identity, and monotonic context
growth. It did not pass the synchronized four-session liveness gate: with
async scheduling enabled, Stage0 EngineCore exited under concurrent append,
none of the four sessions reached its first `response.created`, and the
remaining requests correctly surfaced `native_kv_replica_lost`. With async
scheduling disabled, all four sessions completed two turns in each of two
trials. Therefore the checked-in profile fails closed with async scheduling
disabled and admits the validated four-session shape. The async fence and
regression tests remain groundwork, not an enabled production capability.

The scheduler-native path was added after the historical H20 evidence
summarized at the top of this document. CPU contract tests cover the intended atomicity,
deduplication, capability, deadline, and replica-affinity rules. On 2026-09-04,
the formal affected CPU L1 ran against streaming-prompt vLLM source commit
`ed6e712edd77c041bab62836157b578a76799489` on the H200 host and passed 519
tests: 24 scheduler-mixin, 323 orchestrator/error/duplex-handler, 38 control-
plane, 17 membership, and 117 other affected tests. A separate 40-test router,
correlated-client, AsyncOmni abort/output regression group passed in the same
formal environment. The dependency-isolated local smoke also passed all 38
control-plane tests and 12 focused scheduler append/journal tests; that smaller
run remains state-machine evidence, not the basis of the formal count. The added
control-plane case saturates one
session's four-entry append backlog while 31 independent sessions complete,
then verifies that close preempts and reclaims every hot-session append. On
H200, scheduler-native core and expansion groups passed 4 and 3 tests. The
async-on two-turn run used `OmniARAsyncScheduler`, retained one request and
replica, and grew Stage0 context monotonically under the configured 40960-token
limit. A committed-reply-loss injection proved that Serving retries the same
operation ID and the scheduler deduplicates it to one receipt and one context
increment. A stopped Stage0 replica proved both `input.cancel` and
`session.close` can preempt a pending append within the bounded control path.
A killed resident-KV replica in a two-replica Stage0 topology rebuilt committed
prompt units on the surviving replica under a new physical request generation;
the single-replica loss path still produced `native_kv_replica_lost`, terminated
the session safely, and reclaimed resources. Neither path migrates raw KV
tensors. The reduced-context E2E exercised planned rollover before its
256-token hard ceiling, kept every physical generation within the limit, and
continued on a new request generation. Oversized rebased candidates remain
typed `rate_limit_error/resource_exhausted` failures before old KV is removed.
The async-on Seed-TTS comparison is recorded below. Later synchronized
four-session/two-turn trials with async-off completed 8/8 responses twice;
async-on instead terminated Stage0 EngineCore before any first response. The
default single-GPU profile therefore admits four sessions and keeps Stage0
async scheduling disabled.

## Serving and Compatibility Boundaries

### Normative Realtime event contract

The public contract in this section applies to `/v1/realtime?duplex=1`. Internal
duplex event names such as `response.output_audio.delta` are projection inputs,
not additional public aliases. Events for one attachment preserve mailbox
order; replay after resume preserves the original event order and `event_id`.

Accepted compatibility aliases are normalized before mailbox admission:

| Canonical input event | Accepted aliases | Contract |
| --- | --- | --- |
| `input.text.append` | `input_text.append`, `push_text` | Appends text to the open input item. |
| `input_audio_buffer.append` | `input.audio.append`, `push_chunk` | Appends audio; legacy aliases default to WAV when no format is supplied. |
| `turn.signal` | `signal_turn` | Sends an explicit turn signal; it does not imply response creation. |
| `playback.ack` | `audio.playback_ack` | Advances playback/history only for the identified response and attachment. |
| `session.close` | `close_session`, `close` | Closes the session after previously admitted mailbox events. |

The Realtime projector emits the following response lifecycle. Cardinality is
scoped to one `response_id` unless stated otherwise.

| Public event | Trigger and ordering | Cardinality |
| --- | --- | --- |
| `response.created` | First event for a visible assistant response; precedes its output-item, content-part, speak, text, and audio events. | Exactly once for every visible response. |
| `response.speak` | The model selected speak; emitted after `response.created` and no later than the first `response.output_audio.delta`. It carries decision metadata, not transcript text. | At most once. |
| `response.output_audio.delta` | Carries one ordered audio chunk. Every emitted audio delta is followed by a `response.output_audio_transcript.delta` for the same chunk. | Zero or more. |
| `response.output_audio_transcript.delta` | Append-only transcript contribution paired with an audio delta; it may be empty for a text-less audio unit and must not repeat or overlap earlier text. | One per audio delta. |
| `response.output_audio.done` | Closes the audio stream after its final delta and before response terminal events. It is omitted for a response with no audio or transcript. | At most once. |
| `response.output_audio_transcript.done` | Contains the exact concatenation of all non-overlapping transcript deltas for the response. | At most once, and only for a non-empty transcript. |
| `response.output_item.done` / `conversation.item.done` | Finalize the assistant item after content-part terminal events and before `response.done`. | At most once each. |
| `response.done` | Terminal event for a created response; follows audio, transcript, content-part, and item terminal events. | Exactly once for every created response that reaches a terminal state. |
| `rate_limits.updated` | Compatibility event emitted immediately after `response.done`; the current payload has an empty rate-limit list. | Once per emitted `response.done`. |
| `response.listen` | The model selected listen at a model-unit boundary. It may be emitted without `response.created`; clients must not infer a missing or pending response from it. | Zero or more per session, at most once per model-unit decision. |

Session lifecycle events (`session.created`, `session.updated`,
`session.resumed`, `session.replaced`, `session.expired`, and
`session.resync_required`) are session-scoped rather than response-scoped.
`session.created` is emitted before held response output on a new attachment.
Cancellation and truncation use terminal status/details on the affected
response and never reuse that response ID for later model turns.

Compatibility changes must update this table and its protocol contract tests in
the same change. The golden transcript test requires joined
`response.output_audio_transcript.delta` values to equal
`response.output_audio_transcript.done.transcript`; response tests separately enforce
at-most-once `response.speak` and terminal event cardinality.

### Session internal ledgers

`DuplexSession` remains the serving aggregate root and now composes
`InputBufferState`, `ResponseState`, `PlaybackLedger`, and
`ConversationHistory`. The ledgers are private. Read-only properties return
tuples, snapshots, or mapping proxies, and serving modules mutate them through
Session transition methods for request, response-turn, lifecycle, playback,
capability replacement, and history. The Realtime session-update parser still
normalizes individual fields before atomically retaining or rolling back the
aggregate configuration. Further work may narrow the read-only compatibility
surface, but it must not introduce another reducer or a second session state
machine.

### Realtime input ownership

The browser is a continuous PCM producer. While the session is open and the
microphone is unmuted, it sends `input_audio_buffer.append` every 200 ms,
including while assistant audio is being generated or played. It does not run
VAD and does not send `input_audio_buffer.commit`. Native Stage0 consumes the
stream in approximately one-second model units; MiniCPM listen, speak, and turn
EOS decisions advance the model conversation.

Explicit Realtime clients may still use `input_audio_buffer.commit` to create a
conversation item. The translator validates that wire input before producing a
commit carrying `item_id`. Native auto-response may already have
streamed those PCM samples into the runtime, so the runner accepts a validated
commit even when no runtime-side chunk remains. A truly empty explicit buffer
continues to return `input_audio_buffer_empty`.

The commit-to-response contract is:

| Session state at `input_audio_buffer.commit` | Runtime action | Required event outcome |
| --- | --- | --- |
| Auto-response disabled, valid idle input | Commit and retain the input item. | Emit `input_audio_buffer.committed`; emit no response until an explicit `response.create`. |
| Auto-response enabled, valid idle speech | Submit the final native append for the current model turn. | Emit `input_audio_buffer.committed` before model output. Create a response only on the first visible text/audio output; a model listen decision may end with `response.listen` and no response. |
| Auto-response enabled, short overlap during an active response | Discard the short overlap as an acknowledgement. | Emit the committed/no-response acknowledgement; do not create or defer a response. |
| Auto-response enabled, meaningful overlap during an active response | Retain one deferred payload and promote it after the active response terminates. | Emit `input_audio_buffer.committed` with deferred metadata. A later response follows the same visible-output rule and never reuses the prior response ID. |
| Explicit silence/noise (`is_speech=false`) | Clear the physical input without a runtime append. | Emit the committed/no-response acknowledgement followed by `response.listen`. |
| Empty or invalid input | Reject without changing response ownership. | Emit one typed `error`; do not emit `response.created`. |
| Previous response cancelled or truncated and cleanup complete | Apply the corresponding idle rule above. | The prior response remains terminal; any later visible output receives a new response ID. |

Append tasks, explicit commits, cancellation, and deferred promotion are
serialized by the session actor. Tests cover post-response commits, active
playback deferral, truncated-response ownership, and two-turn
response-required execution; the H20 validation record below is the golden
multi-turn runtime evidence.

During auto-response overlap, `preserve_realtime_input` distinguishes "do not
append this silent chunk to the native buffer" from "clear the open Realtime
item". Silent overlap no longer discards earlier user PCM.

The first chunk of one overlapping input item also reserves its target model
turn. A later Realtime commit uses that reserved identity even if response EOS
has already advanced the Session turn; clear, cancel, close, and successful
promotion release the reservation.

### Physical input and model-turn identity

A Realtime input item, an optional `input_audio_buffer.commit`, and a MiniCPM
model turn are intentionally different identities. The browser normally keeps
one input stream open without commits. Native Stage0 evaluates that stream in
approximately one-second model units. A sampled `<|turn_eos|>` closes the
current model turn and allows a later streamed unit to start another turn.
Explicitly committed input may also produce only model-listen decisions and no
spoken response.

Therefore neither `model_turn_id` nor response cardinality is derived from the
number of physical inputs. `model_turn_id` advances at model EOS. The
`response-required` fixture may require exactly one response per requested
turn, but `model-policy` validation must accept additional model-owned responses
and cannot assert cross-turn transcript independence through physical-input
attribution.
Before closing, the model-policy demo drains every created response through
`response.done`, acknowledges every completed audio playback, and requires a
bounded quiet interval with no newly created response. This validates lifecycle
completion without suppressing or overriding the model's listen/speak policy.

### Non-cancel signal semantics

`turn.signal` remains a public compatibility surface for local Session and
Realtime transitions. Serving forwards only cancellation and `session.update`
to the engine. The engine rejects arbitrary non-cancel signal names instead of
returning a misleading supported acknowledgement. `session.update` remains a
real engine operation because it replaces the configuration consumed by later
append plans.

## Remaining Architecture Work

The following work is deliberately not hidden inside this checkpoint.

### Serving composition

The generic runner and runtime bridge now depend on the model-neutral
`ServingRuntimeAdapter` protocol and no longer import MiniCPM modules. The
`MiniCPMO45ServingRuntimeAdapter` owns MiniCPM serving state, PCM preparation,
data-plane projection, client/runtime configuration validation, prefix policy,
and capability projection. `PipelineConfig.duplex_serving_adapter` explicitly
selects its import path; `AsyncOmniEngine` and `AsyncOmni` carry that path to the
API server, which passes it into the generic handler. The handler has no model
default and fails startup when the duplex endpoint is enabled without an
adapter. Tests may inject an adapter instance directly. Import-boundary tests
verify that loading the generic runner, bridge, or handler does not load MiniCPM
modules.

Behavior is still assembled through `DuplexSessionRunnerMixin`,
`NativeRuntimeBridgeMixin`, and `ChatFallbackProjectorMixin`, so explicit
adapter composition is only the model-policy boundary, not a completed removal
of host-method coupling. A later refactor may replace the remaining mixins with
explicit protocol and effect-runner dependencies while preserving one actor
mailbox and one `DuplexSession` transition owner. Splitting those components
must not create a second reducer or concurrent state machine.

The four private Session ledgers are data partitions, not independent owners.
`DuplexSession` intentionally remains the aggregate root. Further encapsulation
may narrow compatibility properties, but serving code must continue to perform
state changes through Session transitions.

### Plugin descriptor and typed payloads

The pipeline currently selects the MiniCPM engine extension and Serving adapter
through two explicit fields, but there is not yet one versioned plugin descriptor
that binds those identities in an open handshake. Realtime client payloads do
not select either implementation. Adding a second native model should first
introduce that descriptor and reject serving/engine plugin mismatches explicitly.

The stable boundary has typed fences, plans, policies, decisions, and control
messages, but `session_config`, `runtime_config`, and append payloads still use
generic mappings or objects at the extension boundary. Model-neutral typed
session and input-chunk contracts remain follow-up work, especially before
adding video.

### Output modality policy

The current Realtime duplex contract supports text and audio output. Unknown
output modalities fail closed with `unsupported_response_modality`; they are
never silently projected as text. This checkpoint deliberately does not add a
modality registry because there is only one native text/audio projection path
to register.

A registry becomes justified when a second non-audio/non-text implementation
uses the same validation, encoding, lifecycle, and projection contract. At that
point the abstraction must be extracted from two working adapters and retain
the same fail-closed behavior. A speculative registry before that condition
would add selection state without removing any existing branch.

### Performance measurement contract

Stage0 token timing comes only from the engine's per-segment metric snapshot:
`vllm_ttft_ms`, `vllm_tpot_ms`, and `vllm_itls_ms`. A resumable Stage0 request
exports its segment-local `num_generation_tokens`; ordinary requests retain the
existing output-token fallback. Realtime response state clears its metric
accumulator at `response.created`, combines only model units owned by that
response, and ignores session-level events with no matching `response_id`.
Realtime projection carries the result under
`metadata.vllm_omni.stage_metrics`; clients do not estimate TPOT from transcript
deltas or subtract cumulative snapshots.

Audio timing uses the client's monotonic receive clock and is partitioned by
`response_id`. `response_created_to_first_audio_ms` measures projection startup
after `response.created`; `commit_to_first_audio_ms` measures end-to-end output
from an explicit input commit. Neither field is called TTFP without naming its
origin. Reports also include inter-chunk and chunk-duration
count/mean/p50/p95/max plus maximum chunk gap. Engine and client measurements
have different clock origins and must not be merged into one timeline.

### Stage1 CUDA Graph boundary

Native Stage1 streaming advances Python-owned request generators and turn
state on every scheduler step. A full-model CUDA Graph would execute that
Python control flow only while capturing and then replay stale chunk and turn
metadata, even if its host-to-device tensor construction were made
capture-safe. The duplex deploy profile therefore keeps `enforce_eager=false`
but selects `cudagraph_mode=PIECEWISE` for Stage1. Python orchestration runs on
every step while fixed-topology compiled regions remain eligible for CUDA
Graph capture. This is a model execution constraint, not an E2E-only override.

### Resource capabilities

The checked-in MiniCPM duplex deploy profile retains four scheduler sequence
slots in each stage and configures both the engine runtime manager and Stage1
active-stream window with `max_sessions=4`; Stage2 remains ungated so it cannot
choose a disjoint active set. The Realtime capability response therefore
advertises multi-session and same-replica multi-session support with
`session_admission_mode="engine_managed"`. Client session fields cannot raise
this server-owned limit. These are configuration values, not a new capacity
sign-off. Earlier H20/H200 evidence is recorded below against its own source
snapshots. The cleanup requires fresh liveness and complete-answer checks;
the configured four-session shape alone does not establish fairness or an SLO.

The Stage1 active window uses a persistent FIFO across scheduler ticks. A
segment-boundary yield joins its tail, behind every existing waiter, and
terminal cleanup removes the queue entry. A one-promotion cooldown is not
sufficient: with `N > K + 1` the oldest streams can keep alternating and starve
later ones. CPU regressions cover repeated `K=1/2/4, N=K+2` rotations and waiter
cancellation; the four-session CUDA gate also includes `K=2` contention rather
than relying exclusively on the default `K=N=4` shape.

Session leases, lease TTL, resumable attachment, orphan reaping, and bounded
Serving input/backpressure are implemented. Engine append admission is bounded,
different sessions execute independently, and queue/starvation metrics are
available. For MiniCPM with prompt replay enabled, committed prompt units can
rebuild KV on a surviving configured replica under a new physical request
generation; otherwise replica loss safely terminates the affected session.
PersonaPlex and Nemotron do not enable that recovery capability. Raw KV-tensor
live migration is intentionally not claimed. Capability claims must stay limited to the
mechanisms and execution shapes actually validated on the target host.

## Validation Evidence

Production pytest and runtime evidence for this branch must run on a matching
CUDA host; lightweight local state-machine runs are development smoke evidence
only. The H20 record below validates the earlier runtime checkpoint. The H200
record validates the scheduler-native/streaming-prompt delta and does not by
itself sign off H20 or H100.

### Scheduler-native H200 evidence (2026-09-04 through 2026-09-05)

- Streaming-prompt CPU L1: 519 passed; correlated router/AsyncOmni group:
  40 passed; focused scheduler/config/control additions: 94 passed with 18
  deselected, including 26 scheduler and 17 deploy-config cases. The later
  playback-history reservation regression passed 20 focused tests and the
  affected client/runtime/native-hook/input-processor matrix passed 226 tests.
- H200 scheduler-native CUDA groups: 4 core tests passed with one deselected,
  and 3 expansion tests passed. Async-on two-turn continuity passed with one
  request/replica identity and context/receipt observations remaining
  monotonic below the 40960-token ceiling. A later default-profile
  single-session/two-turn run also passed in 171.56 seconds with two responses,
  no errors, stable physical request/replica identity, and monotonic context
  growth. The strengthened terminal-boundary rerun passed in 173.60 seconds;
  both completed responses carried a terminal internal
  `response.output_audio.delta` with `end_of_turn=true`, a non-negative model
  turn ID, and explicit scheduler-data-plane/model-runner/KV-backed metadata.
- The async-off four-session/two-turn gate ran two synchronized trials on one
  H200. Both completed 8/8 responses with no session error. Workload elapsed
  time was 24.7671 and 12.0442 seconds (0.3230 and 0.6642 responses/s); mean
  TTFT was 3703.282 and 1325.641 ms, with maximum TTFT 7371.141 and 4471.330
  ms. Request and semantic isolation held across all four sessions. The fifth
  concurrent session was rejected with typed `resource_exhausted: limit=4`.
  This is the evidence for the default four-session production envelope. A
  later rerun after strengthening the terminal-boundary contract again passed
  both trials at 8/8 responses (19.8517 and 13.0358 seconds; 0.4030 and 0.6137
  responses/s). All 16 completed responses explicitly exposed a genuine model
  turn EOS through the terminal internal audio-delta metadata described above;
  cancelled, failed, or synthetic `response.done` events cannot satisfy this
  assertion.
- The same four-session gate with Stage0 async-on is an explicit negative
  result: Stage0 EngineCore exited under concurrent append, 0/4 sessions reached
  their first `response.created`, and subsequent control surfaced typed
  `native_kv_replica_lost`. The checked-in profile and CPU config contract keep
  Stage0 async scheduling disabled.
- The four-test reliability file covers committed append reply loss,
  pending-append cancel/close preemption, recoverable resident-KV replica loss,
  and reduced-context rollover. It does not include the separate terminal
  single-replica-loss or oversized-candidate rejection cases. Focused H200
  runs passed committed-reply-loss exactly-once retry in 196.95 seconds,
  pending-append cancel/close bounded preemption in 163.32 seconds, surviving-
  replica replay/KV rebuild in 226.37 seconds, and planned context rollover in
  159.15 seconds. A subsequent unified run of the complete four-test H200
  reliability file passed 4/4 in 743.31 seconds, so these results are also
  covered by one clean, serial invocation rather than only by separate focused
  runs. The recovery/rollover tests assert a successful append on a new physical
  generation and bounded metrics, then close; they do **not** establish a
  completed spoken response after rebuild or lossless conversation-history
  equivalence. The normal-path model-EOS gate above does not close this gap.
- The async-off one-session/four-turn Seed-TTS reference completed all turns at
  0.30 request/s, 115.45 ms mean TTFT, 3350.05 ms mean E2EL, 473.05 ms mean
  audio TTFP, and 0.22 mean audio RTF. The matching async-on run also completed
  all four turns at 0.30 request/s, with 122.72 ms mean TTFT, 3358.11 ms mean
  E2EL, 482.31 ms mean audio TTFP, and 0.22 mean audio RTF. This sample shows no
  throughput gain and slightly worse latency; together with the four-session
  EngineCore failure, it does not justify enabling async scheduling.

### Review regressions (2026-09-05)

The local review used base/HEAD
`3cddda978153f3a19ecd691d343854dd074e587d` plus the existing worktree. It
reproduced and fixed four boundary defects: non-FIFO slot readmission;
mislabeling an empty-journal live candidate as replay; stale/concatenated replay
markers on subsequent live output; and suppression that ignored structured
completion payloads or let unrelated request metadata mask their marker.
The focused pre-fix tests failed at the corresponding assertions. After the
fixes, the affected CPU matrix passed **642 tests in 5.79 seconds** on Python
3.12.3 and streaming-prompt vLLM 0.28.0 in the existing H200 container. The five
changed production files and `pyproject.toml` were SHA-256 matched between the
local checkout and that test environment. A separate correlated-router,
AsyncOmni output/abort and duplex-entrypoint group passed **47 tests in 2.80
seconds**, with one non-selected test. These are logic tests, not GPU
performance evidence.

The final-code H200 serial GPU invocation passed **4 tests in 751.89
seconds**: the default `K=4` four-session gate, the new `K=2` contention gate,
surviving-replica rebuild, and reduced-context rollover. Each window gate ran
two trials of four sessions and two turns; all **32 responses** completed with
the scheduler-native/model-runner/KV-backed model-EOS checks and no session
errors. Recovery/rollover retain the narrower append/metrics evidence scope
described above; this run does not establish post-rebuild speech completeness.

The new CPU cases are collected by the existing `core_model and cpu` ready
jobs. The additional four-session/two-slot parameter uses the existing
`advanced_model` merge job and its helper dependency; no new pipeline is
needed. Run from the repo root after activating the matching vLLM environment:

```bash
# Focused local L1 regressions; no model weights required.
python -m pytest -q -o addopts= \
  tests/distributed/omni_connectors/test_chunk_transfer_adapter.py \
  tests/engine/duplex/test_duplex_control_plane.py \
  tests/worker/test_native_duplex_hooks.py \
  tests/engine/test_multimodal_accumulation.py \
  tests/engine/test_orchestrator_error_handling.py \
  -m 'core_model and cpu' --run-level=core_model

# L3 CI-like CUDA gate: both K=4 and K=2, N=4, two turns, two trials.
# Requires one H200-class test GPU and cached MiniCPM-o model/ref-audio assets.
# Set MODEL_PREFIX to the directory containing openbmb/MiniCPM-o-4_5.
python -m pytest -q -o addopts= \
  tests/e2e/online_serving/test_minicpmo_4_5_duplex.py \
  -k test_duplex_four_sessions_rotate_without_starvation_across_multiple_turns \
  -m 'advanced_model and cuda' --run-level=advanced_model

# Exact four-test serial GPU invocation used for the review (same prerequisites).
python -m pytest -q -o addopts= \
  tests/e2e/online_serving/test_minicpmo_4_5_duplex.py::test_duplex_four_sessions_rotate_without_starvation_across_multiple_turns \
  tests/dfx/reliability/test_reliability_minicpmo_4_5_duplex.py::test_native_duplex_committed_kv_recovers_on_surviving_replica \
  tests/dfx/reliability/test_reliability_minicpmo_4_5_duplex.py::test_native_duplex_context_rollover_bounds_kv_growth_and_continues
```

The full CPU matrix additionally includes the entire duplex engine directory,
scheduler mixin, orchestrator and stage-input bridge, output processor, native
Talker batching, MiniCPM stage-input processor, and realtime driver tests.
`-o addopts=` disables repository xdist defaults in the environment without
pytest-xdist; it does not change the selected test level or assertions.

### Historical H20 synchronized tree

The current dirty tree was synchronized to the isolated H20 worktree at
`/home/admin/workspace/aop_lab/model_runner_v2/vllm-omni-worktrees/pr3907-boundary-cg-0721`.

- Final affected matrix: 661 passed, one network-dependent config test
  deselected, 20 warnings. It includes the complete MiniCPM native-duplex hook
  file and all control/session/handler/Orchestrator/import-boundary suites. The
  run is task `be221f67` (`/tmp/remote_gpu_logs/be221f67.log`). The deselected
  `test_resolve_when_autodetect_resolves_none` attempted a Hugging Face model
  lookup and hung in the isolated environment; the duplex deploy-config test
  passed independently.
- Stage1 PIECEWISE CUDA Graph validation: the deploy-config assertion first
  failed with no Stage1 compilation config, then passed in task `3ed6091d`.
  Server task `95827027` started with `enforce_eager=False`, logged
  `cudagraph_mode=PIECEWISE`, and captured batch sizes 1, 2, and 4 without the
  CPU-to-CUDA capture error. The first concurrent E2E exposed graph padding as
  `request_token_spans` covering three valid rows in a four-row capture. A
  focused RED/GREEN now verifies both single-request and batched-request
  padding, and the full MiniCPM hook file passed 47 tests in task `e89dba88`.
  After the fix, two concurrent response-required sessions each completed two
  spoken responses, playback acknowledgements, and history lifecycles with no
  error or stale event. Artifacts are under
  `/tmp/pr3907_piecewise_response_required_2x2_fixed_20260721`.
- An exploratory two-session, two-turn model-policy run still left one session
  waiting after its second input commit while the other session completed both
  turns. The same liveness symptom was reproduced with Stage1 eager mode before
  the padding fix, and no Stage1 fatal occurred after the fix. This run is not
  counted as PIECEWISE acceptance evidence; the remaining Stage0 resumable
  reactivation path requires separate diagnosis. Artifacts are under
  `/tmp/pr3907_piecewise_model_policy_2x2_fixed_20260721`.
- Latest broad affected matrix before the final pre-response continuation fix:
  608 passed, 19 warnings. This includes the
  Stage0 row-routing regression, cleanup-held admission regressions,
  segment-local token accounting, response-owned metric filtering, the generic
  output-processor suite, direct E2E script execution, and continuous-response
  overlap ownership. The run is task `e2146318`
  (`/tmp/remote_gpu_logs/e2146318.log`).
- The pre-response continuation regression group passed 5 focused tests. The
  response-required two-turn scenario then passed 5/5 repeated runs in task
  two-session semantic isolation plus resume, takeover, and
  admission passed in task `49225794`; model-unit overlap passed in task
  `3888387d`. The final affected matrix above supersedes the earlier pending
  rerun requirement.
- Serving plugin selection was first reproduced as three RED tests: the generic
  handler silently selected MiniCPM and the pipeline had no explicit Serving
  adapter field. The final pipeline/import/fail-fast group passed 9 tests in task
  `4db6d959` (`/tmp/remote_gpu_logs/4db6d959.log`). The token-metric and client
  audio-cadence producer-to-consumer group passed 8 tests in task `cf80f6b2`
  (`/tmp/remote_gpu_logs/cf80f6b2.log`).
- Stable-import isolation was first reproduced as a RED that loaded twelve
  `experimental.fullduplex` modules. After extracting the kernel and making
  implementation loading conditional, the isolated import regression passed.
  The focused ControlPlane/ControlClient/session/runtime/Orchestrator/AsyncOmni
  suite then passed 139 tests in task `fad613ea`
  (`/tmp/remote_gpu_logs/fad613ea.log`).
  After the final contract-import cleanup, the import-boundary, runtime, and
  lease suites passed 45 tests in task `bc138887`
  (`/tmp/remote_gpu_logs/bc138887.log`).
- Concurrency and failure-path hardening now has direct regressions for
  single-flight cancel/close cleanup, cleanup retry without dropping request
  resources, failed-replica affinity cleanup and same-address reattachment,
  append ordering across resume, takeover ownership of pending-turn
  reservations, and a closed-transport send immediately after resumable
  credentials are delivered. The final focused resume/takeover group passed 5
  tests in task `dc2ac96e` (`/tmp/remote_gpu_logs/dc2ac96e.log`).
- Stage1 batching regression: the focused test first failed because only the
  first request in `runtime_additional_information` reached the talker. After
  routing every request by `request_token_spans`, preserving per-row terminal
  flags, and emitting request-keyed sparse audio, the complete MiniCPM duplex
  hook file passed 42 tests.
- Stage0 batching regression: two requests in one batch now retain different
  `duplex_prompt_token_ids` and `special_token_ids`. The model emits row-aligned
  payload elements, so Stage0 no longer copies request zero's duplex metadata to
  later sessions.
- Admission finalization regressions cover `max_sessions=1` with blocked and
  failed expiry cleanup plus request-triggered cleanup. Every replacement is
  rejected with `resource_exhausted` until stage resources are cleaned and the
  old session is explicitly finalized.
- Resumable token accounting exports each model unit's native generation-token
  count instead of recounting the request's cumulative token IDs. Realtime
  response state resets and accumulates those segment snapshots, and the client
  ignores unowned session-level metrics while summarizing a response.
- Engine-owned admission and two-session model-policy E2E: two random session
  IDs were admitted and remained isolated, a third was rejected with
  `resource_exhausted`, closing one accepted session returned capacity, and a
  random replacement session was admitted. The capability projection reported
  `configured_limit=2`, `advertised_multi_session=true`, and
  `session_admission_mode=engine_managed`. In the same run one session selected
  speak and the other selected listen without identity leakage. Detach/resume
  advanced attachment generation from one to two and rotated its token; active
  takeover rejected all four writes from the replaced connection. The run is
  task `1d9dd1ed` (`/tmp/remote_gpu_logs/1d9dd1ed.log`), with artifacts under
  `/tmp/pr3907_resume_takeover_native_metrics`.
- Concurrent two-session, two-turn response-required E2E: both sessions produced
  two complete spoken responses, committed two playback acknowledgements and
  history records, and emitted no error, cancel, truncate, or stale events. The
  semantically distinct inputs produced isolated transcripts containing
  "当时" and "一加一" respectively. Per-response Stage0 output counts were 20,
  38, 32, and 33 tokens rather than cumulative request totals. The run is task
  `a16f1467` (`/tmp/remote_gpu_logs/a16f1467.log`), with artifacts under
  `/tmp/pr3907_semantic_2turn_native_metrics`.
- Disconnect-grace E2E: a detached session expired after the configured
  30-second engine grace, and a later resume was rejected with
  `session_resume_expired`. The run is task `9872d50c`, with artifacts under
  `/tmp/pr3907_0719_disconnect_grace_expiry`.
- A separate black-box takeover client exercised the current backend with two
  semantically distinct concurrent inputs. Active resume replayed 17 events
  including five audio deltas and committed playback history; takeover rejected
  all four stale-attachment sends, emitted `session.replaced`, and preserved
  attachment/session isolation. Both transcripts contained only their expected
  semantic token. The client task is `6773bd34`
  (`/tmp/remote_gpu_logs/6773bd34.log`), with artifacts under
  `/tmp/pr3907_gaohan_final_takeover`. The backend task was `5abcd1ee`.
- Across the final response-required, overlap, and resume/takeover runs, engine
  Stage0 TTFT ranged from 31.17 to 310.18 ms, TPOT from 15.18 to 21.18 ms, and
  reported ITL p95 from 16.35 to 32.83 ms. Explicit
  `commit_to_first_audio_ms` ranged from 386.80 to 1033.25 ms, while the
  client-monotonic maximum audio chunk gap ranged from 388.30 to 1316.37 ms.
  These are response-scoped observations from a small, warm-state-sensitive
  sample, not an SLO claim. The slowest response still does not satisfy a 200 ms
  end-to-end first-output target.
- Post-merge focused handler/demo/Web/Stage0/runner/video suite: 324 passed,
  19 warnings. The run is task `f78cb7a8`
  (`/tmp/remote_gpu_logs/f78cb7a8.log`). The Web contract explicitly rejects
  browser VAD markers and browser-generated input commits while the merged
  video path retains camera-frame admission.
- No-response listen and residual accounting: the focused handler regressions
  passed after first reproducing both failures. A model-listen unit without a
  Realtime response no longer defers the next committed input, and three
  consecutive 1.2-second `pcm_f32le` inputs each returned pending input bytes
  to zero.
- Model-unit overlap E2E: the second physical input streamed while the first
  response remained active. Seven overlap decisions had
  `defer_runtime_append=false`; the one-second model-unit boundary occurred
  before the first `response.done`, and no barge-in, cancel, truncate, stale, or
  error event occurred. The model-policy contract accepts a later unit ending
  the continuous response instead of requiring a second physical-input-owned
  response. The two response-owned Stage0 snapshots independently reported 20
  tokens / 152.8 ms TTFT and 19 tokens / 43.3 ms TTFT. The run is task
  `a4977622` (`/tmp/remote_gpu_logs/a4977622.log`), with artifacts under
  `/tmp/pr3907_overlap_native_metrics_scoped`.
- The split multi-session E2E runner is directly executable from the repository
  even when another installed package owns the top-level `tests` name. Its
  script-import and continuous-terminal regressions passed in task `f11b106c`.
- Pinned response-required E2E: the first 1400 ms of the pinned fixture produced
  one complete spoken response, five audio/transcript deltas, and one playback
  history commit, with symmetric created/audio-done/done lifecycle and no
  duplicate `response.speak`. The client task is `a76dec52`
  (`/tmp/remote_gpu_logs/a76dec52.log`).
- Continuous-input model-policy E2E: 36 browser-like 200 ms chunks produced
  seven model-listen decisions with zero input commits and no
  error/cancel/truncate/stale events. The client task is `fb256595`; artifacts
  are in `/tmp/minicpmo_pr3907_continuous_model_policy_20260718`.
- Three-response continuous-input E2E: three consecutive 9469 ms streams,
  composed of the original fixture plus four seconds of silence, produced three
  spoken responses in one session. All three completed `response.done`, audio
  transcript, playback ACK, and history commit; the run had 22 audio/transcript
  deltas, zero input commits, and no error/cancel/truncate/stale events. This is
  lifecycle evidence, not semantic cross-input independence, because the same
  fixture was repeated. The client task is `2a5f29f0`; artifacts are in
  `/tmp/minicpmo_pr3907_continuous_three_response_20260718`.
- The fresh validation backend was stopped after the E2E runs and GPU0/1 returned
  to baseline. Audio-quality and ASR evaluation were not rerun for these
  lifecycle-only changes. Other GPU workloads were not modified.

### Regression lineage

Earlier synchronized checkpoints passed broader affected matrices before the
extension, policy, and ledger consolidation. Those runs were useful while
developing the refactor but are superseded by the current file-identical H20
matrix and E2E evidence above; they are intentionally omitted to avoid mixing
different trees in the acceptance argument.

### Focused suites

- RPC router and AsyncOmni shutdown/error routing;
- orchestrator fatal and duplex control paths;
- WebSocket Actor mailbox and writer;
- duplex protocol and Session ownership;
- Realtime handler, terminal ordering, overlap, playback ACK;
- scheduler resumable boundary;
- MiniCPM runner and Stage0/Stage1 lifecycle.

### E2E contract

Run two deliberately separate fixture contracts:

- model-policy uses three distinct inputs, no transcript hints, 200 ms
  real-time pacing, and continuous input with no browser commit. It allows
  natural listen or speak decisions and assumes no 1:1 mapping between
  physical inputs, model turns, and responses;
- response-required uses `minicpmo_pr3907_jiayan_16k.wav` pinned to SHA-256
  `2e5fd4eb3ee434ce107ee3a0591fa624a33f7683c7462f45fe651c443c9af941`,
  sends its first 1400 ms, and requires one complete spoken response.
  It is an output-lifecycle fixture, not evidence that arbitrary no-hint,
  real-time input must select speak;
- late and old-response playback ACK handling in the affected test matrix;
- final model-policy response drain before `session.close`;
- close/disconnect with no post-terminal audio;
- transcript delta/final consistency;
- generated WAV validity and external ASR sanity.

Distinct-input fixtures must also be semantically distinct. Different files or
waveform hashes containing the same greeting can legitimately produce the same
reply and are not sufficient evidence for cross-turn transcript independence.

For each run record the exact SHA/worktree, command, server log, client log,
WAV path, and cleanup result.

## Acceptance Criteria

This checkpoint is ready to publish only when:

- importing/probing the append contract does not modify the specified upstream
  Core/client classes or status enum, and missing 0.28 fields fail closed;
- all three shared models successfully route native Stage0 through the explicit
  admission/atomic utility contract, while replay remains independently gated;
- initial acknowledgement waits for the full prompt without a finalize RPC;
- one mailbox passes exactly-once and wire-order tests;
- cancel advances the engine fence even when the old append continues in an
  executor, and the next-epoch append remains accepted;
- request completion removes MiniCPM helper and model-owned force-listen state;
- RPC fatal/close wakes every waiter and no new work enters after terminal;
- Session is the only serving response/overlap authority;
- stale terminal events cannot mutate a newer epoch;
- an accepted terminal cannot be dropped by the writer;
- no duplicate `response.speak` or transcript delta appears;
- Stage0 KV continuity and Stage1 turn reset pass;
- a real matching-CUDA multi-turn audio E2E passes for each migrated model,
  including complete-answer checks beyond receipt of chunk/turn EOS;
- ordinary non-duplex paths in the affected matrix still pass;
- no local path, proxy token, temporary profiling probe, or test-only switch is present.

Report only the exact snapshot, models, hardware, concurrency and failure
shapes that passed. Historical passes do not establish this cleanup's result.
MiniCPM rollover/replay and PersonaPlex/Nemotron typed state-loss failure are
separate acceptance items, not a single cross-model recovery claim.

Passing those checks does not support claims for an acoustic-onset barge-in latency target across
arbitrary client chunk sizes, raw KV-tensor append or live migration,
unbounded fairness/starvation guarantees, lossless arbitrary-context
compaction, or scheduler-native H20/H100/NPU sign-off without corresponding
runs. Equivalent complete outputs and repeated baseline measurements are
required before reporting any performance gain.
