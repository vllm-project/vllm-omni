# Unified Full-Duplex Framework

This document is the design record of the full-duplex serving framework:
the engine-resident session model, the typed command/event contract, the
single model plugin, and the serving/API surfaces built on top of them. It is
the owner document for `vllm_omni/engine/duplex/**`,
`vllm_omni/engine/duplex_omni_engine.py`,
`vllm_omni/engine/duplex_orchestrator.py`,
`vllm_omni/entrypoints/duplex_omni.py`, `vllm_omni/entrypoints/duplex/**`,
`vllm_omni/clients/**` and the per-model `duplex/` packages.

`vllm_omni/protocol/realtime/**` (the OpenAI Realtime API) and
`vllm_omni/protocol/duplex/**` (vLLM-Omni's extension of it) are the shared
wire codec duplex is built on. Duplex is its first consumer but not its owner: the codec
is deliberately runtime-agnostic so a second Realtime surface can reuse it
(RFC #6592 P0a).

The public wire contract lives in
[`docs/serving/realtime_duplex_api.md`](../serving/realtime_duplex_api.md);
this page describes how it is produced.

## Purpose

A full-duplex model listens and speaks at the same time: audio streams into
the model while it produces audio, the model decides when to speak, and the
user can interrupt. Serving such a model needs a long-lived session with its
own identity (epoch, turn), ledgers (input, response, playback, history), a
lease (idle TTL, disconnect grace, resume) and a binding to the resumable
stage requests that carry the session's audio through the pipeline.

The framework gives that session exactly one owner, keeps the generic
turn-based stack free of duplex code, and exposes the same session to three
consumers: the OpenAI Realtime WebSocket route, the in-process Python API and
the model plugins.

## Playback completion ownership

`DuplexEngineSession` owns response snapshots and playback cursors. Output
completion and a client's full `playback.ack` can arrive in either order.
`finalize_response_playback` is the shared check used by ACK processing,
normal response completion, draining response completion, and cancellation.
It retains state while that response is active, any of its requests are still
draining, or playback is partial without an explicit hard truncate. A hard
truncate means playback has stopped; once output ends, no further ACK is needed.
At that point, detach the stopped response from the current playback gate even
if calibration keeps its response-scoped cursor. Detach by cursor identity so
an older stop or delayed ACK cannot unblock or overwrite a newer response.
A final output
can add transcript or completion metadata after the ACK, so history is reconciled
against the retained cursor before releasing the snapshot and pending item state.
Deleted history is never reconstructed. Natural drain completion establishes
`audio_complete`; cancellation does not. A later response's cursor and history
position remain independent. Without calibration, aligned history keeps its
timing marks after a hard stop so a later explicit truncate can shorten the
stored prefix. Marks retain their original character offsets; repeated truncation
does not rescale them to the remaining text. Unaligned stops and retired
calibration evidence keep only the stored prefix and cutoff. Delayed progress
ACKs cannot change finalized history or restore unheard text. Plugin runtime
settlement follows all history writes, including cancellation writeback into a
reserved older turn; it must observe the final committed result.

Audio truncation is valid only for an item whose response produced audio.
The session checks its live cursor or retained audio provenance before any
history or playback mutation. Cursor cleanup preserves that provenance only
for stored history items; deleting or replacing finalized context removes it.
This distinguishes a retired audio reply from imported context or a reply that never emitted audio
without retaining its recording or full response snapshot. Realtime commands,
native truncate signals, and hard playback ACKs reject a no-audio target.
An ACK's implicit target is resolved before validation or mutation. Pending and
stored-item fallbacks must match `last_response_id`, which `begin_response`
updates together with the current playback cursor. A retained older reply cannot
consume the current reply's ACK, even when it is the only remaining history item.
Resolution looks up that current item directly in pending or stored history;
unrelated older or imported items do not make its identity ambiguous. This must
continue working when a partial ACK moves it from pending to stored history.
The stored-item fallback also requires the item's live or retained audio
provenance, including for audio-only replies, then passes the same ownership
and late-ACK validation as an explicit target. Imported context without audio
provenance is not selected, even if its ID starts with `item_`. Without an
identifiable target, ordinary ACKs
update only the current playback cursor and hard ACKs fail before any history or
cursor mutation.
The typed `AckPlayback` command preserves the wire `truncate` flag through
translation and mailbox delivery, so hard stops reach the same handler as
internal playback controls.

History edits pass the item's own timing marks and playback cursor explicitly;
the text-prefix calculation never reads the active response implicitly. If both
marks and duration of a known audio reply have been released, a new explicit
truncate cannot establish a heard prefix and clears the text conservatively.
Truncation preserves the empty assistant turn so the command's following playback ACK remains valid;
only deletion removes that turn. Ignored ACKs for finalized history report
`history_committed: false` and do not settle plugin runtime again, including
when calibration evidence has been retired.

Opt-in history calibration is a separate retention owner: incomplete or
hard-capped replies remain in its bounded cache for prefix correction. Every
completion path preserves that evidence until calibration retires it. Sessions
without calibration release stopped output immediately.

## Architecture

```text
              Python users                                WebSocket clients
        ┌──────────────────────┐                    ┌────────────────────────┐
        │  InlineDuplexClient  │                    │      DuplexClient      │
        │  (DuplexClientBase)  │                    │   (DuplexClientBase)   │
        └──────────┬───────────┘                    └───────────┬────────────┘
                   │   DuplexSessionHandle                      │ /v1/realtime?duplex=1 (alias /v1/duplex)
                   │                                            ▼
                   │                          ┌──────────────────────────────────────┐
                   │                          │ OmniDuplexSessionHandler (thin)      │
                   │                          │  websocket I/O, command_from_realtime│
                   │                          │  event.to_realtime(), attachment /   │
                   │                          │  resume tokens / replay journal      │
                   │                          └──────────────────┬───────────────────┘
                   ▼                                             ▼   DuplexSessionHandle
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │ DuplexOmni(AsyncOmni)                      [entrypoints/duplex_omni.py, thin] │
        │   open_session / get_session / resume_session / detach_session / close_session│
        │   DuplexSessionHandle: submit(DuplexCommand) / events() -> DuplexEvent       │
        └──────────────────────────────────────┬───────────────────────────────────────┘
                                               │ engine.open/close/resume/touch_session_async (RPC)
                                               │ engine.submit_command_async (one-way)
                                               ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │ DuplexOmniEngine(AsyncOmniEngine) [engine/duplex_omni_engine.py, thin]        │
        │   loads the plugin, validates session_mode, builds queue messages            │
        │  └─ DuplexOrchestrator(Orchestrator)            [engine/duplex_orchestrator.py]│
        │       template seams + DuplexStagePort over the shared stage machinery       │
        │       ├─ DuplexSessionManager      admission, backpressure, lease reaper,     │
        │       │                            open/close/resume/touch, command dispatch  │
        │       └─ DuplexSessionRunner ×N    ONE session state (DuplexEngineSession),   │
        │                                    ordered mailbox, append planning + stage   │
        │                                    submit, output projection, response /     │
        │                                    playback / history, VAD, typed events     │
        │       plugin: DuplexModelPlugin    (PipelineConfig.duplex_plugin, one/model)  │
        └──────────────────────────────────────────────────────────────────────────────┘
```

In each layer the duplex class extends the turn-based one, which extends
the shared base. The dependency runs duplex -> turn-based and never the
reverse, so the turn-based stack is unchanged and the duplex stack reuses
its request machinery instead of duplicating it:

```text
OmniBase                          OmniEngineBase                        OrchestratorBase
 ├─ Omni       (sync, turn)        └─ AsyncOmniEngine  (turn requests)   └─ Orchestrator        (turn-based admission)
 └─ AsyncOmniBase                      └─ DuplexOmniEngine (session msgs)    └─ DuplexOrchestrator  (hosts DuplexSessionManager,
     └─ AsyncOmni  (EngineClient)                                                                     implements DuplexStagePort)
         └─ DuplexOmni

AsyncOmni._create_engine  -> AsyncOmniEngine     AsyncOmniEngine._create_orchestrator  -> Orchestrator
DuplexOmni._create_engine -> DuplexOmniEngine    DuplexOmniEngine._create_orchestrator -> DuplexOrchestrator
```

Each concrete class constructs its collaborator in an explicit factory
method the base calls at the right moment: `OmniBase.__init__` calls
`self._create_engine(...)`; `OmniEngineBase._bootstrap_orchestrator` calls
`self._create_orchestrator(...)` on the orchestrator thread once the stages
are initialized. `DuplexOmniEngine._create_orchestrator` passes the loaded
plugin and the session runtime config to `DuplexOrchestrator` directly.

### Design rules

1. **One session authority.** The whole duplex session (config, epoch/turn,
   ledgers, lease, stage request resources, model state, projection state) is
   one object, `DuplexEngineSession`, owned by one `DuplexSessionRunner` on
   the orchestrator loop. Nothing above the engine keeps session state beyond
   a handle (session id, event queue, capabilities, public session object).
2. **Typed contract.** Commands into a session are `DuplexCommand`
   dataclasses (`engine/duplex/commands.py`); outputs are `DuplexEvent`
   dataclasses (`engine/duplex/events.py`). `command_from_realtime()` and
   `DuplexEvent.to_realtime()` derive the OpenAI Realtime JSON, so the wire
   format is never hand-built, and the websocket handler and the inline client
   share one conversion.
3. **Generic bases and the turn-based classes have zero duplex vocabulary.**
   `OmniBase`, `AsyncOmniBase`, `AsyncOmni`, `OmniEngineBase`,
   `AsyncOmniEngine`, `OrchestratorBase` and `Orchestrator` carry only
   template seams; `tests/engine/test_duplex_import_boundary.py` checks that
   importing the turn-based stack loads no duplex module.
4. **Deployment selects the serving stack.**
   `vllm-omni serve` constructs `DuplexOmni` when the pipeline declares
   `duplex_plugin` and the deploy configuration sets `session_mode: duplex`.
   An explicit `session_mode: turn` selects `AsyncOmni` instead, without
   changing the pipeline's duplex capability. In duplex mode the server exposes `/v1/realtime?duplex=1` (alias
   `/v1/duplex`), `POST /v1/chat/completions`, `/v1/models` and `/health`, and
   every other turn-based route reports "not available". The chat route uses
   the ordinary chat service on the duplex engine, without a session, when
   the plugin declares `DuplexCapabilities.supports_chat_completions`.
   Turn-based deployments use the ordinary API initialization and retain
   model capability checks and endpoint restrictions.
5. **Serving is transport only.** The websocket handler does socket I/O,
   wire-envelope validation, command translation, event rendering and the
   attachment/resume/replay bookkeeping; it holds no session state.
6. **No `typing.Protocol`** in the duplex surfaces: the plugin, data plane,
   session state, PCM buffer, stage port and client transport seams are ABCs.
7. **The wire codec is not duplex, and a wire type is not a mailbox message.**
   The event and command *vocabulary* plus its wire rendering, the audio format
   negotiation, the conversation-item rules and the error envelope are model-
   and runtime-agnostic, and split along the three tiers
   [`docs/serving/realtime_duplex_api.md`](../serving/realtime_duplex_api.md)
   already defines: **Tier 1** (identical to OpenAI) is
   `vllm_omni/protocol/realtime/**`; **Tier 2** (OpenAI names carrying our
   extensions) and **Tier 3** (ours alone) are `vllm_omni/protocol/duplex/**`.
   A Tier 2 class *subclasses* its Tier 1 twin and declares only what it adds,
   so the OpenAI surface stays honestly pure --- `session.created` without
   `resume_token`, `input_audio_buffer.append` without `video_frames`. Our
   error **codes** are Tier 3 (`protocol/duplex/errors.py`); only the envelope
   shape and OpenAI's three `error.type` classes are Tier 1.
   What stays engine-side is everything that is *internal representation*
   rather than contract: `DuplexCommand.payload()` renders the session runner's
   mailbox dictionary, whose channel genuinely differs from the client event
   (`session.update` and the three `conversation.item.*` commands all travel on
   `turn.signal`); `engine/duplex/realtime_commands.py` decides which command a
   decoded event becomes; `engine/duplex/realtime_events.py` holds the
   session's projection state. The codec may not import the engine, the
   entrypoints, `model_executor` or the clients
   (`tests/protocol/realtime/test_protocol_import_boundary.py`), and there is
   exactly one implementation of each codec behaviour
   (`tests/protocol/realtime/test_realtime_codec_single_source.py`).
   The dependency chain is `protocol/realtime` <- `protocol/duplex` <- engine /
   entrypoints / clients, and a duplex consumer uses **only** the middle link:
   `protocol/duplex/**` re-exports the Tier 1 names it does not extend, so a
   helper that later needs a duplex-specific version (`convert_input_audio_with_rate`
   resamples to MiniCPM-o's 16 kHz rather than the client's rate) is overridden
   in one file instead of at every call site
   (`tests/protocol/duplex/test_duplex_protocol_facade.py`).
   `DuplexEvent` is therefore a plain alias of `RealtimeEvent` --- an event has
   no engine-internal half --- while `DuplexCommand` is a real class, because a
   command does.

## Package layout

```text
vllm_omni/
├── entrypoints/
│   ├── omni_base.py                 OmniBase (+ abstract _create_engine)
│   ├── async_omni_base.py           AsyncOmniBase (output pump, _route_engine_message seam)
│   ├── async_omni.py                AsyncOmni (turn-based requests)
│   ├── duplex_omni.py               DuplexOmni, DuplexSessionHandle (thin pipe)
│   ├── duplex/                      SERVING (thin)
│   │   ├── serving.py               OmniDuplexSessionHandler
│   │   ├── realtime_input.py        RealtimeEnvelope (query rules, first message), parse_resume_request
│   │   ├── session_attachment.py    DuplexSessionAttachmentRegistry (resume tokens, replay journal)
│   │   ├── audio_encoding.py        encode_audio, injected into DuplexOmniEngine for the plugin's data plane
│   │   ├── openai.py                dispatch_realtime_websocket (turn-based /v1/realtime, when duplex mode is not selected)
│   │   ├── warmup.py                startup warmup probe; wait_for_duplex_warmup holds clients until it finishes
│   │   └── websocket.py             websocket send/close/receive helpers
│   └── openai/api_server.py         builds DuplexOmni for duplex models; session-backed app state
├── protocol/                        SHARED WIRE CODEC (no engine / no model / no transport)
│   ├── realtime/                    the OpenAI Realtime API itself
│   │   ├── events.py                RealtimeEvent + OpenAI's 30 server events (22 ship as Tier 1;
│   │   │                            8 have a Tier 2 twin in protocol/duplex)
│   │   ├── commands.py              RealtimeCommand + OpenAI's 10 client commands (8 Tier 1,
│   │   │                            2 with a Tier 2 twin) --- wire_type + fields, no payload()
│   │   ├── formats.py               audio format spellings, what is supported, per-session validation
│   │   ├── session.py               session-object readers; RealtimeInputDefaults (append defaults)
│   │   ├── items.py                 conversation item shape, truncation, transcripts, camera frames
│   │   ├── audio_input.py           decode_audio_append -> RealtimeAudioAppend; client speech hints
│   │   ├── audio.py                 PCM / G.711 / WAV decode, resample, re-encode
│   │   ├── errors.py                RealtimeProtocolError (the envelope's exception type)
│   │   └── capabilities.py          RealtimeProtocolCapabilities: what one consumer can serve
│   └── duplex/                      Tier 2 + Tier 3, and the one door a duplex consumer uses
│       ├── __init__.py              re-exports the Tier 1 helper functions (audio, formats,
│       │                            items, session, capabilities) so the engine never
│       │                            imports protocol/realtime directly
│       ├── events.py                the whole 42-event vocabulary: 22 Tier-1 re-exports,
│       │                            8 Tier-2 subclasses (session.created + resume_token, ...),
│       │                            12 Tier-3 events (listen/speak, playback, resume, ...);
│       │                            DuplexEvent = RealtimeEvent
│       ├── commands.py              the whole 17-command vocabulary: 8 Tier-1 re-exports,
│       │                            2 Tier-2 subclasses (append + hints, commit + final),
│       │                            7 Tier-3 commands (barge_in, playback.ack, ...)
│       └── errors.py                our error-code vocabulary (Tier 3) + RealtimeProtocolError
├── engine/
│   ├── omni_engine_base.py          OmniEngineBase (stage processes, orchestrator thread, queues, RPC)
│   ├── async_omni_engine.py         AsyncOmniEngine (turn-based request building)
│   ├── duplex_omni_engine.py        DuplexOmniEngine (session message surface, creates DuplexOrchestrator)
│   ├── orchestrator.py              OrchestratorBase + Orchestrator
│   ├── duplex_orchestrator.py       DuplexOrchestrator (+ DuplexOrchestratorRequestState; implements DuplexStagePort)
│   └── duplex/
│       ├── commands.py              mailbox half: DuplexCommand.payload() + the mailbox `type`,
│       │                            paired with each protocol command; command_from_realtime
│       ├── realtime_commands.py     duplex binding of the codec: decoded event -> DuplexCommand,
│       │                            DUPLEX_REALTIME_CAPABILITIES, duplex_response_format
│       ├── events.py                re-export shim (DuplexEvent = RealtimeEvent) + the runner's
│       │                            epoch-filter sets DOMAIN_TERMINAL_EVENTS / MODEL_OUTPUT_EVENTS
│       ├── delivery.py              DuplexOutputBuffer (bounded output shared with the session handle)
│       ├── realtime_events.py       RealtimeProjectionState: internal event -> typed events
│       ├── messages.py              queue envelopes (Open/Close/Resume/Touch/Command/Result/Event), DuplexSessionError
│       ├── config.py                DuplexSessionConfig, DuplexCapabilities, ResponseCreateOptions
│       ├── contracts.py             DuplexFence (session_id, epoch, turn_id), stage request records, DuplexStagePort
│       ├── plugin.py                DuplexModelPlugin, DuplexModelSessionState, DuplexDataPlane, PcmAppendBuffer ABCs
│       ├── turn_detection.py        server-side VAD turn detector used by the session
│       ├── audio.py                 compatibility re-export, via protocol/duplex
│       ├── vad.py / intermediate.py
│       └── session/                 one engine-resident session and everything that runs it
│           ├── engine_session.py    DuplexEngineSession: ledgers, lease, fence, stage resources, append sequencing
│           ├── runner.py            DuplexSessionRunner (per-session mailbox on the orchestrator loop)
│           ├── manager.py           DuplexSessionManager (admission, backpressure, reaper, dispatch)
│           ├── context.py           DuplexSessionContext / DuplexRunState: what the runner shares with its components
│           ├── emitter.py           SessionEmitter: projection, epoch filter, domain effects of a terminal event
│           ├── model_channel.py     ModelChannel: submit an append, project stage output, continue a turn
│           ├── control.py           SessionControl: server VAD and the events that reconfigure it
│           ├── append_task.py       AppendAttempt: one append in flight and the rollback its failure owes
│           ├── helpers.py           pure reads and payload builders over a session
│           ├── lease.py             DuplexLeaseState (idle TTL, disconnect grace, resume generation)
│           └── overlap_policy.py / commit_policy.py / playback_ledger.py
├── config/stage_config.py           PipelineConfig.duplex_plugin; DuplexSessionRuntimeConfig
├── model_executor/models/minicpmo_4_5/duplex/plugin.py   MiniCPMO45DuplexPlugin (+ data_plane, input, policy, ...)
├── model_executor/models/nemotron_voicechat/duplex/plugin.py   NemotronVoiceChatDuplexPlugin (+ data_plane, input, capabilities)
├── model_executor/models/personaplex/duplex/plugin.py   PersonaPlexDuplexPlugin (+ stage0, data_plane, input, policy)
├── model_executor/models/qwen3_omni/duplex/plugin.py   Qwen3OmniDuplexPlugin (+ input, session)
├── model_executor/models/aura_omni/duplex/plugin.py   AuraDuplexPlugin
└── clients/
    ├── duplex.py                    DuplexClientBase (ABC), DuplexClient (websocket), client-side events
    ├── inline_duplex.py             InlineDuplexClient (in-process, over DuplexOmni)
    ├── minicpmo_4_5.py              MiniCPM-o 4.5 session preset
    ├── nemotron_voicechat.py        Nemotron VoiceChat session preset
    └── personaplex.py               PersonaPlex session preset
```

## Session identity

- **The server allocates every session id.** `DuplexOmni.open_session`
  generates `duplex-<uuid4 hex>`; a `session_id` / `id` key inside the
  Realtime session object is ignored (Realtime clients echo the session object
  back). The allocated id reaches the client in `session.created` and is the
  only handle for `session.resume`, `close` and every command.
- **Ids are never reused** within an engine's lifetime, so the id alone
  identifies a session. There is no incarnation counter: stage request ids are
  `duplex-s.<b64 session id>.e.<epoch>.r.<role>`, model-side per-session state
  is keyed by `session_id` alone, and a command, close, touch or resume for an
  id the manager no longer holds is answered with `unknown_session`.
- **Staleness inside a live session is epoch-based.** `DuplexFence
  (session_id, epoch, turn_id)` is an engine-internal identity: every stage
  request is bound to the fence it was submitted under, cancels advance the
  epoch, and tracked tasks re-check the epoch after each `await`.
- **Resume identity** is `(session_id, resume_token, expected_lease_generation)`:
  the token belongs to the attachment registry, the lease generation is the
  engine-side compare-and-swap.

## Session lifecycle

```text
open   DuplexOmni.open_session(config)
         -> engine.open_session_async(session_id, DuplexSessionConfig)      [RPC]
         -> DuplexSessionManager.open: admission (max_sessions, closing sessions keep their slot),
            plugin.validate_client_extra_body / prepare_runtime_config, DuplexEngineSession,
            Stage0 request id reserved (ensure_stage_request), DuplexSessionRunner.start()
         -> first event: SessionCreated (announces the allocated id)
command  handle.submit(DuplexCommand)
         -> engine.submit_command_async -> DuplexSessionCommandMessage on the request queue [one-way]
         -> DuplexSessionManager.dispatch: unknown_session / input_backpressure checks, then runner mailbox
output   DuplexOrchestrator._intercept_stage_output -> runner.on_stage_output -> mailbox -> typed events
         -> session's bounded output buffer -> handle.events(); closure and errors without a live session
            use output_sink (DuplexSessionEventMessage) -> DuplexOmni._route_engine_message
detach   DuplexOmni.detach_session(expected_lease_generation) -> touch(DETACH): engine-owned disconnect
         grace, refused for a lease newer than the one the caller held; expiry -> SessionExpired
resume   DuplexOmni.resume_session(expected_lease_generation) -> lease CAS keyed by the control id (a replay
         answers with the generation it produced); the existing handle is re-entered. A caller cancelled
         mid-RPC observes the outcome afterwards (replaying until the engine answers) and settles a landed
         resume: the generation goes to the resume waiting to activate, else to the connection still
         attached, else the lease is detached again
close    DuplexOmni.close_session -> close RPC; the manager tears the runner down, then the stage cleanup
         (abort submitted requests, release reserved ids), then SessionClosed, then the RPC result.
         SessionClosed is emitted after the cleanup attempt (in a finally), so it is also sent when the
         stage cleanup failed; in that case the admission slot is intentionally retained and the cleanup
         is retried by the reaper, and opening a replacement session can still be refused with
         resource_exhausted. Seeing the event therefore does not by itself guarantee a free slot.
reap     DuplexSessionManager.reaper_loop: idle TTL / disconnect grace expiry, cleanup retries
```

### Concurrency and ordering model of the runner

The runner's session state is owned by the orchestrator asyncio loop and
needs no lock. The output buffer shared with the caller thread uses a lock
for queue updates and the final validity check before delivery.

- **Single writer per session.** `DuplexEngineSession` is mutated only by its
  runner. Inputs reach the runner through one mailbox in this order: client
  commands (from `_request_handler`, in request-queue order), stage outputs
  (`on_stage_output` enqueues an internal item; it does not mutate the session
  inline), timers (silence continuation, bounded auto-response) and internal
  items (deferred overlap promotion, VAD-synthesized commits). The mailbox
  worker processes items one at a time.
- **Slow work is not awaited inline.** An `AppendAudio` handler reserves
  bytes, then schedules a tracked append task on the per-session append tail:
  the task awaits `_offload(decode/resample/VAD)` on a dedicated
  `ThreadPoolExecutor` owned by the manager, then `plugin.plan_append`, then
  `stage_port.submit`. The worker returns to the mailbox immediately, so a
  later `CancelResponse` is not blocked behind an in-flight append.
- **Re-validation after every `await`.** A tracked task captures `(epoch,
  turn_id)` when it starts and re-checks them before the stage submit and
  before committing ledgers; a task that finds the epoch advanced rolls back
  its PCM reservation and exits.
- **Buffered commit prefixes.** A model buffer may reserve non-terminal units
  before its terminal commit payload. The runner submits these in wire order
  with `force_listen=True` and waits for append acceptance before acknowledging
  the commit. A commit that must wait for an active response holds those
  prefixes and submits them, still in wire order, when that response ends,
  before the retained terminal payload. Reservations keep their normal rollback
  ownership. MiniCPM uses this path to preserve separate buffered VISION units
  and a following speech tail in manual commit/response sessions. Native
  resumable responses also defer video-only commits. If response creation is
  disabled on the commit, a later explicit `response.create` must drain the
  held prefixes before submitting the terminal payload.
- **Cancel is atomic at the session.** `CancelResponse` / `BargeIn` advance
  the epoch, cancel tracked append tasks, abort the owned stage bindings
  through the stage port and emit the cancelled terminal events before
  yielding. An append task already past its last check is harmless: its stage
  request is bound to the old epoch's request id, which the abort covers, and
  the base never emits session-owned outputs to a client.
- **Tool calls settle once, and the epoch settles them.** A function call the
  runtime exposes to the client (`response.function_call_arguments.done`) is
  recorded in a session-owned ledger before it is emitted, and its
  `function_call_output` is admitted only against that record:
  `pending -> accepted | timed_out | cancelled`, one transition, no rebinding
  of a call id, and terminal records kept as tombstones until the session
  closes. Conversation history is not admission authority -- it is history-wide
  and epoch-blind, so a result arriving after a barge-in could otherwise be
  admitted against a call the session had already moved past. The epoch
  consequences hang off the single epoch-advance path: pending calls from older
  epochs are cancelled, and a result the model accepted but never consumed is
  retired from the runtime so the new epoch cannot speak it. A retired or
  already-consumed generation stays *named* in the worker-facing runtime config
  (model-owned state is per resumable stage request, and a cancel moves the
  session to a different one), so the next request starts above what the
  previous one took instead of re-speaking it or rejecting the next result as
  missing; a generation that only *looks* missing is still refused by the
  worker. Eligibility is the
  origin epoch plus the settlement, never the current turn: a model turn
  completing while the client answers does not invalidate the result, and the
  continuation it triggers targets the turn that is current at admission.
  An optional deadline (`extra_body.tool_result_timeout_s`, off by default)
  times a pending call out; its timer callback only enqueues the call identity
  onto the session's mailbox, so the deadline decision is serialized with every
  other command instead of running inside an `asyncio` timer callback.
- **Ordered event stream.** Because the epoch bump and every emission happen
  on one loop in program order, nothing can be emitted for an old epoch after
  its terminal event; `runner.emit()` still drops a terminal carrying a stale
  epoch and stamps `epoch` on every event for clients that filter
  defensively. In this experimental follow-up, public output goes directly to
  one bounded buffer shared by the engine and `DuplexSessionHandle`. Accepted
  cancellation removes queued `AudioDelta` events for that response through
  the cancelled epoch. Text increments, audio/transcript completion markers,
  other responses and normal completion retain FIFO order. The consumer also
  checks an event already taken from the buffer: WebSocket delivery repeats
  that check under the connection lock, before sequencing and journaling.
  Already sequenced replay entries are unchanged; already delivered audio
  still requires the client to stop its own playback.
- **Backpressure before the mailbox.** `DuplexSessionManager.dispatch`
  checks `max_pending_input_bytes_per_session` and reserves a pending turn for
  `Commit` before the put; a rejected command is answered with
  `ErrorEvent(code="input_backpressure")`.
- **Per-response stage metrics.** `_route_output` computes the per-segment
  `StageMetrics` before calling `_intercept_stage_output`; the runner
  accumulates the snapshot into the active response, so
  `ResponseDone.stage_metrics` and `metadata.vllm_omni.stage_metrics` on the
  wire keep their meaning.

### Semantic turn decisions

`SessionControl` owns a detector per session and offloads its ordered audio
processing to the manager's executor. `SemanticTurnDetector` loads the shared
Smart Turn backend on the first audio chunk; the backend constructor warms the
CPU ONNX session. Silero's candidate pauses trigger subsequent scoring. Each
detector retains the current pause's semantic decision until speech resumes or
the turn resets. At commit, session-correlated logs distinguish a complete
semantic decision from the silence timeout, including whether the preceding
score was incomplete or inference failed. These diagnostics do not change the
wire protocol or the endpoint decision.

Only unfinished response-bound append tasks keep the response gate closed.
A completed task awaiting its cleanup callback does not block a new turn after
barge-in. Active responses and unplayed output retain their own gates.

### Experimental output limits

`max_pending_output_bytes_per_session` defaults to 2 MiB and
`max_pending_output_events_per_session` to 512. The byte limit counts compact
Realtime JSON, including base64 audio, not decoded PCM or Python object
overhead. Audio is already base64-encoded by the producer; counting its length
avoids serializing or decoding that large field again. Neither default has
been tuned from a performance study.

Errors and response endings have an additional 64 KiB / eight-event reserve
and retain FIFO order. One final session-closure notification has an independent
slot, so an exhausted reserve cannot prevent closure or break delivery to
other sessions. If that reserve also fills, an error or response-ending
notification may be omitted; the final session-closure notification still
has its independent slot. Oversized close details (over 4 KiB with generated identity)
are replaced by a compact reason, preserving terminal type and identity.
Late writes after closure are ignored; a locally ended iterator is not reopened
by a late terminal, and a failed open closes the abandoned handle. Events other
than errors are not forwarded once their session's buffer has been removed.
One event held by the consumer is outside the queued budget.

The overflow policy fails the active response without committing its partial
text to history and closes only the affected session. If overflow occurs while
a completion batch is being delivered, its not-yet-queued `ResponseDone` is
converted to failure with its identity and output preserved; any history entry
for that response is removed. An ending already accepted by the buffer is not
replaced or followed by a second ending. After overflow, newly
emitted ordinary events are suppressed, including text/audio completion,
content-part and output-item completion markers. Previously queued non-audio
events retain FIFO order; error and failed response-ending notifications may
use the reserve before the independent closure notification. This differs from
accepted cancellation, which removes only matching audio. See the
[public error and closure fields](../serving/realtime_duplex_api.md#output-limits-and-slow-consumers).
Response-only recovery is not implemented. These limits do not bound the runner's raw stage
output mailbox, replay journal, or client playback queues. The shared buffer
requires the current same-process, separate-thread engine arrangement; it is
not a cross-process delivery protocol. Direct handle iteration and WebSocket
delivery share buffer validity, but end-to-end client behavior still needs
independent real-model verification.

## Session-aware replica placement

The opt-in `duplex_session.replica_policy: least_sessions` policy places each
session on a local replica group mapping every stage to a replica. The
orchestrator owns a separate session reservation ledger; StagePool's request
bindings continue to describe submitted requests only. Admission reserves a
group synchronously after taking the global admission slot. The group must
have space under `max_sessions_per_replica` on every member replica. Replicas
shared by groups share their capacity counter. Eligible groups are ordered by
maximum reserved-to-capacity ratio, total reserved count, and round-robin
tie-breaking.

The reservation covers opening, active and closing sessions. Failed opens
roll it back; ordinary request completion, response cancellation and epoch
changes do not release it. Session cleanup releases it only after the closing
session is finalized. Every Stage0 submission, downstream forward, prewarm and
derived request uses the session's pinned stage replica. Routing is fixed
before multimodal preprocessing and connector sender information are built.
A conflicting or unavailable pin fails instead of falling back to another
replica.

Replica failure marks the replica unavailable and closes every session in its
reverse index, including idle sessions without an in-flight request. Other
groups retain their reservations and can continue serving. This policy does
not migrate model state or support coordinator/distributed deployments.
Worker-local state capacity uses the per-replica cap, while `max_sessions`
remains the global service cap. The default `legacy` policy preserves the
existing routing, admission and worker capacity behavior. See the
[configuration reference](../configuration/stage_configs.md#duplex-replica-placement)
for group validation and model capability requirements.

With metrics enabled, `vllm_omni:duplex_reserved_sessions` reports reservations
by model, stage and replica, and `vllm_omni:duplex_admission_rejected_total`
counts global and replica capacity rejections. These collectors are created
on first use by this policy and carry no session ID labels.

### Multiple frontends

`--api-server-count K` reuses the existing local multi-API launcher. The parent
launches all stage cores, each core exposes one IPC endpoint pair per frontend,
and every frontend attaches to every replica. Each frontend has its own
`DuplexOrchestrator`, `DuplexSessionManager` and Python interpreter. Request
`client_index` routes engine output back to the submitting frontend; there is
no central per-frame session dispatcher.

Placement, rather than engine attachment, enforces exclusive ownership. Group
`g` belongs to frontend `g % K`, and only that frontend may reserve its replicas
or submit its session requests. Groups sharing a physical stage replica must
have the same owner, and `K` must not exceed the group count. The global session
budget is divided by floor/remainder across frontends and bounded by each
frontend's owned replica capacity; budgets do not move between frontends.
The reservation lifetime, strict stage pins and worker caps apply within each
owner's groups. With `K=1`, routing and admission behavior are unchanged.

The multi-frontend WebSocket guard allows only `least_sessions` and plugins
without session resume. Credentials, attachment state and replay journals
remain process-local. The shared public address does not provide resume
affinity; supporting resume-capable models would require a separate design.
INET listeners use the existing per-process `SO_REUSEPORT` path where available
(with a shared-socket fallback). Kernel connection distribution is not capacity
aware, so a locally full frontend can return `resource_exhausted` with
`retryable: true` even when another has room. Clients reconnect with backoff;
an established session never moves to another frontend.

When enabled, warmup runs against the local engine and owned groups rather than
connecting through the public listener. PersonaPlex keeps its opt-in behavior:
`duplex_session.warmup_frames` must be positive. Each frontend keeps its own
warmup completion gate; clients wait at most `DUPLEX_WARMUP_CLIENT_WAIT_S` (180 s)
before being admitted, as in single-frontend mode; there is no all-frontend
warmup barrier. `/health`
retains the engine health check, and a 200 response does not guarantee admission
capacity on the frontend accepting a connection. Only the owner publishes its
replicas' reservation gauges, using Prometheus `livesum` aggregation;
admission rejection counters aggregate across
frontends without changing their label schema.

This mode keeps the current supervisor's whole-service fail-fast behavior on
an engine failure or a frontend's abnormal exit. It does not add frontend
restart, session migration or failure isolation. All-frontend engine attachment
and SHM notification fan-out are unchanged; narrowing SHM wakeups is a separate
optimization. See the [configuration reference](../configuration/stage_configs.md#multiple-duplex-frontends)
and [serving guide](../serving/realtime_duplex_api.md#multiple-frontends).

## Orchestrator seams

`OrchestratorBase` keeps the generic machinery (request handler skeleton,
abort, collective RPC, output loops, stage error and dead-replica handling,
`_route_output`, stage forwarding, prewarm, cleanup, shutdown) and offers
these seams. `DuplexOrchestrator` extends `Orchestrator` rather than sitting
beside it: a seam it does not override falls through to the turn-based
implementation, and `_dispatch_message` hands anything that is not a session
message to `super()`:

| Seam | `Orchestrator` | `DuplexOrchestrator` |
| --- | --- | --- |
| `_dispatch_message(msg) -> bool` | `add_request`, `streaming_update`, `add_companion_request`, `interaction` | open/command/close/resume/touch messages -> `DuplexSessionManager` |
| `_background_tasks()` | — | `[manager.reaper_loop()]` |
| `_shutdown_extensions()` | — | `await manager.shutdown()` |
| `_on_stage_submitted(stage_id, request_id, replica_id, req_state)` | — | bind the forwarded stage request to the owning session |
| `_intercept_stage_output(...) -> bool` | — | hand every session-owned output (Stage0 decision, Stage1 audio) plus its metric snapshot to `runner.on_stage_output`; `True` = do not forward |
| `_handle_forward_failure(...) -> bool` | — | fail the owning session's response and close the session, keep the loop alive |
| `_cleanup_request_ids(ids, abort=, release_owners=)` | ignores `release_owners` | brackets `super()` with `close_sessions_for_request_ids` / `defer_request_cleanups` / `finalize_closed_sessions` |
| `OrchestratorRequestState.session_owned` | never set | set by `ensure_request`: no synthetic terminal, no client emission, no terminal re-forward, no auto-cleanup on finish |

`DuplexOrchestrator` also implements the `DuplexStagePort` ABC the manager and
runner use (`stage_count`, `sampling_defaults`, `ensure_request`, `submit`,
`cleanup`, `abort_requests`): the resumable Stage0 request is preregistered at
open with `DuplexOrchestratorRequestState(session_owned=True, session_id,
fence)`, submitted with `submit_initial` on the first unit and
`submit_update` afterwards, and the request's `streaming.bridge_states["duplex"]`
carries the session identity (`session_id`, `fence`, `epoch`, `turn_id`,
`model_turn_id`), the public session config and the server-owned runtime
config for the worker-side model hooks.

## Model plugin

`DuplexModelPlugin` (ABC, `engine/duplex/plugin.py`) is the one class a model
provides, selected by `PipelineConfig.duplex_plugin`; "is a duplex model" is
`duplex_plugin is not None`. It merges the engine policy and the session
policy that used to be two separately configured objects:

| Half | Members |
| --- | --- |
| engine policy | `configure_sampling_params(runtime_config, defaults)`, `plan_append(...) -> DuplexAppendPlan` (the resumable Stage0 prompt for one unit), `decide_output(...) -> DuplexOutputDecision \| None` (e.g. the listen decision on a finished Stage0 segment) |
| session policy | `capabilities(max_sessions)`, `validate_client_extra_body`, `prepare_runtime_config(config, model_config)` (server-owned runtime keys, reference audio resolution), `runtime_config_for_update`, `parse_function_call` (recognize one call), `runtime_config_for_function_output` (feed an accepted result back into that model), `note_published_function_call` (plugin stores history length at publish), `has_pending_tool_followup` (plugin still has a tool result to answer), `extra_prompt_config`, `runtime_patch_for_submitted_prompt`, `runtime_config_after_response` (keep or drop a result after the response ends), `runtime_config_after_model_output` (consumption acknowledgement), `create_session_state() -> DuplexModelSessionState`, `data_plane: DuplexDataPlane` (projects raw stage outputs into internal events), `data_plane_context(...)` |

A duplex tool call uses one client sequence. The client supplies tools, receives one function call, runs the function, and sends `function_call_output`. The server does not execute the function. Each model plugin implements both hooks. A recognized call is published through `model_channel._expose_function_call`. Before the plugin sees a `function_call_output`, the session copies `name` and `arguments` from the call origin recorded at publish. Conversation history is not admission authority. The client item stays `call_id` and `output`. Qwen3-Omni `parse_function_call` reads a Hermes `<tool_call>` and returns `{call_id, name, arguments}` without retaining them. One Thinker chunk can carry more than one completed call; each is emitted. `note_published_function_call` stores that call's history length in the Qwen runtime. `runtime_config_for_function_output` reads `name` and `arguments` from the preparation item, refuses when either is absent, and appends one `{history_len, messages}` entry there. The stored length wins over the length at result time. Tool pairs stay out of `session.history`. `has_pending_tool_followup` is how the runner learns that a bare `response.create` may open `{"type":"conversation"}`. `plan_append` splices each turn after the history it was accepted against, and `runtime_patch_for_submitted_prompt` holds those turns aside after the append is accepted. The pair stays owned by that response until playback stores its text. The next prompt puts a still-inflight pair back on the waiting list before the usual splice, and the pair keeps that response id so the playback ACK can drop it once the assistant text is stored. A cancel, a playback clear before any audio is committed, or `realtime_response_conversation: none` stores no text and puts the pair back with that response id. A failure before playback can store the answer puts the pair back without the id, so the retry can own it. `tool_choice: none` is rejected. A result whose published anchor has left the prompt window is refused instead of being attached to the latest question. `runtime_config_after_response` is that step. Nemotron VoiceChat `parse_function_call` reads the data-plane function-channel event, and `runtime_config_for_function_output` places the result on a later frame of the request that is still open. The session call book is #8758.

`DuplexOmniEngine._validate_deployment` loads the plugin before any stage
starts; `DuplexSessionManager.__init__` validates it against the stage
sampling defaults once the stage pools exist. Plugin hooks that may block
(`prepare_runtime_config` fetching `ref_audio`) are awaited in `open()` and
offloaded from the loop.

Frame-based plugins use `engine/duplex/intermediate.py::build_duplex_append_prompt`
for the shared request identity, sequencing and config snapshots. Token budgets,
PCM framing and model-specific worker fields remain in the plugins.

Tool result conversion is preparation until the session's final deadline/epoch
check and admission. `runtime_config_for_function_output` receives an isolated
item copy with `name` and `arguments` from the immutable session call origin,
when available. Client-supplied hints for those fields are ignored; a plugin
requiring absent metadata rejects the result before admission. The client
acknowledgement retains the submitted item. Parsing and conversion should not
consume a separate call-metadata cache before admission.

A preparation refusal leaves the call pending and does not reset its deadline.
Missing origin metadata cannot be repaired with client hints: a plugin that
requires it keeps refusing retries until the call times out or is cancelled.
When the optional timeout is disabled, cancellation or session teardown is
required to retire such a call.

`on_tool_call_settled` handles metadata-only terminal cleanup, including old
epochs; `on_tool_calls_closed` disposes registered call metadata during teardown.
Model recovery uses the mailbox-ordered `runtime_config_for_tool_call_timeout`
only for a timeout in the current epoch. These hooks default to no-ops: a model
adapter must implement and validate its recovery behavior. Native Nemotron does
not override the timeout hook: ledger expiry and late-result rejection do not
establish native call abandonment or guarantee a subsequent model call. Qwen
tool-result delivery is separate model-adapter work in
[#8493](https://github.com/vllm-project/vllm-omni/pull/8493); session lifecycle
support alone is not Qwen tool-call end-to-end support. Accepted results use
their prepared candidate, while effective epoch cancellation uses
`retire_pending_function_output` for queued work according to model policy.

JoyAI-VL-Interaction uses the same call book for a different decision shape. Its
model emits an action token (`</response> <note> </delegation> <question>`), so
`parse_function_call` reads a call the plugin itself projected, not a native tool
schema, and the declared function is one opt-in `delegate` entry. A recognised
call short-circuits the rest of that stage output, so the plugin projects the call
alone and the spoken note on its own ordinary path; the note therefore still
reaches the audio stage while the call is published. An accepted result never
enters `session.history`. It becomes a `[Delegated Results]` block in later JoyAI
prompts, and `has_pending_tool_followup` reports whether one is still waiting.
JoyAI's interaction is chunk based: a result is carried by observations until a
stored reply answers it, reused once inside the chunk that answered it, and
expired when the turn that asked it is cancelled. The chunk id and the
"a reply of it stored text" flag live in the session runtime config rather than on
the plugin, because one plugin instance serves every session of a deployment.
Every hook that the session installs returns the complete runtime, not just its
own keys: the session replaces the whole config, so a partial return would drop
the model's TTS and prompt settings.

Nemotron's function-channel wire output is request-aligned. In runtime batches,the worker-provided spans select each request's last scheduled hidden row rather
than CUDA-graph padding; the worker's payload split preserves that mapping for
the session data plane. Intermediate prefill chunks expose no function token
and keep empty slots in mixed batches. The final prefill chunk and decode frames
can expose function tokens. Model-local postprocess feedback alone does not
expose a client tool call. Forced feedback remains queued until an eligible
frame's postprocess consumes it once.

Turn plugins may bind their stage identities in `configure_pipeline` and select
an append's entry with `input_stage_id(payload)` (default 0). The request context
and orchestrator retain that entry stage. Session prompt preparation receives
the request's complete `stage_sampling_params`, so a pipeline with an ASR prefix
budgets the JoyAI prompt against JoyAI's generation settings. Session model config
comes from the first tokenizer-owning stage, matching frontend comprehension
ownership. JoyAI's audio-free requests enter its `joyvl` stage directly; speech
enters ASR and carries the original observation context through the handoff.

While idle, JoyAI retains the latest camera frame and one input chunk of audio
pre-roll. Once speech starts, it keeps the utterance until commit. An idle
camera commit enters JoyAI directly; reserved-byte accounting includes the
pre-roll until the commit is consumed or rolled back.

Completed input transcriptions travel with stage outputs through the runner
mailbox. After checking the request fence, the runner updates the exact committed
audio history item and emits its transcription event. The append captures the
original native audio part before yielding and retains that identity on its
request resource; a reused public item ID does not confer transcript ownership.
Cancellation, item deletion or replacement cannot apply a late transcript to a
newer turn.

See [supported models and deployments](../serving/full_duplex_api.md#enable-full-duplex)
for the current plugin integrations and deployment configurations.

## Heard-text history calibration

With `ack_only`, the client reports how much output audio it played. Some
models have no text-to-audio alignment for an interrupted reply, so the
existing history policy may retain an empty assistant turn. Optional
calibration estimates a prefix of that reply's original text before the next
prompt is built. It remains experimental and disabled by default; configuration
and strategy limitations are described in the
[serving guide](../serving/realtime_duplex_api.md#optional-history-calibration-for-interrupted-qwen3-omni-replies).

### Ownership and callback contract

`DuplexModelPlugin.history_calibrator(runtime_config)` returns either `None`
or a `HistoryCalibrationPolicy`. The plugin selects the strategy and declares
whether it needs retained audio through `requires_audio`. The runner creates
one `HistoryCalibration` controller per session on its own asyncio loop. The
controller owns recordings, task deadlines and result versions;
`DuplexEngineSession` owns the authoritative conversation and validates every
writeback. The callback must yield during slow work and must not mutate session
history.

The async callback receives an immutable `HeardTextSnapshot`: response ID,
original text, client-confirmed playback cutoff in milliseconds, and mono
float32 PCM from the reply's start with its sample rate. A text-only policy
receives empty PCM and a zero sample rate. `complete_audio_duration_ms` is
present only after the final audio stage finishes; an interrupted generation's
partial audio duration cannot estimate the rate of its complete text. The
callback returns an exclusive character offset into the original text (`0`
means an accepted empty prefix), or `None` to keep the existing policy. It
never supplies replacement history text or an ASR transcript to insert.

### Runtime sequence and guarded writeback

1. `ModelChannel` records projected output in the controller. Under `ack_only`,
   cancellation of a response or input, output-audio clearing, barge-in,
   truncation and a final playback acknowledgement can trigger `request_ready()`.
   Each prompt build also calls it through `before_prompt()`, re-evaluating
   eligibility on every append. Only ended replies with a positive, partially
   played cutoff are eligible; active replies and replies whose TTS still drains
   wait.
2. The controller captures the original text and cutoff, starts the callback
   with `asyncio.wait_for`, and identifies the task's evidence by
   `(text, cutoff, retained_audio_bytes)`. Changed evidence replaces the task;
   unchanged evidence is not retried after failure. Callback failure, refusal
   or timeout preserves the existing conservative history policy.
3. Before freezing the stage request's runtime and preparing its conversation,
   `ModelChannel._append_via_data_plane` awaits `before_prompt()`. This has one
   `history_calibration_timeout_ms` budget (default 1000 ms) for all eligible
   tasks, including replacements; replacement work does not extend it. At the
   deadline, the controller revokes task ownership and cancels pending work.
   The append then checks the session epoch and open state before continuing,
   so cancellation or close during the wait cannot submit a stale prompt.
   History and tool runtime are captured after the wait, so a calibration that
   consumes a tool pair cannot leave that pair beside its stored answer in
   the submitted prompt.
4. A result may write back only while its controller is open, its task still
   owns that response and its evidence version still matches. The session
   additionally checks that the original response snapshot and conversation
   item exist, the original text is unchanged, the offset is in range, and
   the result's audio cutoff does not exceed the current permanent cutoff.
   Accepted results replace text at the original assistant item's position in
   history.
   Deletion, replacement, stricter truncation and close therefore reject stale
   results even if cancellation of an async callback arrives late.
5. After an accepted writeback, the controller notifies the runner. If the
   stored assistant item has nonempty text, the runner settles that response's
   runtime as committed, just as a playback ACK does. Qwen then consumes its
   tool pair without requiring another ACK. An empty prefix keeps the pair
   available for a retry, including late ACKs after its evidence is retired.
   Settlement runs outside the calibration-failure handler. A plugin failure
   while settling a stored answer emits `tool_call_settlement_failed` and closes
   that session; it cannot continue with history and tool runtime out of sync.

### Evidence lifetime and limits

The controller retains at most four replies and bounds retained PCM by
`history_audio_max_bytes_per_session` (default 8 MiB). It drops whole replies
rather than retaining a suffix that could be mistaken for audio starting at
zero. Text-only calibration retains no PCM. The byte limit excludes temporary
callback copies and does not bound the conversation's stored text.

Eviction and item deletion discard recordings and revoke pending tasks. For an
ended reply, `retire_history_calibration` also releases its original response
snapshot, playback cursor and pending reconciliation state. A still-generating
or draining reply defers retirement until its output no longer needs that
state. The stored prefix and compact cutoff metadata remain with an undeleted
conversation item. Late ACKs cannot expand a retired prefix; a later, stricter
explicit truncation clears it conservatively because its alignment evidence
has expired. Finishing calibration alone does not retire evidence that a
stricter truncation may still need. Session close closes the controller,
cancels its tasks and releases its recordings.

## Serving

`OmniDuplexSessionHandler.handle_realtime_session(websocket)`:

1. accept; build the `RealtimeEnvelope` (autostart / default session from
   query params, `event_id` correlation, error rendering through the typed
   `ErrorEvent`);
2. handshake: first message `session.update` -> `omni.open_session(session)`
   (the id inside the payload is ignored; the allocated id is announced in
   `session.created` together with `resume_token` / `attachment_generation`
   when the model supports resume); `session.resume(session_id, resume_token,
   last_received_server_event_seq)` -> `attachment.authenticate_resume` ->
   `omni.resume_session`;
3. reader loop: JSON -> `RealtimeEnvelope.translate` (`translate_realtime_command`
   with the session's declared audio defaults) -> `handle.submit`;
   envelope-level errors (invalid JSON, oversize frame, unknown type,
   event acks) are answered locally;
4. writer pump (session-scoped, survives reconnects): `async for ev in
   handle.events(): attachment.send_event(ev.to_realtime())`, journaling
   for replay until the journal overflows (`session.resync_required`);
5. disconnect: a resumable session is detached (`attachment.detach` +
   `omni.detach_session`, engine-owned grace); a superseded socket's
   disconnect is a no-op; a non-resumable session is closed. Takeover sends
   `session.replaced` to the old socket and closes it.

## Validation

- CPU suites: `tests/engine/duplex/**` (session, lease, manager, runner,
  plugin, events, commands), `tests/engine/test_duplex_orchestrator.py`,
  `tests/engine/test_duplex_omni_engine.py`,
  `tests/entrypoints/test_duplex_omni.py`,
  `tests/entrypoints/duplex/test_duplex_serving.py`,
  `tests/entrypoints/openai/test_duplex_session_attachment.py`,
  `tests/entrypoints/openai_api/test_duplex_api_server.py` (the duplex
  server: pipeline probe, app state, routes, warmup gate),
  `tests/entrypoints/duplex/test_duplex_warmup.py`,
  `tests/entrypoints/duplex/test_startup_warmup.py`,
  `tests/clients/**`, `tests/engine/test_duplex_import_boundary.py`,
  `tests/model_executor/models/minicpmo_4_5/duplex/**`,
  `tests/worker/test_native_duplex_hooks.py`.
- Calibration CPU suites: `tests/engine/duplex/test_history_calibration.py`
  covers cache eviction, live-suffix rejection and deferred retirement.
  `tests/model_executor/models/qwen3_omni/test_history_integration.py` covers
  task ownership, cancellation, prompt deadlines, failed-snapshot deduplication
  and the calibrated history in the next submitted prompt. The remaining Qwen
  `test_history_*.py` files cover prefix matching and token-rate estimates.
- GPU: `examples/online_serving/minicpmo/realtime_duplex_demo.py` (one turn
  over the websocket), `examples/online_serving/barge_in_client.py` (cancel
  during playback; `--inline` drives the same scenario through
  `InlineDuplexClient`), `tests/e2e/online_serving/run_minicpmo_realtime_duplex_multi_session.py`
  (admission, resume, takeover, expiry).
