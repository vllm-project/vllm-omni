# Unified Full-Duplex Framework

This document is the design record of the full-duplex serving framework:
the engine-resident session model, the typed command/event contract, the
single model plugin, and the serving/API surfaces built on top of them. It is
the owner document for `vllm_omni/engine/duplex/**`,
`vllm_omni/engine/duplex_omni_engine.py`,
`vllm_omni/engine/duplex_orchestrator.py`,
`vllm_omni/entrypoints/duplex_omni.py`, `vllm_omni/entrypoints/duplex/**`,
`vllm_omni/clients/**` and the per-model `duplex/` packages.

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
        │ DuplexOmni(AsyncOmniBase)                  [entrypoints/duplex_omni.py, thin] │
        │   open_session / get_session / resume_session / detach_session / close_session│
        │   DuplexSessionHandle: submit(DuplexCommand) / events() -> DuplexEvent       │
        └──────────────────────────────────────┬───────────────────────────────────────┘
                                               │ engine.open/close/resume/touch_session_async (RPC)
                                               │ engine.submit_command_async (one-way)
                                               ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │ DuplexOmniEngine(OmniEngineBase)  [engine/duplex_omni_engine.py, thin]        │
        │   loads the plugin, validates session_mode, builds queue messages            │
        │  └─ DuplexOrchestrator(OrchestratorBase)        [engine/duplex_orchestrator.py]│
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

Each layer has one turn-based and one duplex sibling over a shared base:

```text
OmniBase                        OmniEngineBase                       OrchestratorBase
 ├─ Omni        (sync, turn)     ├─ AsyncOmniEngine  (turn requests)  ├─ Orchestrator        (turn-based admission)
 └─ AsyncOmniBase                └─ DuplexOmniEngine (session msgs)   └─ DuplexOrchestrator  (hosts DuplexSessionManager)
     ├─ AsyncOmni   (EngineClient)  _create_engine -> AsyncOmniEngine   _create_orchestrator -> Orchestrator
     └─ DuplexOmni                  _create_engine -> DuplexOmniEngine  _create_orchestrator -> DuplexOrchestrator
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
3. **Generic bases and turn-based siblings have zero duplex vocabulary.**
   `OmniBase`, `AsyncOmniBase`, `AsyncOmni`, `OmniEngineBase`,
   `AsyncOmniEngine`, `OrchestratorBase` and `Orchestrator` carry only
   template seams; `tests/engine/test_duplex_import_boundary.py` checks that
   importing the turn-based stack loads no duplex module.
4. **Duplex models are served duplex-only.** `vllm-omni serve` constructs
   `DuplexOmni` when the pipeline declares `duplex_plugin`; the server exposes
   `/v1/realtime?duplex=1` (alias `/v1/duplex`), `/v1/models` and `/health`,
   and every turn-based route reports "not available". Turn-based use of the
   same model stays available offline through `Omni` / `AsyncOmni`.
5. **Serving is transport only.** The websocket handler does socket I/O,
   wire-envelope validation, command translation, event rendering and the
   attachment/resume/replay bookkeeping; it holds no session state.
6. **No `typing.Protocol`** in the duplex surfaces: the plugin, data plane,
   session state, PCM buffer, stage port and client transport seams are ABCs.

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
│   │   └── websocket.py             websocket send/close/receive helpers
│   └── openai/api_server.py         builds DuplexOmni for duplex models; duplex-only app state
├── engine/
│   ├── omni_engine_base.py          OmniEngineBase (stage processes, orchestrator thread, queues, RPC)
│   ├── async_omni_engine.py         AsyncOmniEngine (turn-based request building)
│   ├── duplex_omni_engine.py        DuplexOmniEngine (session message surface, creates DuplexOrchestrator)
│   ├── orchestrator.py              OrchestratorBase + Orchestrator
│   ├── duplex_orchestrator.py       DuplexOrchestrator (+ DuplexOrchestratorRequestState; implements DuplexStagePort)
│   └── duplex/
│       ├── commands.py              DuplexCommand dataclasses, command_from_realtime
│       ├── realtime_commands.py     Realtime client event -> DuplexCommand translation
│       ├── events.py                DuplexEvent dataclasses (+ to_realtime), REALTIME_ERROR_TYPES_BY_CODE
│       ├── realtime_events.py       RealtimeProjectionState: internal event -> typed events
│       ├── messages.py              queue envelopes (Open/Close/Resume/Touch/Command/Result/Event), DuplexSessionError
│       ├── config.py                DuplexSessionConfig, DuplexCapabilities, ResponseCreateOptions
│       ├── contracts.py             DuplexFence (session_id, epoch, turn_id), stage request records, DuplexStagePort
│       ├── session.py               DuplexEngineSession: ledgers, lease, fence, stage resources, append sequencing
│       ├── session_runner.py        DuplexSessionRunner (per-session mailbox on the orchestrator loop)
│       ├── session_manager.py       DuplexSessionManager (admission, backpressure, reaper, dispatch)
│       ├── plugin.py                DuplexModelPlugin, DuplexModelSessionState, DuplexDataPlane, PcmAppendBuffer ABCs
│       ├── lease.py                 DuplexLeaseState (idle TTL, disconnect grace, resume generation)
│       ├── turn_detection.py        server-side VAD turn detector used by the runner
│       ├── audio.py / vad.py / commit_policy.py / intermediate.py
├── config/stage_config.py           PipelineConfig.duplex_plugin; DuplexSessionRuntimeConfig
├── model_executor/models/minicpmo_4_5/duplex/plugin.py   MiniCPMO45DuplexPlugin (+ data_plane, input, policy, ...)
└── clients/
    ├── duplex.py                    DuplexClientBase (ABC), DuplexClient (websocket), client-side events
    ├── inline_duplex.py             InlineDuplexClient (in-process, over DuplexOmni)
    ├── minicpmo_4_5.py              MiniCPM-o 4.5 session preset
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
         -> output_sink (DuplexSessionEventMessage) -> DuplexOmni._route_engine_message -> handle.events()
detach   DuplexOmni.detach_session -> touch(DETACH): engine-owned disconnect grace; expiry -> SessionExpired
resume   DuplexOmni.resume_session(expected_lease_generation) -> lease CAS; the existing handle is re-entered
close    DuplexOmni.close_session -> close RPC; the manager tears the runner down, then the stage cleanup
         (abort submitted requests, release reserved ids), then SessionClosed, then the RPC result: seeing
         the event means the admission slot is free (it is held until the cleanup succeeded)
reap     DuplexSessionManager.reaper_loop: idle TTL / disconnect grace expiry, cleanup retries
```

### Concurrency and ordering model of the runner

Everything below runs on the orchestrator asyncio loop; there is no lock.

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
- **Cancel is atomic at the session.** `CancelResponse` / `BargeIn` advance
  the epoch, cancel tracked append tasks, abort the owned stage bindings
  through the stage port and emit the cancelled terminal events before
  yielding. An append task already past its last check is harmless: its stage
  request is bound to the old epoch's request id, which the abort covers, and
  the base never emits session-owned outputs to a client.
- **Ordered event stream.** Because the epoch bump and every emission happen
  on one loop in program order, nothing can be emitted for an old epoch after
  its terminal event; `runner.emit()` still drops a terminal carrying a stale
  epoch and stamps `epoch` on every event for clients that filter
  defensively. This is a statement about order, not about latency: the engine
  output queue and each `DuplexSessionHandle` outbox are unbounded and FIFO,
  so a cancellation is delivered behind whatever audio was already emitted for
  the response it cancels. A client that must stop quickly cancels its own
  playback on `response.done` / `audio.cancelled` rather than waiting for the
  stream to drain. Bounding those buffers and letting an accepted invalidation
  skip undelivered media is left to the follow-up RFC.
- **Backpressure before the mailbox.** `DuplexSessionManager.dispatch`
  checks `max_pending_input_bytes_per_session` and reserves a pending turn for
  `Commit` before the put; a rejected command is answered with
  `ErrorEvent(code="input_backpressure")`.
- **Per-response stage metrics.** `_route_output` computes the per-segment
  `StageMetrics` before calling `_intercept_stage_output`; the runner
  accumulates the snapshot into the active response, so
  `ResponseDone.stage_metrics` and `metadata.vllm_omni.stage_metrics` on the
  wire keep their meaning.

## Orchestrator seams

`OrchestratorBase` keeps the generic machinery (request handler skeleton,
abort, collective RPC, output loops, stage error and dead-replica handling,
`_route_output`, stage forwarding, prewarm, cleanup, shutdown) and offers
these seams:

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
| session policy | `capabilities(max_sessions)`, `validate_client_extra_body`, `prepare_runtime_config(config, model_config)` (server-owned runtime keys, reference audio resolution), `runtime_config_for_update`, `runtime_config_for_function_output`, `create_session_state() -> DuplexModelSessionState`, `data_plane: DuplexDataPlane` (projects raw stage outputs into internal events), `data_plane_context(...)` |

`DuplexOmniEngine._validate_deployment` loads the plugin before any stage
starts; `DuplexSessionManager.__init__` validates it against the stage
sampling defaults once the stage pools exist. Plugin hooks that may block
(`prepare_runtime_config` fetching `ref_audio`) are awaited in `open()` and
offloaded from the loop.

The MiniCPM-o 4.5 plugin (`model_executor/models/minicpmo_4_5/duplex/plugin.py`)
is the only integration in this framework version. PersonaPlex and Nemotron
VoiceChat still carry their pre-framework duplex code (runtime extension plus
serving adapter) and are therefore not served over this framework yet: their
pipelines declare no `duplex_plugin`, so they run turn-based until the
follow-up PRs port them (RFC vllm-omni#7181, PR 2/3).

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
  `tests/entrypoints/openai_api/test_duplex_api_server.py` (the duplex-only
  server: pipeline probe, app state, routes, warmup gate),
  `tests/clients/**`, `tests/engine/test_duplex_import_boundary.py`,
  `tests/model_executor/models/minicpmo_4_5/duplex/**`,
  `tests/worker/test_native_duplex_hooks.py`.
- GPU: `examples/online_serving/minicpmo/realtime_duplex_demo.py` (one turn
  over the websocket), `examples/online_serving/barge_in_client.py` (cancel
  during playback; `--inline` drives the same scenario through
  `InlineDuplexClient`), `tests/e2e/online_serving/run_minicpmo_realtime_duplex_multi_session.py`
  (admission, resume, takeover, expiry).
