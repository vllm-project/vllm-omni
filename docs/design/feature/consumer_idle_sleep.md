# Consumer Profile Idle Sleep

Status: Draft for review
Tracking issue: [#10008, item A4](https://github.com/vllm-project/vllm-omni/issues/10008)

## Summary

Consumer deployments should release GPU memory after a configurable period
without inference activity and transparently restore serving on the next
inference request. The policy is opt-in per deploy profile and disabled for all
existing profiles unless they explicitly configure it.

The first implementation uses sleep level 1 only. Level 2 cannot be used
because the current engine cannot reload discarded weights during
`wake_up()`. For AR stages, a successful wake must also call
`resume_generation()` before a request can enter the engine.

The design adds:

- a typed, top-level deploy-profile configuration;
- one API-process lifecycle controller with a serialized state machine;
- ASGI activity tracking for HTTP streams and WebSocket sessions;
- an engine-idle check for work that outlives an HTTP response;
- transparent wake-on-request for auto-sleep transitions;
- transition logs, deterministic tests, and a CUDA qualification procedure;
- one measured consumer-profile default, including reclaimed memory and wake
  latency evidence.

## Problem

The existing `/v1/omni/sleep` and `/v1/omni/wakeup` routes are manual control
operations. They do not define an idle policy and are not sufficient for a
local desktop service:

1. The API process does not know when the server has been idle long enough to
   release GPU memory.
2. A response can finish before engine work finishes, such as an asynchronous
   video job, so HTTP connection count alone is not a safe idle signal.
3. HTTP streaming and realtime WebSocket sessions must remain active for their
   complete transport lifetime, including quiet periods between chunks.
4. Concurrent requests can arrive while sleep or wake is in progress.
5. `AsyncOmni.sleep()` closes frontend admission before offload. For AR stages,
   `wake_up()` restores memory but intentionally does not reopen admission;
   `resume_generation()` is required.
6. `sleep(level=2)` currently discards weights and `wake_up()` raises
   `NotImplementedError`. An automatic policy must therefore not select level
   2.

Without one owner for these transitions, a timer, manual control request, new
inference request, and server shutdown can race and leave stage state
ambiguous.

## Goals

- Release weights and KV-cache memory after a profile-defined idle period.
- Never auto-sleep while an inference request, response stream, realtime
  session, or background engine request is active.
- Transparently wake an auto-slept server before admitting the next inference
  request.
- Ensure one sleep or wake transition runs at a time.
- Preserve existing behavior when the feature is not configured.
- Make transition latency, failures, and reclaimed memory observable through
  lifecycle logs and existing engine sleep acknowledgments.
- Support pure-diffusion and mixed AR/diffusion pipelines on one local server.
- Ship at least one consumer deploy profile with an evidence-backed timeout.

## Non-goals

- Level-2 automatic sleep.
- Defining one universal timeout for every consumer profile.
- Automatic per-stage sleep; the first version sleeps all stages together.
- Multi-node or multi-API-worker coordination.
- Reducing host memory. Level-1 sleep trades GPU memory for host memory.
- Changing trainer-facing `AsyncOmni.sleep()`, `wake_up()`, or
  `resume_generation()` semantics.
- Treating health, model-list, metrics, documentation, CORS preflight, or job
  result retrieval as inference activity.
- Proving an 8/12/16 GB memory tier. A1/A2 qualification supplies those
  measurements.

## RFC alignment

RFC #10008 A4 asks for two deliverables: automatic idle sleep defaults in
consumer profiles and measured wake latency. This design maps those
requirements as follows:

| RFC intent | Design response | Acceptance evidence |
| --- | --- | --- |
| Automatic sleep after configurable idle time | Top-level, opt-in `idle_sleep.timeout_s`; monotonic idle deadline | Deterministic timer tests and one end-to-end idle transition |
| Consumer-profile defaults | At least one consumer profile enables the policy, coordinated with A2 | Profile diff and its qualification report |
| Do not interrupt serving | Transport leases plus an engine-idle snapshot; level-1 `mode="wait"` | Stream, WebSocket, background-job, and race tests |
| Continue serving after sleep | Inference activity transparently performs wake plus resume for controller-owned sleep | First post-sleep request succeeds without a control call |
| Measure wake cost separately | Wake duration and first-request latency are reported separately from steady state | Raw samples, p50/p95, and profile metadata |
| Validate feature interaction | Run the selected profile with its offload, quantization, cache, and tiling settings unchanged | A1/A2 interaction-matrix row and repeated-cycle evidence |

The RFC does not prescribe transparent wake, sleep level, route classification,
or timeout value. Those are proposed here because they are necessary to turn
the existing manual endpoint into a usable localhost lifecycle policy. They
remain reviewer decisions until accepted.

## Current contracts

The design depends on the following current behavior:

- `AsyncOmni.sleep(level=1, mode=...)` blocks new frontend admission, waits for
  in-progress EngineCore submissions, and records sleeping tags.
- AR `wake_up()` restores memory but leaves frontend admission paused.
- `resume_generation()` resumes the AR scheduler and clears the frontend
  admission hold.
- Diffusion-only `wake_up()` clears its temporary admission hold.
- Level-2 wake is not implemented.
- The API process already owns all HTTP and WebSocket routes and can wrap the
  fully assembled ASGI application.
- `AsyncOmni.request_states` tracks engine requests for their full generation
  lifetime, including work that is no longer tied to an open HTTP response.

These are implementation dependencies, not new public contracts.

## Configuration

### Deploy YAML

Idle sleep is a server lifecycle policy, not a stage-engine parameter. It is
configured once at the top level of a deploy profile:

```yaml
idle_sleep:
  timeout_s: 300
```

The new typed configuration is:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `timeout_s` | `float \| null` | `null` | Idle time before automatic sleep; it must be finite and greater than zero. `null` disables the policy. |

Sleep level and wake behavior are intentionally not configurable in the first
version:

- automatic sleep always uses level 1;
- automatic sleep always targets all stages;
- inference routes always wake an auto-slept server transparently.

This avoids exposing combinations that the current engine cannot recover
from. A normal profile remains disabled because `timeout_s` defaults to
`null`. The opt-in `moss_tts_nano_idle_sleep.yaml` profile uses `300` seconds
as an initial interactive policy candidate without changing
`moss_tts_nano.yaml`. Its L20 run validates profile composition and lifecycle
behavior, but only a consumer-GPU qualification run can establish a consumer
hardware support claim or confirm that timeout for a consumer tier.

### Selecting a profile default

For each profile that enables idle sleep:

1. measure five or more level-1 wake cycles with the profile's complete
   offload, quantization, cache, and tiling configuration;
2. report wake duration and first-request latency separately from warm
   steady-state latency;
3. verify the host-memory peak leaves the profile's documented minimum host
   memory usable;
4. choose a timeout that avoids repeated sleep/wake transitions during the
   profile's declared interactive workload.

The mechanism PR may contain an explicit opt-in qualification candidate while
consumer-GPU evidence is collected separately. It must not describe that
candidate as a qualified consumer default. Other profiles stay disabled until
their own compatibility and memory costs are measured. A value such as `300`
seconds may be used in the qualification run, but it is not copied to unrelated
profiles by default.

The qualification report defines the profile's interactive workload and
compares candidate timeouts against its within-session idle gaps. The selected
value must not trigger sleep during that scripted session and must still meet
the profile's stated maximum time to release idle GPU memory. This is a
profile-specific policy decision, not a benchmark-derived global constant.

### Startup validation

When `timeout_s` is configured, startup fails with an actionable error unless:

- every resolved stage has `enable_sleep_mode: true`;
- the engine client implements `sleep`, `wake_up`, and `resume_generation`;
- only one API worker owns the engine lifecycle controller.

Failing startup is preferable to running a profile that advertises memory
reclamation but cannot wake reliably. The first version does not silently
disable idle sleep.

Level-1 sleep requires enough host memory to retain offloaded weights. Each
enabled consumer profile must document its measured host-memory peak; the
runtime does not attempt to predict host OOM from checkpoint size.

### Why deploy YAML only

The first version does not add a CLI flag. Consumer profiles are the policy
owner, deploy YAML is already the user-edited composition surface, and a CLI
override would otherwise be propagated as if it were a per-stage engine
option. A CLI override can be added later if there is a concrete operational
need and a defined precedence rule.

## Architecture

```text
HTTP / WebSocket
       |
       v
IdleActivityMiddleware
  - classify serving activity
  - hold activity until stream/session closes
  - wait for transparent wake
       |
       v
IdleSleepController
  - activity counter and idle deadline
  - one serialized transition task
  - state, errors, logs, shutdown
       |
       v
AsyncOmni
  sleep(level=1, mode="wait")
  wake_up()
  resume_generation()
       |
       v
AR EngineCore and diffusion workers
```

The controller runs in the API event loop but belongs to the engine lifecycle
boundary. The ASGI middleware only translates transport activity into
`begin_activity()` and `end_activity()` calls. It does not call stage workers
or implement cross-stage policy.

### Components

1. `IdleSleepRuntimeConfig`
   - Parsed from the top-level deploy YAML.
   - Exposed by `OmniEngineBase` and `AsyncOmniBase`, following the existing
     `duplex_session_config` pattern.

2. `IdleSleepController`
   - Owns the state machine, idle timer, transition lock, active-transport
     count, auto-sleep ownership, error state, and shutdown.
   - Calls the engine lifecycle methods.
   - Uses monotonic time.
   - Replaces `app.state.sleeping_stages` as the source of truth while exposing
     an equivalent view to the existing manual routes.

3. `IdleActivityMiddleware`
   - Wraps the assembled ASGI app alongside the timestamp middleware.
   - Tracks recognized inference HTTP routes until the ASGI call returns.
     This includes streaming responses.
   - Tracks recognized inference WebSocket routes for the full connection
     lifetime.
   - Waits for controller wake before invoking an inference route.

4. Engine idle snapshot
   - A read-only `AsyncOmni` method reports whether engine-owned work is idle.
   - The minimum snapshot includes `_admitting == 0`, no `request_states`, no
     open realtime or duplex session, no sleeping tags, and no unrelated
     manual pause.
   - Returns only an idle boolean; it does not expose request IDs.

## Activity definition

### Activity that holds the server awake

- Recognized inference HTTP routes from request entry until the response body
  completes or the request is cancelled.
- Batch and streaming inference routes.
- Realtime and duplex WebSocket routes from connection entry until disconnect.
- Engine requests still present after an HTTP response has completed.
- Engine-owned realtime or duplex sessions that outlive one HTTP request.
- Startup warmup requests.

The controller counts transport activity before starting or waiting for wake.
This prevents a new idle timer from being armed while wake is in progress.
The controller starts and arms its first deadline only after app state and the
engine client are ready, immediately before the server begins accepting
requests. Model initialization is therefore outside the idle interval, while a
ready server that receives no user request can still release GPU memory.

### Requests that do not reset the idle deadline

- `/health`
- `/metrics`
- `/v1/models`
- OpenAPI and documentation routes
- CORS `OPTIONS`
- profiler routes
- `/v1/omni/sleep` and `/v1/omni/wakeup`
- video job listing, status, deletion, and content retrieval

Route classification is explicit and covered by a route-census test. A newly
added inference route must be classified before the test passes. Unknown or
404 paths do not wake the model.

An invalid request sent to a recognized inference route can wake the model
before request-body validation. This is accepted for the local-server scope:
the middleware must restore engine admission before the route can validate or
execute safely.

## State machine

### States

| State | Meaning |
| --- | --- |
| `DISABLED` | No timeout is configured. Middleware is a pass-through. |
| `AWAKE` | Engine is available. An idle timer may be armed. |
| `SLEEPING` | One level-1 sleep transition owns the lifecycle. |
| `ASLEEP` | All auto-managed stages completed level-1 sleep. |
| `WAKING` | One wake-and-resume transition owns the lifecycle. |
| `SUSPENDED` | A manual operation left lifecycle state outside the automatic policy's ownership. |
| `ERROR` | A transition failed and stage state may be partial or unknown. |
| `CLOSED` | Server shutdown started; no new transition may begin. |

The timer is not a separate state. It carries a generation number so a stale
timer cannot initiate sleep after newer activity.

### Transitions

| From | Event | To | Action |
| --- | --- | --- | --- |
| `AWAKE` | last activity ends | `AWAKE` | Arm a new idle deadline. |
| `AWAKE` | activity starts | `AWAKE` | Increment activity and invalidate the deadline. |
| `AWAKE` | deadline expires and all idle checks pass | `SLEEPING` | Start one controller-owned sleep task. |
| `SLEEPING` | sleep succeeds | `ASLEEP` | Record reclaimed memory and transition duration. |
| `SLEEPING` | activity starts | `SLEEPING` | Register activity and wait for sleep, then wake. |
| `ASLEEP` | activity starts | `WAKING` | Start one shared wake task. |
| `WAKING` | another activity starts | `WAKING` | Join the shared wake task. |
| `WAKING` | wake and resume succeed | `AWAKE` | Admit all waiting activities. |
| `AWAKE`, `ASLEEP` | manual operation starts | unchanged | Invalidate the timer and serialize with any transition. |
| any managed state | manual operation leaves stages sleeping or admission paused | `SUSPENDED` | Stop automatic timers until state is reconciled. |
| `SUSPENDED` | manual operation leaves every stage awake and admission open | `AWAKE` | Re-arm a full idle interval. |
| any transition | transition fails | `ERROR` | Reject inference with a stable 503 error. |
| any non-closed state | shutdown | `CLOSED` | Cancel timer, reject new activity, and await/cancel controller tasks safely. |

### Locking rules

- A short state lock protects state, activity count, timer generation, and the
  current shared transition task.
- Engine RPCs are never awaited while holding the state lock.
- A separate transition lock serializes auto and manual lifecycle operations.
- Wake and sleep tasks are controller-owned tasks. Request cancellation uses
  `asyncio.shield()` so one disconnected client cannot cancel the transition
  required by other waiters.
- Exactly one caller changes `SLEEPING` or `WAKING` to its terminal state.

## Idle detection

Transport inactivity is necessary but not sufficient.

When the idle deadline expires, the controller rechecks:

1. active transport count is zero;
2. the timer generation is still current;
3. state is `AWAKE`;
4. the engine idle snapshot reports no admitted or active engine request;
5. the engine is not manually paused or already sleeping.

If the engine remains busy after the transport closes, the controller checks
again at a short internal interval. Once engine idleness is first observed, a
fresh full idle interval begins. This is conservative: an imprecise observation
can delay memory release, but cannot sleep active work.

The sleep call uses:

```text
stage_ids = all stages
level = 1
mode = "wait"
```

`mode="wait"` is a final engine-side guard. Automatic idle sleep must not abort
work even if the API-side snapshot races with an internal submission.

## Transparent wake

The first inference request that observes `ASLEEP` creates the shared wake
task. Concurrent inference requests wait on that task. The transition is:

1. `AsyncOmni.wake_up(stage_ids=all)`
2. `AsyncOmni.resume_generation(stage_ids=all)`
3. mark state `AWAKE`
4. release waiting requests into the application

This sequence is required for mixed or AR pipelines. Diffusion-only pipelines
also tolerate the explicit `resume_generation()` call.

Wake latency is charged to the first request and measured separately. It is not
included in steady-state latency reports.

## Manual control interaction

Manual `/v1/omni/sleep` and `/v1/omni/wakeup` retain their existing request and
response contracts. They acquire the same transition lock so they cannot run
concurrently with automatic sleep or wake.

At the start of a manual operation, the controller invalidates the idle timer
and waits for an in-progress automatic transition. After the existing manual
route behavior completes, it queries sleeping stages and frontend admission:

- if every managed stage is awake and admission is open, the controller
  returns to `AWAKE` and starts a fresh full idle interval;
- if any stage remains asleep or admission remains paused, it enters
  `SUSPENDED` and does not transparently wake inference requests;
- a later manual operation that restores every stage and admission returns the
  controller to `AWAKE`;
- a failed manual operation enters `SUSPENDED`, not `ERROR`, because the
  automatic controller did not own the resulting partial state.

This preserves trainer-oriented manual semantics: manual sleep may be
stage-selective and manual wake normally does not imply `resume_generation()`.
There is one ownership-aware exception: a manual wake that targets the complete
auto-slept stage set delegates to the controller's `wake_up()` then
`resume_generation()` sequence and returns the existing wake response shape.
A partial manual wake of an auto-owned all-stage sleep is rejected with
`409 Conflict`; otherwise the controller could no longer provide an atomic
transparent wake.

The controller updates its sleeping-stage view after every automatic or manual
attempt, including partial failure, so the existing route never incorrectly
returns `SKIPPED`. Requests received in `SUSPENDED` get a stable `409 Conflict`
explaining that manual lifecycle state must be resolved; they are not allowed
to wait indefinitely behind the engine admission gate.

## Failure behavior

An exception or unsuccessful acknowledgment during automatic sleep, wake, or
resume moves the controller to `ERROR`. The engine may be partially
transitioned, so the controller does not guess that it is warm.

While in `ERROR`:

- inference HTTP requests receive `503 Service Unavailable`;
- WebSocket inference handshakes close with a service-unavailable error;
- `/health` reports unhealthy with the idle-sleep transition reason;
- passive observability and manual control endpoints remain available;
- no automatic retry loop runs.

The log includes the operation, prior state, elapsed time, exception type, and
affected stage IDs. It does not include prompts or request payloads.

The reliable recovery is server restart. Manual controls remain available for
diagnosis or best-effort recovery, but they do not re-enable automation in the
first version. Automatic retry is deferred until stage-level transition
idempotency and partial-failure semantics are stronger.

## Observability

The first version does not add another Prometheus metric family. Reclaimed
bytes continue to come from existing sleep acknowledgments and engine memory
metrics, avoiding a second source of truth.

Each automatic transition logs its operation, target stages, configured idle
interval for sleep, duration on success, and exception details on failure.
Wake latency is the duration from the automatic wake start log through
successful `resume_generation()`.

## Shutdown

Server shutdown calls `IdleSleepController.close()` before engine teardown.
Close performs the following:

1. set `CLOSED` under the state lock;
2. invalidate and cancel the idle timer;
3. prevent new activity from creating wake tasks;
4. await an already-running transition up to the existing server shutdown
   budget;
5. continue normal engine shutdown.

Shutdown does not wake an asleep engine solely to terminate it.

## Verification

Verification is split into mechanism coverage and hardware qualification:

1. Deterministic CPU tests cover configuration inheritance and validation,
   timer invalidation, HTTP and WebSocket activity leases, concurrent requests
   sharing one wake, manual lifecycle ownership, transition failures, and
   shutdown during an in-flight transition.
2. Route coverage verifies that only inference traffic resets the deadline.
   Streaming responses and background generation retain an activity lease,
   while health, metrics, model-list, lifecycle, and result-retrieval traffic
   remain passive.
3. An opt-in profile must start without an overlay, remain awake for its full
   configured interval, complete one all-stage level-1 sleep, and transparently
   wake for a valid request. The default profile must remain disabled.
4. Repeated qualification runs include at least 20 sleep/wake cycles, concurrent
   wake sharing, stream and disconnect handling, output validation, paired
   device and host-memory measurements, and sleep, wake, warm-request, and
   post-sleep latency distributions.
5. Evidence records the hardware and software environment, pinned code and
   model revisions, resolved profile, raw samples, p50/p95 summaries, and
   controller, CUDA, OOM, and stage-acknowledgment error scans.

The lifecycle mechanism can be reviewed once its CPU gates and L20 mechanism
qualification pass. Full A4 completion still requires one consumer profile to
supply the CUDA qualification report and an evidence-backed timeout. A1/A2
owners may contribute additional 8/12/16/24 GB rows, but those rows do not
replace the A4 lifecycle qualification.

## Acceptance criteria

1. Existing profiles with no `idle_sleep` block have identical lifecycle
   behavior and no controller background task.
2. One opted-in consumer profile performs exactly one all-stage level-1 sleep
   after a complete idle interval and reports successful acknowledgments.
3. No sleep begins while a classified transport or engine request is active.
4. The first inference request after auto sleep succeeds without a manual
   lifecycle request; concurrent arrivals share one wake.
5. Automatic and manual lifecycle operations never overlap, and unresolved
   manual state is surfaced as `SUSPENDED`.
6. Sleep, wake, and resume failures produce explicit state and bounded errors;
   shutdown leaves no controller task running.
7. The qualification report contains pinned environment data, at least 20 raw
   wake samples, p50/p95 wake latency, warm and first-request latency, device
   and host memory, output validation, and at least five consecutive
   successful cycles.
8. The opted-in profile's timeout and host-memory requirement are justified by
   that report; no support claim is made for unmeasured hardware tiers.

## Implementation map

| Area | Planned location | Responsibility |
| --- | --- | --- |
| Config schema | `vllm_omni/config/stage_config.py` | Define and parse the top-level idle-sleep config. |
| Config propagation | `vllm_omni/engine/omni_engine_base.py`, `vllm_omni/entrypoints/async_omni_base.py` | Expose the resolved policy. |
| Lifecycle and transport | `vllm_omni/entrypoints/openai/idle_sleep.py` | Own state, timers, transition serialization, route classification, and activity leases. |
| Server composition | `vllm_omni/entrypoints/openai/api_server.py` | Construct and close the controller, wrap the app, and coordinate manual routes. |
| Engine idle snapshot | `vllm_omni/entrypoints/async_omni.py` | Report whether admission and engine requests are idle. |
| CPU tests | `tests/entrypoints/openai_api/`, `tests/engine/` | Cover config, state transitions, middleware, routes, and shutdown. |
| CUDA test | `tests/e2e/features/` | Validate repeated sleep/wake cycles and memory behavior. |
| User docs | `docs/features/sleep_mode.md` and one consumer recipe | Explain configuration, costs, and measured wake behavior. |

## Delivery plan

1. Add typed config, propagation, and startup validation.
2. Add the controller, engine-idle snapshot, lifecycle logs, and deterministic
   unit tests.
3. Add ASGI activity tracking and integrate controller shutdown.
4. Serialize manual control routes with controller transitions.
5. Add an explicit opt-in profile candidate and validate its composition and
   lifecycle on available CUDA hardware.
6. Qualify that candidate on consumer hardware and confirm or revise its
   timeout from the recorded wake and usage measurements.
7. Add the consumer qualification report.

Each step keeps the feature disabled unless the deploy profile opts in.

## Alternatives considered

### Call `/v1/omni/sleep` from an external timer

Rejected because it cannot observe active streams, background engine work, or
transition races, and it cannot perform the AR wake-plus-resume sequence before
the next request.

### Put the timer only inside `AsyncOmni`

Rejected because the engine does not own HTTP response streams or WebSocket
session lifetime. Engine state remains a required secondary idle signal.

### Use only ASGI connection count

Rejected because asynchronous jobs may outlive their originating response.

### Use level 2 for maximum memory reclamation

Rejected because current level-2 wake is not implemented.

### Poll GPU utilization

Rejected because low utilization is not equivalent to no active request and
would make policy timing device-dependent and nondeterministic.

### Auto-sleep each stage independently

Deferred because requests cross stage boundaries and partial stage sleep
introduces routing and rollback states not required by A4.

## Review decisions

The implementation should not begin until reviewers agree on these proposed
decisions. No maintainer reply in #10008 has confirmed them yet:

1. Transparent wake is the default for auto-owned sleep.
2. Automatic sleep is fixed to level 1 and all stages.
3. Configuration is top-level deploy YAML only in the first version.
4. The first profile's timeout is selected from qualification evidence; 300
   seconds is only the initial measurement candidate.
5. Manual lifecycle operations suspend automation only while sleeping stages
   or paused admission remain unresolved.
6. The first version supports one API worker only.
