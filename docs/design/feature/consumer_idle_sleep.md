# Automatic Idle Sleep

Status: experimental opt-in serving feature (validated on single-L20 multi-stage Omni and pure Diffusion).
Tracking: [#10008, A4](https://github.com/vllm-project/vllm-omni/issues/10008).

## Goal and ownership

Release GPU memory while a local serving pipeline is idle and restore it
before its next generation. The policy is opt-in via
`--idle-sleep-timeout SECONDS`; the default is disabled. Timeout is finite and
positive, and stays at the serving boundary rather than entering engine kwargs
or deploy YAML.

`AsyncOmni` owns the policy because it already owns request admission,
`request_states`, pending submissions and lifecycle calls. The implementation
does not classify HTTP routes or track transport lifetimes. A background video
job reaches the same `generate` entrypoint as a synchronous request.

Six fields hold configuration, the event loop, the timer, the shared
transition, a fatal transition error, and the terminal closed flag. Existing
`_pause_cond`, `_paused`, `_admitting`, `_pending_submits` and sleeping tags
remain the sources of admission and device state.

## Request and idle behavior

With automatic sleep enabled, `generate` uses the existing admission condition
to check availability, cancel the idle timer, and obtain or start shared
recovery. It waits for a transition outside the lock, then rechecks admission.
Without automatic sleep, the original manual pause and sleeping-tag gate still
applies. Stage-by-stage `/v1/run` calls (`run_entry_stage` /
`run_downstream_stage`) are rejected with `OmniClientError(409)` while
automatic idle sleep is enabled, because sleeping between independent stage
calls can release handles required by downstream stages.

There is no suspending await between final admission and request registration.
This lets `request_states` cover accepted work without a second request count
or a hold/release protocol. Async preprocessing added there in the future must
preserve this invariant. Failure or cancellation before registration rearms
the timer in request cleanup without aborting a nonexistent engine request.

The timer is armed after warmup, request cleanup or successful
recovery. An existing deadline is not extended by another cleanup. New work
invalidates the timer by handle identity. Expiry publishes one shared task;
that task checks handle identity and idle conditions under `_pause_cond`
before closing admission or making RPCs. Pending submissions delay sleep for
another full timeout, avoiding changes to individual submission cleanup paths.

During sleep, incoming requests wait on the shared sleep task. Once sleep
finishes, the first remaining waiter starts shared wake and resume. Requests
use `asyncio.shield` so cancellation does not propagate to the lifecycle
task. If every waiter disappears during sleep, the model stays asleep. If
every waiter disappears during wake, successful recovery rearms the timer.

For metrics, `req_start_ts` stays at `submit_ts` and caller-supplied
`arrival_time` is left unchanged, so warm requests see zero shift in TTFT/E2E
preprocessing or queueing semantics when the flag is enabled. Time spent
waiting at admission for an active sleep/wake transition is reported separately
via the opt-in Prometheus histogram `vllm_omni:idle_wake_wait_s` (emitted only
when a request actually waits on a transition). Idle and lifecycle durations
use monotonic time.

## Device transitions and errors

Automatic sleep always uses `sleep(level=1, mode="wait")`. Because automatic
sleep only triggers when `request_states`, `_admitting`, and `_pending_submits`
are empty and frontend admission is closed, diffusion stages rely on
`DiffusionEngine`'s serialized RPC queue and `handle_sleep_task`'s device
synchronization barrier directly without an extra `pause_generation` round-trip.
Recovery runs `wake_up()` and, when admission remains held (AR or mixed
pipelines), `resume_generation()`.

The five public lifecycle methods reject outside callers with
`OmniClientError(409)` while automatic ownership is enabled. Other control RPCs
(`collective_rpc`, `add/remove/list/pin_lora`, `reset_*_cache`,
`start/stop_profile`) are gated through `_guard_control_rpc()`: they return
`OmniClientError(409)` while the engine is asleep, paused, or transitioning,
and cancel the idle timer while pinning admission when awake so sleep cannot
race an in-flight control RPC. The shared transition calls lifecycle RPCs
inside `asyncio.timeout` with a finite control timeout (default 120 seconds,
configurable via `VLLM_OMNI_IDLE_SLEEP_CONTROL_TIMEOUT_S`, independent of the
idle interval).

A transition failure or timeout is recorded before waiters resume.
`errored=True`, `is_running=False` and failing `check_health()` expose the
fault even when the orchestrator and output task remain alive. New generation
raises `OmniEngineDeadError`; no automatic retry or subsequent resume is
attempted. The upstream engine-death watchdog controls process exit and
respects `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH` / `should_keep_api_server_alive`.
An asleep but healthy engine is not an error.

Ordinary request cancellation and validation errors do not poison the policy.
Closing cancellation is also nonfatal. Async timeouts bound local waiting;
they do not guarantee that a synchronous executor or remote device stopped.

## Shutdown

`AsyncOmni.shutdown()` owns policy teardown directly, without modifying
`PureDiffusionLauncherAdapter` or adding a separate shutdown bridge in
`api_server.py`. Both the upstream launcher's signal handler (direct on AR/mixed
or forwarded through `PureDiffusionLauncherAdapter.__getattr__` on pure
diffusion) and `build_async_omni`'s exit handler call `AsyncOmni.shutdown()`.

When idle sleep is enabled, `shutdown()` synchronously sets `_idle_closed = True`
before delegating to `super().shutdown(timeout)`, cancels the timer and shared
transition on the owning event loop (directly when on-loop, or via
`call_soon_threadsafe` when called from another thread), and immediately tears
down the engine after the base frontend shutdown has closed admission,
queued fatal errors and released request waiters on the serving loop. The
admission gate also binds the serving loop before waiting for sleep or wake,
so upstream shutdown can find and release those waiters. Engine teardown
unblocks blocked RPCs without waiting for the idle transition timeout.
Late lifecycle/RPC calls check `_idle_closed` before issuing another
operation, transition cancellation during shutdown is nonfatal, and repeated
`shutdown()` calls are idempotent.

## Supported boundary

- One frontend, complete local pipeline, every stage sleep-enabled.
- AR and diffusion stages; no level-2 automatic sleep.
- No duplex: `DuplexOmni` subclasses `AsyncOmni`, so it must be explicitly
  rejected. Session close acknowledgements do not prove GPU resources are
  released; session-manager changes are outside this work.
- No PD, weight transfer, `/v1/run` stage-by-stage execution
  (`VLLM_OMNI_ENABLE_RUN_ENDPOINT=1`), or dynamic replica coordination.
- Ordinary realtime connections may remain idle across sleep; each generation
  wakes independently.
- No cross-request GPU resource holds or public hold API.
- No qualified consumer deploy profile or consumer-GPU support claim.

## Verification

L1 coverage belongs in `tests/entrypoints/test_async_omni_idle_sleep.py`,
`tests/entrypoints/openai_api/test_idle_sleep.py`, and
`tests/entrypoints/openai_api/test_omni_sleep_wakeup.py`, alongside the
existing pause/sleep routing regressions. Existing L1 CPU Buildkite discovery
collects these paths and verifies:

1. Configuration rejection (including `VLLM_OMNI_ENABLE_RUN_ENDPOINT=1` and
   invalid `VLLM_OMNI_IDLE_SLEEP_CONTROL_TIMEOUT_S`), default-off behavior,
   lifecycle and control-RPC (`collective_rpc`, LoRA, cache reset, profiling)
   gating, `/v1/run` stage rejection, `submit_ts` preservation, and separate
   `vllm_omni:idle_wake_wait_s` observation.
2. Shared recovery, stale timers, admission/expiry races, cancellation of
   one or all waiters, pre-registration failure, and delayed submission cleanup.
3. AR/diffusion sleep/wake/resume ordering, transition failures and timeouts,
   bounded/repeated shutdown, and rejection of late RPCs.
4. Installed-vLLM engine-death watchdog behavior and isolated `SIGTERM` /
   `SIGINT` launcher tests across zero/positive shutdown budgets, AR and
   pure-diffusion adapters, and an uncooperative wake RPC.

GPU E2E CI coverage lives in
`tests/entrypoints/test_omni_sleep_mode.py::TestOmniDiffusionSleepMode::test_diffusion_auto_idle_sleep_wake_e2e`
(`core_model`, L4/MI325): verifies automatic sleep, measured GPU VRAM drop,
409 control-RPC rejection while asleep, automatic wake on `generate()`, and
pixel-identical output. Single-GPU L20 (48 GB) validation additionally covers
`Qwen/Qwen2.5-Omni-7B` and `Qwen/Qwen-Image` FP8.
