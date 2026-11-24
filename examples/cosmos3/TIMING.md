# LIBERO timing aligned with the infer_hw PI benchmark

The complete Chinese protocol, including exact start/end events, throughput
numerators, warmup rules and the PI field mapping, is maintained in the infer_hw
workspace at `docs/LIBERO_TIMING_PROTOCOL.md`.

New policy responses advertise `timing_schema=pi_libero_v1` through `/info`
and `benchmark.timing_schema`. Client reports use `schema_version=3`.

| Metric | Boundary |
| --- | --- |
| `benchmark.policy_wall_ms` / recorded `policy_wall_ms` | Complete frontend policy call: input preparation, lock waiting, backend job completion and cleanup, streaming callbacks, action denormalization, and optional diagnostic dumps. Excludes HTTP request reception and response serialization. Matches PI benchmark `policy_wall_ms` at the call boundary, although the vLLM call includes its remote transport. |
| `policy_timing.infer_ms` | Same complete policy wall time. Previously this field measured only the backend roundtrip. |
| `policy_timing.backend_ms` / recorded `backend_roundtrip_ms` | Backend job roundtrip after lock acquisition. Kept separate from full policy wall time. |
| Recorded `client_ms` | HTTP submission to completed response parsing, excluding client image preparation. Corresponds to PI benchmark `client_ms`. |
| Recorded `ttfa_ms` | Same client start, ending at the first ready action received. Nonstream requests use response completion. Corresponds to PI benchmark `ttfa_ms`. |
| `worker_inference_ms` | Worker job generation and output persistence wall time. Remains separate; it is not CUDA kernel time or a substitute for policy wall time. |

The complete policy clock starts afresh inside each call. It does not include
environment execution or idle time between requests. Streaming requests retain
individual action-ready timestamps in addition to the full-call time.

Both LIBERO client entry points record these fields:

- `examples/cosmos3/closed_loop_eval.py` in this repository.
- `cosmos_framework/simulation/libero/closed_loop_eval.py` in the infer_hw
  Cosmos Framework checkout.

Existing image-inclusive fields such as `client_total_ms` and
`client_time_to_first_action_ms` retain their boundaries. Use `client_ms` and
`ttfa_ms` for the PI benchmark comparison; use the image-inclusive fields to
measure observation preparation plus request latency.

Before evaluation the client sends two separate synthetic requests through
`POST /warmup`, using the configured synchronous/streaming and batch path.
These calls are absent from evaluation request records. Each vLLM job uses its
explicit fixed seed; synthetic images use a local NumPy RNG, so warmup does not
advance evaluation RNG state. `--synthetic_warmup_requests` overrides the count.
Servers that do not advertise warmup support default to zero and record that
fact; an explicit nonzero count requires support.

The default first two real requests remain separately excluded from steady-state
latency and throughput. Stabilization dummy actions are separate from both
forms of request warmup; their time stays in evaluation wall time.

## Matched evaluation scope and debug output

`evaluation_scope=libero_task_loop_v1` starts after client setup, server checks,
task-suite preparation and synthetic warmup. It stops after the last environment
is closed, before final report serialization. Environment creation, resets,
simulation, real requests (including the two timing-warmup requests), and
per-task cleanup are included. The PI evaluator returns this same scope and its
benchmark wrappers use that value instead of their former client-start clock.
Historical summaries retain their original elapsed values.

By default clients reject server dumps/recording/video and client GIF/video/
comparison output. PI servers advertise recording status; vLLM advertises dump
status in `/info`. For intentional debug evaluations use `--allow_debug_output`
(PI: `--allow-debug-output`); reports mark `debug_output_enabled=true`.
Synthetic warmup defaults to zero for such debug runs. Disable the debug outputs
for performance comparisons rather than silently removing their I/O time.

The PI streaming WebSocket `server_timing.infer_ms` also starts after receiving
and decoding the observation, so client startup and gaps spent executing
environment actions are excluded on every request.

## Action counts and throughput

The original `active_actions_per_s` and `sustained_actions_per_s` fields retain
the generated-action numerator. New fields make executable capacity explicit:

- `executable_actions`: sum over returned chunks of
  `min(valid returned actions, client execution window)`. A window of zero means
  the complete valid returned chunk. Each chunk in a batch is capped separately.
- `active_executable_actions_per_s`: that capacity divided by the union of
  measured request intervals.
- `sustained_executable_actions_per_s`: that capacity divided by the first-to-last
  measured request span.

Actual execution remains `evaluation_wall.executed_actions_per_s`; it excludes
dummy actions from the numerator and includes the complete evaluation time.
Capacity may exceed actual execution when an episode succeeds before consuming
its final chunk. Early-stop responses count only the valid returned prefix.

Old logs are not rewritten. A legacy server without the new measurement records
`policy_wall_ms=null`; backend/worker time is never silently substituted.
Reprocessing records without executable-capacity counts leaves the new
executable throughput fields null. Do not combine old backend-only
`policy_infer_ms` samples with `pi_libero_v1` samples as if they shared a boundary.
The infer_hw throughput reporter prioritizes executable active/sustained rates
and `policy_wall_ms`/`client_ms`/`ttfa_ms`. It reconstructs legacy executable
capacity only when the execution window and per-request counts are available;
otherwise it prints `n/a`. Legacy closed-loop scopes are explicitly labelled
`legacy_unaligned`.

## Validation

CPU tests exercise the real frontend and HTTP evaluator against a deterministic
job server, including streaming, batch requests, early stop, and report output.
Clock-controlled tests verify that complete policy time includes input
preparation, queue waiting, backend cleanup and dumps, while a gap between calls
does not affect the measurement. These tests do not measure model performance.
