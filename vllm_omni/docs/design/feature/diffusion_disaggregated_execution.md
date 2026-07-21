# Disaggregated Diffusion Execution (RFC #4590)

## Motivation

A diffusion request classically runs as one monolithic `forward()` on one worker:
validate → encode → denoise loop → decode → output. That couples three phases
with very different resource profiles onto one process and one device group. The
encode phase is encoder-bound (text/image/VAE encoders), the denoise phase is
DiT/attention-bound and owns the session KV, and the decode phase is VAE-decode
bound. Co-locating them wastes capacity: the DiT sits idle during encode/decode,
and encoders/VAE weights occupy memory on the DiT device throughout the rollout.

RFC #4590 lets a diffusion model run its encode, denoise, and decode phases as
**three independent native diffusion stages**, each with its own process,
devices, parallel config, component set, scheduling, and failure handling — while
preserving every existing execution mode unchanged.

## Execution modes (all preserved)

1. **Monolithic** — one worker owns `validate → encode → denoise → decode`
   (`model_stage` = `diffusion` / unset). Unchanged.
2. **Single-worker step execution** — `prepare_encode` once, runner-managed
   `denoise_step` loop, `post_decode` (the `SupportsStepExecution` contract).
   Unchanged.
3. **Native three-stage disaggregation** — `encode → denoise → decode`, each a
   normal diffusion stage on the existing stage engine/runtime. New.

The three disaggregated stages reuse the native diffusion stage runtime — there
is no second execution stack for encoders or VAE submodules.

## Architecture

```
Raw Request
    |
    v
+-----------------------+
| Encode DiffusionStage |
| tokenizer             |
| text/image/VAE encode |
| latent preparation    |
+-----------+-----------+
            |
            | DiffusionStagePayload  (encode -> denoise)
            v
+-----------------------+
| Denoise DiffusionStage|
| DiT / CausalWan       |
| scheduler loop        |
| AR-Diffusion KV       |
+-----------+-----------+
            |
            | DiffusionStagePayload  (denoise -> decode)
            v
+-----------------------+
| Decode DiffusionStage |
| VAE decode            |
| action/video output   |
| postprocess           |
+-----------+-----------+
            |
            v
       Final Output
```

Every box is a `StageExecutionType.DIFFUSION` stage. The role is carried by
`StagePipelineConfig.model_stage` ∈ {`encode`, `denoise`, `decode`} (or
`diffusion` for the monolithic fallback). The **execution type** answers *which
engine family runs this stage*; the **model stage** answers *which portion of the
diffusion model this worker owns*. No `StageType.ENCODE/DENOISE/DECODE` and no
per-phase execution types were added.

### Runner vs pipeline responsibilities

| Concern | Owner |
|---|---|
| stage dispatch, request-state cache, scheduling, batching, scatter/gather, cleanup, profiling aggregation, envelope validation, local/remote integration | **generic runner** (`DiffusionModelRunner`) |
| input validation, text/image/video encoding, latent & timestep prep, model-input construction, denoise math, scheduler advancement, decode, postprocess, state↔payload conversion | **model pipeline** |

The generic runner never interprets model-private fields (DreamZero image
conditioning, prompt-embedding layout, VAE stream state, action tensors,
CFG-private metadata, cross-attention cache, model scheduler internals). There is
**no** `if model_name == "dreamzero": …` in the runner. The stage-role dispatch is
a pure table (`stage_roles.resolve_execution_path`), mirrored in
`DiffusionModelRunner.execute_model`.

### State vs payload

`DiffusionRequestState` is **mutable runner-local state** and never crosses a
process boundary. Inter-stage data travels as a typed, versioned
`DiffusionStagePayload` (`vllm_omni/diffusion/stage_payload.py`):

```python
@dataclass(frozen=True)
class DiffusionStagePayload:
    schema_version: int
    request_id: str
    source_stage: str
    target_stage: str
    payload_type: str
    tensors: dict[str, torch.Tensor]   # the only place tensors live
    metadata: dict[str, Any]           # small, transportable, model-private
```

Invariants (enforced by `validate()`):

* request identity, source/target stages, and `payload_type` are explicit;
* the schema is versioned; consumers reject unknown versions;
* tensors are separate from metadata; the envelope never grows a per-model field;
* modules, generators, CUDA streams, schedulers, and process-local state objects
  are rejected early (`NonTransportableValueError`);
* model-specific keys are interpreted only by the owning pipeline;
* tensors can later be backed by connector handles without changing the envelope.

Tensors are moved to host memory, detached, made contiguous, and cloned at the
**export** boundary (`sanitize_transport_tensor`); device movement happens once at
the **restore** boundary inside the pipeline. `.cpu()`/`.to(device)` are not
scattered through model code.

### Pipeline capability protocol

`vllm_omni/diffusion/models/interface.py` adds:

* `SupportsDisaggregatedDiffusionExecution` (runtime-checkable) —
  `export_stage_payload`, `import_stage_payload`,
  `required_components_for_stage`, plus the `supports_disaggregated_execution`
  flag. `supports_disaggregated_execution(pipeline)` is the capability helper,
  paralleling the existing `supports_step_execution`.
* `SupportsDiffusionAtoms` (additive) — finer atoms `check_inputs`,
  `encode_conditions`, `prepare_latents_and_timesteps`, `decode_latents`,
  `postprocess_outputs`. The runner prefers these via `run_encode_atoms` /
  `run_decode_atoms` and falls back to the four-method step contract
  (`prepare_encode` / `post_decode`) when absent — so existing step pipelines are
  untouched.

A stage that requests `encode`/`denoise`/`decode` on a pipeline lacking the
disaggregated capability fails at **startup** (`load_model`), not mid-forward.

## Runner stage dispatch

`DiffusionModelRunner.execute_model(req)` dispatches on `self.model_stage`:

```python
role = self.model_stage
path = resolve_execution_path(role)          # pure, shared table
if path == EXECUTION_PATH_ENCODE:   return self.execute_encode_stage(req)
if path == EXECUTION_PATH_DENOISE:  return self.execute_denoise_stage(req)
if path == EXECUTION_PATH_DECODE:   return self.execute_decode_stage(req)
if path == EXECUTION_PATH_MODEL_DEFINED: return self.execute_model_defined_stage(role, req)
return self._execute_monolithic(req, kv_prefetch_jobs=kv_prefetch_jobs)   # unchanged
```

* **Encode stage** — creates runner-local state from the raw request, initializes
  the generator, runs the encode atoms, exports an `encode → denoise` payload,
  returns an *intermediate* output (payload in `custom_output`), and releases the
  state. It never runs the DiT, advances the scheduler, or decodes.
* **Denoise stage** — restores state from the payload via `import_stage_payload`
  (which attaches live session/KV through the normal engine mechanism, *never*
  from the payload), caches it, runs the model's `run_denoise`, exports a
  `denoise → decode` payload. It never re-runs encode for a payload-origin
  request — the origin is explicit (payload vs raw), not inferred from a
  new-request id set.
* **Decode stage** — restores decode state, runs the `decode` (no-op for
  DreamZero) + `postprocess` atoms, and returns the user-visible
  `DiffusionOutput` (denormalized actions + latent video). It never instantiates
  or runs the DiT or a scheduler, and — for DreamZero — never invokes the VAE
  decoder (see *Decode ownership matches decode behavior* below). It also
  cross-checks the incoming payload's `request_id` against the request (RFC §B.1).

The historical "new request ⇒ run `prepare_encode`" coupling (`state.request_id in
new_request_ids` in the stepwise path) is *not* used by the disaggregated denoise
path; state origin is carried explicitly by which stage method runs.

## Lifecycle

```
encode:   raw request -> state -> encode atoms -> export(encode->denoise) -> release state
denoise:  payload -> import (attach session KV) -> cache -> run_denoise -> export(denoise->decode) -> release
decode:   payload -> import -> decode_latents + postprocess -> DiffusionOutput -> release
```

For non-streaming requests the decode boundary is normal denoise completion. For
streaming/chunk requests, a chunk decode boundary is distinct from request
completion; emitting a decode payload does not destroy denoise session state.
Exported latents are cloned at the boundary so they never alias a buffer the
denoise worker reuses.

## Partial component loading

`initialize_model` (the single construction point) queries the pipeline class's
`required_components_for_stage(model_stage)` and logs the plan before
construction; the pipeline `__init__` reads `od_config.model_stage` and builds
only the required components (others are `None`, guarded at use sites). Weight
loading self-gates: the loader only fills parameters that exist on constructed
submodules, and the strict-load check derives its expectation from
`named_parameters()`, so a skipped component drops out of both sets automatically
— no loader change needed.

`StageComponentSpec` (in `stage_roles.py`) is the generic, model-agnostic
component vocabulary: `tokenizer`, `text_encoder`, `image_encoder`,
`vae_encoder`, `dit`, `scheduler`, `vae_decoder`, `action_modules`.

### DreamZero component ownership

| Stage | Components |
|---|---|
| encode | tokenizer, text encoder, image encoder, VAE **encoder** |
| denoise | CausalWan DiT, schedulers, action head, AR-Diffusion KV |
| decode | (none built) — action denorm + latent passthrough only |

DreamZero's VAE (`DistributedAutoencoderKLWan`) has a monolithic constructor
(encoder + decoder together), so a stage needing *either* half builds the full
module; the denoise stage needs *neither* and skips it entirely, deriving the
`vae_latents_mean/inv_std` buffers from the Wan-VAE constants instead of the
module.

**Decode ownership matches decode behavior (RFC §B.3).** DreamZero's user-visible
decode phase (`_run_decode_phase`) emits **normalized VAE latents + actions**, not
RGB video: it denormalizes the action tensor and returns `video_out` as the raw
latent. It never invokes the VAE **decoder** — RGB export is an explicit,
out-of-band debug/offline step (`decode_video_latents` / `video_export_worker`).
So `required_components_for_stage(DECODE)` declares **no** heavy component: the
decode worker must not construct or load a VAE decoder it never calls. (If decode
is ever changed to emit RGB, re-add `vae_decoder=True` and call the decoder in the
phase — declaration, construction, and use must agree.)

## DreamZero stage ownership & session/KV

**Critical invariant:** AR-Diffusion session/KV state lives **only** on the
denoise stage. The encode `DiffusionStagePayload` carries stable data
(encoded text/image conditions, `ys`/`clip_feas`, initial latents+noise, timestep
schedule params, embodiment/state metadata, session id, CFG metadata, shape
metadata). The live paged KV, the `FlowUniPCMultistepScheduler`, and the
`DreamZeroState` are **never** serialized. The denoise worker
(`ARDiffusionModelRunner`) acquires/creates the session by the transported
`session_id` through the normal engine mechanism; encode/decode stages attach no
KV and route straight to the base runner.

DreamZero's `forward()` was refactored into three phase methods
(`_run_encode_phase` / `_run_denoise_phase` / `_run_decode_phase`) that share the
existing math helpers. `forward()` now composes all three on one carrier and one
session state — so the monolithic golden path and the disaggregated path run the
**same** code, making numerical equivalence structural rather than coincidental.
The DreamZero-private inter-phase carrier (`DreamZeroStageCarrier`) lives on
`DiffusionRequestState.extra` (model-private), never on the generic state fields.

## Multi-chunk session progress (RFC §A)

DreamZero is auto-regressive: a session's chunks share a sliding AR window whose
cursor is `current_start_frame` (csf). In monolithic `forward()` one shared
`DreamZeroState` advances csf in one process, so ordering is trivially correct.
In the disaggregated path the encode and denoise workers each keep their **own**
process-local `DreamZeroState`, and only the denoise worker's DiT/KV loop advances
csf. Without coordination the encode worker's csf would sit at 0 forever (it
re-runs first-chunk conditioning every request) and the denoise worker would
overwrite its correctly-advanced csf with the stale carrier value — every chunk
silently behaves like the first.

The fix keeps the **denoise stage as the single committed-progress authority** and
adds a small, model-agnostic, torch-free control plane
(`vllm_omni/diffusion/session_progress.py`, `SessionProgressCoordinator`) plus
three routing scalars on the carrier/payload: `session_epoch`, `sequence_no`,
`attempt_id`.

* **Encode (issuer).** After running the encode phase, encode stamps
  `(session_epoch, sequence_no, attempt_id)` on the carrier from its coordinator,
  and advances its **own** csf shadow one AR chunk (mirroring the denoise
  arithmetic: `0 → 1` on the first chunk of a window, then `+= num_frame_per_block`)
  so the next request picks the correct encoding path and reset decision. Encode is
  **not** the commit authority — issuing a sequence does not mean it committed.
* **Denoise (authority).** Before touching the KV/model state, denoise
  `authorize()`s the carrier against committed session state and **rejects a
  stale / duplicate / gapped / pre-reset request** (raising `SessionProgressError`
  with no mutation). Only after a successful DiT loop + KV commit does it
  `commit()` the new csf. The pre-existing overwrite of denoise's csf from the
  carrier is now safe because the guard guarantees the carrier is the expected
  next chunk.
* **Sequencing vs windowing.** `sequence_no` is monotonic per `(session, epoch)`
  and is independent of csf, which resets to 0 at an AR window boundary **without**
  an epoch bump — so a normal window slide is not mistaken for a stale request.
  `session_epoch` bumps only on an explicit/session reset, fencing any pre-reset
  in-flight attempt.
* **Failure containment.** A denoise failure raises before `commit()`, so
  committed progress does not advance and the same sequence can be retried; the
  AR-Diffusion runner tears the KV session down on the exception and also drops
  the session-progress record, so recovery is via an explicit reset (fresh epoch)
  that re-syncs both stages.

`forward()` (monolithic) never touches the coordinator — it stays byte-identical,
preserving it as the golden reference for numerical equivalence.

**Reduced-scope limitation (documented).** The whole-request topology has no
denoise → encode back-channel, so after a denoise failure the encode stage's csf
shadow is one chunk ahead of committed progress. The next request on that session
must be an explicit reset (the runner drops the torn-down session's progress
record so a bare continuation is rejected rather than silently resuming a wiped
session). A full cross-process coordinator that pushes committed csf back to the
encode issuer is deferred as a follow-up; the invariants above (authority,
monotonic ordering, epoch fencing, commit-after-success, idempotent duplicate
rejection) hold without it.

## Generic stage transition

`vllm_omni/model_executor/stage_input_processors/diffusion.py::diffusion_stage_transition`
is the one model-agnostic adapter. It extracts the typed payload from the upstream
`DiffusionOutput.custom_output`, validates identity/transition/version (never the
model-private tensor keys), and places it in the downstream request prompt's
`extra` dict — the channel the diffusion pipeline reads. It also mirrors the
`session_id` for the denoise runner. The payload rides in channels already part of
the msgpack transport contract, so it works in both inline (single-process) and
out-of-process (ZMQ) stage clients without a new serializer.

## Config & topology

* `model_stage` now reaches the worker: added as a field on `OmniDiffusionConfig`
  and wired in `build_diffusion_config` from the stage metadata (previously it was
  dropped at `from_kwargs` and reached only the head-side client).
* `PipelineConfig.validate()` applies the disaggregated linear-topology rules
  (`stage_roles.validate_linear_diffusion_topology`) whenever any stage declares a
  disaggregated role: unique ids, valid `input_sources`, denoise-has-encode-source,
  decode-has-denoise-source, single final-output on the decode stage, single
  upstream per consumer. Unknown/custom roles are accepted (a model may declare
  support) and take the `execute_model_defined_stage` hook.
* Topologies:
  * `dreamzero` (`DREAMZERO_PIPELINE`) — the original single monolithic stage
    (default, unchanged).
  * `dreamzero_disaggregated` (`DREAMZERO_DISAGGREGATED_PIPELINE`) — the three
    `encode → denoise → decode` stages wired with the generic processor.

### Example deploy configuration

`vllm_omni/deploy/dreamzero_disaggregated.yaml` places one GPU per stage
(repository-standard example ids), keeps the AR-Diffusion engine on the denoise
stage, and uses the standard diffusion engine for encode/decode (they own no KV).
Per-stage `devices` and `parallel_config` are independent. Do not treat the
example device ids as production placement.

## Transport limitations of the first implementation

* Uses the existing host-based stage transport (msgpack over ZMQ for
  out-of-process, by-reference inline). No cross-node GPU tensor-transfer
  manager, no zero-copy GPU↔GPU transfer — the payload is structured so tensors
  can later be backed by connector handles without changing runner semantics.
* A stage-boundary clone is taken for correctness (localized, documented).
* No KV-update sub-stage; no decode/KV-update overlap; no DAG scheduling beyond the
  linear `encode → denoise → decode` path; no multi-encoder/multi-decoder pools.

## Backward compatibility

Monolithic diffusion, request-batch forward, step execution, existing stage config
parsing, output formats, profiling fields, cancellation/interruption, prompt-embed
cache, and offload behavior are all preserved. Stagewise behavior is selected
*only* by stage topology / `model_stage`; it is never globally enabled. The
original single-stage DreamZero deploy is untouched and remains the default.

## Non-goals

Cross-node GPU transfer manager, zero-copy GPU↔GPU payloads, generalized DAG
scheduling, multiple parallel encoders/decoders, fused encode/decode pools, a
KV-update sub-stage, decode/KV overlap, autoscaling/replication policy, migrating
every diffusion model, and a session-memory-manager rewrite.

## Manual validation commands

Torch-free foundation unit tests (no GPU, no vllm runtime) — includes the
multi-chunk session-progress ordering/idempotency/epoch tests (RFC §A) and the
payload transport/topology/key-drift guards:

```bash
python -m pytest tests/diffusion/disaggregated/ -q
# or just the session-progress control plane:
python -m pytest tests/diffusion/disaggregated/test_session_progress.py -q
```

Full runtime tests (on the gnr/B70 XPU node, inside the vllm-omni-xpu container):

```bash
# runner stage behavior + guards (request-id cross-check RFC §B.1,
# disaggregated batch-path rejection RFC §B.2), DreamZero atoms/payload
# round-trip incl. session-progress scalars, component ownership, config topology
pytest tests/diffusion/disaggregated/ -m needs_runtime

# numerical equivalence (monolithic forward vs encode->denoise->decode),
# single-chunk AND five-chunk same-session. Set DREAMZERO_MODEL_PATH or rely on
# the HF cache (models--GEAR-Dreams--DreamZero-DROID). Skips (never fails) when
# the checkpoint/accelerator is absent.
DREAMZERO_MODEL_PATH=/models/DreamZero-DROID \
  pytest tests/diffusion/disaggregated/test_dreamzero_disaggregated.py \
  -m needs_runtime -k numerical_equivalence -s
```

Serving the disaggregated topology:

```bash
# denoise stage requires the AR-Diffusion engine (set in the deploy YAML)
vllm-omni serve --deploy-config vllm_omni/deploy/dreamzero_disaggregated.yaml
```
