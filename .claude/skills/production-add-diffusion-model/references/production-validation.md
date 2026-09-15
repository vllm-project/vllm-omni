# Production Serving and Validation

## Contents

1. [Execution-mode decision](#execution-mode-decision)
2. [Request-level batching](#request-level-batching)
3. [Step execution and continuous batching](#step-execution-and-continuous-batching)
4. [Abort and cleanup](#abort-and-cleanup)
5. [Mixed-RPS soak](#mixed-rps-soak)
6. [Four CI tracks](#four-ci-tracks)
7. [PR evidence checklist](#pr-evidence-checklist)

## Execution-mode decision

Request batching and step-wise execution solve different problems:

| Mode | Pipeline contract | Scheduling/abort boundary |
|---|---|---|
| Serial request | `forward(DiffusionRequestBatch)` for one request | Whole request; in-denoise preemption is unavailable |
| Request batching | `supports_request_batch = True`; one forward consumes compatible independent requests | Whole batch/request |
| Single-request step | Full `SupportsStepExecution`; `max_num_seqs=1` | Between denoise steps |
| Step continuous batching | Full step contract plus heterogeneous batched-step evidence | Between denoise waves/steps |

Do not expose four stub methods as an opt-out. The current protocol is
structural; if a native pipeline is not step-capable, do not define the four
step methods. If it is capable, implement and test all four.

Treat structural support, a generic request-batch bridge, and model benefit as
three separate claims. Before recommending step execution, name a hypothesis
that request mode cannot satisfy, define a success threshold, and run a matched
request-mode control. If throughput or latency regresses, retain step execution
as functional/limited and keep request mode as the production recommendation.

Choose compatibility keys from every value that changes tensor shapes,
semantics, hooks, or scheduler behavior: task, output shape/count, frames,
steps/schedule, guidance/CFG, dtype/backend, LoRA/adapter, cache quality policy,
and model-specific conditions. Reject or schedule separately rather than
silently using the first request's settings.

## Request-level batching

A request-batched pipeline returns exactly one `DiffusionOutput` per logical
request. This is model-owned pseudocode: `validate_batch_compatibility()` and
`_generate_batch()` are placeholders to implement from the official contract,
not shared APIs to import.

```python
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.worker.request_batch import (
    DiffusionRequestBatch,
    split_diffusion_output_by_request,
)

class MyPipeline(torch.nn.Module):
    supports_request_batch = True

    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        validate_batch_compatibility(req.sampling_params_list, req.prompts)
        common = req.sampling_params_list[0]

        text_prompts: list[str] | None = [
            prompt
            if isinstance(prompt, str)
            else (prompt.get("prompt") or "")
            for prompt in req.prompts
        ]

        prompt_fields = DiffusionRequestBatch.collate_prompt_field_map(
            req.prompts,
            {"prompt_embeds": None, "prompt_mask": None},
        )
        if prompt_fields["prompt_embeds"] is not None:
            text_prompts = None
        generators = req.collate_request_generators(
            common.num_outputs_per_prompt,
            default_generator=None,
        )
        output = self._generate_batch(
            prompt=text_prompts,
            prompt_fields=prompt_fields,
            generators=generators,
            sampling=common,
        )
        return split_diffusion_output_by_request(
            DiffusionOutput(output=output),
            req,
            num_outputs_per_prompt=common.num_outputs_per_prompt,
        )
```

The prompt-field collator only handles tensor fields; extract raw text prompts
and model-specific negative/alias fields separately, as current in-tree batched
pipelines do. Set a raw prompt field to `None` when the matching precomputed
embedding is supplied. The exact aliases and generator device depend on the
model. Use current collators rather than manual `torch.cat` when they cover the
contract; they reject mixed present/missing tensors and incompatible
shape/dtype/device.

Before reading `sampling_params_list[0]`, validate every field that must be
common. Tests must cover:

- two and N requests, independent seeds/generators, one output each;
- mixed prompt lengths and packed/padded boundaries;
- multiple outputs per prompt and correct result slicing;
- compatible versus incompatible shapes, steps, guidance, LoRA, and quality;
- one request failure without cross-request output or state corruption;
- output ordering and request IDs under different completion times.

After the model advertises support, candidate serving is:

```bash
vllm serve '<model>' --omni \
  --max-num-seqs 4 \
  --request-batch-max-wait-ms 20
```

The wait window trades admission latency for batch formation. Benchmark at 0
and candidate nonzero values under the target arrival process.

## Step execution and continuous batching

### Full protocol skeleton

Use the current shared state types and make every mutable value request-local:

The helpers beginning with `_` and `validate_step_batch()` below are
model-owned placeholders; scheduler initialization is scheduler-specific.

```python
import copy
from collections.abc import Sequence
from typing import Any, ClassVar

import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.interface import SupportsStepExecution
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.utils import StepRequestState

class MyPipeline(torch.nn.Module, SupportsStepExecution):
    supports_step_execution: ClassVar[bool] = True

    def prepare_encode(
        self,
        state: StepRequestState,
        **kwargs: Any,
    ) -> StepRequestState:
        prepared = self._prepare_request(state.prompt, state.sampling)
        state.prompt_embeds = prepared.prompt_embeds
        state.latents = prepared.latents
        state.timesteps = prepared.timesteps
        state.scheduler = copy.deepcopy(self.scheduler)
        state.scheduler.set_begin_index(0)
        state.step_index = 0
        state.extra["task_state"] = prepared.task_state
        return state

    def denoise_step(
        self,
        input_batch: InputBatch,
        *,
        states: Sequence[StepRequestState] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | None:
        validate_step_batch(input_batch, states)
        return self.transformer(**build_batched_denoise_inputs(input_batch))

    def step_scheduler(
        self,
        state: StepRequestState,
        noise_pred: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        timestep = state.current_timestep
        if timestep is None:
            raise RuntimeError("step_scheduler called without a current timestep")
        state.latents = state.scheduler.step(
            noise_pred,
            timestep,
            state.latents,
            return_dict=False,
        )[0].to(state.latents.dtype)
        state.step_index += 1

    def post_decode(
        self,
        state: StepRequestState,
        **kwargs: Any,
    ) -> DiffusionOutput:
        return DiffusionOutput(output=self.decode(state.latents))
```

This is a shape guide, not drop-in code: verify the target revision's
`StepRequestState.extra`, scheduler signature, CFG handling, state fields, and
decode contract. Follow a current native pipeline such as Qwen-Image for the
exact revision.

Never use the shared `self.scheduler` to advance a request. Store scheduler,
latents, embeddings, timesteps, masks, RNG/generator, task metadata, cache
policy, adapter, and output assembly state on `StepRequestState` or its
request-local extension. Pipeline fields may hold immutable weights/config and
bounded immutable caches only.

### Qualification sequence

1. Compare request-mode and step-mode at `max_num_seqs=1` for every task using
   identical seed/schedule/shape and intermediate latents.
2. Abort queued and in-flight requests at several step boundaries.
3. Increase to two compatible requests; verify independent scheduler/RNG/state
   and output order.
4. Increase to the target capacity and mix prompt lengths.
5. Exercise mixed shape, frames, steps/schedule, task, guidance, and quality.
   Batch only compatible rows; otherwise split/reject predictably.
6. Inject a failure in encode, denoise, scheduler, and decode independently.
7. Run the production soak and fault matrix.

Candidate commands only after capability validation:

```bash
# Conservative step execution and step-boundary abort.
vllm serve '<model>' --omni \
  --step-execution \
  --max-num-seqs 1

# Experimental batched-step candidate after multi-request E2E.
vllm serve '<model>' --omni \
  --step-execution \
  --max-num-seqs 4
```

If an attention backend is only correct for single-request step mode, fail
early or keep the recipe at `max_num_seqs=1`. Do not describe generic runtime
support as model support.

## Abort and cleanup

### Required semantics

For queued and in-flight abort/disconnect:

- emit exactly one terminal aborted/error result;
- do not run `post_decode` after abort unless a documented partial-output mode
  explicitly requires it;
- do not increment `step_index` or apply a scheduler result after cancellation;
- remove scheduler/state cache entries and request IDs;
- release latents, prompt/reference tensors, cache hooks, DLO buffers, connector
  messages, uploaded/temp files, and output buffers;
- keep other requests alive;
- make cleanup idempotent when disconnect, timeout, and worker error race;
- drop late results/exceptions for already-resolved or cancelled futures
  without terminating the shared result-pump thread;
- complete the next valid request with its requested cache/offload policy.

Test cancellation before engine admission, while queued, after encode, during
early/middle/final denoise steps, before decode, and while returning/persisting
output. Include async job cancellation and client socket disconnect.

### Failure injection matrix

| Location | Injection | Assertions |
|---|---|---|
| Validation/upload | bad MIME, count/bytes overflow, truncated media | 400/413, no job/engine submit, no temp leak |
| Load/start | missing fused shard, unsupported topology | startup fails, actionable error, no silent init |
| Encode | exception/OOM/timeout | one error, request state/temp released |
| Denoise | exception/OOM/worker exit | peers handled, hooks/buffers released, health behavior explicit |
| Scheduler | invalid state/shape | no latent/step partial commit |
| Decode/post | exception/codec error | no corrupt successful result, partial output removed |
| Result/IPC | late result after cancel, large split tensor, orphan SHM | pump remains alive, one terminal result, exact ownership cleanup |
| Connector/disagg | timeout/retry/duplicate | idempotency, backpressure, no orphan payload |

After each injection, check server health and issue a known-good request. If
the process must restart, document and test the restart policy; do not claim
transparent recovery.

### Large-output transport

Treat worker-to-engine IPC and engine-to-client transport as separate hops.
For batched video, prove that each logical output does not accidentally retain
or serialize the complete batch storage. Exercise the target revision's shared
memory or artifact-handle path above and below its threshold, including
non-contiguous views, multiple outputs, consumer failure, timeout, abort, and
worker exit. Verify every shared segment/file is released exactly once.

Freeze the media contract at each hop: dtype, value range, shape/layout,
contiguity, request ownership, encoding/color model, payload bytes, and pending
consumers. Device-side float-to-uint8 preparation, D2H/IPC, planar/interleaved
conversion, CPU encode/mux, and delivery off the asyncio event loop are
different optimizations. Validate their route selection and fallbacks
independently. Byte-identical HTTP MP4 output does not prove that a raw offline
tensor kept its dtype or range.

For long-window output, validate segment ordering and audio/video continuity,
then exercise payloads around serializer, shared-memory, HTTP, and codec size
limits. Chunking must reassemble exactly, reject malformed/missing chunks, and
release partial state after abort, timeout, or peer failure.

Qualify chunk production, transport, and public streaming as separate layers.
A VAE callback must define owned representation, frame offsets, final marker,
rank ownership, source/config gate, and collective-safe failure, but it does not
prove SHM/ZMQ delivery, bounded backpressure, cancellation, codec reuse, or
client streaming. Measure time to first owned chunk, first encoded fragment,
steady cadence, mux finalization, and complete client artifact independently.

Measure remote MP4 encoding, serialization/copies, network transfer, and client
materialization separately from denoise/decode. An inference-kernel speedup does
not establish serving throughput if output transport dominates or leaks.

## Mixed-RPS soak

Use a realistic request mix and arrival distribution. Include small/large
shapes, short/long prompts, all advertised tasks, cache tiers, and a small rate
of abort/disconnect/error injections. Run phases below, near, and above measured
saturation so queue/backpressure behavior is visible.

Example serving benchmark phase:

```bash
python benchmarks/diffusion/diffusion_benchmark_serving.py \
  --base-url http://127.0.0.1:8091 \
  --endpoint /v1/videos \
  --model '<model>' \
  --dataset trace \
  --dataset-path /path/to/versioned-production-trace.jsonl \
  --task t2v \
  --num-prompts 500 \
  --request-rate 0.5 \
  --max-concurrency 8 \
  --warmup-requests 8 \
  --warmup-concurrency 8 \
  --output-file soak-near-saturation.json
```

Use a task-appropriate trace/media dataset. Run the same phase at rates selected
from an initial saturation curve; a literal `0.5` is illustrative only.

On supported GPU hosts, wrap a test or soak with the repository monitor:

```bash
bash tests/dfx/stability/scripts/resource_monitor.sh run --backend gpu -- \
  pytest -s -v tests/dfx/stability/scripts/test_stability_<model>.py -m slow
```

Check the target revision's monitor backends. Do not claim CPU/NPU monitoring
from a reserved CLI choice unless collection is implemented and verified.

Report:

- offered and achieved RPS, success/error/abort rates;
- p50/p95/p99 and queue/service/stage times, with the successful sample count
  and arrival process used for each percentile;
- throughput per device and batch/wave utilization;
- per-rank HBM and process-tree PSS trend/slope;
- cache size/hit rate, temp-file/disk growth, open FDs/handles;
- outstanding shared-memory/artifact handles and encoded/network bytes;
- worker/engine health, restarts, OOM/timeouts, backpressure response;
- total duration, request count/mix, raw logs/JSON/monitor artifact hashes.

A short burst or repeated serial curl loop is not a stability test.

## Four CI tracks

Keep the tracks independent so a failure answers a specific question. These
four readiness tracks are not the repository's numbered CI levels. Map them
onto the current L1-L5 taxonomy:

| Readiness track | Repository CI placement |
|---|---|
| Function | L1 unit/logic plus L2 basic online/offline E2E; extend real-model scenarios in L3/L4 |
| Accuracy | L3 real-model gates for important cases; deeper and broader nightly coverage in L4 |
| Performance | Important thresholds in L3 where they fit the time budget; full baselines/regression jobs in nightly L4 |
| Reliability | Cheap rejection/recovery assertions can run earlier, but long stability, fault injection, and reliability suites belong to weekly L5 |

Distributed topology is a test case within the appropriate level, not a
replacement definition for a level.

### 1. Function CI

Cover:

- strict unquantized and advertised quantized loading;
- fused source-shard completeness and prefix routing;
- official task positive boundaries and neighboring negative cases;
- offline, sync, async normalization/parity;
- single-device reference plus every advertised topology smoke;
- output type/count/order and request ID isolation;
- feature rejection paths that must fail early.

### 2. Accuracy CI

Pin official implementation/checkpoint/vLLM-Omni revisions, prompts and media
hashes, seed, scheduler/sigma/steps, dimensions/frames/FPS, guidance, dtype,
backend, and topology. Compare at suitable levels:

- component outputs or selected blocks;
- one or more intermediate denoise latents;
- final image/video/audio artifact metrics;
- temporal consistency and audio/video synchronization where relevant.

Set tolerance per modality/hardware/backend with rationale. Keep failure
artifacts and metadata. A single visual inspection does not define CI.

For quantization trajectory tooling, inspect the target revision's current
`vllm_omni/quantization/tools/compare_diffusion_trajectory_similarity.py` and
reuse it when compatible instead of creating an ad hoc comparator.

### 3. Performance CI

Pin workload and best deployment. Include a resident recommended row and a
small-HBM/DLO row when both are production targets. Declare:

- one explicit warmup exclusion and at least three measured fixed-work
  repetitions;
- raw result JSON and environment identity;
- median/mean plus range for small fixed-work A/Bs; p50/p95/p99 only for an
  arrival-load run with enough samples, plus throughput/RPS, stage time,
  HBM/PSS;
- regression metric, threshold, baseline owner, and update procedure;
- correctness/accuracy check linked to the same configuration.

Use the current harness config style, for example:

```bash
pytest -s -v tests/dfx/perf/scripts/run_diffusion_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_<model>_vllm_omni.json
```

Verify parser options on the target revision. Do not copy a JSON schema from a
different model without matching its endpoint, task, dataset, and deploy path.

### 4. Reliability CI

Run a bounded CI soak plus a longer scheduled soak. Cover:

- below/near/above-saturation arrival phases;
- queued/in-flight abort and disconnect;
- OOM/timeout and worker/process failure appropriate to the platform;
- DLO/cache/connector cleanup;
- health, fast-fail/backpressure, restart/recovery policy;
- memory/temp/FD trend and successful known-good request after each fault.

Keep reliability thresholds independent from latency thresholds. A fast server
that leaks memory or hangs after cancellation fails production readiness.

## PR evidence checklist

The PR description must include or link:

- official source/checkpoint revisions and API/limit matrix;
- scoped feature compatibility matrix with the four allowed states;
- implementation notes for FP8, DLO, request cache, attention/fusion, batching;
- exact environment, deploy/serve, every task curl, benchmark and soak commands;
- fixed accuracy inputs and artifacts;
- raw before/after performance runs and memory traces;
- output-contract manifests and stage accounting from device preparation
  through the complete client artifact;
- abort/disconnect/fault results and cleanup evidence;
- Function/Accuracy/Performance/Reliability CI files and ownership;
- limitations/TODOs that remain `not tested` rather than implied support.

Separate the Day-0 milestone from production readiness. If the PR only closes
some gates, title and description should say which scoped gates it closes.
