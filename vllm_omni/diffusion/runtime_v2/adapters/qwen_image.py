# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen-Image runtime_v2 adapter, task compiler, and step-exec executors.

Rather than re-implementing the diffusion forward path with low-level
primitives, every executor drives the ``QwenImagePipeline`` *step-execution
stage methods* — ``prepare_encode`` / ``denoise_step`` / ``step_scheduler`` /
``post_decode`` — exactly as ``DiffusionModelRunner.execute_stepwise`` does.

The single shared artifact passed along the linear DAG is a
``DiffusionRequestState`` (kind ``REQUEST_STATE``, layout ``WORKER_LOCAL``):

    TEXT_ENCODE  ->  [DIT_STEP_CHUNK] * ceil(num_inference_steps / chunk)
                 ->  VAE_DECODE   (terminal; its DiffusionOutput is the result)

PR1 is single-group / host-local: there is no cross-group artifact migration.
The executors require a real GPU pipeline + model and are exercised by the GPU end-to-end test;
what is CPU-testable now is the compiler, ``validate_pipeline``, import, and
``build_executors``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

from vllm.logger import init_logger

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.runtime_v2.interfaces import (
    RuntimeV2Adapter,
    TaskCompiler,
    WorkerExecutor,
)
from vllm_omni.diffusion.runtime_v2.protocol import (
    ArtifactHandle,
    ArtifactKind,
    ArtifactLayout,
    ArtifactValue,
    InferenceTask,
    ParallelSpec,
    RequestExecutionPlan,
    StepRange,
    TaskKind,
)

logger = init_logger(__name__)


# Task kinds emitted by the Qwen-Image linear compiler. prepare_encode() bundles
# prompt encode + latents + timesteps + scheduler setup into the single
# TEXT_ENCODE task, so there is no separate dit/timestep prepare stage.
QWEN_TASK_KINDS: tuple[TaskKind, ...] = (
    TaskKind.TEXT_ENCODE,
    TaskKind.DIT_STEP_CHUNK,
    TaskKind.VAE_DECODE,
)

# Stage methods that define the step-execution contract this adapter drives.
_REQUIRED_STAGE_METHODS: tuple[str, ...] = (
    "prepare_encode",
    "denoise_step",
    "step_scheduler",
    "post_decode",
)


# --------------------------------------------------------------------------- #
# Request wrapper                                                             #
# --------------------------------------------------------------------------- #


class QwenRuntimeRequest:
    """Normalized runtime_v2 request: an OmniDiffusionRequest + chunk size.

    The compiler only needs the request id, the sampling params (for
    ``num_inference_steps``) and the denoise chunk size; everything else the
    executors read straight off the wrapped ``OmniDiffusionRequest``.
    """

    def __init__(
        self,
        diffusion_request: OmniDiffusionRequest,
        denoise_chunk_size: int = 1,
        *,
        priority: int = 0,
        group_id: str | None = None,
    ) -> None:
        if denoise_chunk_size < 1:
            raise ValueError(f"denoise_chunk_size must be >= 1, got {denoise_chunk_size}")
        self.diffusion_request = diffusion_request
        self.request_id = diffusion_request.request_id
        self.denoise_chunk_size = int(denoise_chunk_size)
        self.priority = int(priority)
        self.group_id = group_id


# --------------------------------------------------------------------------- #
# Task compiler                                                              #
# --------------------------------------------------------------------------- #


class QwenImageTaskCompiler(TaskCompiler):
    """Emit a linear stage DAG for one Qwen-Image request.

    The denoise loop is partitioned into ``ceil(num_steps / chunk)`` contiguous
    ``DIT_STEP_CHUNK`` tasks, each carrying a ``StepRange(start, end)`` over the
    step index. Every task consumes the previous task's ``DiffusionRequestState``
    artifact and produces the (mutated) state, so the scheduler sees a strict
    linear dependency chain.
    """

    def __init__(
        self,
        default_denoise_chunk_size: int = 1,
        *,
        od_config: Any = None,
        pipeline: Any = None,
    ) -> None:
        if default_denoise_chunk_size < 1:
            raise ValueError("default_denoise_chunk_size must be >= 1")
        self.default_denoise_chunk_size = int(default_denoise_chunk_size)
        self.od_config = od_config
        self.pipeline = pipeline

    def compile_request(self, request: Any) -> RequestExecutionPlan:
        if not isinstance(request, QwenRuntimeRequest):
            raise TypeError(f"unsupported runtime_v2 request type: {type(request)!r}")

        req = request.diffusion_request
        request_id = request.request_id
        if req.sampling_params.sigmas is not None:
            raise NotImplementedError("runtime_v2 does not support custom sigmas for Qwen-Image")
        raw_num_steps = req.sampling_params.num_inference_steps
        # `or 50` would mask an explicit 0 (making it slip past the `< 1` guard
        # below and silently run 50 steps); only substitute when unset.
        num_steps = int(raw_num_steps) if raw_num_steps is not None else 50
        if num_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {num_steps}")

        chunk_size = int(request.denoise_chunk_size or self.default_denoise_chunk_size)
        if chunk_size < 1:
            raise ValueError(f"denoise_chunk_size must be >= 1, got {chunk_size}")
        num_chunks = math.ceil(num_steps / chunk_size)

        group_id = request.group_id
        priority = request.priority
        parallel_spec = ParallelSpec()

        tasks: dict[str, InferenceTask] = {}

        def _state_handle(artifact_id: str, producer_task_id: str | None) -> ArtifactHandle:
            return ArtifactHandle(
                request_id=request_id,
                artifact_id=artifact_id,
                kind=ArtifactKind.REQUEST_STATE,
                layout=ArtifactLayout.WORKER_LOCAL,
                producer_task_id=producer_task_id,
            )

        # ── TEXT_ENCODE: prepare_encode (encode + latents + timesteps) ──
        prep_task_id = f"{request_id}:text_encode"
        # The initial input artifact is the raw request itself, seeded into the
        # plan as an initial_artifact so QwenPrepareExecutor can build the state.
        request_handle = ArtifactHandle(
            request_id=request_id,
            artifact_id=f"{request_id}:request",
            kind=ArtifactKind.REQUEST_STATE,
            layout=ArtifactLayout.HOST,
            producer_task_id=None,
        )
        prep_out = _state_handle(f"{request_id}:state_text", prep_task_id)
        tasks[prep_task_id] = InferenceTask(
            task_id=prep_task_id,
            request_id=request_id,
            kind=TaskKind.TEXT_ENCODE,
            group_id=group_id,
            parallel_spec=parallel_spec,
            priority=priority,
            dependencies=(),
            inputs=(request_handle,),
            outputs=(prep_out,),
        )

        # ── DIT_STEP_CHUNK * num_chunks ──
        prev_task_id = prep_task_id
        prev_out = prep_out
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, num_steps)
            dit_task_id = f"{request_id}:dit:{chunk_idx}"
            dit_out = _state_handle(f"{request_id}:state_dit:{chunk_idx}", dit_task_id)
            tasks[dit_task_id] = InferenceTask(
                task_id=dit_task_id,
                request_id=request_id,
                kind=TaskKind.DIT_STEP_CHUNK,
                group_id=group_id,
                parallel_spec=parallel_spec,
                priority=priority,
                dependencies=(prev_task_id,),
                inputs=(prev_out,),
                outputs=(dit_out,),
                step_range=StepRange(start, end),
            )
            prev_task_id = dit_task_id
            prev_out = dit_out

        # ── VAE_DECODE: post_decode -> DiffusionOutput (terminal) ──
        # The decoded DiffusionOutput IS the request result: it is fetched from
        # the worker and unpacked in StageDiffusionProc, which runs the CPU
        # postprocess. No separate finalize stage is needed.
        vae_task_id = f"{request_id}:vae_decode"
        vae_out = ArtifactHandle(
            request_id=request_id,
            artifact_id=f"{request_id}:output",
            kind=ArtifactKind.OUTPUT,
            layout=ArtifactLayout.WORKER_LOCAL,
            producer_task_id=vae_task_id,
        )
        tasks[vae_task_id] = InferenceTask(
            task_id=vae_task_id,
            request_id=request_id,
            kind=TaskKind.VAE_DECODE,
            group_id=group_id,
            parallel_spec=parallel_spec,
            priority=priority,
            dependencies=(prev_task_id,),
            inputs=(prev_out,),
            outputs=(vae_out,),
        )

        plan = RequestExecutionPlan(
            request_id=request_id,
            tasks=tasks,
            terminal_task_ids=(vae_task_id,),
            initial_artifacts=(ArtifactValue(handle=request_handle, value=req),),
            metadata={
                "adapter": "qwen_image",
                "num_steps": num_steps,
                "chunk_size": chunk_size,
                "num_chunks": num_chunks,
            },
        )
        logger.debug(
            "runtime_v2 compile: request_id=%s adapter=qwen_image steps=%s chunk=%s denoise_tasks=%s total_tasks=%s",
            request_id,
            num_steps,
            chunk_size,
            num_chunks,
            len(plan.tasks),
        )
        return plan


# --------------------------------------------------------------------------- #
# Executors (drive QwenImagePipeline step-exec stage methods)               #
# --------------------------------------------------------------------------- #


class _BaseQwenExecutor(WorkerExecutor):
    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    @staticmethod
    def _single_input(task: InferenceTask, resolved_inputs: Mapping[str, Any]) -> Any:
        if len(task.inputs) != 1:
            raise ValueError(f"{task.kind} expects exactly one input artifact")
        artifact_id = task.inputs[0].artifact_id
        if artifact_id not in resolved_inputs:
            raise KeyError(f"missing resolved input for artifact {artifact_id}")
        return resolved_inputs[artifact_id]


class QwenPrepareExecutor(_BaseQwenExecutor):
    """TEXT_ENCODE: build the per-request state and run ``prepare_encode``.

    Mirrors ``DiffusionModelRunner._update_states`` (state construction) +
    ``_prepare_batch_inputs`` (the ``prepare_encode`` call for new requests).
    ``prepare_encode`` populates prompt embeds, latents, timesteps, the
    per-request scheduler, and CFG config on the state.
    """

    def execute(self, task: InferenceTask, resolved_inputs: Mapping[str, Any]) -> tuple[ArtifactValue, ...]:
        # Imported lazily so this module imports without torch/worker deps on
        # the host / in CPU-only collection.
        import torch

        from vllm_omni.diffusion.worker.utils import DiffusionRequestState

        req = cast(OmniDiffusionRequest, self._single_input(task, resolved_inputs))

        state = DiffusionRequestState(
            request_id=task.request_id,
            sampling=req.sampling_params,
            prompt=req.prompt,
            kv_sender_info=req.kv_sender_info,
        )

        # Match _prepare_batch_inputs: materialize the generator from the seed
        # before encoding so the initial latent noise is deterministic and
        # matches the single-group baseline.
        sampling = state.sampling
        if sampling.generator is None and sampling.seed is not None:
            device = getattr(self.pipeline, "device", None)
            if sampling.generator_device is not None:
                gen_device = sampling.generator_device
            elif device is not None and getattr(device, "type", None) == "cpu":
                gen_device = "cpu"
            elif device is not None:
                gen_device = device
            else:
                gen_device = "cpu"
            sampling.generator = torch.Generator(device=gen_device).manual_seed(int(sampling.seed))

        self.pipeline.prepare_encode(state)
        return (ArtifactValue(handle=task.outputs[0], value=state),)


class QwenDenoiseChunkExecutor(_BaseQwenExecutor):
    """DIT_STEP_CHUNK: run one chunk of denoise steps over the input state.

    Mirrors the per-step inner loop of ``execute_stepwise``: build an
    ``InputBatch`` from the single state, call ``denoise_step`` to get the
    batched ``noise_pred``, then slice the per-state row offset and feed it to
    ``step_scheduler`` (which mutates ``state.latents`` and advances
    ``step_index``). For a single-request batch the offset slice is the full
    ``noise_pred`` (``offset == 0``, ``row_num == state.latents.shape[0]``).
    """

    def execute(self, task: InferenceTask, resolved_inputs: Mapping[str, Any]) -> tuple[ArtifactValue, ...]:
        if task.step_range is None:
            raise ValueError("dit_step_chunk requires step_range")

        from vllm_omni.diffusion.worker.input_batch import InputBatch
        from vllm_omni.diffusion.worker.utils import DiffusionRequestState

        state = cast(DiffusionRequestState, self._single_input(task, resolved_inputs))
        if state.latents is None or state.timesteps is None:
            raise RuntimeError("state is not ready for denoising (missing latents/timesteps)")

        # Clamp the chunk's step range to the request's actual step count so a
        # final ragged chunk does not over-run the timestep schedule.
        start = max(0, task.step_range.start)
        end = min(task.step_range.end, state.total_steps)

        cached_batch: InputBatch | None = None
        for _ in range(start, end):
            input_batch = InputBatch.make_batch([state], cached_batch=cached_batch)
            cached_batch = input_batch
            noise_pred = self.pipeline.denoise_step(input_batch, states=[state])

            if noise_pred is None:
                # Interrupted: leave the state as-is and stop the chunk early.
                break

            # Single-request batch: consume exactly this state's rows, matching
            # the offset slicing in execute_stepwise.
            row_num = state.latents.shape[0]
            self.pipeline.step_scheduler(state, noise_pred[0:row_num])

        return (ArtifactValue(handle=task.outputs[0], value=state),)


class QwenDecodeExecutor(_BaseQwenExecutor):
    """VAE_DECODE: ``post_decode(state)`` -> ``DiffusionOutput``."""

    def execute(self, task: InferenceTask, resolved_inputs: Mapping[str, Any]) -> tuple[ArtifactValue, ...]:
        from vllm_omni.diffusion.worker.utils import DiffusionRequestState

        state = cast(DiffusionRequestState, self._single_input(task, resolved_inputs))
        if state.latents is None:
            raise RuntimeError("state.latents is None in decode stage")
        result = self.pipeline.post_decode(state)
        return (ArtifactValue(handle=task.outputs[0], value=result),)


def build_executors(pipeline: Any) -> dict[TaskKind, WorkerExecutor]:
    """Map each Qwen-Image task kind to its step-exec executor."""
    return {
        TaskKind.TEXT_ENCODE: QwenPrepareExecutor(pipeline),
        TaskKind.DIT_STEP_CHUNK: QwenDenoiseChunkExecutor(pipeline),
        TaskKind.VAE_DECODE: QwenDecodeExecutor(pipeline),
    }


def validate_pipeline(pipeline: Any, od_config: Any) -> None:
    """Assert the pipeline implements the step-execution contract."""
    if pipeline is None:
        raise ValueError("runtime_v2 qwen-image adapter requires an initialized pipeline")
    if not getattr(pipeline, "supports_step_execution", False):
        raise ValueError("runtime_v2 qwen-image adapter requires a pipeline with supports_step_execution=True")
    for method_name in _REQUIRED_STAGE_METHODS:
        method = getattr(pipeline, method_name, None)
        if not callable(method):
            raise ValueError(f"qwen-image runtime_v2 adapter requires callable pipeline.{method_name}")
    # Cache backends are OUT of PR1 scope. The runtime_v2 executors call the
    # pipeline step methods directly, bypassing DiffusionModelRunner.execute_stepwise
    # -- and thus both its per-request cache refresh (_refresh_cache_for_requests)
    # AND its "Step mode does not support cache_backend" guard. With a cache
    # backend enabled on the pipeline (the worker load path does enable it), the
    # first request would run with stale / unrefreshed cache state instead of
    # failing. Reject it loudly at startup, mirroring the legacy stepwise guard.
    cache_backend = getattr(od_config, "cache_backend", None)
    if cache_backend not in (None, "none"):
        raise ValueError(
            f"runtime_v2 (PR1) does not support cache_backend={cache_backend!r}; "
            "use the legacy path or set cache_backend='none'"
        )


# --------------------------------------------------------------------------- #
# Adapter                                                                    #
# --------------------------------------------------------------------------- #


class QwenImageRuntimeV2Adapter(RuntimeV2Adapter):
    model_class_name = "QwenImagePipeline"

    @property
    def supported_task_kinds(self) -> tuple[TaskKind, ...]:
        return QWEN_TASK_KINDS

    def normalize_request(self, request: Any, denoise_chunk_size: int) -> QwenRuntimeRequest:
        if isinstance(request, QwenRuntimeRequest):
            return request
        if not isinstance(request, OmniDiffusionRequest):
            raise TypeError(f"unsupported runtime_v2 request type: {type(request)!r}")
        return QwenRuntimeRequest(
            diffusion_request=request,
            denoise_chunk_size=denoise_chunk_size,
        )

    def build_task_compiler(
        self,
        default_denoise_chunk_size: int,
        *,
        od_config: Any = None,
        pipeline: Any = None,
    ) -> TaskCompiler:
        return QwenImageTaskCompiler(
            default_denoise_chunk_size=default_denoise_chunk_size,
            od_config=od_config,
            pipeline=pipeline,
        )

    def build_executors(self, pipeline: Any) -> dict[TaskKind, WorkerExecutor]:
        return build_executors(pipeline)

    def validate_pipeline(self, pipeline: Any, od_config: Any) -> None:
        validate_pipeline(pipeline, od_config)
