# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bridge from step-execution pipelines to request-batch ``forward()``.

Step-level and request-level batching are two different entry points into
the same underlying pipeline: step mode drives
``prepare_encode``/``denoise_step``/``step_scheduler``/``post_decode`` one
scheduler wave at a time (see ``DiffusionModelRunner._execute_stepwise_core``),
while request-batch mode expects a single ``pipeline.forward(DiffusionRequestBatch)``
call. Today, adding request-batch support to a pipeline that already
implements ``SupportsStepExecution`` means hand-writing a second, parallel
implementation of the same merge/scatter mechanics (see e.g.
``QwenImagePipeline.forward``) that can drift from the step-mode path.

This module drives the exact same step-execution contract to completion
in-process, so a pipeline only has to implement it once. It has no scheduler
interleaving other requests mid-stream, so it is strictly simpler than the
runner's per-wave loop: every request starts together and runs to
completion before ``post_decode``.

Streaming/chunked output (``chunk_num_steps`` set) is out of scope here -
request-batch callers expect one final result per request, not incremental
chunks, so pipelines that only support chunked streaming should not use
this bridge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

from vllm_omni.diffusion.worker.input_batch import InputBatch, scatter_latents
from vllm_omni.diffusion.worker.utils import (
    StepRequestState,
    clear_pipeline_stage_durations,
    consume_pipeline_stage_durations,
    merge_stage_durations,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.models.interface import SupportsStepExecution
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    class _StepExecutionPipeline(SupportsStepExecution, Protocol):
        """The bridge's actual requirement: the step-execution contract plus
        the ``device`` every concrete pipeline already carries (not part of
        ``SupportsStepExecution`` itself, since step mode never needs it)."""

        device: torch.device


def _initialize_generator(pipeline: _StepExecutionPipeline, sampling_params: OmniDiffusionSamplingParams) -> None:
    """Seed a request's generator from its seed, mirroring the runner's
    ``DiffusionModelRunner._initialize_generator`` so step-execution pipelines
    see the same generator semantics regardless of entry point."""
    if sampling_params.generator is not None or sampling_params.seed is None:
        return
    if sampling_params.generator_device is not None:
        gen_device = sampling_params.generator_device
    elif pipeline.device.type == "cpu":
        gen_device = "cpu"
    else:
        gen_device = pipeline.device
    sampling_params.generator = torch.Generator(device=gen_device).manual_seed(sampling_params.seed)


def run_step_execution_to_completion(
    pipeline: _StepExecutionPipeline,
    requests: list[OmniDiffusionRequest],
) -> list[DiffusionOutput]:
    """Run a step-execution pipeline's full denoise loop for a batch of
    independent requests, returning one output per request in ``requests``
    order.

    Raises:
        NotImplementedError: if any request's ``prepare_encode`` sets up
            chunked/streaming decode, which this bridge does not support.
        ValueError: if a per-step ``denoise_step`` result does not account
            for every request row (mirrors the runner's own consistency
            check for a batched ``noise_pred``).
    """
    if not requests:
        return []

    states: list[StepRequestState] = []
    for req in requests:
        _initialize_generator(pipeline, req.sampling_params)
        state = StepRequestState(
            request_id=req.request_id,
            sampling=req.sampling_params,
            prompt=req.prompt,
            kv_sender_info=req.kv_sender_info,
            prepared_layout=req.prepared_layout,
        )
        clear_pipeline_stage_durations(pipeline)
        pipeline.prepare_encode(state)
        merge_stage_durations(state, consume_pipeline_stage_durations(pipeline))
        if state.chunk_num_steps is not None:
            raise NotImplementedError(
                f"{type(pipeline).__name__} requested chunked/streaming decode for "
                f"{req.request_id!r}; request-batch forward only supports a single "
                "final decode per request."
            )
        states.append(state)

    input_batch: InputBatch | None = None
    pending = list(states)
    while pending:
        input_batch = InputBatch.make_batch(pending, cached_batch=input_batch)
        clear_pipeline_stage_durations(pipeline)
        noise_pred = pipeline.denoise_step(input_batch, states=pending)
        denoise_stage_durations = consume_pipeline_stage_durations(pipeline)
        for state in pending:
            merge_stage_durations(state, denoise_stage_durations)

        if noise_pred is None:
            raise RuntimeError(
                f"{type(pipeline).__name__}.denoise_step returned no prediction while running a request-batch forward."
            )

        offset = 0
        for state in pending:
            assert state.latents is not None, f"Request {state.request_id} has no latents after denoise_step."
            row_num = state.latents.shape[0]
            pipeline.step_scheduler(state, noise_pred[offset : offset + row_num])
            offset += row_num
        if offset != noise_pred.shape[0]:
            raise ValueError(
                f"Step-execution batch forward consumed {offset} noise_pred rows, "
                f"but {type(pipeline).__name__}.denoise_step returned {noise_pred.shape[0]}."
            )

        gathered_latents = torch.cat([state.latents for state in pending], dim=0)
        input_batch.latents = gathered_latents
        scatter_latents(pending, input_batch)

        pending = [state for state in pending if not state.denoise_completed]

    outputs: dict[str, DiffusionOutput] = {}
    for state in states:
        clear_pipeline_stage_durations(pipeline)
        result = pipeline.post_decode(state)
        merge_stage_durations(state, consume_pipeline_stage_durations(pipeline))
        if result.stage_durations is None and state.stage_durations:
            result.stage_durations = dict(state.stage_durations)
        outputs[state.request_id] = result

    return [outputs[req.request_id] for req in requests]


class StepExecutionRequestBatchMixin:
    """Grants ``DiffusionRequestBatch`` request-batch ``forward()`` to any
    pipeline that already implements ``SupportsStepExecution``.

    Mix this in ahead of the pipeline base class instead of hand-writing a
    batched ``forward()``: it replays the exact
    prepare_encode/denoise_step/step_scheduler/post_decode contract the
    pipeline already maintains for scheduler-driven step mode, so the two
    entry points cannot drift apart.
    """

    supports_request_batch = True

    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        # A mixin's `self` only satisfies `_StepExecutionPipeline` once mixed
        # into a concrete pipeline that provides `device` and the
        # SupportsStepExecution methods; mypy can't see that from the mixin
        # alone.
        return run_step_execution_to_completion(self, req.requests)  # type: ignore[arg-type]
