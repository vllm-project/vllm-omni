# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chunk-step execution policy below the request/session runner."""

from collections.abc import Callable, MutableMapping

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.distributed.member_batch import ChunkStepPlan, planned_task
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, StepRequestState


def _chunk_step_pending(
    request_id: str,
    states: MutableMapping[str, StepRequestState],
    output: BatchRunnerOutput,
) -> bool:
    """More steps of this request's current chunk remain after ``output``."""
    member = output.get_request_output(request_id)
    return member is not None and not member.finished and member.result is None and request_id in states


def advance_chunk(
    request_id: str,
    states: MutableMapping[str, StepRequestState],
    output: BatchRunnerOutput,
    advance: Callable[[], BatchRunnerOutput],
    *,
    previous_step: int | None,
) -> BatchRunnerOutput:
    """Advance through the penultimate step, preserving the interaction boundary.

    The scheduler gets a fresh cycle before the final denoise step can decode
    and apply interactions. A model may finish early, fail, or drop its state;
    none of those cases may trigger another compute. The initial step has
    already executed and counts against the chunk's bounded compute budget.
    """

    if not _chunk_step_pending(request_id, states, output):
        return output
    state = states[request_id]
    if state.chunk_num_steps is None:
        return output
    plan = ChunkStepPlan(state.chunk_index, state.chunk_num_steps)
    calls = 1
    while _chunk_step_pending(request_id, states, output):
        state = states[request_id]
        final = planned_task(plan, plan.num_slots - 1, 0)
        assert final is not None
        if state.step_in_chunk > 0 and state.step_in_chunk == final[1] and state.step_in_chunk != previous_step:
            break
        if calls >= plan.num_slots:
            states.pop(request_id)
            member = output.get_request_output(request_id)
            assert member is not None
            member.finished = True
            member.result = DiffusionOutput(error=f"Chunk did not complete within {plan.num_slots} denoise steps")
            break
        previous_step = state.step_in_chunk
        calls += 1
        output = advance()
    return output
