# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chunk execution preserves the scheduler's final-step interaction boundary."""

import pytest

from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput, StepRequestState
from vllm_omni.experimental.ar_diffusion.chunk_executor import advance_chunk
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _ChunkStepper:
    """Advance callback that records the step it saw and moves the state on."""

    def __init__(self, state: StepRequestState, *, stuck: bool = False) -> None:
        self.state = state
        self.stuck = stuck
        self.calls: list[int] = []

    def __call__(self) -> BatchRunnerOutput:
        self.calls.append(self.state.step_in_chunk)
        if not self.stuck:
            self.state.step_in_chunk += 1
        return BatchRunnerOutput.from_list([RunnerOutput(request_id="a", step_index=self.state.step_in_chunk)])


class _NoAdvance:
    """Advance callback that fails the test if a chunk asks for more compute."""

    def __call__(self) -> BatchRunnerOutput:
        pytest.fail("A finished/cancelled request must not compute again")


def test_chunk_advances_to_penultimate_step_not_interaction_boundary():
    state = StepRequestState("a", OmniDiffusionSamplingParams(), chunk_num_steps=4, step_in_chunk=1)
    states = {"a": state}
    output = BatchRunnerOutput.from_list([RunnerOutput(request_id="a", step_index=1)])
    advance = _ChunkStepper(state)

    result = advance_chunk("a", states, output, advance, previous_step=0)
    assert advance.calls == [1, 2]
    assert state.step_in_chunk == 3
    assert not result.get_request_output("a").finished


def test_chunk_stuck_model_is_bounded_and_drops_state():
    state = StepRequestState("a", OmniDiffusionSamplingParams(), chunk_num_steps=2, step_in_chunk=0)
    states = {"a": state}
    output = BatchRunnerOutput.from_list([RunnerOutput(request_id="a", step_index=0)])
    advance = _ChunkStepper(state, stuck=True)

    result = advance_chunk("a", states, output, advance, previous_step=0)
    assert advance.calls == [0]
    assert "a" not in states
    member = result.get_request_output("a")
    assert member.finished
    assert member.result.error == "Chunk did not complete within 2 denoise steps"


def test_chunk_does_not_advance_after_output_or_cancellation():
    state = StepRequestState("a", OmniDiffusionSamplingParams(), chunk_num_steps=2)
    output = BatchRunnerOutput.from_list([RunnerOutput(request_id="a", step_index=0, finished=True)])
    unexpected = _NoAdvance()

    assert advance_chunk("a", {"a": state}, output, unexpected, previous_step=0) is output
    output.get_request_output("a").finished = False
    assert advance_chunk("a", {}, output, unexpected, previous_step=0) is output
