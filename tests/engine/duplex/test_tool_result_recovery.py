# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tool recovery ordering through the real runner and Nemotron worker consumer.

Stage outputs use the existing recording port; these are not model inference
or WebSocket end-to-end tests.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.engine.duplex.test_session_runner import append_audio
from tests.engine.duplex.test_session_runner_nemotron_voicechat import (
    Thinker,
    answer,
    batches,
    deliver,
    error_codes,
    generation,
    open_nemotron_harness,
    open_timed_harness,
    prepare,
)
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.session.tool_calls import ToolCallState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("acknowledged", [False, True])
async def test_cancel_then_new_result_before_first_worker_sync(monkeypatch, acknowledged):
    async with open_nemotron_harness(monkeypatch) as h:
        old_call = await prepare(h)
        assert error_codes(await answer(h, old_call)) == []
        if acknowledged:
            await deliver(h, nvc_function_response_consumed_generation=1)
            assert batches(h) == []
        await h.run(commands.CancelInput())
        assert batches(h) == []

        # No worker synchronized the retired runtime snapshot. The new epoch
        # can receive another result before its first worker frame executes.
        new_call = await prepare(h)
        assert error_codes(await answer(h, new_call)) == []
        assert generation(h) == 2
        worker = {"function_response_generation": 0, "forced_function_tokens": []}
        Thinker._sync_forced_function_response(worker, h.session.runtime_config)
        assert worker["function_response_generation"] == 2
        assert worker["forced_function_tokens"] == [42]


@pytest.mark.asyncio
async def test_pending_tool_does_not_block_incoming_audio(monkeypatch):
    async with open_nemotron_harness(monkeypatch) as h:
        call_id = await prepare(h)
        submissions = len(h.port.submissions)
        assert error_codes(await h.run(append_audio(samples=1280))) == []
        assert len(h.port.submissions) == submissions + 1
        assert h.runner.tool_calls.record(call_id).state is ToolCallState.PENDING
        assert error_codes(await answer(h, call_id)) == []
        assert generation(h) == 1


def test_missing_batches_without_retirement_are_still_rejected():
    worker = {"function_response_generation": 0, "forced_function_tokens": []}
    with pytest.raises(ValueError, match="incomplete"):
        Thinker._sync_forced_function_response(
            worker,
            {"nvc_function_response_generation": 1, "nvc_function_response_batches": []},
        )
    assert worker["function_response_generation"] == 0
    assert worker["forced_function_tokens"] == []


def test_out_of_order_batches_are_refused_without_mutating_worker():
    worker = {"function_response_generation": 0, "forced_function_tokens": [9]}
    with pytest.raises(ValueError, match="not contiguous"):
        Thinker._sync_forced_function_response(
            worker,
            {
                "nvc_function_response_generation": 2,
                "nvc_function_response_batches": [
                    {"generation": 2, "token_ids": [22]},
                    {"generation": 1, "token_ids": [11]},
                ],
            },
        )
    assert worker == {"function_response_generation": 0, "forced_function_tokens": [9]}


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ["timeout", "cancel"])
async def test_fresh_response_completes_after_abandoned_tool(monkeypatch, interruption):
    async with open_timed_harness(monkeypatch) as h:
        call_id = await prepare(h)
        if interruption == "timeout":
            h.clock.advance(5.0)
            h.scheduler.fire_all()
            assert error_codes(await h.settle()) == ["tool_result_timeout"]
        else:
            await h.run(commands.CancelInput())
        assert error_codes(await answer(h, call_id)) == ["invalid_function_call_output"]

        # A new user input proceeds through the real runner. Model text/EOS and
        # codec output are supplied at the existing stage-port seam, rather
        # than injecting response.done or changing session state directly.
        await h.run(append_audio(samples=1280))
        started = await deliver(h, token_id=7)
        response_id = next(event.response["id"] for event in started if event.type == "response.created")
        await deliver(h, token_id=1)
        output = SimpleNamespace(
            request_id=h.stage0_request_id(),
            finished=False,
            outputs=[],
            multimodal_output={"model_outputs": [np.ones(16_000, dtype=np.float32)], "sr": [16_000]},
        )
        events = await h.deliver_and_settle(output, stage_id=2, segment_finished=True)
        completed = [event for event in events if event.type == "response.done"]
        assert len(completed) == 1
        assert completed[0].response["id"] == response_id
        assert completed[0].status == "completed"
        assert error_codes(events) == []
