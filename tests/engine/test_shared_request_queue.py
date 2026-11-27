# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared admission preserves cancellation and submission ordering on the CPU."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest

from vllm_omni.distributed.weight_sharing.runtime import SharingRuntime
from vllm_omni.engine.messages import AddCompanionRequestMessage, EngineQueueMessage, StageSubmissionMessage
from vllm_omni.engine.shared_request_queue import SharedRequestQueue

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def request(rid):
    return StageSubmissionMessage(
        type="add_request",
        request_id=rid,
        prompt="cat",
        original_prompt="cat",
        output_prompt_text=None,
        sampling_params_list=[],
        final_stage_id=1,
        preprocess_ms=0,
        request_timestamp=0,
        enqueue_ts=0,
    )


@asynccontextmanager
async def admission(dispatch=None):
    group = Mock(spec=SharingRuntime, state="READY")
    group.drain_requests = AsyncMock()
    gate = SharedRequestQueue(group)
    submitted: asyncio.Queue[EngineQueueMessage] = asyncio.Queue()

    async def submit(message):
        await submitted.put(message)
        if dispatch:
            await dispatch(gate, message)
        return True

    task = asyncio.create_task(gate.run(submit, asyncio.Event()))
    try:
        yield gate, group, submitted, task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_queued_cancel_and_duplicate_finish_do_not_release_active_request():
    async with admission() as (gate, group, submitted, _):
        for rid in ("one", "two", "three"):
            gate.enqueue(request(rid))
        assert (await submitted.get()).request_id == "one"
        assert [m.request_id for m in gate.discard(["two"])] == ["two"]
        gate.finish(["two"], aborted=True)
        await asyncio.sleep(0)
        assert submitted.empty()
        gate.finish(["one"], aborted=False)
        assert (await submitted.get()).request_id == "three"
        gate.finish(["one"], aborted=True)
        await asyncio.sleep(0)
        assert gate.active == "three"
        group.drain_requests.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_waits_for_late_submission_then_drain_before_next_request():
    submission_started, submission_done = asyncio.Event(), asyncio.Event()

    async def dispatch(gate, message):
        if message.request_id == "one":
            submission_started.set()
            await submission_done.wait()

    async with admission(dispatch) as (gate, group, submitted, _):
        drain_started, drained = asyncio.Event(), asyncio.Event()

        async def drain():
            drain_started.set()
            await drained.wait()

        group.drain_requests.side_effect = drain
        gate.enqueue(request("one"))
        gate.enqueue(request("two"))
        await submitted.get()
        await submission_started.wait()
        gate.finish(["one"], aborted=True)
        await asyncio.sleep(0)
        group.drain_requests.assert_not_called()
        submission_done.set()
        await drain_started.wait()
        assert submitted.empty() and gate.active == "one"
        drained.set()
        assert (await submitted.get()).request_id == "two"


@pytest.mark.asyncio
async def test_cancel_waits_for_inflight_downstream_forward():
    async with admission() as (gate, group, submitted, _):
        gate.enqueue(request("one"))
        gate.enqueue(request("two"))
        await submitted.get()
        async with gate.submission("one") as allowed:
            assert allowed
            gate.finish(["one"], aborted=True)
            await asyncio.sleep(0)
            group.drain_requests.assert_not_called()
            assert submitted.empty()
        assert (await submitted.get()).request_id == "two"
        group.drain_requests.assert_awaited_once()
        async with gate.submission("one") as allowed:
            assert not allowed


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [TimeoutError("drain timed out"), RuntimeError("rank failed")])
async def test_uncertain_drain_never_admits_next_request(failure):
    async with admission() as (gate, group, submitted, task):
        group.drain_requests.side_effect = failure
        gate.enqueue(request("one"))
        gate.enqueue(request("two"))
        await submitted.get()
        gate.finish(["one"], aborted=True)
        with pytest.raises(type(failure), match=str(failure)):
            await task
        assert submitted.empty() and gate.active == "one"
        group.fail.assert_called_once()


@pytest.mark.asyncio
async def test_companions_wait_for_their_queued_parent():
    async with admission() as (gate, _, submitted, _):
        gate.enqueue(request("one"))
        gate.enqueue(request("two"))
        companion = AddCompanionRequestMessage(
            companion_id="two-negative",
            parent_id="two",
            role="negative",
            prompt=None,
            companion_prompt_text=None,
            sampling_params_list=[],
        )
        assert gate.enqueue(companion)
        assert (await submitted.get()).request_id == "one"
        gate.finish(["one"], aborted=False)
        assert (await submitted.get()).request_id == "two"
        assert await submitted.get() is companion


@pytest.mark.asyncio
async def test_dispatch_failure_freezes_group_and_shutdown_does_not_admit_pending():
    async def fail(gate, message):
        raise RuntimeError("submit failed")

    async with admission(fail) as (gate, group, submitted, task):
        gate.enqueue(request("one"))
        gate.enqueue(request("two"))
        with pytest.raises(RuntimeError, match="submit failed"):
            await task
        assert (await submitted.get()).request_id == "one"
        assert submitted.empty()
        group.fail.assert_called_once()
        assert [m.request_id for m in gate.discard(list(gate.pending))] == ["two"]
