# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Native cancellation acknowledgements must release central admission slots."""

import asyncio
from contextlib import suppress
from types import SimpleNamespace

import pytest
import zmq

from tests.engine.test_central_scheduling_admission import _message
from vllm_omni.diffusion.data import DiffusionRequestAbortedError
from vllm_omni.diffusion.inline_stage_diffusion_client import InlineStageDiffusionClient
from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient
from vllm_omni.distributed.omni_connectors.utils.serialization import OmniMsgpackDecoder, OmniMsgpackEncoder
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.metrics.definitions import DIFFUSION_SCHEDULER_WAITING_KEY

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion, pytest.mark.asyncio]
pytest_plugins = [
    "tests.diffusion.test_stage_diffusion_proc",
    "tests.engine.test_central_scheduling_admission",
]


class AbortingEngine:
    def __init__(self, metrics):
        self.metrics = metrics

    async def step_streaming(self, request):
        error = DiffusionRequestAbortedError(f"{request.request_id} cancelled")
        error.diffusion_metrics = self.metrics
        raise error
        yield  # Preserve the engine's async-generator interface.


@pytest.fixture
async def cancelled_runtime(runtime):
    orch, pool, (backend,), _ = runtime()
    await orch._handle_add_request(_message("A"))
    admission = orch._central_scheduling_admission_tasks["A"]
    assert await asyncio.wait_for(backend.started.get(), 1) == "A"
    backend.finish_submit.set()
    await admission
    await orch._handle_add_request(_message("B"))
    for _ in range(3):
        await asyncio.sleep(0)
    await orch._cleanup_request_ids(["A"], abort=True)
    assert pool._central_scheduler.active_count == pool._central_scheduler.pending_count == 1
    assert backend.submitted == ["A"]
    return orch, pool, backend


async def assert_waiter_resumes(orch, pool, backend, metrics):
    output = pool.poll_diffusion_output(0)
    assert output.request_id == "A" and output.finished and output.aborted
    assert output.error is None and output.metrics == metrics
    assert orch._absorb_diffusion_metrics(0, 0, output)
    assert orch.output_async_queue.empty()
    assert await asyncio.wait_for(backend.started.get(), 1) == "B"
    assert pool._central_scheduler.pending_count == 0
    assert pool._central_scheduler.active_count == 1
    assert "A" not in pool._central_scheduling_submitted
    # Replaying A's acknowledgement cannot release B's execution slot.
    backend.outputs.put_nowait(output)
    pool.poll_diffusion_output(0)
    assert pool._central_scheduler.active_count == 1
    assert pool.get_bound_replica_id("B") == 0
    assert backend.submitted == ["A", "B"]


@pytest.mark.parametrize("metrics", [{}, {DIFFUSION_SCHEDULER_WAITING_KEY: 0}], ids=["no-metrics", "with-metrics"])
async def test_inline_cancel_acknowledgement_releases_central_slot(cancelled_runtime, metrics):
    orch, pool, backend = cancelled_runtime
    client = object.__new__(InlineStageDiffusionClient)
    client._engine = AbortingEngine(metrics)
    client.od_config = SimpleNamespace(streaming_output=False)
    client._output_queue = backend.outputs
    client._tasks = {}
    await client._dispatch_request("A", {}, OmniDiffusionSamplingParams())
    await assert_waiter_resumes(orch, pool, backend, metrics)


@pytest.mark.parametrize("metrics", [{}, {DIFFUSION_SCHEDULER_WAITING_KEY: 0}], ids=["no-metrics", "with-metrics"])
async def test_process_cancel_acknowledgement_releases_central_slot(
    cancelled_runtime, proc_control_loop, mocker, metrics
):
    orch, pool, backend = cancelled_runtime
    proc, submit, outgoing, *_ = proc_control_loop
    proc._od_config = SimpleNamespace(streaming_output=False)
    proc._engine.step_streaming = AbortingEngine(metrics).step_streaming
    loop_task = asyncio.create_task(proc.run_loop("test-input", "test-output"))
    try:
        submit({"type": "add_request", "request_id": "A", "prompt": {}, "sampling_params": {}})
        message = await asyncio.wait_for(outgoing.get(), 1)
        assert message["type"] == "result"
        client = object.__new__(StageDiffusionClient)
        client._response_socket = mocker.Mock()
        client._response_socket.recv.side_effect = [OmniMsgpackEncoder().encode(message), zmq.Again()]
        client._decoder = OmniMsgpackDecoder()
        client._output_queue = backend.outputs
        client._drain_responses()
        await assert_waiter_resumes(orch, pool, backend, metrics)
    finally:
        loop_task.cancel()
        with suppress(asyncio.CancelledError):
            await loop_task
