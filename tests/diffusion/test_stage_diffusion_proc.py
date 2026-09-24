# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

import vllm_omni.diffusion.stage_diffusion_proc as stage_diffusion_proc
import vllm_omni.plugins as omni_plugins
from vllm_omni.diffusion.data import DIFFUSION_REQUEST_LIFECYCLE_KEY, DIFFUSION_REQUEST_STARTED
from vllm_omni.diffusion.stage_diffusion_proc import StageDiffusionProc
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@dataclass
class MockOmniRequestOutput:
    request_id: str = ""
    status: str = "success"


BASE_HEIGHT = 512
BASE_WIDTH = 512
BASE_INFER_STEPS = 10
DELAY_BASE = 0.01


def test_run_diffusion_proc_sets_lifecycle_before_loading_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class StopProcessError(Exception):
        pass

    class TestStageDiffusionProc(StageDiffusionProc):
        def __init__(self, model, od_config):
            events.append("proc")
            raise StopProcessError

    monkeypatch.setattr(omni_plugins, "load_omni_general_plugins", lambda: events.append("plugins"))
    monkeypatch.setattr(stage_diffusion_proc, "set_death_signal", lambda _: events.append("death_signal"))
    monkeypatch.setattr(stage_diffusion_proc.signal, "signal", lambda *_: events.append("signal_handler"))

    with pytest.raises(StopProcessError):
        TestStageDiffusionProc.run_diffusion_proc(
            model="test-model",
            od_config=None,
            handshake_address="test-address",
            local_client=True,
            headless=False,
        )

    assert events == ["death_signal", "signal_handler", "signal_handler", "plugins", "proc"]


class MockDiffusionEngine:
    async def step_streaming(self, request):
        def simulate_step_delay(height, width, num_inference_steps) -> float:
            return (height / BASE_HEIGHT) * (width / BASE_WIDTH) * (num_inference_steps / BASE_INFER_STEPS)

        DELAY_BASE = 0.01
        delay_scale = simulate_step_delay(
            request.sampling_params.height, request.sampling_params.width, request.sampling_params.num_inference_steps
        )
        delay = DELAY_BASE + delay_scale * DELAY_BASE
        await asyncio.sleep(delay)
        yield [MockOmniRequestOutput(request_id=request.request_id)]


@pytest.mark.asyncio
async def test_proc_streaming_request_yields_each_engine_chunk():
    """Ensure that the streaming output chunks from DiffusionEngine reaches StageDiffusionProc"""
    captured = {}
    chunks = [
        OmniRequestOutput.from_diffusion(request_id="", images=[], finished=False),
        OmniRequestOutput.from_diffusion(request_id="", images=[], finished=True),
    ]

    class _StreamingEngine:
        async def step_streaming(self, request):
            captured["request"] = request
            for chunk in chunks:
                yield [chunk]

    stage_proc = object.__new__(StageDiffusionProc)
    stage_proc._engine = _StreamingEngine()

    outputs = [
        output
        async for output in stage_proc._process_streaming_request(
            request_id="req-stream",
            prompt="prompt",
            sampling_params_dict=asdict(OmniDiffusionSamplingParams()),
            kv_sender_info={0: {"host": "127.0.0.1"}},
        )
    ]

    assert outputs == chunks
    assert [output.request_id for output in outputs] == ["req-stream", "req-stream"]
    assert [output.finished for output in outputs] == [False, True]
    assert captured["request"].kv_sender_info == {0: {"host": "127.0.0.1"}}


@pytest.mark.asyncio
async def test_proc_non_streaming_forwards_lifecycle_before_final_output():
    lifecycle = OmniRequestOutput.from_diffusion(
        request_id="",
        images=[],
        custom_output={DIFFUSION_REQUEST_LIFECYCLE_KEY: DIFFUSION_REQUEST_STARTED},
        finished=False,
    )
    intermediate = OmniRequestOutput.from_diffusion(
        request_id="",
        images=[],
        custom_output={"chunk": 0},
        finished=False,
    )
    final = OmniRequestOutput.from_diffusion(request_id="", images=[], finished=True)

    class _LifecycleEngine:
        async def step_streaming(self, request):
            del request
            yield [lifecycle]
            yield [intermediate]
            yield [final]

    stage_proc = object.__new__(StageDiffusionProc)
    stage_proc._engine = _LifecycleEngine()
    intermediate_outputs = []

    async def _capture(output):
        intermediate_outputs.append(output)

    result = await stage_proc._process_request(
        request_id="req-lifecycle",
        prompt="prompt",
        sampling_params_dict=asdict(OmniDiffusionSamplingParams()),
        on_request_started=_capture,
    )

    assert intermediate_outputs == [lifecycle]
    assert lifecycle.request_id == "req-lifecycle"
    assert result is final
    assert result.request_id == "req-lifecycle"


@pytest.mark.asyncio
async def test_proc_process_request_with_batching_async_output():
    stage_proc = object.__new__(StageDiffusionProc)
    stage_proc._engine = MockDiffusionEngine()

    test_requests = [
        {
            "request_id": "req_1",
            "prompt": "prompt1",
            "params": {"height": BASE_HEIGHT * 2, "width": BASE_WIDTH * 2, "num_inference_steps": BASE_INFER_STEPS * 1},
        },
        {
            "request_id": "req_2",
            "prompt": "prompt2",
            "params": {"height": BASE_HEIGHT * 2, "width": BASE_WIDTH * 2, "num_inference_steps": BASE_INFER_STEPS * 2},
        },
        {
            "request_id": "req_3",
            "prompt": "prompt3",
            "params": {"height": BASE_HEIGHT * 2, "width": BASE_WIDTH * 2, "num_inference_steps": BASE_INFER_STEPS * 3},
        },
    ]

    async def run_task(req_data):
        start_time = time.time()
        result = await stage_proc._process_request(
            request_id=req_data["request_id"], prompt=req_data["prompt"], sampling_params_dict=req_data["params"]
        )
        end_time = time.time()
        return result, end_time - start_time

    coros = [run_task(req) for req in test_requests]
    results = await asyncio.gather(*coros)

    assert len(results) == len(test_requests)
    base_time = DELAY_BASE
    time_gap_std = DELAY_BASE * 2 * 2 * 1  # height/width/steps infer time scale
    eps = 0.1
    for i, (res, elapsed_time) in enumerate(results):
        assert res.request_id == test_requests[i]["request_id"]
        assert isinstance(res, MockOmniRequestOutput)
        time_gap = elapsed_time - base_time
        assert time_gap > time_gap_std - eps and time_gap < time_gap_std + eps
        base_time = elapsed_time


@pytest.fixture
def proc_control_loop(monkeypatch: pytest.MonkeyPatch):
    incoming: asyncio.Queue[bytes] = asyncio.Queue()
    outgoing: asyncio.Queue[dict | bytes] = asyncio.Queue()
    encoder = stage_diffusion_proc.OmniMsgpackEncoder()
    decoder = stage_diffusion_proc.OmniMsgpackDecoder()
    request_socket = MagicMock()
    response_socket = MagicMock()
    request_socket.recv = lambda: asyncio.ensure_future(incoming.get())

    async def send(payload):
        assert not response_socket.close.called
        await outgoing.put(payload if payload == StageDiffusionProc.DIFFUSION_PROC_DEAD else decoder.decode(payload))

    response_socket.send = AsyncMock(side_effect=send)
    context = MagicMock()
    context.socket.side_effect = [request_socket, response_socket]
    monkeypatch.setattr(stage_diffusion_proc.zmq.asyncio, "Context", lambda: context)
    proc = StageDiffusionProc("test-model", None)
    proc._engine = MagicMock()
    proc._engine.executor.is_dead = False
    proc._engine.update_streaming_playback.return_value = None
    proc._executor = ThreadPoolExecutor(max_workers=1)

    def submit(message):
        incoming.put_nowait(encoder.encode(message))

    yield proc, submit, outgoing, request_socket, response_socket, context
    proc._executor.shutdown(wait=True, cancel_futures=True)


@pytest.mark.asyncio
async def test_proc_rpc_wait_allows_playback_and_abort_in_order(proc_control_loop):
    proc, submit, outgoing, request_socket, response_socket, context = proc_control_loop
    entered = threading.Event()
    release = threading.Event()
    aborted = asyncio.Event()
    calls = []

    def rpc(method, *_args):
        calls.append(method)
        if method == "submit_interaction":
            entered.set()
            assert release.wait(timeout=5)
        return method

    proc._engine.collective_rpc.side_effect = rpc
    proc._engine.abort.side_effect = lambda _rid: aborted.set()
    task = asyncio.create_task(proc.run_loop("request", "response"))
    try:
        submit({"type": "collective_rpc", "rpc_id": "interaction", "method": "submit_interaction"})
        assert await asyncio.to_thread(entered.wait, 1)
        submit({"type": "collective_rpc", "rpc_id": "second", "method": "list_adapters"})
        submit(
            {
                "type": "collective_rpc",
                "rpc_id": "feedback",
                "method": "update_streaming_playback",
                "args": ["video", 0.5],
            }
        )
        assert await asyncio.wait_for(outgoing.get(), 1) == {
            "type": "rpc_result",
            "rpc_id": "feedback",
            "result": None,
        }
        proc._engine.update_streaming_playback.assert_called_once_with("video", 0.5)
        submit({"type": "abort", "request_ids": ["video"]})
        await asyncio.wait_for(aborted.wait(), 1)
        proc._engine.abort.assert_called_once_with("video")
        assert calls == ["submit_interaction"]

        release.set()
        replies = [await asyncio.wait_for(outgoing.get(), 1) for _ in range(2)]
        assert [reply["rpc_id"] for reply in replies] == ["interaction", "second"]
        assert calls == ["submit_interaction", "list_adapters"]
        submit({"type": "shutdown"})
        await asyncio.wait_for(task, 1)
        request_socket.close.assert_called_once()
        response_socket.close.assert_called_once()
        context.term.assert_called_once()
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_proc_shutdown_cancels_waiting_rpc_before_closing_sockets(proc_control_loop, monkeypatch):
    proc, submit, outgoing, request_socket, response_socket, _context = proc_control_loop
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def rpc(*_args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    def close():
        assert cancelled.is_set()

    monkeypatch.setattr(proc, "_handle_collective_rpc", rpc)
    request_socket.close.side_effect = close
    response_socket.close.side_effect = close
    task = asyncio.create_task(proc.run_loop("request", "response"))
    try:
        submit({"type": "collective_rpc", "rpc_id": "interaction", "method": "submit_interaction"})
        await asyncio.wait_for(entered.wait(), 1)
        submit({"type": "shutdown"})
        await asyncio.wait_for(task, 1)
        assert cancelled.is_set()
        assert outgoing.empty()
        assert proc._active_tasks is None
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_proc_full_rpc_queue_rejects_without_blocking_abort(proc_control_loop, monkeypatch):
    proc, submit, outgoing, _request_socket, _response_socket, _context = proc_control_loop
    entered = asyncio.Event()
    aborted = asyncio.Event()
    calls = []

    async def rpc(method, *_args):
        calls.append(method)
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(proc, "_handle_collective_rpc", rpc)
    proc._engine.abort.side_effect = lambda _rid: aborted.set()
    task = asyncio.create_task(proc.run_loop("request", "response"))
    try:
        submit({"type": "collective_rpc", "rpc_id": "interaction", "method": "submit_interaction"})
        await asyncio.wait_for(entered.wait(), 1)
        for index in range(65):
            submit({"type": "collective_rpc", "rpc_id": str(index), "method": "list_adapters"})
        assert await asyncio.wait_for(outgoing.get(), 1) == {
            "type": "error",
            "rpc_id": "64",
            "error": "Too many pending collective RPCs",
        }
        submit({"type": "abort", "request_ids": ["video"]})
        await asyncio.wait_for(aborted.wait(), 1)
        submit({"type": "shutdown"})
        await asyncio.wait_for(task, 1)
        assert calls == ["submit_interaction"]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_proc_rpc_engine_death_wakes_recv_without_new_messages(proc_control_loop):
    proc, submit, outgoing, request_socket, response_socket, context = proc_control_loop
    proc._engine.executor.is_dead = True
    proc._engine.collective_rpc.side_effect = RuntimeError("DiffusionExecutor is closed")
    task = asyncio.create_task(proc.run_loop("request", "response"))
    try:
        submit({"type": "collective_rpc", "rpc_id": "interaction", "method": "submit_interaction"})
        with pytest.raises(RuntimeError, match="executor reported permanent failure"):
            await asyncio.wait_for(task, 1)
        assert outgoing.get_nowait() == {
            "type": "error",
            "rpc_id": "interaction",
            "error": "DiffusionExecutor is closed",
        }
        assert outgoing.get_nowait() == StageDiffusionProc.DIFFUSION_PROC_DEAD
        request_socket.close.assert_called_once()
        response_socket.close.assert_called_once()
        context.term.assert_called_once()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
