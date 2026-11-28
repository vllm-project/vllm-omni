# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.diffusion.data import (
    DIFFUSION_REQUEST_LIFECYCLE_KEY,
    DIFFUSION_REQUEST_STARTED,
    OmniDiffusionConfig,
)
from vllm_omni.diffusion.inline_stage_diffusion_client import InlineStageDiffusionClient
from vllm_omni.engine.stage_init_utils import StageMetadata
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def mock_engine():
    with patch("vllm_omni.diffusion.inline_stage_diffusion_client.DiffusionEngine") as mock:
        engine_instance = MagicMock()
        engine_instance.executor = SimpleNamespace(
            register_failure_callback=MagicMock(),
            check_health=MagicMock(),
        )
        mock.make_engine.return_value = engine_instance
        yield engine_instance


@pytest.fixture
def client(mock_engine):
    metadata = StageMetadata(
        stage_id=0,
        stage_type="diffusion",
        engine_output_type="image",
        is_comprehension=False,
        requires_multimodal_data=False,
        engine_input_source="prompt",
        final_output=True,
        final_output_type="image",
        default_sampling_params={},
        custom_process_input_func=None,
        model_stage=None,
        runtime_cfg=None,
    )
    with patch.object(InlineStageDiffusionClient, "_enrich_config"):
        od_config = MagicMock(spec=OmniDiffusionConfig)
        od_config.streaming_output = False
        c = InlineStageDiffusionClient(model="test_model", od_config=od_config, metadata=metadata)
        yield c
        c.shutdown()


def test_inline_client_implements_shared_prompt_hook_protocol(client):
    assert client.prompt_transform_func is None
    assert client.prompt_expand_func is None


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "timeout", "cancel", "error"])
async def test_inline_drain_preserves_owned_generation_and_terminal_output(client, mock_engine, outcome):
    entered, aborted, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def step_streaming(request):
        entered.set()
        await finish.wait()
        if outcome == "error":
            raise RuntimeError("Generation failed")
        yield [OmniRequestOutput.from_diffusion(request_id=request.request_id, images=[])]

    mock_engine.step_streaming = step_streaming
    mock_engine.abort.side_effect = lambda request_id: aborted.set()
    await client.add_request_async("owned", "prompt", OmniDiffusionSamplingParams())
    generation = client._tasks["owned"]
    await asyncio.wait_for(entered.wait(), 1)
    drain = asyncio.create_task(
        client.collective_rpc_async("weight_sharing_drain", timeout=0 if outcome == "timeout" else 2)
    )
    try:
        await asyncio.wait_for(aborted.wait(), 1)
        mock_engine.abort.assert_called_once_with("owned")
        if outcome in ("timeout", "cancel"):
            if outcome == "cancel":
                drain.cancel()
            with pytest.raises(TimeoutError if outcome == "timeout" else asyncio.CancelledError):
                await drain
            assert not generation.done()
        else:
            assert not drain.done()
        finish.set()
        await asyncio.wait_for(generation, 1)
        if outcome in ("success", "error"):
            assert await drain is True
        assert client._tasks == {}
        output = client.get_diffusion_output_nowait()
        assert output.request_id == "owned"
        assert output.error == ("Generation failed" if outcome == "error" else None)
        mock_engine.collective_rpc.assert_not_called()
        assert await client.collective_rpc_async("weight_sharing_drain", timeout=0) is True
    finally:
        finish.set()
        await asyncio.gather(generation, drain, return_exceptions=True)


@pytest.mark.asyncio
async def test_inline_drain_waits_only_for_the_requested_sessions(client, mock_engine):
    finished = {rid: asyncio.Event() for rid in ("old", "current")}

    async def step_streaming(request):
        await finished[request.request_id].wait()
        yield [OmniRequestOutput.from_diffusion(request_id=request.request_id, images=[])]

    mock_engine.step_streaming = step_streaming
    mock_engine.abort.side_effect = lambda rid: finished[rid].set()
    for request_id in finished:
        await client.add_request_async(request_id, "prompt", OmniDiffusionSamplingParams())
    tasks = tuple(client._tasks.values())
    try:
        result = await client.collective_rpc_async("weight_sharing_drain", timeout=2, kwargs={"request_ids": ["old"]})
        assert result is True
        mock_engine.abort.assert_called_once_with("old")
        assert set(client._tasks) == {"current"}
        assert not client._tasks["current"].done()
    finally:
        for event in finished.values():
            event.set()
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "rejected", "cancelled"])
async def test_inline_restore_acks_staging_and_routes_owned_output(client, mock_engine, outcome):
    entered, stage, resume, cleaned = (asyncio.Event() for _ in range(4))
    snapshots = [{"saved": b"snapshot"}]
    output = OmniRequestOutput.from_diffusion(request_id="saved", images=[])

    @asynccontextmanager
    async def restore(payload, *, policy_ref, max_bytes, timeout):
        assert (payload, policy_ref, max_bytes, timeout) == (snapshots, "v2", 128, 3)
        entered.set()
        try:
            await stage.wait()
            if outcome == "rejected":
                raise ValueError("policy mismatch")

            async def stream():
                await resume.wait()
                yield [output]

            yield stream()
        finally:
            cleaned.set()

    mock_engine.restore_step_streaming = restore
    control = asyncio.create_task(
        client.restore_step_request_async("saved", snapshots, policy_ref="v2", max_bytes=128, timeout=3)
    )
    await asyncio.wait_for(entered.wait(), 1)
    task = client._tasks["saved"]
    try:
        assert not control.done()
        assert client.get_diffusion_output_nowait() is None
        if outcome == "cancelled":
            control.cancel()
            with pytest.raises(asyncio.CancelledError):
                await control
        else:
            stage.set()
            if outcome == "rejected":
                with pytest.raises(ValueError, match="policy mismatch"):
                    await control
            else:
                await asyncio.wait_for(control, 1)
                assert not task.done()
                with pytest.raises(ValueError, match="already has an output task"):
                    await client.restore_step_request_async(
                        "saved", snapshots, policy_ref="v2", max_bytes=128, timeout=3
                    )
                resume.set()
                await asyncio.wait_for(task, 1)
        assert cleaned.is_set()
        assert not client._tasks
        assert client.get_diffusion_output_nowait() is (output if outcome == "success" else None)
        assert client.get_diffusion_output_nowait() is None
        mock_engine.step_streaming.assert_not_called()
    finally:
        stage.set()
        resume.set()
        await asyncio.gather(control, task, return_exceptions=True)


@pytest.mark.asyncio
async def test_inline_dispatch_request_success(client, mock_engine):
    mock_result = OmniRequestOutput.from_diffusion(request_id="req-1", images=[MagicMock()])

    async def _step_streaming(_request):
        yield [mock_result]

    mock_engine.step_streaming = _step_streaming

    sampling_params = OmniDiffusionSamplingParams()
    await client.add_request_async("req-1", "A test prompt", sampling_params)

    # Wait for the task to be processed
    for _ in range(10):
        output = client.get_diffusion_output_nowait()
        if output is not None:
            break
        await asyncio.sleep(0.01)

    assert output is not None
    assert output.request_id == "req-1"


@pytest.mark.parametrize("streaming", [False, True])
def test_inline_dispatch_preserves_payload_sender_info(client, mock_engine, streaming):
    async def _run_test():
        requests = []

        async def _step_streaming(request):
            requests.append(request)
            yield [OmniRequestOutput.from_diffusion(request_id=request.request_id, images=[MagicMock()])]

        mock_engine.step_streaming = _step_streaming
        client.od_config.streaming_output = streaming
        payload_sender_info = {"host": "10.0.0.1", "zmq_port": 50071}
        kv_transfer_params = {"remote_engine_id": "mooncake-producer", "remote_block_ids": [1, 2]}
        await client.add_request_async(
            "req-payload",
            "A test prompt",
            OmniDiffusionSamplingParams(),
            kv_transfer_params=kv_transfer_params,
            payload_sender_info=payload_sender_info,
        )

        for _ in range(10):
            if requests:
                break
            await asyncio.sleep(0.01)

        assert requests[0].payload_sender_info == payload_sender_info
        assert requests[0].kv_transfer_params == kv_transfer_params

    asyncio.run(_run_test())


@pytest.mark.asyncio
async def test_inline_non_streaming_dispatches_lifecycle_before_final(client, mock_engine):
    lifecycle = OmniRequestOutput.from_diffusion(
        request_id="req-lifecycle",
        images=[],
        custom_output={DIFFUSION_REQUEST_LIFECYCLE_KEY: DIFFUSION_REQUEST_STARTED},
        finished=False,
    )
    intermediate = OmniRequestOutput.from_diffusion(
        request_id="req-lifecycle",
        images=[],
        custom_output={"chunk": 0},
        finished=False,
    )
    final = OmniRequestOutput.from_diffusion(request_id="req-lifecycle", images=[MagicMock()])

    async def _step_streaming(_request):
        yield [lifecycle]
        yield [intermediate]
        yield [final]

    mock_engine.step_streaming = _step_streaming

    await client.add_request_async("req-lifecycle", "A test prompt", OmniDiffusionSamplingParams())

    outputs = []
    for _ in range(20):
        output = client.get_diffusion_output_nowait()
        if output is not None:
            outputs.append(output)
            if output.finished:
                break
        await asyncio.sleep(0.01)

    assert outputs == [lifecycle, final]


@pytest.mark.asyncio
async def test_inline_requests_clone_shared_sampling_params(client, mock_engine):
    requests = []

    async def _step_streaming(request):
        requests.append(request)
        yield [OmniRequestOutput.from_diffusion(request_id=request.request_id, images=[MagicMock()])]

    mock_engine.step_streaming = _step_streaming

    shared = OmniDiffusionSamplingParams()
    await client.add_request_async("req-1", "First prompt", shared)
    await client.add_request_async("req-2", "Second prompt", shared)

    for _ in range(10):
        if len(requests) == 2:
            break
        await asyncio.sleep(0.01)

    assert len(requests) == 2
    assert requests[0].sampling_params is not requests[1].sampling_params
    assert requests[0].sampling_params.guidance_scale_provided is False
    assert requests[1].sampling_params.guidance_scale_provided is False
    assert shared.guidance_scale is None


@pytest.mark.asyncio
async def test_inline_dispatch_request_streaming_success(client, mock_engine):
    """Inline StageDiffusionClient correctly receives streaming chunk output from Diffusion Engine"""
    chunks = [
        OmniRequestOutput.from_diffusion(request_id="req-stream", images=[], finished=False),
        OmniRequestOutput.from_diffusion(request_id="req-stream", images=[], finished=True),
    ]

    async def _step_streaming(_request):
        for chunk in chunks:
            yield [chunk]

    client.od_config.streaming_output = True
    mock_engine.step_streaming = _step_streaming

    await client.add_request_async("req-stream", "A test prompt", OmniDiffusionSamplingParams())

    outputs = []
    for _ in range(20):
        output = client.get_diffusion_output_nowait()
        if output is not None:
            outputs.append(output)
            if output.finished:
                break
        await asyncio.sleep(0.01)

    assert outputs == chunks


@pytest.mark.asyncio
async def test_inline_dispatch_request_error(client, mock_engine):
    async def _step_streaming(_request):
        raise RuntimeError("Engine failure")
        yield  # pragma: no cover

    mock_engine.step_streaming = _step_streaming

    sampling_params = OmniDiffusionSamplingParams()
    await client.add_request_async("req-err", "A test prompt", sampling_params)

    for _ in range(10):
        output = client.get_diffusion_output_nowait()
        if output is not None:
            break
        await asyncio.sleep(0.01)

    assert output is not None
    assert output.request_id == "req-err"
    assert output.error == "Engine failure"
    assert not output.images


def test_inline_shutdown(client, mock_engine):
    assert not client._shutting_down

    # Shutting down should cleanly cancel anything queued and close engine
    client.shutdown()

    assert client._shutting_down
    mock_engine.close.assert_called_once()


def test_inline_shutdown_runs_after_orchestrator_premark(client, mock_engine):
    # The orchestrator pre-marks clients to suppress false worker-death errors
    # before StagePool invokes the actual teardown.
    client._shutting_down = True

    client.shutdown()

    mock_engine.close.assert_called_once()
    assert client._shutdown_complete

    client.shutdown()
    mock_engine.close.assert_called_once()


def test_inline_registers_executor_failure_callback(client, mock_engine):
    mock_engine.executor.register_failure_callback.assert_called_once()


def test_inline_executor_failure_marks_engine_dead(client, mock_engine):
    callback = mock_engine.executor.register_failure_callback.call_args.args[0]

    callback()

    assert client._engine_dead is True
    with pytest.raises(EngineDeadError, match="inline diffusion engine is dead"):
        client.get_diffusion_output_nowait()


def test_inline_returns_queued_output_before_engine_dead(client, mock_engine):
    callback = mock_engine.executor.register_failure_callback.call_args.args[0]
    output = OmniRequestOutput.from_diffusion(request_id="req-queued", images=[])
    client._output_queue.put_nowait(output)

    callback()

    assert client.get_diffusion_output_nowait() is output
    with pytest.raises(EngineDeadError, match="inline diffusion engine is dead"):
        client.get_diffusion_output_nowait()


def test_inline_check_health_marks_engine_dead(client, mock_engine):
    mock_engine.executor.check_health.side_effect = EngineDeadError()

    with pytest.raises(EngineDeadError):
        client.check_health()

    assert client._engine_dead is True


def test_inline_client_requires_replica_id(mock_engine):
    metadata = SimpleNamespace(
        stage_id=0,
        final_output=True,
        final_output_type="image",
        default_sampling_params={},
        requires_multimodal_data=False,
        custom_process_input_func=None,
        engine_input_source=[],
    )
    with patch.object(InlineStageDiffusionClient, "_enrich_config"):
        od_config = MagicMock(spec=OmniDiffusionConfig)
        with pytest.raises(AttributeError, match="replica_id"):
            InlineStageDiffusionClient(
                model="test_model",
                od_config=od_config,
                metadata=metadata,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method", ["update_streaming_playback", "track_streaming_interaction", "release_streaming_output"]
)
async def test_playback_feedback_bypasses_a_waiting_worker_rpc(client, mock_engine, method):
    entered = threading.Event()
    release = threading.Event()
    feedback = []

    def rpc(method, timeout, args, kwargs, unique_reply_rank):
        if method == "submit_interaction":
            entered.set()
            assert release.wait(timeout=5)
        elif method == "update_streaming_playback":
            feedback.append(args)

    mock_engine.collective_rpc.side_effect = rpc
    mock_engine.update_streaming_playback.side_effect = lambda *args, **kwargs: feedback.append(args)
    mock_engine.track_streaming_interaction.side_effect = lambda *args, **kwargs: feedback.append(args)
    mock_engine.release_streaming_output.side_effect = lambda *args, **kwargs: feedback.append(args)
    interaction = asyncio.create_task(client.submit_interaction_async("video", {"event": {}}))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        await asyncio.wait_for(client.collective_rpc_async(method, args=("video", 0.5)), 1)
        assert feedback == [("video", 0.5)]
        assert not interaction.done()
    finally:
        release.set()
        await interaction
