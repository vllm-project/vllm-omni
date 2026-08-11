from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.stage.inline_stage_diffusion_client import InlineStageDiffusionClient
from vllm_omni.diffusion.stage.stage_diffusion_core_client import StageDiffusionCoreClient
from vllm_omni.engine.stage.stage_core_types import StageDiffusionCoreOutput, StageDiffusionCoreRequest
from vllm_omni.engine.stage_init_utils import StageMetadata
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def mock_engine():
    with patch("vllm_omni.diffusion.stage.inline_stage_diffusion_client.DiffusionEngine") as mock:
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
        c = InlineStageDiffusionClient(model="test_model", od_config=od_config, metadata=metadata, batch_size=1)
        yield c
        c.shutdown()


@pytest.mark.asyncio
async def test_inline_dispatch_request_success(client, mock_engine):
    mock_result = OmniRequestOutput.from_diffusion(request_id="req-1", images=[MagicMock()])

    async def _step_streaming(_request):
        yield [mock_result]

    mock_engine.step_streaming = _step_streaming

    await client.add_request_async(
        StageDiffusionCoreRequest(request_id="req-1", prompt="A test prompt", sampling_params={})
    )

    # Wait for the task to be processed
    batch = None
    for _ in range(10):
        batch = client.get_outputs_nowait()
        if batch is not None:
            break
        await asyncio.sleep(0.01)

    assert batch is not None
    assert len(batch.outputs) == 1
    core_output = batch.outputs[0]
    assert core_output.request_id == "req-1"
    assert core_output.output is mock_result


@pytest.mark.asyncio
async def test_inline_preserves_advanced_generators(client, mock_engine):
    """Regression: inline single-stage requests must keep the original
    ``OmniDiffusionSamplingParams`` (per-output generator list + advanced
    generator state + ``modules``), not the lossy dict the wire path uses.

    The pool routes the dataclass straight through for inline clients, so
    ``add_request_async`` receives it in ``StageDiffusionCoreRequest`` and hands
    it to the engine unchanged.
    """
    # Two distinct per-output generators; advance the first so it carries state
    # beyond its initial seed (what a single-seed collapse would discard).
    gen0 = torch.Generator().manual_seed(111)
    gen1 = torch.Generator().manual_seed(222)
    torch.randn(4, generator=gen0)  # advance gen0 past its initial seed
    modules = {"vae": object()}
    params = OmniDiffusionSamplingParams(
        seed=None,
        generator=[gen0, gen1],
        modules=modules,
    )

    # Sanity-check the wire conversion IS lossy (why inline must bypass it):
    # the generator list collapses to a single seed and ``modules`` is dropped.
    wire = StageDiffusionCoreClient.sampling_params_to_dict(params)
    assert "generator" not in wire
    assert "modules" not in wire
    assert wire["seed"] == 111  # first generator's initial seed only

    captured: dict[str, object] = {}

    async def _step_streaming(request):
        captured["request"] = request
        yield [OmniRequestOutput.from_diffusion(request_id="req-gen", images=[MagicMock()])]

    mock_engine.step_streaming = _step_streaming

    # Inline path: the pool passes the dataclass object (not a dict) through.
    await client.add_request_async(
        StageDiffusionCoreRequest(request_id="req-gen", prompt="A test prompt", sampling_params=params)
    )

    for _ in range(10):
        if client.get_outputs_nowait() is not None:
            break
        await asyncio.sleep(0.01)

    engine_params = captured["request"].sampling_params
    # The full per-output generator list survived, in order, with state intact.
    assert engine_params.generator == [gen0, gen1]
    assert engine_params.generator[0] is gen0
    assert engine_params.generator[1] is gen1
    # And the subprocess-local ``modules`` were preserved, not stripped.
    assert engine_params.modules is modules


@pytest.mark.asyncio
async def test_pool_routes_object_to_inline_and_dict_to_wire(client):
    """The fix point: ``StageReplicaPool._diffusion_add_request`` must pass the
    original sampling-params object to inline clients and the serialized dict to
    the out-of-process client."""
    from vllm_omni.engine.stage.stage_replica_pool import StageReplicaPool

    params = OmniDiffusionSamplingParams(seed=None, generator=[torch.Generator().manual_seed(5)])
    fake_pool = SimpleNamespace(stage_id=0)

    # Inline branch (real InlineStageDiffusionClient): dataclass passes straight through.
    captured_inline: dict[str, object] = {}

    async def _inline_add(request):
        captured_inline["sp"] = request.sampling_params

    client.add_request_async = _inline_add
    await StageReplicaPool._diffusion_add_request(fake_pool, client, "r1", "p", params, None)
    assert captured_inline["sp"] is params

    # Out-of-process branch: serialized to a plain dict (generator stripped).
    wire_client = MagicMock(spec=StageDiffusionCoreClient)
    captured_wire: dict[str, object] = {}

    async def _wire_add(request):
        captured_wire["sp"] = request.sampling_params

    wire_client.add_request_async = _wire_add
    await StageReplicaPool._diffusion_add_request(fake_pool, wire_client, "r2", "p", params, None)
    assert isinstance(captured_wire["sp"], dict)
    assert "generator" not in captured_wire["sp"]


@pytest.mark.asyncio
async def test_inline_accepts_plain_dict_sampling_params(client, mock_engine):
    """Robustness: the inline client still accepts the plain-dict wire form and
    reconstructs the dataclass in-process (out-of-process path compatibility)."""
    captured: dict[str, object] = {}

    async def _step_streaming(request):
        captured["request"] = request
        yield [OmniRequestOutput.from_diffusion(request_id="req-dict", images=[MagicMock()])]

    mock_engine.step_streaming = _step_streaming

    await client.add_request_async(
        StageDiffusionCoreRequest(request_id="req-dict", prompt="A test prompt", sampling_params={"seed": 7})
    )

    for _ in range(10):
        if client.get_outputs_nowait() is not None:
            break
        await asyncio.sleep(0.01)

    engine_params = captured["request"].sampling_params
    assert isinstance(engine_params, OmniDiffusionSamplingParams)
    assert engine_params.seed == 7


@pytest.mark.asyncio
async def test_inline_requests_clone_shared_sampling_params(client, mock_engine):
    requests = []

    async def _step_streaming(request):
        requests.append(request)
        yield [OmniRequestOutput.from_diffusion(request_id=request.request_id, images=[MagicMock()])]

    mock_engine.step_streaming = _step_streaming

    shared = OmniDiffusionSamplingParams()
    await client.add_request_async(
        StageDiffusionCoreRequest(request_id="req-1", prompt="First prompt", sampling_params=shared)
    )
    await client.add_request_async(
        StageDiffusionCoreRequest(request_id="req-2", prompt="Second prompt", sampling_params=shared)
    )

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

    await client.add_request_async(
        StageDiffusionCoreRequest(request_id="req-stream", prompt="A test prompt", sampling_params={})
    )

    collected: list[StageDiffusionCoreOutput] = []
    for _ in range(20):
        batch = client.get_outputs_nowait()
        if batch is not None:
            collected.extend(batch.outputs)
            if any(o.finished for o in batch.outputs):
                break
        await asyncio.sleep(0.01)

    assert [o.output for o in collected] == chunks


@pytest.mark.asyncio
async def test_inline_dispatch_request_error(client, mock_engine):
    async def _step_streaming(_request):
        raise RuntimeError("Engine failure")
        yield  # pragma: no cover

    mock_engine.step_streaming = _step_streaming

    await client.add_request_async(
        StageDiffusionCoreRequest(request_id="req-err", prompt="A test prompt", sampling_params={})
    )

    batch = None
    for _ in range(10):
        batch = client.get_outputs_nowait()
        if batch is not None:
            break
        await asyncio.sleep(0.01)

    assert batch is not None
    core_output = batch.outputs[0]
    assert core_output.request_id == "req-err"
    assert core_output.error == "Engine failure"
    assert core_output.output is None


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
        client.get_outputs_nowait()


def test_inline_returns_queued_output_before_engine_dead(client, mock_engine):
    callback = mock_engine.executor.register_failure_callback.call_args.args[0]
    core_output = StageDiffusionCoreOutput(
        request_id="req-queued",
        finished=True,
        output=OmniRequestOutput.from_diffusion(request_id="req-queued", images=[]),
    )
    client._output_queue.put_nowait(core_output)

    callback()

    batch = client.get_outputs_nowait()
    assert batch is not None
    assert batch.outputs == [core_output]
    with pytest.raises(EngineDeadError, match="inline diffusion engine is dead"):
        client.get_outputs_nowait()


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
                batch_size=1,
            )
