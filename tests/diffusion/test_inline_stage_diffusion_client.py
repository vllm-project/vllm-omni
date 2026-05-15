from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.inline_stage_diffusion_client import InlineStageDiffusionClient
from vllm_omni.engine.stage_init_utils import StageMetadata
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def mock_engine():
    with patch("vllm_omni.diffusion.inline_stage_diffusion_client.DiffusionEngine") as mock:
        engine_instance = MagicMock()
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
        c = InlineStageDiffusionClient(model="test_model", od_config=od_config, metadata=metadata, batch_size=1)
        yield c
        c.shutdown()


@pytest.mark.asyncio
async def test_inline_dispatch_request_success(client, mock_engine):
    # Setup mock engine step to return a successful result
    mock_result = OmniRequestOutput.from_diffusion(request_id="req-1", images=[MagicMock()])
    mock_engine.step.return_value = [mock_result]

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
    mock_engine.step.assert_called_once()


@pytest.mark.asyncio
async def test_inline_dispatch_request_error(client, mock_engine):
    # Setup mock engine step to raise an exception
    mock_engine.step.side_effect = RuntimeError("Engine failure")

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
