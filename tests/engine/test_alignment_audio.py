# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio

import pytest
import torch
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.engine.cfg_companion_tracker import CfgCompanionTracker
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.model_executor.stage_input_processors.forced_aligner import code2wav2aligner
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.outputs.audio_accumulation import AudioChunkBuffer
from vllm_omni.outputs.mm_outputs import MultimodalCompletionOutput, MultimodalPayload

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def chunk(values, finished=False, rate=24000):
    return OmniRequestOutput(
        request_id="audio",
        finished=finished,
        outputs=[
            MultimodalCompletionOutput(
                index=0,
                text="",
                token_ids=[],
                cumulative_logprob=None,
                logprobs=None,
                multimodal_output=MultimodalPayload.from_dict({"audio": torch.tensor(values), "sr": rate}),
            )
        ],
    )


def setup_route(mocker, output_kind=RequestOutputKind.DELTA, model_stage="forced_aligner"):
    orchestrator = object.__new__(Orchestrator)
    orchestrator.async_chunk = True
    orchestrator._pd_pair = None
    orchestrator._cfg_tracker = CfgCompanionTracker()
    source = mocker.Mock(spec=StagePool)
    source.final_output = True
    target = mocker.Mock(spec=StagePool)
    target.stage_vllm_config = mocker.Mock()
    target.stage_vllm_config.model_config.model_stage = model_stage
    target.stage_vllm_config.model_config.async_chunk = False
    orchestrator.stage_pools = [source, target]
    orchestrator.output_async_queue = asyncio.Queue()
    orchestrator._forward_to_next_stage = mocker.AsyncMock()
    state = OrchestratorRequestState(
        request_id="audio",
        final_stage_id=1,
        final_output_stage_ids={0, 1},
        sampling_params_list=[SamplingParams(output_kind=output_kind)],
    )
    return orchestrator, state


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_final", [False, True])
async def test_route_preserves_delta_and_aligns_complete_audio(mocker, empty_final):
    orchestrator, state = setup_route(mocker)
    outputs = [chunk([0.1, 0.2]), chunk([0.3], finished=not empty_final)]
    if empty_final:
        outputs.append(chunk([], finished=True))
    for output in outputs:
        await orchestrator._route_output(0, 0, output, state, None)
    orchestrator._forward_to_next_stage.assert_awaited_once()
    forwarded = orchestrator._forward_to_next_stage.call_args.args[2]
    client_outputs = [orchestrator.output_async_queue.get_nowait().engine_outputs for _ in outputs]
    assert all(a is b for a, b in zip(client_outputs, outputs))
    expected = torch.cat([out.outputs[0].multimodal_output["audio"] for out in client_outputs])
    torch.testing.assert_close(forwarded.outputs[0].multimodal_output["audio"], expected)
    prompt = {"additional_information": {"text": ["Hello"]}}
    aligned_input = code2wav2aligner([forwarded], prompt)[0]
    waveform, rate = aligned_input["multi_modal_data"]["audio"]
    torch.testing.assert_close(torch.from_numpy(waveform), expected)
    assert rate == 24000
    assert not state.alignment_audio


@pytest.mark.parametrize(
    "kind,model_stage",
    [
        (RequestOutputKind.CUMULATIVE, "forced_aligner"),
        (RequestOutputKind.FINAL_ONLY, "forced_aligner"),
        (RequestOutputKind.DELTA, "other"),
    ],
)
def test_unrelated_paths_unchanged(mocker, kind, model_stage):
    orchestrator, state = setup_route(mocker, kind, model_stage)
    output = chunk([1.0], finished=True)
    assert orchestrator._assemble_alignment_audio(0, output, state, True) is output
    assert not state.alignment_audio


def test_request_isolation_and_segment_reset(mocker):
    orchestrator, first = setup_route(mocker)
    _, second = setup_route(mocker)
    orchestrator._assemble_alignment_audio(0, chunk([1.0]), first, False)
    orchestrator._assemble_alignment_audio(0, chunk([2.0]), second, False)
    one = orchestrator._assemble_alignment_audio(0, chunk([3.0], True), first, True)
    two = orchestrator._assemble_alignment_audio(0, chunk([4.0], True), second, True)
    torch.testing.assert_close(one.outputs[0].multimodal_output["audio"], torch.tensor([1.0, 3.0]))
    torch.testing.assert_close(two.outputs[0].multimodal_output["audio"], torch.tensor([2.0, 4.0]))
    next_segment = orchestrator._assemble_alignment_audio(0, chunk([5.0], True), first, True)
    torch.testing.assert_close(next_segment.outputs[0].multimodal_output["audio"], torch.tensor([5.0]))


def test_buffer_owns_audio_and_rejects_rate_change():
    buffer = AudioChunkBuffer()
    output = chunk([1.0])
    buffer.append(output, finished=False)
    output.outputs[0].multimodal_output["audio"].zero_()
    result = buffer.append(chunk([], True), finished=True)
    assert result.outputs[0].multimodal_output["audio"].item() == 1.0
    buffer.append(chunk([1.0]), finished=False)
    with pytest.raises(ValueError, match="sample rate changed"):
        buffer.append(chunk([2.0], rate=16000), finished=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("abort", [False, True])
async def test_cleanup_releases_partial_audio(mocker, abort):
    orchestrator, state = setup_route(mocker)
    orchestrator._assemble_alignment_audio(0, chunk([1.0]), state, False)
    orchestrator.request_states = {"audio": state}
    orchestrator._pd_kv_params = {}
    orchestrator._release_request_bindings = mocker.Mock()
    orchestrator._abort_request_ids = mocker.AsyncMock(return_value=[])
    orchestrator._running_counter = None
    await orchestrator._cleanup_request_ids(["audio"], abort=abort)
    assert not state.alignment_audio
    assert not orchestrator.request_states


@pytest.mark.asyncio
async def test_invalid_audio_fails_only_affected_request(mocker):
    orchestrator, state = setup_route(mocker)
    orchestrator._handle_stage_error = mocker.AsyncMock()
    await orchestrator._route_output(0, 0, chunk([1.0]), state, None)
    await orchestrator._route_output(0, 0, chunk([2.0], True, rate=16000), state, None)
    orchestrator._handle_stage_error.assert_awaited_once()
    error = orchestrator._handle_stage_error.call_args.args[1]
    assert error.request_id == state.request_id
    assert "sample rate changed" in error.error
    orchestrator._forward_to_next_stage.assert_not_awaited()


@pytest.mark.parametrize("rate", [None, [], torch.tensor([])])
def test_missing_sample_rate_uses_default(rate):
    result = AudioChunkBuffer().append(chunk([1.0], True, rate=rate), finished=True)
    assert result.outputs[0].multimodal_output["sr"] == 24000
