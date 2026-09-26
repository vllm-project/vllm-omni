# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import FinishReason

from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs
from vllm_omni.engine.cfg_companion_tracker import CfgCompanionTracker
from vllm_omni.engine.messages import ErrorMessage, OutputMessage
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState
from vllm_omni.outputs.output_processor import MultimodalOutputProcessor, OmniRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _CodecPool:
    final_output = True

    def __init__(self, output_kind, registered, replica_id):
        self.output_processor = MultimodalOutputProcessor(
            tokenizer=None, log_stats=False, engine_core_output_type="audio"
        )
        self.output_kind = output_kind
        self.replica_id = replica_id
        self.bound_replica = None
        if registered:
            self.register()

    def register(self):
        self.bound_replica = self.replica_id
        self.output_processor.request_states["r"] = OmniRequestState(
            request_id="r",
            external_req_id="r",
            parent_req=None,
            request_index=0,
            lora_request=None,
            prompt=None,
            prompt_token_ids=[0],
            prompt_embeds=None,
            logprobs_processor=None,
            detokenizer=None,
            max_tokens_param=None,
            arrival_time=0.0,
            queue=None,
            log_stats=False,
            stream_interval=1,
            output_kind=self.output_kind,
        )
        self.output_processor.external_req_ids["r"].append("r")

    def get_bound_replica_id(self, request_id):
        return self.bound_replica

    async def process_llm_raw_outputs(self, replica_id, raw, **kwargs):
        assert replica_id == self.replica_id
        return self.output_processor.process_outputs(raw.outputs, raw.timestamp).request_outputs


def _orchestrator(output_kind=RequestOutputKind.DELTA, *, registered=True, replica_id=2, final_stage_id=1):
    obj = Orchestrator.__new__(Orchestrator)
    obj.stage_pools = [SimpleNamespace(final_output=False), _CodecPool(output_kind, registered, replica_id)]
    obj.request_states = {"r": OrchestratorRequestState(request_id="r", final_stage_id=final_stage_id)}
    obj.output_async_queue = asyncio.Queue()
    obj._cfg_tracker = CfgCompanionTracker()
    obj._pd_kv_params = {}
    obj._running_counter = None
    obj._abort_request_ids = AsyncMock(return_value=[])
    obj._release_request_bindings = Mock()
    obj._finish_raw_terminal_requests = AsyncMock()

    async def process(stage, replica, raw, terminals):
        await obj._route_upstream_first_audio(stage, replica, raw)
        return await obj.stage_pools[stage].process_llm_raw_outputs(replica, raw)

    async def route(stage, replica, outputs):
        for output in outputs:
            await obj.output_async_queue.put(
                OutputMessage(
                    request_id=output.request_id,
                    stage_id=stage,
                    replica_id=replica,
                    engine_outputs=output,
                    metrics=None,
                    finished=output.finished,
                )
            )

    obj._process_llm_stage_outputs = process
    obj._handle_processed_outputs = route
    return obj


def _raw(*, first=False, required=False, terminal=False, samples=(1, 1, 1, 1)):
    return OmniEngineCoreOutputs(
        outputs=[
            OmniEngineCoreOutput(
                request_id="r",
                new_token_ids=[],
                finish_reason=FinishReason.STOP if terminal else None,
                multimodal_output={
                    "model_outputs": torch.tensor(samples, dtype=torch.float32),
                    "sr": torch.tensor(24000),
                    "_omni_first_audio": torch.tensor(first),
                    "_omni_first_audio_required": torch.tensor(required),
                },
            )
        ]
    )


async def _codec_output(obj, raw):
    replica = obj.stage_pools[1].replica_id
    processed = await obj._process_llm_stage_outputs(1, replica, raw, set())
    await obj._handle_processed_outputs(1, replica, processed)


def _messages(obj):
    messages = []
    while not obj.output_async_queue.empty():
        messages.append(obj.output_async_queue.get_nowait())
    return messages


def _audio(output):
    audio = output.outputs[0].multimodal_output["audio"]
    return torch.cat(audio) if isinstance(audio, list) else audio


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(RequestOutputKind))
@pytest.mark.parametrize("codec_overtook", [False, True])
async def test_first_audio_obeys_codec_output_kind(kind, codec_overtook):
    obj = _orchestrator(kind)
    suffix = _raw(required=True, samples=(20, 21))
    terminal = _raw(terminal=True, samples=(30, 31))
    if codec_overtook:
        await _codec_output(obj, suffix)
        await _codec_output(obj, terminal)
        assert obj.output_async_queue.empty()
    await obj._route_upstream_first_audio(0, 0, _raw(first=True, samples=(10, 11)))
    if kind == RequestOutputKind.FINAL_ONLY and not codec_overtook:
        assert obj.output_async_queue.empty()
    if not codec_overtook:
        await _codec_output(obj, suffix)
        await _codec_output(obj, terminal)
    messages = _messages(obj)
    # The source replica is zero; output ownership belongs to codec replica two.
    assert all((message.stage_id, message.replica_id) == (1, 2) for message in messages)
    actual = [_audio(message.engine_outputs).tolist() for message in messages]
    expected = {
        RequestOutputKind.DELTA: [[10, 11], [20, 21], [30, 31]],
        RequestOutputKind.CUMULATIVE: [[10, 11], [10, 11, 20, 21], [10, 11, 20, 21, 30, 31]],
        RequestOutputKind.FINAL_ONLY: [[10, 11, 20, 21, 30, 31]],
    }
    assert actual == expected[kind]
    assert messages[-1].finished
    assert obj.request_states["r"].pending_first_audio_outputs == []
    await obj._route_upstream_first_audio(0, 0, _raw(first=True))
    assert obj.output_async_queue.empty()


@pytest.mark.asyncio
async def test_first_audio_waits_for_codec_registration_and_ignores_duplicates():
    obj = _orchestrator(registered=False)
    req_state = obj.request_states["r"]
    await obj._route_upstream_first_audio(0, 0, _raw(first=True, samples=(10, 11)))
    await obj._route_upstream_first_audio(0, 0, _raw(first=True, samples=(90, 91)))
    assert obj.output_async_queue.empty()
    assert not req_state.upstream_first_audio
    assert req_state.pending_upstream_first_audio is not None
    obj.stage_pools[1].register()
    await obj._flush_upstream_first_audio(req_state)
    messages = _messages(obj)
    assert len(messages) == 1
    assert _audio(messages[0].engine_outputs).tolist() == [10, 11]
    assert req_state.upstream_first_audio
    assert req_state.pending_upstream_first_audio is None


@pytest.mark.asyncio
@pytest.mark.parametrize("processing_failed", [False, True])
async def test_prewarm_flushes_first_audio_after_successful_submission(mocker, processing_failed):
    obj = _orchestrator(registered=False, final_stage_id=2 if processing_failed else 1)
    req_state = obj.request_states["r"]
    req_state.sampling_params_list = [None] * (req_state.final_stage_id + 1)
    await obj._route_upstream_first_audio(0, 0, _raw(first=True, samples=(10, 11)))
    pool = obj.stage_pools[1]
    pool.stage_type = "llm"
    pool.stage_vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=None))
    obj.stage_pools[0].get_bound_replica_id = lambda request_id: 0
    obj._stage_receives_async_chunks = lambda stage: True
    obj._build_payload_sender_info = Mock(return_value={})
    obj._on_stage_submitted = Mock()
    obj._emit_tx_edge = Mock()
    mocker.patch(
        "vllm_omni.engine.orchestrator.build_engine_core_request_from_tokens",
        return_value=SimpleNamespace(request_id="r"),
    )

    async def submit(*args, **kwargs):
        pool.register()

    pool.submit_initial = submit
    if processing_failed:
        pool.process_llm_raw_outputs = AsyncMock(side_effect=RuntimeError("processing failed"))
        later_pool = SimpleNamespace(submit_initial=AsyncMock())
        obj.stage_pools.append(later_pool)
    submitted = await obj._prewarm_async_chunk_stages("r", SimpleNamespace(prompt_token_ids=[0]), req_state)
    assert submitted is not processing_failed
    messages = _messages(obj)
    assert len(messages) == 1
    if processing_failed:
        assert isinstance(messages[0], ErrorMessage)
        later_pool.submit_initial.assert_not_called()
        assert "r" not in obj.request_states
    else:
        assert _audio(messages[0].engine_outputs).tolist() == [10, 11]


@pytest.mark.asyncio
async def test_pending_first_audio_is_discarded_on_cancellation():
    obj = _orchestrator(registered=False)
    req_state = obj.request_states["r"]
    await obj._route_upstream_first_audio(0, 0, _raw(first=True))
    await obj._cleanup_request_ids(["r"], abort=True)
    obj.stage_pools[1].register()
    await obj._flush_upstream_first_audio(req_state)
    assert obj.output_async_queue.empty()


@pytest.mark.asyncio
async def test_cancellation_while_routing_first_audio_does_not_release_codec_outputs():
    obj = _orchestrator()
    await obj._route_upstream_first_audio(1, 2, _raw(required=True, terminal=True))

    async def cancel(stage, replica, outputs):
        await obj._cleanup_request_ids(["r"], abort=True)

    obj._handle_processed_outputs = cancel
    obj._process_llm_stage_outputs = AsyncMock()
    await obj._route_upstream_first_audio(0, 0, _raw(first=True))
    obj._process_llm_stage_outputs.assert_not_called()
    assert obj.output_async_queue.empty()


@pytest.mark.asyncio
async def test_request_ending_at_source_does_not_emit_first_audio():
    obj = _orchestrator(final_stage_id=0)
    raw = _raw(first=True)
    await obj._route_upstream_first_audio(0, 0, raw)
    assert not raw.outputs and obj.output_async_queue.empty()
    assert obj.request_states["r"].pending_upstream_first_audio is None


@pytest.mark.asyncio
async def test_later_aligner_receives_full_codec_waveform():
    from vllm_omni.model_executor.stage_input_processors.forced_aligner import code2wav2aligner

    obj = _orchestrator(RequestOutputKind.CUMULATIVE, final_stage_id=2)
    await obj._route_upstream_first_audio(0, 0, _raw(first=True, samples=(10, 11)))
    await _codec_output(obj, _raw(required=True, terminal=True, samples=(20, 21)))
    messages = _messages(obj)
    assert all(message.stage_id == 1 for message in messages)
    inputs = code2wav2aligner(
        [messages[-1].engine_outputs], prompt={"additional_information": {"text": ["Hello world"]}}
    )
    assert inputs[0]["multi_modal_data"]["audio"][0].tolist() == [10, 11, 20, 21]


@pytest.mark.asyncio
async def test_first_audio_processing_failure_fails_only_its_request():
    obj = _orchestrator()
    obj.request_states["healthy"] = OrchestratorRequestState(request_id="healthy", final_stage_id=1)
    obj.stage_pools[1].process_llm_raw_outputs = AsyncMock(side_effect=RuntimeError("processing failed"))
    await obj._route_upstream_first_audio(0, 0, _raw(first=True))
    error = obj.output_async_queue.get_nowait()
    assert isinstance(error, ErrorMessage)
    assert error.request_id == "r" and error.stage_id == 1
    assert "processing failed" in error.error
    assert set(obj.request_states) == {"healthy"}


@pytest.mark.asyncio
async def test_unmarked_upstream_audio_and_regular_codec_outputs_are_untouched():
    obj = _orchestrator()
    for stage in (0, 1):
        raw = _raw()
        await obj._route_upstream_first_audio(stage, 0, raw)
        assert len(raw.outputs) == 1
        assert torch.equal(raw.outputs[0].multimodal_output["model_outputs"], torch.ones(4))
    assert obj.output_async_queue.empty()


@pytest.mark.asyncio
async def test_cancelled_request_does_not_receive_late_first_audio():
    obj = _orchestrator()
    obj.request_states.clear()
    raw = _raw(first=True)
    await obj._route_upstream_first_audio(0, 0, raw)
    assert not raw.outputs and obj.output_async_queue.empty()


@pytest.mark.asyncio
async def test_first_audio_supports_both_speech_and_chat_consumers():
    from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

    obj = _orchestrator()
    await obj._route_upstream_first_audio(0, 0, _raw(first=True, samples=(10, 11)))
    result = obj.output_async_queue.get_nowait().engine_outputs
    assert not result.finished
    assert _audio(result).tolist() == [10, 11]
    payload, key = OmniOpenAIServingSpeech._extract_audio_output(result)
    assert payload is not None and key is not None
    audio = payload[key]
    audio = torch.cat(audio) if isinstance(audio, list) else audio
    assert audio.tolist() == [10, 11]


@pytest.mark.asyncio
@pytest.mark.parametrize("audio", [None, [], torch.empty(0)])
@pytest.mark.parametrize("codec_overtook", [False, True])
async def test_invalid_first_audio_reports_error_and_releases_request(audio, codec_overtook):
    obj = _orchestrator()
    obj.request_states["healthy"] = OrchestratorRequestState(request_id="healthy", final_stage_id=1)
    if codec_overtook:
        await obj._route_upstream_first_audio(1, 2, _raw(required=True, terminal=True))
        assert obj.request_states["r"].pending_first_audio_outputs
    first = _raw(first=True)
    first.outputs[0].multimodal_output["model_outputs"] = audio
    await obj._route_upstream_first_audio(0, 0, first)

    error = obj.output_async_queue.get_nowait()
    assert isinstance(error, ErrorMessage)
    assert error.request_id == "r" and "first-audio" in error.error
    assert not first.outputs and set(obj.request_states) == {"healthy"}
    obj._abort_request_ids.assert_awaited_once_with(["r"])
    obj._release_request_bindings.assert_called_once_with(["r"])
    await obj._route_upstream_first_audio(0, 0, _raw(first=True))
    assert obj.output_async_queue.empty()
