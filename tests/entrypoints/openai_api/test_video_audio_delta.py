# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Preserve every sample from the engine's fresh DELTA audio payloads."""

import asyncio
import base64
import io
import wave
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import FinishReason

from tests.helpers.serving_chat import build_serving_chat
from vllm_omni.entrypoints.omni_base import OmniBase
from vllm_omni.entrypoints.openai.serving_video_stream import QwenOmniStreamingVideoHandler
from vllm_omni.entrypoints.openai.video_stream_base import (
    _CODEC_FRAME_SAMPLES,
    OmniStreamingVideoHandler,
    StreamingVideoSessionConfig,
)
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.outputs.mm_outputs import MultimodalCompletionOutput, MultimodalPayload
from vllm_omni.outputs.output_processor import OmniRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _audio_result(payload):
    completion = MultimodalCompletionOutput(
        index=0,
        text="",
        token_ids=[],
        cumulative_logprob=None,
        logprobs=None,
        multimodal_output=MultimodalPayload.from_dict({"audio": payload}),
    )
    return OmniRequestOutput(
        outputs=[completion],
        final_output_type="audio",
    )


def _pcm(encoded):
    assert encoded is not None
    with wave.open(io.BytesIO(base64.b64decode(encoded)), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (24000, 1, 2)
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")


def _chunks():
    # Exactly representable PCM16 values; distinct ordered samples catch loss,
    # repetition, reordering, and trimming more than once.
    return [torch.arange(start, start + 5000, dtype=torch.float32) / 32768 for start in (0, 5000, 10000)]


@pytest.mark.parametrize("mode", ["fast", "slow"])
@pytest.mark.parametrize("shape", ["tensor", "singleton_list", "batched_list"])
def test_fresh_delta_samples_are_forwarded_once(monkeypatch, mode, shape):
    monkeypatch.setenv("VLLM_VIDEO_AUDIO_DELTA_MODE", mode)
    chunks = _chunks()
    payloads = chunks if shape == "tensor" else [[chunk] for chunk in chunks]
    if shape == "batched_list":
        payloads = [[chunks[0]], chunks[1:]]
    seen = 0
    pcm = []
    for payload in payloads:
        encoded, seen = OmniStreamingVideoHandler._extract_audio_delta_b64(_audio_result(payload), seen)
        pcm.append(_pcm(encoded))
    np.testing.assert_array_equal(np.concatenate(pcm), np.arange(_CODEC_FRAME_SAMPLES, 15000))
    assert seen == 3


@pytest.mark.parametrize("mode", ["fast", "slow"])
@pytest.mark.parametrize("empty", [None, [], torch.empty(0), [torch.empty(0)]])
def test_empty_delta_does_not_consume_first_emission(monkeypatch, mode, empty):
    monkeypatch.setenv("VLLM_VIDEO_AUDIO_DELTA_MODE", mode)
    encoded, seen = OmniStreamingVideoHandler._extract_audio_delta_b64(_audio_result(empty), 0)
    assert encoded is None
    assert seen == 0
    encoded, seen = OmniStreamingVideoHandler._extract_audio_delta_b64(_audio_result(_chunks()[0]), seen)
    np.testing.assert_array_equal(_pcm(encoded), np.arange(_CODEC_FRAME_SAMPLES, 5000))
    assert seen == 1
    encoded, after = OmniStreamingVideoHandler._extract_audio_delta_b64(_audio_result(empty), seen)
    assert encoded is None
    assert after == seen


@pytest.mark.parametrize("first_length", [1, 3840, 3841])
def test_first_chunk_trim_threshold_is_preserved(first_length):
    first = torch.arange(first_length, dtype=torch.float32) / 32768
    later = torch.arange(5000, 10000, dtype=torch.float32).reshape(1, -1) / 32768
    encoded, seen = OmniStreamingVideoHandler._delta_fast(first, 0)
    start = _CODEC_FRAME_SAMPLES if first_length > 2 * _CODEC_FRAME_SAMPLES else 0
    np.testing.assert_array_equal(_pcm(encoded), np.arange(start, first_length))
    encoded, seen = OmniStreamingVideoHandler._delta_fast(later, seen)
    np.testing.assert_array_equal(_pcm(encoded), np.arange(5000, 10000))
    assert seen == 2


@pytest.mark.parametrize("explicit", [False, True])
def test_only_omitted_sampling_parameters_are_coerced(explicit):
    base = object.__new__(OmniBase)
    base.engine = SimpleNamespace(num_stages=1)
    params = SamplingParams(output_kind=RequestOutputKind.CUMULATIVE)
    base.default_sampling_params_list = [params]
    base.sampling_constraints_list = [{}]
    resolved = base.resolve_sampling_params_list([params] if explicit else None, allow_delta_coercion=True)
    expected = RequestOutputKind.CUMULATIVE if explicit else RequestOutputKind.DELTA
    assert resolved[0].output_kind == expected
    assert params.output_kind == RequestOutputKind.CUMULATIVE


def _state(output_kind):
    return OmniRequestState(
        request_id="audio-delta",
        external_req_id="audio-delta",
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
        output_kind=output_kind,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("wire_mode", ["on", "off"])
@pytest.mark.parametrize("delta_mode", ["fast", "slow"])
@pytest.mark.parametrize("batch_later_chunks", [False, True])
@pytest.mark.parametrize("finish_with_payload", [False, True])
@pytest.mark.parametrize("explicit_sampling", [False, True])
@pytest.mark.parametrize("lengths", [(5000, 5000, 5000), (0, 5000, 0, 2000, 5000)])
async def test_real_producer_deltas_reach_video_client(
    monkeypatch, wire_mode, delta_mode, batch_later_chunks, finish_with_payload, explicit_sampling, lengths
):
    monkeypatch.setenv("VLLM_VIDEO_ASYNC_CHUNK", wire_mode)
    monkeypatch.setenv("VLLM_VIDEO_AUDIO_DELTA_MODE", delta_mode)
    chunks = []
    offset = 0
    for length in lengths:
        # Qwen's generation runner hands off [1, samples] CPU tensors.
        chunks.append(torch.arange(offset, offset + length, dtype=torch.float32).reshape(1, -1) / 32768)
        offset += length
    groups = [[chunks[0]], chunks[1:]] if batch_later_chunks else [[chunk] for chunk in chunks]
    state = _state(RequestOutputKind.DELTA)
    snapshots = []
    produced = []
    # Complete production before consumption: retained DELTA snapshots must
    # survive subsequent accumulation and draining under consumer backpressure.
    for index, group in enumerate(groups):
        for chunk in group:
            state.add_multimodal_tensor(chunk, mm_type="audio")
        finish = FinishReason.STOP if finish_with_payload and index == len(groups) - 1 else None
        result = state.make_request_output([], None, finish, None)
        assert result is not None
        assert "audio" not in state.mm_accumulated
        snapshots.append(result.outputs[0].multimodal_output["audio"])
        produced.append(OmniRequestOutput.from_stage_output(result, final_output_type="audio"))
    if not finish_with_payload:
        result = state.make_request_output([], None, FinishReason.STOP, None)
        assert result is not None
        produced.append(OmniRequestOutput.from_stage_output(result, final_output_type="audio"))

    class Engine:
        async def generate(self, *, prompt, request_id, output_modalities, sampling_params_list=None):
            if explicit_sampling:
                assert sampling_params_list is not None
                assert sampling_params_list[0].output_kind == RequestOutputKind.DELTA
            else:
                assert sampling_params_list is None
            # AsyncOmni coerces omitted parameters to DELTA as well.
            for result in produced:
                yield result

    class Handler(QwenOmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            return {"prompt": "describe"}

    class WebSocket:
        def __init__(self):
            self.sent = []

        async def send_json(self, message):
            self.sent.append(message)

    ws = WebSocket()
    handler = Handler(chat_service=build_serving_chat(), engine_client=Engine())
    config = StreamingVideoSessionConfig(
        model="test",
        modalities=["text", "audio"],
        sampling_params_list=[{"temperature": 0, "output_kind": 0}] if explicit_sampling else None,
    )
    await handler._process_query_engine(
        ws,
        config,
        [],
        bytearray(),
        [],
        "describe",
        "audio-delta",
        asyncio.Event(),
        {},
    )
    assert not any(message["type"] == "error" for message in ws.sent)
    audio = [message for message in ws.sent if message["type"] == "response.output_audio.delta"]
    nonempty_groups = sum(any(chunk.numel() for chunk in group) for group in groups)
    assert len(audio) == (nonempty_groups if wire_mode == "on" else 1)
    np.testing.assert_array_equal(
        np.concatenate([_pcm(message["data"]) for message in audio]),
        np.arange(_CODEC_FRAME_SAMPLES, sum(lengths)),
    )
    assert sum(message["type"] == "response.output_audio.done" for message in ws.sent) == 1
    assert torch.equal(snapshots[0], chunks[0])
    last = torch.cat(snapshots[-1], dim=-1) if isinstance(snapshots[-1], list) else snapshots[-1]
    assert torch.equal(last, torch.cat(groups[-1], dim=-1))


def test_cumulative_producer_is_a_different_contract():
    # The fix must not change the producer globally: explicit CUMULATIVE users
    # still receive accumulated history, not fresh deltas.
    state = _state(RequestOutputKind.CUMULATIVE)
    chunks = _chunks()
    for chunk in chunks:
        state.add_multimodal_tensor(chunk, mm_type="audio")
        result = state.make_request_output([], None, None, None)
        assert result is not None
        assert "audio" in state.mm_accumulated
    payload = result.outputs[0].multimodal_output["audio"]
    assert isinstance(payload, torch.Tensor)
    assert torch.equal(payload, torch.cat(chunks))
