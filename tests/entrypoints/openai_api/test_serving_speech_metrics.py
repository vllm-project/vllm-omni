# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Prometheus metric coverage for the Speech API audio path."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _MetricsStub:
    def __init__(self) -> None:
        self.ttfp_calls: list[tuple[str, str, float]] = []
        self.underrun_calls: list[tuple[str, str, float]] = []
        self.continuity_calls: list[tuple[str, str, int]] = []

    def observe_audio_ttfp(self, stage: str, replica: str, seconds: float) -> None:
        self.ttfp_calls.append((stage, replica, seconds))

    def observe_audio_underrun(self, stage: str, replica: str, seconds: float) -> None:
        self.underrun_calls.append((stage, replica, seconds))

    def inc_audio_continuity_ok(self, stage: str, replica: str, threshold_ms: int) -> None:
        self.continuity_calls.append((stage, replica, threshold_ms))


def _serving(metrics: _MetricsStub) -> OmniOpenAIServingSpeech:
    serving = OmniOpenAIServingSpeech.__new__(OmniOpenAIServingSpeech)
    serving._tts_model_type = "qwen3_tts"
    serving.engine_client = SimpleNamespace(mod_metrics=metrics, request_states={})
    serving.create_audio = lambda audio_obj: SimpleNamespace(
        audio_data=b"\0\0" * int(audio_obj.audio_tensor.size),
        media_type="audio/pcm",
    )
    serving._mark_ref_audio_artifact_ready_for_request = lambda request_id: None
    serving._discard_ref_audio_artifact_warmup = lambda request_id: None
    return serving


def _result(samples: int = 320, *, stage_id: int = 1, replica_id: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        multimodal_output={"audio": torch.zeros(samples), "sr": 16000},
        stage_id=stage_id,
        replica_id=replica_id,
    )


async def _generate(*results):
    for result in results:
        yield result


@pytest.mark.asyncio
async def test_streaming_speech_observes_ttfp_once_on_first_pcm_payload(monkeypatch):
    metrics = _MetricsStub()
    serving = _serving(metrics)
    monkeypatch.setattr("vllm_omni.entrypoints.openai.serving_speech.time.time", lambda: 100.25)

    chunks = serving._generate_audio_chunks(
        _generate(_result(), _result()),
        request_id="speech-test",
        request_arrival_ts=100.0,
    )
    assert len([chunk async for chunk in chunks]) == 2
    assert metrics.ttfp_calls == [("1", "2", pytest.approx(0.25))]
    assert metrics.underrun_calls == [("1", "2", pytest.approx(0.0))]
    assert metrics.continuity_calls == [("1", "2", 100)]


@pytest.mark.asyncio
async def test_streaming_speech_does_not_count_empty_payload_as_first_packet(monkeypatch):
    metrics = _MetricsStub()
    serving = _serving(metrics)
    monkeypatch.setattr("vllm_omni.entrypoints.openai.serving_speech.time.time", lambda: 100.25)

    chunks = serving._generate_audio_chunks(
        _generate(_result(samples=0), _result()),
        request_id="speech-test",
        request_arrival_ts=100.0,
    )
    assert len([chunk async for chunk in chunks]) == 2
    assert metrics.ttfp_calls == [("1", "2", pytest.approx(0.25))]
    assert metrics.underrun_calls == [("1", "2", pytest.approx(0.0))]
    assert metrics.continuity_calls == [("1", "2", 100)]


@pytest.mark.asyncio
async def test_streaming_speech_does_not_finalize_continuity_on_error(monkeypatch):
    metrics = _MetricsStub()
    serving = _serving(metrics)
    monkeypatch.setattr("vllm_omni.entrypoints.openai.serving_speech.time.time", lambda: 100.25)

    async def failing_stream():
        yield _result()
        raise RuntimeError("stream failed")

    chunks = serving._generate_audio_chunks(
        failing_stream(),
        request_id="speech-test",
        request_arrival_ts=100.0,
    )
    with pytest.raises(RuntimeError, match="stream failed"):
        _ = [chunk async for chunk in chunks]

    assert len(metrics.ttfp_calls) == 1
    assert metrics.underrun_calls == []
    assert metrics.continuity_calls == []


@pytest.mark.asyncio
async def test_streaming_speech_reports_late_chunk_as_underrun(monkeypatch):
    metrics = _MetricsStub()
    serving = _serving(metrics)
    monkeypatch.setattr("vllm_omni.entrypoints.openai.serving_speech.time.time", lambda: 100.25)
    perf_times = iter((0.0, 0.1, 0.1, 2.0, 2.0))
    monkeypatch.setattr("vllm_omni.entrypoints.openai.serving_speech.time.perf_counter", lambda: next(perf_times))

    chunks = serving._generate_audio_chunks(
        _generate(_result(), _result()),
        request_id="speech-test",
        request_arrival_ts=100.0,
    )
    assert len([chunk async for chunk in chunks]) == 2
    assert metrics.underrun_calls[0][:2] == ("1", "2")
    assert metrics.underrun_calls[0][2] > 0.1
    assert metrics.continuity_calls == []


@pytest.mark.asyncio
async def test_non_streaming_speech_does_not_observe_ttfp():
    metrics = _MetricsStub()
    serving = _serving(metrics)
    serving._audio_encode_speed = lambda _request: 1.0

    async def prepare(_request, **_kwargs):
        result = _result()
        result.metrics = {}
        return "speech-test", _generate(result), {}

    serving._prepare_speech_generation = prepare
    request = OpenAICreateSpeechRequest(input="hello", response_format="pcm")

    audio_data, media_type = await serving._generate_audio_bytes(request, request_arrival_ts=100.0)

    assert audio_data
    assert media_type == "audio/pcm"
    assert metrics.ttfp_calls == []
