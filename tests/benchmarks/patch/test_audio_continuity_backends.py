# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Deterministic transport tests for chat continuity and measured-only rollups."""

import asyncio
import base64
import io
import json
import wave
from collections.abc import Callable

import pytest
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput
from vllm.benchmarks.serve import TaskType

from vllm_omni.benchmarks.metrics.metrics import calculate_metrics
from vllm_omni.benchmarks.patch import patch

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


def wav_chunk(rate=24000, duration=0.1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        writer.writeframes(bytes(round(rate * duration) * 2))
    return buffer.getvalue()


def sse_audio(data):
    payload = {"modality": "audio", "choices": [{"delta": {"content": base64.b64encode(data).decode()}}]}
    return ("data: " + json.dumps(payload) + "\n\n").encode()


class Response:
    status = 200
    headers: dict[str, str] = {}
    reason = ""

    def __init__(self, clock, chunks):
        self.clock = clock
        self.chunks = chunks
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def iter_any(self):
        for at, chunk in self.chunks:
            self.clock[0] = at
            yield chunk


class Session:
    """Minimal HTTP transport fake; every post returns a Response context manager."""

    def __init__(self, post: Callable[..., Response]) -> None:
        self.post = post


@pytest.mark.parametrize("gap,expected", [(0.05, 0.0), (0.5, 0.4)])
def test_chat_measures_decoded_audio_not_wav_headers(monkeypatch, gap, expected):
    clock = [0.0]
    monkeypatch.setattr(patch.time, "perf_counter", lambda: clock[0])
    encoded = sse_audio(wav_chunk())
    # First SSE event spans two network reads; arrival is its final byte.
    chunks = [(10.0, encoded[:20]), (10.1, encoded[20:]), (10.1 + gap, encoded), (20.0, b"data: [DONE]\n\n")]
    session = Session(post=lambda **kwargs: Response(clock, chunks))
    request = RequestFuncInput(
        model="test",
        model_name="test",
        prompt="hello",
        prompt_len=1,
        output_len=8,
        api_url="http://unused/v1/chat/completions",
    )
    output = asyncio.run(patch.async_request_openai_chat_omni_completions(request, session))
    assert output.success, output.error
    assert output.audio_duration == pytest.approx(0.2)
    (sample,) = output.audio_continuity_samples
    assert sample["audio_chunk_count"] == 2
    assert sample["audio_underrun_s"] == pytest.approx(expected)
    assert sample["audio_continuity_ok"] is (expected <= 0.1)


@pytest.mark.parametrize("chunks", [[], [b"broken-wav"], [wav_chunk(), wav_chunk(16000)]])
def test_chat_no_audio_or_invalid_wav_has_no_continuity_sample(monkeypatch, chunks):
    clock = [0.0]
    monkeypatch.setattr(patch.time, "perf_counter", lambda: clock[0])
    events = [(i + 1.0, sse_audio(chunk)) for i, chunk in enumerate(chunks)]
    events.append((10.0, b"data: [DONE]\n\n"))
    session = Session(post=lambda **kwargs: Response(clock, events))
    request = RequestFuncInput(
        model="test",
        model_name="test",
        prompt="hello",
        prompt_len=1,
        output_len=8,
        api_url="http://unused/v1/chat/completions",
    )
    output = asyncio.run(patch.async_request_openai_chat_omni_completions(request, session))
    assert output.success, output.error
    assert not output.audio_continuity_samples


def calculate(outputs, selected):
    return calculate_metrics(
        input_requests=[],
        outputs=outputs,
        dur_s=1.0,
        tokenizer=None,
        selected_percentiles=[50.0, 99.0],
        goodput_config_dict={},
        task_type=TaskType.GENERATION,
        selected_percentile_metrics=selected,
        max_concurrency=1,
        request_rate=float("inf"),
        benchmark_duration=1.0,
    )[0]


def test_rollup_weights_replies_and_excludes_unmeasured_and_failed(capsys):
    first = patch.MixRequestFuncOutput(success=True, audio_duration=1.0)
    first.audio_continuity_samples = [
        {"audio_underrun_s": 0.4, "audio_continuity_ok": False},
        {"audio_underrun_s": 0.0, "audio_continuity_ok": True},
    ]
    second = patch.MixRequestFuncOutput(success=True, audio_duration=1.0)
    second.audio_continuity_samples = [{"audio_underrun_s": 0.0, "audio_continuity_ok": True}]
    missing = patch.MixRequestFuncOutput(success=True, audio_duration=1.0)
    failed = patch.MixRequestFuncOutput(success=False, audio_continuity_samples=first.audio_continuity_samples)
    metrics = calculate([first, second, missing, failed], ["audio_underrun", "audio_continuity_rate"])
    assert metrics.num_audio_continuity_samples == 3
    assert metrics.audio_continuity_ok_rate == pytest.approx(2 / 3)
    assert metrics.mean_audio_underrun_s == pytest.approx(0.4 / 3)
    assert "66.67%" in capsys.readouterr().out


def test_unmeasured_rate_is_na_not_100_percent(capsys):
    metrics = calculate(
        [patch.MixRequestFuncOutput(success=True, audio_duration=1.0)], ["audio_underrun", "audio_continuity_ok_rate"]
    )
    assert metrics.num_audio_continuity_samples == 0
    assert metrics.audio_continuity_ok_rate is None
    printed = capsys.readouterr().out
    assert "N/A" in printed
    assert "100.00%" not in printed


def test_chat_retry_discards_previous_attempt_audio(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(patch.time, "perf_counter", lambda: clock[0])

    class BrokenResponse(Response):
        async def iter_any(self):
            async for chunk in super().iter_any():
                yield chunk
            raise patch.aiohttp.ClientError("connection lost")

    responses = iter(
        [
            BrokenResponse(clock, [(1.0, sse_audio(wav_chunk()))]),
            Response(
                clock, [(100.0, sse_audio(wav_chunk())), (100.05, sse_audio(wav_chunk())), (101.0, b"data: [DONE]\n\n")]
            ),
        ]
    )
    session = Session(post=lambda **kwargs: next(responses))
    request = RequestFuncInput(
        model="test",
        model_name="test",
        prompt="hello",
        prompt_len=1,
        output_len=8,
        api_url="http://unused/v1/chat/completions",
    )
    output = asyncio.run(patch.async_request_openai_chat_omni_completions(request, session))
    assert output.success, output.error
    (sample,) = output.audio_continuity_samples
    assert sample["audio_chunk_count"] == 2
    assert sample["audio_underrun_s"] == 0


def speech_request():
    return RequestFuncInput(
        model="test",
        model_name="test",
        prompt="hello",
        prompt_len=1,
        output_len=0,
        api_url="http://unused/v1/audio/speech",
    )


@pytest.mark.parametrize("rate,channels", [(16000, 1), (24000, 1), (48000, 1), (24000, 2)])
@pytest.mark.parametrize("gap,expected,continuous", [(0.05, 0.0, True), (0.2, 0.1, True), (0.5, 0.4, False)])
def test_speech_pcm_timeline_format_and_rollup(monkeypatch, rate, channels, gap, expected, continuous):
    clock = [0.0]
    monkeypatch.setattr(patch.time, "perf_counter", lambda: clock[0])
    monkeypatch.setenv("VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE", str(rate))
    monkeypatch.setenv("VLLM_OMNI_BENCH_AUDIO_CHANNELS", str(channels))
    monkeypatch.setenv("VLLM_OMNI_BENCH_AUDIO_CONTINUITY_THRESHOLD_S", "0.1")
    pcm = bytes(rate // 10 * channels * 2)
    # Empty reads, first-packet waiting, and waiting for EOF are excluded from
    # continuity. Each real read contains exactly 100 ms of raw PCM.
    chunks = [(1.0, b""), (10.0, pcm), (10.0 + gap, pcm), (100.0, b"")]
    posted = {}

    def post(**kwargs):
        posted.update(kwargs)
        return Response(clock, chunks)

    output = asyncio.run(patch.async_request_openai_audio_speech(speech_request(), Session(post=post)))
    assert output.success, output.error
    assert posted["json"]["stream"] is True
    assert posted["json"]["response_format"] == "pcm"
    assert output.audio_ttfp == pytest.approx(10.0)
    assert output.audio_duration == pytest.approx(0.2)
    assert output.audio_frames == rate // 5
    (sample,) = output.audio_continuity_samples
    assert sample["audio_chunk_count"] == 2
    assert sample["audio_underrun_s"] == pytest.approx(expected)
    assert sample["audio_continuity_ok"] is continuous
    assert output.audio_underrun_s == pytest.approx(expected)
    assert output.audio_continuity_ok is continuous
    metrics = calculate([output], ["audio_underrun", "audio_continuity_rate"])
    assert metrics.num_audio_continuity_samples == 1  # scalar + list must not double count
    assert metrics.mean_audio_underrun_s == pytest.approx(expected)
    assert metrics.audio_continuity_ok_rate == float(continuous)


@pytest.mark.parametrize("failure", ["empty", "http", "disconnect"])
def test_speech_empty_or_failed_stream_has_no_continuity_sample(monkeypatch, failure):
    clock = [0.0]
    monkeypatch.setattr(patch.time, "perf_counter", lambda: clock[0])

    class BrokenResponse(Response):
        async def iter_any(self):
            async for chunk in super().iter_any():
                yield chunk
            raise patch.aiohttp.ClientError("connection lost")

    response = Response(clock, [(1.0, b"")])
    if failure == "http":
        response.status = 500
        response.reason = "server error"
    elif failure == "disconnect":
        response = BrokenResponse(clock, [(1.0, bytes(4800))])
    session = Session(post=lambda **kwargs: response)
    output = asyncio.run(patch.async_request_openai_audio_speech(speech_request(), session))
    assert output.success is (failure == "empty")
    assert not output.audio_continuity_samples
    assert output.audio_underrun_s is None
    assert output.audio_continuity_ok is None
    metrics = calculate([output], ["audio_underrun", "audio_continuity_rate"])
    assert metrics.num_audio_continuity_samples == 0
    assert metrics.audio_continuity_ok_rate is None


@pytest.mark.parametrize(
    "invalid",
    [
        {"audio_underrun_s": None, "audio_continuity_ok": True},
        {"audio_underrun_s": -0.1, "audio_continuity_ok": True},
        {"audio_underrun_s": float("nan"), "audio_continuity_ok": True},
        {"audio_underrun_s": float("inf"), "audio_continuity_ok": False},
        {"audio_underrun_s": True, "audio_continuity_ok": True},
        {"audio_underrun_s": "0.1", "audio_continuity_ok": True},
        {"audio_underrun_s": 0.0, "audio_continuity_ok": None},
        {"audio_underrun_s": 0.0, "audio_continuity_ok": 1},
    ],
)
def test_rollup_invalid_samples_do_not_change_denominator_or_percentiles(invalid):
    valid = patch.MixRequestFuncOutput(
        success=True,
        audio_continuity_samples=[
            {"audio_underrun_s": 0.0, "audio_continuity_ok": True},
            {"audio_underrun_s": 0.05, "audio_continuity_ok": True},
            {"audio_underrun_s": 0.4, "audio_continuity_ok": False},
        ],
    )
    invalid_output = patch.MixRequestFuncOutput(success=True, audio_continuity_samples=[invalid])
    failed = patch.MixRequestFuncOutput(success=False, audio_continuity_samples=valid.audio_continuity_samples)
    metrics = calculate([valid, invalid_output, failed], ["audio_underrun", "audio_continuity_rate"])
    assert metrics.num_audio_continuity_samples == 3
    assert metrics.audio_continuity_ok_rate == pytest.approx(2 / 3)
    assert metrics.mean_audio_underrun_s == pytest.approx(0.15)
    assert metrics.median_audio_underrun_s == pytest.approx(0.05)
    assert dict(metrics.percentiles_audio_underrun_s)[99.0] == pytest.approx(0.393)


@pytest.mark.parametrize("samples,expected_count", [(None, 1), ([], 0)])
def test_scalar_fallback_requires_missing_sample_list(samples, expected_count):
    output = patch.MixRequestFuncOutput(
        success=True,
        audio_underrun_s=0.4,
        total_underrun_s=0.4,
        audio_underrun_event_count=1,
        audio_underrun_ratio=2 / 3,
        audio_underrun_events_per_minute=300,
        audio_continuity_ok=False,
        audio_continuity_samples=samples,
    )
    metrics = calculate([output], ["audio_underrun", "audio_continuity_rate"])
    assert metrics.num_audio_continuity_samples == expected_count
    assert metrics.audio_continuity_ok_rate == (0.0 if expected_count else None)
    assert metrics.num_total_underrun_samples == expected_count
    assert metrics.total_audio_underrun_events == expected_count
    if expected_count:
        assert metrics.mean_total_underrun_s == 0.4
        assert metrics.mean_audio_underrun_ratio == pytest.approx(2 / 3)
        assert metrics.mean_audio_underrun_events_per_minute == 300


@pytest.mark.parametrize("backend", ["speech", "chat"])
def test_multiple_stalls_transport_and_all_rollups(monkeypatch, backend):
    clock = [0.0]
    monkeypatch.setattr(patch.time, "perf_counter", lambda: clock[0])
    monkeypatch.setenv("VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE", "24000")
    monkeypatch.setenv("VLLM_OMNI_BENCH_AUDIO_CHANNELS", "1")
    monkeypatch.setenv("VLLM_OMNI_BENCH_AUDIO_CONTINUITY_THRESHOLD_S", "0.1")
    arrivals, durations = [0.3, 0.45, 0.85], [0.1, 0.2, 0.2]
    if backend == "speech":
        chunks = [(t, bytes(round(d * 48000))) for t, d in zip(arrivals, durations)]
        request, func = speech_request(), patch.async_request_openai_audio_speech
    else:
        chunks = [(t, sse_audio(wav_chunk(duration=d))) for t, d in zip(arrivals, durations)]
        chunks.append((10.0, b"data: [DONE]\n\n"))
        request = RequestFuncInput(
            model="test",
            model_name="test",
            prompt="hello",
            prompt_len=1,
            output_len=0,
            api_url="http://unused/v1/chat/completions",
        )
        func = patch.async_request_openai_chat_omni_completions
    output = asyncio.run(func(request, Session(post=lambda **kwargs: Response(clock, chunks))))
    assert output.success, output.error
    (sample,) = output.audio_continuity_samples
    expected = {
        "audio_underrun_s": 0.25,
        "audio_max_single_underrun_s": 0.2,
        "total_underrun_s": 0.25,
        "audio_underrun_event_count": 2,
        "audio_underrun_ratio": 1 / 3,
        "audio_underrun_events_per_minute": 240,
    }
    for key, value in expected.items():
        assert sample[key] == pytest.approx(value)
        if backend == "speech":
            assert getattr(output, key) == pytest.approx(value)
    metrics = calculate([output], [key.removesuffix("_s") for key in expected])
    for key, value in expected.items():
        assert getattr(metrics, f"mean_{key}") == pytest.approx(value)
        assert getattr(metrics, f"median_{key}") == pytest.approx(value)
        assert dict(getattr(metrics, f"percentiles_{key}"))[99.0] == pytest.approx(value)
    assert metrics.total_audio_underrun_events == 2
    assert metrics.audio_continuity_ok_rate == 0.0


def test_reply_weighting_of_stall_details_and_total_threshold():
    # First reply: two 62.5ms stalls, exceeds the 100ms *total* budget.
    first = patch.continuity_sample([0, 0.1875, 0.375], [0.125] * 3, threshold_s=0.1)
    second = patch.continuity_sample([0, 0.125], [0.125, 1.0], threshold_s=0.1)
    output = patch.MixRequestFuncOutput(success=True, audio_continuity_samples=[first, second])
    metrics = calculate([output], ["total_underrun", "audio_underrun_ratio", "audio_underrun_events_per_minute"])
    assert metrics.audio_continuity_ok_rate == 0.5
    assert metrics.mean_audio_max_single_underrun_s == 0.03125
    assert metrics.mean_audio_underrun_s == 0.0625
    assert metrics.mean_total_underrun_s == 0.0625
    assert metrics.mean_audio_underrun_event_count == 1
    assert metrics.mean_audio_underrun_ratio == 0.125
    assert metrics.mean_audio_underrun_events_per_minute == 160
    assert metrics.total_audio_underrun_events == 2


def test_missing_stall_details_are_unmeasured():
    output = patch.MixRequestFuncOutput(success=True, audio_underrun_s=0.2, audio_continuity_ok=False)
    metrics = calculate([output], ["total_underrun", "audio_underrun_event_count"])
    assert metrics.num_audio_continuity_samples == 1
    assert metrics.num_total_underrun_samples == 0
    assert metrics.num_audio_underrun_event_count_samples == 0
    assert metrics.total_audio_underrun_events == 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("total_underrun_s", float("nan")),
        ("audio_underrun_event_count", 1.5),
        ("audio_underrun_ratio", 1.1),
        ("audio_underrun_events_per_minute", -1),
    ],
)
def test_invalid_stall_detail_excluded_from_its_rollup(key, value):
    sample = patch.continuity_sample([0, 0.5], [0.1, 0.1], threshold_s=0.1)
    sample[key] = value
    metrics = calculate([patch.MixRequestFuncOutput(success=True, audio_continuity_samples=[sample])], [])
    assert metrics.num_audio_continuity_samples == 1
    assert getattr(metrics, f"num_{key.removesuffix('_s')}_samples") == 0


def test_omniinteract_backend_preserves_reply_metrics_and_prints_report(monkeypatch, tmp_path, capsys):
    from vllm_omni.benchmarks.data_modules.omniinteract_dataset import (
        OmniInteractCase,
        OmniInteractPreparedInput,
        OmniInteractSessionOptions,
    )
    from vllm_omni.benchmarks.duplex_session_metrics import collect_duplex_session_metrics
    from vllm_omni.clients.duplex import EventCollector

    collector = EventCollector()
    for at, event in (
        (0.0, {"type": "response.created", "response": {"id": "a"}}),
        (0.3, pcm_delta("a", 0.1)),
        (0.45, pcm_delta("a", 0.2)),
        (0.85, pcm_delta("a", 0.2)),
        (1.0, {"type": "response.done", "response": {"id": "a", "status": "completed"}}),
        (100.0, {"type": "response.created", "response": {"id": "b"}}),
        (100.1, pcm_delta("b", 1.0)),
        (102.0, {"type": "response.done", "response": {"id": "b", "status": "completed"}}),
    ):
        collector.add(event, received_at_s=at)
    bundle = collect_duplex_session_metrics(collector, stream_start=0, session_id="test-session")

    async def run_case(*args, **kwargs):
        return patch.OmniInteractCaseResult(
            subset="1qna",
            video="test.mp4",
            output_dir=str(tmp_path),
            success=True,
            session_id="test-session",
            latency_s=102,
            audio_bytes=72000,
            duplex_request_metrics=bundle.request_metrics,
            duplex_session_metrics=bundle.session_metrics,
        )

    monkeypatch.setattr(patch, "run_omniinteract_case", run_case)
    request = speech_request()
    request.api_url = "http://unused/v1/realtime"
    request.omniinteract_case = OmniInteractCase(
        "1qna", "test.mp4", tmp_path / "test.mp4", tmp_path / "test.json", "test"
    )
    request.omniinteract_options = OmniInteractSessionOptions(tmp_path, 10, 10, "unused.wav")
    request.omniinteract_prepared_input = OmniInteractPreparedInput(1, b"", (), "data:audio/wav;base64,")
    output = asyncio.run(patch.async_request_openai_realtime_duplex(request, None))
    assert output.success, output.error
    assert len(output.audio_continuity_samples) == 2
    metrics = calculate(
        [output],
        [
            "audio_underrun",
            "audio_max_single_underrun",
            "total_underrun",
            "audio_underrun_event_count",
            "audio_underrun_ratio",
            "audio_underrun_events_per_minute",
            "audio_continuity_rate",
        ],
    )
    assert metrics.mean_audio_max_single_underrun_s == pytest.approx(0.1)
    assert metrics.mean_audio_underrun_s == pytest.approx(0.125)
    assert metrics.mean_total_underrun_s == pytest.approx(0.125)
    assert metrics.mean_audio_underrun_event_count == 1
    assert metrics.mean_audio_underrun_ratio == pytest.approx(1 / 6)
    assert metrics.mean_audio_underrun_events_per_minute == pytest.approx(120)
    assert metrics.audio_continuity_ok_rate == 0.5
    printed = capsys.readouterr().out
    assert "Mean TOTAL_UNDERRUN (s):" in printed
    assert "Mean AUDIO_UNDERRUN_EVENT_COUNT (events):" in printed
    assert "Mean AUDIO_UNDERRUN_RATIO (ratio):" in printed
    assert "Mean AUDIO_UNDERRUN_EVENTS_PER_MINUTE (events/min):" in printed


def pcm_delta(response_id, duration):
    return {
        "type": "response.output_audio.delta",
        "response_id": response_id,
        "sample_rate_hz": 24000,
        "delta": base64.b64encode(bytes(round(duration * 48000))).decode(),
    }
