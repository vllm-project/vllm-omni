import io
import wave
from types import SimpleNamespace

import pytest

from benchmarks.diffusion.backends import (
    RequestFuncInput,
    RequestFuncOutput,
    async_request_audio_generate,
    async_request_chat_completions,
    endpoint_filename_token,
    normalize_endpoint,
)
from benchmarks.diffusion.diffusion_benchmark_serving import (
    _compute_expected_latency_ms_from_base,
    _infer_slo_base_time_ms_from_warmups,
    calculate_metrics,
)


class _MockResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return str(self._payload)

    async def read(self):
        return self._payload


class _MockSession:
    def __init__(self, payload: dict):
        self._payload = payload
        self.last_json = None

    def post(self, *args, **kwargs):
        self.last_json = kwargs.get("json")
        return _MockResponse(self._payload)


@pytest.mark.core_model
@pytest.mark.benchmark
@pytest.mark.cpu
def test_endpoint_normalization_accepts_optional_leading_slash():
    assert normalize_endpoint("v1/videos") == "/v1/videos"
    assert normalize_endpoint("/v1/videos") == "/v1/videos"
    assert normalize_endpoint("v1/chat/completions") == "/v1/chat/completions"
    assert normalize_endpoint("v1/images/generations") == "/v1/images/generations"


@pytest.mark.core_model
@pytest.mark.benchmark
@pytest.mark.cpu
def test_endpoint_normalization_accepts_legacy_backend_aliases():
    assert normalize_endpoint("vllm-omni") == "/v1/chat/completions"
    assert normalize_endpoint("openai") == "/v1/images/generations"


@pytest.mark.core_model
@pytest.mark.benchmark
@pytest.mark.cpu
def test_endpoint_filename_token_drops_leading_slash():
    assert endpoint_filename_token("/v1/videos") == "v1_videos"
    assert endpoint_filename_token("v1/chat/completions") == "v1_chat_completions"


@pytest.mark.core_model
@pytest.mark.benchmark
@pytest.mark.cpu
@pytest.mark.asyncio
async def test_chat_completions_metrics_fallback_to_top_level():
    payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc"},
                        }
                    ]
                }
            }
        ],
        "metrics": {
            "stage_durations": {"diffusion": 1.25},
            "peak_memory_mb": 4096.0,
        },
    }

    output = await async_request_chat_completions(
        RequestFuncInput(
            prompt="draw a cat",
            api_url="http://test.local/v1/chat/completions",
            model="ByteDance-Seed/BAGEL-7B-MoT",
        ),
        session=_MockSession(payload),
    )

    assert output.success is True
    assert output.stage_durations == {"diffusion": 1.25}
    assert output.peak_memory_mb == 4096.0


@pytest.mark.core_model
@pytest.mark.benchmark
@pytest.mark.cpu
@pytest.mark.asyncio
async def test_chat_completions_metrics_message_level_takes_precedence():
    payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc"},
                            "stage_durations": {"message_stage": 0.7},
                            "peak_memory_mb": 1234.0,
                        }
                    ]
                }
            }
        ],
        "metrics": {
            "stage_durations": {"top_level_stage": 9.9},
            "peak_memory_mb": 9999.0,
        },
    }

    output = await async_request_chat_completions(
        RequestFuncInput(
            prompt="draw a dog",
            api_url="http://test.local/v1/chat/completions",
            model="ByteDance-Seed/BAGEL-7B-MoT",
        ),
        session=_MockSession(payload),
    )

    assert output.success is True
    assert output.stage_durations == {"message_stage": 0.7}
    assert output.peak_memory_mb == 1234.0


def _make_wav(duration: float = 1.0, sample_rate: int = 8000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\0\0" * int(duration * sample_rate))
    return buffer.getvalue()


@pytest.mark.core_model
@pytest.mark.benchmark
@pytest.mark.cpu
@pytest.mark.asyncio
async def test_audio_generate_builds_payload_and_measures_wav_duration():
    session = _MockSession(_make_wav(duration=2.0))
    request = RequestFuncInput(
        prompt="ocean waves",
        api_url="http://test.local/v1/audio/generate",
        model="stabilityai/stable-audio-open-1.0",
        audio_length=2.0,
        num_inference_steps=20,
        seed=42,
        extra_body={"guidance_scale": 7.0},
    )

    output = await async_request_audio_generate(request, session=session)

    assert output.success is True
    assert output.audio_duration == pytest.approx(2.0)
    assert output.response_body
    assert session.last_json == {
        "input": "ocean waves",
        "model": "stabilityai/stable-audio-open-1.0",
        "response_format": "wav",
        "audio_length": 2.0,
        "num_inference_steps": 20,
        "seed": 42,
        "guidance_scale": 7.0,
    }


@pytest.mark.core_model
@pytest.mark.benchmark
@pytest.mark.cpu
def test_audio_metrics_include_rtf_and_generated_audio_throughput():
    outputs = [
        RequestFuncOutput(success=True, latency=1.0, audio_duration=2.0),
        RequestFuncOutput(success=True, latency=3.0, audio_duration=2.0),
    ]
    requests = [
        RequestFuncInput(prompt="a", api_url="http://test", model="model"),
        RequestFuncInput(prompt="b", api_url="http://test", model="model"),
    ]

    metrics = calculate_metrics(outputs, 4.0, requests, SimpleNamespace(), False)

    assert metrics["audio_duration_total"] == pytest.approx(4.0)
    assert metrics["audio_throughput_seconds_per_second"] == pytest.approx(1.0)
    assert metrics["audio_rtf_mean"] == pytest.approx(1.0)
    assert metrics["audio_rtf_median"] == pytest.approx(1.0)


@pytest.mark.core_model
@pytest.mark.benchmark
@pytest.mark.cpu
def test_audio_slo_scales_with_duration_and_diffusion_steps():
    args = SimpleNamespace(task="t2a", audio_length=10.0, num_inference_steps=20)
    warmup_request = RequestFuncInput(
        prompt="warmup",
        api_url="http://test",
        model="model",
        audio_length=2.0,
        num_inference_steps=5,
    )
    base_ms = _infer_slo_base_time_ms_from_warmups(
        [(warmup_request, RequestFuncOutput(success=True, latency=1.0))], args
    )
    request = RequestFuncInput(
        prompt="benchmark",
        api_url="http://test",
        model="model",
        audio_length=4.0,
        num_inference_steps=10,
    )

    assert base_ms == pytest.approx(100.0)
    assert _compute_expected_latency_ms_from_base(request, args, base_ms) == pytest.approx(4000.0)
