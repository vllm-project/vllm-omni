import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.diffusion.backends import (
    RequestFuncInput,
    async_request_chat_completions,
    async_request_v1_videos,
    endpoint_filename_token,
    normalize_endpoint,
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


class _MockSession:
    def __init__(self, payload: dict):
        self._payload = payload

    def post(self, *args, **kwargs):
        return _MockResponse(self._payload)


class _MockVideoSession:
    def __init__(self):
        self.get_calls = 0
        self.delete_calls = 0

    def post(self, *args, **kwargs):
        return _MockResponse({"id": "job-1", "status": "queued"})

    def get(self, *args, **kwargs):
        self.get_calls += 1
        return _MockResponse({"status": "in_progress"})

    def delete(self, *args, **kwargs):
        self.delete_calls += 1
        return _MockResponse({})


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


@pytest.mark.core_model
@pytest.mark.benchmark
@pytest.mark.cpu
@pytest.mark.asyncio
async def test_video_job_uses_request_client_timeout(mocker):
    session = _MockVideoSession()
    clock = iter([0.0, 0.0, 1.0, 601.0])
    mocker.patch(
        "benchmarks.diffusion.backends.time.perf_counter",
        side_effect=lambda: next(clock),
    )
    mocker.patch(
        "benchmarks.diffusion.backends.asyncio.sleep",
        new=mocker.AsyncMock(),
    )

    output = await async_request_v1_videos(
        RequestFuncInput(
            prompt="generate a video",
            api_url="http://test.local/v1/videos",
            model="test-model",
            client_timeout=0.5,
        ),
        session=session,
    )

    assert output.success is False
    assert output.error == "Timed out waiting 0.5s for video job job-1 to complete."
    assert session.get_calls == 1
    assert session.delete_calls == 1


@pytest.mark.core_model
@pytest.mark.benchmark
@pytest.mark.cpu
def test_video_job_client_timeout_default_is_backward_compatible():
    request = RequestFuncInput(
        prompt="generate a video",
        api_url="http://test.local/v1/videos",
        model="test-model",
    )

    assert request.client_timeout == 600.0


@pytest.mark.core_model
@pytest.mark.benchmark
@pytest.mark.cpu
def test_diffusion_benchmark_cli_exposes_positive_client_timeout():
    script = Path(__file__).resolve().parents[2] / "benchmarks" / "diffusion" / "diffusion_benchmark_serving.py"

    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--client-timeout CLIENT_TIMEOUT" in help_result.stdout

    invalid_result = subprocess.run(
        [sys.executable, str(script), "--client-timeout", "0"],
        capture_output=True,
        text=True,
    )
    assert invalid_result.returncode == 2
    assert "must be a finite number greater than zero" in invalid_result.stderr
