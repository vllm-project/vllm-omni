# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import json
import time
from argparse import Namespace
from collections.abc import AsyncIterator

import aiohttp
import pytest
from pytest_mock import MockerFixture
from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput
from vllm.benchmarks.serve import TaskType

from vllm_omni.benchmarks import serve
from vllm_omni.benchmarks.patch import patch

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


async def _chunks(chunks: list[bytes], error: Exception | None = None) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk
    if error is not None:
        raise error


@pytest.fixture
def speech_request() -> RequestFuncInput:
    return RequestFuncInput(
        model="test-model",
        prompt="test speech",
        prompt_len=2,
        output_len=20,
        api_url="http://unused/v1/audio/speech",
    )


@pytest.fixture
def response_session(mocker: MockerFixture):
    response = mocker.MagicMock(spec=aiohttp.ClientResponse)
    response.status = 200
    response.headers = {"X-Audio-Sample-Rate": "48000", "X-Audio-Channels": "2"}
    response.content = mocker.Mock(spec=aiohttp.StreamReader)
    response.__aenter__.return_value = response
    response.__aexit__.return_value = False
    session = mocker.MagicMock(spec=aiohttp.ClientSession)
    session.post.return_value = response
    return response, session


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_audio_details_preserve_transport_reads_and_pcm_format(
    mocker: MockerFixture, speech_request, response_session, enabled: bool
) -> None:
    response, session = response_session
    response.content.iter_any.return_value = _chunks([b"\x01", b"", b"\x00" * 7])
    mocker.patch.object(patch.time, "perf_counter", side_effect=[10.0, 10.1, 10.3, 10.8])
    token = patch.CAPTURE_AUDIO_STREAM_DETAILS.set(enabled)
    try:
        output = await patch.async_request_openai_audio_speech(speech_request, session)
    finally:
        patch.CAPTURE_AUDIO_STREAM_DETAILS.reset(token)

    assert output.success
    assert output.audio_ttfp == pytest.approx(0.1)
    assert output.latency == pytest.approx(0.8)
    assert output.audio_duration == pytest.approx(8 / (48_000 * 2 * 2))
    if not enabled:
        assert output.audio_stream_details is None
        return
    saved = json.loads(json.dumps(output.audio_stream_details, allow_nan=False))
    assert saved["sample_rate"] == 48_000
    assert saved["channels"] == 2
    assert saved["sample_width"] == 2
    assert saved["chunk_sizes_bytes"] == [1, 7]
    assert saved["chunk_arrival_times_s"] == pytest.approx([0.1, 0.3])
    assert saved["response_end_s"] == pytest.approx(0.8)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chunks", "content_type", "error"),
    [
        ([], "application/json", "no PCM data"),
        ([b"\x00"], "audio/pcm", "incomplete PCM frame"),
        ([], None, "no PCM data"),
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
async def test_invalid_pcm_is_not_a_successful_stream(
    speech_request, response_session, chunks, content_type, error, enabled
) -> None:
    response, session = response_session
    if content_type is not None:
        response.headers["Content-Type"] = content_type
    response.content.iter_any.return_value = _chunks(chunks)
    token = patch.CAPTURE_AUDIO_STREAM_DETAILS.set(enabled)
    try:
        output = await patch.async_request_openai_audio_speech(speech_request, session)
    finally:
        patch.CAPTURE_AUDIO_STREAM_DETAILS.reset(token)
    assert not output.success
    assert error in output.error
    assert f"Content-Type={content_type!r}" in output.error


@pytest.mark.asyncio
async def test_failed_stream_keeps_partial_arrivals(speech_request, response_session) -> None:
    response, session = response_session
    response.content.iter_any.return_value = _chunks([b"\x00" * 8], ConnectionResetError("stream reset"))
    token = patch.CAPTURE_AUDIO_STREAM_DETAILS.set(True)
    try:
        output = await patch.async_request_openai_audio_speech(speech_request, session)
    finally:
        patch.CAPTURE_AUDIO_STREAM_DETAILS.reset(token)
    assert not output.success
    assert "stream reset" in output.error
    assert output.audio_stream_details is not None
    assert output.audio_stream_details["chunk_sizes_bytes"] == [8]
    assert len(output.audio_stream_details["chunk_arrival_times_s"]) == 1
    assert output.audio_stream_details["response_end_s"] is None


@pytest.mark.parametrize("enabled", [False, True])
def test_benchmark_exports_audio_details_in_request_order(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    details = patch.AudioStreamDetails(
        sample_rate=24_000,
        channels=1,
        sample_width=2,
        chunk_arrival_times_s=[0.1, 0.2],
        chunk_sizes_bytes=[24_000, 24_000],
        response_end_s=0.5,
    )

    async def request_func(request_func_input, session, pbar=None):
        success = request_func_input.prompt == "complete"
        return patch.MixRequestFuncOutput(
            success=success,
            start_time=time.perf_counter(),
            audio_ttfp=0.1 if success else 0.0,
            audio_duration=1.0 if success else 0.0,
            audio_rtf=0.5 if success else 0.0,
            latency=0.5 if success else 0.0,
            audio_stream_details=details if enabled and success else None,
            error="" if success else "no audio",
        )

    monkeypatch.setitem(patch.ASYNC_REQUEST_FUNCS, "openai-audio-speech", request_func)
    token = patch.CAPTURE_AUDIO_STREAM_DETAILS.set(enabled)
    try:
        result = asyncio.run(
            patch.benchmark(
                task_type=TaskType.GENERATION,
                endpoint_type="openai-audio-speech",
                api_url="http://unused/v1/audio/speech",
                base_url="http://unused",
                model_id="test-model",
                model_name="test-model",
                tokenizer=None,
                input_requests=[
                    SampleRequest(prompt=text, prompt_len=2, expected_output_len=20) for text in ("failed", "complete")
                ],
                logprobs=None,
                request_rate=float("inf"),
                burstiness=1.0,
                disable_tqdm=True,
                num_warmups=0,
                profile=False,
                selected_percentile_metrics=[],
                selected_percentiles=[50.0],
                ignore_eos=False,
                goodput_config_dict={},
                max_concurrency=2,
                lora_modules=None,
                extra_headers=None,
                extra_body=None,
                ready_check_timeout_sec=0,
            )
        )
    finally:
        patch.CAPTURE_AUDIO_STREAM_DETAILS.reset(token)
    assert result["completed"] == 1
    if enabled:
        saved = json.loads(json.dumps(result["audio_stream_details"], allow_nan=False))
        assert saved == [None, details]
    else:
        assert "audio_stream_details" not in result


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_cli_scopes_audio_details_to_one_run(monkeypatch: pytest.MonkeyPatch, enabled: bool, fails: bool) -> None:
    async def run(args: Namespace) -> dict[str, bool]:
        assert patch.CAPTURE_AUDIO_STREAM_DETAILS.get() is enabled
        if fails:
            raise RuntimeError("benchmark failed")
        return {"ok": True}

    monkeypatch.setattr(serve, "main_async", run)
    token = patch.CAPTURE_AUDIO_STREAM_DETAILS.set(not enabled)
    try:
        args = Namespace(backend="openai-audio-speech", save_detailed=enabled)
        if fails:
            with pytest.raises(RuntimeError, match="benchmark failed"):
                serve.main(args)
        else:
            assert serve.main(args) == {"ok": True}
        assert patch.CAPTURE_AUDIO_STREAM_DETAILS.get() is not enabled
    finally:
        patch.CAPTURE_AUDIO_STREAM_DETAILS.reset(token)
