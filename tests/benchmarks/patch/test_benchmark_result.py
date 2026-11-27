# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Verify metric sample counts survive benchmark JSON serialization."""

import asyncio
import base64
import datetime as dt
import io
import json
import subprocess
import sys
import time
import wave
from pathlib import Path
from unittest.mock import Mock

import pytest
from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.serve import TaskType

from tests.benchmarks.patch.test_patch import MockResponse, create_sse_chunk
from vllm_omni.benchmarks.patch import patch

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


@pytest.mark.parametrize("backend", ["chat", "speech"])
@pytest.mark.parametrize("include_failed", [False, True])
def test_last_audio_survives_benchmark_export(monkeypatch, tmp_path, backend, include_failed):
    class Clock:
        current = 100.0

        @classmethod
        def perf_counter(cls):
            return cls.current

    pcm = b"\0\0" * 48000  # Two seconds at 24 kHz, mono.
    if backend == "chat":
        wav = io.BytesIO()
        with wave.open(wav, "wb") as writer:
            writer.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            writer.writeframes(pcm[: len(pcm) // 2])
        audio_event = create_sse_chunk(
            {"modality": "audio", "choices": [{"delta": {"content": base64.b64encode(wav.getvalue()).decode()}}]}
        )
        events = [
            (100.5, audio_event),
            (101.0, audio_event),
            (102.0, create_sse_chunk({"modality": "audio", "choices": [{"delta": {}}]})),
            (103.0, create_sse_chunk({"choices": [], "usage": {"completion_tokens": 1}})),
            (103.0, b"data: [DONE]\n\n"),
        ]
        request = patch.async_request_openai_chat_omni_completions
        endpoint = "chat/completions"
    else:
        events = [(100.5, pcm[: len(pcm) // 2]), (101.0, pcm[len(pcm) // 2 :])]
        request = patch.async_request_openai_audio_speech
        endpoint = "audio/speech"

    async def stream():
        for stamp, chunk in events:
            Clock.current = stamp
            yield chunk
        Clock.current = 103.0  # Delayed usage/EOF must stay out of audio RTF.

    response = MockResponse(200, [])
    response.headers = {"X-Audio-Sample-Rate": "24000", "X-Audio-Channels": "1"}
    response.content.iter_any = stream
    session = Mock()
    session.post.return_value = response

    async def request_func(request_func_input, **kwargs):
        if request_func_input.prompt == "fail":
            # A partial audio response that failed must not enter the ledger.
            return patch.MixRequestFuncOutput(
                success=False, error="stream failed", audio_duration=2.0, audio_latency=1.0, audio_rtf=0.5
            )
        return await request(request_func_input, session)

    monkeypatch.setattr(patch, "time", Clock)
    monkeypatch.setitem(patch.ASYNC_REQUEST_FUNCS, "test-audio-completion", request_func)
    result = asyncio.run(
        patch.benchmark(
            task_type=TaskType.GENERATION,
            endpoint_type="test-audio-completion",
            api_url=f"http://unused/v1/{endpoint}",
            base_url="http://unused",
            model_id="test-model",
            model_name="test-model",
            tokenizer=None,
            input_requests=[
                SampleRequest(prompt=prompt, prompt_len=2, expected_output_len=1)
                for prompt in (["hi", "fail"] if include_failed else ["hi"])
            ],
            logprobs=None,
            request_rate=float("inf"),
            burstiness=1.0,
            disable_tqdm=True,
            num_warmups=0,
            profile=False,
            selected_percentile_metrics=["e2el", "audio_rtf", "audio_duration"],
            selected_percentiles=[50.0],
            ignore_eos=False,
            goodput_config_dict={},
            max_concurrency=1,
            lora_modules=None,
            extra_headers=None,
            extra_body=None,
            ready_check_timeout_sec=0,
        )
    )
    saved = json.loads(json.dumps(result, allow_nan=False))
    assert saved["completed"] == 1
    assert saved["mean_e2el_ms"] == 3000
    assert saved["mean_audio_rtf"] == 0.5
    expected_audio: list[dict[str, float] | None] = [{"audio_duration_s": 2.0, "audio_latency_s": 1.0}]
    if include_failed:
        expected_audio.append(None)
    assert saved["request_audio_metrics"] == expected_audio
    assert "request_stage_metrics" not in saved

    # Feed the real producer result through the report CLI (serve adds C/failed).
    saved.update(max_concurrency=1, failed=int(include_failed))
    result_path, ledger_path = tmp_path / "result.json", tmp_path / "ledger.json"
    result_path.write_text(json.dumps(saved))
    report = Path(__file__).resolve().parents[3] / ".claude/skills/omni-perf-ceiling/scripts/ceiling_report.py"
    proc = subprocess.run(
        [sys.executable, str(report), "--cell", f"audio={result_path}", "--json", str(ledger_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    cell = json.loads(ledger_path.read_text())[0]
    assert cell["aggregate_rtf"] == 0.5
    assert cell["request_e2el_rtf"] == 1.5
    assert cell["little_residual"] == pytest.approx(0)


def test_benchmark_end_time_precedes_postprocessing(monkeypatch):
    measured_end = dt.datetime(2026, 11, 12, 12, 0, 10)

    class Clock(dt.datetime):
        current = measured_end

        @classmethod
        def now(cls, tz=None):
            return cls.current

    async def evaluate(*args):
        Clock.current = measured_end + dt.timedelta(seconds=60)
        return None

    async def request_func(**kwargs):
        return patch.MixRequestFuncOutput(
            success=True,
            prompt_len=2,
            output_tokens=1,
            generated_text="hi",
            start_time=time.perf_counter(),
            ttft=0.01,
            latency=0.02,
        )

    monkeypatch.setattr(patch, "datetime", Clock)
    monkeypatch.setattr(patch, "_evaluate_omniinteract_batch", evaluate)
    monkeypatch.setitem(patch.ASYNC_REQUEST_FUNCS, "test-window", request_func)
    result = asyncio.run(
        patch.benchmark(
            task_type=TaskType.GENERATION,
            endpoint_type="test-window",
            api_url="http://unused/v1/chat/completions",
            base_url="http://unused",
            model_id="test-model",
            model_name="test-model",
            tokenizer=None,
            input_requests=[SampleRequest(prompt="hi", prompt_len=2, expected_output_len=1)],
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
            max_concurrency=1,
            lora_modules=None,
            extra_headers=None,
            extra_body=None,
            ready_check_timeout_sec=0,
        )
    )
    saved = json.loads(json.dumps(result))
    assert saved["benchmark_end_time"] == measured_end.isoformat()
    assert Clock.current > measured_end


@pytest.mark.parametrize("selected_metrics", [["ttft", "tpot", "itl"], []])
@pytest.mark.parametrize("measured", [True, False])
def test_benchmark_result_preserves_tpot_sample_count(monkeypatch, measured, selected_metrics):
    async def request_func(**kwargs):
        return patch.MixRequestFuncOutput(
            success=True,
            prompt_len=2,
            output_tokens=3,
            generated_text="hello world!",
            start_time=time.perf_counter(),
            ttft=0.01,
            latency=0.03,
            text_latency=0.03,
            itl=[0.01, 0.01] if measured else [0.0, 0.0],
        )

    monkeypatch.setitem(patch.ASYNC_REQUEST_FUNCS, "test-result", request_func)
    result = asyncio.run(
        patch.benchmark(
            task_type=TaskType.GENERATION,
            endpoint_type="test-result",
            api_url="http://unused/v1/chat/completions",
            base_url="http://unused",
            model_id="test-model",
            model_name="test-model",
            tokenizer=None,
            input_requests=[SampleRequest(prompt="hi", prompt_len=2, expected_output_len=3)],
            logprobs=None,
            request_rate=float("inf"),
            burstiness=1.0,
            disable_tqdm=True,
            num_warmups=0,
            profile=False,
            selected_percentile_metrics=selected_metrics,
            selected_percentiles=[50.0],
            ignore_eos=False,
            goodput_config_dict={},
            max_concurrency=1,
            lora_modules=None,
            extra_headers=None,
            extra_body=None,
            ready_check_timeout_sec=0,
        )
    )
    # DFX reads this dictionary from the benchmark's JSON result file.
    saved_result = json.loads(json.dumps(result, allow_nan=False))
    assert saved_result["completed"] == 1
    assert saved_result["request_audio_metrics"] == [None]
    assert saved_result["max_output_tokens_per_s"] > 0
    assert saved_result.get("num_tpot_samples") == int(measured)
    assert saved_result["num_ttft_samples"] == 1
    assert saved_result["num_itl_samples"] == 2
    assert "num_audio_ttfp_samples" in saved_result
    assert "num_audio_rtf_samples" in saved_result
    if measured and "tpot" in selected_metrics:
        assert saved_result["mean_tpot_ms"] == pytest.approx(10.0)
    else:
        assert "mean_tpot_ms" not in saved_result


def test_benchmark_result_flattens_duplex_per_response_latency(monkeypatch):
    request_metrics = [
        {
            "response_id": "r1",
            "ttft_ms": 50.0,
            "ttfp_ms": 150.0,
            "client_ttft_ms": 300.0,
            "client_ttfp_ms": 500.0,
            "client_rtf": 0.5,
            "busy_rtf": 0.2,
            "stall_ratio": 0.1,
        },
        {
            "response_id": "r2",
            "ttft_ms": 70.0,
            "ttfp_ms": 170.0,
            "client_ttft_ms": 320.0,
            "client_ttfp_ms": 700.0,
            "client_rtf": 0.7,
            "busy_rtf": 0.4,
            "stall_ratio": 0.3,
        },
    ]

    async def request_func(**kwargs):
        return patch.MixRequestFuncOutput(
            success=True,
            prompt_len=0,
            output_tokens=2,
            generated_text="hi",
            start_time=time.perf_counter(),
            ttft=0.06,
            latency=1.0,
            duplex_request_metrics=request_metrics,
            duplex_session_metrics={"ttft_ms": {"count": 2, "mean": 60.0, "p50": 50.0, "p99": 70.0}},
        )

    monkeypatch.setitem(patch.ASYNC_REQUEST_FUNCS, "test-duplex-result", request_func)
    result = asyncio.run(
        patch.benchmark(
            task_type=TaskType.GENERATION,
            endpoint_type="test-duplex-result",
            api_url="http://unused/v1/realtime",
            base_url="http://unused",
            model_id="test-model",
            model_name="test-model",
            tokenizer=None,
            input_requests=[SampleRequest(prompt="", prompt_len=0, expected_output_len=0)],
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
            max_concurrency=1,
            lora_modules=None,
            extra_headers=None,
            extra_body=None,
            ready_check_timeout_sec=0,
        )
    )
    saved_result = json.loads(json.dumps(result, allow_nan=False))
    assert saved_result["max_output_tokens_per_s"] is None
    assert saved_result["request_audio_metrics"] == [None]
    assert saved_result["num_duplex_client_ttfp_ms_samples"] == 2
    assert saved_result["mean_duplex_client_ttfp_ms"] == pytest.approx(600.0)
    assert saved_result["num_duplex_client_rtf_samples"] == 2
    assert saved_result["mean_duplex_client_rtf"] == pytest.approx(0.6)
    assert saved_result["num_duplex_busy_rtf_samples"] == 2
    assert saved_result["mean_duplex_busy_rtf"] == pytest.approx(0.3)
    assert saved_result["num_duplex_stall_ratio_samples"] == 2
    assert saved_result["mean_duplex_stall_ratio"] == pytest.approx(0.2)
    assert saved_result["mean_duplex_response_ttft_ms"] == pytest.approx(60.0)
    assert saved_result["p99_duplex_client_ttft_ms"] == pytest.approx(320.0)
