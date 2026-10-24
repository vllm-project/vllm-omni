# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Verify metric sample counts survive benchmark JSON serialization."""

import argparse
import asyncio
import json
import time

import pytest
from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.serve import TaskType

from vllm_omni.benchmarks.patch import patch

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


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
        {"response_id": "r1", "ttft_ms": 50.0, "ttfp_ms": 150.0, "client_ttft_ms": 300.0, "client_ttfp_ms": 500.0},
        {"response_id": "r2", "ttft_ms": 70.0, "ttfp_ms": 170.0, "client_ttft_ms": 320.0, "client_ttfp_ms": 700.0},
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
    assert saved_result["num_duplex_client_ttfp_ms_samples"] == 2
    assert saved_result["mean_duplex_client_ttfp_ms"] == pytest.approx(600.0)
    assert saved_result["mean_duplex_response_ttft_ms"] == pytest.approx(60.0)
    assert saved_result["p99_duplex_client_ttft_ms"] == pytest.approx(320.0)


@pytest.mark.parametrize(
    "selected_metrics,has_audio",
    [("audio_ttfp,audio_rtf", True), ("audio_ttfp", False), ("ttft", True), ("ttft", False)],
)
def test_audio_request_metrics_survive_detailed_result_save(monkeypatch, tmp_path, selected_metrics, has_audio):
    from vllm.benchmarks import serve as upstream_serve

    from vllm_omni.benchmarks import serve
    from vllm_omni.entrypoints.cli.benchmark.serve import OmniBenchmarkServingSubcommand

    requests = [SampleRequest(prompt="hi", prompt_len=2, expected_output_len=3, request_id=f"r{i}") for i in range(3)]
    outputs = [
        patch.MixRequestFuncOutput(
            success=True,
            latency=0.5,
            audio_ttfp=0.1 if has_audio else 0.0,
            audio_rtf=0.2 if has_audio else 0.0,
            audio_duration=2.0 if has_audio else 0.0,
            audio_frames=48000 if has_audio else 0,
        ),
        patch.MixRequestFuncOutput(success=True, latency=0.7),
        patch.MixRequestFuncOutput(
            success=False, error="stream failed", latency=0.8, audio_ttfp=float("nan"), audio_rtf=float("inf")
        ),
    ]

    async def request_func(request_func_input, session, pbar=None):
        output = outputs[int(request_func_input.request_id[1:])]
        output.start_time = time.perf_counter()
        return output

    monkeypatch.setattr(upstream_serve, "get_samples", lambda args, tokenizer: requests)
    monkeypatch.setitem(patch.ASYNC_REQUEST_FUNCS, "openai-chat-omni", request_func)
    parser = argparse.ArgumentParser()
    OmniBenchmarkServingSubcommand.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--model",
            "test-model",
            "--backend",
            "openai-chat-omni",
            "--skip-tokenizer-init",
            "--disable-tqdm",
            "--num-prompts",
            "3",
            "--num-warmups",
            "0",
            "--max-concurrency",
            "1",
            "--ready-check-timeout-sec",
            "0",
            "--percentile-metrics",
            selected_metrics,
            "--save-result",
            "--save-detailed",
            "--result-dir",
            str(tmp_path),
            "--result-filename",
            "audio.json",
        ]
    )
    result = serve.main(args)
    saved = json.loads((tmp_path / "audio.json").read_text())
    if not has_audio and selected_metrics == "ttft":
        assert "audio_request_metrics" not in saved
        return
    rows = saved["audio_request_metrics"]
    assert rows == result["audio_request_metrics"]
    assert [row["request_index"] for row in rows] == [0, 1, 2]
    assert [row["request_id"] for row in rows] == ["r0", "r1", "r2"]
    assert all(row["start_time"] > 0 for row in rows)
    assert [row["success"] for row in rows] == [True, True, False]
    assert [row["latency"] for row in rows] == [0.5, 0.7, 0.8]
    assert rows[2]["error"] == "stream failed"
    fields = ("audio_ttfp", "audio_rtf", "audio_duration", "audio_frames")
    expected_audio: list[float | None] = [0.1, 0.2, 2.0, 48000] if has_audio else [None] * 4
    assert [rows[0][field] for field in fields] == expected_audio
    assert all(row[field] is None for row in rows[1:] for field in fields)
    json.dumps(rows, allow_nan=False)
