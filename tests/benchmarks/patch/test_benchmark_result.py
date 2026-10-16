# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Verify metric sample counts survive benchmark JSON serialization."""

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
    saved_result = json.loads(json.dumps(result))
    assert saved_result["completed"] == 1
    assert saved_result.get("num_tpot_samples") == int(measured)
    assert saved_result["num_ttft_samples"] == 1
    assert saved_result["num_itl_samples"] == 2
    assert "num_audio_ttfp_samples" in saved_result
    assert "num_audio_rtf_samples" in saved_result
    if measured and "tpot" in selected_metrics:
        assert saved_result["mean_tpot_ms"] == pytest.approx(10.0)
    else:
        assert "mean_tpot_ms" not in saved_result


@pytest.mark.parametrize("selector", ["audio_continuity_rate", "audio_continuity_ok_rate"])
@pytest.mark.parametrize("measured", [True, False])
def test_benchmark_result_preserves_continuity_rate(monkeypatch, capsys, selector, measured):
    async def request_func(**kwargs):
        return patch.MixRequestFuncOutput(
            success=True,
            prompt_len=2,
            start_time=time.perf_counter(),
            latency=0.3,
            audio_duration=0.2,
            audio_continuity_samples=[
                {"response_id": "a", **patch.continuity_sample([0.3, 0.45, 0.85], [0.1, 0.2, 0.2], threshold_s=0.1)},
                {"response_id": "b", **patch.continuity_sample([0], [1], threshold_s=0.1)},
            ]
            if measured
            else None,
        )

    monkeypatch.setattr(patch, "_PRINT_STAGE", True)
    monkeypatch.setitem(patch.ASYNC_REQUEST_FUNCS, "test-continuity", request_func)
    result = asyncio.run(
        patch.benchmark(
            task_type=TaskType.GENERATION,
            endpoint_type="test-continuity",
            api_url="http://unused/v1/chat/completions",
            base_url="http://unused",
            model_id="test-model",
            model_name="test-model",
            tokenizer=None,
            input_requests=[SampleRequest(prompt="hi", prompt_len=2, expected_output_len=0)],
            logprobs=None,
            request_rate=float("inf"),
            burstiness=1.0,
            disable_tqdm=True,
            num_warmups=0,
            profile=False,
            selected_percentile_metrics=[
                "audio_underrun",
                "audio_max_single_underrun",
                "total_underrun",
                "audio_underrun_event_count",
                "audio_underrun_ratio",
                "audio_underrun_events_per_minute",
                selector,
            ],
            selected_percentiles=[50.0, 90.0, 95.0, 99.0],
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
    printed = capsys.readouterr().out
    assert saved["completed"] == 1
    assert saved["num_audio_continuity_samples"] == (2 if measured else 0)
    assert saved["audio_continuity_ok_rate"] == (0.5 if measured else None)
    assert len(saved["audio_continuity_samples"]) == (2 if measured else 0)
    assert "mean_audio_continuity_rate_ms" not in saved
    assert "mean_audio_continuity_ok_rate_ms" not in saved
    if measured:
        assert "Maximum Single Audio Underrun" in printed
        assert "Total Audio Underrun per Response" in printed
        assert "Streaming Audio Underrun (Cumulative)" in printed
        for metric, unit in (
            ("AUDIO_UNDERRUN", "s"),
            ("AUDIO_MAX_SINGLE_UNDERRUN", "s"),
            ("TOTAL_UNDERRUN", "s"),
            ("AUDIO_UNDERRUN_EVENT_COUNT", "events"),
            ("AUDIO_UNDERRUN_RATIO", "ratio"),
            ("AUDIO_UNDERRUN_EVENTS_PER_MINUTE", "events/min"),
        ):
            assert f"Mean {metric} ({unit}):" in printed
            assert f"P90 {metric} ({unit}):" in printed
            assert f"P95 {metric} ({unit}):" in printed
            assert f"P99 {metric} ({unit}):" in printed
            assert f"{metric} (ms):" not in printed
        assert saved["mean_audio_max_single_underrun_s"] == pytest.approx(0.1)
        assert saved["mean_audio_underrun_s"] == pytest.approx(0.125)
        assert saved["p99_audio_max_single_underrun_s"] == pytest.approx(0.198)
        assert saved["p99_audio_underrun_s"] == pytest.approx(0.2475)
        assert saved["mean_total_underrun_s"] == pytest.approx(0.125)
        assert saved["p99_total_underrun_s"] == pytest.approx(0.2475)
        assert saved["mean_audio_underrun_event_count"] == 1
        assert saved["mean_audio_underrun_ratio"] == pytest.approx(1 / 6)
        assert saved["mean_audio_underrun_events_per_minute"] == pytest.approx(120)
        assert saved["p90_total_underrun_s"] == pytest.approx(0.225)
        assert saved["p95_total_underrun_s"] == pytest.approx(0.2375)
        assert saved["p99_audio_underrun_event_count"] == pytest.approx(1.98)
        assert saved["p99_audio_underrun_ratio"] == pytest.approx(0.33)
        assert saved["p99_audio_underrun_events_per_minute"] == pytest.approx(237.6)
        assert saved["total_audio_underrun_events"] == 2
        assert saved["audio_continuity_samples"][0]["request_index"] == 0
    else:
        assert "mean_audio_underrun_s" not in saved
        assert "mean_total_underrun_s" not in saved
        assert saved["num_total_underrun_samples"] == 0
        assert saved["total_audio_underrun_events"] == 0
    assert "mean_total_underrun_ms" not in saved
    assert "mean_audio_underrun_ratio_ms" not in saved
    assert "mean_audio_underrun_event_count_ms" not in saved
    assert "mean_audio_underrun_events_per_minute_ms" not in saved
