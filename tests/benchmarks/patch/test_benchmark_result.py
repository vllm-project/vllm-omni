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
