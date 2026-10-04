# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for diffusion benchmark tokenizer bypass (Issue #6873)."""

import asyncio
from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from vllm_omni.benchmarks.serve import (
    _main_async_diffusion,
    is_diffusion_benchmark,
    main,
)

DIFFUSION_ENDPOINTS = ["/v1/images/generations", "/v1/images/edits", "/v1/videos"]
DIFFUSION_BACKENDS = ["openai-image-gen-omni", "openai-image-edits-omni", "openai-video-omni"]


@pytest.mark.parametrize("endpoint", DIFFUSION_ENDPOINTS)
def test_diffusion_endpoints_detected(endpoint):
    assert is_diffusion_benchmark(Namespace(endpoint=endpoint, backend=""))


@pytest.mark.parametrize(
    "endpoint,backend",
    [
        ("/v1/chat/completions", ""),
        ("", "openai-chat-omni"),
        ("", ""),
    ],
)
def test_non_diffusion_not_detected(endpoint, backend):
    assert not is_diffusion_benchmark(Namespace(endpoint=endpoint, backend=backend))


def test_main_routes_diffusion_to_diffusion_path():
    args = Namespace(
        endpoint="/v1/images/generations",
        backend="openai-image-gen-omni",
        seed_tts_wer_eval=False,
        seed_tts_wer_save_items=False,
        daily_omni_save_eval_items=False,
        print_stage=False,
        extra_body=None,
        dataset_name=None,
    )
    with patch(
        "vllm_omni.benchmarks.serve._main_async_diffusion",
        new_callable=AsyncMock,
        return_value={"routed": True},
    ) as mock:
        result = main(args)
        assert mock.called
        assert result == {"routed": True}


def test_tokenizer_is_none_throughout():
    args = Namespace(
        endpoint="/v1/images/generations",
        backend="openai-image-gen-omni",
        seed=42,
        base_url="http://localhost:8000",
        model="stabilityai/stable-diffusion-3.5-medium",
        served_model_name="stabilityai/stable-diffusion-3.5-medium",
        extra_body={},
        header=None,
        insecure=False,
        num_prompts=5,
        request_rate=float("inf"),
        burstiness=1.0,
        disable_tqdm=True,
        label=None,
        percentile_metrics="ttft,tpot,itl",
        metric_percentiles="99",
        num_warmups=1,
        max_concurrency=1,
        logprobs=None,
        ready_check_timeout_sec=600,
        metadata=None,
        save_result=False,
        append_result=False,
        save_detailed=False,
        goodput=None,
        plot_timeline=False,
        result_filename=None,
        result_dir=None,
        ramp_up_strategy=None,
    )
    with (
        patch(
            "vllm_omni.benchmarks.patch.patch.get_samples",
            return_value=[],
        ) as mock_samples,
        patch(
            "vllm_omni.benchmarks.patch.patch.benchmark",
            new_callable=AsyncMock,
            return_value={},
        ) as mock_bench,
    ):
        asyncio.run(_main_async_diffusion(args))
        assert mock_samples.call_args[0][1] is None
        assert mock_bench.call_args[1]["tokenizer"] is None
