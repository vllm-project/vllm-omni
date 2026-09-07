# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for diffusion benchmark tokenizer bypass (Issue #6873, Task 1).

Verifies that diffusion models are correctly detected, routed to
``_main_async_diffusion`` (which skips tokenizer loading), and that
``get_samples`` / ``benchmark`` both receive ``tokenizer=None``.

Run:
    python -m pytest tests/diffusion/test_diffusion_bench_tokenizer_bypass.py -v
"""

import asyncio
from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm_omni.benchmarks.serve import (
    _main_async_diffusion,
    is_diffusion_benchmark,
    main,
)


# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------

class TestIsDiffusionBenchmark:

    @pytest.mark.parametrize("endpoint", [
        "/v1/images/generations",
        "/v1/images/edits",
        "/v1/videos",
    ])
    def test_diffusion_endpoints_detected(self, endpoint):
        assert is_diffusion_benchmark(Namespace(endpoint=endpoint, backend=""))

    @pytest.mark.parametrize("backend", [
        "openai-image-gen-omni",
        "openai-image-edits-omni",
        "openai-video-omni",
    ])
    def test_diffusion_backends_detected(self, backend):
        assert is_diffusion_benchmark(Namespace(endpoint="", backend=backend))

    @pytest.mark.parametrize("endpoint,backend", [
        ("/v1/chat/completions", ""),
        ("", "openai-chat-omni"),
        ("", ""),
    ])
    def test_non_diffusion_not_detected(self, endpoint, backend):
        assert not is_diffusion_benchmark(Namespace(endpoint=endpoint, backend=backend))

    def test_missing_attrs_not_detected(self):
        assert not is_diffusion_benchmark(Namespace())


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

class TestDiffusionRouting:

    def test_diffusion_args_route_to_diffusion_path(self):
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
        ) as mock_diffusion:
            result = main(args)
            assert mock_diffusion.called
            assert result == {"routed": True}

    def test_non_diffusion_args_skip_diffusion_path(self):
        args = Namespace(
            endpoint="/v1/chat/completions",
            backend="openai-chat-omni",
            seed_tts_wer_eval=False,
            seed_tts_wer_save_items=False,
            daily_omni_save_eval_items=False,
            print_stage=False,
            extra_body=None,
            dataset_name="daily-omni",
            omniinteract_output_dir="omniinteract-output",
        )
        with patch(
            "vllm_omni.benchmarks.serve._main_async_diffusion",
            new_callable=AsyncMock,
        ) as mock_diffusion, patch(
            "vllm_omni.benchmarks.serve.main_async",
            new_callable=AsyncMock,
            return_value={"upstream": True},
        ):
            main(args)
            assert not mock_diffusion.called


# ---------------------------------------------------------------------------
# Tokenizer bypass
# ---------------------------------------------------------------------------

def _make_diffusion_args():
    return Namespace(
        endpoint="/v1/images/generations",
        backend="openai-image-gen-omni",
        seed=42,
        base_url="http://localhost:8000",
        model="stabilityai/sd3",
        served_model_name="stabilityai/sd3",
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


class TestTokenizerBypass:

    def test_get_samples_receives_tokenizer_none(self):
        with patch(
            "vllm_omni.benchmarks.patch.patch.get_samples",
            return_value=[],
        ) as mock_get_samples, patch(
            "vllm_omni.benchmarks.patch.patch.benchmark",
            new_callable=AsyncMock,
            return_value={},
        ):
            asyncio.run(_main_async_diffusion(_make_diffusion_args()))
            assert mock_get_samples.called
            tokenizer_arg = mock_get_samples.call_args[0][1]
            assert tokenizer_arg is None

    def test_benchmark_receives_tokenizer_none(self):
        with patch(
            "vllm_omni.benchmarks.patch.patch.get_samples",
            return_value=[],
        ), patch(
            "vllm_omni.benchmarks.patch.patch.benchmark",
            new_callable=AsyncMock,
            return_value={},
        ) as mock_benchmark:
            asyncio.run(_main_async_diffusion(_make_diffusion_args()))
            assert mock_benchmark.called
            assert mock_benchmark.call_args[1]["tokenizer"] is None


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

class TestImports:

    def test_upstream_imports(self):
        from vllm.benchmarks.serve import (  # noqa: F401
            TaskType,
            check_goodput_args,
            compute_result_filename,
            get_first_model_from_server,
            main_async,
        )
        from vllm.utils.gc_utils import freeze_gc_heap  # noqa: F401
        from vllm.utils.network_utils import join_host_port  # noqa: F401

    def test_serve_module_exports(self):
        assert callable(is_diffusion_benchmark)
        assert callable(_main_async_diffusion)
        assert callable(main)
