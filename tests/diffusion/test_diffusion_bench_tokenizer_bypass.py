# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for diffusion benchmark tokenizer bypass (Issue #6873)."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from vllm_omni.benchmarks.serve import (
    _prepare_diffusion_args,
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


def test_prepare_diffusion_args_sets_skip_tokenizer():
    args = Namespace()
    _prepare_diffusion_args(args)
    assert args.skip_tokenizer_init is True


def test_main_sets_skip_tokenizer_for_diffusion():
    args = Namespace(
        endpoint="/v1/images/generations",
        backend="openai-image-gen-omni",
        seed_tts_wer_eval=False,
        seed_tts_wer_save_items=False,
        daily_omni_save_eval_items=False,
        videomme_save_eval_items=False,
        omni_request_timeout_s=None,
        print_stage=False,
        extra_body=None,
        dataset_name=None,
    )
    with patch(
        "vllm_omni.benchmarks.serve.main_async",
        new_callable=AsyncMock,
        return_value={"ok": True},
    ) as mock:
        result = main(args)
        assert mock.called
        assert result == {"ok": True}
        called_args = mock.call_args[0][0]
        assert called_args.skip_tokenizer_init is True
