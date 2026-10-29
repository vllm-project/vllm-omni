# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CovoAudio token2wav attention must not use the deprecated SDPA context manager."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest
import torch
from torch.nn.attention import SDPBackend

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.covo_audio import token2wav
from vllm_omni.model_executor.models.covo_audio.token2wav import MultiHeadAttention
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model]

_BASE_BACKENDS = {SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION}


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("is_cuda", "major", "minor", "expected"),
    [
        (True, 8, 0, _BASE_BACKENDS | {SDPBackend.FLASH_ATTENTION}),
        (True, 8, 9, _BASE_BACKENDS),
        (True, 9, 0, _BASE_BACKENDS),
        (False, 0, 0, _BASE_BACKENDS),
    ],
    ids=["sm80", "sm89", "sm90", "not_cuda"],
)
def test_cuda_backends_add_flash_only_on_sm80(monkeypatch, is_cuda, major, minor, expected):
    monkeypatch.setattr(token2wav, "current_omni_platform", SimpleNamespace(is_cuda=lambda: is_cuda))
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(major=major, minor=minor))

    assert set(MultiHeadAttention(hidden_size=64, num_heads=4).cuda_backends) == expected


@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="CUDA required")
@hardware_test(res={"cuda": "L4"})
@torch.inference_mode()
def test_cuda_attention_does_not_warn_about_sdp_kernel():
    attn = MultiHeadAttention(hidden_size=64, num_heads=4).to("cuda")
    x = torch.randn(1, 8, 64, device="cuda")

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*sdp_kernel", category=FutureWarning)
        out = attn(x)

    assert out.shape == x.shape
