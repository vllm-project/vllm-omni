# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CovoAudio token2wav attention must not use the deprecated SDPA context manager."""

from __future__ import annotations

import warnings

import pytest
import torch
from torch.nn.attention import SDPBackend

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.covo_audio.token2wav import MultiHeadAttention
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model]

_BASE_BACKENDS = {SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION}


@pytest.mark.cpu
def test_cuda_backends_excludes_flash(monkeypatch):
    """cuda_backends never includes FLASH_ATTENTION.

    forward() always passes a dense attn_bias tensor as attn_mask, which
    FlashAttention2 does not accept, so adding it to the allowed list has
    no effect and is omitted to keep the backend list accurate.
    """
    assert set(MultiHeadAttention(hidden_size=64, num_heads=4).cuda_backends) == _BASE_BACKENDS


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
