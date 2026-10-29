# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CovoAudio token2wav attention must not use the deprecated SDPA context manager."""

from __future__ import annotations

import warnings

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.covo_audio.token2wav import MultiHeadAttention
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model]


@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="CUDA required")
@hardware_test(res={"cuda": "L4"})
@torch.inference_mode()
def test_cuda_attention_does_not_warn_about_sdp_kernel():
    attn = MultiHeadAttention(hidden_size=64, num_heads=4).to("cuda")
    x = torch.randn(1, 8, 64, device="cuda")

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        out = attn(x)

    assert out.shape == x.shape
