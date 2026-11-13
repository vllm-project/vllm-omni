# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""NPU numerical smoke test for MiniMax H3 VAE MXFP8 attention."""

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test
from vllm_omni.platforms import current_omni_platform
from vllm_omni.platforms.npu import is_a5

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.fixture
def a5_device():
    if not current_omni_platform.is_npu():
        pytest.skip("requires an NPU")
    device = current_omni_platform.get_torch_device()
    if not is_a5(device):
        pytest.skip("requires an Ascend A5")
    return device


@hardware_test(res={"npu": "A5"}, num_cards=1)
def test_mxfp8_attention_unaligned_batch_matches_reference(a5_device):
    pytest.importorskip("mindiesd")
    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
    from vllm_omni.diffusion.models.minimax_h3.vae_mxfp8 import _run_mxfp8_fa

    torch.manual_seed(37)
    query, key, value = (
        torch.randn(2, 195, 32, 64, device=a5_device, dtype=torch.float16) for _ in range(3)
    )
    backend = FlashAttentionImpl(num_heads=32, head_size=64, softmax_scale=64**-0.5, qkv_layout="BSND")
    with torch.inference_mode():
        actual = _run_mxfp8_fa(backend, query, key, value)
        expected = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).transpose(1, 2)
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    error = (actual.float() - expected.float()).abs()
    # A smoke bound for this fixed random input, not a full-video quality limit.
    assert error.max().item() < 0.3
    assert error.mean().item() < 0.03
