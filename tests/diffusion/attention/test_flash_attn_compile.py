# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compile the production dense and packed vLLM FlashAttention fallback."""

from functools import partial

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.attention.backends.utils import fa

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cuda]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("packed", [False, True])
def test_bundled_flash_attention_fullgraph_matches_eager(monkeypatch, packed):
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    from vllm.vllm_flash_attn.flash_attn_interface import is_fa_version_supported

    if not is_fa_version_supported(2):
        pytest.skip("The vLLM FA2 kernel is unavailable")
    monkeypatch.setattr(fa, "flash_attn_func", None)
    monkeypatch.setattr(fa, "flash_attn_varlen_func", partial(flash_attn_varlen_func, fa_version=2))
    impl = FlashAttentionImpl(num_heads=2, num_kv_heads=1, head_size=64, softmax_scale=0.125)
    torch.manual_seed(6106)
    batch = 1 if packed else 2
    q = torch.randn(batch, 4, 2, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, 4, 1, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    metadata = None
    if packed:
        boundaries = torch.tensor([0, 1, 4], device="cuda", dtype=torch.int32)
        metadata = AttentionMetadata(
            extra={
                "cu_seqlens_q": boundaries,
                "cu_seqlens_k": boundaries,
                "max_seqlen_q": 3,
                "max_seqlen_k": 3,
            }
        )

    with torch.inference_mode():
        expected = impl.forward_cuda(q, k, v, metadata)
        compiled = torch.compile(impl.forward_cuda, fullgraph=True)
        actual = compiled(q, k, v, metadata)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
