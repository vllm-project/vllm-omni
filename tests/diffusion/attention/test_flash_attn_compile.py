# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import functools

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.attention.backends.utils import fa

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cuda]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_fa4_dense_dispatch_is_opaque_to_dynamic_torch_compile(monkeypatch, tmp_path):
    """Verify the compile boundary with a mocked FA4 dispatcher; real FA4 hardware is not required."""
    marker = tmp_path / "kernel"
    marker.write_text("loaded", encoding="utf-8")

    @functools.cache
    def cached_kernel_loader():
        with open(marker, encoding="utf-8") as handle:
            handle.read()

    def fake_attention(query, key, value, **_kwargs):
        cached_kernel_loader()
        return torch.empty_like(query)

    monkeypatch.setattr(fa, "HAS_FLASH_ATTN", True)
    monkeypatch.setattr(fa, "IS_FLASH_ATTN_4", True)
    monkeypatch.setattr(fa, "flash_attn_func", fake_attention)

    impl = FlashAttentionImpl(
        num_heads=8,
        head_size=64,
        softmax_scale=0.125,
        causal=False,
    )
    q = torch.randn(1, 16, 8, 64, device="cuda", dtype=torch.bfloat16)
    compiled = torch.compile(
        lambda query, key, value: impl.forward_cuda(query, key, value),
        fullgraph=True,
        dynamic=True,
    )
    out = compiled(q, q, q)

    assert out.shape == q.shape
