# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.backends.ring import ring_kernels
from vllm_omni.diffusion.attention.backends.ring.ring_selector import AttnType
from vllm_omni.diffusion.attention.parallel import ring as ring_parallel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_fa4_wrapper_returns_lse_and_normalizes_unbounded_window(monkeypatch):
    expected_out = object()
    expected_lse = object()
    captured = {}

    def fake_fa4(q, k, v, **kwargs):
        captured.update(q=q, k=k, v=v, **kwargs)
        return expected_out, expected_lse

    monkeypatch.setattr(ring_kernels, "HAS_FA4", True)
    monkeypatch.setattr(ring_kernels, "fa4_attn_func", fake_fa4)

    out, lse = ring_kernels.flash_attn4_func_forward(
        "q",
        "k",
        "v",
        softmax_scale=0.125,
        causal=True,
        window_size=(-1, -1),
    )

    assert (out, lse) == (expected_out, expected_lse)
    assert captured == {
        "q": "q",
        "k": "k",
        "v": "v",
        "softmax_scale": 0.125,
        "causal": True,
        "window_size": (None, None),
        "softcap": 0.0,
        "return_lse": True,
    }


def test_ring_prefers_fa4_on_blackwell(monkeypatch):
    captured = {}

    def fake_ring_flash_attn_func(*args, **kwargs):
        captured.update(kwargs)
        return "output"

    monkeypatch.setattr(ring_parallel, "HAS_FA4", True)
    monkeypatch.setattr(ring_parallel, "HAS_FA3", True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (10, 3))

    from vllm_omni.diffusion.attention.backends import ring_flash_attn

    monkeypatch.setattr(ring_flash_attn, "ring_flash_attn_func", fake_ring_flash_attn_func)

    query = SimpleNamespace(
        dtype=torch.bfloat16,
        shape=(1, 16, 4, 128),
        is_cuda=True,
        device=torch.device("cuda:0"),
    )
    strategy = ring_parallel.RingParallelAttention(SimpleNamespace(ring_group=object()))

    assert strategy.run_attention(query, "key", "value", None) == "output"
    assert captured["attn_type"] is AttnType.FA4


def test_flashinfer_wrapper_canonicalizes_unbatched_lse_to_bhs(monkeypatch):
    seq_len = num_heads = 4
    query = torch.randn(1, seq_len, num_heads, 8)
    unbatched_out = torch.randn(seq_len, num_heads, 8)
    unbatched_lse = torch.arange(num_heads * seq_len, dtype=torch.float32).reshape(num_heads, seq_len)
    assert not torch.equal(unbatched_lse, unbatched_lse.transpose(0, 1))

    def fake_prefill(_q, _k, _v, **_kwargs):
        return unbatched_out.clone(), unbatched_lse.clone()

    monkeypatch.setattr(ring_kernels, "HAS_FLASHINFER", True)
    # FlashInfer symbols are only bound when HAS_FLASHINFER was true at import.
    monkeypatch.setattr(ring_kernels, "single_prefill_with_kv_cache", fake_prefill, raising=False)
    monkeypatch.setattr(ring_kernels, "_LOG2_E", 1.0, raising=False)

    out, lse = ring_kernels.flashinfer_attn_forward(query, query, query, softmax_scale=0.1)

    assert out.shape == query.shape
    assert lse.shape == (1, num_heads, seq_len)
    assert torch.equal(lse[0], unbatched_lse)
