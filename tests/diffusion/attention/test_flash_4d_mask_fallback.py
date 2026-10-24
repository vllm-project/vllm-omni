# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression test for #8830: FlashAttention must accept 4D block-causal masks.

``QwenImage21Transformer._build_block_causal_mask`` emits a dense 4D
``(batch, 1, seq, seq)`` mask for padded/heterogeneous batches, but the FlashAttention
``_forward_varlen_masked`` path requires a 2D key-padding mask and asserts
``attention_mask.ndim == 2``, crashing every request on such batches. The backend now
falls back to SDPA (which consumes arbitrary 4D masks) when a non-2D mask arrives.

These tests run on CPU and verify the fallback produces output identical to a reference
masked attention over the same 4D block-causal mask.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.attention.backends.utils import fa  # noqa: F401  (import side-effects on CPU)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

DEVICE = torch.device("cpu")


def _block_causal_mask(seq_len: int, blocks: list[tuple[int, int]]) -> torch.Tensor:
    """Cartesian 4D block-causal mask: `(q_idx >= kv_idx) or same_image_block`."""
    ids = -torch.ones(seq_len, dtype=torch.long)
    for bid, (start, end) in enumerate(blocks):
        ids[start:end] = bid
    q_idx = torch.arange(seq_len, device=DEVICE).view(-1, 1)
    kv_idx = torch.arange(seq_len, device=DEVICE).view(1, -1)
    same_block = (ids.view(-1, 1) == ids.view(1, -1)) & (ids.view(-1, 1) >= 0)
    return ((q_idx >= kv_idx) | same_block).view(1, 1, seq_len, seq_len)


def _reference_attn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    """Reference masked attention (exact matmul) matching the 4D mask semantics."""
    b, sq, h, d = q.shape
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * softmax_scale
    m = mask.expand(b, h, sq, sq)
    scores = scores.masked_fill(~m, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = attn.masked_fill(~m, 0.0)
    return torch.einsum("bhqk,bkhd->bqhd", attn, v)


def _make_impl(**overrides) -> FlashAttentionImpl:
    head_size = overrides.get("head_size", 8)
    kwargs = dict(num_heads=4, head_size=head_size, softmax_scale=1.0 / (head_size**0.5), causal=False)
    kwargs.update(overrides)
    return FlashAttentionImpl(**kwargs)


def _run_fallback(q, k, v, mask, impl) -> torch.Tensor:
    """Invoke the SDPA-fallback path exactly as forward_cuda would dispatch it."""
    assert mask.ndim != 2
    return impl._forward_4d_masked_sdpa(q, k, v, mask)


@pytest.mark.parametrize(
    "blocks",
    [
        [(0, 6)],  # single full-attention block
        [(1, 3), (4, 6)],  # two bidirectional blocks with causal gaps
        [(2, 4)],  # middle block
    ],
    ids=["full-block", "two-blocks-causal-gaps", "middle-block"],
)
def test_4d_masked_fallback_matches_reference(blocks):
    torch.manual_seed(0)
    b, sq, h, dh = 2, 6, 4, 8
    q = torch.randn(b, sq, h, dh, dtype=torch.float64)
    k = torch.randn(b, sq, h, dh, dtype=torch.float64)
    v = torch.randn(b, sq, h, dh, dtype=torch.float64)
    mask = _block_causal_mask(sq, blocks).expand(b, 1, -1, -1).to(torch.bool)

    impl = _make_impl()
    out = _run_fallback(q, k, v, mask, impl)
    ref = _reference_attn(q, k, v, mask, impl.softmax_scale)

    assert out.shape == q.shape
    assert torch.allclose(out, ref, atol=1e-6)


def test_4d_masked_fallback_with_key_padding():
    """The same mask combined with a 2D key-validity mask (the #8830 shape)."""
    torch.manual_seed(1)
    b, sq, h, dh = 2, 6, 4, 8
    q = torch.randn(b, sq, h, dh, dtype=torch.float64)
    k = torch.randn(b, sq, h, dh, dtype=torch.float64)
    v = torch.randn(b, sq, h, dh, dtype=torch.float64)

    block_mask = _block_causal_mask(sq, [(1, 3), (4, 6)])
    pad = torch.ones(b, sq, dtype=torch.bool)
    pad[0, 4:] = False  # one sample pads its tail (prompt padding)
    mask = (block_mask.expand(b, 1, -1, -1) & pad.view(b, 1, 1, sq)).to(torch.bool)

    impl = _make_impl()
    out = _run_fallback(q, k, v, mask, impl)
    ref = _reference_attn(q, k, v, mask, impl.softmax_scale)
    assert torch.allclose(out, ref, atol=1e-6)


def test_forward_cuda_dispatches_4d_mask_to_sdpa(monkeypatch):
    """Regression guard: forward_cuda must route a 4D mask to the SDPA fallback.

    Patches HAS_FLASH_ATTN so the CUDA-only kernel gate passes on CPU, then drives
    the real dispatch with a 4D block-causal mask. If the dispatch line
    (`attention_mask.ndim != 2`) is ever lost, this test fails — either by calling
    the varlen path (stubbed to raise) or by producing wrong output.
    """
    import vllm_omni.diffusion.attention.backends.utils.fa as fa_mod

    monkeypatch.setattr(fa_mod, "HAS_FLASH_ATTN", True)

    def _must_not_call_varlen(*args, **kwargs):
        raise AssertionError("4D mask must NOT reach the varlen path")

    monkeypatch.setattr(fa_mod, "flash_attn_varlen_func", _must_not_call_varlen)

    torch.manual_seed(0)
    b, sq, h, dh = 2, 6, 4, 8
    q = torch.randn(b, sq, h, dh, dtype=torch.float64)
    k = torch.randn(b, sq, h, dh, dtype=torch.float64)
    v = torch.randn(b, sq, h, dh, dtype=torch.float64)
    mask = _block_causal_mask(sq, [(1, 3), (4, 6)]).expand(b, 1, -1, -1).to(torch.bool)

    impl = _make_impl()
    out = impl.forward_cuda(
        q,
        k,
        v,
        AttentionMetadata(attn_mask=mask, full_attn_spans=None, extra={"attention_mask_mode": "arbitrary"}),
    )
    ref = _reference_attn(q, k, v, mask, impl.softmax_scale)
    assert out.shape == q.shape
    assert torch.allclose(out, ref, atol=1e-6)


def test_forward_cuda_keeps_2d_mask_on_varlen_path(monkeypatch):
    """A 2D key-padding mask must NOT be diverted to the SDPA fallback."""
    import vllm_omni.diffusion.attention.backends.utils.fa as fa_mod

    monkeypatch.setattr(fa_mod, "HAS_FLASH_ATTN", True)

    b, sq, h, dh = 1, 6, 4, 8
    q = torch.randn(b, sq, h, dh)
    k = torch.randn(b, sq, h, dh)
    v = torch.randn(b, sq, h, dh)
    mask2d = torch.ones(b, sq, dtype=torch.bool)
    mask2d[0, 3:] = False

    impl = _make_impl()
    calls = []

    def _fake_varlen(query, key, value, attention_mask, qwen21_decode_varlen=None):
        calls.append(attention_mask.ndim)
        return query

    impl._forward_varlen_masked = _fake_varlen
    out = impl.forward_cuda(
        q,
        k,
        v,
        AttentionMetadata(attn_mask=mask2d, full_attn_spans=None, extra={"attention_mask_mode": "arbitrary"}),
    )
    assert calls == [2], "2D mask must reach the varlen path"
    assert out is q


def test_4d_masked_fallback_preserves_output_shape():
    """A broadcastable 4D mask preserves the BSHD output shape."""
    torch.manual_seed(2)
    b, sq, h, dh = 1, 6, 4, 8
    q = torch.randn(b, sq, h, dh)
    k = torch.randn(b, sq, h, dh)
    v = torch.randn(b, sq, h, dh)
    # This case exercises the Qwen 4D layout, not rejection of other mask ranks.
    mask4d = torch.ones(b, 1, sq, sq, dtype=torch.bool)
    impl = _make_impl()
    out = impl._forward_4d_masked_sdpa(q, k, v, mask4d)
    assert out.shape == q.shape
