# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end check of the Cosmos3 multiview FlashAttention-4 backend.

Requires a datacenter-Blackwell GPU and the optional ``vllm-omni[fa4]`` extra;
everything that can be verified without one lives in
``test_multiview_flex_attention.py``.
"""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.B200]

# Fixture UND capacity; production uses DEFAULT_MAX_UND_TOKENS (4098). 128 is
# the smallest legal value here because real_und_len is parametrized up to 128,
# and being exactly one FA4 KV block it reproduces the padded length the old
# prompt-dependent rule gave for both 96 and 128 -- so the dense oracle below
# compares against the same geometry it did before capacity padding. The 96 case
# leaves a pad region and the 128 case fills the block exactly, covering both.
_MAX_UND = 128


def _fa4_unavailable() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA is not available"
    if torch.cuda.get_device_capability()[0] != 10:
        return "FlashAttention-4 multiview backend targets datacenter Blackwell (SM100)"
    try:
        import flash_attn.cute  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"flash-attn-4 is not importable: {exc}"
    return None


_SKIP_REASON = _fa4_unavailable()
pytestmark.append(pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or ""))


def _dense_oracle(q, k_und, k, v_und, v, metadata, multiview_pair_predicate):
    """Masked dense attention over the unpadded sequence, in float32."""
    heads = q.shape[2]
    kv_heads = k.shape[2]
    repeat = heads // kv_heads
    dense_k = torch.cat([k_und, k], dim=1).repeat_interleave(repeat, dim=2).float()
    dense_v = torch.cat([v_und, v], dim=1).repeat_interleave(repeat, dim=2).float()
    allowed = multiview_pair_predicate(
        metadata,
        torch.arange(q.shape[1], device=q.device)[:, None],
        torch.arange(dense_k.shape[1], device=q.device)[None, :],
    )
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), dense_k) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~allowed[None, None], float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", scores.softmax(dim=-1), dense_v)


# Cosmos3 defaults to 32 query heads over 8 KV heads (transformer_cosmos3.py:
# num_attention_heads / num_key_value_heads), so the production path is 4:1 GQA,
# not MHA.  FA4 packs those query heads by default; upstream covers that exact
# shape against a head-broadcast block map in tests/cute/test_mask_mod.py, and
# these cases pin it down for the multiview mask.
HEAD_GEOMETRIES = [
    pytest.param(4, 4, id="mha"),
    pytest.param(32, 8, id="gqa-production-4to1"),
    pytest.param(8, 1, id="mqa"),
]


@pytest.mark.parametrize("num_heads,num_kv_heads", HEAD_GEOMETRIES)
@pytest.mark.parametrize("real_und_len", [128, 96])
def test_fa4_multiview_attention_matches_dense_oracle(real_und_len: int, num_heads: int, num_kv_heads: int) -> None:
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
        MultiviewAttentionContext,
        MultiviewLayout,
        build_multiview_flex_metadata,
        multiview_pair_predicate,
        padded_multiview_flex_attention,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim = 128

    # Two views x four latent frames x 8x8 patches -> 256 tokens per item,
    # 512 packed GEN tokens, which is exactly two 256-row FA4 query blocks.
    layout = MultiviewLayout(2, 4, 8, 8, condition_frame_indexes=(0, 2), backend="fa4", max_und_tokens=_MAX_UND)
    gen = layout.gen_tokens
    context = MultiviewAttentionContext(layout, {})

    q = torch.randn(1, gen, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, gen, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    k_und = torch.randn(1, real_und_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_und = torch.randn_like(k_und)

    actual = padded_multiview_flex_attention(q, k, v, k_und, v_und, context)
    assert actual.shape == (1, gen, num_heads, head_dim)

    metadata = build_multiview_flex_metadata(
        seq_len=real_und_len + gen,
        full_q_offsets=(real_und_len, real_und_len + layout.item_tokens, real_und_len + gen),
        items_per_sample=layout.mask_items(device),
        device=device,
        num_und=real_und_len,
        attention_scope=layout.attention_scope,
    )
    expected = _dense_oracle(q, k_und, k, v_und, v, metadata, multiview_pair_predicate)

    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("num_heads,num_kv_heads", HEAD_GEOMETRIES)
def test_fa4_and_triton_backends_agree(num_heads: int, num_kv_heads: int) -> None:
    """Both backends project the same predicate, so their outputs must match."""
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
        MultiviewAttentionContext,
        MultiviewLayout,
        padded_multiview_flex_attention,
    )

    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim = 128

    tensors = None
    outputs = {}
    for backend in ("triton", "fa4"):
        layout = MultiviewLayout(2, 4, 8, 8, condition_frame_indexes=(0, 2), backend=backend, max_und_tokens=_MAX_UND)
        gen = layout.gen_tokens
        if tensors is None:
            tensors = (
                torch.randn(1, gen, num_heads, head_dim, device=device, dtype=dtype),
                torch.randn(1, gen, num_kv_heads, head_dim, device=device, dtype=dtype),
                torch.randn(1, gen, num_kv_heads, head_dim, device=device, dtype=dtype),
                torch.randn(1, 128, num_kv_heads, head_dim, device=device, dtype=dtype),
                torch.randn(1, 128, num_kv_heads, head_dim, device=device, dtype=dtype),
            )
        context = MultiviewAttentionContext(layout, {})
        outputs[backend] = padded_multiview_flex_attention(*tensors, context)

    torch.testing.assert_close(outputs["fa4"].float(), outputs["triton"].float(), atol=2e-2, rtol=2e-2)
