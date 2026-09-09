# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F
from vllm.triton_utils import HAS_TRITON

from vllm_omni.diffusion.layers.rope import apply_rotary_emb_torch

pytestmark = [pytest.mark.core_model, pytest.mark.cuda, pytest.mark.diffusion]

_HEAD_DIM = 128
_ROTARY_DIM = 96
_EPS = 1e-5


def _reference(q, k, q_weight, k_weight, rope_table):
    q = F.rms_norm(q, (_HEAD_DIM,), q_weight, _EPS)
    k = F.rms_norm(k, (_HEAD_DIM,), k_weight, _EPS)
    half = _ROTARY_DIM // 2
    cos = rope_table[..., :half].unsqueeze(1)
    sin = rope_table[..., half:].unsqueeze(1)

    def apply(x):
        first = x[..., :half]
        second = x[..., half:_ROTARY_DIM]
        return torch.cat(
            (
                first * cos - second * sin,
                second * cos + first * sin,
                x[..., _ROTARY_DIM:],
            ),
            dim=-1,
        )

    return apply(q), apply(k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("seq_len", [1, 257, 1024])
def test_fused_qk_norm_rope_matches_bf16_reference(seq_len):
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
        fused_qk_norm_rope,
    )

    torch.manual_seed(17)
    heads = 14
    qkv = torch.randn(
        seq_len,
        heads * _HEAD_DIM * 3,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q = qkv[:, : heads * _HEAD_DIM].view(seq_len, heads, _HEAD_DIM)
    k = qkv[:, heads * _HEAD_DIM : 2 * heads * _HEAD_DIM].view(
        seq_len,
        heads,
        _HEAD_DIM,
    )
    q_weight = torch.randn(_HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn(_HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    freqs = torch.randn(seq_len, _ROTARY_DIM // 2, device="cuda")
    rope_table = torch.cat((torch.cos(freqs), torch.sin(freqs)), dim=-1).to(torch.bfloat16)

    expected_q, expected_k = _reference(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
    )
    actual_q, actual_k = fused_qk_norm_rope(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        _EPS,
    )

    torch.testing.assert_close(actual_q, expected_q, atol=0.0625, rtol=0.02)
    torch.testing.assert_close(actual_k, expected_k, atol=0.0625, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("batched", [False, True])
def test_fused_qk_norm_rope_interleaved_supports_shared_layouts(batched):
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
        fused_qk_norm_rope_interleaved,
        fused_qk_norm_rope_interleaved_supported,
    )

    torch.manual_seed(23)
    batch = 2
    seq_len = 17
    q_heads = 7
    k_heads = 3
    head_dim = 64
    rotary_dim = head_dim
    shape_prefix = (batch, seq_len) if batched else (seq_len,)
    q = torch.randn(*shape_prefix, q_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(*shape_prefix, k_heads, head_dim, device="cuda", dtype=torch.float16)
    q_weight = torch.randn(head_dim, device="cuda")
    k_weight = torch.randn(head_dim, device="cuda")
    freqs = torch.randn(seq_len, rotary_dim // 2, device="cuda")
    cos = torch.cos(freqs).to(q.dtype)
    sin = torch.sin(freqs).to(q.dtype)

    expected_q = apply_rotary_emb_torch(
        F.rms_norm(q, (head_dim,), q_weight, _EPS),
        cos,
        sin,
        interleaved=True,
    )
    expected_k = apply_rotary_emb_torch(
        F.rms_norm(k, (head_dim,), k_weight, _EPS),
        cos,
        sin,
        interleaved=True,
    )

    assert fused_qk_norm_rope_interleaved_supported(q, k, cos, sin)
    actual_q, actual_k = fused_qk_norm_rope_interleaved(q, k, q_weight, k_weight, cos, sin, _EPS)

    torch.testing.assert_close(actual_q, expected_q, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual_k, expected_k, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_fused_qk_norm_rope_interleaved_supports_boogu_geometry():
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
        fused_qk_norm_rope_interleaved,
        fused_qk_norm_rope_interleaved_supported,
    )

    torch.manual_seed(29)
    seq_len = 4139
    q_heads = 28
    k_heads = 7
    head_dim = 120
    qkv = torch.randn(
        seq_len,
        (q_heads + 2 * k_heads) * head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q, k, _ = qkv.split(
        (q_heads * head_dim, k_heads * head_dim, k_heads * head_dim),
        dim=-1,
    )
    q = q.unflatten(-1, (q_heads, head_dim))
    k = k.unflatten(-1, (k_heads, head_dim))
    q_weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
    freqs = torch.randn(seq_len, head_dim // 2, device="cuda")
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    expected_q = apply_rotary_emb_torch(
        F.rms_norm(q, (head_dim,), q_weight, _EPS),
        cos,
        sin,
        interleaved=True,
    ).to(q.dtype)
    expected_k = apply_rotary_emb_torch(
        F.rms_norm(k, (head_dim,), k_weight, _EPS),
        cos,
        sin,
        interleaved=True,
    ).to(k.dtype)

    assert fused_qk_norm_rope_interleaved_supported(q, k, cos, sin)
    actual_q, actual_k = fused_qk_norm_rope_interleaved(q, k, q_weight, k_weight, cos, sin, _EPS)

    torch.testing.assert_close(actual_q, expected_q, atol=0.0625, rtol=0.02)
    torch.testing.assert_close(actual_k, expected_k, atol=0.0625, rtol=0.02)
