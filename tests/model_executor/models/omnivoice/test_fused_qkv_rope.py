# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused attention prologue must equal the six steps it replaces.

``fused_qkv_norm_rope`` collapses split, per-head RMSNorm of Q and K, RoPE on
both, the GQA broadcast of K and V, and the transpose into SDPA's layout into
one Triton pass. That is a lot of behaviour to fold into one kernel, so these
check it against the unfused reference across the geometries and dtypes it
claims to support, and check that unsupported geometries fall back rather than
producing something wrong.
"""

from __future__ import annotations

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from vllm_omni.model_executor.models.omnivoice.fused_qkv_rope import (
    _eager_qkv_norm_rope,
    fused_cuda_supported,
    fused_qkv_norm_rope,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
triton_only = pytest.mark.skipif(not HAS_TRITON, reason="Triton required")

EPS = 1e-6
TOL = {
    torch.float32: dict(atol=2e-5, rtol=2e-5),
    torch.float16: dict(atol=4e-3, rtol=4e-3),
    torch.bfloat16: dict(atol=3e-2, rtol=3e-2),
}


def _inputs(batch, seq_len, num_heads, num_kv_heads, head_dim, dtype, device="cuda"):
    torch.manual_seed(11)
    total = num_heads + 2 * num_kv_heads
    qkv = torch.randn(batch, seq_len, total, head_dim, device=device, dtype=dtype)
    q_weight = torch.randn(head_dim, device=device, dtype=dtype)
    k_weight = torch.randn(head_dim, device=device, dtype=dtype)
    freqs = torch.randn(seq_len, head_dim // 2, device=device)
    rope_table = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(dtype)
    return qkv, q_weight, k_weight, rope_table


@cuda_only
@triton_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "batch,seq_len,num_heads,num_kv_heads,head_dim",
    [
        (2, 44, 16, 8, 128),  # OmniVoice: the CFG pair, 2:1 GQA
        (2, 130, 16, 8, 128),
        (1, 7, 8, 8, 128),  # no GQA broadcast
        (3, 16, 16, 8, 64),
        (2, 33, 32, 8, 128),  # 4:1 GQA
    ],
)
def test_matches_the_unfused_reference(dtype, batch, seq_len, num_heads, num_kv_heads, head_dim):
    qkv, q_weight, k_weight, rope_table = _inputs(batch, seq_len, num_heads, num_kv_heads, head_dim, dtype)
    assert fused_cuda_supported(qkv, num_heads, num_kv_heads)

    got = fused_qkv_norm_rope(qkv, q_weight, k_weight, rope_table, EPS, num_heads, num_kv_heads)
    expected = _eager_qkv_norm_rope(qkv, q_weight, k_weight, rope_table, EPS, num_heads, num_kv_heads)

    for name, a, b in zip(("q", "k", "v"), got, expected):
        assert a.shape == (batch, num_heads, seq_len, head_dim), name
        assert a.is_contiguous(), name
        torch.testing.assert_close(a, b, msg=lambda m, n=name: f"{n}: {m}", **TOL[dtype])


@cuda_only
@triton_only
def test_v_is_the_untouched_projection_broadcast_into_place():
    """V gets no norm and no rotation, only the GQA broadcast and the transpose."""
    batch, seq_len, num_heads, num_kv_heads, head_dim = 2, 44, 16, 8, 128
    qkv, q_weight, k_weight, rope_table = _inputs(batch, seq_len, num_heads, num_kv_heads, head_dim, torch.float32)

    _, _, v = fused_qkv_norm_rope(qkv, q_weight, k_weight, rope_table, EPS, num_heads, num_kv_heads)

    raw_v = qkv[:, :, num_heads + num_kv_heads :]
    expected = raw_v.repeat_interleave(num_heads // num_kv_heads, dim=2).permute(0, 2, 1, 3)
    torch.testing.assert_close(v, expected.contiguous(), atol=0, rtol=0)


@cuda_only
@triton_only
def test_kv_heads_are_broadcast_not_reordered():
    """Query group g must see KV head g // repeat, not some transposed pairing."""
    batch, seq_len, num_heads, num_kv_heads, head_dim = 1, 4, 16, 8, 128
    qkv, q_weight, k_weight, rope_table = _inputs(batch, seq_len, num_heads, num_kv_heads, head_dim, torch.float32)

    _, k, _ = fused_qkv_norm_rope(qkv, q_weight, k_weight, rope_table, EPS, num_heads, num_kv_heads)

    repeat = num_heads // num_kv_heads
    for head in range(num_heads):
        torch.testing.assert_close(k[:, head], k[:, (head // repeat) * repeat], atol=0, rtol=0)


@cuda_only
@triton_only
def test_position_zero_is_a_pure_rmsnorm():
    """A rope table row of cos=1, sin=0 must leave the normalized value alone."""
    batch, seq_len, num_heads, num_kv_heads, head_dim = 1, 4, 8, 8, 128
    qkv, q_weight, k_weight, _ = _inputs(batch, seq_len, num_heads, num_kv_heads, head_dim, torch.float32)
    half = head_dim // 2
    rope_table = torch.cat(
        [torch.ones(seq_len, half, device="cuda"), torch.zeros(seq_len, half, device="cuda")], dim=-1
    )

    q, _, _ = fused_qkv_norm_rope(qkv, q_weight, k_weight, rope_table, EPS, num_heads, num_kv_heads)

    raw_q = qkv[:, :, :num_heads].to(torch.float32)
    variance = raw_q.pow(2).mean(-1, keepdim=True)
    expected = (raw_q * torch.rsqrt(variance + EPS) * q_weight).permute(0, 2, 1, 3)
    torch.testing.assert_close(q, expected.contiguous(), **TOL[torch.float32])


@cuda_only
@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_dim",
    [(12, 4, 128), (16, 8, 96), (16, 6, 128)],
)
def test_unsupported_geometry_falls_back_and_still_matches(num_heads, num_kv_heads, head_dim):
    """Head counts the tiling cannot split, or a head_dim that is not a power of two."""
    batch, seq_len = 2, 8
    qkv, q_weight, k_weight, rope_table = _inputs(batch, seq_len, num_heads, num_kv_heads, head_dim, torch.float32)
    assert not fused_cuda_supported(qkv, num_heads, num_kv_heads)

    got = fused_qkv_norm_rope(qkv, q_weight, k_weight, rope_table, EPS, num_heads, num_kv_heads)
    expected = _eager_qkv_norm_rope(qkv, q_weight, k_weight, rope_table, EPS, num_heads, num_kv_heads)
    for a, b in zip(got, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.cpu
def test_cpu_tensors_take_the_eager_path():
    batch, seq_len, num_heads, num_kv_heads, head_dim = 1, 5, 16, 8, 128
    qkv, q_weight, k_weight, rope_table = _inputs(
        batch, seq_len, num_heads, num_kv_heads, head_dim, torch.float32, device="cpu"
    )
    assert not fused_cuda_supported(qkv, num_heads, num_kv_heads)

    q, k, v = fused_qkv_norm_rope(qkv, q_weight, k_weight, rope_table, EPS, num_heads, num_kv_heads)
    assert q.shape == k.shape == v.shape == (batch, num_heads, seq_len, head_dim)


@pytest.mark.cpu
def test_a_mismatched_head_count_raises():
    qkv = torch.randn(1, 4, 30, 128)  # 16 + 2*8 would be 32
    w = torch.randn(128)
    table = torch.zeros(4, 128)
    with pytest.raises(ValueError, match="packed QKV"):
        fused_qkv_norm_rope(qkv, w, w, table, EPS, 16, 8)


@pytest.mark.cpu
def test_a_short_rope_table_raises():
    qkv = torch.randn(1, 10, 32, 128)
    w = torch.randn(128)
    table = torch.zeros(4, 128)
    with pytest.raises(ValueError, match="rope_table"):
        fused_qkv_norm_rope(qkv, w, w, table, EPS, 16, 8)
