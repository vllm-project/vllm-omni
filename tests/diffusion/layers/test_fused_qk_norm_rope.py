# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F
from vllm.triton_utils import HAS_TRITON

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


# ---------------------------------------------------------------------------
# General geometry (any even head_dim, here Boogu-Image's 120) in both
# pairing modes, against the module's own eager reference at the same
# tolerance as the MiniMax-H3 test above.
# ---------------------------------------------------------------------------

_BOOGU_HEAD_DIM = 120


def _boogu_inputs(strided: bool):
    torch.manual_seed(11)
    if strided:
        # Slices of a wider head axis: the op must honour q/k strides
        # (merged-QKV projections hand the op such views).
        q = torch.randn(4139, 35, _BOOGU_HEAD_DIM, device="cuda", dtype=torch.bfloat16)[:, :28]
        k = torch.randn(4139, 35, _BOOGU_HEAD_DIM, device="cuda", dtype=torch.bfloat16)[:, :7]
    else:
        q = torch.randn(4139, 28, _BOOGU_HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(4139, 7, _BOOGU_HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    q_weight = torch.randn(_BOOGU_HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn(_BOOGU_HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    freqs = torch.randn(4139, _BOOGU_HEAD_DIM // 2, device="cuda", dtype=torch.float32)
    rope_table = torch.cat((torch.cos(freqs), torch.sin(freqs)), dim=-1)
    return q, k, q_weight, k_weight, rope_table


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("strided", [False, True])
def test_fused_qk_norm_rope_interleaved(strided):
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
        _eager_qk_norm_rope,
        _launch_fused_qk_norm_rope,
        fused_qk_norm_rope,
    )

    q, k, q_weight, k_weight, rope_table = _boogu_inputs(strided)
    expected = _eager_qk_norm_rope(q, k, q_weight, k_weight, rope_table, _EPS, _BOOGU_HEAD_DIM, _BOOGU_HEAD_DIM, True)
    # Check the public op AND the launcher directly: the latter cannot fall
    # back to eager, so a silent dispatch regression cannot go green here.
    for actual in (
        fused_qk_norm_rope(q, k, q_weight, k_weight, rope_table, _EPS, interleaved=True),
        _launch_fused_qk_norm_rope(q, k, q_weight, k_weight, rope_table, _EPS, interleaved=True),
    ):
        torch.testing.assert_close(actual[0], expected[0], atol=0.0625, rtol=0.02)
        torch.testing.assert_close(actual[1], expected[1], atol=0.0625, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_fused_qk_norm_rope_half_split_general_dim():
    """head_dim=120 off the MiniMax-H3 128 pin, half-split pairing.

    Exercises the generalised combined kernel's half-split mode via the
    launcher. It is not routed in production (half-split traffic keeps the
    untouched pre-existing per-tensor kernel and its 128/96 contract, so the
    public op falls back to eager at this geometry), pending the
    maintainers' call.
    """
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
        _eager_qk_norm_rope,
        _launch_fused_qk_norm_rope,
    )

    q, k, q_weight, k_weight, rope_table = _boogu_inputs(strided=False)
    expected = _eager_qk_norm_rope(q, k, q_weight, k_weight, rope_table, _EPS, _BOOGU_HEAD_DIM, _BOOGU_HEAD_DIM)
    actual = _launch_fused_qk_norm_rope(q, k, q_weight, k_weight, rope_table, _EPS, interleaved=False)
    torch.testing.assert_close(actual[0], expected[0], atol=0.0625, rtol=0.02)
    torch.testing.assert_close(actual[1], expected[1], atol=0.0625, rtol=0.02)
