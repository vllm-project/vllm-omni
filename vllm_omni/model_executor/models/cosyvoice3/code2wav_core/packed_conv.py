# ruff: noqa: N803
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Grouped causal Conv1d + Mish directly on packed DiT rows.

The DiT position embedding is two grouped (16 groups), 31-tap left-padded
convolutions. cuDNN runs them as one small kernel per group and needs a padded
``(rows, width)`` scatter, transposes and a gather around every call. This
implicit-GEMM kernel reads the packed ``[tokens, channels]`` sequence directly:
taps before a row's first frame are masked, which is exactly the left zero
padding, so rows never mix and no padded layout is materialized.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _packed_causal_conv_mish_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    pos_ptr,
    out_ptr,
    total,
    channels,
    TAPS: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    group = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < total
    positions = tl.load(pos_ptr + rows, mask=row_mask, other=0)
    cin = tl.arange(0, GROUP)
    cout = tl.arange(0, GROUP)
    acc = tl.zeros((BLOCK_M, GROUP), dtype=tl.float32)
    for tap in range(TAPS):
        shift = TAPS - 1 - tap
        valid = row_mask & (positions >= shift)
        src = (rows - shift).to(tl.int64)
        a = tl.load(
            x_ptr + src[:, None] * channels + group * GROUP + cin[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        b = tl.load(w_ptr + ((group * TAPS + tap) * GROUP + cin[:, None]) * GROUP + cout[None, :])
        acc = tl.dot(a, b, acc)
    acc += tl.load(bias_ptr + group * GROUP + cout).to(tl.float32)[None, :]
    # Match eager rounding: the convolution output is stored in the input
    # dtype before Mish computes in float32 and rounds again.
    conv = acc.to(out_ptr.dtype.element_ty).to(tl.float32)
    softplus = tl.where(conv > 20.0, conv, tl.log(1.0 + tl.exp(conv)))
    e2 = tl.exp(-2.0 * softplus)
    mish = conv * (1.0 - e2) / (1.0 + e2)
    tl.store(
        out_ptr + rows.to(tl.int64)[:, None] * channels + group * GROUP + cout[None, :],
        mish.to(out_ptr.dtype.element_ty),
        mask=row_mask[:, None],
    )


def pack_conv_weight(conv: torch.nn.Conv1d) -> torch.Tensor:
    """[Cout, Cin/G, K] grouped weight -> [G, K, Cin/G, Cout/G] contiguous."""
    cout, cin_g, taps = conv.weight.shape
    groups = conv.groups
    return conv.weight.detach().view(groups, cout // groups, cin_g, taps).permute(0, 3, 2, 1).contiguous()


def packed_causal_conv_mish(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """x: [T, C] packed rows; positions: [T] int frame index within each row."""
    total, channels = x.shape
    groups, taps, group, _ = weight.shape
    out = torch.empty_like(x)
    if total:
        block_m = 64
        _packed_causal_conv_mish_kernel[(triton.cdiv(total, block_m), groups)](
            x, weight, bias, positions, out, total, channels, TAPS=taps, GROUP=group, BLOCK_M=block_m
        )
    return out
