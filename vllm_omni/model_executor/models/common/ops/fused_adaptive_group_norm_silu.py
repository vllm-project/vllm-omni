# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# ruff: noqa: N803

"""Fused Adaptive Group Normalization (AdaGN) + SiLU operator.

This module implements the fused AdaGN+SiLU pattern commonly used in Diffusion
Transformer ResBlocks:

    output = SiLU(GroupNorm(x, weight, bias) * (1 + scale) + shift)

where ``scale`` and ``shift`` are per-(batch, channel) conditioning signals
derived from the timestep embedding. Eager PyTorch spends four kernels and three
full-size intermediates on this; the fused version does it in one pass.

Large activations additionally take a split reduction so the work is spread over
the whole device instead of ``B * num_groups`` CTAs; see
:mod:`._group_norm_reduction` for why and how. On an A10G (80 SMs), bf16, batch
1, that is worth 1.46-1.48x over the unsplit kernel at every level of the
HunyuanImage3 decode ladder, and puts the 1024^2 activation at 498 GB/s -- the
same rate a plain device-to-device copy of that footprint sustains, so the
kernel is now at the streaming limit rather than short of it. Small activations
keep the single-launch path, where the widened autotune space is what helps
instead (1.25-1.36x on the 32x32 sizes, which the old fixed 4096-wide block
served badly).

Falls back to native PyTorch ops when Triton is unavailable.
"""

import torch
import torch.nn.functional as F
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_omni.model_executor.models.common.ops._group_norm_reduction import (
    SPLIT_REDUCTION_CONFIGS,
    SPLIT_REDUCTION_KEY,
    SPLIT_REDUCTION_PRUNE,
    launch_partial_stats,
    pick_split,
    welford_combine,
    welford_group_range,
)


@triton.autotune(configs=SPLIT_REDUCTION_CONFIGS, key=SPLIT_REDUCTION_KEY, prune_configs_by=SPLIT_REDUCTION_PRUNE)
@triton.jit
def _adaptive_group_norm_silu_kernel(
    # Input/output pointers
    x_ptr,
    out_ptr,
    # Affine parameters
    weight_ptr,
    bias_ptr,
    # Conditioning signals, both (B, C) contiguous
    scale_ptr,
    shift_ptr,
    # Partial-statistics workspace, unused (and passed as 0) when SPLIT == 1
    ws_ptr,
    # Shape info; x is contiguous (B, C, spatial_size)
    C,
    spatial_size,
    split_chunk,
    num_groups: tl.constexpr,
    eps: tl.constexpr,
    # Split reduction
    SPLIT: tl.constexpr,
    SPLIT_POW2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    """
    Fused AdaGN + SiLU kernel: SiLU(GroupNorm(x) * (1 + scale) + shift).
    Mean and variance use a parallel Welford reduction, which avoids the
    catastrophic cancellation of E[x^2] - E[x]^2 on large-offset inputs. x is read
    once for the statistics and once for normalization (2 reads total).
    Moments are accumulated in fp32 to match PyTorch numerics.

    ``SPLIT`` programs cooperate on each (batch, group) pair; see the sibling
    ``_group_norm_silu_kernel`` and :mod:`._group_norm_reduction` for the
    rationale. Only the epilogue below differs between the two operators.
    """
    pid = tl.program_id(0)
    bg = pid // SPLIT
    s = pid % SPLIT

    group_size = C // num_groups
    n_idx = bg // num_groups
    g_idx = bg % num_groups

    # === Pass 1: group statistics (fp32) ===
    if SPLIT == 1:
        lo = 0
        hi = spatial_size
        n_total, mean, m2_total = welford_group_range(
            x_ptr, n_idx, g_idx, C, spatial_size, group_size, lo, hi, BLOCK_SIZE, num_stages
        )
    else:
        lo = s * split_chunk
        hi = tl.minimum(lo + split_chunk, spatial_size)
        n_total, mean, m2_total = welford_combine(ws_ptr, bg, SPLIT, SPLIT_POW2)

    var = m2_total / n_total
    rstd = 1.0 / tl.sqrt(var + eps)

    # === Pass 2: normalize, affine, then adaptive modulation ===
    for c_offset in range(group_size):
        c_idx = g_idx * group_size + c_offset
        base = n_idx * C * spatial_size + c_idx * spatial_size

        weight_val = tl.load(weight_ptr + c_idx).to(tl.float32)
        bias_val = tl.load(bias_ptr + c_idx).to(tl.float32)
        scale_val = tl.load(scale_ptr + n_idx * C + c_idx).to(tl.float32)
        shift_val = tl.load(shift_ptr + n_idx * C + c_idx).to(tl.float32)

        for s_start in tl.range(lo, hi, BLOCK_SIZE, num_stages=num_stages):
            offsets = s_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < hi

            x_val = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
            x_val = x_val.to(tl.float32)

            # GroupNorm: (x - mean) * rstd * weight + bias
            norm_val = (x_val - mean) * rstd * weight_val + bias_val

            # AdaGN: norm * (1 + scale) + shift
            out_val = norm_val * (1.0 + scale_val) + shift_val

            # SiLU: out * sigmoid(out)
            out_val = out_val * tl.sigmoid(out_val)

            # ``tl.store`` casts to the output pointer's dtype, which the
            # caller picked to match eager GroupNorm's autocast behaviour.
            tl.store(out_ptr + base + offsets, out_val, mask=mask)


def fused_adaptive_group_norm_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    num_groups: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused Adaptive GroupNorm + SiLU.

    Computes SiLU(GroupNorm(x) * (1 + scale) + shift) in one fused operation, avoiding intermediate tensors.

    - x: (B, C, *spatial), any spatial rank >= 1.
    - weight, bias: per-channel parameters, shape (C,).
    - scale, shift: adaptive parameters with B * C elements, broadcastable per channel.
    - num_groups: GroupNorm group count.
    - eps: numerical stability epsilon.

    Returns a tensor with the same shape as x, matching eager F.group_norm dtype behavior.

    Spatial dimensions are flattened during computation and restored afterward.
    GroupNorm statistics cover each channel group across all spatial positions, so the result is exact.
    """
    assert x.ndim >= 3, f"Expected at least 3D input (B, C, *spatial), got {x.ndim}D"
    B, C = x.shape[:2]
    assert C % num_groups == 0, f"num_channels ({C}) must be divisible by num_groups ({num_groups})"
    assert weight.ndim == 1 and weight.size(0) == C, f"Weight shape {tuple(weight.shape)} doesn't match channels {C}"
    assert bias.ndim == 1 and bias.size(0) == C, f"Bias shape {tuple(bias.shape)} doesn't match channels {C}"
    assert scale.numel() == B * C, f"scale has {scale.numel()} elements, expected B*C = {B * C}"
    assert shift.numel() == B * C, f"shift has {shift.numel()} elements, expected B*C = {B * C}"

    # Fallback if Triton not available (NPU, CPU, ...)
    if not HAS_TRITON:
        broadcast = (B, C) + (1,) * (x.ndim - 2)
        normed = F.group_norm(x, num_groups, weight, bias, eps)
        return F.silu(normed * (1.0 + scale.reshape(broadcast)) + shift.reshape(broadcast))

    orig_shape = x.shape

    # The kernel indexes x as a dense (B, C, spatial) block, so normalize the
    # layout here. ``reshape``/``contiguous`` are no-ops for the common case of
    # a contiguous activation coming out of a conv.
    x_flat = x.contiguous().reshape(B, C, -1)
    spatial_size = x_flat.size(2)

    # ``reshape`` rather than ``view``: scale/shift typically arrive as halves
    # of a torch.chunk along dim 1, which is not contiguous when B > 1.
    scale_2d = scale.reshape(B, C).contiguous()
    shift_2d = shift.reshape(B, C).contiguous()

    # Allocate with the dtype eager GroupNorm would return, so the fused path
    # stays a drop-in replacement inside autocast regions.
    out_dtype = x_flat.dtype
    if torch.is_autocast_enabled(x_flat.device.type):
        out_dtype = torch.float32
    out_flat = torch.empty_like(x_flat, dtype=out_dtype)

    # One program per (batch, group) pair would be B*num_groups CTAs -- 32 for a
    # B=1 decode, well under any modern SM count. Split the spatial axis so the
    # largest activations get enough CTAs to saturate memory; small ones fall
    # back to split=1, which is the original single-launch kernel.
    split, split_chunk = pick_split(spatial_size, B, num_groups, x_flat.device)

    ws = 0
    if split > 1:
        ws = launch_partial_stats(x_flat, B, C, spatial_size, num_groups, split, split_chunk)

    _adaptive_group_norm_silu_kernel[(B * num_groups * split,)](
        x_flat,
        out_flat,
        weight,
        bias,
        scale_2d,
        shift_2d,
        ws,
        C,
        spatial_size,
        split_chunk,
        num_groups=num_groups,
        eps=eps,
        SPLIT=split,
        SPLIT_POW2=1 << (split - 1).bit_length(),
    )

    return out_flat.reshape(orig_shape)


__all__ = ["fused_adaptive_group_norm_silu"]
