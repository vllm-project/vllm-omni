# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ruff: noqa: N803

"""Fused Adaptive Group Normalization (AdaGN) + SiLU operator.

This module implements the fused AdaGN+SiLU pattern commonly used in Diffusion
Transformer ResBlocks:

    output = SiLU(GroupNorm(x, weight, bias) * (1 + scale) + shift)

where ``scale`` and ``shift`` are per-(batch, channel) conditioning signals
derived from the timestep embedding. Eager PyTorch spends four kernels and three
full-size intermediates on this; the fused version does it in one pass.

Falls back to native PyTorch ops when Triton is unavailable.
"""

import torch
import torch.nn.functional as F
from vllm.triton_utils import HAS_TRITON, tl, triton


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=6),
    ],
    key=["spatial_size", "C"],
)
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
    # Shape info; x is contiguous (B, C, spatial_size)
    C,
    spatial_size,
    num_groups: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    """
    Fused AdaGN + SiLU kernel: SiLU(GroupNorm(x) * (1 + scale) + shift).
    Mean and variance use a parallel Welford reduction (block-level register
    reduction merged with the Chan formula), which avoids the catastrophic
    cancellation of E[x^2] - E[x]^2 on large-offset inputs. x is read once for
    the statistics and once for normalization (2 reads total).
    One program handles each (batch, group) pair. Channels are processed serially,
    while the spatial axis is vectorized for diffusion workloads.
    Moments are accumulated in fp32 to match PyTorch numerics.
    """
    pid = tl.program_id(0)

    group_size = C // num_groups
    n_idx = pid // num_groups
    g_idx = pid % num_groups

    # === Pass 1: Welford reduction over the whole group (fp32) ===
    # Each block is reduced in registers (block mean, then centered M2), so the
    # loaded values are reused instead of re-reading memory; blocks are merged
    # with the Chan et al. parallel-Welford formula. This keeps both the mean and
    # the variance accurate for inputs like ``10000 +- 0.1``, where the naive
    # ``E[x^2] - E[x]^2`` cancels catastrophically, and reads x only once for the
    # statistics (2 reads total including normalize).
    n_total = tl.zeros([1], dtype=tl.float32)
    mean_total = tl.zeros([1], dtype=tl.float32)
    m2_total = tl.zeros([1], dtype=tl.float32)

    for c_offset in range(group_size):
        c_idx = g_idx * group_size + c_offset
        base = n_idx * C * spatial_size + c_idx * spatial_size

        for s_start in tl.range(0, spatial_size, BLOCK_SIZE, num_stages=num_stages):
            offsets = s_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < spatial_size

            x_val = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
            x_val = x_val.to(tl.float32)

            # block-level reduction in registers (no extra memory traffic)
            n = tl.sum(tl.where(mask, 1.0, 0.0), axis=0)
            bsum = tl.sum(x_val, axis=0)
            bmean = bsum / n
            bm2 = tl.sum(tl.where(mask, (x_val - bmean) * (x_val - bmean), 0.0), axis=0)

            # Chan et al. merge into the running (n, mean, m2)
            delta = bmean - mean_total
            new_n = n_total + n
            mean_total = mean_total + delta * (n / new_n)
            m2_total = m2_total + bm2 + delta * delta * (n_total * n / new_n)
            n_total = new_n

    mean = mean_total
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

        for s_start in tl.range(0, spatial_size, BLOCK_SIZE, num_stages=num_stages):
            offsets = s_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < spatial_size

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

    - x: (B, C, *spatial), any spatial rank ≥ 1.
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

    # Only B*num_groups programs are launched, which is well under the SM count
    # for typical diffusion batches. A memory-bound kernel can still saturate
    # HBM from few CTAs, but only with enough loads in flight, so widen the CTA
    # for the large activations instead of leaving it at the 4-warp default.

    # One program per (batch, group) pair.
    grid = (B * num_groups,)

    _adaptive_group_norm_silu_kernel[grid](
        x_flat,
        out_flat,
        weight,
        bias,
        scale_2d,
        shift_2d,
        C,
        spatial_size,
        num_groups=num_groups,
        eps=eps,
    )

    return out_flat.reshape(orig_shape)


__all__ = ["fused_adaptive_group_norm_silu"]
