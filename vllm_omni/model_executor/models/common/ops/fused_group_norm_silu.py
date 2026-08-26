# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# ruff: noqa: N803

"""Fused GroupNorm + SiLU operator.

This operator fuses GroupNorm followed by SiLU activation into a single kernel,
reducing memory traffic and kernel launch overhead. The implementation uses
Triton for CUDA/ROCm compatibility, and falls back to native PyTorch ops when
Triton is unavailable (NPU, CPU, ...), so callers never need a platform check.

Measured against eager ``F.silu(F.group_norm(...))`` on one L20X, bf16, 32
groups: 1.1-1.5x at the DiT ResBlock's activation sizes, where both paths are
dominated by launch overhead, and 2.2-2.9x at the VAE's decode-resolution
activations, where the saved memory traffic is what pays.

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
def _group_norm_silu_kernel(
    # Input/Output pointers
    x_ptr,
    out_ptr,
    # Normalization parameters
    weight_ptr,
    bias_ptr,
    # Partial-statistics workspace, unused (and passed as 0) when SPLIT == 1
    ws_ptr,
    # Shape info; x is contiguous (N, C, spatial_size)
    C,
    spatial_size,
    split_chunk,
    num_groups: tl.constexpr,
    eps: tl.constexpr,
    # Split reduction
    SPLIT: tl.constexpr,
    SPLIT_POW2: tl.constexpr,
    # Block sizes
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    """
    Fused GroupNorm + SiLU kernel.
    Computes SiLU(GroupNorm(x)) in one kernel, avoiding intermediate tensors.
    Mean and variance use a parallel Welford reduction, which avoids the
    catastrophic cancellation of E[x^2] - E[x]^2 on large-offset inputs. x is read
    once for the statistics and once for normalization (2 reads total).
    Moments are accumulated in fp32 to match PyTorch numerics.

    ``SPLIT`` programs cooperate on each (batch, group) pair. At SPLIT == 1 one
    program owns the whole group and computes its statistics inline, which is the
    right shape when the activation is small enough that launch overhead
    dominates. Above that, a preceding pass has already reduced each slice into
    ``ws`` and this kernel only merges the partials before normalizing its own
    slice -- the reason being that one CTA per group leaves most of the device
    idle on decode-resolution activations.
    """
    pid = tl.program_id(0)
    bg = pid // SPLIT
    s = pid % SPLIT

    group_size = C // num_groups
    n_idx = bg // num_groups
    g_idx = bg % num_groups

    # === Pass 1: group statistics (fp32) ===
    # Triton resolves this constexpr branch at compile time, so only one of the
    # two forms is ever emitted.
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

    # === Pass 2: normalize this program's slice, apply affine, and SiLU ===
    for c_offset in range(group_size):
        c_idx = g_idx * group_size + c_offset
        base = n_idx * C * spatial_size + c_idx * spatial_size

        weight_val = tl.load(weight_ptr + c_idx).to(tl.float32)
        bias_val = tl.load(bias_ptr + c_idx).to(tl.float32)

        for s_start in tl.range(lo, hi, BLOCK_SIZE, num_stages=num_stages):
            offsets = s_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < hi

            x_val = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
            x_val = x_val.to(tl.float32)

            # Normalize and apply affine
            norm_val = (x_val - mean) * rstd * weight_val + bias_val

            # Apply SiLU: x * sigmoid(x)
            out_val = norm_val * tl.sigmoid(norm_val)

            # ``tl.store`` casts to the output pointer's dtype, which the
            # caller picked to match eager GroupNorm's autocast behaviour.
            tl.store(out_ptr + base + offsets, out_val, mask=mask)


def fused_group_norm_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int = 32,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused GroupNorm + SiLU activation.
    Computes SiLU(GroupNorm(x, num_groups, weight, bias, eps)) in a single Triton kernel,
    avoiding intermediate tensors and reducing memory traffic and launch overhead.

    - x: (N, C, *spatial), any spatial rank.
    - weight, bias: per-channel parameters, shape (C,).
    - num_groups: GroupNorm group count, default 32.
    - eps: numerical stability epsilon, default 1e-6.
    Uses fp32 accumulation for numeric alignment with PyTorch.
    Returns the same shape and eager F.group_norm-compatible dtype.
    Spatial dimensions are flattened during computation and restored afterward;
    this is exact since GroupNorm reduces across each channel group and all spatial positions.
    Non-contiguous inputs are materialized before launch.
    """
    # Fallback if Triton not available (NPU, CPU, ...)
    if not HAS_TRITON:
        return F.silu(F.group_norm(x, num_groups, weight, bias, eps))

    # Validate inputs
    assert x.ndim >= 3, f"Expected at least 3D input (N, C, *spatial), got {x.ndim}D"
    assert x.size(1) % num_groups == 0, f"Channels {x.size(1)} must be divisible by num_groups {num_groups}"
    assert weight.ndim == 1 and weight.size(0) == x.size(1), (
        f"Weight shape {weight.shape} doesn't match channels {x.size(1)}"
    )
    assert bias.ndim == 1 and bias.size(0) == x.size(1), f"Bias shape {bias.shape} doesn't match channels {x.size(1)}"

    # Collapse arbitrary spatial ranks into a single axis so one kernel serves
    # both the 2D DiT blocks and the 3D VAE blocks.
    #
    # ``contiguous()`` is not just defensive. HunyuanImage3's UNetUp feeds this
    # op straight out of ``rearrange(x, "b (h w) c -> b c h w")``, i.e. a
    # permuted view whose *channel* stride is 1. Indexing that layout directly
    # from the kernel makes every warp's spatial load stride by C elements, and
    # the resulting uncoalesced traffic turned the op into 0.44x of eager at
    # (1, 4096, 64, 64). One coalesced pre-pass costs far less than that, so
    # normalize the layout here and let the kernel assume a dense block.
    orig_shape = x.shape
    B, C = orig_shape[0], orig_shape[1]
    x_flat = x.contiguous().reshape(B, C, -1)
    spatial_size = x_flat.size(2)

    # Allocate output with the dtype eager GroupNorm would return, so that the
    # fused path stays a drop-in replacement inside autocast regions.
    out_dtype = x_flat.dtype
    if torch.is_autocast_enabled(x_flat.device.type):
        out_dtype = torch.float32
    out_flat = torch.empty_like(x_flat, dtype=out_dtype)

    # One program per (batch, group) pair would be B*num_groups CTAs -- 32 for a
    # B=1 VAE decode, well under any modern SM count. Split the spatial axis so
    # the largest activations get enough CTAs to saturate memory; small ones fall
    # back to split=1, which is the original single-launch kernel.
    split, split_chunk = pick_split(spatial_size, B, num_groups, x_flat.device)

    ws = 0
    if split > 1:
        ws = launch_partial_stats(x_flat, B, C, spatial_size, num_groups, split, split_chunk)

    _group_norm_silu_kernel[(B * num_groups * split,)](
        x_flat,
        out_flat,
        weight,
        bias,
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


__all__ = ["fused_group_norm_silu"]
