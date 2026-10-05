# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
def _group_norm_silu_kernel(
    # Input/Output pointers
    x_ptr,
    out_ptr,
    # Normalization parameters
    weight_ptr,
    bias_ptr,
    # Shape info; x is contiguous (N, C, spatial_size)
    C,
    spatial_size,
    num_groups: tl.constexpr,
    eps: tl.constexpr,
    # Block sizes
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    """
    Fused GroupNorm + SiLU kernel.
    Computes SiLU(GroupNorm(x)) in one kernel, avoiding intermediate tensors.
    Mean and variance use a parallel Welford reduction (block-level register
    reduction merged with the Chan formula), which avoids the catastrophic
    cancellation of E[x^2] - E[x]^2 on large-offset inputs. x is read once for
    the statistics and once for normalization (2 reads total).
    One program handles each (batch, group) pair. Channels are processed serially,
    while spatial positions are vectorized for diffusion workloads.
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

    # === Pass 2: normalize, apply affine, and SiLU ===
    for c_offset in range(group_size):
        c_idx = g_idx * group_size + c_offset
        base = n_idx * C * spatial_size + c_idx * spatial_size

        weight_val = tl.load(weight_ptr + c_idx).to(tl.float32)
        bias_val = tl.load(bias_ptr + c_idx).to(tl.float32)

        for s_start in tl.range(0, spatial_size, BLOCK_SIZE, num_stages=num_stages):
            offsets = s_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < spatial_size

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

    # Only B*num_groups programs are launched, which is well under the SM count
    # for typical diffusion batches. A memory-bound kernel can still saturate
    # HBM from few CTAs, but only with enough loads in flight, so widen the CTA
    # for the large activations instead of leaving it at the 4-warp default.

    # One program per (batch, group) pair.
    grid = (B * num_groups,)

    _group_norm_silu_kernel[grid](
        x_flat,
        out_flat,
        weight,
        bias,
        C,
        spatial_size,
        num_groups=num_groups,
        eps=eps,
    )

    return out_flat.reshape(orig_shape)
