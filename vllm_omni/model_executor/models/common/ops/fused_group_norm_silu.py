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

from vllm_omni.model_executor.models.common.ops._dtype_utils import (
    group_norm_output_dtype,
)

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _group_norm_silu_kernel(
        # Input/Output pointers
        x_ptr, out_ptr,
        # Normalization parameters
        weight_ptr, bias_ptr,
        # Shape info; x is contiguous (N, C, spatial_size)
        C, spatial_size,
        num_groups: tl.constexpr,
        eps: tl.constexpr,
        # Block sizes
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused GroupNorm + SiLU kernel.

        Computes: SiLU(GroupNorm(x)) in a single pass.

        One program per (batch, group) pair. Channels within the group are
        walked serially while the spatial axis is vectorized, which is the right
        way round for diffusion workloads: a group holds at most a few hundred
        channels but thousands of spatial positions.

        Uses fp32 accumulation for moments to match PyTorch's numeric behavior.
        """
        pid = tl.program_id(0)

        group_size = C // num_groups
        n_idx = pid // num_groups
        g_idx = pid % num_groups

        # === Pass 1: Compute mean and variance (fp32 accumulation) ===
        mean_acc = tl.zeros([1], dtype=tl.float32)
        var_acc = tl.zeros([1], dtype=tl.float32)

        for c_offset in range(group_size):
            c_idx = g_idx * group_size + c_offset
            base = n_idx * C * spatial_size + c_idx * spatial_size

            for s_start in range(0, spatial_size, BLOCK_SIZE):
                offsets = s_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < spatial_size

                x_val = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
                x_val = x_val.to(tl.float32)

                mean_acc += tl.sum(x_val, axis=0)
                var_acc += tl.sum(x_val * x_val, axis=0)

        group_total = group_size * spatial_size
        mean = mean_acc / group_total
        var = var_acc / group_total - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)

        # === Pass 2: Normalize, apply affine transform, and SiLU ===
        for c_offset in range(group_size):
            c_idx = g_idx * group_size + c_offset
            base = n_idx * C * spatial_size + c_idx * spatial_size

            weight_val = tl.load(weight_ptr + c_idx).to(tl.float32)
            bias_val = tl.load(bias_ptr + c_idx).to(tl.float32)

            for s_start in range(0, spatial_size, BLOCK_SIZE):
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
    """Fused GroupNorm + SiLU activation.
    
    Computes: SiLU(GroupNorm(x, num_groups, weight, bias, eps))
    
    This is mathematically equivalent to:
        F.silu(F.group_norm(x, num_groups, weight, bias, eps))
    
    But fuses the operations into a single Triton kernel to:
    1. Reduce memory traffic (no materialized intermediate tensors)
    2. Reduce kernel launch overhead
    3. Maintain fp32 accumulation precision for numeric alignment
    
    Args:
        x: Input tensor of shape (N, C, *spatial); any spatial rank is accepted,
            e.g. (N, C, H, W) for the 2D case or (N, C, T, H, W) for the 3D VAE.
        weight: Per-channel scale of shape (C,)
        bias: Per-channel bias of shape (C,)
        num_groups: Number of groups for GroupNorm (default: 32)
        eps: Small constant for numerical stability (default: 1e-6)

    Returns:
        Output tensor of the same shape as ``x``, with the dtype eager
        ``F.group_norm`` would produce (see ``group_norm_output_dtype``).

    Examples:
        >>> x = torch.randn(2, 64, 32, 32, device='cuda')
        >>> weight = torch.randn(64, device='cuda')
        >>> bias = torch.randn(64, device='cuda')
        >>> out = fused_group_norm_silu(x, weight, bias, num_groups=32)
        >>> out.shape
        torch.Size([2, 64, 32, 32])

    Note:
        Spatial axes are collapsed into one before the launch and restored
        afterwards. GroupNorm reduces over the whole channel group and every
        spatial position, so collapsing the spatial axes is exact, not an
        approximation. Non-contiguous inputs are materialized first; see the
        comment in the body for why that is a win rather than a cost.
    """
    # Fallback if Triton not available (NPU, CPU, ...)
    if not HAS_TRITON:
        return F.silu(F.group_norm(x, num_groups, weight, bias, eps))

    # Validate inputs
    assert x.ndim >= 3, f"Expected at least 3D input (N, C, *spatial), got {x.ndim}D"
    assert x.size(1) % num_groups == 0, \
        f"Channels {x.size(1)} must be divisible by num_groups {num_groups}"
    assert weight.ndim == 1 and weight.size(0) == x.size(1), \
        f"Weight shape {weight.shape} doesn't match channels {x.size(1)}"
    assert bias.ndim == 1 and bias.size(0) == x.size(1), \
        f"Bias shape {bias.shape} doesn't match channels {x.size(1)}"

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
    out_flat = torch.empty_like(x_flat, dtype=group_norm_output_dtype(x))

    # Only B*num_groups programs are launched, which is well under the SM count
    # for typical diffusion batches. A memory-bound kernel can still saturate
    # HBM from few CTAs, but only with enough loads in flight, so widen the CTA
    # for the large activations instead of leaving it at the 4-warp default.
    BLOCK_SIZE = min(4096, triton.next_power_of_2(spatial_size))
    num_warps = 16 if BLOCK_SIZE >= 4096 else (8 if BLOCK_SIZE >= 2048 else 4)

    # One program per (batch, group) pair.
    grid = (B * num_groups,)

    _group_norm_silu_kernel[grid](
        x_flat, out_flat,
        weight, bias,
        C, spatial_size,
        num_groups=num_groups,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return out_flat.reshape(orig_shape)