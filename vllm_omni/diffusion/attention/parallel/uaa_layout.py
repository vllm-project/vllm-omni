# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Local head padding/layout copies for advanced Ulysses.

Collectives, head-count selection, and sequence splits remain in ``ulysses``.
The CUDA copies preserve element bit patterns and allocate contiguous outputs.
Only measured BF16 A100 layouts are accelerated; the original view/copy
sequence handles all other inputs, including autograd and no-padding cases.
"""

from functools import cache

import torch
import torch.nn.functional as F
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:

    @triton.jit
    def _pad_pack_kernel(
        x_ptr,
        out_ptr,
        batch_size: tl.constexpr,
        seq_len: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        local_heads: tl.constexpr,
        numel: tl.constexpr,
        stride_b: tl.constexpr,
        stride_s: tl.constexpr,
        stride_h: tl.constexpr,
        stride_d: tl.constexpr,
        block_size: tl.constexpr,
    ):
        # Widen before multiplying: long sequences can exceed int32 offsets.
        i = tl.program_id(0).to(tl.int64) * block_size + tl.arange(0, block_size)
        d = i % head_dim
        h = (i // head_dim) % local_heads
        b = (i // (head_dim * local_heads)) % batch_size
        s = (i // (head_dim * local_heads * batch_size)) % seq_len
        u = i // (head_dim * local_heads * batch_size * seq_len)
        original_h = u * local_heads + h
        value = tl.load(
            x_ptr + b * stride_b + s * stride_s + original_h * stride_h + d * stride_d,
            mask=(i < numel) & (original_h < num_heads),
            other=0,
        )
        tl.store(out_ptr + i, value, mask=i < numel)

    @triton.jit
    def _unpack_unpad_kernel(
        x_ptr,
        out_ptr,
        batch_size: tl.constexpr,
        seq_len: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        local_heads: tl.constexpr,
        numel: tl.constexpr,
        block_size: tl.constexpr,
    ):
        i = tl.program_id(0).to(tl.int64) * block_size + tl.arange(0, block_size)
        d = i % head_dim
        h = (i // head_dim) % num_heads
        s = (i // (head_dim * num_heads)) % seq_len
        b = i // (head_dim * num_heads * seq_len)
        u = h // local_heads
        local_h = h % local_heads
        src = (((u * seq_len + s) * batch_size + b) * local_heads + local_h) * head_dim + d
        value = tl.load(x_ptr + src, mask=i < numel, other=0)
        tl.store(out_ptr + i, value, mask=i < numel)


def _can_fuse(x: torch.Tensor) -> bool:
    return (
        HAS_TRITON
        and current_platform.is_cuda()
        and x.is_cuda
        and x.layout == torch.strided
        and x.dtype in (torch.float32, torch.bfloat16)
        and x.numel() > 0
        and not (torch.is_grad_enabled() and x.requires_grad)
    )


@cache
def _is_benchmarked_device(device_index: int) -> bool:
    return current_platform.get_device_name(device_index) == "NVIDIA A100-SXM4-40GB"


def _pack_has_measured_speedup(x: torch.Tensor, world_size: int, padded_heads: int) -> bool:
    # All six local-copy cases improved on both A100s in seven paired blocks.
    # Compilation has correctness coverage but no independent timing evidence.
    if torch.compiler.is_compiling():
        return False
    b, s, h, d = x.shape
    return (
        x.dtype == torch.bfloat16
        and b == 1
        and s in (2048, 2136)
        and d == 120
        and world_size == 2
        and (h, padded_heads) in ((21, 24), (7, 8))
        and x.is_contiguous()
        and _is_benchmarked_device(x.device.index)
    )


def _unpack_has_measured_speedup(x: torch.Tensor, world_size: int, seq_len: int, heads: int) -> bool:
    if torch.compiler.is_compiling():
        return False
    return (
        x.dtype == torch.bfloat16
        and world_size == 2
        and seq_len in (2048, 2136)
        and heads == 21
        and x.shape == (2 * seq_len, 1, 12, 120)
        and _is_benchmarked_device(x.device.index)
    )


def pad_pack_heads(x: torch.Tensor, world_size: int, padded_heads: int) -> torch.Tensor:
    """Map [B,S,H,D] to contiguous [U*S,B,Hpad/U,D], padding heads with zero.

    ``padded_heads`` is chosen by the caller, including the GQA ratio. Reading
    uses all four input strides, including packed projection slices. A single
    rank and an unpadded input retain the original reshape/contiguous behavior.
    """
    b, s, h, d = x.shape
    local_h = padded_heads // world_size
    if padded_heads > h and world_size > 1 and _can_fuse(x) and _pack_has_measured_speedup(x, world_size, padded_heads):
        out = torch.empty((world_size * s, b, local_h, d), device=x.device, dtype=x.dtype)
        _pad_pack_kernel[(triton.cdiv(out.numel(), 1024),)](
            x,
            out,
            b,
            s,
            h,
            d,
            local_h,
            out.numel(),
            *x.stride(),
            1024,
        )
        return out
    if padded_heads > h:
        x = F.pad(x, (0, 0, 0, padded_heads - h))
    return x.reshape(b, s, world_size, local_h, d).permute(2, 1, 0, 3, 4).contiguous().flatten(0, 1)


def unpack_unpad_heads(
    x: torch.Tensor,
    world_size: int,
    local_seq_len: int,
    original_heads: int,
) -> torch.Tensor:
    """Map receive [U*S,B,Hlocal,D] directly to contiguous [B,S,H,D]."""
    _, b, local_h, d = x.shape
    padded_heads = world_size * local_h
    # A single row's native head slice is already a contiguous view. Keep
    # that allocation/stride behavior instead of introducing a kernel/copy.
    if (
        0 < original_heads < padded_heads
        and world_size > 1
        and b * local_seq_len > 1
        and x.is_contiguous()
        and _can_fuse(x)
        and _unpack_has_measured_speedup(x, world_size, local_seq_len, original_heads)
    ):
        out = torch.empty((b, local_seq_len, original_heads, d), device=x.device, dtype=x.dtype)
        _unpack_unpad_kernel[(triton.cdiv(out.numel(), 1024),)](
            x,
            out,
            b,
            local_seq_len,
            original_heads,
            d,
            local_h,
            out.numel(),
            1024,
        )
        return out
    out = x.reshape(world_size, local_seq_len, b, local_h, d).permute(2, 1, 0, 3, 4).contiguous()
    out = out.reshape(b, local_seq_len, padded_heads, d)
    if padded_heads != original_heads:
        out = out[:, :, :original_heads, :].contiguous()
    return out
