# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: N803
"""Small fused elementwise kernels used by the MiniMax-H3 DiT blocks."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

_BLOCK_SIZE = 256
_QK_BLOCK_SIZE = 128


@triton.jit
def _pack_qkv_destination_major_kernel(
    output_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    total_elements,
    rows,
    local_heads,
    head_size,
    stride_q_row,
    stride_q_head,
    stride_k_row,
    stride_k_head,
    stride_v_row,
    stride_v_head,
    BLOCK_SIZE: tl.constexpr,
):
    """Pack [S, H, D] Q/K/V into [world, S, H/world, 3D]."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    dim = offsets % head_size
    head_slot = offsets // head_size
    local_head = head_slot % local_heads
    row_slot = head_slot // local_heads
    row = row_slot % rows
    destination = row_slot // rows
    global_head = destination * local_heads + local_head

    q_value = tl.load(
        q_ptr + row * stride_q_row + global_head * stride_q_head + dim,
        mask=mask,
    )
    k_value = tl.load(
        k_ptr + row * stride_k_row + global_head * stride_k_head + dim,
        mask=mask,
    )
    v_value = tl.load(
        v_ptr + row * stride_v_row + global_head * stride_v_head + dim,
        mask=mask,
    )
    output_base = head_slot * (3 * head_size) + dim
    tl.store(output_ptr + output_base, q_value, mask=mask)
    tl.store(output_ptr + output_base + head_size, k_value, mask=mask)
    tl.store(output_ptr + output_base + 2 * head_size, v_value, mask=mask)


@triton.jit
def _merge_ulysses_heads_kernel(
    output_ptr,
    input_ptr,
    total_elements,
    seq,
    world,
    local_heads,
    head_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Materialize [1, S, W, H/W, D] from [W, S, 1, H/W, D]."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    dim = offsets % head_size
    head_slot = offsets // head_size
    local_head = head_slot % local_heads
    world_slot = (head_slot // local_heads) % world
    row = (head_slot // (local_heads * world)) % seq

    input_offset = (((world_slot * seq + row) * local_heads + local_head) * head_size) + dim
    tl.store(output_ptr + offsets, tl.load(input_ptr + input_offset, mask=mask), mask=mask)


@triton.jit
def _merge_ulysses_heads_vec_kernel(
    output_ptr,
    input_ptr,
    total_vectors,
    seq,
    world,
    local_heads,
    head_size,
    VEC_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Vectorized [W, S, 1, h, D] -> [1, S, W, h, D] copy.

    The H3 head dimension is 128, so every vector stays within one head and
    can be loaded from the all-to-all result and stored contiguously in the
    merged output.  This mirrors the 16-byte copy granularity used by the
    fixed-shape CUDA path while retaining a scalar fallback for other layouts.
    """
    vector_ids = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    vector_mask = vector_ids < total_vectors
    lanes = tl.arange(0, VEC_SIZE)

    vectors_per_head = head_size // VEC_SIZE
    scalar_offsets = vector_ids * VEC_SIZE
    dim_vector = vector_ids % vectors_per_head
    head_slot = vector_ids // vectors_per_head
    local_head = head_slot % local_heads
    world_slot = (head_slot // local_heads) % world
    row = (head_slot // (local_heads * world)) % seq
    input_offset = (((world_slot * seq + row) * local_heads + local_head) * head_size) + dim_vector * VEC_SIZE

    mask = vector_mask[:, None]
    values = tl.load(input_ptr + input_offset[:, None] + lanes[None, :], mask=mask)
    tl.store(output_ptr + scalar_offsets[:, None] + lanes[None, :], values, mask=mask)


@triton.jit
def _indexed_scale_shift_kernel(
    x_ptr,
    scale_ptr,
    shift_ptr,
    indices_ptr,
    n_cols,
    x_stride_0,
    x_stride_1,
    scale_stride_0,
    scale_stride_1,
    shift_stride_0,
    shift_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    index = tl.load(indices_ptr + row)
    x_offset = row * x_stride_0 + cols * x_stride_1
    scale_offset = index * scale_stride_0 + cols * scale_stride_1
    shift_offset = index * shift_stride_0 + cols * shift_stride_1
    x_value = tl.load(x_ptr + x_offset, mask=mask, other=0.0).to(tl.float32)
    scale_value = tl.load(scale_ptr + scale_offset, mask=mask, other=0.0).to(tl.float32)
    shift_value = tl.load(shift_ptr + shift_offset, mask=mask, other=0.0).to(tl.float32)
    result = (x_value * (1.0 + scale_value) + shift_value).to(x_ptr.dtype.element_ty)
    tl.store(x_ptr + x_offset, result, mask=mask)


@triton.jit
def _fused_rmsnorm_indexed_scale_shift_kernel(
    output_ptr,
    x_ptr,
    weight_ptr,
    scale_ptr,
    shift_ptr,
    indices_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)

    # Match the existing two-op path: RMSNorm stores BF16 before the indexed
    # scale/shift kernel reloads and promotes the normalized values to FP32.
    normalized = (x * inv_rms * weight).to(x_ptr.dtype.element_ty)
    index = tl.load(indices_ptr + row)
    parameter_offset = index * n_cols + cols
    scale = tl.load(scale_ptr + parameter_offset, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr + parameter_offset, mask=mask, other=0.0).to(tl.float32)
    output = (normalized.to(tl.float32) * (1.0 + scale) + shift).to(output_ptr.dtype.element_ty)
    tl.store(output_ptr + row * n_cols + cols, output, mask=mask)


@triton.jit
def _indexed_gate_kernel(
    x_ptr,
    gate_ptr,
    other_ptr,
    indices_ptr,
    n_cols,
    x_stride_0,
    x_stride_1,
    gate_stride_0,
    gate_stride_1,
    other_stride_0,
    other_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    index = tl.load(indices_ptr + row)
    x_offset = row * x_stride_0 + cols * x_stride_1
    gate_offset = index * gate_stride_0 + cols * gate_stride_1
    other_offset = row * other_stride_0 + cols * other_stride_1
    x_value = tl.load(x_ptr + x_offset, mask=mask, other=0.0).to(tl.float32)
    gate_value = tl.load(gate_ptr + gate_offset, mask=mask, other=0.0).to(tl.float32)
    other_value = tl.load(other_ptr + other_offset, mask=mask, other=0.0).to(tl.float32)
    result = (x_value + gate_value * other_value).to(x_ptr.dtype.element_ty)
    tl.store(x_ptr + x_offset, result, mask=mask)


@triton.jit
def _qknorm_rope_kernel(
    x_ptr,
    norm_weight_ptr,
    rope_cache_ptr,
    n_tokens,
    n_heads,
    head_dim,
    rope_dim,
    x_stride_0,
    x_stride_1,
    x_stride_2,
    norm_stride,
    rope_stride_0,
    rope_stride_1,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < head_dim
    x_base = x_ptr + token * x_stride_0 + head * x_stride_1
    values = tl.load(x_base + cols * x_stride_2, mask=mask, other=0.0).to(tl.float32)
    norm_weight = tl.load(norm_weight_ptr + cols * norm_stride, mask=mask, other=1.0).to(tl.float32)
    rms = tl.rsqrt(tl.sum(values * values, axis=0) / head_dim + eps)
    normed = (values * rms * norm_weight).to(x_ptr.dtype.element_ty)

    half_rope = rope_dim // 2
    pair = tl.where(cols < half_rope, cols + half_rope, cols - half_rope)
    pair_mask = (cols < rope_dim) & (pair < head_dim)
    pair_values = tl.load(x_base + pair * x_stride_2, mask=pair_mask, other=0.0).to(tl.float32)
    pair_weight = tl.load(norm_weight_ptr + pair * norm_stride, mask=pair_mask, other=1.0).to(tl.float32)
    pair_normed = (pair_values * rms * pair_weight).to(x_ptr.dtype.element_ty)

    rope_base = rope_cache_ptr + token * rope_stride_0
    cos = tl.load(rope_base + cols * rope_stride_1, mask=pair_mask, other=1.0).to(tl.float32)
    sin = tl.load(rope_base + (rope_dim + cols) * rope_stride_1, mask=pair_mask, other=0.0).to(tl.float32)
    sign = tl.where(cols < half_rope, -1.0, 1.0)
    # Match the reference order: each BF16 product is rounded before the
    # rotated pair is added, just as the unfused Torch path multiplies BF16
    # tensors and then adds the two BF16 products.
    first_product = (normed.to(tl.float32) * cos).to(x_ptr.dtype.element_ty)
    second_product = (pair_normed.to(tl.float32) * sin).to(x_ptr.dtype.element_ty)
    rotated = (first_product.to(tl.float32) + sign * second_product.to(tl.float32)).to(x_ptr.dtype.element_ty)
    result = tl.where(cols < rope_dim, rotated, normed)
    tl.store(x_base + cols * x_stride_2, result, mask=mask)


@triton.jit
def _qk_norm_rope_kernel(
    q_ptr,
    k_ptr,
    q_weight_ptr,
    k_weight_ptr,
    rope_cache_ptr,
    n_tokens,
    n_heads,
    head_dim,
    rope_dim,
    q_stride_0,
    q_stride_1,
    q_stride_2,
    k_stride_0,
    k_stride_1,
    k_stride_2,
    q_norm_stride,
    k_norm_stride,
    rope_stride_0,
    rope_stride_1,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply the same token/head's Q and K norm+RoPE work in one launch."""
    token = tl.program_id(0)
    head = tl.program_id(1)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < head_dim

    q_base = q_ptr + token * q_stride_0 + head * q_stride_1
    k_base = k_ptr + token * k_stride_0 + head * k_stride_1
    q_values = tl.load(q_base + cols * q_stride_2, mask=mask, other=0.0).to(tl.float32)
    k_values = tl.load(k_base + cols * k_stride_2, mask=mask, other=0.0).to(tl.float32)
    q_weight = tl.load(q_weight_ptr + cols * q_norm_stride, mask=mask, other=1.0).to(tl.float32)
    k_weight = tl.load(k_weight_ptr + cols * k_norm_stride, mask=mask, other=1.0).to(tl.float32)
    q_rms = tl.rsqrt(tl.sum(q_values * q_values, axis=0) / head_dim + eps)
    k_rms = tl.rsqrt(tl.sum(k_values * k_values, axis=0) / head_dim + eps)
    q_normed = (q_values * q_rms * q_weight).to(q_ptr.dtype.element_ty)
    k_normed = (k_values * k_rms * k_weight).to(k_ptr.dtype.element_ty)

    half_rope = rope_dim // 2
    pair = tl.where(cols < half_rope, cols + half_rope, cols - half_rope)
    pair_mask = (cols < rope_dim) & (pair < head_dim)
    q_pair_values = tl.load(q_base + pair * q_stride_2, mask=pair_mask, other=0.0).to(tl.float32)
    k_pair_values = tl.load(k_base + pair * k_stride_2, mask=pair_mask, other=0.0).to(tl.float32)
    q_pair_weight = tl.load(q_weight_ptr + pair * q_norm_stride, mask=pair_mask, other=1.0).to(tl.float32)
    k_pair_weight = tl.load(k_weight_ptr + pair * k_norm_stride, mask=pair_mask, other=1.0).to(tl.float32)
    q_pair_normed = (q_pair_values * q_rms * q_pair_weight).to(q_ptr.dtype.element_ty)
    k_pair_normed = (k_pair_values * k_rms * k_pair_weight).to(k_ptr.dtype.element_ty)

    rope_base = rope_cache_ptr + token * rope_stride_0
    cos = tl.load(rope_base + cols * rope_stride_1, mask=pair_mask, other=1.0).to(tl.float32)
    sin = tl.load(rope_base + (rope_dim + cols) * rope_stride_1, mask=pair_mask, other=0.0).to(tl.float32)
    sign = tl.where(cols < half_rope, -1.0, 1.0)
    # Keep the reference BF16 rounding order for both projections.
    q_first = (q_normed.to(tl.float32) * cos).to(q_ptr.dtype.element_ty)
    q_second = (q_pair_normed.to(tl.float32) * sin).to(q_ptr.dtype.element_ty)
    k_first = (k_normed.to(tl.float32) * cos).to(k_ptr.dtype.element_ty)
    k_second = (k_pair_normed.to(tl.float32) * sin).to(k_ptr.dtype.element_ty)
    q_rotated = (q_first.to(tl.float32) + sign * q_second.to(tl.float32)).to(q_ptr.dtype.element_ty)
    k_rotated = (k_first.to(tl.float32) + sign * k_second.to(tl.float32)).to(k_ptr.dtype.element_ty)
    q_result = tl.where(cols < rope_dim, q_rotated, q_normed)
    k_result = tl.where(cols < rope_dim, k_rotated, k_normed)
    tl.store(q_base + cols * q_stride_2, q_result, mask=mask)
    tl.store(k_base + cols * k_stride_2, k_result, mask=mask)


@triton.jit
def _qk_rope_inplace_kernel(
    q_ptr,
    k_ptr,
    rope_cache_ptr,
    n_tokens,
    n_heads,
    head_dim,
    rope_dim,
    q_stride_0,
    q_stride_1,
    q_stride_2,
    k_stride_0,
    k_stride_1,
    k_stride_2,
    rope_stride_0,
    rope_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply H3's partial RoPE in place to already-normalized Q and K."""
    token = tl.program_id(0)
    head = tl.program_id(1)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < head_dim

    half_rope = rope_dim // 2
    pair = tl.where(cols < half_rope, cols + half_rope, cols - half_rope)
    pair_mask = (cols < rope_dim) & (pair < head_dim)
    rope_base = rope_cache_ptr + token * rope_stride_0
    cos = tl.load(rope_base + cols * rope_stride_1, mask=pair_mask, other=1.0).to(tl.float32)
    sin = tl.load(rope_base + (rope_dim + cols) * rope_stride_1, mask=pair_mask, other=0.0).to(tl.float32)
    sign = tl.where(cols < half_rope, -1.0, 1.0)

    q_base = q_ptr + token * q_stride_0 + head * q_stride_1
    k_base = k_ptr + token * k_stride_0 + head * k_stride_1
    q_values = tl.load(q_base + cols * q_stride_2, mask=mask, other=0.0).to(tl.float32)
    k_values = tl.load(k_base + cols * k_stride_2, mask=mask, other=0.0).to(tl.float32)
    q_pair = tl.load(q_base + pair * q_stride_2, mask=pair_mask, other=0.0).to(tl.float32)
    k_pair = tl.load(k_base + pair * k_stride_2, mask=pair_mask, other=0.0).to(tl.float32)

    q_first = (q_values * cos).to(q_ptr.dtype.element_ty)
    q_second = (q_pair * sin).to(q_ptr.dtype.element_ty)
    k_first = (k_values * cos).to(k_ptr.dtype.element_ty)
    k_second = (k_pair * sin).to(k_ptr.dtype.element_ty)
    q_rotated = (q_first.to(tl.float32) + sign * q_second.to(tl.float32)).to(q_ptr.dtype.element_ty)
    k_rotated = (k_first.to(tl.float32) + sign * k_second.to(tl.float32)).to(k_ptr.dtype.element_ty)
    tl.store(q_base + cols * q_stride_2, tl.where(cols < rope_dim, q_rotated, q_values), mask=mask)
    tl.store(k_base + cols * k_stride_2, tl.where(cols < rope_dim, k_rotated, k_values), mask=mask)


def _can_use_indexed_common(
    x: torch.Tensor,
    parameter: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and parameter.dtype == torch.bfloat16
        and indices.dtype in (torch.int32, torch.int64)
        and x.dim() == 2
        and parameter.dim() == 2
        and indices.dim() == 1
        and x.shape[0] == indices.shape[0]
        and x.shape[1] == parameter.shape[1]
        and x.is_contiguous()
        and parameter.is_contiguous()
        and indices.is_contiguous()
    )


def indexed_scale_shift_bf16_(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    """Apply ``x = x * (1 + scale[index]) + shift[index]`` in place."""
    if not (
        _can_use_indexed_common(x, scale, indices)
        and shift.dtype == torch.bfloat16
        and shift.shape == scale.shape
        and shift.is_contiguous()
    ):
        return False
    grid = (x.shape[0], triton.cdiv(x.shape[1], _BLOCK_SIZE))
    _indexed_scale_shift_kernel[grid](
        x,
        scale,
        shift,
        indices,
        x.shape[1],
        x.stride(0),
        x.stride(1),
        scale.stride(0),
        scale.stride(1),
        shift.stride(0),
        shift.stride(1),
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=4,
    )
    return True


def indexed_gate_bf16_(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    """Apply ``x += gate[index] * other`` in place."""
    if not (
        _can_use_indexed_common(x, gate, indices)
        and other.dtype == torch.bfloat16
        and other.shape == x.shape
        and other.is_contiguous()
    ):
        return False
    grid = (x.shape[0], triton.cdiv(x.shape[1], _BLOCK_SIZE))
    _indexed_gate_kernel[grid](
        x,
        gate,
        other,
        indices,
        x.shape[1],
        x.stride(0),
        x.stride(1),
        gate.stride(0),
        gate.stride(1),
        other.stride(0),
        other.stride(1),
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=4,
    )
    return True


def _launch_fused_rmsnorm_indexed_scale_shift(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    output = torch.empty_like(x)
    if x.shape[0] == 0:
        return output
    block_size = triton.next_power_of_2(x.shape[1])
    _fused_rmsnorm_indexed_scale_shift_kernel[(x.shape[0],)](
        output,
        x,
        weight,
        scale,
        shift,
        indices,
        x.shape[1],
        eps,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return output


def fused_rmsnorm_indexed_scale_shift_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    """Fuse out-of-place RMSNorm and indexed AdaLN for H3's BF16 path."""
    if not (
        _can_use_indexed_common(x, scale, indices)
        and weight.is_cuda
        and shift.is_cuda
        and indices.is_cuda
        and weight.device == x.device
        and shift.device == x.device
        and scale.device == x.device
        and indices.device == x.device
        and weight.dtype == torch.bfloat16
        and shift.dtype == torch.bfloat16
        and weight.shape == (x.shape[1],)
        and shift.shape == scale.shape
        and weight.is_contiguous()
        and shift.is_contiguous()
        and 0 < x.shape[1] <= 65536
    ):
        return None
    if torch.compiler.is_compiling():
        return torch.ops.vllm_omni.minimax_h3_rmsnorm_indexed_scale_shift(
            x,
            weight,
            shift,
            scale,
            indices,
            eps,
        )
    return _launch_fused_rmsnorm_indexed_scale_shift(x, weight, shift, scale, indices, eps)


def pack_qkv_destination_major_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    world_size: int,
) -> torch.Tensor | None:
    """Pack Q/K/V for one Ulysses input collective.

    The output uses destination-major storage, so a single all-to-all can
    exchange all three projections while preserving a unit stride in the
    head dimension after splitting the received ``3 * D`` payload.  Return
    ``None`` for layouts outside the fixed H3 CUDA fast path.
    """
    if not (
        world_size > 1
        and q.is_cuda
        and q.dtype == torch.bfloat16
        and q.shape == k.shape == v.shape
        and q.ndim == 3
        and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    ):
        return None
    rows, global_heads, head_size = q.shape
    if global_heads % world_size:
        return None
    local_heads = global_heads // world_size
    output = torch.empty(
        (world_size, rows, local_heads, 3 * head_size),
        dtype=q.dtype,
        device=q.device,
    )
    total_elements = rows * global_heads * head_size
    if total_elements == 0:
        return output
    block_size = 1024
    _pack_qkv_destination_major_kernel[(triton.cdiv(total_elements, block_size),)](
        output,
        q,
        k,
        v,
        total_elements,
        rows,
        local_heads,
        head_size,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return output


def merge_ulysses_heads_bf16(
    x: torch.Tensor,
    *,
    world_size: int,
    seq: int,
    local_heads: int,
    head_size: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Fuse the inverse Ulysses output transpose and contiguous copy."""
    if not (
        world_size > 1
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim == 5
        and tuple(x.shape) == (world_size, seq, 1, local_heads, head_size)
        and x.is_contiguous()
    ):
        return None
    if output is None:
        output = torch.empty_like(x).new_empty((1, seq, world_size, local_heads, head_size))
    else:
        if output.numel() != x.numel() or not output.is_contiguous():
            return None
        # The linear storage of [1, seq, world, local_heads, head_size] is
        # identical to [1, seq, world * local_heads, head_size].  Reusing the
        # completed attention output here removes one full-size allocator
        # request without changing the fused copy's indexing.
        output = output.view(1, seq, world_size, local_heads, head_size)
    total_elements = output.numel()
    if total_elements == 0:
        return output
    # BF16 vectors of eight values are 16-byte aligned for the H3 contiguous
    # buffers.  Use the vector path only when a complete vector fits inside a
    # head; the generic scalar kernel preserves the helper's wider contract.
    vec_size = 8
    if (
        head_size % vec_size == 0
        and total_elements % vec_size == 0
        and (x.data_ptr() % 16) == 0
        and (output.data_ptr() % 16) == 0
    ):
        block_size = 256
        total_vectors = total_elements // vec_size
        _merge_ulysses_heads_vec_kernel[(triton.cdiv(total_vectors, block_size),)](
            output,
            x,
            total_vectors,
            seq,
            world_size,
            local_heads,
            head_size,
            VEC_SIZE=vec_size,
            BLOCK_SIZE=block_size,
            num_warps=8,
        )
    else:
        block_size = 1024
        _merge_ulysses_heads_kernel[(triton.cdiv(total_elements, block_size),)](
            output,
            x,
            total_elements,
            seq,
            world_size,
            local_heads,
            head_size,
            BLOCK_SIZE=block_size,
            num_warps=8,
        )
    return output


def fused_qknorm_rope_bf16_(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_cache: torch.Tensor,
    *,
    eps: float,
    rope_dim: int,
) -> bool:
    """Fuse q/k RMSNorm and H3's partial RoPE into one in-place kernel."""
    if (
        not q.is_cuda
        or q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or q_weight.dtype != torch.bfloat16
        or k_weight.dtype != torch.bfloat16
        or rope_cache.dtype != torch.bfloat16
        or q.dim() != 3
        or k.dim() != 3
        or q.shape[-1] != _QK_BLOCK_SIZE
        or k.shape[-1] != _QK_BLOCK_SIZE
        or q.shape[0] != k.shape[0]
        or q_weight.shape != (q.shape[-1],)
        or k_weight.shape != (k.shape[-1],)
        or rope_cache.dim() != 2
        or rope_cache.shape[0] != q.shape[0]
        or rope_cache.shape[1] < 2 * rope_dim
        or not q.is_contiguous()
        or not k.is_contiguous()
        or not q_weight.is_contiguous()
        or not k_weight.is_contiguous()
        or not rope_cache.is_contiguous()
        or q.requires_grad
        or k.requires_grad
    ):
        return False

    if q.shape == k.shape:
        grid = (q.shape[0], q.shape[1])
        _qk_norm_rope_kernel[grid](
            q,
            k,
            q_weight,
            k_weight,
            rope_cache,
            q.shape[0],
            q.shape[1],
            q.shape[-1],
            rope_dim,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            q_weight.stride(0),
            k_weight.stride(0),
            rope_cache.stride(0),
            rope_cache.stride(1),
            eps,
            BLOCK_SIZE=_QK_BLOCK_SIZE,
            num_warps=4,
        )
    else:
        # Preserve the generic GQA fallback when Q and K have different head
        # counts; the fused H3 path always has equal Q/K shapes.
        for x, weight in ((q, q_weight), (k, k_weight)):
            grid = (x.shape[0], x.shape[1])
            _qknorm_rope_kernel[grid](
                x,
                weight,
                rope_cache,
                x.shape[0],
                x.shape[1],
                x.shape[-1],
                rope_dim,
                x.stride(0),
                x.stride(1),
                x.stride(2),
                weight.stride(0),
                rope_cache.stride(0),
                rope_cache.stride(1),
                eps,
                BLOCK_SIZE=_QK_BLOCK_SIZE,
                num_warps=4,
            )
    return True


def fused_rope_bf16_(
    q: torch.Tensor,
    k: torch.Tensor,
    rope_cache: torch.Tensor,
    *,
    rope_dim: int,
) -> bool:
    """Apply H3's partial RoPE to normalized Q/K without materializing cats."""
    if not (
        q.is_cuda
        and k.is_cuda
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and q.shape == k.shape
        and q.dim() == 3
        and q.shape[-1] == _QK_BLOCK_SIZE
        and q.stride(-1) == k.stride(-1) == 1
        and q.is_contiguous()
        and k.is_contiguous()
        and rope_cache.dtype == torch.bfloat16
        and rope_cache.dim() == 2
        and rope_cache.shape[0] == q.shape[0]
        and rope_cache.shape[1] >= 2 * rope_dim
        and rope_cache.is_contiguous()
        and not q.requires_grad
        and not k.requires_grad
    ):
        return False

    if torch.compiler.is_compiling():
        torch.ops.vllm_omni.minimax_h3_rope(q, k, rope_cache, rope_dim)
        return True

    grid = (q.shape[0], q.shape[1])
    _qk_rope_inplace_kernel[grid](
        q,
        k,
        rope_cache,
        q.shape[0],
        q.shape[1],
        q.shape[-1],
        rope_dim,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        rope_cache.stride(0),
        rope_cache.stride(1),
        BLOCK_SIZE=_QK_BLOCK_SIZE,
        num_warps=4,
    )
    return True


if not hasattr(torch.ops.vllm_omni, "minimax_h3_rope"):

    @torch.library.custom_op(
        "vllm_omni::minimax_h3_rope",
        mutates_args=("q", "k"),
    )
    def _minimax_h3_rope_op(
        q: torch.Tensor,
        k: torch.Tensor,
        rope_cache: torch.Tensor,
        rope_dim: int,
    ) -> None:
        grid = (q.shape[0], q.shape[1])
        _qk_rope_inplace_kernel[grid](
            q,
            k,
            rope_cache,
            q.shape[0],
            q.shape[1],
            q.shape[-1],
            rope_dim,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            rope_cache.stride(0),
            rope_cache.stride(1),
            BLOCK_SIZE=_QK_BLOCK_SIZE,
            num_warps=4,
        )
        return None

    @_minimax_h3_rope_op.register_fake
    def _minimax_h3_rope_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        rope_cache: torch.Tensor,
        rope_dim: int,
    ) -> None:
        del q, k, rope_cache, rope_dim
        return None


if not hasattr(torch.ops.vllm_omni, "minimax_h3_rmsnorm_indexed_scale_shift"):

    @torch.library.custom_op(
        "vllm_omni::minimax_h3_rmsnorm_indexed_scale_shift",
        mutates_args=(),
    )
    def _minimax_h3_rmsnorm_indexed_scale_shift_op(
        x: torch.Tensor,
        weight: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        indices: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        return _launch_fused_rmsnorm_indexed_scale_shift(x, weight, shift, scale, indices, eps)

    @_minimax_h3_rmsnorm_indexed_scale_shift_op.register_fake
    def _minimax_h3_rmsnorm_indexed_scale_shift_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        indices: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        del weight, shift, scale, indices, eps
        return torch.empty_like(x)
