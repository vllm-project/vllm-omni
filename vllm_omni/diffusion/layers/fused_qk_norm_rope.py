# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Q/K RMSNorm followed by RoPE.

The packed non-interleaved contract is shared by diffusion attention
implementations:

* ``q`` and ``k`` are ``[tokens, heads, head_dim]``;
* norm weights are one-dimensional ``[head_dim]`` tensors;
* ``rope_table`` is ``[tokens, rotary_dim]`` and stores
  ``[cos(theta), sin(theta)]`` with ``theta`` of width ``rotary_dim // 2``.

The interleaved contract accepts either ``[tokens, heads, head_dim]`` or
``[batch, seq, heads, head_dim]`` Q/K tensors and separate 2-D half-width
cosine and sine tables in the activation dtype or FP32. CUDA fast paths fuse
RMSNorm and RoPE without materializing normalized Q/K or rotary-product
intermediates. Ascend composes its RMSNorm and rotary fused primitives for the
packed contract; unsupported inputs use the eager reference.
"""

from __future__ import annotations

from importlib.util import find_spec

import torch
import torch.nn.functional as F
from torch.library import Library
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_omni.diffusion.layers.rope import apply_rotary_emb_torch
from vllm_omni.platforms import current_omni_platform

_FUSED_HEAD_DIM = 128
_FUSED_ROTARY_DIM = 96
_HEADS_PER_PROGRAM = 8


def _apply_rope_table(
    x: torch.Tensor,
    rope_table: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    half = rotary_dim // 2
    cos = rope_table[..., :half].to(x.dtype).unsqueeze(1)
    sin = rope_table[..., half:].to(x.dtype).unsqueeze(1)
    first = x[..., :half]
    second = x[..., half:rotary_dim]
    return torch.cat(
        (
            first * cos - second * sin,
            second * cos + first * sin,
            x[..., rotary_dim:],
        ),
        dim=-1,
    )


if HAS_TRITON:

    @triton.jit
    def _rms_norm_rope_kernel(
        x_ptr,
        weight_ptr,
        rope_table_ptr,
        out_ptr,
        x_stride_t,
        x_stride_h,
        x_stride_d,
        rope_stride_t,
        out_stride_t,
        out_stride_h,
        out_stride_d,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        rotary_half: tl.constexpr,
        eps: tl.constexpr,
        heads_per_program: tl.constexpr,
    ):
        token = tl.program_id(0)
        head_group = tl.program_id(1)
        heads = head_group * heads_per_program + tl.arange(0, heads_per_program)
        dims = tl.arange(0, head_dim)
        mask = heads[:, None] < num_heads
        offsets = token * x_stride_t + heads[:, None] * x_stride_h + dims[None, :] * x_stride_d

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + dims).to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(x * x, axis=1) / head_dim + eps)
        normalized = (x * inv_rms[:, None] * weight[None, :]).to(tl.bfloat16)

        rotary_dim = rotary_half * 2
        pair_dims = tl.where(
            dims < rotary_half,
            dims + rotary_half,
            tl.where(dims < rotary_dim, dims - rotary_half, dims),
        )
        pair_offsets = token * x_stride_t + heads[:, None] * x_stride_h + pair_dims[None, :] * x_stride_d
        pair_x = tl.load(x_ptr + pair_offsets, mask=mask, other=0.0).to(tl.float32)
        pair_weight = tl.load(weight_ptr + pair_dims).to(tl.float32)
        pair_normalized = (pair_x * inv_rms[:, None] * pair_weight[None, :]).to(tl.bfloat16)

        freq_dims = tl.where(
            dims < rotary_half,
            dims,
            tl.where(dims < rotary_dim, dims - rotary_half, 0),
        )
        table_offsets = token * rope_stride_t + freq_dims
        cos = tl.load(rope_table_ptr + table_offsets).to(tl.float32)
        sin = tl.load(rope_table_ptr + table_offsets + rotary_half).to(tl.float32)
        first = normalized.to(tl.float32) * cos - pair_normalized.to(tl.float32) * sin
        second = normalized.to(tl.float32) * cos + pair_normalized.to(tl.float32) * sin
        output = tl.where(
            dims < rotary_dim,
            tl.where(dims < rotary_half, first, second),
            normalized.to(tl.float32),
        )

        out_offsets = token * out_stride_t + heads[:, None] * out_stride_h + dims[None, :] * out_stride_d
        tl.store(out_ptr + out_offsets, output, mask=mask)

    @triton.jit
    def _qk_norm_rope_interleaved_kernel(
        q_ptr,
        k_ptr,
        q_out_ptr,
        k_out_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_ptr,
        sin_ptr,
        q_stride_b,
        q_stride_s,
        q_stride_h,
        q_stride_d,
        k_stride_b,
        k_stride_s,
        k_stride_h,
        k_stride_d,
        q_out_stride_b,
        q_out_stride_s,
        q_out_stride_h,
        q_out_stride_d,
        k_out_stride_b,
        k_out_stride_s,
        k_out_stride_h,
        k_out_stride_d,
        cos_stride_s,
        cos_stride_d,
        sin_stride_s,
        sin_stride_d,
        num_q_heads: tl.constexpr,
        head_dim: tl.constexpr,
        eps,
        input_dtype: tl.constexpr,
        rope_fp32: tl.constexpr,
        head_block: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        token_idx = tl.program_id(1)
        head_pid = tl.program_id(2)
        is_k = head_pid >= num_q_heads
        head_idx = tl.where(is_k, head_pid - num_q_heads, head_pid)

        offs = tl.arange(0, head_block)
        mask = offs < head_dim
        pair_offs = offs ^ 1

        in_base = tl.where(
            is_k,
            k_ptr + batch_idx * k_stride_b + token_idx * k_stride_s + head_idx * k_stride_h,
            q_ptr + batch_idx * q_stride_b + token_idx * q_stride_s + head_idx * q_stride_h,
        )
        in_stride_d = tl.where(is_k, k_stride_d, q_stride_d)
        vals = tl.load(in_base + offs * in_stride_d, mask=mask, other=0.0).to(tl.float32)
        pair_vals = tl.load(in_base + pair_offs * in_stride_d, mask=mask, other=0.0).to(tl.float32)

        rms = tl.rsqrt(tl.sum(vals * vals, axis=0) / head_dim + eps)
        weight_base = tl.where(is_k, k_weight_ptr, q_weight_ptr)
        weights = tl.load(weight_base + offs, mask=mask, other=0.0).to(tl.float32)
        pair_weights = tl.load(weight_base + pair_offs, mask=mask, other=0.0).to(tl.float32)
        # The existing path materializes RMSNorm output in the activation dtype
        # before RoPE. Keep the same rounding point for tighter equivalence.
        normed = (vals * rms * weights).to(input_dtype)
        pair_normed = (pair_vals * rms * pair_weights).to(input_dtype)

        rope_offs = offs // 2
        cos_vals = tl.load(
            cos_ptr + token_idx * cos_stride_s + rope_offs * cos_stride_d,
            mask=mask,
            other=1.0,
        )
        sin_vals = tl.load(
            sin_ptr + token_idx * sin_stride_s + rope_offs * sin_stride_d,
            mask=mask,
            other=0.0,
        )
        sign = tl.where(offs % 2 == 0, -1.0, 1.0)
        # Preserve each caller's existing RoPE rounding: Qwen supplies
        # activation-dtype tables, while callers with FP32 tables rotate in FP32.
        if rope_fp32:
            lhs = normed.to(tl.float32) * cos_vals.to(tl.float32)
            rhs = pair_normed.to(tl.float32) * sin_vals.to(tl.float32)
        else:
            lhs = (normed * cos_vals).to(input_dtype).to(tl.float32)
            rhs = (pair_normed * sin_vals).to(input_dtype).to(tl.float32)
        out = (lhs + sign * rhs).to(input_dtype)

        out_base = tl.where(
            is_k,
            k_out_ptr + batch_idx * k_out_stride_b + token_idx * k_out_stride_s + head_idx * k_out_stride_h,
            q_out_ptr + batch_idx * q_out_stride_b + token_idx * q_out_stride_s + head_idx * q_out_stride_h,
        )
        out_stride_d = tl.where(is_k, k_out_stride_d, q_out_stride_d)
        tl.store(out_base + offs * out_stride_d, out, mask=mask)


def _eager_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_norm = F.rms_norm(q, (head_dim,), q_weight, eps)
    k_norm = F.rms_norm(k, (head_dim,), k_weight, eps)
    return (
        _apply_rope_table(q_norm, rope_table, rotary_dim),
        _apply_rope_table(k_norm, rope_table, rotary_dim),
    )


def _triton_input_dtype(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float16:
        return tl.float16
    raise TypeError(f"Interleaved fused QK RMSNorm/RoPE only supports BF16/FP16 on Triton, got {dtype}")


def _interleaved_shape(q: torch.Tensor) -> tuple[int, int, int]:
    if q.ndim == 3:
        return 1, q.shape[0], q.shape[-1]
    if q.ndim == 4:
        return q.shape[0], q.shape[1], q.shape[-1]
    raise ValueError(f"q and k must be [tokens, heads, head_dim] or [batch, seq, heads, head_dim], got {q.shape}")


def _validate_interleaved_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[int, int, int, int]:
    batch, seq_len, head_dim = _interleaved_shape(q)
    k_batch, k_seq_len, k_head_dim = _interleaved_shape(k)
    if q.ndim != k.ndim or (batch, seq_len, head_dim) != (k_batch, k_seq_len, k_head_dim):
        raise ValueError(f"q and k shapes are incompatible: {q.shape} and {k.shape}")
    if q.dtype != k.dtype or q.device != k.device:
        raise ValueError("q and k must have the same dtype and device")
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"Fused QK RMSNorm/RoPE requires floating inputs, got {q.dtype}")
    if q_weight.shape != (head_dim,) or k_weight.shape != (head_dim,):
        raise ValueError(f"Expected norm weights [{head_dim}], got {tuple(q_weight.shape)} and {tuple(k_weight.shape)}")
    if q_weight.device != q.device or k_weight.device != q.device:
        raise ValueError("Q/K norm weights must be on the activation device")
    if cos.shape != sin.shape:
        raise ValueError(f"cos and sin shapes must match, got {cos.shape} and {sin.shape}")
    if cos.dtype != sin.dtype or cos.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"cos and sin must have the same floating dtype, got {cos.dtype} and {sin.dtype}")
    if cos.dtype not in (q.dtype, torch.float32):
        raise TypeError(f"cos and sin must use the activation dtype or float32, got {cos.dtype} for {q.dtype} inputs")
    if cos.device != q.device or sin.device != q.device:
        raise ValueError("RoPE tables must be on the activation device")
    if cos.ndim != 2 or cos.shape[0] != seq_len:
        raise ValueError(f"Expected cos/sin [{seq_len}, rotary_dim/2], got {tuple(cos.shape)}")

    rotary_dim = cos.shape[-1] * 2
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(f"rotary_dim must be even and in [2, {head_dim}], got {rotary_dim}")
    return batch, seq_len, head_dim, rotary_dim


def _normalize_interleaved_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    squeezed = q.ndim == 3
    if squeezed:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
    return q, k, cos, sin, squeezed


def _eager_qk_norm_rope_interleaved(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_norm = F.rms_norm(q, (head_dim,), q_weight, eps)
    k_norm = F.rms_norm(k, (head_dim,), k_weight, eps)
    return (
        apply_rotary_emb_torch(q_norm, cos, sin, interleaved=True).to(q.dtype),
        apply_rotary_emb_torch(k_norm, cos, sin, interleaved=True).to(k.dtype),
    )


def _interleaved_cuda_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    seq_len: int,
    head_dim: int,
    rotary_dim: int,
) -> bool:
    return (
        HAS_TRITON
        and current_platform.is_cuda()
        and q.is_cuda
        and k.is_cuda
        and q.dtype in (torch.bfloat16, torch.float16)
        and k.dtype == q.dtype
        and seq_len > 0
        and rotary_dim == head_dim
    )


def fused_qk_norm_rope_interleaved_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
    """Return whether inputs can use the shared interleaved Triton path."""
    try:
        _, seq_len, head_dim = _interleaved_shape(q)
        k_batch, k_seq_len, k_head_dim = _interleaved_shape(k)
    except ValueError:
        return False
    if q.ndim != k.ndim or k_seq_len != seq_len or k_head_dim != head_dim:
        return False
    if q.ndim == 4 and k_batch != q.shape[0]:
        return False
    if cos.shape != sin.shape or cos.ndim != 2:
        return False
    if cos.dtype != sin.dtype or cos.dtype not in (q.dtype, torch.float32):
        return False
    table_shape_ok = cos.shape[0] == seq_len
    rotary_dim = cos.shape[-1] * 2
    return (
        table_shape_ok
        and rotary_dim == head_dim
        and rotary_dim % 2 == 0
        and cos.device == q.device
        and sin.device == q.device
        and _interleaved_cuda_supported(q, k, seq_len, head_dim, rotary_dim)
    )


def _npu_apply_rope_table(
    x: torch.Tensor,
    rope_table: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    """Apply H3's rotary dimensions through Ascend's fused rotary kernel.

    ``mindiesd.rotary_position_embedding`` takes a 4-D BSND tensor, whereas
    MiniMax-H3 keeps packed activations as ``[tokens, heads, head_dim]``.
    The shared MindIE-SD wrapper normalizes this 3-D layout to BSND and
    restores it afterwards. It receives the half-width cos/sin values from
    the packed table; the wrapper expands them to H3's non-interleaved 96-D
    rotary layout. The 32 non-rotary head dimensions bypass the kernel.

    Some CANN environments package ``torch_npu`` without MindIE-SD. Keep
    those deployments on the same Ascend fused rotary primitive rather than
    silently falling back to eager elementwise RoPE.
    """
    import torch_npu

    half = rotary_dim // 2
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    cos = rope_table[..., :half]
    sin = rope_table[..., half:]

    if find_spec("mindiesd") is not None:
        from vllm_omni.diffusion.layers.rope import apply_rotary_emb_mindiesd

        x_rot = apply_rotary_emb_mindiesd(
            x_rot,
            cos,
            sin,
            interleaved=False,
            half_head_dim=True,
        )
    else:
        # npu_rotary_mul uses BSND and full rotary-width cos/sin. H3 uses
        # NeoX/rotated-half ordering, so duplicate rather than interleave the
        # half-width table along the final dimension.
        cos = cos.unsqueeze(0).unsqueeze(2).repeat(1, 1, 1, 2)
        sin = sin.unsqueeze(0).unsqueeze(2).repeat(1, 1, 1, 2)
        x_rot = torch_npu.npu_rotary_mul(x_rot.unsqueeze(0), cos, sin).squeeze(0)

    return torch.cat((x_rot, x_pass), dim=-1)


def _npu_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use Ascend RMSNorm and RoPE fused primitives for MiniMax-H3 DiT."""
    import torch_npu

    q_norm = torch_npu.npu_rms_norm(q, q_weight, epsilon=eps)[0]
    k_norm = torch_npu.npu_rms_norm(k, k_weight, epsilon=eps)[0]
    return (
        _npu_apply_rope_table(q_norm, rope_table, rotary_dim),
        _npu_apply_rope_table(k_norm, rope_table, rotary_dim),
    )


def _fused_cuda_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
) -> bool:
    return (
        HAS_TRITON
        and current_platform.is_cuda()
        and q.is_cuda
        and k.is_cuda
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and head_dim == _FUSED_HEAD_DIM
        and rotary_dim == _FUSED_ROTARY_DIM
    )


def _fused_npu_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
) -> bool:
    """Return whether the MiniMax-H3 Ascend fused-op contract is satisfied."""
    return (
        current_omni_platform.is_npu()
        and q.device.type == "npu"
        and k.device.type == "npu"
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and head_dim == _FUSED_HEAD_DIM
        and rotary_dim == _FUSED_ROTARY_DIM
    )


def _launch_fused_rms_norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    tokens, heads, head_dim = x.shape
    rotary_half = rope_table.shape[-1] // 2
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if tokens == 0:
        return out
    grid = (tokens, triton.cdiv(heads, _HEADS_PER_PROGRAM))
    _rms_norm_rope_kernel[grid](
        x,
        weight,
        rope_table,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        rope_table.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads=heads,
        head_dim=head_dim,
        rotary_half=rotary_half,
        eps=eps,
        heads_per_program=_HEADS_PER_PROGRAM,
        num_warps=8,
    )
    return out


def _launch_fused_qk_norm_rope_interleaved(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    *,
    num_warps: int,
    num_stages: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, seq_len, head_dim, rotary_dim = _validate_interleaved_inputs(q, k, q_weight, k_weight, cos, sin)
    if not _interleaved_cuda_supported(q, k, seq_len, head_dim, rotary_dim):
        raise RuntimeError("Interleaved fused QK RMSNorm/RoPE Triton path is not supported for these inputs")

    q, k, cos, sin, squeezed = _normalize_interleaved_inputs(q, k, cos, sin)
    q_weight = q_weight.contiguous()
    k_weight = k_weight.contiguous()
    q_out, k_out = _launch_prepared_qk_norm_rope_interleaved(
        q,
        k,
        q_weight,
        k_weight,
        cos,
        sin,
        eps,
        head_dim,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if squeezed:
        return q_out.squeeze(0), k_out.squeeze(0)
    return q_out, k_out


def _launch_prepared_qk_norm_rope_interleaved(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    head_dim: int,
    *,
    num_warps: int,
    num_stages: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    grid = (q.shape[0], q.shape[1], q.shape[2] + k.shape[2])
    _qk_norm_rope_interleaved_kernel[grid](
        q,
        k,
        q_out,
        k_out,
        q_weight,
        k_weight,
        cos,
        sin,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        q_out.stride(0),
        q_out.stride(1),
        q_out.stride(2),
        q_out.stride(3),
        k_out.stride(0),
        k_out.stride(1),
        k_out.stride(2),
        k_out.stride(3),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        num_q_heads=q.shape[2],
        head_dim=head_dim,
        eps=eps,
        input_dtype=_triton_input_dtype(q.dtype),
        rope_fp32=cos.dtype == torch.float32,
        head_block=triton.next_power_of_2(head_dim),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return q_out, k_out


def _fused_qk_norm_rope_interleaved_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _interleaved_cuda_supported(q, k, q.shape[1], head_dim, rotary_dim):
        return _eager_qk_norm_rope_interleaved(q, k, q_weight, k_weight, cos, sin, eps, head_dim)
    num_warps = 4
    num_stages = 4
    return _launch_prepared_qk_norm_rope_interleaved(
        q,
        k,
        q_weight,
        k_weight,
        cos,
        sin,
        eps,
        head_dim,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _fused_qk_norm_rope_interleaved_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del q_weight, k_weight, cos, sin, eps, head_dim, rotary_dim
    return torch.empty_like(q), torch.empty_like(k)


def _fused_qk_norm_rope_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _fused_npu_supported(q, k, head_dim, rotary_dim):
        return _npu_qk_norm_rope(
            q,
            k,
            q_weight,
            k_weight,
            rope_table,
            eps,
            rotary_dim,
        )
    if not _fused_cuda_supported(q, k, head_dim, rotary_dim):
        return _eager_qk_norm_rope(
            q,
            k,
            q_weight,
            k_weight,
            rope_table,
            eps,
            head_dim,
            rotary_dim,
        )
    return (
        _launch_fused_rms_norm_rope(q, q_weight, rope_table, eps),
        _launch_fused_rms_norm_rope(k, k_weight, rope_table, eps),
    )


def _fused_qk_norm_rope_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del q_weight, k_weight, rope_table, eps, head_dim, rotary_dim
    return torch.empty_like(q), torch.empty_like(k)


_OMNI_OP_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, "fused_qk_norm_rope"):
    direct_register_custom_op(
        op_name="fused_qk_norm_rope",
        op_func=_fused_qk_norm_rope_impl,
        fake_impl=_fused_qk_norm_rope_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )
if not hasattr(torch.ops.vllm_omni, "fused_qk_norm_rope_interleaved"):
    direct_register_custom_op(
        op_name="fused_qk_norm_rope_interleaved",
        op_func=_fused_qk_norm_rope_interleaved_impl,
        fake_impl=_fused_qk_norm_rope_interleaved_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )


def fused_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    *,
    head_dim: int | None = None,
    rotary_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Q/K RMSNorm and packed non-interleaved RoPE."""
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError(f"q and k must be [tokens, heads, head_dim], got {q.shape} and {k.shape}")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError(f"q and k shapes are incompatible: {q.shape} and {k.shape}")
    if q.dtype != k.dtype or q.device != k.device:
        raise ValueError("q and k must have the same dtype and device")
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"Fused QK RMSNorm/RoPE requires floating inputs, got {q.dtype}")

    head_dim = q.shape[-1] if head_dim is None else head_dim
    rotary_dim = rope_table.shape[-1] if rotary_dim is None else rotary_dim
    if q.shape[-1] != head_dim:
        raise ValueError(f"Expected q/k head_dim={head_dim}, got {q.shape[-1]}")
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(f"rotary_dim must be even and in [2, {head_dim}], got {rotary_dim}")
    if q_weight.shape != (head_dim,) or k_weight.shape != (head_dim,):
        raise ValueError(f"Expected norm weights [{head_dim}], got {tuple(q_weight.shape)} and {tuple(k_weight.shape)}")
    if q_weight.device != q.device or k_weight.device != q.device:
        raise ValueError("Q/K norm weights must be on the activation device")
    if rope_table.device != q.device or rope_table.dtype != q.dtype:
        raise ValueError("rope_table must have the same dtype and device as q/k")
    if rope_table.shape != (q.shape[0], rotary_dim):
        raise ValueError(f"Expected rope_table [{q.shape[0]}, {rotary_dim}], got {tuple(rope_table.shape)}")

    q_weight = q_weight.contiguous()
    k_weight = k_weight.contiguous()
    rope_table = rope_table.contiguous()
    if _fused_npu_supported(q, k, head_dim, rotary_dim):
        return _npu_qk_norm_rope(
            q,
            k,
            q_weight,
            k_weight,
            rope_table,
            eps,
            rotary_dim,
        )
    if not _fused_cuda_supported(q, k, head_dim, rotary_dim):
        return _fused_qk_norm_rope_impl(
            q,
            k,
            q_weight,
            k_weight,
            rope_table,
            eps,
            head_dim,
            rotary_dim,
        )
    return torch.ops.vllm_omni.fused_qk_norm_rope(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        eps,
        head_dim,
        rotary_dim,
    )


def fused_qk_norm_rope_interleaved(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Q/K RMSNorm and adjacent-pair RoPE with a shared CUDA fast path."""
    _, seq_len, head_dim, rotary_dim = _validate_interleaved_inputs(q, k, q_weight, k_weight, cos, sin)
    q, k, cos, sin, squeezed = _normalize_interleaved_inputs(q, k, cos, sin)
    q_weight = q_weight.contiguous()
    k_weight = k_weight.contiguous()

    if not _interleaved_cuda_supported(q, k, seq_len, head_dim, rotary_dim):
        q_out, k_out = _eager_qk_norm_rope_interleaved(
            q,
            k,
            q_weight,
            k_weight,
            cos,
            sin,
            eps,
            head_dim,
        )
    else:
        q_out, k_out = torch.ops.vllm_omni.fused_qk_norm_rope_interleaved(
            q,
            k,
            q_weight,
            k_weight,
            cos,
            sin,
            eps,
            head_dim,
            rotary_dim,
        )
    if squeezed:
        return q_out.squeeze(0), k_out.squeeze(0)
    return q_out, k_out


__all__ = [
    "fused_qk_norm_rope",
    "fused_qk_norm_rope_interleaved",
    "fused_qk_norm_rope_interleaved_supported",
]
