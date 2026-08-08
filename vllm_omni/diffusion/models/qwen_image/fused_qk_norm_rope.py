# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Qwen-Image Q/K RMSNorm followed by interleaved RoPE.

This module keeps the Qwen-Image-specific tensor layout and RoPE convention out
of the transformer modeling file. Inputs match ``QwenImageCrossAttention`` after
Q/K split and head unflattening:

* ``q`` and ``k`` are ``[batch, seq, heads, head_dim]``.
* ``cos`` and ``sin`` are Qwen-Image RoPE tables with width ``head_dim // 2``.
* RoPE uses the interleaved/GPT-J pairing used by
  ``RotaryEmbedding(is_neox_style=False)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.library import Library
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton

try:
    from vllm.utils import direct_register_custom_op
except ImportError:
    from vllm.utils.torch_utils import direct_register_custom_op


def _prepare_qwen_image_cos_sin(
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 3:
        # Match RotaryEmbedding._prepare_half_head_dim_cos_sin: Qwen-Image
        # shares one text/image RoPE table across the request batch.
        cos = cos[0]
        sin = sin[0]
    return cos.to(dtype), sin.to(dtype)


def _apply_interleaved_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    rotary_dim = cos.shape[-1] * 2
    x_rot = x[..., :rotary_dim]
    x_tail = x[..., rotary_dim:]
    x_even = x_rot[..., ::2]
    x_odd = x_rot[..., 1::2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    x_rotated = torch.stack(
        (
            x_even * cos - x_odd * sin,
            x_odd * cos + x_even * sin,
        ),
        dim=-1,
    ).flatten(-2)
    return torch.cat((x_rotated, x_tail), dim=-1)


def _eager_qwen_image_qk_norm_rope(
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
        _apply_interleaved_rope(q_norm, cos, sin),
        _apply_interleaved_rope(k_norm, cos, sin),
    )


def _triton_input_dtype(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float16:
        return tl.float16
    raise TypeError(f"Qwen-Image fused QK RMSNorm/RoPE only supports BF16/FP16 fast path, got {dtype}")


def _triton_supported(q: torch.Tensor, head_dim: int, rotary_dim: int) -> bool:
    return (
        HAS_TRITON
        and current_platform.is_cuda()
        and q.is_cuda
        and q.dtype in (torch.bfloat16, torch.float16)
        and head_dim > 0
        and rotary_dim == head_dim
        and q.shape[1] > 0
    )


def qwen_image_qk_norm_rope_fast_path_supported(q: torch.Tensor, cos: torch.Tensor) -> bool:
    """Return whether Qwen-Image Q/K RMSNorm+RoPE can use the Triton fast path."""
    if q.ndim != 4:
        return False
    if cos.dim() == 3:
        if cos.shape[0] == 0:
            return False
        cos = cos[0]
    if cos.ndim != 2 or cos.shape[0] != q.shape[1]:
        return False
    return _triton_supported(q, q.shape[-1], cos.shape[-1] * 2)


if HAS_TRITON:

    @triton.jit
    def _qwen_image_qk_norm_rope_kernel(
        q_ptr,
        k_ptr,
        q_out_ptr,
        k_out_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_ptr,
        sin_ptr,
        q_stride_b: tl.constexpr,
        q_stride_s: tl.constexpr,
        q_stride_h: tl.constexpr,
        q_stride_d: tl.constexpr,
        k_stride_b: tl.constexpr,
        k_stride_s: tl.constexpr,
        k_stride_h: tl.constexpr,
        k_stride_d: tl.constexpr,
        q_out_stride_b: tl.constexpr,
        q_out_stride_s: tl.constexpr,
        q_out_stride_h: tl.constexpr,
        q_out_stride_d: tl.constexpr,
        k_out_stride_b: tl.constexpr,
        k_out_stride_s: tl.constexpr,
        k_out_stride_h: tl.constexpr,
        k_out_stride_d: tl.constexpr,
        cos_stride_s: tl.constexpr,
        cos_stride_d: tl.constexpr,
        sin_stride_s: tl.constexpr,
        sin_stride_d: tl.constexpr,
        num_q_heads: tl.constexpr,
        head_dim: tl.constexpr,
        eps: tl.constexpr,
        input_dtype: tl.constexpr,
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
        cos_vals = tl.load(cos_ptr + token_idx * cos_stride_s + rope_offs * cos_stride_d, mask=mask, other=1.0)
        sin_vals = tl.load(sin_ptr + token_idx * sin_stride_s + rope_offs * sin_stride_d, mask=mask, other=0.0)
        sign = tl.where(offs % 2 == 0, -1.0, 1.0)
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


def _qwen_image_fused_qk_norm_rope_impl(
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
    if not _triton_supported(q, head_dim, rotary_dim):
        return _eager_qwen_image_qk_norm_rope(q, k, q_weight, k_weight, cos, sin, eps, head_dim)

    num_warps = 4
    num_stages = 4
    return _qwen_image_fused_qk_norm_rope_triton(
        q,
        k,
        q_weight,
        k_weight,
        cos,
        sin,
        eps,
        head_dim,
        rotary_dim,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _qwen_image_fused_qk_norm_rope_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
    *,
    num_warps: int,
    num_stages: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _triton_supported(q, head_dim, rotary_dim):
        raise RuntimeError("Qwen-Image fused QK RMSNorm/RoPE Triton path is not supported for these inputs")
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    head_block = triton.next_power_of_2(head_dim)
    grid = (q.shape[0], q.shape[1], q.shape[2] + k.shape[2])
    _qwen_image_qk_norm_rope_kernel[grid](
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
        head_block=head_block,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return q_out, k_out


def _qwen_image_fused_qk_norm_rope_fake(
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


def _qwen_image_fused_qk_norm_rope_op(
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
    return torch.ops.vllm_omni.qwen_image_fused_qk_norm_rope(
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


# The operator lifetime is tied to the Library object, so keep it alive at
# module scope while avoiding a config-looking uppercase global.
_omni_op_lib = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, "qwen_image_fused_qk_norm_rope"):
    direct_register_custom_op(
        op_name="qwen_image_fused_qk_norm_rope",
        op_func=_qwen_image_fused_qk_norm_rope_impl,
        fake_impl=_qwen_image_fused_qk_norm_rope_fake,
        mutates_args=[],
        target_lib=_omni_op_lib,
    )


def qwen_image_fused_qk_norm_rope_fast_path(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the Triton fast path after the caller has checked support."""
    head_dim = q.shape[-1]
    cos, sin = _prepare_qwen_image_cos_sin(cos, sin, dtype=q.dtype)
    rotary_dim = cos.shape[-1] * 2

    if not q_weight.is_contiguous():
        q_weight = q_weight.contiguous()
    if not k_weight.is_contiguous():
        k_weight = k_weight.contiguous()
    if not cos.is_contiguous():
        cos = cos.contiguous()
    if not sin.is_contiguous():
        sin = sin.contiguous()

    return _qwen_image_fused_qk_norm_rope_op(
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


def qwen_image_fused_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Q/K RMSNorm and Qwen-Image interleaved RoPE with a CUDA fast path."""
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(f"q and k must be [batch, seq, heads, head_dim], got {q.shape} and {k.shape}")
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[3] != k.shape[3]:
        raise ValueError(f"q and k shapes are incompatible: {q.shape} and {k.shape}")
    if q.dtype != k.dtype:
        raise ValueError(f"q and k must have the same dtype, got {q.dtype} and {k.dtype}")
    if q.device != k.device:
        raise ValueError(f"q and k must be on the same device, got {q.device} and {k.device}")
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"Qwen-Image fused QK RMSNorm/RoPE only supports floating inputs, got {q.dtype}")

    head_dim = q.shape[-1]
    cos, sin = _prepare_qwen_image_cos_sin(cos, sin, dtype=q.dtype)
    rotary_dim = cos.shape[-1] * 2
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(f"rotary_dim must be even and in [2, {head_dim}], got {rotary_dim}")
    if cos.shape != sin.shape:
        raise ValueError(f"cos and sin shapes must match, got {cos.shape} and {sin.shape}")
    if cos.shape != (q.shape[1], rotary_dim // 2):
        raise ValueError(f"Expected cos/sin [{q.shape[1]}, {rotary_dim // 2}], got {tuple(cos.shape)}")
    if q_weight.shape != (head_dim,) or k_weight.shape != (head_dim,):
        raise ValueError(
            f"Expected one norm weight of shape [{head_dim}], got {tuple(q_weight.shape)} and {tuple(k_weight.shape)}"
        )
    if q_weight.device != q.device or k_weight.device != q.device:
        raise ValueError("Q/K norm weights must be on the activation device")
    if cos.device != q.device or sin.device != q.device:
        raise ValueError("RoPE tables must be on the activation device")

    if not q_weight.is_contiguous():
        q_weight = q_weight.contiguous()
    if not k_weight.is_contiguous():
        k_weight = k_weight.contiguous()
    if not cos.is_contiguous():
        cos = cos.contiguous()
    if not sin.is_contiguous():
        sin = sin.contiguous()

    if not _triton_supported(q, head_dim, rotary_dim):
        return _qwen_image_fused_qk_norm_rope_impl(q, k, q_weight, k_weight, cos, sin, eps, head_dim, rotary_dim)
    return _qwen_image_fused_qk_norm_rope_op(
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


__all__ = [
    "qwen_image_fused_qk_norm_rope",
]
