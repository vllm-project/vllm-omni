# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fused FP8 prefix dequantization and prefix/target pack for KV-cache decode.

With ``prefix_kv_cache_dtype`` enabled, the timestep-independent prefix K/V is stored in FP8
E4M3 with one FP32 scale per `(batch, token, head)`. A non-SP decode step materializes that
prefix, prepends it to the fresh target K/V and hands the concatenation to the unchanged
attention backend. Eagerly that costs a cast, a multiply, a rounding cast and a copy per
tensor before the backend ever runs::

    prefix  -> fp32 -> * scale -> bf16 -> cat(prefix, target)

Both halves of the concatenation have a known destination, so this module writes them
straight into one output buffer instead: the prefix region receives the dequantized FP8
payload and the target region receives the fresh K/V. That removes the intermediate
dequantized prefix and the second full-size copy.

A prefix that is already in the native dtype has nothing to dequantize and keeps
``torch.cat``: folding a plain copy into this kernel measured slower than the copy kernel
it would replace, so the fusion is limited to the case that removes real work.

The attention kernels still consume BF16; nothing here changes the cache format, the scale
granularity, the quantization formula or the attention backend. The SP joint-prefix path
keeps its original eager dequantization because that buffer is handed to the all-to-all
entry rather than to ``torch.cat``.

Rounding contract, matching ``_dequantize_prefix_kv_fp8``::

    out = round_bf16(prefix_fp32 * scale)
"""

from __future__ import annotations

import torch
from torch.library import Library
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_omni.diffusion.layers.numerics import mul_rn_f32, round_bf16_to_fp32
from vllm_omni.platforms import current_omni_platform

_HEADS_PER_PROGRAM = 4
_OP_NAME = "qwen_image_21_concat_prefix_kv"


@triton.jit
def _concat_prefix_kv_kernel(
    prefix_ptr,
    scale_ptr,
    target_ptr,
    out_ptr,
    prefix_stride_b,
    prefix_stride_r,
    prefix_stride_h,
    scale_stride_b,
    scale_stride_r,
    scale_stride_h,
    target_stride_b,
    target_stride_r,
    target_stride_h,
    out_stride_b,
    out_stride_r,
    out_stride_h,
    num_heads,
    prefix_len,
    head_dim: tl.constexpr,
    heads_per_program: tl.constexpr,
):
    token = tl.program_id(0)
    batch = tl.program_id(1)
    heads = tl.program_id(2) * heads_per_program + tl.arange(0, heads_per_program)
    live = (heads < num_heads)[:, None]
    dims = tl.arange(0, head_dim)[None, :]
    out_offset = batch * out_stride_b + token * out_stride_r + heads[:, None] * out_stride_h + dims

    if token < prefix_len:
        # Distinct names per branch: Triton unifies same-named variables across an if/else
        # and these carry different dtypes.
        cached = tl.load(
            prefix_ptr + batch * prefix_stride_b + token * prefix_stride_r + heads[:, None] * prefix_stride_h + dims,
            mask=live,
            other=0.0,
        ).to(tl.float32)
        scale = tl.load(
            scale_ptr + batch * scale_stride_b + token * scale_stride_r + heads * scale_stride_h,
            mask=heads < num_heads,
            other=0.0,
        )
        cached = mul_rn_f32(cached, scale[:, None])
        tl.store(out_ptr + out_offset, round_bf16_to_fp32(cached), mask=live)
    else:
        fresh = tl.load(
            target_ptr
            + batch * target_stride_b
            + (token - prefix_len) * target_stride_r
            + heads[:, None] * target_stride_h
            + dims,
            mask=live,
            other=0.0,
        )
        tl.store(out_ptr + out_offset, fresh, mask=live)


def _reference(prefix: torch.Tensor, prefix_scale: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
    if prefix_scale is not None:
        prefix = (prefix.float() * prefix_scale).to(target.dtype)
    return torch.cat([prefix, target], dim=1)


def _supported(prefix: torch.Tensor, prefix_scale: torch.Tensor | None, target: torch.Tensor) -> bool:
    """Only an FP8 prefix with a scale has a dequantization worth folding into the pack."""
    return (
        prefix_scale is not None
        and HAS_TRITON
        and current_omni_platform.is_cuda()
        and target.is_cuda
        and target.dtype is torch.bfloat16
        and target.ndim == 4
        # The fresh K/V are strided views into the packed `to_qkv` output, so only the
        # innermost stride has to be unit; the kernel addresses them by full strides.
        and target.stride(-1) == 1
        and prefix.dtype is torch.float8_e4m3fn
        and prefix.stride(-1) == 1
        and prefix.shape[:1] == target.shape[:1]
        and prefix.shape[2:] == target.shape[2:]
        and prefix_scale.dtype is torch.float32
        and prefix_scale.is_cuda
        and prefix_scale.shape == (*prefix.shape[:3], 1)
        and prefix_scale.stride(-1) == 1
    )


def _launch(prefix: torch.Tensor, prefix_scale: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
    batch, prefix_len, num_heads, head_dim = prefix.shape
    out = torch.empty(
        (batch, prefix_len + target.shape[1], num_heads, head_dim), dtype=target.dtype, device=target.device
    )
    if out.numel() == 0:
        return out
    grid = (out.shape[1], batch, triton.cdiv(num_heads, _HEADS_PER_PROGRAM))
    _concat_prefix_kv_kernel[grid](
        prefix,
        prefix_scale,
        target,
        out,
        prefix.stride(0),
        prefix.stride(1),
        prefix.stride(2),
        prefix_scale.stride(0),
        prefix_scale.stride(1),
        prefix_scale.stride(2),
        target.stride(0),
        target.stride(1),
        target.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads,
        prefix_len,
        head_dim=head_dim,
        heads_per_program=_HEADS_PER_PROGRAM,
        num_warps=4,
    )
    return out


def _concat_prefix_kv_fake(
    prefix: torch.Tensor, prefix_scale: torch.Tensor | None, target: torch.Tensor
) -> torch.Tensor:
    del prefix_scale
    return torch.empty(
        (target.shape[0], prefix.shape[1] + target.shape[1], target.shape[2], target.shape[3]),
        dtype=target.dtype,
        device=target.device,
    )


_OMNI_OP_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, _OP_NAME):
    direct_register_custom_op(
        op_name=_OP_NAME,
        op_func=_launch,
        fake_impl=_concat_prefix_kv_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )


def concat_prefix_kv(prefix: torch.Tensor, prefix_scale: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
    """Return ``cat([dequant(prefix), target], dim=1)`` written in a single launch.

    ``prefix`` is ``[batch, prefix_len, heads, head_dim]`` FP8 E4M3 and ``prefix_scale`` is
    its ``[batch, prefix_len, heads, 1]`` FP32 scale; ``target`` is the current step's K or V.
    A native-dtype prefix (``prefix_scale is None``) takes the eager ``torch.cat``.
    """
    if target.ndim != 4 or prefix.ndim != 4:
        raise ValueError(f"expected 4-D prefix/target, got {tuple(prefix.shape)} and {tuple(target.shape)}")
    if not _supported(prefix, prefix_scale, target):
        return _reference(prefix, prefix_scale, target)
    return torch.ops.vllm_omni.qwen_image_21_concat_prefix_kv(prefix, prefix_scale, target)


__all__ = ["concat_prefix_kv"]
