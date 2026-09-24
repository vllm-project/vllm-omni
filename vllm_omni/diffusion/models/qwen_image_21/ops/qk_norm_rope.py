# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fused Q/K RMSNorm + interleaved RoPE for the Qwen-Image 2.1 single-stream block.

The eager block splits the packed ``to_qkv`` projection, normalizes Q and K and then
rotates them with an interleaved complex-pair RoPE. Per block per step that is a chain of
pointwise kernels over two full head stacks (FP32 reduction, rounding cast, learned scale,
FP32 promotion, complex multiply, output cast). This module performs the whole chain in one
Triton launch over the pack region holding Q followed by K, so no BF16 Q/K copy is
materialized between the normalization and the rotation.

Rounding contract, matching ``QwenImage21Attention.forward``::

    var  = fp32 mean of x^2 over the head dim       # fp32 reduction
    unit = (x * rsqrt(var + eps)).to(bfloat16)      # round BEFORE the learned scale
    norm = unit * weight                            # bfloat16 x bfloat16
    out  = interleaved_complex_mul(norm, freqs)     # fp32, then one final round

The round-to-BF16 between the normalization and the scale is load bearing: the upstream
zero-tolerance Q/K test pins it, so the kernel reproduces that rounding point explicitly
instead of folding the scale into the FP32 reduction. RoPE here is the *interleaved*
complex-pair layout rather than rotate-half, which is why ``layers/fused_qk_norm_rope.py``
(rotate-half, real cos/sin table) cannot serve this model.
"""

from __future__ import annotations

import torch
from torch.library import Library
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_omni.diffusion.layers.numerics import (
    add_rn_f32,
    fma_rn_f32,
    mul_rn_f32,
    round_bf16_to_fp32,
    rsqrt_approx_f32,
)
from vllm_omni.platforms import current_omni_platform

_HEADS_PER_PROGRAM = 4
_SUPPORTED_DTYPE = torch.bfloat16
_OP_NAME = "qwen_image_21_qk_norm_rope"


@triton.jit
def _qk_norm_rope_kernel(
    qk_ptr,
    q_weight_ptr,
    k_weight_ptr,
    freqs_ptr,
    out_ptr,
    qk_stride_b,
    qk_stride_s,
    qk_stride_h,
    qk_stride_d,
    freqs_stride_s,
    out_stride_b,
    out_stride_s,
    out_stride_h,
    out_stride_d,
    num_q_heads,
    num_heads,
    mean_factor,
    eps,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    heads_per_program: tl.constexpr,
):
    token = tl.program_id(0)
    batch = tl.program_id(1)
    slot = tl.program_id(2) * heads_per_program + tl.arange(0, heads_per_program)
    is_q = (slot < num_q_heads)[:, None]
    live = (slot < num_heads)[:, None]

    # The packed input already holds Q heads followed by K heads, and the packed output
    # keeps that order, so one head index serves both sides.
    token_base = batch * qk_stride_b + token * qk_stride_s
    dims = tl.arange(0, head_dim)
    row = token_base + slot[:, None] * qk_stride_h
    x = tl.load(qk_ptr + row + dims[None, :] * qk_stride_d, mask=live, other=0.0).to(tl.float32)

    variance = mul_rn_f32(tl.sum(mul_rn_f32(x, x), axis=1), mean_factor)
    inv_rms = rsqrt_approx_f32(add_rn_f32(variance, eps))

    q_weight = tl.load(q_weight_ptr + dims).to(tl.float32)
    k_weight = tl.load(k_weight_ptr + dims).to(tl.float32)
    weight = tl.where(is_q, q_weight[None, :], k_weight[None, :])

    unit = round_bf16_to_fp32(mul_rn_f32(x, inv_rms[:, None]))
    normed = round_bf16_to_fp32(mul_rn_f32(unit, weight))

    # Pair p owns head dims (2p, 2p + 1). `freqs` is complex64, so its FP32 view holds
    # cos at [..., 2p] and sin at [..., 2p + 1] over the same pair index.
    real, imag = tl.split(tl.reshape(normed, (heads_per_program, half_dim, 2)))
    pairs = tl.arange(0, half_dim)
    freq_offset = token * freqs_stride_s + pairs * 2
    cos = tl.load(freqs_ptr + freq_offset)
    sin = tl.load(freqs_ptr + freq_offset + 1)
    # ATen's complex64 multiply contracts one of the two products in each part, and the
    # fused form is what reproduces it bit for bit: re = fma(a, c, -(b * d)),
    # im = fma(b, c, a * d).
    out_real = fma_rn_f32(real, cos[None, :], -mul_rn_f32(imag, sin[None, :]))
    out_imag = fma_rn_f32(imag, cos[None, :], mul_rn_f32(real, sin[None, :]))

    out_row = batch * out_stride_b + token * out_stride_s + slot[:, None] * out_stride_h
    pair_offset = out_row + pairs[None, :] * (2 * out_stride_d)
    tl.store(out_ptr + pair_offset, out_real, mask=live)
    tl.store(out_ptr + pair_offset + out_stride_d, out_imag, mask=live)


def _launch(
    qk: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    batch, seq, _, head_dim = qk.shape
    num_heads = num_q_heads + num_kv_heads
    out = torch.empty((batch, seq, num_heads, head_dim), dtype=qk.dtype, device=qk.device)
    if out.numel() == 0:
        return out
    rows = batch * seq * num_heads
    # Complex64 -> FP32 doubles the stride of every outer axis, so read the view's own
    # strides rather than the complex tensor's.
    view = freqs.view(torch.float32)
    grid = (seq, batch, triton.cdiv(num_heads, _HEADS_PER_PROGRAM))
    _qk_norm_rope_kernel[grid](
        qk,
        q_weight,
        k_weight,
        view,
        out,
        qk.stride(0),
        qk.stride(1),
        qk.stride(2),
        qk.stride(3),
        view.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        num_q_heads,
        num_heads,
        float(rows) / float(rows * head_dim),
        eps,
        head_dim=head_dim,
        half_dim=head_dim // 2,
        heads_per_program=_HEADS_PER_PROGRAM,
        num_warps=4,
    )
    return out


def _reference(
    qk: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    q, k = qk.split([num_q_heads, num_kv_heads], dim=-2)
    variance = q.float().square().mean(-1, keepdim=True)
    q = (q * torch.rsqrt(variance + eps)).to(q.dtype) * q_weight
    variance = k.float().square().mean(-1, keepdim=True)
    k = (k * torch.rsqrt(variance + eps)).to(k.dtype) * k_weight
    return torch.cat(
        [
            torch.view_as_real(torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2)) * freqs.unsqueeze(1))
            .flatten(-2)
            .to(q.dtype),
            torch.view_as_real(torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2)) * freqs.unsqueeze(1))
            .flatten(-2)
            .to(k.dtype),
        ],
        dim=-2,
    )


def _supported(qk: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor, freqs: torch.Tensor) -> bool:
    head_dim = qk.shape[-1]
    return (
        HAS_TRITON
        and current_omni_platform.is_cuda()
        and qk.is_cuda
        and qk.dtype is _SUPPORTED_DTYPE
        and qk.ndim == 4
        and qk.stride(-1) == 1
        and head_dim >= 2
        and head_dim & (head_dim - 1) == 0
        and q_weight.shape == (head_dim,)
        and k_weight.shape == (head_dim,)
        and q_weight.dtype is _SUPPORTED_DTYPE
        and k_weight.dtype is _SUPPORTED_DTYPE
        and freqs.dtype is torch.complex64
        and freqs.shape[0] == qk.shape[1]
        and freqs.shape[1] * 2 == head_dim
        and freqs.stride(-1) == 1
    )


def _fused_qk_norm_rope(
    qk: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    if not _supported(qk, q_weight, k_weight, freqs):
        return _reference(qk, q_weight, k_weight, freqs, eps, num_q_heads, num_kv_heads)
    return _launch(qk, q_weight, k_weight, freqs, eps, num_q_heads, num_kv_heads)


def _fused_qk_norm_rope_fake(
    qk: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    del q_weight, k_weight, freqs, eps
    return torch.empty(
        (qk.shape[0], qk.shape[1], num_q_heads + num_kv_heads, qk.shape[3]),
        dtype=qk.dtype,
        device=qk.device,
    )


_OMNI_OP_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, _OP_NAME):
    direct_register_custom_op(
        op_name=_OP_NAME,
        op_func=_fused_qk_norm_rope,
        fake_impl=_fused_qk_norm_rope_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )


def apply_qk_norm_rope(
    qk: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    eps: float,
    num_q_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize and rotate the packed Q-then-K head region of a ``to_qkv`` output.

    ``qk`` is ``[batch, seq, num_q_heads + num_kv_heads, head_dim]``. For a
    ``QKVParallelLinear`` output that is exactly ``qkv[..., : q_size + kv_size]``
    unflattened, so it stays a strided view and nothing is repacked. The returned Q and K
    own independent storage.
    """
    if qk.ndim != 4:
        raise ValueError(f"qk must be [batch, seq, heads, head_dim], got {tuple(qk.shape)}")
    if not 0 < num_q_heads <= qk.shape[2]:
        raise ValueError(f"num_q_heads={num_q_heads} is outside the packed head count {qk.shape[2]}")
    num_kv_heads = qk.shape[2] - num_q_heads
    if not _supported(qk, q_weight, k_weight, freqs):
        # CPU and every ineligible layout stay on plain torch, so the op never needs a
        # non-CUDA kernel registration.
        out = _reference(qk, q_weight, k_weight, freqs, eps, num_q_heads, num_kv_heads)
    else:
        out = torch.ops.vllm_omni.qwen_image_21_qk_norm_rope(
            qk, q_weight, k_weight, freqs, eps, num_q_heads, num_kv_heads
        )
    return out.split([num_q_heads, num_kv_heads], dim=-2)


__all__ = ["apply_qk_norm_rope"]
