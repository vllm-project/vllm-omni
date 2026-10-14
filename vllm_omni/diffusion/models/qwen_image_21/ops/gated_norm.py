# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fuse the attention residual and following norm/modulation in compiled blocks."""

import torch
from torch.library import Library
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_omni.diffusion.layers.numerics import add_rn_f32, mul_rn_f32, round_bf16_to_fp32, tanh_f32
from vllm_omni.diffusion.models.qwen_image_21.ops.modulation import (
    _select_row,
    _supported,
    apply_gated_residual,
    apply_modulation,
)


@triton.jit
def _kernel(
    residual,
    sublayer,
    gate,
    scale,
    mask,
    hidden,
    out,
    seq: tl.constexpr,
    rows,
    gate_stride,
    scale_stride,
    eps: tl.constexpr,
    has_mask: tl.constexpr,
    prepared: tl.constexpr,
):
    batch, token = tl.program_id(0), tl.program_id(1)
    col = tl.arange(0, 4096)
    offset = (batch * seq + token) * 4096 + col
    row = _select_row(batch, token, mask, rows, has_mask)
    g = tl.load(gate + row * gate_stride + col).to(tl.float32)
    s = tl.load(sublayer + offset).to(tl.float32)
    r = tl.load(residual + offset).to(tl.float32)
    if not prepared:
        g = round_bf16_to_fp32(tanh_f32(g))
    x = round_bf16_to_fp32(add_rn_f32(r, round_bf16_to_fp32(mul_rn_f32(g, s))))
    mean = tl.sum(x, 0) / 4096
    centered = x - mean
    variance = tl.sum(centered * centered, 0) / 4096
    norm = centered * tl.rsqrt(variance + eps)
    weight = tl.load(scale + row * scale_stride + col).to(tl.float32)
    if not prepared:
        weight = round_bf16_to_fp32(add_rn_f32(1.0, weight))
    tl.store(hidden + offset, x)
    tl.store(out + offset, mul_rn_f32(norm, weight))


def _launch(
    residual: torch.Tensor,
    sublayer: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    token_mask: torch.Tensor | None,
    eps: float,
    prepared: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden, out = torch.empty_like(residual), torch.empty_like(residual)
    if residual.numel():
        _kernel[(residual.shape[0], residual.shape[1])](
            residual,
            sublayer,
            gate,
            scale,
            token_mask,
            hidden,
            out,
            residual.shape[1],
            gate.shape[0],
            gate.stride(0),
            scale.stride(0),
            eps,
            token_mask is not None,
            prepared,
            num_warps=8,
            enable_fp_fusion=False,
        )
    return hidden, out


def _fake(
    residual: torch.Tensor,
    sublayer: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    token_mask: torch.Tensor | None,
    eps: float,
    prepared: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(residual), torch.empty_like(residual)


_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, "qwen_image_21_gated_norm"):
    direct_register_custom_op(
        op_name="qwen_image_21_gated_norm",
        op_func=_launch,
        fake_impl=_fake,
        mutates_args=[],
        target_lib=_LIB,
    )


def apply_gated_norm_modulation(
    residual: torch.Tensor,
    sublayer: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    token_mask: torch.Tensor | None,
    norm: torch.nn.LayerNorm,
    prepared: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the residual and modulated LayerNorm input to the MLP.

    Eager uses native LayerNorm. The compiled BF16/4096 path preserves residual
    rounding and keeps the norm/scale epilogue in FP32. Its reduction and fused
    epilogue can differ from native Welford at BF16 rounding boundaries.
    """
    if (
        torch.compiler.is_compiling()
        and residual.shape[-1] == 4096
        and _supported(residual, gate, token_mask)
        and _supported(residual, scale, token_mask)
        and sublayer.dtype == residual.dtype
        and sublayer.shape == residual.shape
        and sublayer.is_contiguous()
    ):
        return torch.ops.vllm_omni.qwen_image_21_gated_norm(
            residual,
            sublayer,
            gate,
            scale,
            token_mask,
            norm.eps,
            prepared,
        )
    hidden = apply_gated_residual(residual, sublayer, gate, token_mask, prepared)
    normed = norm(hidden)
    return hidden, apply_modulation(normed, scale, token_mask, prepared)
