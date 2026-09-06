# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exact Triton Q/K RMSNorm plus split-RoPE for LTX-2."""

from __future__ import annotations

import logging

import torch
from vllm.triton_utils import tl, triton

from ..numerics import fma_rn_f32, mul_rn_f32, rms_reciprocal_fma, round_bf16_to_fp32
from ..platform import is_ltx2_ops_eligible

_PAIR_BLOCK = 1024
_FAILED_DEVICES: set[int | None] = set()

logger = logging.getLogger(__name__)


@triton.jit
def _qknorm_split_rope_kernel(
    output_ptr,
    input_ptr,
    cos_ptr,
    sin_ptr,
    weight_ptr,
    sequence_length,
    input_stride_batch,
    input_stride_token,
    cos_stride_batch,
    cos_stride_head,
    cos_stride_token,
    sin_stride_batch,
    sin_stride_head,
    sin_stride_token,
    eps,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    hidden_size: tl.constexpr,
    has_weight: tl.constexpr,
    rope_is_bf16: tl.constexpr,
    pair_block: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    batch = row // sequence_length
    token = row % sequence_length
    input_base = batch * input_stride_batch + token * input_stride_token
    output_base = row * hidden_size
    reciprocal_rms = rms_reciprocal_fma(
        input_ptr,
        input_base,
        eps,
        hidden_size,
    )

    half_dim: tl.constexpr = head_dim // 2
    pair_count: tl.constexpr = hidden_size // 2
    for block in tl.static_range(pair_count // pair_block):
        pairs = block * pair_block + tl.arange(0, pair_block)
        valid = pairs < pair_count
        heads = pairs // half_dim
        head_offsets = pairs % half_dim
        first_columns = heads * head_dim + head_offsets
        second_columns = first_columns + half_dim

        first_norm = mul_rn_f32(
            tl.load(input_ptr + input_base + first_columns, mask=valid).to(tl.float32),
            reciprocal_rms,
        )
        second_norm = mul_rn_f32(
            tl.load(input_ptr + input_base + second_columns, mask=valid).to(tl.float32),
            reciprocal_rms,
        )
        if has_weight:
            first_weight = tl.load(weight_ptr + first_columns, mask=valid).to(tl.float32)
            second_weight = tl.load(weight_ptr + second_columns, mask=valid).to(tl.float32)
            first_norm = mul_rn_f32(first_norm, first_weight)
            second_norm = mul_rn_f32(second_norm, second_weight)
        first_norm = round_bf16_to_fp32(first_norm)
        second_norm = round_bf16_to_fp32(second_norm)

        cos_offsets = batch * cos_stride_batch + heads * cos_stride_head + token * cos_stride_token + head_offsets
        sin_offsets = batch * sin_stride_batch + heads * sin_stride_head + token * sin_stride_token + head_offsets
        cos = tl.load(cos_ptr + cos_offsets, mask=valid).to(tl.float32)
        sin = tl.load(sin_ptr + sin_offsets, mask=valid).to(tl.float32)
        first_base = mul_rn_f32(first_norm, cos)
        second_base = mul_rn_f32(second_norm, cos)
        if rope_is_bf16:
            first_base = round_bf16_to_fp32(first_base)
            second_base = round_bf16_to_fp32(second_base)
        first_output = fma_rn_f32(-sin, second_norm, first_base)
        second_output = fma_rn_f32(sin, first_norm, second_base)
        tl.store(output_ptr + output_base + first_columns, first_output, mask=valid)
        tl.store(output_ptr + output_base + second_columns, second_output, mask=valid)


def _supported_side(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    weight: torch.Tensor | None,
    *,
    num_heads: int,
    head_dim: int,
) -> bool:
    weight_ok = weight is None or (
        weight.is_cuda
        and weight.device == x.device
        and weight.dtype is x.dtype
        and weight.shape == (x.shape[2],)
        and weight.is_contiguous()
    )
    return (
        x.is_cuda
        and cos.is_cuda
        and sin.is_cuda
        and x.device == cos.device == sin.device
        and x.dtype is torch.bfloat16
        and cos.dtype in (torch.float32, torch.bfloat16)
        and sin.dtype is cos.dtype
        and x.ndim == 3
        and cos.ndim == 4
        and sin.ndim == 4
        and x.stride(-1) == 1
        and weight_ok
        and cos.shape == sin.shape
        and cos.shape[0] == x.shape[0]
        and cos.shape[1] == num_heads
        and cos.shape[2] == x.shape[1]
        and cos.shape[3] * 2 == head_dim
        and x.shape[2] == num_heads * head_dim
        and x.shape[2] in (2048, 4096)
        and head_dim % 2 == 0
        and x.shape[2] % 4 == 0
        and cos.stride(-1) == 1
        and sin.stride(-1) == 1
    )


def _supported_inputs(
    q: torch.Tensor,
    q_cos: torch.Tensor,
    q_sin: torch.Tensor,
    q_weight: torch.Tensor | None,
    k: torch.Tensor,
    k_cos: torch.Tensor,
    k_sin: torch.Tensor,
    k_weight: torch.Tensor | None,
    *,
    num_heads: int,
    head_dim: int,
) -> bool:
    return (
        is_ltx2_ops_eligible(q)
        and q.device.index not in _FAILED_DEVICES
        and q_cos.dtype is k_cos.dtype
        and q.device == k.device
        and _supported_side(q, q_cos, q_sin, q_weight, num_heads=num_heads, head_dim=head_dim)
        and _supported_side(k, k_cos, k_sin, k_weight, num_heads=num_heads, head_dim=head_dim)
    )


def _run_side(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    weight: torch.Tensor | None,
    *,
    eps: float,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    weight_arg = x if weight is None else weight
    with torch.accelerator.device_index(x.device.index):
        _qknorm_split_rope_kernel[(x.shape[0] * x.shape[1],)](
            output,
            x,
            cos,
            sin,
            weight_arg,
            x.shape[1],
            x.stride(0),
            x.stride(1),
            cos.stride(0),
            cos.stride(1),
            cos.stride(2),
            sin.stride(0),
            sin.stride(1),
            sin.stride(2),
            eps,
            num_heads=num_heads,
            head_dim=head_dim,
            hidden_size=x.shape[2],
            has_weight=weight is not None,
            rope_is_bf16=cos.dtype is torch.bfloat16,
            pair_block=_PAIR_BLOCK,
            num_warps=4,
        )
    return output


def _run_qknorm_split_rope(
    q: torch.Tensor,
    q_cos: torch.Tensor,
    q_sin: torch.Tensor,
    q_weight: torch.Tensor | None,
    k: torch.Tensor,
    k_cos: torch.Tensor,
    k_sin: torch.Tensor,
    k_weight: torch.Tensor | None,
    *,
    eps: float,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _supported_inputs(
        q,
        q_cos,
        q_sin,
        q_weight,
        k,
        k_cos,
        k_sin,
        k_weight,
        num_heads=num_heads,
        head_dim=head_dim,
    ):
        raise ValueError("unsupported tensors for LTX-2 Triton QKNorm split-RoPE")
    return (
        _run_side(q, q_cos, q_sin, q_weight, eps=eps, num_heads=num_heads, head_dim=head_dim),
        _run_side(k, k_cos, k_sin, k_weight, eps=eps, num_heads=num_heads, head_dim=head_dim),
    )


def try_qknorm_split_rope_exact(
    q: torch.Tensor,
    q_cos: torch.Tensor,
    q_sin: torch.Tensor,
    q_weight: torch.Tensor | None,
    k: torch.Tensor,
    k_cos: torch.Tensor,
    k_sin: torch.Tensor,
    k_weight: torch.Tensor | None,
    eps: float,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return the exact eager result, or ``None`` outside its contract."""

    attn_eps = float(eps)
    if attn_eps <= 0 or not _supported_inputs(
        q,
        q_cos,
        q_sin,
        q_weight,
        k,
        k_cos,
        k_sin,
        k_weight,
        num_heads=num_heads,
        head_dim=head_dim,
    ):
        return None
    try:
        return _run_qknorm_split_rope(
            q,
            q_cos,
            q_sin,
            q_weight,
            k,
            k_cos,
            k_sin,
            k_weight,
            eps=attn_eps,
            num_heads=num_heads,
            head_dim=head_dim,
        )
    except Exception as exc:  # noqa: BLE001 - fail closed after Triton failure
        _FAILED_DEVICES.add(q.device.index)
        logger.warning(
            "Disabling LTX-2 Triton QKNorm+split-RoPE on %s after failure: %s",
            q.device,
            exc,
        )
        return None


__all__ = ["try_qknorm_split_rope_exact"]
