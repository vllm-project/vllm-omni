# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from vllm.triton_utils import tl, triton
except ImportError:
    import triton
    import triton.language as tl


@triton.jit
def _layernorm_select01_kernel(
    output_ptr,
    gate_out_ptr,
    x_ptr,
    weight_ptr,
    bias_ptr,
    scale0_ptr,
    shift0_ptr,
    gate0_ptr,
    scale1_ptr,
    shift1_ptr,
    gate1_ptr,
    index_ptr,
    inner_dim,
    seq_len,
    stride_x_row,
    stride_out_row,
    stride_gate_out_row,
    stride_w,
    stride_b,
    stride_s0_b,
    stride_s0_c,
    stride_sh0_b,
    stride_sh0_c,
    stride_g0_b,
    stride_g0_c,
    stride_s1_b,
    stride_s1_c,
    stride_sh1_b,
    stride_sh1_c,
    stride_g1_b,
    stride_g1_c,
    stride_i_b,
    stride_i_l,
    eps,
    has_weight: tl.constexpr,
    has_bias: tl.constexpr,
    block_n: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block_n)
    mask = cols < inner_dim

    x_row_ptr = x_ptr + row * stride_x_row
    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / inner_dim
    xbar = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / inner_dim
    x_hat = (x - mean) * tl.rsqrt(var + eps)

    if has_weight:
        weight = tl.load(weight_ptr + cols * stride_w, mask=mask, other=1.0).to(tl.float32)
        x_hat = x_hat * weight
    if has_bias:
        bias = tl.load(bias_ptr + cols * stride_b, mask=mask, other=0.0).to(tl.float32)
        x_hat = x_hat + bias

    batch_idx = row // seq_len
    seq_idx = row % seq_len
    idx = tl.load(index_ptr + batch_idx * stride_i_b + seq_idx * stride_i_l).to(tl.int1)

    scale0_ptrs = scale0_ptr + batch_idx * stride_s0_b + cols * stride_s0_c
    shift0_ptrs = shift0_ptr + batch_idx * stride_sh0_b + cols * stride_sh0_c
    gate0_ptrs = gate0_ptr + batch_idx * stride_g0_b + cols * stride_g0_c
    scale1_ptrs = scale1_ptr + batch_idx * stride_s1_b + cols * stride_s1_c
    shift1_ptrs = shift1_ptr + batch_idx * stride_sh1_b + cols * stride_sh1_c
    gate1_ptrs = gate1_ptr + batch_idx * stride_g1_b + cols * stride_g1_c

    scale_ptrs = tl.where(idx, scale1_ptrs, scale0_ptrs)
    shift_ptrs = tl.where(idx, shift1_ptrs, shift0_ptrs)
    gate_ptrs = tl.where(idx, gate1_ptrs, gate0_ptrs)

    scale = tl.load(scale_ptrs, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptrs, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptrs, mask=mask, other=0.0)

    y = x_hat * (1.0 + scale) + shift
    tl.store(output_ptr + row * stride_out_row + cols, y, mask=mask)
    tl.store(gate_out_ptr + row * stride_gate_out_row + cols, gate, mask=mask)


@triton.jit
def _residual_layernorm_select01_kernel(
    output_ptr,
    residual_out_ptr,
    gate_out_ptr,
    x_ptr,
    residual_ptr,
    residual_gate_ptr,
    weight_ptr,
    bias_ptr,
    scale0_ptr,
    shift0_ptr,
    gate0_ptr,
    scale1_ptr,
    shift1_ptr,
    gate1_ptr,
    index_ptr,
    inner_dim,
    seq_len,
    stride_x_row,
    stride_res_row,
    stride_res_gate_row,
    stride_out_row,
    stride_res_out_row,
    stride_gate_out_row,
    stride_w,
    stride_b,
    stride_s0_b,
    stride_s0_c,
    stride_sh0_b,
    stride_sh0_c,
    stride_g0_b,
    stride_g0_c,
    stride_s1_b,
    stride_s1_c,
    stride_sh1_b,
    stride_sh1_c,
    stride_g1_b,
    stride_g1_c,
    stride_i_b,
    stride_i_l,
    eps,
    has_weight: tl.constexpr,
    has_bias: tl.constexpr,
    block_n: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block_n)
    mask = cols < inner_dim

    x = tl.load(x_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + row * stride_res_row + cols, mask=mask, other=0.0).to(tl.float32)
    residual_gate = tl.load(
        residual_gate_ptr + row * stride_res_gate_row + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    residual_out = residual + residual_gate * x
    tl.store(residual_out_ptr + row * stride_res_out_row + cols, residual_out, mask=mask)

    mean = tl.sum(residual_out, axis=0) / inner_dim
    xbar = tl.where(mask, residual_out - mean, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / inner_dim
    x_hat = (residual_out - mean) * tl.rsqrt(var + eps)

    if has_weight:
        weight = tl.load(weight_ptr + cols * stride_w, mask=mask, other=1.0).to(tl.float32)
        x_hat = x_hat * weight
    if has_bias:
        bias = tl.load(bias_ptr + cols * stride_b, mask=mask, other=0.0).to(tl.float32)
        x_hat = x_hat + bias

    batch_idx = row // seq_len
    seq_idx = row % seq_len
    idx = tl.load(index_ptr + batch_idx * stride_i_b + seq_idx * stride_i_l).to(tl.int1)

    scale0_ptrs = scale0_ptr + batch_idx * stride_s0_b + cols * stride_s0_c
    shift0_ptrs = shift0_ptr + batch_idx * stride_sh0_b + cols * stride_sh0_c
    gate0_ptrs = gate0_ptr + batch_idx * stride_g0_b + cols * stride_g0_c
    scale1_ptrs = scale1_ptr + batch_idx * stride_s1_b + cols * stride_s1_c
    shift1_ptrs = shift1_ptr + batch_idx * stride_sh1_b + cols * stride_sh1_c
    gate1_ptrs = gate1_ptr + batch_idx * stride_g1_b + cols * stride_g1_c

    scale_ptrs = tl.where(idx, scale1_ptrs, scale0_ptrs)
    shift_ptrs = tl.where(idx, shift1_ptrs, shift0_ptrs)
    gate_ptrs = tl.where(idx, gate1_ptrs, gate0_ptrs)

    scale = tl.load(scale_ptrs, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptrs, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptrs, mask=mask, other=0.0)

    y = x_hat * (1.0 + scale) + shift
    tl.store(output_ptr + row * stride_out_row + cols, y, mask=mask)
    tl.store(gate_out_ptr + row * stride_gate_out_row + cols, gate, mask=mask)


def split_select01_mod_params(mod_params: torch.Tensor) -> tuple[torch.Tensor, ...]:
    shift, scale, gate = mod_params.chunk(3, dim=-1)
    actual_batch = shift.size(0) // 2
    if actual_batch == 0 or shift.size(0) != actual_batch * 2:
        raise ValueError("select01 modulation expects mod_params batch dim to be 2 * batch")
    return (
        scale[:actual_batch],
        shift[:actual_batch],
        gate[:actual_batch],
        scale[actual_batch:],
        shift[actual_batch:],
        gate[actual_batch:],
    )


def select01_modulation_native(
    mod_params: torch.Tensor,
    index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale0, shift0, gate0, scale1, shift1, gate1 = split_select01_mod_params(mod_params)
    idx = index.to(dtype=torch.bool).unsqueeze(-1)
    scale = torch.where(idx, scale1.unsqueeze(1), scale0.unsqueeze(1))
    shift = torch.where(idx, shift1.unsqueeze(1), shift0.unsqueeze(1))
    gate = torch.where(idx, gate1.unsqueeze(1), gate0.unsqueeze(1))
    return scale, shift, gate


def _validate_select01_inputs(
    x: torch.Tensor,
    index: torch.Tensor,
    tensors: tuple[torch.Tensor, ...],
) -> None:
    if x.dim() != 3:
        raise ValueError("x must be 3D [B, L, C]")
    if index.dim() != 2 or index.shape != x.shape[:2]:
        raise ValueError("index must be 2D [B, L] and match x batch/sequence")
    batch_size, _, hidden_size = x.shape
    for tensor in tensors:
        if tensor.dim() != 2 or tensor.shape != (batch_size, hidden_size):
            raise ValueError("scale/shift/gate tensors must be 2D [B, C]")


def _maybe_contiguous(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_contiguous() else x.contiguous()


def _launch_layernorm_select01(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_select01_inputs(x, index, (scale0, shift0, gate0, scale1, shift1, gate1))
    batch_size, seq_len, hidden_size = x.shape
    output = torch.empty_like(x)
    gate_out = torch.empty_like(x)

    x_2d = _maybe_contiguous(x).view(batch_size * seq_len, hidden_size)
    output_2d = output.view(batch_size * seq_len, hidden_size)
    gate_out_2d = gate_out.view(batch_size * seq_len, hidden_size)
    index = _maybe_contiguous(index)
    scale0 = _maybe_contiguous(scale0)
    shift0 = _maybe_contiguous(shift0)
    gate0 = _maybe_contiguous(gate0)
    scale1 = _maybe_contiguous(scale1)
    shift1 = _maybe_contiguous(shift1)
    gate1 = _maybe_contiguous(gate1)

    weight_arg = _maybe_contiguous(weight) if weight is not None else x_2d
    bias_arg = _maybe_contiguous(bias) if bias is not None else x_2d
    block_n = min(65536 // x_2d.element_size(), triton.next_power_of_2(hidden_size))
    if hidden_size > block_n:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    _layernorm_select01_kernel[(batch_size * seq_len,)](
        output_2d,
        gate_out_2d,
        x_2d,
        weight_arg,
        bias_arg,
        scale0,
        shift0,
        gate0,
        scale1,
        shift1,
        gate1,
        index,
        hidden_size,
        seq_len,
        x_2d.stride(0),
        output_2d.stride(0),
        gate_out_2d.stride(0),
        weight_arg.stride(0) if weight is not None else 0,
        bias_arg.stride(0) if bias is not None else 0,
        scale0.stride(0),
        scale0.stride(1),
        shift0.stride(0),
        shift0.stride(1),
        gate0.stride(0),
        gate0.stride(1),
        scale1.stride(0),
        scale1.stride(1),
        shift1.stride(0),
        shift1.stride(1),
        gate1.stride(0),
        gate1.stride(1),
        index.stride(0),
        index.stride(1),
        eps,
        has_weight=weight is not None,
        has_bias=bias is not None,
        block_n=block_n,
        num_warps=4,
        num_stages=4,
    )
    return output, gate_out


def _launch_residual_layernorm_select01(
    x: torch.Tensor,
    residual: torch.Tensor,
    residual_gate: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_select01_inputs(x, index, (scale0, shift0, gate0, scale1, shift1, gate1))
    if residual.shape != x.shape:
        raise ValueError("residual must have the same shape as x")
    if residual_gate.shape != x.shape:
        raise ValueError("residual_gate must have the same shape as x")

    batch_size, seq_len, hidden_size = x.shape
    output = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    gate_out = torch.empty_like(x)

    x_2d = _maybe_contiguous(x).view(batch_size * seq_len, hidden_size)
    residual_2d = _maybe_contiguous(residual).view(batch_size * seq_len, hidden_size)
    residual_gate_2d = _maybe_contiguous(residual_gate).view(batch_size * seq_len, hidden_size)
    output_2d = output.view(batch_size * seq_len, hidden_size)
    residual_out_2d = residual_out.view(batch_size * seq_len, hidden_size)
    gate_out_2d = gate_out.view(batch_size * seq_len, hidden_size)
    index = _maybe_contiguous(index)
    scale0 = _maybe_contiguous(scale0)
    shift0 = _maybe_contiguous(shift0)
    gate0 = _maybe_contiguous(gate0)
    scale1 = _maybe_contiguous(scale1)
    shift1 = _maybe_contiguous(shift1)
    gate1 = _maybe_contiguous(gate1)

    weight_arg = _maybe_contiguous(weight) if weight is not None else x_2d
    bias_arg = _maybe_contiguous(bias) if bias is not None else x_2d
    block_n = min(65536 // x_2d.element_size(), triton.next_power_of_2(hidden_size))
    if hidden_size > block_n:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    _residual_layernorm_select01_kernel[(batch_size * seq_len,)](
        output_2d,
        residual_out_2d,
        gate_out_2d,
        x_2d,
        residual_2d,
        residual_gate_2d,
        weight_arg,
        bias_arg,
        scale0,
        shift0,
        gate0,
        scale1,
        shift1,
        gate1,
        index,
        hidden_size,
        seq_len,
        x_2d.stride(0),
        residual_2d.stride(0),
        residual_gate_2d.stride(0),
        output_2d.stride(0),
        residual_out_2d.stride(0),
        gate_out_2d.stride(0),
        weight_arg.stride(0) if weight is not None else 0,
        bias_arg.stride(0) if bias is not None else 0,
        scale0.stride(0),
        scale0.stride(1),
        shift0.stride(0),
        shift0.stride(1),
        gate0.stride(0),
        gate0.stride(1),
        scale1.stride(0),
        scale1.stride(1),
        shift1.stride(0),
        shift1.stride(1),
        gate1.stride(0),
        gate1.stride(1),
        index.stride(0),
        index.stride(1),
        eps,
        has_weight=weight is not None,
        has_bias=bias is not None,
        block_n=block_n,
        num_warps=4,
        num_stages=4,
    )
    return output, residual_out, gate_out


def fused_layernorm_select01(
    x: torch.Tensor,
    mod_params: torch.Tensor,
    index: torch.Tensor,
    eps: float,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale0, shift0, gate0, scale1, shift1, gate1 = split_select01_mod_params(mod_params)
    is_compiling = torch.compiler.is_compiling()
    if x.is_cuda and not is_compiling:
        return _launch_layernorm_select01(
            x,
            weight,
            bias,
            scale0,
            shift0,
            gate0,
            scale1,
            shift1,
            gate1,
            index,
            eps,
        )

    scale, shift, gate = select01_modulation_native(mod_params, index)
    out = F.layer_norm(
        x.float(),
        (x.shape[-1],),
        weight=weight.float() if weight is not None else None,
        bias=bias.float() if bias is not None else None,
        eps=eps,
    ).to(x.dtype)
    return out * (1 + scale) + shift, gate


def fused_residual_layernorm_select01(
    x: torch.Tensor,
    residual: torch.Tensor,
    residual_gate: torch.Tensor,
    mod_params: torch.Tensor,
    index: torch.Tensor,
    eps: float,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale0, shift0, gate0, scale1, shift1, gate1 = split_select01_mod_params(mod_params)
    is_compiling = torch.compiler.is_compiling()
    if x.is_cuda and not is_compiling:
        return _launch_residual_layernorm_select01(
            x,
            residual,
            residual_gate,
            weight,
            bias,
            scale0,
            shift0,
            gate0,
            scale1,
            shift1,
            gate1,
            index,
            eps,
        )

    scale, shift, gate = select01_modulation_native(mod_params, index)
    residual_out = residual + residual_gate * x
    out = F.layer_norm(
        residual_out.float(),
        (residual_out.shape[-1],),
        weight=weight.float() if weight is not None else None,
        bias=bias.float() if bias is not None else None,
        eps=eps,
    ).to(residual_out.dtype)
    return out * (1 + scale) + shift, residual_out, gate
