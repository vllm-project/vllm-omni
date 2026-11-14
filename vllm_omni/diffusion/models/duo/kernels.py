# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pointwise Realization kernels preserving eager rounding and finite coverage."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["count"])
def _rotary_kernel(
    input_ptr,
    cos_ptr,
    sin_ptr,
    output_ptr,
    count,
    heads: tl.constexpr,
    dim: tl.constexpr,
    width: tl.constexpr,
    s0: tl.constexpr,
    s1: tl.constexpr,
    s2: tl.constexpr,
    c0: tl.constexpr,
    c2: tl.constexpr,
    block: tl.constexpr,
):
    i = tl.program_id(0) * block + tl.arange(0, block)
    row, head, d = i // (heads * dim), (i // dim) % heads, i % dim
    offset = row * s0 + head * s1
    x = tl.load(input_ptr + offset + d * s2, i < count, 0).to(tl.float32)
    half = width // 2
    partner = tl.where(d < half, d + half, d - half)
    other = tl.load(input_ptr + offset + partner * s2, (i < count) & (d < width), 0).to(tl.float32)
    other = tl.where(d < half, -other, other)
    c = tl.load(cos_ptr + row * c0 + d * c2, (i < count) & (d < width), 0).to(tl.float32)
    s = tl.load(sin_ptr + row * c0 + d * c2, (i < count) & (d < width), 0).to(tl.float32)
    # Eager has TWO separately rounded BF16/FP16 products before its add.
    # Explicit casts and disabled FMA preserve that operation boundary.
    a = (x * c).to(output_ptr.dtype.element_ty).to(tl.float32)
    b = (other * s).to(output_ptr.dtype.element_ty).to(tl.float32)
    y = tl.where(d < width, a + b, x)
    tl.store(output_ptr + i, y, i < count)


def rotary_cuda(value, coefficients):
    cos, sin = coefficients
    rows, heads, dim = value.shape
    width = cos.shape[-1]
    if (
        value.device.type != "cuda"
        or value.dtype not in (torch.bfloat16, torch.float16)
        or cos.shape != sin.shape
        or cos.shape != (rows, 1, width)
        or cos.dtype != value.dtype
        or sin.dtype != value.dtype
        or cos.device != value.device
        or sin.device != value.device
        or cos.stride() != sin.stride()
        or width > dim
        or width % 2
    ):
        raise ValueError("Unsupported fused Realization rotary signature")
    output = torch.empty(value.shape, device=value.device, dtype=value.dtype)
    _rotary_kernel[(triton.cdiv(value.numel(), 256),)](
        value,
        cos,
        sin,
        output,
        value.numel(),
        heads,
        dim,
        width,
        *value.stride(),
        cos.stride(0),
        cos.stride(2),
        256,
        enable_fp_fusion=False,
    )
    return output


@triton.jit(do_not_specialize=["count", "chunks"])
def _new_kv_finite(
    storage_ptr,
    slots_ptr,
    partial_ptr,
    count,
    row_width: tl.constexpr,
    side_size: tl.constexpr,
    slot_stride: tl.constexpr,
    chunks,
    block: tl.constexpr,
):
    i = tl.program_id(0) * block + tl.arange(0, block)
    # Real 50-layer pools exceed 2**32 elements; widen BEFORE multiply.
    side = tl.program_id(1).to(tl.int64)
    slot = tl.load(slots_ptr + (i // row_width) * slot_stride, i < count, 0)
    valid_slot = (slot >= 0) & (slot < side_size // row_width)
    x = tl.load(
        storage_ptr + side * side_size + slot * row_width + i % row_width,
        (i < count) & valid_slot,
        0,
    ).to(tl.float32)
    finite = (tl.abs(x) < float("inf")) & ((i >= count) | valid_slot)
    good = tl.sum((~finite).to(tl.int32), 0) == 0
    tl.store(partial_ptr + side * chunks + tl.program_id(0), good)


def new_kv_finite(storage, slots):
    """Check all layers and K/V, restricted to the transaction's exact slots."""
    if (
        storage.device.type != "cuda"
        or not storage.is_contiguous()
        or storage.dtype not in (torch.bfloat16, torch.float16, torch.float32)
    ):
        checks = [
            torch.isfinite(side.flatten(0, 1).index_select(0, slots)).all() for layer in storage for side in layer
        ]
        return torch.stack(checks).all()
    if (
        storage.ndim != 6
        or storage.shape[1] != 2
        or slots.ndim != 1
        or slots.dtype != torch.long
        or slots.device != storage.device
    ):
        raise ValueError("Invalid Realization finite-check storage/slots")
    count = slots.numel() * storage.shape[-2] * storage.shape[-1]
    if not count:
        return torch.ones((), dtype=torch.bool, device=storage.device)
    chunks = triton.cdiv(count, 2048)
    sides = storage.shape[0] * 2
    partial = torch.empty((sides, chunks), dtype=torch.bool, device=storage.device)
    _new_kv_finite[(chunks, sides)](
        storage,
        slots,
        partial,
        count,
        storage.shape[-2] * storage.shape[-1],
        storage[0, 0].numel(),
        slots.stride(0),
        chunks,
        2048,
    )
    return partial.all()


@triton.jit(do_not_specialize=["count"])
def _modulate_kernel(
    value_ptr,
    scale_ptr,
    shift_ptr,
    indices_ptr,
    output_ptr,
    count,
    width: tl.constexpr,
    v0: tl.constexpr,
    v1: tl.constexpr,
    s0: tl.constexpr,
    s1: tl.constexpr,
    t0: tl.constexpr,
    t1: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * block + tl.arange(0, block)
    i = row * width + col
    index = tl.load(indices_ptr + row)
    value = tl.load(value_ptr + row * v0 + col * v1, col < width, 0).to(tl.float32)
    scale = tl.load(scale_ptr + index * s0 + col * s1, col < width, 0).to(tl.float32)
    shift = tl.load(shift_ptr + index * t0 + col * t1, col < width, 0).to(tl.float32)
    scale = (1.0 + scale).to(output_ptr.dtype.element_ty).to(tl.float32)
    product = (value * scale).to(output_ptr.dtype.element_ty).to(tl.float32)
    tl.store(output_ptr + i, product + shift, col < width)


@triton.jit(do_not_specialize=["count"])
def _gated_residual_kernel(
    hidden_ptr,
    gate_ptr,
    value_ptr,
    indices_ptr,
    output_ptr,
    count,
    width: tl.constexpr,
    h0: tl.constexpr,
    h1: tl.constexpr,
    g0: tl.constexpr,
    g1: tl.constexpr,
    v0: tl.constexpr,
    v1: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * block + tl.arange(0, block)
    i = row * width + col
    index = tl.load(indices_ptr + row)
    hidden = tl.load(hidden_ptr + row * h0 + col * h1, col < width, 0).to(tl.float32)
    value = tl.load(value_ptr + row * v0 + col * v1, col < width, 0).to(tl.float32)
    gate = tl.load(gate_ptr + index * g0 + col * g1, col < width, 0).to(tl.float32)
    product = (value * gate).to(output_ptr.dtype.element_ty).to(tl.float32)
    tl.store(output_ptr + i, hidden + product, col < width)


def _pointwise_supported(value, tables, indices):
    return (
        value.device.type == "cuda"
        and value.dtype in (torch.bfloat16, torch.float16)
        and value.ndim == 2
        and indices.ndim == 1
        and indices.is_contiguous()
        and indices.numel() == value.shape[0]
        and indices.dtype == torch.long
        and indices.device == value.device
        and all(
            t.ndim == 2 and t.shape[1] == value.shape[1] and t.device == value.device and t.dtype == value.dtype
            for t in tables
        )
    )


def modulate(value, scale, shift, indices):
    """Preserve eager low-precision rounding after 1+scale and multiply."""
    if not _pointwise_supported(value, (scale, shift), indices):
        return value * (1 + scale[indices]) + shift[indices]
    output = torch.empty_like(value, memory_format=torch.contiguous_format)
    _modulate_kernel[(value.shape[0], triton.cdiv(value.shape[1], 1024))](
        value,
        scale,
        shift,
        indices,
        output,
        value.numel(),
        value.shape[1],
        *value.stride(),
        *scale.stride(),
        *shift.stride(),
        1024,
        enable_fp_fusion=False,
    )
    return output


def gated_residual(hidden, gate, value, indices):
    """Preserve eager gate-product rounding before residual addition."""
    if not _pointwise_supported(hidden, (gate, value), indices) or value.shape != hidden.shape:
        return hidden + gate[indices] * value
    output = torch.empty_like(hidden, memory_format=torch.contiguous_format)
    _gated_residual_kernel[(hidden.shape[0], triton.cdiv(hidden.shape[1], 1024))](
        hidden,
        gate,
        value,
        indices,
        output,
        hidden.numel(),
        hidden.shape[1],
        *hidden.stride(),
        *gate.stride(),
        *value.stride(),
        1024,
        enable_fp_fusion=False,
    )
    return output


@triton.jit(do_not_specialize=["count"])
def _lora_epilogue_kernel(base_ptr, delta_ptr, output_ptr, count, scale: tl.constexpr, block: tl.constexpr):
    i = tl.program_id(0) * block + tl.arange(0, block)
    base = tl.load(base_ptr + i, i < count, 0).to(tl.float32)
    delta = tl.load(delta_ptr + i, i < count, 0).to(tl.float32)
    # Separate FP32 multiply/add, followed by the original output cast.
    scaled = delta * tl.full((), scale, tl.float32)
    tl.store(output_ptr + i, base + scaled, i < count)


def lora_epilogue(base, delta, scale):
    """Keep PEFT's FP32 scale/add, then cast once to the base output dtype."""
    if (
        base.device.type != "cuda"
        or base.device != delta.device
        or base.shape != delta.shape
        or not base.is_contiguous()
        or not delta.is_contiguous()
        or delta.dtype != torch.float32
        or base.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or (torch.is_grad_enabled() and (base.requires_grad or delta.requires_grad))
    ):
        return (base + delta * scale).to(base.dtype)
    result = torch.empty_like(base)
    _lora_epilogue_kernel[(triton.cdiv(base.numel(), 256),)](
        base, delta, result, base.numel(), scale, 256, enable_fp_fusion=False
    )
    return result


@triton.jit
def _split_lora_epilogue_kernel(
    base, q, k, v, output, count: tl.constexpr, width: tl.constexpr, scale: tl.constexpr, block: tl.constexpr
):
    i = tl.program_id(0) * block + tl.arange(0, block)
    col = i % (3 * width)
    j = i // (3 * width) * width + col % width
    dq = tl.load(q + j, (i < count) & (col < width), 0)
    dk = tl.load(k + j, (i < count) & (col >= width) & (col < 2 * width), 0)
    dv = tl.load(v + j, (i < count) & (col >= 2 * width), 0)
    delta = tl.where(col < width, dq, tl.where(col < 2 * width, dk, dv))
    value = tl.load(base + i, i < count, 0).to(tl.float32)
    scaled = delta * tl.full((), scale, tl.float32)
    tl.store(output + i, value + scaled, i < count)


def split_lora_epilogue(base, deltas, scale):
    """Fuse packing Q/K/V with the original FP32 add and final base cast."""
    if (
        len(deltas) != 3
        or base.device.type != "cuda"
        or not base.is_contiguous()
        or base.dtype not in (torch.bfloat16, torch.float16)
        or base.shape[:-1] != deltas[0].shape[:-1]
        or base.shape[-1] != 3 * deltas[0].shape[-1]
        or any(
            x.shape != deltas[0].shape or x.dtype != torch.float32 or x.device != base.device or not x.is_contiguous()
            for x in deltas
        )
        or (torch.is_grad_enabled() and (base.requires_grad or any(x.requires_grad for x in deltas)))
    ):
        return lora_epilogue(base, torch.cat(deltas, -1), scale)
    output = torch.empty_like(base)
    _split_lora_epilogue_kernel[(triton.cdiv(base.numel(), 256),)](
        base, *deltas, output, base.numel(), deltas[0].shape[-1], scale, 256, enable_fp_fusion=False
    )
    return output


@triton.jit(do_not_specialize=["count"])
def _packed_lora_epilogue_kernel(
    base,
    q,
    k,
    v,
    output,
    count,
    qw: tl.constexpr,
    kw: tl.constexpr,
    vw: tl.constexpr,
    scale: tl.constexpr,
    block: tl.constexpr,
):
    i = tl.program_id(0) * block + tl.arange(0, block)
    row, col = i // (qw + kw + vw), i % (qw + kw + vw)
    dq = tl.load(q + row * qw + col, (i < count) & (col < qw), 0)
    dk = tl.load(k + row * kw + col - qw, (i < count) & (col >= qw) & (col < qw + kw), 0)
    dv = tl.load(v + row * vw + col - qw - kw, (i < count) & (col >= qw + kw), 0)
    delta = tl.where(col < qw, dq, tl.where(col < qw + kw, dk, dv))
    value = tl.load(base + i, i < count, 0).to(tl.float32)
    scaled = delta * tl.full((), scale, tl.float32)
    tl.store(output + i, value + scaled, i < count)


def packed_lora_epilogue(base, deltas, scale):
    """Pack native unequal Q/K/V projections with FP32 multiply/add and one cast."""
    q, k, v = deltas
    output = torch.empty_like(base)
    _packed_lora_epilogue_kernel[(triton.cdiv(base.numel(), 256),)](
        base,
        q,
        k,
        v,
        output,
        base.numel(),
        q.shape[-1],
        k.shape[-1],
        v.shape[-1],
        scale,
        256,
        enable_fp_fusion=False,
    )
    return output
