# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Aligned block-int8 transport for the MiniMax-H3 Ulysses QKV exchange.

The UE5M3 packet format - 128-value vectors, group-32 finite-positive scale codes, and the
144-byte output record - follows Sol-H3's ``h3_runtime/comm_quant.py``, which ships with
NVlabs/Sana's sol-engine (Apache-2.0) alongside the SOL_ATTN backend this lane's attention path
uses. The format is kept byte-compatible so a packet produced here decodes against that reference.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

from .quant_kernels import _encode_ue5m3_int8

VECTOR = 128
# One output record carries one 128-wide attention head, four group-32
# finite-positive UE5M3 scale codes, and twelve zero padding bytes, i.e. 144
# bytes per 128 values (0.5625x the bf16 payload).
OUTPUT_PACKET = 144
VALUE_BIAS = 127
OUTPUT_RECORDS_PER_PROGRAM = 16


_MXFP8_UNIT_SCALES: dict[tuple[int, int, int], torch.Tensor] = {}


@triton.jit
def _encode_output_kernel(
    input_ptr, packet_ptr, records, RECORDS: tl.constexpr, PACKET: tl.constexpr, VALUE_OFFSET: tl.constexpr
):
    record = tl.program_id(0) * RECORDS + tl.arange(0, RECORDS)
    columns = tl.arange(0, 128)
    valid = record[:, None] < records
    values = tl.load(input_ptr + record[:, None] * 128 + columns[None, :], mask=valid, other=0.0)
    stored, scale_codes = _encode_ue5m3_int8(values, GROUPS=RECORDS * 4)
    stored = tl.reshape(stored, RECORDS, 128)
    scale_codes = tl.reshape(scale_codes, RECORDS, 4)
    base = record[:, None] * PACKET
    tl.store(packet_ptr + base + columns[None, :], stored, mask=valid)
    groups = tl.arange(0, 4)
    tl.store(
        packet_ptr + base + 128 + groups[None, :],
        scale_codes,
        mask=record[:, None] < records,
    )
    # Initialize padding so collectives never carry allocator contents.
    padding = tl.arange(0, 16)
    tl.store(
        packet_ptr + base + 132 + padding[None, :],
        0,
        mask=(record[:, None] < records) & (padding[None, :] < PACKET - 132),
    )


@triton.jit
def _decode_merge_output_kernel(
    packet_ptr,
    output_ptr,
    rows,
    heads_local,
    world,
    output_records,
    RECORDS: tl.constexpr,
    PACKET: tl.constexpr,
    VALUE_OFFSET: tl.constexpr,
):
    """Decode rank-major packets directly into row-major global-head order."""
    output_record = tl.program_id(0) * RECORDS + tl.arange(0, RECORDS)
    valid = output_record[:, None] < output_records
    global_head = output_record % (heads_local * world)
    row = output_record // (heads_local * world)
    source = global_head // heads_local
    local_head = global_head - source * heads_local
    source_record = (source * rows + row) * heads_local + local_head

    columns = tl.arange(0, 128)
    base = source_record[:, None] * PACKET
    stored = tl.load(packet_ptr + base + columns[None, :], mask=valid, other=VALUE_OFFSET).to(tl.int32)
    group = columns // 32
    code = tl.load(packet_ptr + base + 128 + group[None, :], mask=valid, other=0).to(tl.int32)
    scale_bits = (((code >> 3) - 15 + 127) << 23) | ((code & 7) << 20)
    scale = scale_bits.to(tl.float32, bitcast=True)
    decoded = (stored - VALUE_OFFSET).to(tl.float32) * scale
    tl.store(output_ptr + output_record[:, None] * 128 + columns[None, :], decoded, mask=valid)


def quantize_output(x: torch.Tensor) -> torch.Tensor:
    """Encode contiguous ``[..., 128]`` BF16 attention output for transport."""
    if x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("output transport requires a contiguous CUDA BF16 tensor")
    if x.ndim != 3 or x.shape[-1] != VECTOR:
        raise ValueError(f"expected [tokens, heads, {VECTOR}], got {tuple(x.shape)}")
    records = x.numel() // VECTOR
    packet = torch.empty((*x.shape[:-1], OUTPUT_PACKET), dtype=torch.uint8, device=x.device)
    if records:
        _encode_output_kernel[(triton.cdiv(records, OUTPUT_RECORDS_PER_PROGRAM),)](
            x,
            packet,
            records,
            RECORDS=OUTPUT_RECORDS_PER_PROGRAM,
            PACKET=OUTPUT_PACKET,
            VALUE_OFFSET=VALUE_BIAS,
            num_warps=8,
        )
    return packet


def dequantize_merge_output(packet: torch.Tensor, world: int) -> torch.Tensor:
    """Decode packed rank-major output records and merge global heads."""
    if packet.dtype != torch.uint8 or not packet.is_cuda or not packet.is_contiguous():
        raise ValueError("output transport requires contiguous CUDA uint8 packets")
    if packet.ndim != 4 or packet.shape[0] != world or packet.shape[-1] != OUTPUT_PACKET:
        raise ValueError(f"expected [{world}, rows, heads_local, {OUTPUT_PACKET}], got {tuple(packet.shape)}")
    _, rows, heads_local, _ = packet.shape
    output = torch.empty((rows, world * heads_local, VECTOR), dtype=torch.bfloat16, device=packet.device)
    records = rows * world * heads_local
    if records:
        _decode_merge_output_kernel[(triton.cdiv(records, OUTPUT_RECORDS_PER_PROGRAM),)](
            packet,
            output,
            rows,
            heads_local,
            world,
            records,
            RECORDS=OUTPUT_RECORDS_PER_PROGRAM,
            PACKET=OUTPUT_PACKET,
            VALUE_OFFSET=VALUE_BIAS,
            num_warps=8,
        )
    return output


@triton.jit
def _encode_output_fp8_kernel(input_ptr, packet_ptr, elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < elements
    values = tl.load(input_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    # Direct E4M3FN conversion maps overflow to NaN. Saturation keeps every
    # transmitted attention value finite without carrying a separate scale.
    values = tl.maximum(-448.0, tl.minimum(448.0, values))
    encoded = values.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(packet_ptr + offsets, encoded, mask=valid)


@triton.jit
def _merge_output_fp8_kernel(
    packet_ptr,
    output_ptr,
    elements,
    world,
    rows,
    inner,
    BLOCK: tl.constexpr,
):
    """Merge rank-major E4M3 bytes into a row-major BF16 or E4M3 output."""
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < elements
    tail = offsets % inner
    slot = offsets // inner
    source = slot % world
    row = slot // world
    source_offset = (source * rows + row) * inner + tail
    encoded = tl.load(packet_ptr + source_offset, mask=valid, other=0)
    decoded = encoded.to(tl.float8e4nv, bitcast=True)
    tl.store(output_ptr + offsets, decoded, mask=valid)


def quantize_output_fp8(x: torch.Tensor) -> torch.Tensor:
    """Encode a contiguous BF16 attention output as saturated raw E4M3 bytes."""
    if x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("FP8 output transport requires a contiguous CUDA BF16 tensor")
    if x.ndim != 3 or x.shape[-1] != VECTOR:
        raise ValueError(f"expected [tokens, heads, {VECTOR}], got {tuple(x.shape)}")
    packet = torch.empty_like(x, dtype=torch.uint8)
    if x.numel():
        block = 1024
        _encode_output_fp8_kernel[(triton.cdiv(x.numel(), block),)](x, packet, x.numel(), BLOCK=block, num_warps=8)
    return packet


def dequantize_merge_output_fp8(packet: torch.Tensor, world: int) -> torch.Tensor:
    """Decode raw E4M3 bytes while merging rank-major heads."""
    if packet.dtype != torch.uint8 or not packet.is_cuda or not packet.is_contiguous():
        raise ValueError("FP8 output transport requires contiguous CUDA uint8 packets")
    if packet.ndim != 4 or packet.shape[0] != world or packet.shape[-1] != VECTOR:
        raise ValueError(f"expected [{world}, rows, heads_local, {VECTOR}], got {tuple(packet.shape)}")
    _, rows, heads_local, _ = packet.shape
    output = torch.empty((rows, world * heads_local, VECTOR), dtype=torch.bfloat16, device=packet.device)
    if output.numel():
        block = 1024
        _merge_output_fp8_kernel[(triton.cdiv(output.numel(), block),)](
            packet,
            output,
            output.numel(),
            world,
            rows,
            heads_local * VECTOR,
            BLOCK=block,
            num_warps=8,
        )
    return output


def merge_output_fp8_as_mxfp8(
    packet: torch.Tensor,
    world: int,
    shape: tuple[int, ...],
):
    """Reuse raw FP8 transport bytes as an MXFP8 linear activation.

    The return collective already carries E4M3 values.  With an E8M0 scale of
    one they are also a valid block-scaled MXFP8 activation, so the only work
    left is the rank-major to row-major head merge.  This removes both the
    intermediate BF16 tensor and the following dynamic MXFP8 quantization.
    """
    if packet.dtype != torch.uint8 or not packet.is_cuda or not packet.is_contiguous():
        raise ValueError("FP8 output transport requires contiguous CUDA uint8 packets")
    if packet.ndim != 4 or packet.shape[0] != world or packet.shape[-1] != VECTOR:
        raise ValueError(f"expected [{world}, rows, heads_local, {VECTOR}], got {tuple(packet.shape)}")
    _, rows, heads_local, _ = packet.shape
    k = world * heads_local * VECTOR
    leading_rows = 1
    for dimension in shape[:-1]:
        leading_rows *= dimension
    if not shape or shape[-1] != k or leading_rows != rows:
        raise ValueError(f"MXFP8 output shape {shape} does not match merged packet shape ({rows}, {k})")

    from .mxfp8 import MXActivation

    quantized = torch.empty((rows, k), dtype=torch.float8_e4m3fn, device=packet.device)
    if quantized.numel():
        block = 1024
        # The same merge kernel stores either BF16 or E4M3 according to the
        # output pointer type. An E4M3 destination preserves the packet bytes.
        _merge_output_fp8_kernel[(triton.cdiv(quantized.numel(), block),)](
            packet,
            quantized,
            quantized.numel(),
            world,
            rows,
            heads_local * VECTOR,
            BLOCK=block,
            num_warps=8,
        )

    num_groups = k // 32
    scale_numel = triton.cdiv(rows, 128) * 128 * triton.cdiv(num_groups, 4) * 4
    device_index = packet.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    scale_key = (device_index, rows, k)
    scale = _MXFP8_UNIT_SCALES.get(scale_key)
    if scale is None:
        # E8M0 byte 127 is 2**0. The buffer is immutable and shape-stable for a
        # resident request, so allocate/fill it only once instead of per block.
        scale = torch.full((scale_numel,), 127, dtype=torch.uint8, device=packet.device)
        _MXFP8_UNIT_SCALES[scale_key] = scale
    return MXActivation(quantized, scale, shape, torch.bfloat16)
