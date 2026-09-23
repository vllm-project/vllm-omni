# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Lossless eager Q/K RoPE fusion for ERNIE-Image."""

from __future__ import annotations

import logging
from collections import OrderedDict

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)

_FAILED_KEYS_MAX_SIZE = 128
_FAILED_KEYS: OrderedDict[tuple[object, ...], None] = OrderedDict()


if HAS_TRITON:

    @triton.jit
    def _round_bf16_to_fp32(value):
        """RNE-round FP32 to BF16 precision in an FP32 register."""

        bits = value.to(tl.int32, bitcast=True)
        rounding_bias = 0x7FFF + ((bits >> 16) & 1)
        rounded_bits = (bits + rounding_bias) & -65536
        return rounded_bits.to(tl.float32, bitcast=True)

    @triton.jit
    def _qk_rope_kernel(
        out_q_ptr,
        out_k_ptr,
        q_ptr,
        k_ptr,
        cos_ptr,
        sin_ptr,
        heads,
        head_dim: tl.constexpr,
        rotary_dim: tl.constexpr,
        rotary_half: tl.constexpr,
        heads_block: tl.constexpr,
        half_block: tl.constexpr,
        tail_block: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        is_key = tl.program_id(1)
        base = row * heads * head_dim
        head_offsets = tl.arange(0, heads_block)[:, None]
        head_mask = head_offsets < heads
        columns = tl.arange(0, half_block)[None, :]
        column_mask = columns < rotary_half
        mask = head_mask & column_mask

        if is_key == 0:
            input_ptr = q_ptr
            output_ptr = out_q_ptr
        else:
            input_ptr = k_ptr
            output_ptr = out_k_ptr

        first_offsets = base + head_offsets * head_dim + columns
        second_offsets = first_offsets + rotary_half
        first = tl.load(input_ptr + first_offsets, mask=mask, other=0.0).to(tl.float32)
        second = tl.load(input_ptr + second_offsets, mask=mask, other=0.0).to(tl.float32)

        # The eager path casts FP32 frequencies to BF16, duplicates adjacent
        # values, rounds both BF16 multiplies, and finally rounds the add.
        first_freq = columns // 2
        second_freq = (rotary_half + columns) // 2
        cos_first = _round_bf16_to_fp32(tl.load(cos_ptr + row * rotary_half + first_freq, mask=column_mask))
        cos_second = _round_bf16_to_fp32(tl.load(cos_ptr + row * rotary_half + second_freq, mask=column_mask))
        sin_first = _round_bf16_to_fp32(tl.load(sin_ptr + row * rotary_half + first_freq, mask=column_mask))
        sin_second = _round_bf16_to_fp32(tl.load(sin_ptr + row * rotary_half + second_freq, mask=column_mask))

        out_first = _round_bf16_to_fp32(first * cos_first) + _round_bf16_to_fp32(-second * sin_first)
        out_second = _round_bf16_to_fp32(second * cos_second) + _round_bf16_to_fp32(first * sin_second)
        tl.store(output_ptr + first_offsets, out_first, mask=mask)
        tl.store(output_ptr + second_offsets, out_second, mask=mask)

        if head_dim > rotary_dim:
            tail_columns = rotary_dim + tl.arange(0, tail_block)[None, :]
            tail_mask = head_mask & (tail_columns < head_dim)
            tail_offsets = base + head_offsets * head_dim + tail_columns
            tail = tl.load(input_ptr + tail_offsets, mask=tail_mask, other=0.0)
            tl.store(output_ptr + tail_offsets, tail, mask=tail_mask)


def _runtime_key(
    query: torch.Tensor,
    key: torch.Tensor,
    freqs_cos: torch.Tensor,
) -> tuple[object, ...]:
    return (
        query.device.index,
        tuple(query.shape),
        tuple(key.shape),
        tuple(freqs_cos.shape),
        query.dtype,
        freqs_cos.dtype,
    )


def _is_failed_runtime_key(runtime_key: tuple[object, ...]) -> bool:
    if runtime_key not in _FAILED_KEYS:
        return False
    _FAILED_KEYS.move_to_end(runtime_key)
    return True


def _record_failed_runtime_key(runtime_key: tuple[object, ...]) -> None:
    _FAILED_KEYS[runtime_key] = None
    _FAILED_KEYS.move_to_end(runtime_key)
    while len(_FAILED_KEYS) > _FAILED_KEYS_MAX_SIZE:
        _FAILED_KEYS.popitem(last=False)


def _supported_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> bool:
    if torch.compiler.is_compiling():
        return False
    if any(tensor.requires_grad for tensor in (query, key, freqs_cos, freqs_sin)):
        return False
    if not HAS_TRITON or not current_omni_platform.is_cuda():
        return False
    if query.dtype is not torch.bfloat16 or not query.is_cuda:
        return False
    if query.ndim != 4 or key.shape != query.shape or key.dtype is not query.dtype or key.device != query.device:
        return False
    if not query.is_contiguous() or not key.is_contiguous():
        return False
    if freqs_cos.dtype is not torch.float32 or freqs_sin.dtype is not torch.float32:
        return False
    if not freqs_cos.is_cuda or freqs_cos.device != query.device:
        return False
    if freqs_sin.device != query.device or freqs_sin.shape != freqs_cos.shape:
        return False
    if not freqs_cos.is_contiguous() or not freqs_sin.is_contiguous():
        return False
    batch, sequence, _, head_dim = query.shape
    rotary_half = freqs_cos.shape[-1] if freqs_cos.ndim == 3 else 0
    return (
        query.numel() > 0
        and freqs_cos.shape == (batch, sequence, rotary_half)
        and rotary_half > 0
        and rotary_half * 2 <= head_dim
    )


def _launch_fused_qk_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, sequence, heads, head_dim = query.shape
    rotary_half = freqs_cos.shape[-1]
    rotary_dim = rotary_half * 2
    tail = head_dim - rotary_dim
    output_query = torch.empty_like(query)
    output_key = torch.empty_like(key)
    with torch.accelerator.device_index(query.device.index):
        _qk_rope_kernel[(batch * sequence, 2)](
            output_query,
            output_key,
            query,
            key,
            freqs_cos,
            freqs_sin,
            heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            rotary_half=rotary_half,
            heads_block=triton.next_power_of_2(heads),
            half_block=triton.next_power_of_2(rotary_half),
            tail_block=triton.next_power_of_2(max(tail, 1)),
        )
    return output_query, output_key


def try_fused_qk_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return fused Q/K outputs, or ``None`` to request the eager fallback."""

    if not _supported_inputs(query, key, freqs_cos, freqs_sin):
        return None

    runtime_key = _runtime_key(query, key, freqs_cos)
    if _is_failed_runtime_key(runtime_key):
        return None

    try:
        return _launch_fused_qk_rotary_emb(query, key, freqs_cos, freqs_sin)
    except Exception:
        _record_failed_runtime_key(runtime_key)
        logger.exception("ERNIE-Image fused Q/K RoPE failed; using the eager fallback")
        return None


__all__ = ["try_fused_qk_rotary_emb"]
