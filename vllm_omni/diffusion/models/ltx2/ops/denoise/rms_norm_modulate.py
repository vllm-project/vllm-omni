# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exact Triton BF16 RMSNorm and LTX-2 AdaLN modulation."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from vllm.triton_utils import tl, triton

from ..numerics import rms_reciprocal_fma, round_bf16_to_fp32
from ..platform import is_ltx2_ops_eligible

_ALIGNMENT = 8
_OUTPUT_BLOCK = 1024
_FAILED_RUNTIME_KEYS: set[tuple[int | None, int, bool, bool]] = set()
_VERIFIED_RUNTIME_KEYS: set[tuple[int | None, int, bool, bool]] = set()

logger = logging.getLogger(__name__)


@triton.jit
def _rms_norm_modulate_kernel(
    output_a_ptr,
    output_b_ptr,
    input_ptr,
    scale_a_ptr,
    shift_a_ptr,
    scale_b_ptr,
    shift_b_ptr,
    scale_a_table_ptr,
    shift_a_table_ptr,
    scale_b_table_ptr,
    shift_b_table_ptr,
    rows_per_batch,
    scale_a_batch_stride,
    scale_a_row_stride,
    shift_a_batch_stride,
    shift_a_row_stride,
    scale_b_batch_stride,
    scale_b_row_stride,
    shift_b_batch_stride,
    shift_b_row_stride,
    eps,
    hidden_size: tl.constexpr,
    dual: tl.constexpr,
    has_tables: tl.constexpr,
    output_block: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    batch = row // rows_per_batch
    token = row % rows_per_batch
    row_base = row * hidden_size
    reciprocal_rms = rms_reciprocal_fma(
        input_ptr,
        row_base,
        eps,
        hidden_size,
    )

    scale_a_base = batch * scale_a_batch_stride + token * scale_a_row_stride
    shift_a_base = batch * shift_a_batch_stride + token * shift_a_row_stride
    scale_b_base = batch * scale_b_batch_stride + token * scale_b_row_stride
    shift_b_base = batch * shift_b_batch_stride + token * shift_b_row_stride
    for block in tl.static_range(hidden_size // output_block):
        columns = block * output_block + tl.arange(0, output_block)
        value = tl.load(input_ptr + row_base + columns).to(tl.float32)
        normalized = round_bf16_to_fp32(value * reciprocal_rms)

        scale_a = tl.load(scale_a_ptr + scale_a_base + columns).to(tl.float32)
        shift_a = tl.load(shift_a_ptr + shift_a_base + columns).to(tl.float32)
        if has_tables:
            scale_a_table = tl.load(scale_a_table_ptr + columns).to(tl.float32)
            shift_a_table = tl.load(shift_a_table_ptr + columns).to(tl.float32)
            scale_a = round_bf16_to_fp32(scale_a_table + scale_a)
            shift_a = round_bf16_to_fp32(shift_a_table + shift_a)
        one_plus_scale_a = round_bf16_to_fp32(1.0 + scale_a)
        product_a = round_bf16_to_fp32(normalized * one_plus_scale_a)
        tl.store(output_a_ptr + row_base + columns, product_a + shift_a)

        if dual:
            scale_b = tl.load(scale_b_ptr + scale_b_base + columns).to(tl.float32)
            shift_b = tl.load(shift_b_ptr + shift_b_base + columns).to(tl.float32)
            if has_tables:
                scale_b_table = tl.load(scale_b_table_ptr + columns).to(tl.float32)
                shift_b_table = tl.load(shift_b_table_ptr + columns).to(tl.float32)
                scale_b = round_bf16_to_fp32(scale_b_table + scale_b)
                shift_b = round_bf16_to_fp32(shift_b_table + shift_b)
            one_plus_scale_b = round_bf16_to_fp32(1.0 + scale_b)
            product_b = round_bf16_to_fp32(normalized * one_plus_scale_b)
            tl.store(output_b_ptr + row_base + columns, product_b + shift_b)


def _modulation_view_matches(tensor: torch.Tensor, x: torch.Tensor) -> bool:
    return (
        tensor.dtype is torch.bfloat16
        and tensor.is_cuda
        and tensor.device == x.device
        and tensor.dim() == 3
        and tensor.shape[0] == x.shape[0]
        and tensor.shape[1] in (1, x.shape[1])
        and tensor.shape[2] == x.shape[2]
        and tensor.stride(-1) == 1
        and tensor.stride(0) % 4 == 0
        and tensor.stride(1) % 4 == 0
        and tensor.data_ptr() % _ALIGNMENT == 0
    )


def _table_matches(table: torch.Tensor | None, x: torch.Tensor) -> bool:
    return table is None or (
        table.dtype is torch.bfloat16
        and table.is_cuda
        and table.device == x.device
        and table.shape == (x.shape[-1],)
        and table.is_contiguous()
        and table.data_ptr() % _ALIGNMENT == 0
    )


def _common_matches(x: torch.Tensor, values: tuple[torch.Tensor, ...]) -> bool:
    return (
        is_ltx2_ops_eligible(x)
        and x.dtype is torch.bfloat16
        and x.dim() == 3
        and x.is_contiguous()
        and x.numel() > 0
        and x.shape[-1] in (2048, 4096)
        and x.shape[-1] % 4 == 0
        and x.data_ptr() % _ALIGNMENT == 0
        and all(_modulation_view_matches(value, x) for value in values)
    )


def can_use_triton_rms_norm_modulate(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    scale_table: torch.Tensor | None = None,
    shift_table: torch.Tensor | None = None,
) -> bool:
    del eps
    return (
        _common_matches(x, (scale, shift))
        and _table_matches(scale_table, x)
        and _table_matches(shift_table, x)
        and (scale_table is None) == (shift_table is None)
    )


def can_use_triton_rms_norm_dual_modulate(
    x: torch.Tensor,
    scale_a: torch.Tensor,
    shift_a: torch.Tensor,
    scale_b: torch.Tensor,
    shift_b: torch.Tensor,
    eps: float,
    scale_a_table: torch.Tensor | None = None,
    shift_a_table: torch.Tensor | None = None,
    scale_b_table: torch.Tensor | None = None,
    shift_b_table: torch.Tensor | None = None,
) -> bool:
    del eps
    tables = (scale_a_table, shift_a_table, scale_b_table, shift_b_table)
    return (
        _common_matches(x, (scale_a, shift_a, scale_b, shift_b))
        and all(_table_matches(table, x) for table in tables)
        and (all(table is None for table in tables) or all(table is not None for table in tables))
    )


def _table_or_value(table: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    return value if table is None else table


def _launch(
    x: torch.Tensor,
    scale_a: torch.Tensor,
    shift_a: torch.Tensor,
    scale_b: torch.Tensor,
    shift_b: torch.Tensor,
    eps: float,
    scale_a_table: torch.Tensor | None,
    shift_a_table: torch.Tensor | None,
    scale_b_table: torch.Tensor | None,
    shift_b_table: torch.Tensor | None,
    *,
    dual: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_a = torch.empty_like(x)
    output_b = torch.empty_like(x) if dual else output_a
    with torch.accelerator.device_index(x.device.index):
        _rms_norm_modulate_kernel[(x.shape[0] * x.shape[1],)](
            output_a,
            output_b,
            x,
            scale_a,
            shift_a,
            scale_b,
            shift_b,
            _table_or_value(scale_a_table, scale_a),
            _table_or_value(shift_a_table, shift_a),
            _table_or_value(scale_b_table, scale_b),
            _table_or_value(shift_b_table, shift_b),
            x.shape[1],
            scale_a.stride(0),
            0 if scale_a.shape[1] == 1 else scale_a.stride(1),
            shift_a.stride(0),
            0 if shift_a.shape[1] == 1 else shift_a.stride(1),
            scale_b.stride(0),
            0 if scale_b.shape[1] == 1 else scale_b.stride(1),
            shift_b.stride(0),
            0 if shift_b.shape[1] == 1 else shift_b.stride(1),
            eps,
            hidden_size=x.shape[2],
            dual=dual,
            has_tables=scale_a_table is not None,
            output_block=_OUTPUT_BLOCK,
            num_warps=4,
        )
    return output_a, output_b


def _reference(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    scale_table: torch.Tensor | None,
    shift_table: torch.Tensor | None,
) -> torch.Tensor:
    if scale_table is not None and shift_table is not None:
        scale = scale_table + scale
        shift = shift_table + shift
    normalized = F.rms_norm(x, (x.shape[-1],), eps=eps)
    return normalized * (1 + scale) + shift


def _disable_after_failure(
    runtime_key: tuple[int | None, int, bool, bool],
    x: torch.Tensor,
    exc: Exception | None = None,
) -> None:
    _FAILED_RUNTIME_KEYS.add(runtime_key)
    reason = f"JIT/runtime failure: {exc}" if exc is not None else "bit-exactness mismatch"
    logger.warning(
        "Disabling LTX-2 Triton RMSNorm modulation on %s after %s",
        x.device,
        reason,
    )


def try_rms_norm_modulate_exact(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    scale_table: torch.Tensor | None = None,
    shift_table: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Run the verified-CUDA Triton fusion, or expose the reference path."""

    if not can_use_triton_rms_norm_modulate(x, scale, shift, eps, scale_table, shift_table):
        return None
    runtime_key = (x.device.index, x.shape[-1], False, scale_table is not None)
    if runtime_key in _FAILED_RUNTIME_KEYS:
        return None
    try:
        output, _ = _launch(
            x,
            scale,
            shift,
            scale,
            shift,
            eps,
            scale_table,
            shift_table,
            scale_table,
            shift_table,
            dual=False,
        )
    except Exception as exc:  # noqa: BLE001 - fail closed after Triton failure
        _disable_after_failure(runtime_key, x, exc)
        return None
    if runtime_key not in _VERIFIED_RUNTIME_KEYS:
        reference = _reference(x, scale, shift, eps, scale_table, shift_table)
        if not torch.equal(output, reference):
            _disable_after_failure(runtime_key, x)
            return None
        _VERIFIED_RUNTIME_KEYS.add(runtime_key)
    return output


def try_rms_norm_dual_modulate_exact(
    x: torch.Tensor,
    scale_a: torch.Tensor,
    shift_a: torch.Tensor,
    scale_b: torch.Tensor,
    shift_b: torch.Tensor,
    eps: float,
    scale_a_table: torch.Tensor | None = None,
    shift_a_table: torch.Tensor | None = None,
    scale_b_table: torch.Tensor | None = None,
    shift_b_table: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run the verified-CUDA dual-output Triton fusion when supported."""

    if not can_use_triton_rms_norm_dual_modulate(
        x,
        scale_a,
        shift_a,
        scale_b,
        shift_b,
        eps,
        scale_a_table,
        shift_a_table,
        scale_b_table,
        shift_b_table,
    ):
        return None
    runtime_key = (x.device.index, x.shape[-1], True, scale_a_table is not None)
    if runtime_key in _FAILED_RUNTIME_KEYS:
        return None
    try:
        outputs = _launch(
            x,
            scale_a,
            shift_a,
            scale_b,
            shift_b,
            eps,
            scale_a_table,
            shift_a_table,
            scale_b_table,
            shift_b_table,
            dual=True,
        )
    except Exception as exc:  # noqa: BLE001 - fail closed after Triton failure
        _disable_after_failure(runtime_key, x, exc)
        return None
    if runtime_key not in _VERIFIED_RUNTIME_KEYS:
        references = (
            _reference(x, scale_a, shift_a, eps, scale_a_table, shift_a_table),
            _reference(x, scale_b, shift_b, eps, scale_b_table, shift_b_table),
        )
        if not all(torch.equal(output, reference) for output, reference in zip(outputs, references, strict=True)):
            _disable_after_failure(runtime_key, x)
            return None
        _VERIFIED_RUNTIME_KEYS.add(runtime_key)
    return outputs


__all__ = [
    "try_rms_norm_dual_modulate_exact",
    "try_rms_norm_modulate_exact",
]
