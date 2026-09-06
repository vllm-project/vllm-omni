# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Bit-exact multi-row Q/K RMSNorm for the LTX diffusion VAE decoder."""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F
from vllm.triton_utils import tl, triton

from ..numerics import (
    add_rn_f32,
    fma_rn_f32,
    mul_rn_f32,
    round_bf16_to_fp32,
    rsqrt_approx_f32,
    shfl_down_f32,
)
from ..platform import is_ltx2_ops_eligible
from .rope_tables import get_rope_tables

_HEAD_DIM = 64
_ROWS_PER_PROGRAM = 2
_VERIFY_ROWS = 256
_FAILED_ROPE_KEYS: set[tuple[int | None, float, float, tuple[int, int, int], float]] = set()
_VERIFIED_ROPE_KEYS: set[tuple[int | None, float, float, tuple[int, int, int], float]] = set()

logger = logging.getLogger(__name__)


@triton.jit
def _shuffle_lane_zero(value):
    return tl.inline_asm_elementwise(
        asm="shfl.sync.idx.b32 $0, $1, 0, 0x1f, 0xffffffff;",
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _warp_sum(value):
    value = add_rn_f32(value, shfl_down_f32(value, 16))
    value = add_rn_f32(value, shfl_down_f32(value, 8))
    value = add_rn_f32(value, shfl_down_f32(value, 4))
    value = add_rn_f32(value, shfl_down_f32(value, 2))
    value = add_rn_f32(value, shfl_down_f32(value, 1))
    return _shuffle_lane_zero(value)


@triton.jit
def _qk_rms_norm_scale_rope_3d_kernel(
    query_output_ptr,
    key_output_ptr,
    query_ptr,
    key_ptr,
    query_weight_ptr,
    key_weight_ptr,
    cos_t_ptr,
    sin_t_ptr,
    cos_h_ptr,
    sin_h_ptr,
    cos_w_ptr,
    sin_w_ptr,
    rows,
    tokens_per_batch,
    height,
    width,
    heads,
    eps,
    query_scale,
    rows_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    pairs_t: tl.constexpr,
    pairs_h: tl.constexpr,
):
    threads = tl.arange(0, rows_per_program * 32)
    local_row = threads // 32
    lane = threads % 32
    row = tl.program_id(0).to(tl.int64) * rows_per_program + local_row
    valid_row = row < rows
    reduction_valid = valid_row & (lane < 16)
    reduction_base = row * head_dim + lane * 4

    query_accumulator = tl.zeros((rows_per_program * 32,), dtype=tl.float32)
    key_accumulator = tl.zeros((rows_per_program * 32,), dtype=tl.float32)
    for element in tl.static_range(4):
        query = tl.load(query_ptr + reduction_base + element, mask=reduction_valid, other=0.0).to(tl.float32)
        key = tl.load(key_ptr + reduction_base + element, mask=reduction_valid, other=0.0).to(tl.float32)
        query_accumulator = fma_rn_f32(query, query, query_accumulator)
        key_accumulator = fma_rn_f32(key, key, key_accumulator)
    query_rstd = rsqrt_approx_f32(_warp_sum(query_accumulator) / head_dim + eps)
    key_rstd = rsqrt_approx_f32(_warp_sum(key_accumulator) / head_dim + eps)

    tokens = (row // heads) % tokens_per_batch
    pos_w = tokens % width
    pos_h = (tokens // width) % height
    pos_t = tokens // (height * width)
    pairs_per_head: tl.constexpr = head_dim // 2
    pairs_w: tl.constexpr = pairs_per_head - pairs_t - pairs_h
    is_t = lane < pairs_t
    is_h = (lane >= pairs_t) & (lane < pairs_t + pairs_h)
    is_w = ~(is_t | is_h)
    pair_t = lane
    pair_h = lane - pairs_t
    pair_w = lane - pairs_t - pairs_h
    cos_t = tl.load(cos_t_ptr + pos_t * pairs_t + pair_t, mask=valid_row & is_t, other=0.0)
    sin_t = tl.load(sin_t_ptr + pos_t * pairs_t + pair_t, mask=valid_row & is_t, other=0.0)
    cos_h = tl.load(cos_h_ptr + pos_h * pairs_h + pair_h, mask=valid_row & is_h, other=0.0)
    sin_h = tl.load(sin_h_ptr + pos_h * pairs_h + pair_h, mask=valid_row & is_h, other=0.0)
    cos_w = tl.load(cos_w_ptr + pos_w * pairs_w + pair_w, mask=valid_row & is_w, other=0.0)
    sin_w = tl.load(sin_w_ptr + pos_w * pairs_w + pair_w, mask=valid_row & is_w, other=0.0)
    cos = cos_t + cos_h + cos_w
    sin = sin_t + sin_h + sin_w

    even_column = lane * 2
    odd_column = even_column + 1
    even_offset = row * head_dim + even_column
    odd_offset = row * head_dim + odd_column
    query_even = tl.load(query_ptr + even_offset, mask=valid_row).to(tl.float32)
    query_odd = tl.load(query_ptr + odd_offset, mask=valid_row).to(tl.float32)
    key_even = tl.load(key_ptr + even_offset, mask=valid_row).to(tl.float32)
    key_odd = tl.load(key_ptr + odd_offset, mask=valid_row).to(tl.float32)
    query_even_weight = tl.load(query_weight_ptr + even_column).to(tl.float32)
    query_odd_weight = tl.load(query_weight_ptr + odd_column).to(tl.float32)
    key_even_weight = tl.load(key_weight_ptr + even_column).to(tl.float32)
    key_odd_weight = tl.load(key_weight_ptr + odd_column).to(tl.float32)

    query_even = round_bf16_to_fp32(mul_rn_f32(query_even_weight, mul_rn_f32(query_rstd, query_even)))
    query_odd = round_bf16_to_fp32(mul_rn_f32(query_odd_weight, mul_rn_f32(query_rstd, query_odd)))
    query_even = round_bf16_to_fp32(mul_rn_f32(query_even, query_scale))
    query_odd = round_bf16_to_fp32(mul_rn_f32(query_odd, query_scale))
    key_even = round_bf16_to_fp32(mul_rn_f32(key_even_weight, mul_rn_f32(key_rstd, key_even)))
    key_odd = round_bf16_to_fp32(mul_rn_f32(key_odd_weight, mul_rn_f32(key_rstd, key_odd)))

    tl.store(
        query_output_ptr + even_offset,
        add_rn_f32(mul_rn_f32(query_even, cos), -mul_rn_f32(query_odd, sin)),
        mask=valid_row,
    )
    tl.store(
        query_output_ptr + odd_offset,
        add_rn_f32(mul_rn_f32(query_even, sin), mul_rn_f32(query_odd, cos)),
        mask=valid_row,
    )
    tl.store(
        key_output_ptr + even_offset,
        add_rn_f32(mul_rn_f32(key_even, cos), -mul_rn_f32(key_odd, sin)),
        mask=valid_row,
    )
    tl.store(
        key_output_ptr + odd_offset,
        add_rn_f32(mul_rn_f32(key_even, sin), mul_rn_f32(key_odd, cos)),
        mask=valid_row,
    )


def _weight_matches(value: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        weight.is_cuda
        and weight.device == value.device
        and weight.dtype is torch.bfloat16
        and weight.shape == (_HEAD_DIM,)
        and weight.is_contiguous()
    )


def _supported_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    eps: float,
    query_scale: float,
) -> bool:
    return (
        is_ltx2_ops_eligible(query)
        and query.dtype is torch.bfloat16
        and query.ndim == 6
        and query.shape[-1] == _HEAD_DIM
        and query.numel() > 0
        and query.is_contiguous()
        and key.is_cuda
        and key.device == query.device
        and key.dtype is query.dtype
        and key.shape == query.shape
        and key.is_contiguous()
        and _weight_matches(query, query_weight)
        and _weight_matches(query, key_weight)
        and math.isfinite(eps)
        and eps > 0
        and math.isfinite(query_scale)
    )


def _combined_supported_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    eps: float,
    query_scale: float,
    dim_split: tuple[int, int, int],
    base: float,
) -> bool:
    return (
        _supported_inputs(query, key, query_weight, key_weight, eps, query_scale)
        and len(dim_split) == 3
        and sum(dim_split) == _HEAD_DIM
        and all(dim > 0 and dim % 2 == 0 for dim in dim_split)
        and math.isfinite(base)
        and base > 0
    )


def _launch_combined(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    eps: float,
    query_scale: float,
    dim_split: tuple[int, int, int],
    base: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_output = torch.empty_like(query)
    key_output = torch.empty_like(key)
    rows = query.numel() // _HEAD_DIM
    tables = get_rope_tables(query, dim_split, base)
    cos_t, sin_t = tables[0]
    cos_h, sin_h = tables[1]
    cos_w, sin_w = tables[2]
    with torch.accelerator.device_index(query.device.index):
        _qk_rms_norm_scale_rope_3d_kernel[(triton.cdiv(rows, _ROWS_PER_PROGRAM),)](
            query_output,
            key_output,
            query,
            key,
            query_weight,
            key_weight,
            cos_t,
            sin_t,
            cos_h,
            sin_h,
            cos_w,
            sin_w,
            rows,
            query.shape[1] * query.shape[2] * query.shape[3],
            query.shape[2],
            query.shape[3],
            query.shape[4],
            eps,
            query_scale,
            rows_per_program=_ROWS_PER_PROGRAM,
            head_dim=_HEAD_DIM,
            pairs_t=dim_split[0] // 2,
            pairs_h=dim_split[1] // 2,
            num_warps=_ROWS_PER_PROGRAM,
        )
    return query_output, key_output


def _verify_combined_prefix(
    output: tuple[torch.Tensor, torch.Tensor],
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    eps: float,
    query_scale: float,
    dim_split: tuple[int, int, int],
    base: float,
) -> bool:
    rows = min(query.numel() // _HEAD_DIM, _VERIFY_ROWS)
    query_rows = query.reshape(-1, _HEAD_DIM)[:rows]
    key_rows = key.reshape(-1, _HEAD_DIM)[:rows]
    query_rows = F.rms_norm(query_rows, (_HEAD_DIM,), query_weight, eps) * query_scale
    key_rows = F.rms_norm(key_rows, (_HEAD_DIM,), key_weight, eps)
    tables = get_rope_tables(query, dim_split, base)
    row_indices = torch.arange(rows, device=query.device)
    tokens_per_batch = query.shape[1] * query.shape[2] * query.shape[3]
    tokens = (row_indices // query.shape[4]) % tokens_per_batch
    positions = (
        tokens // (query.shape[2] * query.shape[3]),
        (tokens // query.shape[3]) % query.shape[2],
        tokens % query.shape[3],
    )
    cos = torch.cat([axis_tables[0][position] for axis_tables, position in zip(tables, positions, strict=True)], -1)
    sin = torch.cat([axis_tables[1][position] for axis_tables, position in zip(tables, positions, strict=True)], -1)

    def rotate(value: torch.Tensor) -> torch.Tensor:
        pairs = value.reshape(rows, _HEAD_DIM // 2, 2)
        even = pairs[..., 0].float()
        odd = pairs[..., 1].float()
        return (
            torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).reshape(rows, _HEAD_DIM).to(value.dtype)
        )

    reference = rotate(query_rows), rotate(key_rows)
    actual = tuple(value.reshape(-1, _HEAD_DIM)[:rows] for value in output)
    return all(torch.equal(a, b) for a, b in zip(actual, reference, strict=True))


def try_qk_rms_norm_scale_rope_3d_exact(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    eps: float,
    query_scale: float,
    dim_split: tuple[int, int, int],
    base: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Normalize Q/K, scale Q, and apply paired 3D RoPE in one kernel."""

    if not _combined_supported_inputs(
        query,
        key,
        query_weight,
        key_weight,
        eps,
        query_scale,
        dim_split,
        base,
    ):
        return None
    runtime_key = (query.device.index, float(eps), float(query_scale), dim_split, float(base))
    if runtime_key in _FAILED_ROPE_KEYS:
        return None
    try:
        output = _launch_combined(
            query,
            key,
            query_weight,
            key_weight,
            eps,
            query_scale,
            dim_split,
            base,
        )
        if runtime_key not in _VERIFIED_ROPE_KEYS:
            if not _verify_combined_prefix(
                output,
                query,
                key,
                query_weight,
                key_weight,
                eps,
                query_scale,
                dim_split,
                base,
            ):
                _FAILED_ROPE_KEYS.add(runtime_key)
                logger.warning(
                    "Disabling LTX DiffVAE Q/K RMSNorm+3D RoPE fusion on %s after a bit-exactness mismatch",
                    query.device,
                )
                return None
            _VERIFIED_ROPE_KEYS.add(runtime_key)
    except Exception as exc:  # noqa: BLE001 - fail closed after optimized-path failure
        _FAILED_ROPE_KEYS.add(runtime_key)
        logger.warning(
            "Disabling LTX DiffVAE Q/K RMSNorm+3D RoPE fusion on %s after failure: %s",
            query.device,
            exc,
        )
        return None
    return output


__all__ = ["try_qk_rms_norm_scale_rope_3d_exact"]
