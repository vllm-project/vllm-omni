# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""BF16 norm and softmax with audited PyTorch CUDA reduction orders.

The fast path follows Torch 2.13.0's Reduce.cuh: four serial accumulators per
lane, then descending warp shuffles. It is intentionally limited to the two
Pi0.5 widths and sufficiently many rows. Other versions, layouts and dtypes
use the actual PyTorch operations, rather than assuming the reduction is stable
across releases. Softmax additionally follows PersistentSoftmax.cuh for the
default prefix and action-expert attention shapes. This module does not modify model dispatch.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton


def _supported_torch_reduction() -> bool:
    # Extend only after auditing Reduce.cuh/ReduceMomentKernel.cu and
    # PersistentSoftmax.cuh/SoftMax.cu, then running bit-exact tests on that
    # release. Local CUDA build suffix is irrelevant.
    return torch.version.hip is None and str(torch.__version__).split("+")[0] == "2.13.0"


def can_use_exact_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    add: torch.Tensor | None = None,
    shift: torch.Tensor | None = None,
) -> bool:
    """Whether the audited fused implementation can replace eager operations."""
    if not HAS_TRITON or not _supported_torch_reduction():
        return False
    if x.ndim < 2 or x.device.type != "cuda" or x.dtype != torch.bfloat16 or not x.is_contiguous():
        return False
    width = x.shape[-1]
    if width not in (1024, 2048) or x.numel() // width < 16:
        return False
    for value in (weight, shift):
        if value is not None and (
            value.shape != (width,)
            or value.device != x.device
            or value.dtype not in (torch.float32, torch.bfloat16)
            or not value.is_contiguous()
        ):
            return False
    if add is not None and (
        add.shape != x.shape or add.device != x.device or add.dtype != x.dtype or not add.is_contiguous()
    ):
        return False
    # This is inference code. Preserve ordinary autograd when called directly.
    if torch.is_grad_enabled() and any(value.requires_grad for value in (x, weight, add, shift) if value is not None):
        return False
    return True


def can_use_exact_scaled_masked_softmax(
    scores: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float,
    *,
    output_dtype: torch.dtype | None = None,
) -> bool:
    """Limit the audited persistent-softmax tree to validated BF16 geometries."""
    if not HAS_TRITON or not _supported_torch_reduction():
        return False
    if scores.shape == (1, 8, 968, 968):
        # The prefix geometry has only been validated with this CUDA build.
        if torch.version.cuda != "13.2":
            return False
    elif scores.shape != (1, 8, 50, 1018):
        return False
    if (
        scores.device.type != "cuda"
        or scores.dtype != torch.bfloat16
        or not scores.is_contiguous()
        or output_dtype not in (None, torch.bfloat16)
        or not isinstance(scale, (float, int))
        or scale != 1.0 / 16
    ):
        return False
    if (
        mask is None
        or mask.shape != (1, 1, *scores.shape[-2:])
        or mask.device != scores.device
        or mask.dtype != torch.float32
        or not mask.is_contiguous()
    ):
        return False
    return not (torch.is_grad_enabled() and (scores.requires_grad or mask.requires_grad))


if HAS_TRITON:

    @triton.jit
    def _residual_value(X, ADD, offset, HAS_ADD: tl.constexpr):
        value = tl.load(X + offset).to(tl.float32)
        if HAS_ADD:
            value = (value + tl.load(ADD + offset).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        return value

    @triton.jit
    def torch_order_row_mean_square(X, ADD, row, WIDTH: tl.constexpr, HAS_ADD: tl.constexpr):
        """Audited contiguous BF16 row; callers must disable FP fusion.

        The host guard is mandatory: rows <16 can alter Torch's block width,
        and widths outside 1024/2048 have not been validated.
        """
        lanes = tl.arange(0, 32)
        a0 = tl.full((32,), 0, tl.float32)
        a1 = tl.full((32,), 0, tl.float32)
        a2 = tl.full((32,), 0, tl.float32)
        a3 = tl.full((32,), 0, tl.float32)
        for block in range(WIDTH // 128):
            offset = row * WIDTH + lanes * 4 + block * 128
            x0 = _residual_value(X, ADD, offset, HAS_ADD)
            x1 = _residual_value(X, ADD, offset + 1, HAS_ADD)
            x2 = _residual_value(X, ADD, offset + 2, HAS_ADD)
            x3 = _residual_value(X, ADD, offset + 3, HAS_ADD)
            a0 = a0 + x0 * x0
            a1 = a1 + x1 * x1
            a2 = a2 + x2 * x2
            a3 = a3 + x3 * x3
        value = ((a0 + a1) + a2) + a3
        for log_offset in tl.static_range(5):
            offset = 16 >> log_offset
            other = tl.gather(value, tl.minimum(lanes + offset, 31), axis=0)
            value = value + other
        # Broadcast lane zero by adding exact zeros, without changing its tree.
        total = tl.sum(tl.where(lanes == 0, value, 0), axis=0)
        return total * (1.0 / WIDTH)

    @triton.jit
    def _exact_rms_norm_kernel(
        X,
        ADD,
        WEIGHT,
        SHIFT,
        RESIDUAL,
        OUT,
        WIDTH: tl.constexpr,
        EPS: tl.constexpr,
        HAS_ADD: tl.constexpr,
        ADARMS: tl.constexpr,
    ):
        row = tl.program_id(0)
        variance = torch_order_row_mean_square(X, ADD, row, WIDTH, HAS_ADD)
        cols = tl.arange(0, WIDTH)
        x = _residual_value(X, ADD, row * WIDTH + cols, HAS_ADD)
        if HAS_ADD:
            tl.store(RESIDUAL + row * WIDTH + cols, x)
        normalized = x * tldevice.rsqrt(variance + EPS)
        weight = tl.load(WEIGHT + cols).to(tl.float32)
        result = normalized * (1.0 + weight)
        if ADARMS:
            result = result + tl.load(SHIFT + cols).to(tl.float32)
        tl.store(OUT + row * WIDTH + cols, result.to(tl.bfloat16))

    @triton.jit
    def _exact_scaled_masked_softmax_kernel(
        SCORES,
        MASK,
        OUT,
        WIDTH: tl.constexpr,
        QUERIES: tl.constexpr,
        ROWS: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        # Torch's persistent dispatch uses four warps, one row per warp. Each
        # lane accumulates 32 values in order before the descending XOR tree.
        rows = tl.program_id(0) * 4 + tl.arange(0, 4)[:, None]
        lanes = tl.arange(0, 32)[None, :]
        values = ()
        maximum = tl.full((4, 32), float("-inf"), tl.float32)
        for block in tl.static_range(32):
            cols = block * 32 + lanes
            valid = (rows < ROWS) & (cols < WIDTH)
            score = tl.load(SCORES + rows * WIDTH + cols, mask=valid, other=0).to(tl.float32)
            # The eager multiply rounds to BF16 before the FP32 mask addition.
            scaled = (score * SCALE).to(tl.bfloat16).to(tl.float32)
            mask = tl.load(MASK + (rows % QUERIES) * WIDTH + cols, mask=valid, other=0)
            value = tl.where(valid, scaled + mask, float("-inf"))
            values += (value,)
            maximum = tl.where(maximum > value, maximum, value)
        for log_offset in tl.static_range(5):
            index = tl.broadcast_to(lanes ^ (16 >> log_offset), (4, 32))
            other = tl.gather(maximum, index, axis=1)
            maximum = tl.where(maximum < other, other, maximum)
        exponentials = ()
        denominator = tl.full((4, 32), 0, tl.float32)
        for block in tl.static_range(32):
            exponential = tldevice.exp(values[block] - maximum)
            exponentials += (exponential,)
            denominator = denominator + exponential
        for log_offset in tl.static_range(5):
            index = tl.broadcast_to(lanes ^ (16 >> log_offset), (4, 32))
            denominator = denominator + tl.gather(denominator, index, axis=1)
        for block in tl.static_range(32):
            cols = block * 32 + lanes
            valid = (rows < ROWS) & (cols < WIDTH)
            probability = tl.div_rn(exponentials[block], denominator)
            tl.store(OUT + rows * WIDTH + cols, probability.to(tl.bfloat16), mask=valid)


def scaled_masked_softmax(
    scores: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Preserve eager scale rounding, FP32 masked softmax and final output cast.

    Unsupported shapes, versions, layouts and differentiable inputs execute
    the actual PyTorch operations. QK and PV matrix multiplies stay with callers.
    """
    if not can_use_exact_scaled_masked_softmax(scores, mask, scale, output_dtype=output_dtype):
        logits = scores * scale
        if mask is not None:
            logits = logits + mask
        return torch.softmax(logits, dim=-1, dtype=torch.float32).to(output_dtype or scores.dtype)
    output = torch.empty_like(scores)
    queries, width = scores.shape[-2:]
    rows = scores.numel() // width
    _exact_scaled_masked_softmax_kernel[(triton.cdiv(rows, 4),)](
        scores,
        mask,
        output,
        WIDTH=width,
        QUERIES=queries,
        ROWS=rows,
        SCALE=1.0 / 16,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


def residual_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    add: torch.Tensor | None = None,
    shift: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return residual and normalized output, preserving eager dtype boundaries.

    ``weight`` is Gemma's norm weight, or AdaRMS's already-computed scale when
    ``shift`` is provided. Conditioning projections and the gate stay with the
    caller. Unsupported inputs use PyTorch, including batched modulation rows.
    """
    if not can_use_exact_rms_norm(x, weight, add=add, shift=shift):
        residual = x if add is None else x + add
        value = residual.float()
        variance = value.square().mean(dim=-1, keepdim=True)
        normalized = value * torch.rsqrt(variance + eps)
        normalized = normalized * (1.0 + weight.float())
        if shift is not None:
            normalized = normalized + shift.float()
        return residual, normalized.to(residual.dtype)
    residual = x if add is None else torch.empty_like(x)
    normalized = torch.empty_like(x)
    width = x.shape[-1]
    _exact_rms_norm_kernel[(x.numel() // width,)](
        x,
        x if add is None else add,
        weight,
        weight if shift is None else shift,
        residual,
        normalized,
        WIDTH=width,
        EPS=eps,
        HAS_ADD=add is not None,
        ADARMS=shift is not None,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return residual, normalized
