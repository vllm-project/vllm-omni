# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PixelUMM split Q/K normalization and MRoPE fused while preserving eager arithmetic.

Split QK norms keep FP32 through weighting and each T/H/W rotation, then round
once for attention. The shared diffusion QK fusion and vLLM's RMSNorm round
normalized values before rotation and cannot express this contract. The
ordinary hidden-size norms and the MLP's SiLU*mul use vLLM's ``RMSNorm`` and
vLLM-Omni's ``SiluAndMul`` instead.

The fusion takes the fastest exact implementation that applies:

1. A Triton kernel for PixelUMM's BF16 CUDA Q/K shape (128-wide heads, any
   batch), reading heads in place from the stacked QKV projection. It
   reproduces ATen's reduction order, so no FP32 squares are materialized.
2. ``torch.compile`` with ``emulate_precision_casts`` for other widths, dtypes,
   strides and accelerators. Mean squares stay in ATen to keep its summation
   order, at the cost of one extra FP32 pass.
3. ``None`` on CPU, under autograd or inside an outer compile; callers then run
   the eager math.

The eager MRoPE helpers live here too, so the decoder imports this module
without an import cycle.
"""

from functools import cache

import torch
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton

# Bound the FP32 squares the compiled QK path materializes per chunk.
_COMPILED_QK_CHUNK_BYTES = 256 * 2**20


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_mrope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, sections) -> torch.Tensor:
    """x [B,N,H,D]; each section is rotated within itself (rotate_half per axis)."""
    parts = x.split(tuple(sections), dim=-1)
    rotated = torch.cat([_rotate_half(part) for part in parts], dim=-1)
    return x * cos + rotated * sin


if HAS_TRITON:

    @triton.jit
    def _split_norm_rope(
        x_ptr,
        wt_ptr,
        whw_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        rows,
        tokens,
        token_stride,
        heads: tl.constexpr,
        eps: tl.constexpr,
        rows_per_block: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64) * rows_per_block + tl.arange(0, rows_per_block)
        lane = tl.arange(0, 32)
        mask = row[:, None] < rows
        # Input rows may be heads of a stacked QKV projection; the output is contiguous.
        token_index = row // heads
        head_index = row % heads
        base = token_index[:, None] * token_stride + head_index[:, None] * 128
        out_base = row[:, None] * 128
        t0 = tl.load(x_ptr + base + lane[None, :], mask, 0).to(tl.float32)
        t1 = tl.load(x_ptr + base + lane[None, :] + 32, mask, 0).to(tl.float32)
        h0 = tl.load(x_ptr + base + lane[None, :] + 64, mask, 0).to(tl.float32)
        h1 = tl.load(x_ptr + base + lane[None, :] + 96, mask, 0).to(tl.float32)
        inv_t = tldevice.rsqrt(tl.sum(t0 * t0 + t1 * t1, 1) * (1.0 / 64) + eps)
        inv_hw = tldevice.rsqrt(tl.sum(h0 * h0 + h1 * h1, 1) * (1.0 / 64) + eps)
        col = tl.arange(0, 128)
        # Pair within T(64), H(32), W(32), rather than across the whole head.
        half = tl.where(col < 64, 32, 16)
        pair = col ^ half
        sign = tl.where((col & half) == 0, -1.0, 1.0)
        inv = tl.where(col[None, :] < 64, inv_t[:, None], inv_hw[:, None])
        weight_t = tl.load(wt_ptr + col % 64)
        weight_hw = tl.load(whw_ptr + col % 64)
        weight = tl.where(col < 64, weight_t, weight_hw).to(tl.float32)
        pair_weight_t = tl.load(wt_ptr + pair % 64)
        pair_weight_hw = tl.load(whw_ptr + pair % 64)
        pair_weight = tl.where(pair < 64, pair_weight_t, pair_weight_hw).to(tl.float32)
        x = tl.load(x_ptr + base + col[None, :], mask, 0).to(tl.float32)
        other = tl.load(x_ptr + base + pair[None, :], mask, 0).to(tl.float32)
        normalized = (x * inv) * weight[None, :]
        rotated = ((other * inv) * pair_weight[None, :]) * sign[None, :]
        # cos/sin have one row per token, shared by all heads and batches.
        token = token_index % tokens
        table_offset = token[:, None] * 128 + col[None, :]
        cosine = tl.load(cos_ptr + table_offset, mask, 0)
        sine = tl.load(sin_ptr + table_offset, mask, 0)
        tl.store(out_ptr + out_base + col[None, :], normalized * cosine + rotated * sine, mask)


def _fusable(x: torch.Tensor) -> bool:
    """Whether a fused path may serve ``x``: Inductor devices outside autograd and outer compiles."""
    # CUDA covers ROCm. Other accelerators keep eager rather than an untested compiler backend.
    return x.device.type in ("cuda", "xpu") and not torch.is_grad_enabled() and not torch.compiler.is_compiling()


def _triton_device(x: torch.Tensor) -> bool:
    return HAS_TRITON and current_platform.is_cuda() and x.device.type == "cuda"


def _compile_exact(fn):
    # emulate_precision_casts keeps every eager rounding; reductions stay in ATen.
    return torch.compile(fn, dynamic=True, fullgraph=True, options={"emulate_precision_casts": True})


def _square(x):
    return x.float().pow(2)


def _qk_norm_mrope(x, mean_t, mean_hw, eps, weight_t, weight_hw, cos, sin, sections):
    temporal, spatial = x.float().chunk(2, dim=-1)
    temporal = weight_t * (temporal * torch.rsqrt(mean_t + eps))
    spatial = weight_hw * (spatial * torch.rsqrt(mean_hw + eps))
    normalized = torch.cat((temporal, spatial), dim=-1)
    return apply_mrope(normalized, cos, sin, sections).to(x.dtype)


@cache
def _compiled(name: str):
    if name == "square":
        return _compile_exact(_square)
    return _compile_exact(_qk_norm_mrope)


def _mean_square(x: torch.Tensor) -> torch.Tensor:
    return _compiled("square")(x).mean(-1, keepdim=True)


def split_norm_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, weight_t: torch.Tensor, weight_hw: torch.Tensor, eps: float
) -> torch.Tensor | None:
    """Fuse PixelUMM's two FP32 Q/K norms and three rotations for ``x`` ``[B, N, heads, D]``.

    ``cos``/``sin`` are ``[1, N, 1, D]``, shared by every batch element and head.
    """
    if not _fusable(x):
        return None
    batch, tokens, heads, dim = x.shape
    if (
        _triton_device(x)
        and x.dtype == torch.bfloat16
        and dim == 128
        and x.stride(3) == 1
        and x.stride(2) == 128
        and x.stride(0) == tokens * x.stride(1)
        and cos.dtype == torch.float32
        and sin.dtype == torch.float32
        and cos.shape == (1, tokens, 1, 128)
        and sin.shape == cos.shape
        and cos.is_contiguous()
        and sin.is_contiguous()
        and weight_t.dtype == x.dtype
        and weight_hw.dtype == x.dtype
        and weight_t.shape == (64,)
        and weight_hw.shape == (64,)
        and x.numel() > 0
    ):
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        rows = x.numel() // 128
        with torch.get_device_module().device(x.device):
            _split_norm_rope[(triton.cdiv(rows, 4),)](
                x,
                weight_t,
                weight_hw,
                cos,
                sin,
                out,
                rows,
                tokens,
                x.stride(1),
                heads,
                eps,
                rows_per_block=4,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return out
    sections = (dim // 2, dim // 4, dim // 4)
    half = dim // 2
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    chunk = max(1, _COMPILED_QK_CHUNK_BYTES // (batch * heads * half * 4))
    for first in range(0, tokens, chunk):
        last = first + chunk
        part = x[:, first:last]
        mean_t = _mean_square(part[..., :half])
        mean_hw = _mean_square(part[..., half:])
        out[:, first:last] = _compiled("qk")(
            part, mean_t, mean_hw, eps, weight_t, weight_hw, cos[:, first:last], sin[:, first:last], sections
        )
    return out
