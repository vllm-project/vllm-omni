# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Two-branch CFG followed by last-dimension L2 normalization.

Preserve the eager rounding of difference, scale, sum, both norms, norm ratio,
and final multiplication. In particular zero norms retain NaN/Inf behavior;
there is no epsilon, clamp, or standard-deviation rescale.
"""

import torch
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:
    # Explicit PTX keeps both rounding and subnormals. CUDA libdevice rn
    # intrinsics inherit Triton's FTZ mode, including when traced by Inductor.
    @triton.jit
    def _add_rn(x, y):
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[x, y],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _sub_rn(x, y):
        return tl.inline_asm_elementwise(
            "sub.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[x, y],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _mul_rn(x, y):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[x, y],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _sqrt_rn(x):
        return tl.inline_asm_elementwise(
            "sqrt.rn.f32 $0, $1;",
            constraints="=f,f",
            args=[x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _cfg_l2_kernel(
        positive_ptr,
        negative_ptr,
        out_ptr,
        seq_len: tl.constexpr,
        hidden_size: tl.constexpr,
        rows: tl.constexpr,
        positive_stride_b: tl.constexpr,
        positive_stride_s: tl.constexpr,
        negative_stride_b: tl.constexpr,
        negative_stride_s: tl.constexpr,
        guidance_scale: tl.constexpr,
        block_d: tl.constexpr,
        block_rows: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64) * block_rows + tl.arange(0, block_rows)
        d = tl.arange(0, block_d)
        batch, token = row // seq_len, row % seq_len
        mask = (row[:, None] < rows) & (d[None, :] < hidden_size)
        dtype = positive_ptr.dtype.element_ty
        p = tl.load(
            positive_ptr + batch[:, None] * positive_stride_b + token[:, None] * positive_stride_s + d[None, :],
            mask,
            other=0,
        ).to(tl.float32)
        n = tl.load(
            negative_ptr + batch[:, None] * negative_stride_b + token[:, None] * negative_stride_s + d[None, :],
            mask,
            other=0,
        ).to(tl.float32)
        diff = _sub_rn(p, n).to(dtype).to(tl.float32)
        scaled = _mul_rn(tl.full((), guidance_scale, tl.float32), diff).to(dtype).to(tl.float32)
        combined = _add_rn(n, scaled).to(dtype).to(tl.float32)
        p_sum = tl.sum(tl.where(d[None, :] < hidden_size, _mul_rn(p, p), 0), 1)
        c_sum = tl.sum(tl.where(d[None, :] < hidden_size, _mul_rn(combined, combined), 0), 1)
        # The approximate sqrt flushes subnormal squared norms to zero.
        # Keep the reference norm behavior for tiny but nonzero rows.
        p_norm = _sqrt_rn(p_sum).to(dtype).to(tl.float32)
        c_norm = _sqrt_rn(c_sum).to(dtype).to(tl.float32)
        ratio = tl.div_rn(p_norm, c_norm).to(dtype).to(tl.float32)
        out = _mul_rn(combined, ratio[:, None]).to(dtype)
        tl.store(out_ptr + row[:, None] * hidden_size + d[None, :], out, mask)


def try_fused_cfg_l2(
    positive: torch.Tensor,
    negative: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor | None:
    """Fuse CUDA FP32/BF16 [B,S,D] rows; return None for the caller's fallback.

    Supports unit channel stride and slicing along batch/sequence, D <= 8192.
    Autograd keeps the model's original implementation; torch.compile traces
    the functional Triton launch.
    The helper has no collective or model state and does not mutate inputs.
    """
    if (
        not HAS_TRITON
        or not current_platform.is_cuda()
        or not positive.is_cuda
        or positive.ndim != 3
        or positive.dtype not in (torch.float32, torch.bfloat16)
        or positive.numel() == 0
        or not 0 < positive.shape[-1] <= 8192
        or type(guidance_scale) not in (float, int)
    ):
        return None
    if negative.shape != positive.shape or negative.dtype != positive.dtype or negative.device != positive.device:
        return None
    if any(t.layout != torch.strided or t.stride(-1) != 1 for t in (positive, negative)):
        return None
    if torch.is_grad_enabled() and (positive.requires_grad or negative.requires_grad):
        return None
    b, s, d = positive.shape
    out = torch.empty(positive.shape, dtype=positive.dtype, device=positive.device)
    rows_per_program = 4 if d <= 256 else 1
    _cfg_l2_kernel[(triton.cdiv(b * s, rows_per_program),)](
        positive,
        negative,
        out,
        s,
        d,
        b * s,
        positive.stride(0),
        positive.stride(1),
        negative.stride(0),
        negative.stride(1),
        guidance_scale,
        triton.next_power_of_2(d),
        rows_per_program,
        num_warps=4 if d <= 2048 else 8,
        enable_fp_fusion=False,
    )
    return out
