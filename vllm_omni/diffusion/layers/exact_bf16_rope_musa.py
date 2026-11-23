# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Interleaved RoPE and fused Q/K RMSNorm + RoPE Triton kernels for Qwen-Image-2.1 on MUSA.

They follow the CUDA kernels in ``exact_bf16_pointwise`` and ``exact_bf16_qk_norm_rope`` with two
differences: the rotation rounds every product, as MUSA's native real-arithmetic rotation does, and
the batch is a grid axis with the per-batch size passed at runtime, so a new sequence length reuses
the compiled kernel and the index math needs no runtime division. Each kernel runs through an opaque
custom op registered on MUSA only, which keeps the launch, and its enable_fp_fusion=False, out of
regionally compiled graphs.
"""

import torch
from torch.library import Library
from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_omni.platforms import current_omni_platform

# The ops are registered on MUSA only; the helpers below decline wherever they are not.
_HAS_MUSA_OPS = HAS_TRITON and current_omni_platform.is_musa()

if HAS_TRITON:

    @triton.jit
    def _rope_kernel(
        input_ptr,
        freq_ptr,
        output_ptr,
        pairs_per_batch,
        heads: tl.constexpr,
        x_stride0,
        x_stride1: tl.constexpr,
        x_stride2: tl.constexpr,
        freq_stride0: tl.constexpr,
        freq_stride1: tl.constexpr,
        block_size: tl.constexpr,
    ):
        # Grid: (blocks of one batch, batch).
        index = tl.program_id(0) * block_size + tl.arange(0, block_size)
        batch = tl.program_id(1)
        mask = index < pairs_per_batch
        pair = index % 64
        head = (index // 64) % heads
        token = index // (64 * heads)
        base = batch * x_stride0 + token * x_stride1 + head * x_stride2 + pair * 2
        real = tl.load(input_ptr + base, mask, 0).to(tl.float32)
        imag = tl.load(input_ptr + base + 1, mask, 0).to(tl.float32)
        cosine = tl.load(freq_ptr + token * freq_stride0 + pair * freq_stride1, mask, 0)
        sine = tl.load(freq_ptr + token * freq_stride0 + pair * freq_stride1 + 1, mask, 0)
        # The real-arithmetic rotation rounds every product first.
        rotated_real = real * cosine - imag * sine
        rotated_imag = imag * cosine + real * sine
        output = (batch * pairs_per_batch + index) * 2
        tl.store(output_ptr + output, rotated_real.to(tl.bfloat16), mask)
        tl.store(output_ptr + output + 1, rotated_imag.to(tl.bfloat16), mask)

    @triton.jit
    def _qk_norm_rope_kernel(
        q_ptr,
        k_ptr,
        qweight_ptr,
        kweight_ptr,
        freq_ptr,
        oq_ptr,
        ok_ptr,
        rows_per_batch,
        heads: tl.constexpr,
        q_stride0,
        q_stride1: tl.constexpr,
        q_stride2: tl.constexpr,
        k_stride0,
        k_stride1: tl.constexpr,
        k_stride2: tl.constexpr,
        freq_stride0: tl.constexpr,
        freq_stride1: tl.constexpr,
        eps: tl.constexpr,
        rows_per_block: tl.constexpr,
    ):
        # Grid: (row blocks of one batch, q or k, batch).
        row = tl.program_id(0) * rows_per_block + tl.arange(0, rows_per_block)
        lane = tl.arange(0, 32)
        head = row % heads
        token = row // heads
        batch = tl.program_id(2)
        plane = tl.program_id(1)
        if plane == 0:
            base = batch * q_stride0 + token * q_stride1 + head * q_stride2
            input_ptr = q_ptr
            weight_ptr = qweight_ptr
            output_ptr = oq_ptr
        else:
            base = batch * k_stride0 + token * k_stride1 + head * k_stride2
            input_ptr = k_ptr
            weight_ptr = kweight_ptr
            output_ptr = ok_ptr
        i = base[:, None] + lane[None, :] * 4
        mask = row[:, None] < rows_per_batch
        a = tl.load(input_ptr + i, mask, 0).to(tl.float32)
        b = tl.load(input_ptr + i + 1, mask, 0).to(tl.float32)
        c = tl.load(input_ptr + i + 2, mask, 0).to(tl.float32)
        d = tl.load(input_ptr + i + 3, mask, 0).to(tl.float32)
        # ATen Reduce.cuh vectorizes four adjacent values per lane, combines
        # them sequentially, then reduces lanes in descending warp offsets.
        local = ((a * a + b * b) + c * c) + d * d
        variance = tl.sum(local, axis=1) / 128
        rrms = tldevice.rsqrt(variance + eps)[:, None]
        wa = tl.load(weight_ptr + lane * 4).to(tl.float32)
        wb = tl.load(weight_ptr + lane * 4 + 1).to(tl.float32)
        wc = tl.load(weight_ptr + lane * 4 + 2).to(tl.float32)
        wd = tl.load(weight_ptr + lane * 4 + 3).to(tl.float32)
        # Eager RMSNorm materializes BF16 after normalization and again after
        # multiplying by the BF16 weight. Each BF16 cast reproduces that rounding;
        # widening back to FP32 preserves the rounded value for the next opmath
        # operation (weight multiplication, then complex-FP32 RoPE).
        a = ((a * rrms).to(tl.bfloat16).to(tl.float32) * wa[None, :]).to(tl.bfloat16).to(tl.float32)
        b = ((b * rrms).to(tl.bfloat16).to(tl.float32) * wb[None, :]).to(tl.bfloat16).to(tl.float32)
        c = ((c * rrms).to(tl.bfloat16).to(tl.float32) * wc[None, :]).to(tl.bfloat16).to(tl.float32)
        d = ((d * rrms).to(tl.bfloat16).to(tl.float32) * wd[None, :]).to(tl.bfloat16).to(tl.float32)
        fi = token[:, None] * freq_stride0 + lane[None, :] * 2 * freq_stride1
        ca = tl.load(freq_ptr + fi, mask, 0)
        sa = tl.load(freq_ptr + fi + 1, mask, 0)
        cb = tl.load(freq_ptr + fi + freq_stride1, mask, 0)
        sb = tl.load(freq_ptr + fi + freq_stride1 + 1, mask, 0)
        out = (batch * rows_per_batch + row)[:, None] * 128 + lane[None, :] * 4
        # The real-arithmetic rotation rounds every product first.
        tl.store(output_ptr + out, (a * ca - b * sa).to(tl.bfloat16), mask)
        tl.store(output_ptr + out + 1, (b * ca + a * sa).to(tl.bfloat16), mask)
        tl.store(output_ptr + out + 2, (c * cb - d * sb).to(tl.bfloat16), mask)
        tl.store(output_ptr + out + 3, (d * cb + c * sb).to(tl.bfloat16), mask)


def _interleaved_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    real_freqs = torch.view_as_real(freqs)
    _rope_kernel[(triton.cdiv(x[0].numel() // 2, 256), x.shape[0])](
        x,
        real_freqs,
        output,
        x[0].numel() // 2,
        x.shape[2],
        *x.stride()[:3],
        *real_freqs.stride()[:2],
        256,
        enable_fp_fusion=False,
    )
    return output


def _interleaved_rope_fake(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


def _qk_norm_rope(
    q: torch.Tensor, k: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, f: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    oq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    ok = torch.empty_like(oq)
    ff = torch.view_as_real(f)
    _qk_norm_rope_kernel[(triton.cdiv(q[0].numel() // 128, 4), 2, q.shape[0])](
        q,
        k,
        wq,
        wk,
        ff,
        oq,
        ok,
        q[0].numel() // 128,
        q.shape[2],
        *q.stride()[:3],
        *k.stride()[:3],
        *ff.stride()[:2],
        eps,
        4,
        enable_fp_fusion=False,
    )
    return oq, ok


def _qk_norm_rope_fake(
    q: torch.Tensor, k: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, f: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty(q.shape, dtype=q.dtype, device=q.device),
        torch.empty(k.shape, dtype=k.dtype, device=k.device),
    )


if _HAS_MUSA_OPS and not hasattr(torch.ops.vllm_omni, "musa_interleaved_rope"):
    _ROPE_OP_LIB = Library("vllm_omni", "FRAGMENT")
    direct_register_custom_op(
        op_name="musa_interleaved_rope",
        op_func=_interleaved_rope,
        fake_impl=_interleaved_rope_fake,
        mutates_args=[],
        target_lib=_ROPE_OP_LIB,
    )

if _HAS_MUSA_OPS and not hasattr(torch.ops.vllm_omni, "musa_qk_norm_rope"):
    _QK_OP_LIB = Library("vllm_omni", "FRAGMENT")
    direct_register_custom_op(
        op_name="musa_qk_norm_rope",
        op_func=_qk_norm_rope,
        fake_impl=_qk_norm_rope_fake,
        mutates_args=[],
        target_lib=_QK_OP_LIB,
    )


def musa_rotary(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor | None:
    """Interleaved RoPE on MUSA with the rounding of the real-arithmetic rotation, or ``None``.

    It also runs inside compiled regions, through the opaque op.
    """
    if (
        not _HAS_MUSA_OPS
        or x.device.type != "musa"
        or x.dtype != torch.bfloat16
        or x.ndim != 4
        or x.shape[-1] != 128
        or x.stride(-1) != 1
        or freqs.dtype != torch.complex64
        or freqs.device != x.device
        or freqs.shape != (x.shape[1], 64)
        or torch.is_grad_enabled()
        or x.numel() == 0
    ):
        return None
    return torch.ops.vllm_omni.musa_interleaved_rope(x, freqs)


def musa_qk_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    f: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Q/K RMSNorm and interleaved RoPE on MUSA with the rotation's unfused products, or ``None``.

    It also runs inside compiled regions, through the opaque op.
    """
    if (
        not _HAS_MUSA_OPS
        or q.device.type != "musa"
        or q.dtype != torch.bfloat16
        or q.ndim != 4
        or q.shape != k.shape
        or q.shape[-1] != 128
        or q.stride(-1) != 1
        or k.stride(-1) != 1
        or k.device != q.device
        or k.dtype != q.dtype
        or wq.dtype != q.dtype
        or wk.dtype != q.dtype
        or wq.shape != (128,)
        or wk.shape != (128,)
        or not wq.is_contiguous()
        or not wk.is_contiguous()
        or wq.device != q.device
        or wk.device != q.device
        or f.dtype != torch.complex64
        or f.device != q.device
        or f.shape != (q.shape[1], 64)
        or q.numel() == 0
        or torch.is_grad_enabled()
    ):
        return None
    return torch.ops.vllm_omni.musa_qk_norm_rope(q, k, wq, wk, f, eps)
