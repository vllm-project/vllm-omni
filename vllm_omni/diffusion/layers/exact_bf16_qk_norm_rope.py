# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""QK RMSNorm/RoPE matching ATen's CUDA head128 reduction and BF16 roundings."""

import os

import torch
from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton

from vllm_omni.diffusion.layers.device_gates import TRITON_FASTPATH_DEVICE_TYPES
from vllm_omni.diffusion.layers.exact_bf16_pointwise import supports_exact_rope

if HAS_TRITON:

    @triton.jit
    def _full(
        q_ptr,
        k_ptr,
        qweight_ptr,
        kweight_ptr,
        freq_ptr,
        oq_ptr,
        ok_ptr,
        row_count: tl.constexpr,
        sequence: tl.constexpr,
        heads: tl.constexpr,
        q_stride0: tl.constexpr,
        q_stride1: tl.constexpr,
        q_stride2: tl.constexpr,
        k_stride0: tl.constexpr,
        k_stride1: tl.constexpr,
        k_stride2: tl.constexpr,
        freq_stride0: tl.constexpr,
        freq_stride1: tl.constexpr,
        eps: tl.constexpr,
        rows_per_block: tl.constexpr,
        fuse_products: tl.constexpr,
    ):
        row = tl.program_id(0) * rows_per_block + tl.arange(0, rows_per_block)
        lane = tl.arange(0, 32)
        head = row % heads
        token = (row // heads) % sequence
        batch = row // (sequence * heads)
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
        mask = row[:, None] < row_count
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
        out = row[:, None] * 128 + lane[None, :] * 4
        if fuse_products:
            # CUDA's complex64 multiply fuses the real leading and imaginary
            # trailing products; the other two round first.
            r0 = tl.fma(a, ca, -b * sa)
            r1 = tl.fma(b, ca, a * sa)
            r2 = tl.fma(c, cb, -d * sb)
            r3 = tl.fma(d, cb, c * sb)
        else:
            # MUSA's complex multiply does not fuse its products, so the
            # reference rotation is a plain fp32 multiply/add sequence and this
            # path keeps that order to stay bit-exact.
            r0 = a * ca - b * sa
            r1 = b * ca + a * sa
            r2 = c * cb - d * sb
            r3 = d * cb + c * sb
        tl.store(output_ptr + out, r0.to(tl.bfloat16), mask)
        tl.store(output_ptr + out + 1, r1.to(tl.bfloat16), mask)
        tl.store(output_ptr + out + 2, r2.to(tl.bfloat16), mask)
        tl.store(output_ptr + out + 3, r3.to(tl.bfloat16), mask)


def qk_fma_products(device: torch.device) -> bool | None:
    """Whether the fused QK norm+RoPE fuses its products, or None when unqualified."""
    if device.type == "musa":
        return False
    if supports_exact_rope(device):
        return True
    return None


def _launch_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    f: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    fuse_products = qk_fma_products(q.device)
    if fuse_products is None:
        raise RuntimeError("qk_norm_rope is unqualified on this device")
    oq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    ok = torch.empty_like(oq)
    ff = torch.view_as_real(f)
    _full[(triton.cdiv(q.numel() // 128, 4), 2)](
        q,
        k,
        wq,
        wk,
        ff,
        oq,
        ok,
        q.numel() // 128,
        q.shape[1],
        q.shape[2],
        *q.stride()[:3],
        *k.stride()[:3],
        *ff.stride()[:2],
        eps,
        4,
        fuse_products,
        enable_fp_fusion=False,
    )
    return oq, ok


if HAS_TRITON:

    @torch.library.custom_op("vllm_omni::qk_norm_rope", mutates_args=())
    def _qk_norm_rope(
        q: torch.Tensor, k: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, f: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Q/K RMSNorm plus interleaved RoPE as an opaque op for compiled regions."""
        return _launch_qk(q, k, wq, wk, f, eps)

    @_qk_norm_rope.register_fake
    def _(
        q: torch.Tensor, k: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, f: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.empty(q.shape, dtype=q.dtype, device=q.device),
            torch.empty(k.shape, dtype=k.dtype, device=k.device),
        )


MUSA_QK_FUSION_ENV = "VLLM_OMNI_MUSA_FUSED_QK_NORM_ROPE"


def _musa_qk_fusion_enabled() -> bool:
    """Whether MUSA may use the fused QK norm+RoPE, off unless the env asks for it.

    The kernel is qualified on MUSA but is not bit-exact there, so a caller that
    requires bit-exactness must be able to keep the reference path.
    """
    return os.environ.get(MUSA_QK_FUSION_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def qk_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    f: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if (
        not HAS_TRITON
        or q.device.type not in TRITON_FASTPATH_DEVICE_TYPES
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
    if qk_fma_products(q.device) is None:
        return None
    if q.device.type == "musa" and not _musa_qk_fusion_enabled():
        return None
    return _qk_norm_rope(q, k, wq, wk, f, eps)
