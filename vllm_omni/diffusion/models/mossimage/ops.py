# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Inference kernels preserving MOSSImage's eager BF16 rounding boundaries."""

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

# The vectorized reduction below matches the torch 2.13 CUDA mean kernel.
_MATCHED_REDUCTION = torch.__version__.split("+")[0].split(".")[:2] == ["2", "13"]

if HAS_TRITON:
    from triton.language.extra.cuda import libdevice

    # Shape-dependent sizes (sequence length, row counts, batch strides) are
    # runtime arguments and the batch index comes from the launch grid, so a
    # new prompt length or batch size reuses the compiled kernel. Only values
    # fixed by the model (hidden sizes, head counts, per-row strides) are
    # constexpr. `seq` is not specialized on divisibility by 16 either.
    @triton.jit
    def _gated_residual_kernel(x, y, gate, out, per_batch, dim: tl.constexpr, gs: tl.constexpr, block: tl.constexpr):
        batch = tl.program_id(1)
        j = tl.program_id(0) * block + tl.arange(0, block)
        i = batch * per_batch + j
        a = tl.load(x + i, j < per_batch, 0).to(tl.float32)
        b = tl.load(y + i, j < per_batch, 0).to(tl.float32)
        g = tl.load(gate + batch * gs + j % dim, j < per_batch, 0).to(tl.float32)
        product = (b * g).to(tl.bfloat16).to(tl.float32)
        tl.store(out + i, a + product, j < per_batch)

    @triton.jit(do_not_specialize=["seq"])
    def _qk_norm_rope_kernel(
        q,
        k,
        qw,
        kw,
        rope,
        oq,
        ok,
        seq,
        heads: tl.constexpr,
        bs,
        ss: tl.constexpr,
        hs: tl.constexpr,
        rbs,
        rs: tl.constexpr,
    ):
        if tl.program_id(1) == 0:
            src, weight, dst = q, qw, oq
        else:
            src, weight, dst = k, kw, ok
        batch = tl.program_id(2)
        local = tl.program_id(0) * 4 + tl.arange(0, 4)
        s = local // heads
        base = batch * bs + s * ss + local % heads * hs
        rows = (batch + 1) * (seq * heads)
        row = batch * (seq * heads) + local
        lane = tl.arange(0, 32)
        offset = base[:, None] + lane[None, :] * 4
        v0 = tl.load(src + offset, row[:, None] < rows, 0).to(tl.float32)
        v1 = tl.load(src + offset + 1, row[:, None] < rows, 0).to(tl.float32)
        v2 = tl.load(src + offset + 2, row[:, None] < rows, 0).to(tl.float32)
        v3 = tl.load(src + offset + 3, row[:, None] < rows, 0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(((v0 * v0 + v1 * v1) + v2 * v2) + v3 * v3, 1) / 128.0 + 1e-6)
        d = tl.arange(0, 128)
        pair = (d + 64) % 128
        x = tl.load(src + base[:, None] + d[None, :], row[:, None] < rows, 0).to(tl.float32)
        y = tl.load(src + base[:, None] + pair[None, :], row[:, None] < rows, 0).to(tl.float32)
        w = tl.load(weight + d).to(tl.float32)
        pw = tl.load(weight + pair).to(tl.float32)
        x = ((x * inv[:, None]).to(tl.bfloat16).to(tl.float32) * w[None, :]).to(tl.bfloat16).to(tl.float32)
        y = ((y * inv[:, None]).to(tl.bfloat16).to(tl.float32) * pw[None, :]).to(tl.bfloat16).to(tl.float32)
        rope_offset = batch * rbs + s[:, None] * rs + d[None, :] % 64
        sin = tl.load(rope + rope_offset, row[:, None] < rows, 0).to(tl.float32)
        cos = tl.load(rope + rope_offset + 64, row[:, None] < rows, 0).to(tl.float32)
        y = tl.where(d[None, :] < 64, -y, y)
        a = (x * cos).to(tl.bfloat16).to(tl.float32)
        b = (y * sin).to(tl.bfloat16).to(tl.float32)
        tl.store(dst + row[:, None] * 128 + d[None, :], a + b, row[:, None] < rows)

    @triton.jit(do_not_specialize=["seq"])
    def _rms_full_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        scale_ptr,
        shift_ptr,
        dim: tl.constexpr,
        seq,
        heads: tl.constexpr,
        bs,
        ss: tl.constexpr,
        hs: tl.constexpr,
        modulate: tl.constexpr,
        scale_stride: tl.constexpr,
        shift_stride: tl.constexpr,
        block: tl.constexpr,
        row_tile: tl.constexpr,
    ):
        batch = tl.program_id(1)
        local = tl.program_id(0) * row_tile + tl.arange(0, row_tile)[:, None]
        base = batch * bs + (local // heads) * ss + (local % heads) * hs
        rows = (batch + 1) * (seq * heads)
        row = batch * (seq * heads) + local
        lane = tl.arange(0, 32)[None, :]
        a0 = tl.full((row_tile, 32), 0, tl.float32)
        a1 = tl.full((row_tile, 32), 0, tl.float32)
        a2 = tl.full((row_tile, 32), 0, tl.float32)
        a3 = tl.full((row_tile, 32), 0, tl.float32)
        # torch Reduce.cuh: four independent vector lanes per thread, followed
        # by left-associated combination and descending-offset warp reduction.
        # Four rows/four warps keep 32 lanes per reduction. Other launch layouts
        # can change tl.sum's order; do not tune them without exactness checks.
        # Unroll loads while retaining each accumulator's sequential additions.
        for chunk in tl.static_range(dim // 128):
            offset = base + chunk * 128 + lane * 4
            v0 = tl.load(x_ptr + offset, row < rows, 0).to(tl.float32)
            v1 = tl.load(x_ptr + offset + 1, row < rows, 0).to(tl.float32)
            v2 = tl.load(x_ptr + offset + 2, row < rows, 0).to(tl.float32)
            v3 = tl.load(x_ptr + offset + 3, row < rows, 0).to(tl.float32)
            a0 = a0 + v0 * v0
            a1 = a1 + v1 * v1
            a2 = a2 + v2 * v2
            a3 = a3 + v3 * v3
        inv = tl.rsqrt(tl.sum(((a0 + a1) + a2) + a3, 1) * (1.0 / dim) + 1e-6)[:, None]
        d = tl.arange(0, block)[None, :]
        mask = (row < rows) & (d < dim)
        x = tl.load(x_ptr + base + d, mask, other=0).to(tl.float32)
        w = tl.load(w_ptr + d, d < dim, other=0).to(tl.float32)
        result = (x * inv).to(tl.bfloat16).to(tl.float32) * w
        if modulate:
            scale = tl.load(scale_ptr + batch * scale_stride + d, mask, other=0).to(tl.float32)
            shift = tl.load(shift_ptr + batch * shift_stride + d, mask, other=0).to(tl.float32)
            result = result.to(tl.bfloat16).to(tl.float32)
            factor = (1.0 + scale).to(tl.bfloat16).to(tl.float32)
            result = (result * factor).to(tl.bfloat16).to(tl.float32) + shift
        tl.store(out_ptr + row * dim + d, result, mask)

    @triton.jit
    def _situ_kernel(h_ptr, g_ptr, out_ptr, n, dim: tl.constexpr, stride: tl.constexpr, block: tl.constexpr):
        i = tl.program_id(0) * block + tl.arange(0, block)
        offset = i // dim * stride + i % dim
        h = tl.load(h_ptr + offset, i < n, other=0).to(tl.float32)
        g = tl.load(g_ptr + offset, i < n, other=0).to(tl.float32)
        gt = (g * 0.25).to(tl.bfloat16).to(tl.float32)
        gt = libdevice.tanh(gt).to(tl.bfloat16).to(tl.float32)
        gt = (4.0 * gt).to(tl.bfloat16).to(tl.float32)
        sigmoid = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-g)).to(tl.bfloat16).to(tl.float32)
        a = (gt * sigmoid).to(tl.bfloat16).to(tl.float32)
        ht = (h * (1.0 / 25.0)).to(tl.bfloat16).to(tl.float32)
        ht = libdevice.tanh(ht).to(tl.bfloat16).to(tl.float32)
        b = (25.0 * ht).to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + i, a * b, i < n)

    @triton.jit
    def _rms_tail_kernel(
        x_ptr,
        inv_ptr,
        weight_ptr,
        out_ptr,
        scale_ptr,
        shift_ptr,
        per_batch,
        dim: tl.constexpr,
        block: tl.constexpr,
        modulate: tl.constexpr,
        scale_stride: tl.constexpr,
        shift_stride: tl.constexpr,
    ):
        batch = tl.program_id(1)
        j = tl.program_id(0) * block + tl.arange(0, block)
        i = batch * per_batch + j
        mask = j < per_batch
        x = tl.load(x_ptr + i, mask, other=0)
        inv = tl.load(inv_ptr + i // dim, mask, other=0)
        weight = tl.load(weight_ptr + j % dim, mask, other=0).to(tl.float32)
        normalized = (x * inv).to(tl.bfloat16).to(tl.float32)
        result = normalized * weight
        if modulate:
            scale = tl.load(scale_ptr + batch * scale_stride + j % dim, mask, other=0).to(tl.float32)
            shift = tl.load(shift_ptr + batch * shift_stride + j % dim, mask, other=0).to(tl.float32)
            # Preserve all three eager BF16 rounding boundaries.
            result = result.to(tl.bfloat16).to(tl.float32)
            factor = (1.0 + scale).to(tl.bfloat16).to(tl.float32)
            result = (result * factor).to(tl.bfloat16).to(tl.float32) + shift
        tl.store(out_ptr + i, result, mask)

    @triton.jit
    def _rotate_qk_kernel(
        q_ptr,
        k_ptr,
        rope_ptr,
        out_q_ptr,
        out_k_ptr,
        per_batch,
        heads: tl.constexpr,
        dim: tl.constexpr,
        rope_batch_stride,
        rope_stride: tl.constexpr,
        block: tl.constexpr,
    ):
        batch = tl.program_id(2)
        j = tl.program_id(0) * block + tl.arange(0, block)
        i = batch * per_batch + j
        n = (batch + 1) * per_batch
        d = j % dim
        s = j // (heads * dim)
        pair = i + tl.where(d < dim // 2, dim // 2, -dim // 2)
        if tl.program_id(1) == 0:
            src = q_ptr
            dst = out_q_ptr
        else:
            src = k_ptr
            dst = out_k_ptr
        x = tl.load(src + i, i < n, other=0).to(tl.float32)
        y = tl.load(src + pair, i < n, other=0).to(tl.float32)
        rope_offset = batch * rope_batch_stride + s * rope_stride + d % (dim // 2)
        sin = tl.load(rope_ptr + rope_offset, i < n, other=0).to(tl.float32)
        cos = tl.load(rope_ptr + rope_offset + dim // 2, i < n, other=0).to(tl.float32)
        y = tl.where(d < dim // 2, -y, y)
        # Eager BF16 multiplication rounds both products before adding.
        first = (x * cos).to(tl.bfloat16).to(tl.float32)
        second = (y * sin).to(tl.bfloat16).to(tl.float32)
        tl.store(dst + i, first + second, i < n)


def gated_residual(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Fuse gated residual addition, preserving eager BF16 product rounding."""
    if not (
        HAS_TRITON
        and x.is_cuda
        and torch.version.hip is None
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and x.ndim == 3
        and x.shape == y.shape
        and x.numel() > 0
        and gate.shape == (x.shape[0], x.shape[2])
        and gate.stride(-1) == 1
        and x.is_contiguous()
        and y.is_contiguous()
        and x.dtype == y.dtype == gate.dtype == torch.bfloat16
        and x.device == y.device == gate.device
    ):
        return x + y * gate[:, None]
    out = torch.empty_like(x)
    per_batch = x.shape[1] * x.shape[2]
    _gated_residual_kernel[(triton.cdiv(per_batch, 1024), x.shape[0])](
        x,
        y,
        gate,
        out,
        per_batch,
        x.shape[2],
        gate.stride(0),
        1024,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


def try_qk_norm_rope(q, k, q_weight, k_weight, rope):
    """Fuse the matched 128-wide normalization and half-split RoPE."""
    if not (
        HAS_TRITON
        and _MATCHED_REDUCTION
        and q.is_cuda
        and torch.version.hip is None
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and q.ndim == 4
        and q.shape == k.shape
        and q.shape[-1] == 128
        and q.numel() // 128 >= 16
        and q.stride() == k.stride()
        and q.stride(-1) == 1
        and q_weight.shape == k_weight.shape == (128,)
        and q_weight.is_contiguous()
        and k_weight.is_contiguous()
        and rope.ndim == 3
        and rope.shape[0] in (1, q.shape[0])
        and rope.shape[1:] == (q.shape[1], 128)
        and rope.stride(-1) == 1
        and all(t.dtype == torch.bfloat16 and t.device == q.device for t in (q, k, q_weight, k_weight, rope))
    ):
        return None
    oq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    ok = torch.empty_like(oq)
    _qk_norm_rope_kernel[(triton.cdiv(q.shape[1] * q.shape[2], 4), 2, q.shape[0])](
        q,
        k,
        q_weight,
        k_weight,
        rope,
        oq,
        ok,
        q.shape[1],
        q.shape[2],
        *q.stride()[:3],
        rope.stride(0) if rope.shape[0] > 1 else 0,
        rope.stride(1),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return oq, ok


def try_rotate_qk(q: torch.Tensor, k: torch.Tensor, rope: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fuse two half-split RoPE calls, or return None outside the supported path."""
    if not (
        HAS_TRITON
        and q.is_cuda
        and torch.version.hip is None
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and q.dtype == k.dtype == rope.dtype == torch.bfloat16
        and q.device == k.device == rope.device
        and q.ndim == 4
        and q.shape == k.shape
        and q.numel() > 0
        and q.is_contiguous()
        and k.is_contiguous()
        and q.shape[-1] == 128
        and rope.ndim == 3
        and rope.shape[0] in (1, q.shape[0])
        and rope.shape[1:] == (q.shape[1], q.shape[-1])
        and rope.stride(-1) == 1
    ):
        return None
    oq, ok = torch.empty_like(q), torch.empty_like(k)
    _, s, h, d = q.shape
    _rotate_qk_kernel[(triton.cdiv(s * h * d, 1024), 2, q.shape[0])](
        q,
        k,
        rope,
        oq,
        ok,
        s * h * d,
        h,
        d,
        rope.stride(0) if rope.shape[0] > 1 else 0,
        rope.stride(1),
        1024,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return oq, ok


def try_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor | None = None, shift: torch.Tensor | None = None
) -> torch.Tensor | None:
    """Use the matched reduction where supported, otherwise fuse only the tail."""
    if not (
        HAS_TRITON
        and x.is_cuda
        and torch.version.hip is None
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and x.dtype == weight.dtype == torch.bfloat16
        and x.device == weight.device
        and x.ndim >= 2
        and x.numel() > 0
        and weight.shape == (x.shape[-1],)
        and weight.is_contiguous()
    ):
        return None
    modulate = scale is not None or shift is not None
    if modulate and not (
        scale is not None
        and shift is not None
        and x.ndim == 3
        and scale.shape == shift.shape == (x.shape[0], x.shape[-1])
        and scale.dtype == shift.dtype == x.dtype
        and scale.device == shift.device == x.device
        and scale.stride(-1) == shift.stride(-1) == 1
    ):
        return None
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if (
        _MATCHED_REDUCTION
        and x.ndim in (3, 4)
        and x.stride(-1) == 1
        and x.shape[-1] in (128, 3072)
        and x.numel() // x.shape[-1] >= 16
    ):
        heads = x.shape[2] if x.ndim == 4 else 1
        _rms_full_kernel[(triton.cdiv(x.shape[1] * heads, 4), x.shape[0])](
            x,
            weight,
            output,
            scale if modulate else weight,
            shift if modulate else weight,
            x.shape[-1],
            x.shape[1],
            heads,
            x.stride(0),
            x.stride(1),
            x.stride(2) if x.ndim == 4 else 0,
            modulate,
            scale.stride(0) if modulate else 0,
            shift.stride(0) if modulate else 0,
            triton.next_power_of_2(x.shape[-1]),
            4,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return output
    xf = x.float().contiguous()
    inv = torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
    # Modulation is per batch row (x is [B, S, D]); otherwise treat x as one block.
    batch = x.shape[0] if modulate else 1
    per_batch = x.numel() // batch
    _rms_tail_kernel[(triton.cdiv(per_batch, 1024), batch)](
        xf,
        inv,
        weight,
        output,
        scale if modulate else weight,
        shift if modulate else weight,
        per_batch,
        x.shape[-1],
        1024,
        modulate,
        scale.stride(0) if modulate else 0,
        shift.stride(0) if modulate else 0,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


def try_situ(hidden: torch.Tensor, gate: torch.Tensor) -> torch.Tensor | None:
    """Fuse SiTU with BF16 rounding; near-zero underflow may differ from eager."""
    if not (
        HAS_TRITON
        and hidden.is_cuda
        and torch.version.hip is None
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and hidden.dtype == gate.dtype == torch.bfloat16
        and hidden.device == gate.device
        and hidden.ndim == 3
        and hidden.shape == gate.shape
        and hidden.numel() > 0
        and hidden.stride() == gate.stride()
        and hidden.stride(-1) == 1
        and (hidden.shape[0] == 1 or hidden.stride(0) == hidden.shape[1] * hidden.stride(1))
    ):
        return None
    output = torch.empty(hidden.shape, dtype=hidden.dtype, device=hidden.device)
    _situ_kernel[(triton.cdiv(hidden.numel(), 1024),)](
        hidden,
        gate,
        output,
        hidden.numel(),
        hidden.shape[-1],
        hidden.stride(-2),
        1024,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
