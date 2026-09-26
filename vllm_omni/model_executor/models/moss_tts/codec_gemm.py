# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline-selected codec FFN kernels. No tuning or synchronization in forward.

MOSS_CODEC_GEMM_CONFIG points to the JSON produced by benchmark_moss_codec_kernels.
Absent an exact shape entry, retain PyTorch. Epilogues preserve BF16 boundaries.
"""

import json
import os

import torch
from vllm.triton_utils import tl, triton

libdevice = tl.extra.cuda.libdevice

_path = os.getenv("MOSS_CODEC_GEMM_CONFIG")
CONFIG = {}
if _path:
    with open(_path) as config_file:
        CONFIG = json.load(config_file)
FUSE = os.getenv("MOSS_CODEC_FFN_FUSION", "1") == "1"


@triton.jit
def _mm(
    a,
    w,
    p,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    bm: tl.constexpr,
    bn: tl.constexpr,
    bk: tl.constexpr,
    split: tl.constexpr,
    mode: tl.constexpr,
    scale,
    residual,
):
    rows = tl.program_id(0) * bm + tl.arange(0, bm)
    cols = tl.program_id(1) * bn + tl.arange(0, bn)
    kk = tl.program_id(2) * bk + tl.arange(0, bk)
    acc = tl.zeros((bm, bn), tl.float32)
    for block in range(tl.cdiv(k, bk * split)):
        ki = kk + block * bk * split
        av = tl.load(a + rows[:, None] * k + ki[None, :], (rows[:, None] < m) & (ki[None, :] < k), 0)
        wv = tl.load(w + cols[None, :] * k + ki[:, None], (cols[None, :] < n) & (ki[:, None] < k), 0)
        acc += tl.dot(av, wv)
    if split == 1:
        val = acc.to(tl.bfloat16).to(tl.float32)
        if mode == 1:
            val = 0.5 * val * (1.0 + libdevice.erf(val * 0.7071067811865476))
        elif mode == 2:
            scale = tl.load(scale + cols, cols < n, 0).to(tl.float32)
            residual = tl.load(
                residual + rows[:, None] * n + cols[None, :], (rows[:, None] < m) & (cols[None, :] < n), 0
            ).to(tl.float32)
            val = (val * scale[None, :]).to(tl.bfloat16).to(tl.float32) + residual
        tl.store(p + rows[:, None] * n + cols[None, :], val, (rows[:, None] < m) & (cols[None, :] < n))
    else:
        tl.store(
            p + tl.program_id(2) * m * n + rows[:, None] * n + cols[None, :],
            acc,
            (rows[:, None] < m) & (cols[None, :] < n),
        )


@triton.jit
def _finish(
    p,
    out,
    scale,
    residual,
    size: tl.constexpr,
    n: tl.constexpr,
    split: tl.constexpr,
    mode: tl.constexpr,
    block: tl.constexpr,
):
    ix = tl.program_id(0) * block + tl.arange(0, block)
    value = tl.full((block,), 0, tl.float32)
    for part in range(split):
        value += tl.load(p + part * size + ix, ix < size, 0)
    value = value.to(tl.bfloat16).to(tl.float32)
    if mode == 1:
        value = 0.5 * value * (1.0 + libdevice.erf(value * 0.7071067811865476))
    elif mode == 2:
        scale = tl.load(scale + ix % n).to(tl.float32)
        residual = tl.load(residual + ix, ix < size, 0).to(tl.float32)
        value = (value * scale).to(tl.bfloat16).to(tl.float32) + residual
    tl.store(out + ix, value, ix < size)


@torch.library.custom_op("moss_codec::ffn_gemm", mutates_args=())
def ffn_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    residual: torch.Tensor,
    mode: int,
    bm: int,
    bn: int,
    bk: int,
    split: int,
) -> torch.Tensor:
    n, k = weight.shape
    m = x.numel() // k
    out = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
    partial = out if split == 1 else torch.empty((split, m, n), device=x.device, dtype=torch.float32)
    _mm[(triton.cdiv(m, bm), triton.cdiv(n, bn), split)](
        x,
        weight,
        partial,
        m,
        n,
        k,
        bm,
        bn,
        bk,
        split,
        mode,
        scale,
        residual,
        num_warps=4,
        num_stages=3,
        enable_fp_fusion=False,
    )
    if split > 1:
        _finish[(triton.cdiv(m * n, 256),)](
            partial, out, scale, residual, m * n, n, split, mode, 256, enable_fp_fusion=False
        )
    return out


@ffn_gemm.register_fake
def _(x, weight, scale, residual, mode, bm, bn, bk, split):
    return torch.empty((*x.shape[:-1], weight.shape[0]), device=x.device, dtype=x.dtype)


@torch.library.custom_op("moss_codec::selected_linear", mutates_args=())
def selected_linear(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, residual: torch.Tensor, mode: int
) -> torch.Tensor:
    """Dispatch inside the opaque op so batch/frame dimensions remain symbolic."""
    key = f"{x.numel() // x.shape[-1]},{weight.shape[0]},{weight.shape[1]},{mode}"
    config = CONFIG.get(key)
    supported = (
        x.is_cuda
        and x.dtype == weight.dtype == scale.dtype == residual.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.is_contiguous()
        and scale.is_contiguous()
        and residual.is_contiguous()
    )
    if config is not None and supported:
        return ffn_gemm(x, weight, scale, residual, mode, *config)
    value = torch.nn.functional.linear(x, weight)
    if mode == 1:
        return torch.nn.functional.gelu(value)
    if mode == 2:
        if supported:
            out = torch.empty_like(value)
            _finish[(triton.cdiv(value.numel(), 256),)](
                value, out, scale, residual, value.numel(), weight.shape[0], 1, 2, 256, enable_fp_fusion=False
            )
            return out
        return residual + value * scale
    return value


@selected_linear.register_fake
def _(x, weight, scale, residual, mode):
    return torch.empty((*x.shape[:-1], weight.shape[0]), device=x.device, dtype=x.dtype)
