# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fused audio embedding reduction, preserving pad/clamp semantics."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _embed(
    c_ptr,
    w_ptr,
    o_ptr,
    cs0: tl.constexpr,
    cs1: tl.constexpr,
    nq: tl.constexpr,
    v: tl.constexpr,
    h: tl.constexpr,
    pad: tl.constexpr,
    bq: tl.constexpr,
    bd: tl.constexpr,
):
    row = tl.program_id(0)
    dims = tl.program_id(1) * bd + tl.arange(0, bd)
    quantizers = tl.arange(0, bq)
    codes = tl.load(c_ptr + row * cs0 + quantizers * cs1, quantizers < nq, pad)
    valid = (quantizers < nq) & (codes != pad)
    safe_codes = tl.minimum(tl.maximum(codes, 0), v - 1)
    values = tl.load(
        w_ptr + (quantizers[:, None] * v + safe_codes[:, None]) * h + dims[None, :],
        valid[:, None] & (dims[None, :] < h),
        0,
    ).to(tl.float32)
    total = tl.sum(values, 0)
    tl.store(o_ptr + row * h + dims, total, dims < h)


@torch.library.custom_op("vllm_omni::moss_audio_embed", mutates_args=())
def audio_embed(codes: torch.Tensor, weights: torch.Tensor, pad: int) -> torch.Tensor:
    t, nq = codes.shape
    assert weights.is_contiguous() and weights.shape[0] == nq
    assert codes.device == weights.device and codes.device.type == "cuda"
    out = torch.empty((t, weights.shape[2]), device=codes.device, dtype=weights.dtype)
    if t:
        _embed[(t, triton.cdiv(weights.shape[2], 256))](
            codes,
            weights,
            out,
            *codes.stride(),
            nq,
            weights.shape[1],
            weights.shape[2],
            pad,
            triton.next_power_of_2(nq),
            256,
            num_warps=4,
        )
    return out


@audio_embed.register_fake
def _(codes: torch.Tensor, weights: torch.Tensor, pad: int) -> torch.Tensor:
    return weights.new_empty((codes.shape[0], weights.shape[2]))
