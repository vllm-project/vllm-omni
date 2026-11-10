# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Dense FP8 EagleFIA using MindIE-SD's coupled C8V16/V512 mode."""

import math

import torch


def eagle_fp8_fia(query, key, value, *, scale, kv_head_major=False):
    if query.ndim != 3 or key.ndim != 3 or key.shape != value.shape:
        raise ValueError("EagleFIA requires TND Q and matching rank-three K/V")
    kv_heads = key.shape[0 if kv_head_major else 1]
    kv_len = key.shape[1 if kv_head_major else 0]
    if query.shape[1] != kv_heads or query.shape[-1] != 128 or key.shape[-1] != 128:
        raise ValueError("LingBot EagleFIA requires matching heads and head_dim128")
    if min(query.shape[0], kv_len) <= 0:
        raise ValueError("EagleFIA requires nonempty Q/K")
    if any(x.dtype != torch.bfloat16 for x in (query, key, value)):
        raise ValueError("LingBot EagleFIA requires BF16 inputs")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("EagleFIA scale must be positive and finite")
    from mindiesd.layers.flash_attn.quant_flash_attn import quant_attention
    from mindiesd.quantization.mode import FP8FAMode

    q = query.transpose(0, 1).unsqueeze(0).contiguous()
    if kv_head_major:
        if not key.is_contiguous() or not value.is_contiguous():
            raise ValueError("EagleFIA head-major K/V must be contiguous")
        k, v = key.unsqueeze(0), value.unsqueeze(0)
    else:
        k, v = (x.transpose(0, 1).unsqueeze(0).contiguous() for x in (key, value))
    out = quant_attention(
        q,
        k,
        v,
        precision="fp8",
        layout="BNSD",
        scale=scale,
        fp8_fa_mode=FP8FAMode.C8V16_TILING512,
    )
    if out.shape != q.shape or out.dtype != query.dtype:
        raise RuntimeError("MindIE FP8 EagleFIA output contract mismatch")
    return out[0].permute(1, 0, 2).contiguous()
