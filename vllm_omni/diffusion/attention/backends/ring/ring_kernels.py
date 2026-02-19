# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Jiarui Fang.
# Adapted from https://github.com/feifeibear/long-context-attention

import math

import torch

from .ring_globals import (
    HAS_AITER,
    HAS_FA3,
    HAS_FLASH_ATTN,
    HAS_FLASHINFER,
    fa3_fwd_func,
)

_scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_flash_attention
_scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention

try:
    import torch_musa  # noqa: F401

    _scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_attention_flash_musa
    _scaled_dot_product_efficient_attention = None
except ModuleNotFoundError:
    pass

if HAS_AITER:
    from aiter import flash_attn_func as flash_attn_func_aiter

if HAS_FLASH_ATTN:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward

if HAS_FLASHINFER:
    from flashinfer.prefill import single_prefill_with_kv_cache

    _LOG2_E = math.log2(math.e)


def pytorch_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p=0.0,
    softmax_scale=None,
    causal=True,
    window_size=(-1, -1),
    softcap=None,
    alibi_slopes=None,
    return_softmax=False,
    op_type="efficient",
):
    assert op_type in ["flash", "efficient"], f"Invalid op_type: {op_type}"
    """
    q shape (bs, seqlen, nhead, hs)
    k shape (bs, seqlen, nhead, hs)
    v shape (bs, seqlen, nhead, hs)
    """
    # Fallback logic: Flash Attention does not support float32.
    # If op_type is 'flash' but dtype is float32, force 'efficient'.
    if op_type == "flash" and q.dtype == torch.float32:
        op_type = "efficient"

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    if op_type == "flash":
        out, lse = _scaled_dot_product_flash_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )[:2]
    elif op_type == "efficient":
        out, lse = _scaled_dot_product_efficient_attention(
            q,
            k,
            v,
            attn_bias=None,
            compute_log_sumexp=True,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )[:2]
    else:
        raise ValueError(f"Invalid op_type: {op_type}")

    out = out.transpose(1, 2)
    lse = lse.to(q.dtype)

    return out, lse


def flash_attn_forward(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=None,
    alibi_slopes=None,
    return_softmax=False,
):
    assert HAS_FLASH_ATTN, "FlashAttention is not available"
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if flash_attn.__version__ < "2.6.3":
        block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax,
        )
    else:
        block_out, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax,
        )
    return block_out, block_lse


def fa3_forward(q, k, v, dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes, return_softmax):
    """FA3 forward pass for inference.

    FA3 supports Ampere, Ada, and Hopper GPUs. Dropout is ignored since FA3 is inference-only.
    Uses low-level API (_flash_attn_forward) which always returns softmax_lse,
    required for Ring Attention's correct accumulation.
    """
    assert HAS_FA3, "FA3 is not available"
    assert fa3_fwd_func is not None, "FA3 low-level API (fa3_fwd_func) not available"

    # Low-level API always returns (out, softmax_lse, S_dmask, rng_state)
    out, softmax_lse, *_ = fa3_fwd_func(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0] if window_size else -1,
        window_size_right=window_size[1] if window_size else -1,
        softcap=softcap if softcap else 0.0,
    )

    return out, softmax_lse


# Legacy alias for backward compatibility
flash_attn3_func_forward = fa3_forward


def flash_attn_forward_aiter(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=None,
    alibi_slopes=None,
    return_softmax=False,
):
    assert HAS_AITER, "Aiter is not available"
    block_out, block_lse = flash_attn_func_aiter(
        q,
        k,
        v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        return_lse=True,
    )

    return block_out, block_lse


def flashinfer_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float | None = None,
    alibi_slopes: torch.Tensor | None = None,
    return_softmax: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert HAS_FLASHINFER, "FlashInfer is not available"
    if q.ndim == 4:
        if q.shape[0] > 1:
            raise ValueError("batch size > 1 is not supported")
        out, lse = single_prefill_with_kv_cache(
            q[0],
            k[0],
            v[0],
            sm_scale=softmax_scale,
            causal=causal,
            logits_soft_cap=softcap,
            window_left=window_size[0],
            return_lse=True,
        )
        lse = lse.transpose(0, 1)
        out, lse = out.unsqueeze(0), lse.unsqueeze(0)
    elif q.ndim == 3:
        out, lse = single_prefill_with_kv_cache(
            q,
            k,
            v,
            sm_scale=softmax_scale,
            causal=causal,
            logits_soft_cap=softcap,
            window_left=window_size[0],
            return_lse=True,
        )
        lse = lse.transpose(0, 1)
    else:
        raise ValueError(f"Invalid input shape: {q.shape}")
    lse = lse / _LOG2_E
    return out, lse
