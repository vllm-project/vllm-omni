# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""310P diffusion attention: route SDPA through ``npu_prompt_flash_attention``.

The 310P op library has no working ``aclnnFlashAttentionScore`` /
``scaled_dot_product_attention`` implementation for NPU tensors: both the
default SDPA kernel and ``npu_fusion_attention`` fail with ``ERR01007``.
``npu_prompt_flash_attention`` does run on 310P, including GQA and arbitrary
sequence lengths, but only for non-causal attention and without a mask.

This module installs a process-wide SDPA replacement for NPU tensors:

* non-causal, unmasked, reduced-precision calls go to prompt flash attention;
* causal or masked calls (the Qwen3-VL text tower) use a chunked
  matmul/softmax implementation that keeps the same numerics.

Every diffusion call site uses the ``[B, H, S, D]`` layout, which is exactly what
prompt flash attention consumes. Calls that are not NPU 4D tensors are passed
through untouched.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False
_ORIGINAL_SDPA: Callable[..., torch.Tensor] | None = None
# Query rows per attention chunk. Bounds the [B, H, chunk, S] score buffer.
_QUERY_CHUNK = 256

# Exact power-of-two pre-scales in front of prompt flash attention.
#
# The 310P kernel has two FP16 range cliffs, and both of them turn whole query
# rows into inf/NaN on data that is perfectly representable in the output:
#
# * QK^T is accumulated in an FP16 workspace. A raw dot product above 65504
#   saturates (measured on Atlas 300I Duo: 65000 is exact, 70000 returns inf).
#   Diffusion attention heads, with RMSNorm'd q/k of dimension 128, reach
#   ~59000 raw at their largest, close enough to the cliff that one TP rank
#   overflowed on the last blocks. Dividing q and k by 2^4 is exact in FP16
#   and multiplies `scale_value` by 2^8, so the logits are unchanged while the
#   workspace keeps a 256x margin.
# * The P/V product is also range limited: on a captured layer-48 call, 145
#   query rows came back inf even though the softmax weights summed to one and
#   the fp64 result peaked at 237. Halving V was already enough to fix it;
#   scaling V by 2^-4 and rescaling the output by 2^4 is exact, costs nothing
#   measurable, and leaves 16x more room.
#
# The exactness above is per-value (powers of two only move the exponent), but
# the kernel is only empirically bounded, so every call is still checked and
# recomputed in fp32 by `chunked_attention` if anything slipped through.
_PFA_QK_SCALE = 2.0**4
_PFA_VALUE_SCALE = 2.0**4
# Fallback attempt with a much deeper value pre-scale before paying for fp32.
# Not deeper than 2^8: below that the small V entries go subnormal in FP16.
_PFA_VALUE_RETRY_SCALE = 2.0**8


class PFANonFiniteError(RuntimeError):
    """The 310P kernel returned inf/NaN for a call it should have handled."""


def prompt_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Non-causal ``[B, H, S, D]`` attention through the 310P PFA kernel."""
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    return torch_npu_pfa(query, key, value, scale)


def _pfa_once(query, key, value, scale, value_scale):
    import torch_npu

    out = torch_npu.npu_prompt_flash_attention(
        query / _PFA_QK_SCALE,
        key / _PFA_QK_SCALE,
        value / value_scale,
        num_heads=query.shape[1],
        num_key_value_heads=key.shape[1],
        input_layout="BNSD",
        scale_value=scale * _PFA_QK_SCALE**2,
    )
    return out * value_scale


def torch_npu_pfa(query, key, value, scale):
    """PFA with the 310P pre-scales, retried once, then validated.

    Raises ``PFANonFiniteError`` when the kernel overflows even at the deepest
    pre-scale, which lets ``_sdpa_310p`` fall back to the fp32 implementation.
    """
    out = _pfa_once(query, key, value, scale, _PFA_VALUE_SCALE)
    if bool(torch.isfinite(out).all()):
        return out
    out = _pfa_once(query, key, value, scale, _PFA_VALUE_RETRY_SCALE)
    if bool(torch.isfinite(out).all()):
        return out
    raise PFANonFiniteError(f"non-finite prompt flash attention output (Sq={query.shape[2]}, Skv={key.shape[2]})")


def _expand_gqa(key: torch.Tensor, value: torch.Tensor, q_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    kv_heads = key.shape[1]
    if kv_heads == q_heads:
        return key, value
    if kv_heads == 0 or q_heads % kv_heads:
        raise ValueError(f"GQA requires q_heads % kv_heads == 0, got {q_heads} and {kv_heads}")
    repeat = q_heads // kv_heads
    return key.repeat_interleave(repeat, dim=1), value.repeat_interleave(repeat, dim=1)


def _mask_4d(attn_mask: torch.Tensor) -> torch.Tensor:
    """Normalize an SDPA mask (2D/3D/4D) to ``[B, H or 1, Sq, Sk]``."""
    if attn_mask.dim() == 2:
        return attn_mask[None, None]
    if attn_mask.dim() == 3:
        return attn_mask[:, None]
    return attn_mask


def chunked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
    causal: bool,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked/causal fallback that mirrors ``F.scaled_dot_product_attention``.

    ``attn_mask`` follows PyTorch semantics: a bool tensor where ``True``
    means the key participates, or an additive float tensor.
    """
    query, key, value = (t.contiguous() for t in (query, key, value))
    key, value = _expand_gqa(key, value, query.shape[1])
    batch, heads, seq_q, _ = query.shape
    seq_k = key.shape[2]
    out = torch.empty_like(query)
    neg_inf = torch.finfo(torch.float32).min
    for start in range(0, seq_q, _QUERY_CHUNK):
        stop = min(start + _QUERY_CHUNK, seq_q)
        q_chunk = query[:, :, start:stop]
        logits = torch.matmul(q_chunk.float(), key.float().transpose(-1, -2)) * scale
        if causal:
            rows = torch.arange(start, stop, device=query.device).view(1, 1, -1, 1)
            cols = torch.arange(seq_k, device=query.device).view(1, 1, 1, -1)
            logits = logits.masked_fill(cols > rows, neg_inf)
        if attn_mask is not None:
            mask = _mask_4d(attn_mask)
            # SDPA accepts a mask that broadcasts over the query axis, e.g. a
            # key-padding mask shaped [B, 1, 1, Sk]. Slicing that singleton
            # axis with the chunk bounds would leave every chunk but the first
            # with zero query rows.
            if mask.shape[2] != 1:
                mask = mask[:, :, start:stop]
            if attn_mask.dtype == torch.bool:
                logits = logits.masked_fill(~mask, neg_inf)
            else:
                logits = logits + mask.float()
        probs = torch.softmax(logits, dim=-1).to(query.dtype)
        out[:, :, start:stop] = torch.matmul(probs, value)
    return out


def _sdpa_310p(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    **kwargs,
) -> torch.Tensor:
    if query.device.type != "npu" or query.dim() != 4 or dropout_p != 0.0:
        if _ORIGINAL_SDPA is None:  # pragma: no cover - set by apply_attention_patches()
            raise RuntimeError("310P attention patch has no original SDPA to fall back to")
        return _ORIGINAL_SDPA(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
            **kwargs,
        )
    softmax_scale = float(scale) if scale is not None else float(query.shape[-1]) ** -0.5
    can_use_pfa = (
        attn_mask is None
        and not is_causal
        and query.dtype in (torch.float16, torch.bfloat16)
        and key.dtype == query.dtype
        and value.dtype == query.dtype
    )
    if can_use_pfa:
        try:
            return torch_npu_pfa(query, key, value, softmax_scale)
        except PFANonFiniteError:
            logger.warning_once(
                "310P prompt flash attention returned non-finite output; "
                "recomputing those calls with the fp32 fallback."
            )
        except Exception as exc:  # pragma: no cover - kernel/SoC dependent
            logger.warning_once(
                "310P prompt flash attention declined (%s: %s); using the chunked fallback.",
                type(exc).__name__,
                exc,
            )
    if enable_gqa:
        key, value = _expand_gqa(key, value, query.shape[1])
    return chunked_attention(query, key, value, scale=softmax_scale, causal=is_causal, attn_mask=attn_mask)


def apply_attention_patches() -> None:
    global _PATCHED, _ORIGINAL_SDPA
    if _PATCHED:
        return
    _ORIGINAL_SDPA = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _sdpa_310p
    # ``torch.nn.functional`` is the same module object for both spellings.
    torch.nn.functional.scaled_dot_product_attention = _sdpa_310p
    _PATCHED = True
    logger.info("310P diffusion attention: SDPA routed through npu_prompt_flash_attention")
