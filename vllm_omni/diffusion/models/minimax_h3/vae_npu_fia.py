# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in NPU fused-infer attention for the MiniMax-H3 Video VAE decoder.

Use ``video_vae_fused_infer_attention`` in the model's additional config.
FIA requires nonempty, equal-shape BSND Q/K/V in FP16 or BF16 and unmasked,
non-causal inference. Other calls keep the original attention.
"""

from collections.abc import Mapping
from types import MethodType

import torch

from .ops.vae.dispatch import supports_h3_vae_npu_fia

_CONFIG_KEY = "video_vae_fused_infer_attention"
_STATS_LOG_INTERVAL = 1000


def resolve_video_vae_fused_infer_attention(additional_config: Mapping[str, object] | None) -> bool:
    """Read the MiniMax-H3 launch option without coercing non-boolean values."""
    enabled = (additional_config or {}).get(_CONFIG_KEY, False)
    if not isinstance(enabled, bool):
        raise ValueError(f"additional_config.{_CONFIG_KEY} must be a boolean")
    return enabled


def _shape_fallback_reason(query, key, value) -> str | None:
    if query.ndim != 4 or query.shape != key.shape or query.shape != value.shape:
        return "expected matching BSND Q/K/V shapes"
    if 0 in query.shape:
        return "empty Q/K/V"
    return None


def _fallback_reason(attn, query, key, value, pack_info):
    """Return why the fused path cannot run, or None when it can."""
    if attn.training:
        return "training mode"
    if any(t.requires_grad for t in (query, key, value)):
        return "gradient tracking"
    if query.device.type != "npu" or key.device != query.device or value.device != query.device:
        return "non-NPU or mismatched devices"
    if query.dtype not in (torch.float16, torch.bfloat16):
        return f"dtype outside the FP16/BF16 envelope ({query.dtype})"
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return "mismatched Q/K/V dtypes"
    shape_reason = _shape_fallback_reason(query, key, value)
    if shape_reason is not None:
        return shape_reason
    if pack_info.get("cu_seqlens") is not None:
        return "varlen attention"
    if pack_info.get("mask_mod") is not None:
        return "masked attention"
    if pack_info.get("block_sparse") is not None:
        return "block-sparse attention"
    return None


def _fused_infer_attention(query, key, value):
    import torch_npu

    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()
    hidden_states, _ = torch_npu.npu_fused_infer_attention_score(
        q,
        k,
        v,
        num_heads=q.shape[1],
        num_key_value_heads=k.shape[1],
        input_layout="BNSD",
        scale=q.shape[-1] ** -0.5,
        pre_tokens=2147483647,
        next_tokens=2147483647,
        sparse_mode=0,
        inner_precise=0,
        softmax_lse_flag=False,
    )
    return hidden_states.transpose(1, 2).nan_to_num(0.0)


def install_vae_npu_fia(decoder, device: torch.device, logger, *, enabled: bool) -> None:
    """Install the opted-in NPU attention path before the VAE stagers are built."""
    if not enabled or not supports_h3_vae_npu_fia(device) or decoder is None:
        return
    if getattr(decoder, "_h3_npu_fia_stats", None) is not None:
        return
    blocks = list(getattr(decoder, "transformer_blocks", []))
    if not blocks:
        return
    stats = {"calls": 0, "fallbacks": 0}
    seen_fallback_reasons = set()

    def _maybe_log_stats():
        if (stats["calls"] + stats["fallbacks"]) % _STATS_LOG_INTERVAL == 0:
            logger.info("[H3_VAE_NPU_FIA] calls=%d fallbacks=%d", stats["calls"], stats["fallbacks"])

    def wrap(original):
        @torch.compiler.disable
        def perform(attn, query, key, value, pack_info=None):
            pack_info = {} if pack_info is None else pack_info
            reason = _fallback_reason(attn, query, key, value, pack_info)
            if reason is not None:
                stats["fallbacks"] += 1
                if reason not in seen_fallback_reasons:
                    seen_fallback_reasons.add(reason)
                    logger.info("[H3_VAE_NPU_FIA] falling back to the original attention: %s", reason)
                _maybe_log_stats()
                return original(query, key, value, pack_info)
            out = _fused_infer_attention(query, key, value)
            stats["calls"] += 1
            if stats["calls"] == 1:
                logger.info(
                    "[H3_VAE_NPU_FIA] first fast-path call: shape=%s dtype=%s",
                    tuple(query.shape),
                    query.dtype,
                )
            _maybe_log_stats()
            return out

        return perform

    installed = 0
    for block in blocks:
        attn = getattr(block, "attn", None)
        if attn is None or not hasattr(attn, "perform_attention"):
            continue
        attn.perform_attention = MethodType(wrap(attn.perform_attention), attn)
        attn._h3_decoder_fused_infer_attention = True
        installed += 1
    if installed == 0:
        return
    decoder._h3_npu_fia_stats = stats
    logger.info("[H3_VAE_NPU_FIA] installed on %d blocks", installed)


__all__ = ["install_vae_npu_fia", "resolve_video_vae_fused_infer_attention"]
