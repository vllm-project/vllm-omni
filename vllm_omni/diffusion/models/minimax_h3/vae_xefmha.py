# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""XeFMHA fast path for the MiniMax-H3 VAE decoder self-attention on XPU.

``install_vae_xefmha(decoder, device, logger)`` rebinds each decoder
block's ``attn.perform_attention`` to ``xpu_ops.flash_attn_varlen_func``
for calls inside the validated envelope: inference, single sequence,
non-causal, FP16, 32 heads x 64 head-dim, 64-byte-aligned strides.
Anything else falls back to the original implementation and logs the
reason once per reason. Controlled by ``H3_VAE_XEFMHA`` (default on).
Call it after ``install_h3_vae_optimizations()`` and before
``H3_VAE_COMPILE`` wraps ``block.forward``.
"""

import os
from types import MethodType

import torch

_STATS_LOG_INTERVAL = 1000


def install_vae_xefmha(decoder, device, logger):
    if os.environ.get("H3_VAE_XEFMHA", "1") != "1":
        return
    if device.type != "xpu" or decoder is None:
        # Not an error: every non-XPU platform takes this path on every VAE
        # load, so raising here would put an ERROR+traceback in their logs
        # for what is simply "this fast path does not apply".
        return
    if getattr(decoder, "_h3_xefmha_stats", None) is not None:
        return
    from vllm._xpu_ops import xpu_ops

    blocks = list(decoder.transformer_blocks)
    if not blocks or decoder.t_causal:
        raise ValueError("H3_VAE_XEFMHA requires nonempty, non-causal blocks")
    # Share one metadata entry across blocks; never cache QKV or outputs.
    metadata = {}
    stats = {"calls": 0, "fallbacks": 0}
    seen_fallback_reasons = set()

    def _maybe_log_stats():
        if (stats["calls"] + stats["fallbacks"]) % _STATS_LOG_INTERVAL == 0:
            logger.info("[H3_VAE_XEFMHA] calls=%d fallbacks=%d", stats["calls"], stats["fallbacks"])

    def _fallback_reason(attn, query, key, value, pack_info):
        if attn.training or attn.spatial_parallel or pack_info:
            return "unsupported mode (training, spatial-parallel or packed input)"
        if query.device.type != "xpu" or key.device != query.device or value.device != query.device:
            return "non-XPU or mismatched devices"
        if query.dtype != torch.float16 or key.dtype != query.dtype or value.dtype != query.dtype:
            return f"dtype outside the validated FP16 envelope ({query.dtype})"
        if (
            query.ndim != 4
            or query.shape != key.shape
            or query.shape != value.shape
            or query.shape[0] != 1
            or query.shape[1] <= 0
            or query.shape[2:] != (32, 64)
        ):
            return "shape outside the validated single-sequence 32-head x 64-dim envelope"
        return None

    def wrap(original):
        @torch.compiler.disable
        def perform(attn, query, key, value, pack_info=None):
            pack_info = {} if pack_info is None else pack_info
            reason = _fallback_reason(attn, query, key, value, pack_info)
            q = k = v = None
            if reason is None:
                batch, length, heads, dim = query.shape
                q, k, v = (x.flatten(0, 1) for x in (query, key, value))
                # Native API supports strided heads/rows; ensure its alignment contract.
                if any(
                    x.stride(-1) != 1
                    or x.data_ptr() % 64 != 0
                    or any(s * x.element_size() % 64 for s in x.stride()[:-1])
                    for x in (q, k, v)
                ):
                    reason = "strides break the kernel's 64-byte alignment contract"
            if reason is not None:
                stats["fallbacks"] += 1
                if reason not in seen_fallback_reasons:
                    seen_fallback_reasons.add(reason)
                    logger.info("[H3_VAE_XEFMHA] falling back to the original attention: %s", reason)
                _maybe_log_stats()
                return original(query, key, value, pack_info)
            cache_key = (query.device, batch, length)
            if metadata.get("key") != cache_key:
                metadata.clear()
                metadata["cu"] = torch.arange(0, (batch + 1) * length, length, dtype=torch.int32, device=query.device)
                metadata["key"] = cache_key
            cu = metadata["cu"]
            out = xpu_ops.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                max_seqlen_q=length,
                max_seqlen_k=length,
                softmax_scale=dim**-0.5,
                causal=False,
                dropout_p=0.0,
                return_softmax_lse=False,
            )
            stats["calls"] += 1
            if stats["calls"] == 1:
                logger.info(
                    "[H3_VAE_XEFMHA] first fast-path call: shape=%s dtype=%s strides=%s",
                    tuple(query.shape),
                    query.dtype,
                    [x.stride() for x in (q, k, v)],
                )
            _maybe_log_stats()
            return out.reshape(batch, length, heads, dim).nan_to_num(0.0)

        return perform

    for block in blocks:
        attn = block.attn
        attn.perform_attention = MethodType(wrap(attn.perform_attention), attn)
    decoder._h3_xefmha_stats = stats
    logger.info("[H3_VAE_XEFMHA] installed on %d blocks", len(blocks))
