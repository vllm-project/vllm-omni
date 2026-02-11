# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Workarounds for cuBLAS compatibility issues across GPU drivers."""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


def patch_qwen25vl_rope_for_cublas_compat(model: nn.Module) -> None:
    """Patch Qwen2.5-VL rotary embeddings to avoid cuBLAS stride-0 bug.

    ``Tensor.expand()`` produces stride-0 dimensions that share memory.
    Some cuBLAS versions / GPU drivers reject stride-0 in
    ``cublasSgemmStridedBatched``, raising ``CUBLAS_STATUS_INVALID_VALUE``.

    This replaces the ``forward`` of every ``Qwen2_5_VLRotaryEmbedding``
    inside *model* with a version that calls ``.contiguous()`` after
    ``expand()``, eliminating the stride-0 tensors.

    The patch is safe for all GPUs – ``.contiguous()`` is a no-op when
    the tensor is already contiguous, and the extra memory for the
    small ``inv_freq`` buffer is negligible.
    """
    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLRotaryEmbedding,
        )
    except ImportError:
        return

    for module in model.modules():
        if not isinstance(module, Qwen2_5_VLRotaryEmbedding):
            continue

        def _make_safe_forward(rope_mod: Qwen2_5_VLRotaryEmbedding):
            @torch.no_grad()
            def _safe_forward(x: torch.Tensor, position_ids: torch.Tensor):
                inv_freq_expanded = (
                    rope_mod.inv_freq[None, None, :, None]
                    .float()
                    .expand(3, position_ids.shape[1], -1, 1)
                    .contiguous()  # avoid stride-0 in cuBLAS
                )
                position_ids_expanded = (
                    position_ids[:, :, None, :].float().contiguous()
                )  # position_ids also comes from expand() with stride-0

                device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
                with torch.autocast(device_type=device_type, enabled=False):
                    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
                    emb = torch.cat((freqs, freqs), dim=-1)
                    cos = emb.cos() * rope_mod.attention_scaling
                    sin = emb.sin() * rope_mod.attention_scaling

                return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

            return _safe_forward

        module.forward = _make_safe_forward(module)
        logger.info(
            "Patched %s for cuBLAS stride-0 compatibility",
            type(module).__name__,
        )
