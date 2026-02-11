# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Workarounds for cuBLAS compatibility issues across GPU drivers.

Some cuBLAS versions reject the batched GEMM dispatched by the ``@``
operator inside ``RotaryEmbedding.forward``, raising
``CUBLAS_STATUS_INVALID_VALUE`` from ``cublasSgemmStridedBatched``.

Because the inner dimension (k) of the multiplication is always **1**,
the matmul is mathematically equivalent to element-wise broadcast
multiplication (``A * B``).  The patches in this module replace ``@``
with ``*``, which runs through CUDA element-wise kernels instead of
cuBLAS and is therefore immune to the driver bug.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Qwen2.5-VL  (4-D M-RoPE with 3 grids: temporal, height, width)
# ---------------------------------------------------------------------------


def patch_qwen25vl_rope_for_cublas_compat(model: nn.Module) -> None:
    """Patch every ``Qwen2_5_VLRotaryEmbedding`` in *model*."""
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
                # inv_freq: (dim//2,)
                # After unsqueeze: (1, 1, dim//2, 1)
                inv_freq_expanded = rope_mod.inv_freq[None, None, :, None].float()
                # position_ids: (3, bs, seq_len)  →  (3, bs, 1, seq_len)
                position_ids_expanded = position_ids[:, :, None, :].float()

                device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
                with torch.autocast(device_type=device_type, enabled=False):
                    # k == 1, so matmul ≡ broadcast multiply:
                    #   (1,1,dim//2,1) * (3,bs,1,seq_len) → (3,bs,dim//2,seq_len)
                    freqs = (inv_freq_expanded * position_ids_expanded).transpose(2, 3)
                    emb = torch.cat((freqs, freqs), dim=-1)
                    cos = emb.cos() * rope_mod.attention_scaling
                    sin = emb.sin() * rope_mod.attention_scaling

                return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

            return _safe_forward

        module.forward = _make_safe_forward(module)
        logger.info(
            "Patched %s – replaced matmul with broadcast multiply",
            type(module).__name__,
        )


# ---------------------------------------------------------------------------
# Qwen3  (standard 3-D RoPE, used by Qwen3Model / Qwen3ForCausalLM)
# ---------------------------------------------------------------------------


def patch_qwen3_rope_for_cublas_compat(model: nn.Module) -> None:
    """Patch every ``Qwen3RotaryEmbedding`` in *model*.

    Same cuBLAS issue as Qwen2.5-VL but with standard 3-D shapes::

        inv_freq_expanded  : (bs, dim//2, 1)
        position_ids_expanded : (bs, 1, seq_len)

    k == 1, so ``@`` is replaced with broadcast ``*``.
    """
    try:
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3RotaryEmbedding,
        )
    except ImportError:
        return

    for module in model.modules():
        if not isinstance(module, Qwen3RotaryEmbedding):
            continue

        def _make_safe_forward(rope_mod: Qwen3RotaryEmbedding):
            @torch.no_grad()
            def _safe_forward(x: torch.Tensor, position_ids: torch.Tensor):
                # inv_freq: (dim//2,)  →  (1, dim//2, 1)
                inv_freq_expanded = rope_mod.inv_freq[None, :, None].float().to(x.device)
                # position_ids: (bs, seq_len)  →  (bs, 1, seq_len)
                position_ids_expanded = position_ids[:, None, :].float()

                device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
                with torch.autocast(device_type=device_type, enabled=False):
                    # k == 1, so matmul ≡ broadcast multiply:
                    #   (1,dim//2,1) * (bs,1,seq_len) → (bs,dim//2,seq_len)
                    freqs = (inv_freq_expanded * position_ids_expanded).transpose(1, 2)
                    emb = torch.cat((freqs, freqs), dim=-1)
                    cos = emb.cos() * rope_mod.attention_scaling
                    sin = emb.sin() * rope_mod.attention_scaling

                return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

            return _safe_forward

        module.forward = _make_safe_forward(module)
        logger.info(
            "Patched %s – replaced matmul with broadcast multiply",
            type(module).__name__,
        )
