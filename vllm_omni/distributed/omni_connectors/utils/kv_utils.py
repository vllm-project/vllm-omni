# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility helpers for KV cache manipulation."""

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

LayerKV = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


def normalize_layer_kv(
    layer_kv: LayerKV,
    *,
    req_id: str = "",
    layer_idx: int = -1,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Normalize one layer KV cache to a ``(key_blocks, value_blocks)`` tuple.

    In vLLM, different attention backends return paged-attention KV blocks
    with different layouts.  For example:

    * **FlashAttention** (``vllm/v1/attention/backends/flash_attn.py``)
      returns shape ``(2, num_blocks, block_size, num_kv_heads, head_size)``
      – the key/value dimension is at **dim 0**.
    * **FlashInfer** (``vllm/v1/attention/backends/flashinfer.py``)
      returns shape ``(num_blocks, 2, block_size, num_kv_heads, head_size)``
      – the key/value dimension is at **dim 1**.

    This utility handles both layouts (and the tuple case) so that
    downstream code can work with any backend.

    Supported layouts:

    * **Stacked tensor** ``[2, num_blocks, block_size, n_heads, head_dim]`` –
      dim-0 selects key / value.
    * **Stacked tensor** ``[num_blocks, 2, block_size, n_heads, head_dim]`` –
      dim-1 selects key / value.
    * **Tuple** ``(key_tensor, value_tensor)`` – returned as-is after
      validation.

    Args:
        layer_kv: The raw KV cache (tensor or tuple) for the layer.
        req_id: Request ID used only for diagnostic log messages.
        layer_idx: Layer index used only for diagnostic log messages.

    Returns:
        ``(key_blocks, value_blocks)`` if *layer_kv* is valid, ``None``
        otherwise.
    """
    if isinstance(layer_kv, torch.Tensor):
        if layer_kv.ndim >= 3 and layer_kv.shape[0] == 2:
            key_blocks = layer_kv[0]
            value_blocks = layer_kv[1]
        elif layer_kv.ndim >= 3 and layer_kv.shape[1] == 2:
            key_blocks = layer_kv[:, 0]
            value_blocks = layer_kv[:, 1]
        else:
            logger.warning(
                f"Layer {layer_idx} for request {req_id} has invalid stacked KV shape: "
                f"expected [2, ...] or [..., 2, ...] at dim 0/1, got {tuple(layer_kv.shape)}"
            )
            return None
    elif isinstance(layer_kv, tuple):
        if len(layer_kv) != 2:
            logger.warning(f"Layer {layer_idx} for request {req_id} has KV pair length {len(layer_kv)} (expected 2)")
            return None
        key_blocks, value_blocks = layer_kv
        if not isinstance(key_blocks, torch.Tensor) or not isinstance(value_blocks, torch.Tensor):
            logger.warning(f"Layer {layer_idx} for request {req_id} has non-tensor KV pair entries")
            return None
    else:
        logger.warning(f"Layer {layer_idx} for request {req_id} has unsupported KV type {type(layer_kv).__name__}")
        return None

    # ensure key/value blocks are at least 2D for block indexing
    if key_blocks.ndim < 2 or value_blocks.ndim < 2:
        logger.warning(
            f"Layer {layer_idx} for request {req_id} has invalid KV block shape: "
            f"got key={tuple(key_blocks.shape)} value={tuple(value_blocks.shape)}"
        )
        return None

    return key_blocks, value_blocks
