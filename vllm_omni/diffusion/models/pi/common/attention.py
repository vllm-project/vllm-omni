# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared attention primitives for Pi-family action models."""

import torch
import torch.nn.functional as F

# Matches OpenPI's finite mask value exactly.
OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """Build token visibility from padding and autoregressive block markers.

    The cumulative block ids make tokens within a block bidirectional while
    preventing an earlier block from attending to a later one. Padding is then
    applied on both the query and key axes.

    Ref: openpi/models_pytorch/pi0_pytorch.py ``make_att_2d_masks``.
    """
    if att_masks.ndim != 2:
        raise ValueError(f"att_masks must be 2-D, got {att_masks.ndim}-D")
    if pad_masks.ndim != 2:
        raise ValueError(f"pad_masks must be 2-D, got {pad_masks.ndim}-D")

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def prepare_attention_masks_4d(att_2d_masks: torch.Tensor) -> torch.Tensor:
    """Convert bool masks to additive attention masks.

    The singleton dimension in ``(B, 1, query_len, key_len)`` broadcasts over
    attention heads. ``True`` becomes 0.0; ``False`` becomes OpenPI's exact
    finite negative mask value.

    Ref: OpenPI ``PI0Pytorch._prepare_attention_masks_4d``.
    """
    att_2d_masks_4d = att_2d_masks[:, None, :, :]
    return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for grouped-query attention.

    ``(B, num_kv_heads, S, D)`` becomes
    ``(B, num_kv_heads * n_rep, S, D)`` while preserving head-group order.
    Matches ``transformers.models.gemma.modeling_gemma.repeat_kv``.
    """
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def eager_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    num_kv_groups: int,
    scaling: float,
) -> torch.Tensor:
    """Compute ``softmax(Q K^T * scale + mask) V`` in eager mode.

    The mask is intentionally sliced to the actual key length. This lets one
    ``(B, 1, query_len, prefix_len + suffix_len)`` mask serve the prefix pass
    and the later suffix pass without rebuilding it. The softmax is evaluated
    in float32, then cast back to the query dtype for reference parity.
    """
    key_states = repeat_kv(key_states, num_kv_groups)
    value_states = repeat_kv(value_states, num_kv_groups)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    return torch.matmul(attn_weights, value_states)
