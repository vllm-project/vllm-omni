# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Device-independent reference operations for Phonon's streaming state."""

import torch
import torch.nn.functional as F


def ring_attention_step(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    slots: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Append one position and attend within the causal window.

    Q/K/V are [batch, heads, dim], pools [slots, window, heads, dim].
    Slots must be distinct and positions nonnegative. The caller owns reset
    and position advancement, shared across all decoder layers.
    """
    window = key_pool.shape[1]
    write_indices = positions.remainder(window)
    key_pool[slots, write_indices] = key
    value_pool[slots, write_indices] = value
    physical_indices = torch.arange(window, device=positions.device)
    ages = (positions[:, None] - physical_indices).remainder(window)
    valid = ages <= positions[:, None]
    keys = key_pool[slots].transpose(1, 2)
    values = value_pool[slots].transpose(1, 2)
    scores = (query.unsqueeze(2) * keys).sum(-1) * query.shape[-1] ** -0.5
    scores = scores.masked_fill(~valid[:, None], float("-inf"))
    probabilities = scores.float().softmax(-1).to(values.dtype)
    return (probabilities.unsqueeze(-1) * values).sum(2)


def causal_conv_step(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    history_pool: torch.Tensor,
    slots: torch.Tensor,
    *,
    dilation: int = 1,
    groups: int = 1,
    scratch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stride-one causal convolution with alias-safe history update.

    Inputs are [batch, channels, time]. History is [slots, channels,
    dilation * (kernel_size - 1)]. No tensor retained by convolution is
    overwritten when the new history is installed.
    """
    history_length = history_pool.shape[-1]
    if scratch is None:
        combined = torch.cat((history_pool[slots], inputs), dim=-1)
    else:
        combined = scratch[: inputs.shape[0]]
        if history_length:
            combined[..., :history_length].copy_(history_pool.index_select(0, slots))
        combined[..., history_length:].copy_(inputs)
    output = F.conv1d(combined, weight, bias, dilation=dilation, groups=groups)
    if history_length:
        history_pool[slots] = combined[..., -history_length:]
    return output


def ring_attention_chunk(query, key, value, key_pool, value_pool, slots, positions):
    """Attend a complete chunk against the physical ring and current keys."""
    if query.is_cuda:
        from .mimi_attention import ring_attention_cuda

        return ring_attention_cuda(query, key, value, key_pool, value_pool, slots, positions)
    frames = query.shape[1]
    window = key_pool.shape[1]
    offsets = torch.arange(frames, device=positions.device)
    physical = torch.arange(window, device=positions.device)
    old_positions = positions[:, None] - 1 - (positions[:, None] - 1 - physical).remainder(window)
    current_positions = positions[:, None] + offsets
    key_positions = torch.cat((old_positions, current_positions), -1)
    delta = current_positions[:, :, None] - key_positions[:, None, :]
    mask = (key_positions[:, None, :] >= 0) & (delta >= 0) & (delta < window)
    keys = torch.cat((key_pool[slots], key), 1)
    values = torch.cat((value_pool[slots], value), 1)
    output = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        keys.transpose(1, 2),
        values.transpose(1, 2),
        attn_mask=mask[:, None],
    ).transpose(1, 2)
    retained = min(frames, window)
    indices = current_positions[:, -retained:].remainder(window)
    key_pool[slots[:, None], indices] = key[:, -retained:]
    value_pool[slots[:, None], indices] = value[:, -retained:]
    return output
