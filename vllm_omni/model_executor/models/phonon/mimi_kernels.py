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
) -> torch.Tensor:
    """Stride-one causal convolution with alias-safe history update.

    Inputs are [batch, channels, time]. History is [slots, channels,
    dilation * (kernel_size - 1)]. No tensor retained by convolution is
    overwritten when the new history is installed.
    """
    history_length = history_pool.shape[-1]
    combined = torch.cat((history_pool[slots], inputs), dim=-1)
    output = F.conv1d(combined, weight, bias, dilation=dilation, groups=groups)
    if history_length:
        history_pool[slots] = combined[..., -history_length:]
    return output
