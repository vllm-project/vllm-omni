# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Adapted from transformers.models.mimi.modeling_mimi.MimiConv1d.forward.
# Copyright 2024 Kyutai, and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Based on Transformers 5.14.1 MimiConv1d.forward; parity also validated
# against Transformers 5.19.0. Recheck this override on Transformers upgrades.
# Remove it once upstream provides equivalent integer-only padding arithmetic.
"""Model-local, integer-only padding sizes for the Mimi reference encoder."""

from types import MethodType

import torch
from torch import nn
from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiConv1dPaddingCache


def _integer_padding_forward(
    self: MimiConv1d,
    hidden_states: torch.Tensor,
    padding_cache: MimiConv1dPaddingCache | None = None,
) -> torch.Tensor:
    # Mimi's tensor-buffer arithmetic simplifies to ceil(length / stride).
    # Read Conv1d's Python attributes to avoid device scalars and their .item()
    # synchronizations when the encoder has been moved to an accelerator.
    length = hidden_states.shape[-1]
    stride = self.conv.stride[0]
    kernel_size = (self.conv.kernel_size[0] - 1) * self.conv.dilation[0] + 1
    padding_total = kernel_size - stride
    extra_padding = ((length + stride - 1) // stride) * stride - length

    if not self.causal and padding_cache is not None:
        raise ValueError("`padding_cache` is not supported for non-causal convolutions.")

    if self.causal and padding_cache is not None:
        previous = padding_cache.update(hidden_states, self.layer_idx)
        hidden_states = torch.cat([previous, hidden_states], dim=2)
    elif self.causal:
        hidden_states = self._pad1d(hidden_states, (padding_total, extra_padding), mode=self.pad_mode)
    else:
        padding_right = padding_total // 2
        hidden_states = self._pad1d(
            hidden_states, (padding_total - padding_right, padding_right + extra_padding), mode=self.pad_mode
        )

    # Retain Mimi's short-input reflection handling, including zero-width pads.
    return self.conv(hidden_states)


def use_integer_padding(encoder: nn.Module) -> None:
    """Override only this encoder's convolutions, preserving checkpoint keys.

    Called once during model construction, before weights are loaded. Keeping
    the modules themselves preserves parameters, buffers and weight-norm state;
    the upstream MimiConv1d class and other model instances are untouched.
    """
    for module in encoder.modules():
        if isinstance(module, MimiConv1d):
            module.forward = MethodType(_integer_padding_forward, module)
