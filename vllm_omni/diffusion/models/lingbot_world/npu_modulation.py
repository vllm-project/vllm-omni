# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in LingBot NPU modulation kernels."""

import torch


def adaln(norm, hidden, scale, shift, tokens_per_frame):
    """Use the provider's fused LayerNorm and scale/shift with FP32 inputs."""
    if hidden.device.type != "npu" or hidden.dtype != torch.bfloat16:
        return None
    try:
        from mindiesd import layernorm_scale_shift
    except ImportError:
        return None
    x = hidden.float()
    if tokens_per_frame != 1:
        x = x.view(-1, tokens_per_frame, x.shape[-1])
        scale, shift = (value.reshape(-1, value.shape[-1]) for value in (scale, shift))
    else:
        scale, shift = scale.squeeze(2), shift.squeeze(2)
    return layernorm_scale_shift(norm, x, scale, shift, fused=True).reshape_as(hidden).to(hidden.dtype)
