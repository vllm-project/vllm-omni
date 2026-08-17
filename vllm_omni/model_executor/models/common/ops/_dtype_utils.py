# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""dtype helpers shared by the fused normalization operators."""

import torch


def group_norm_output_dtype(x: torch.Tensor) -> torch.dtype:
    """Return the dtype eager ``F.group_norm`` would produce for ``x``.

    ``group_norm`` sits on autocast's fp32 cast policy: under an autocast
    region it upcasts its inputs and returns fp32 *regardless* of the input
    dtype, while outside autocast it simply preserves the input dtype.

    Fused kernels replacing a ``GroupNorm`` must follow the same rule.
    Allocating the output with ``torch.empty_like(x)`` instead would silently
    downcast the activation to fp16/bf16 inside autocast regions, which is a
    behavioural change the caller never asked for. HunyuanImage3's VAE hits
    exactly this case: it holds fp32 weights and runs under fp16 autocast, so
    eager keeps the post-norm activation in fp32.
    """
    if torch.is_autocast_enabled(x.device.type):
        return torch.float32
    return x.dtype


__all__ = ["group_norm_output_dtype"]
