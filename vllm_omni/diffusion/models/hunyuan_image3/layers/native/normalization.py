# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Native GroupNorm and activation for VAE convolution inputs."""

import torch
import torch.nn.functional as F
from torch import nn


def vae_group_norm_for_conv(x: torch.Tensor, norm: nn.GroupNorm, activation: str = "swish") -> torch.Tensor:
    y = norm(x)
    if activation == "swish":
        return F.silu(y)
    if activation == "identity":
        return y
    raise ValueError(f"Unsupported GroupNorm activation: {activation}")
