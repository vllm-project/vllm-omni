# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Native VAE entry points shared by non-NPU backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_hunyuan import DistributedAutoencoderKLHunyuan

VaeConv3d = nn.Conv3d


def vae_group_norm_for_conv(x: torch.Tensor, norm: nn.GroupNorm, activation: str = "swish") -> torch.Tensor:
    y = norm(x)
    if activation == "swish":
        return F.silu(y)
    if activation == "identity":
        return y
    raise ValueError(f"Unsupported GroupNorm activation: {activation}")


def spatial_vae_forward(
    vae: DistributedAutoencoderKLHunyuan, x: torch.Tensor, stage: Literal["encode", "decode"]
) -> torch.Tensor | None:
    return None
