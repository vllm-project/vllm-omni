# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""NPU GroupNorm fusion with explicit spatial-collective context.

Independent VAE tiles normalize locally; spatial shards share statistics.
"""

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.hunyuan_image3.layers.native.vae_context import VAE_SPATIAL_CONTEXT


def global_group_norm(x: torch.Tensor, norm: torch.nn.GroupNorm, activation: str = "swish") -> torch.Tensor:
    if activation not in ("swish", "identity"):
        raise ValueError(f"Unsupported GroupNorm activation: {activation}")
    context = VAE_SPATIAL_CONTEXT.get()
    if context is None:
        raise RuntimeError("Global VAE normalization requires an active spatial context")
    from vllm_omni.platforms.npu.layers.dist_group_norm import dist_group_norm

    return dist_group_norm(
        x,
        norm.num_groups,
        norm.weight,
        norm.bias,
        norm.eps,
        activation=activation,
        device_group=context.group,
        layout="NCTHW",
        communication_guard=context.communication_guard,
    )


def vae_group_norm_silu(x: torch.Tensor, norm: torch.nn.GroupNorm) -> torch.Tensor:
    return vae_group_norm_for_conv(x, norm)


def vae_group_norm_for_conv(x: torch.Tensor, norm: torch.nn.GroupNorm, activation: str = "swish") -> torch.Tensor:
    """Use distributed fusion only inside an explicit spatial VAE context."""
    if activation not in ("swish", "identity"):
        raise ValueError(f"Unsupported GroupNorm activation: {activation}")
    if VAE_SPATIAL_CONTEXT.get() is not None:
        return global_group_norm(x, norm, activation)
    # Preserve conditional-image encoding numerics on the independent-tile path.
    y = norm(x)
    return F.silu(y) if activation == "swish" else y
