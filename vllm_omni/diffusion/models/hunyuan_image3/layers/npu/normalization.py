# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""NPU GroupNorm fusion with explicit spatial-collective context.

Independent VAE tiles normalize locally; spatial shards share statistics.
"""

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.hunyuan_image3.layers.npu.vae_context import VAE_SPATIAL_CONTEXT


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
    )


def vae_group_norm_silu(x: torch.Tensor, norm: torch.nn.GroupNorm) -> torch.Tensor:
    return vae_group_norm_for_conv(x, norm)


def vae_group_norm_for_conv(x: torch.Tensor, norm: torch.nn.GroupNorm, activation: str = "swish") -> torch.Tensor:
    """Select the NPU fusion by capability; retain native dtype/autograd fallback."""
    if activation not in ("swish", "identity"):
        raise ValueError(f"Unsupported GroupNorm activation: {activation}")
    if VAE_SPATIAL_CONTEXT.get() is not None:
        return global_group_norm(x, norm, activation)
    eligible = (
        x.device.type in ("npu", "privateuseone")
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.ndim == 5
        and norm.num_groups == 32
        and norm.weight is not None
        and norm.bias is not None
        and x.is_contiguous()
        and 16 * 1024 * 1024 <= x.numel() < 2**31
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
    )
    if eligible:
        from vllm_omni.platforms.npu.layers.dist_group_norm import dist_group_norm

        # Like eager GroupNorm, NPU autocast promotes normalization to FP32.
        value = x.float() if torch.is_autocast_enabled("npu") else x
        return dist_group_norm(
            value,
            norm.num_groups,
            norm.weight,
            norm.bias,
            norm.eps,
            activation=activation,
            local_only=True,
            layout="NCTHW",
        )
    y = norm(x)
    return F.silu(y) if activation == "swish" else y
