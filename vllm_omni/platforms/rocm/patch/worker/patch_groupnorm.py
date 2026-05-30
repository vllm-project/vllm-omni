# SPDX-License-Identifier: Apache-2.0

"""Patch ``initialize_model`` to replace VAE GroupNorm with AITER GroupNorm on ROCm."""

import torch.nn as nn
from vllm.logger import init_logger

import vllm_omni.diffusion.registry as _registry_mod

logger = init_logger(__name__)

_original_initialize_model = _registry_mod.initialize_model


def _replace_groupnorm_with_aiter(vae: nn.Module) -> bool:
    from aiter.ops.groupnorm import GroupNorm as AiterGroupNorm

    targets = [
        (parent, name, child)
        for parent in vae.modules()
        for name, child in parent.named_children()
        if isinstance(child, nn.GroupNorm) and child.affine
    ]

    for parent, name, child in targets:
        new_group_norm = AiterGroupNorm(
            num_groups=child.num_groups,
            num_channels=child.num_channels,
            eps=child.eps,
            affine=True,
            device=child.weight.device,
            dtype=child.weight.dtype,
        )
        new_group_norm.weight = child.weight
        new_group_norm.bias = child.bias
        setattr(parent, name, new_group_norm)

    return len(targets) > 0


def _patched_initialize_model(od_config):
    model = _original_initialize_model(od_config)

    if hasattr(model, "vae"):
        try:
            from vllm._aiter_ops import is_aiter_found_and_supported

            if is_aiter_found_and_supported() and _replace_groupnorm_with_aiter(model.vae):
                logger.info("AITER GroupNorm is enabled for VAE.")
        except Exception:
            logger.warning("Failed to apply AITER GroupNorm to VAE.")

    return model


_registry_mod.initialize_model = _patched_initialize_model
