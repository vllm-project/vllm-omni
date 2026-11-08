# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Replace VAE GroupNorm with AITER GroupNorm after model initialization."""

import torch.nn as nn
from vllm.logger import init_logger

logger = init_logger(__name__)


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


def apply_aiter_groupnorm(model: nn.Module, model_class_name: str | None) -> None:
    if hasattr(model, "vae"):
        # AITER GroupNorm does not preserve the PyTorch autocast behavior required
        # by Hunyuan Image 3.0. Keep PyTorch GroupNorm until the fix is released:
        # https://github.com/ROCm/aiter/issues/4780
        # https://github.com/ROCm/aiter/pull/4779
        if model_class_name == "HunyuanImage3ForCausalMM":
            return

        try:
            from vllm._aiter_ops import is_aiter_found_and_supported

            if is_aiter_found_and_supported() and _replace_groupnorm_with_aiter(model.vae):
                logger.info("AITER GroupNorm is enabled for VAE.")
        except Exception:
            logger.warning("Failed to apply AITER GroupNorm to VAE.")
