# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ming-Image adapter for the shared Z-Image transformer."""

from __future__ import annotations

import torch

from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.diffusion.models.z_image.z_image_transformer import ZImageTransformer2DModel


class MingImageTransformer2DModel(ZImageTransformer2DModel):
    """Runtime adapter for vendor DiffusionTransformer checkpoints.

    This adapter reuses the shared Z-Image DiT and injects Ming-specific request conditions.
    """

    def forward(
        self,
        x: list[torch.Tensor],
        t,
        cap_feats: list[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
    ):
        generated_frames = [item.shape[1] for item in x]
        ref_x = None
        cap_feats_2 = None
        if is_forward_context_available():
            context = get_forward_context()
            if context.cfg_branch is not None:
                branch = 0 if context.cfg_branch == "positive" else 1
                batch_size = len(x)
                if context.ref_latent is not None and context.ref_latent.shape[0] == batch_size * 2:
                    context_ref = context.ref_latent.chunk(2, dim=0)[branch]
                else:
                    context_ref = context.ref_latent
                if context.direct_condition is not None and context.direct_condition.shape[0] == batch_size * 2:
                    context_direct = context.direct_condition.chunk(2, dim=0)[branch]
                else:
                    context_direct = context.direct_condition
            else:
                context_ref = context.ref_latent
                context_direct = context.direct_condition
            if context_ref is not None:
                ref_x = [item.to(device=x[0].device, dtype=x[0].dtype) for item in context_ref.unbind(dim=0)]
            if context_direct is not None:
                cap_feats_2 = [item.to(device=x[0].device, dtype=x[0].dtype) for item in context_direct.unbind(dim=0)]

        output, metadata = super().forward(
            x,
            t,
            cap_feats,
            patch_size=patch_size,
            f_patch_size=f_patch_size,
            ref_x=ref_x,
            cap_feats_2=cap_feats_2,
        )
        output = [item[:, :frames] for item, frames in zip(output, generated_frames)]
        return output, metadata


__all__ = ["MingImageTransformer2DModel"]
