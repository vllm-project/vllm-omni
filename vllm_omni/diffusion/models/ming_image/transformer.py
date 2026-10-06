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
            batch_size = len(x)
            start = batch_size if context.cfg_branch == "negative" else 0
            reference = context.ref_latent
            direct = context.direct_condition
            if context.cfg_branch is not None:
                if reference is not None and reference.shape[0] == 2 * batch_size:
                    reference = reference[start : start + batch_size]
                if direct is not None and direct.shape[0] == 2 * batch_size:
                    direct = direct[start : start + batch_size]
            if reference is not None:
                ref_x = [item.to(device=x[0].device, dtype=x[0].dtype) for item in reference.unbind(dim=0)]
            if direct is not None:
                cap_feats_2 = [item.to(device=x[0].device, dtype=x[0].dtype) for item in direct.unbind(dim=0)]

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
