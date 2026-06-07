# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.

"""Ming-specific subclass of ZImageTransformer2DModel that supports ``ref_x``.

Ming's img2img path concatenates a VAE-encoded reference latent along the
frame axis before patchification, then drops the reference portion from the
unpatchified output. This is a surgical override — everything else (attention,
RoPE, final layer) stays on the parent implementation.

"""

from __future__ import annotations

import torch

from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.diffusion.models.z_image.z_image_transformer import ZImageTransformer2DModel


class MingZImageTransformer2DModel(ZImageTransformer2DModel):
    """ZImage DiT with Ming's reference-latent support."""

    def forward(
        self,
        x: list[torch.Tensor],
        t,
        cap_feats: list[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
    ):
        ref_latent = get_forward_context().ref_latent if is_forward_context_available() else None
        if ref_latent is not None:
            per_item = ref_latent[0].unsqueeze(1).to(dtype=x[0].dtype, device=x[0].device)  # [C, 1, H, W]
            x = [torch.cat([img, per_item], dim=1) for img in x]
        return super().forward(x, t, cap_feats, patch_size=patch_size, f_patch_size=f_patch_size)

    def unpatchify(
        self,
        x: list[torch.Tensor],
        size: list[tuple],
        patch_size,
        f_patch_size,
    ) -> list[torch.Tensor]:
        out = super().unpatchify(x, size, patch_size, f_patch_size)
        # No-op when F==1 (pure t2i); drops the reference-frame prediction when F==2.
        return [t[:, :1, :, :] for t in out]


__all__ = ["MingZImageTransformer2DModel"]
