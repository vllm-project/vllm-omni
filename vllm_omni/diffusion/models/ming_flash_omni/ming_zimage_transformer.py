# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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
            if ref_latent.dim() == 3:
                ref_latent = ref_latent.unsqueeze(0)
            if ref_latent.dim() != 4 or ref_latent.shape[0] != len(x):
                raise ValueError(
                    "Ming reference latent batch must match transformer batch: "
                    f"latents={tuple(ref_latent.shape)}, requests={len(x)}"
                )
            x = [
                torch.cat(
                    [img, ref_latent[i].unsqueeze(1).to(dtype=img.dtype, device=img.device)],
                    dim=1,
                )
                for i, img in enumerate(x)
            ]
        num_rows = len(x)
        od_config = get_forward_context().omni_diffusion_config if is_forward_context_available() else None
        capacity = max(1, int(getattr(od_config, "max_num_seqs", num_rows)))
        predictions = []
        for start in range(0, num_rows, capacity):
            batch_x = x[start : start + capacity]
            batch_cap_feats = cap_feats[start : start + capacity]
            batch_t = t[start : start + capacity]
            valid_rows = len(batch_x)
            # Keep the DiT's GEMM/attention shapes stable across admission and
            # retirement. Duplicate an existing row; padding owns no request,
            # RNG or scheduler state and its prediction is discarded below.
            padding = capacity - valid_rows
            if padding:
                batch_x = [*batch_x, *([batch_x[-1]] * padding)]
                batch_cap_feats = [*batch_cap_feats, *([batch_cap_feats[-1]] * padding)]
                batch_t = torch.cat([batch_t, batch_t[-1:].expand(padding)])
            result = super().forward(
                batch_x, batch_t, batch_cap_feats, patch_size=patch_size, f_patch_size=f_patch_size
            )
            predictions.extend(result[0][:valid_rows])
        return predictions, {}

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
