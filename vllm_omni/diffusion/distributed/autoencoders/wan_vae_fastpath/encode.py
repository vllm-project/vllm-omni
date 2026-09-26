# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Untiled Wan encoding with chunk-local patchification and linear output assembly."""

from __future__ import annotations

import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import patchify


def can_encode_frames(vae, x: torch.Tensor) -> bool:
    """Select only the supported inference schedule; other inputs use the parent."""
    return (
        not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and x.ndim == 5
        and x.numel() > 0
        and x.shape[1] == 3
        and (x.shape[2] - 1) % 4 == 0
        and x.shape[3] % vae.config.scale_factor_spatial == 0
        and x.shape[4] % vae.config.scale_factor_spatial == 0
        and vae.config.patch_size == 2
        and vae.config.scale_factor_temporal == 4
    )


def encode_frames(vae, x: torch.Tensor) -> torch.Tensor:
    """Return posterior parameters; public encode still constructs the distribution.

    The caller dispatches tiling using raw pixel dimensions. ``quant_conv`` is
    applied once to the assembled encoder features, matching untiled Diffusers.
    """
    vae.clear_cache()
    try:
        count = 1 + (x.shape[2] - 1) // 4
        output = None
        offset = 0
        for index in range(count):
            start, end = (0, 1) if index == 0 else (1 + 4 * (index - 1), 1 + 4 * index)
            chunk = patchify(x[:, :, start:end], patch_size=vae.config.patch_size)
            vae._enc_conv_idx = [0]
            chunk = vae.encoder(chunk, feat_cache=vae._enc_feat_map, feat_idx=vae._enc_conv_idx)
            if output is None:
                # Supported encoders produce exactly one latent frame per chunk.
                output = torch.empty(
                    (chunk.shape[0], chunk.shape[1], count, chunk.shape[3], chunk.shape[4]),
                    dtype=chunk.dtype,
                    device=chunk.device,
                    memory_format=(
                        torch.channels_last_3d
                        if chunk.is_contiguous(memory_format=torch.channels_last_3d)
                        else torch.contiguous_format
                    ),
                )
            if chunk.shape[2] != 1 or offset >= output.shape[2]:
                raise RuntimeError(f"Wan encoder chunk {index} produced {chunk.shape[2]} frames; expected one")
            output[:, :, offset : offset + 1].copy_(chunk)
            offset += 1
        if output is None or offset != output.shape[2]:
            raise RuntimeError("Wan encoder did not fill the expected temporal output")
        return vae.quant_conv(output)
    finally:
        vae.clear_cache()
