# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunked Wan decode with a preallocated output buffer."""

from __future__ import annotations

import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify


def decode_frames(vae, z: torch.Tensor) -> torch.Tensor:
    """``AutoencoderKLWan._decode``'s frame loop, writing every chunk straight into the result.

    Upstream grows the result with ``torch.cat([out, out_], 2)`` on every chunk
    (quadratic copying) and then materializes ``unpatchify`` and ``clamp`` as two
    more full-size copies, so three or four copies of the decoded video are live
    at once. Here the final ``[B, C, T, H, W]`` buffer is allocated after the
    first chunk and each chunk is unpatchified and clamped directly into its
    slot. Every step is a permutation or an elementwise clamp, so the values are
    identical to upstream. Tiling dispatch is the caller's responsibility.
    """
    vae.clear_cache()
    x = vae.post_quant_conv(z)
    patch_size = vae.config.patch_size
    num_frames = x.shape[2]
    # Each temporal upsampler doubles the frames of every chunk after the first
    # (the first chunk drops the extra leading frames), so steady-state chunks
    # produce ``2 ** num_temporal_upsamplers`` frames each.
    frames_per_chunk = 2 ** sum(bool(flag) for flag in getattr(vae.decoder, "temporal_upsample", ()))

    output: torch.Tensor | None = None
    offset = 0
    for index in range(num_frames):
        vae._conv_idx = [0]
        chunk = vae.decoder(
            x[:, :, index : index + 1],
            feat_cache=vae._feat_map,
            feat_idx=vae._conv_idx,
            first_chunk=index == 0,
        )
        if patch_size is not None:
            chunk = unpatchify(chunk, patch_size=patch_size)
        if output is None:
            total_frames = chunk.shape[2] + (num_frames - 1) * frames_per_chunk
            output = chunk.new_empty((chunk.shape[0], chunk.shape[1], total_frames, chunk.shape[3], chunk.shape[4]))
        end = offset + chunk.shape[2]
        if end > output.shape[2]:
            raise RuntimeError(
                f"Wan decoder produced more frames than expected: chunk {index} ends at {end}, "
                f"allocated {output.shape[2]}"
            )
        torch.clamp(chunk, min=-1.0, max=1.0, out=output[:, :, offset:end])
        offset = end

    if output is None or offset != output.shape[2]:
        expected = None if output is None else output.shape[2]
        raise RuntimeError(f"Wan decoder produced {offset} frames, expected {expected}")
    vae.clear_cache()
    return output


__all__ = ["decode_frames"]
