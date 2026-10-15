# SPDX-License-Identifier: Apache-2.0
"""Session-owned incremental decode for the causal Wan video VAE."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify


def snapshot_feature_cache(cache: list[Any]) -> list[Any]:
    """Snapshot a Wan decoder cache list without losing non-tensor sentinels.

    Wan's temporal upsamplers use the string sentinel ``"Rep"`` alongside
    tensors and ``None``. Decoder layers replace list entries rather than
    mutating cached tensors in place, so a detached list copy is sufficient to
    isolate the committed session state from the next decode transaction.

    DreamZero's ``_vae_clone_feat_map`` looks interchangeable but is not: it
    snapshots the *encoder* cache and clones, because its chunked encode hands
    the same list back for in-place reuse. Cloning here would copy the whole
    decoder cache on every realtime tick to protect against a write the
    decoder never makes.
    """

    return [entry.detach() if isinstance(entry, torch.Tensor) else entry for entry in cache]


@dataclass(frozen=True, slots=True)
class WanStreamingDecodeResult:
    video: torch.Tensor
    feature_cache: list[Any]


def decode_wan_causal_chunk(
    vae,
    denormalized_latents: torch.Tensor,
    *,
    feature_cache: list[Any] | None,
    initialized: bool,
) -> WanStreamingDecodeResult:
    """Decode only new latent frames while preserving causal decoder state.

    Same per-frame loop as ``wan_spatial_shard.spatial_shard_decode`` and
    diffusers ``AutoencoderKLWan._decode``, and it cannot call either: both
    clear the feature cache on entry, which is exactly the state this must
    carry across ticks. The session therefore owns the cache, and
    ``first_chunk`` tracks the session rather than this call's frame index.
    """

    if denormalized_latents.ndim != 5 or denormalized_latents.shape[0] != 1:
        raise ValueError(
            f"Incremental Wan decode expects latents shaped [1,C,T,H,W], got {tuple(denormalized_latents.shape)}."
        )
    if denormalized_latents.shape[2] <= 0:
        raise ValueError("Incremental Wan decode requires at least one new latent frame.")
    if feature_cache is None:
        vae.clear_cache()
        raw_cache = getattr(vae, "_feat_map", None)
        if not isinstance(raw_cache, list):
            raise TypeError("Wan VAE clear_cache() did not expose the expected decoder _feat_map list.")
        feature_cache = snapshot_feature_cache(raw_cache)

    vae._feat_map = snapshot_feature_cache(feature_cache)
    try:
        with vae._execution_context() if hasattr(vae, "_execution_context") else nullcontext():
            projected = vae.post_quant_conv(denormalized_latents)
            decoded_parts: list[torch.Tensor] = []
            for latent_idx in range(projected.shape[2]):
                vae._conv_idx = [0]
                decoded = vae.decoder(
                    projected[:, :, latent_idx : latent_idx + 1],
                    feat_cache=vae._feat_map,
                    feat_idx=vae._conv_idx,
                    first_chunk=not initialized and latent_idx == 0,
                )
                decoded_parts.append(decoded)

            video = torch.cat(decoded_parts, dim=2)
            if vae.config.patch_size is not None:
                video = unpatchify(video, patch_size=vae.config.patch_size)
        committed_cache = snapshot_feature_cache(vae._feat_map)
    finally:
        # The state owns the committed cache; the shared VAE must not retain a
        # second session reference after success, reset, or decoder failure.
        vae.clear_cache()
    return WanStreamingDecodeResult(
        video=video,
        feature_cache=committed_cache,
    )


__all__ = [
    "WanStreamingDecodeResult",
    "decode_wan_causal_chunk",
    "snapshot_feature_cache",
]
