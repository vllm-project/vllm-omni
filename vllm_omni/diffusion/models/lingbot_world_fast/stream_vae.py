"""Per-chunk streaming VAE encode wrapper around ``Wan2_1_VAE``."""

from __future__ import annotations

import torch
import torch.cuda.amp as amp

from .vae2_1 import Wan2_1_VAE


class StreamVAE:
    def __init__(self, vae: Wan2_1_VAE) -> None:
        self._vae = vae
        self._model = vae.model
        self._scale = vae.scale
        self._dtype = vae.dtype

    def reset(self) -> None:
        """Clear encoder feat_map cache. Call at the start of each new request."""
        self._model.clear_cache()

    def decode(self, zs):
        return self._vae.decode(zs)

    @property
    def dtype(self) -> torch.dtype:
        return self._vae.dtype

    @torch.no_grad()
    def init(self, frame: torch.Tensor) -> torch.Tensor:
        """Encode the single init pixel frame and return its latent.

        Caller keeps the latent for fresh starts (anchor encoding) or
        discards it for extension starts (just sets up the init bias).

        Args:
            frame: ``[C, 1, H, W]`` or ``[B, C, 1, H, W]``.
        Returns:
            ``[z_dim, 1, H, W]`` latent.
        """
        with amp.autocast(dtype=self._dtype):
            pixels = frame.unsqueeze(0) if frame.dim() == 4 else frame
            out = self._encode_group(pixels)
            mu = self._apply_conv1_and_normalize(out)
        return mu.float().squeeze(0)

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode ``4*N`` pixel frames using the preserved state.

        Args:
            pixels: ``[C, 4N, H, W]`` or ``[B, C, 4N, H, W]``.
        Returns:
            ``[z_dim, N, H, W]`` latents.
        """
        pixels = pixels.unsqueeze(0) if pixels.dim() == 4 else pixels
        T = pixels.shape[2]
        assert T % 4 == 0, f"StreamVAE.encode expects a multiple of 4 pixel frames, got {T}"
        N = T // 4
        with amp.autocast(dtype=self._dtype):
            outs = [self._encode_group(pixels[:, :, i * 4 : (i + 1) * 4]) for i in range(N)]
            out = torch.cat(outs, dim=2)
            mu = self._apply_conv1_and_normalize(out)
        return mu.float().squeeze(0)

    # ── internals ──────────────────────────────────────────────────────────

    def _encode_group(self, pixels: torch.Tensor) -> torch.Tensor:
        """One ``(1|4)``-frame encoder pass using the live ``_enc_feat_map``."""
        self._model._enc_conv_idx = [0]
        return self._model.encoder(
            pixels, feat_cache=self._model._enc_feat_map, feat_idx=self._model._enc_conv_idx
        )

    def _apply_conv1_and_normalize(self, out: torch.Tensor) -> torch.Tensor:
        mu, _ = self._model.conv1(out).chunk(2, dim=1)
        z = self._model.z_dim
        if isinstance(self._scale[0], torch.Tensor):
            mu = (mu - self._scale[0].view(1, z, 1, 1, 1)) * self._scale[1].view(1, z, 1, 1, 1)
        else:
            mu = (mu - self._scale[0]) * self._scale[1]
        return mu