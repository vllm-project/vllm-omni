"""VAEProtocol and DummyVAE placeholder.

The ``VAEProtocol`` captures the minimal interface that the
preprocessors need from a VAE-like encoder:

* ``encode(x)`` — ``(B, C, T, H, W) → (B, D, Tl, Hl, Wl)``
* ``decode(z)`` — ``(B, D, Tl, Hl, Wl) → (B, C, T, H, W)``
* ``latent_dim`` — number of latent channels ``D``
* ``compression_modes`` — supported ``(Td, Hd, Wd)`` compression ratios
* ``set_compression_mode()`` — select the active compression mode
* ``temporal_chunk_size`` — raw frames per temporal chunk
* ``tokenization_config`` — ``TokenizationConfig`` for downstream consumers

Both ``DummyVAE`` (placeholder, conv-based) and the real
``TwoStageVAEInference`` satisfy this protocol.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, Sequence, runtime_checkable

import torch
from torch import nn

from .tokenization_config import TokenizationConfig


@runtime_checkable
class VAEProtocol(Protocol):
    """Encode raw visual input (image or video) into latent tokens.

    Input:  ``(B, C, T, H, W)`` — raw pixels, ``T=1`` for images.
    Output: ``(B, D, Tl, Hl, Wl)`` — latent embeddings.
    """

    @property
    def latent_dim(self) -> int:
        """Number of latent channels (D)."""
        ...

    @property
    def compression_modes(self) -> Sequence[tuple[int, int, int]]:
        """All supported downsampling compression factors ``(Td, Hd, Wd)``."""
        ...

    def set_compression_mode(self, mode: tuple[int, int, int]) -> None:
        """Select the default compression mode for subsequent ``encode`` calls."""
        ...

    @property
    def temporal_chunk_size(self) -> int:
        """Number of raw frames consumed per temporal chunk."""
        ...

    @property
    def tokenization_config(self) -> TokenizationConfig:
        """Compression config for token-count estimation and packing."""
        ...

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, C, T, H, W) → (B, D, Tl, Hl, Wl)``."""
        ...

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode ``(B, D, Tl, Hl, Wl) → (B, C, T, H, W)``."""
        ...


@dataclasses.dataclass
class DummyVAEConfig:
    """Config for the placeholder conv-based VAE encoder.

    All compression parameters are read from ``tokenization_config``,
    making the VAE the single source of truth for spatial/temporal
    compression ratios and latent dimensionality.
    """

    tokenization_config: TokenizationConfig = dataclasses.field(
        default_factory=TokenizationConfig
    )

    def make(self, device: torch.device) -> DummyVAE:
        """Build a ``DummyVAE`` on *device*."""
        return DummyVAE(self).to(device)


class DummyVAE(nn.Module):
    """Conv-based placeholder that follows the ``VAEProtocol``."""

    def __init__(self, cfg: DummyVAEConfig) -> None:
        super().__init__()
        self._cfg = cfg
        tc = cfg.tokenization_config
        self.conv = nn.Conv2d(
            3, tc.visual_latent_dim, kernel_size=tc.patch_size, stride=tc.patch_size
        )
        self.deconv = nn.ConvTranspose2d(
            tc.visual_latent_dim, 3, kernel_size=tc.patch_size, stride=tc.patch_size
        )
        self._temporal_compression = tc.vae_temporal_compression_factor

    @property
    def latent_dim(self) -> int:
        return self._cfg.tokenization_config.visual_latent_dim

    @property
    def compression_modes(self) -> Sequence[tuple[int, int, int]]:
        return (self._cfg.tokenization_config.video_compression,)

    def set_compression_mode(self, mode: tuple[int, int, int]) -> None:
        if mode not in self.compression_modes:
            raise ValueError(
                f"{mode!r} is not a supported compression mode; "
                f"available: {self.compression_modes}"
            )

    @property
    def temporal_chunk_size(self) -> int:
        return self._cfg.tokenization_config.vae_temporal_chunk_size

    @property
    def tokenization_config(self) -> TokenizationConfig:
        return self._cfg.tokenization_config

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, T, H, W) → (B, D, T', H', W')``."""
        b, c, t, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)  # (B*T, C, H, W)
        latents = self.conv(frames)  # (B*T, D, H', W')
        _, d, h2, w2 = latents.shape
        out: torch.Tensor = latents.reshape(b, t, d, h2, w2).permute(0, 2, 1, 3, 4)
        tc = self._temporal_compression
        if tc > 1 and t > 1:
            out = out[:, :, ::tc]
        return out

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """``(B, D, T', H', W') → (B, C, T, H, W)``."""
        b, d, t, h2, w2 = z.shape
        flat = z.permute(0, 2, 1, 3, 4).reshape(b * t, d, h2, w2)  # (B*T, D, H', W')
        pixels = self.deconv(flat)  # (B*T, C, H, W)
        _, c, h, w = pixels.shape
        out: torch.Tensor = pixels.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return out
