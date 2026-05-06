from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class TokenizationConfig:
    """Compression ratios for estimating token counts before GPU encoding.

    Shared between the dataloader (packing estimation) and the preprocessor
    (actual encoding) to keep them consistent.

    All fields are derived from the ``VAEProtocol`` instance at runtime
    so that a single source of truth (the VAE) drives spatial compression,
    temporal compression, chunk size and latent dimensionality.
    """

    patch_size: int = 32
    vae_temporal_compression_factor: int = 4
    max_text_seq_len: int = 512
    vae_temporal_chunk_size: int = 16
    visual_latent_dim: int = 4
    tokens_per_image_tile: int | None = None

    @property
    def image_compression(self) -> tuple[int, int]:
        return (self.patch_size, self.patch_size)

    @property
    def video_compression(self) -> tuple[int, int, int]:
        return (self.vae_temporal_compression_factor, self.patch_size, self.patch_size)

    def estimate_image_tokens(self, h: int, w: int) -> int:
        if self.tokens_per_image_tile is not None:
            return self.tokens_per_image_tile
        hr, wr = self.image_compression
        return (h // hr) * (w // wr)

    def estimate_video_tokens(self, t: int, h: int, w: int) -> int:
        tr, hr, wr = self.video_compression
        # T=1 mirrors the House VAE: maybe_replicate_single_frame tiles the
        # single frame up to the temporal downsampling factor, producing one
        # latent frame.  A naive t // tr would say 0.
        t_latent = 1 if t == 1 else t // tr
        return t_latent * (h // hr) * (w // wr)

    def estimate_text_tokens(self, text: str) -> int:
        """Word-based approximation, clamped to ``max_text_seq_len``.

        TODO(mik): this is a temporary proxy until text is loaded untokenized. Remove later.
        """
        return min(len(text.split()), self.max_text_seq_len)

    def align_to_chunk_size(self, t: int) -> int:
        # Match the House VAE contract: only T=1 or T=k*chunk_size is
        # valid. T=1 is kept as-is (the VAE replicates it internally);
        # 1 < T < chunk_size cannot be aligned and is aligned to a single frame;
        # longer videos are floored to a multiple of chunk_size.
        if t < self.vae_temporal_chunk_size:
            return 1
        return (t // self.vae_temporal_chunk_size) * self.vae_temporal_chunk_size
