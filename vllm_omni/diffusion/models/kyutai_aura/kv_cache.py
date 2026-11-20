# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-owned rolling K/V history and RoPE positions for AURA's chunk-wise generation.

The history keeps, per layer, the keys and values of the first chunk (an identity anchor that is
never evicted) and of the latest ``window_chunks`` chunks, the current one included:

    [ chunk 0 | chunk r - window + 1 | ... | chunk r ]

Every denoising step of a chunk overwrites the current chunk's rows, so the history that the next
chunk sees holds the K/V of the last (noisy) step: AURA uses no extra clean pass. Keys are stored
before the rotary embedding; the model rotates the whole key set at every step with the positions
below, which lets AURA-1.3B freeze its temporal positions on long streams.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm_omni.diffusion.models.kyutai_aura.config import KyutaiAuraConfig


@dataclass(frozen=True)
class ChunkPositions:
    """Temporal RoPE positions of the current chunk (queries) and of the visible history (keys).

    Positions are latent-frame indices. The reference frame sits at ``ref_offset`` inside the
    training segment of the chunk's last frame. With ``max_chunk`` set, positions are computed as
    if the current chunk were ``min(chunk, max_chunk)`` (bounded re-indexing).
    """

    frames_per_chunk: int
    window_chunks: int
    segment_frames: int
    ref_offset: int
    max_chunk: int | None = None

    @classmethod
    def for_config(cls, cfg: KyutaiAuraConfig) -> ChunkPositions:
        """The position layout of one released checkpoint."""
        return cls(cfg.frames_per_chunk, cfg.window_chunks, cfg.segment_frames, cfg.ref_rope_offset, cfg.rope_max_chunk)

    def _rope_chunk(self, chunk: int) -> int:
        return chunk if self.max_chunk is None else min(chunk, self.max_chunk)

    def _frames(self, chunks: list[int]) -> list[int]:
        n = self.frames_per_chunk
        return [c * n + i for c in chunks for i in range(n)]

    def _ref_position(self, last_frame: int) -> int:
        return last_frame - last_frame % self.segment_frames + self.ref_offset

    def query_positions(self, chunk: int) -> list[int]:
        """Temporal positions of the chunk's latent frames, then of the reference frame."""
        frames = self._frames([self._rope_chunk(chunk)])
        return frames + [self._ref_position(frames[-1])]

    def key_positions(self, chunk: int) -> list[int]:
        """Temporal positions of the visible history (anchor, latest chunks), then of the reference frame."""
        r = self._rope_chunk(chunk)
        visible = [0] + list(range(max(1, r - self.window_chunks + 1), r + 1))
        frames = self._frames(visible)
        return frames + [self._ref_position(frames[-1])]


class RollingChunkKVCache:
    """Per-layer pre-RoPE keys and values of the anchor chunk plus the latest chunks."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        chunk_tokens: int,
        window_chunks: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.chunk_tokens = chunk_tokens
        self.num_slots = window_chunks + 1
        shape = (1, self.num_slots * chunk_tokens, num_heads, head_dim)
        self.keys = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self.values = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self.chunk = 0

    @classmethod
    def for_config(cls, cfg: KyutaiAuraConfig, chunk_tokens: int, device: torch.device) -> RollingChunkKVCache:
        """An empty history for one request on ``cfg``'s model; ``chunk_tokens`` tokens per chunk."""
        return cls(cfg.num_layers, cfg.num_heads, cfg.head_dim, chunk_tokens, cfg.window_chunks, device)

    def current_rows(self) -> tuple[int, int]:
        """Row range [start, end) of the current chunk; the visible history is rows [0, end)."""
        slot = min(self.chunk, self.num_slots - 1)
        return slot * self.chunk_tokens, (slot + 1) * self.chunk_tokens

    @torch.no_grad()
    def advance(self) -> None:
        """Close the current chunk. Once all slots are used, drop the oldest non-anchor chunk."""
        self.chunk += 1
        if self.chunk >= self.num_slots:
            n = self.chunk_tokens
            for t in (*self.keys, *self.values):
                t[:, n:-n] = t[:, 2 * n :].clone()

    def nbytes(self) -> int:
        """Memory held by the history, in bytes."""
        return sum(t.numel() * t.element_size() for t in (*self.keys, *self.values))
