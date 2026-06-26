# SPDX-License-Identifier: Apache-2.0
"""Configuration for AR-Diffusion engine-level KV cache management."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ARDiffusionKVConfig:
    """Settings for the AR-Diffusion paged KV cache.

    Disabled by default: when ``enable`` is False the AR-Diffusion engine behaves exactly
    like the base ``DiffusionEngine`` (no pool, no paged KV).
    """

    enable: bool = False
    # Persistent KV tokens materialized per AR chunk. For DreamZero this is
    # ``num_frame_per_block * frame_seqlen``.
    chunk_size: int = 0
    # Resident window in chunks. ``None`` means full attention (no eviction).
    window_chunks: int | None = None
    # Protected leading chunks (attention sink); never evicted.
    sink_chunks: int = 0
    # DreamZero-style window reset vs. VGGT-style sliding replace.
    reset_at_boundary: bool = False
    # Fraction of free device memory budgeted for the AR-Diffusion KV pool.
    gpu_memory_fraction: float = 0.1

    @property
    def sliding_window(self) -> int | None:
        """Window size in tokens, or ``None`` for full attention."""
        if self.window_chunks is None:
            return None
        return self.window_chunks * self.chunk_size
