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
    # Persistent KV tokens materialized per paged cache block.
    chunk_size: int = 0
    # Resident window in chunks. ``None`` means full attention (no eviction).
    window_chunks: int | None = None
    # Protected leading chunks (attention sink); never evicted.
    sink_chunks: int = 0
    # Boundary reset vs. sliding replacement.
    reset_at_boundary: bool = False
    # Fraction of free device memory budgeted for the AR-Diffusion KV pool.
    gpu_memory_fraction: float = 0.1
    # Maximum number of concurrently-tracked AR-Diffusion sessions. Each live
    # session owns pool blocks (two CFG adapters), so the runner LRU-evicts past
    # this cap to bound pool ownership under session-id churn. Generic default;
    # a model whose pipeline caps its own session map at a different value can
    # override this in ``ar_diffusion_kv_config`` to keep the two maps in step.
    max_sessions: int = 64
    # When CUDA graph / torch.compile is on (not enforce_eager), pre-capture the
    # DiT graphs for every window-fill shape at load time via a synthetic rollout,
    # so the serving run is fast from the first chunk. No effect when eager.
    warmup_cudagraph: bool = True
    # Also capture the post-window-boundary (reset-cycle) forward during warm-up.
    warmup_capture_reset: bool = False

    @property
    def sliding_window(self) -> int | None:
        """Window size in tokens, or ``None`` for full attention."""
        if self.window_chunks is None:
            return None
        return self.window_chunks * self.chunk_size
