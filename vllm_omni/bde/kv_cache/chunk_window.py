# SPDX-License-Identifier: Apache-2.0
"""Chunk-granularity sliding-window KV spec + manager for the BDE engine.

A world-model keeps only the last ``W`` chunks of KV resident. That is a sliding
window whose unit is a *chunk*, expressed as a ``SlidingWindowSpec`` subclass with
``sliding_window = window_chunks * chunk_size`` plus a manager that evicts at
chunk boundaries. Memory policy, refcounting, and ``null_block`` replacement stay
in vLLM's ``BlockPool`` — only the token-skip math is overridden here.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.kv_cache_spec_registry import register_kv_cache_spec


def chunk_window_skipped_tokens(
    num_computed_tokens: int,
    *,
    chunk_size: int,
    sliding_window: int,
    sink_chunks: int,
    reset_at_boundary: bool,
) -> int:
    """Tokens outside the resident chunk window, snapped to a chunk boundary.

    Pure function so the eviction policy is unit-testable without constructing a
    manager. Two strategies:

    - ``reset_at_boundary`` (DreamZero): at each chunk boundary everything past
      the sink is dropped.
    - otherwise (VGGT-style sliding replace): keep the last ``window`` tokens
      (plus the sink); the skip count snaps down to a chunk boundary so a chunk
      is never half-evicted.
    """
    sink = sink_chunks * chunk_size
    if reset_at_boundary:
        completed = (num_computed_tokens // chunk_size) * chunk_size
        return max(0, completed - sink)
    skipped = max(0, num_computed_tokens - sliding_window - sink)
    return (skipped // chunk_size) * chunk_size


class ChunkWindowManager(SlidingWindowManager):
    """``SlidingWindowManager`` that evicts at chunk boundaries.

    ``self.sliding_window`` is set by the base ``__init__``; the chunk fields are
    read from ``self.kv_cache_spec`` (a :class:`ChunkWindowSpec`).
    """

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        spec = self.kv_cache_spec
        return chunk_window_skipped_tokens(
            num_computed_tokens,
            chunk_size=spec.chunk_size,
            sliding_window=self.sliding_window,
            sink_chunks=spec.sink_chunks,
            reset_at_boundary=spec.reset_at_boundary,
        )


# Register so KVCacheManager resolves ChunkWindowSpec to ChunkWindowManager.
# Dispatch walks the spec's MRO, so without explicit registration the subclass
# would silently fall back to the parent SlidingWindowManager (override ignored).
# uniform_type_base_spec=None => its own KV cache group.
@register_kv_cache_spec(manager_class=ChunkWindowManager, uniform_type_base_spec=None)
@dataclass(frozen=True, kw_only=True)
class ChunkWindowSpec(SlidingWindowSpec):
    # sliding_window (inherited) MUST equal window_chunks * chunk_size.
    chunk_size: int
    window_chunks: int
    sink_chunks: int = 0
    reset_at_boundary: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.sliding_window != self.window_chunks * self.chunk_size:
            raise ValueError(
                "ChunkWindowSpec.sliding_window must equal "
                f"window_chunks * chunk_size ({self.window_chunks} * "
                f"{self.chunk_size}), got {self.sliding_window}"
            )
