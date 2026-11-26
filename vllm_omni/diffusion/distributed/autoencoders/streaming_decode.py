# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Session-owned state and protocol for incremental diffusion VAE decode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch


@dataclass
class StreamingDecodeState:
    """One session's temporal decoder cache.

    ``feat_map`` holds one entry per causal convolution. Entries are tensors,
    ``None`` before that convolution has run, or a non-tensor sentinel the
    decoder uses for its first upsample step, so byte accounting must skip
    anything that is not a tensor.

    The cache has **no recompute source**: it is the compressed tail of every
    frame decoded so far and cannot be rebuilt from the prompt or the latents.
    Releasing it ends the stream; there is no partial eviction.
    """

    session_id: str
    feat_map: list[Any]
    conv_idx: list[int] = field(default_factory=lambda: [0])
    frames_decoded: int = 0
    chunks_decoded: int = 0

    def nbytes(self) -> int:
        """Resident bytes of the cache, skipping non-tensor entries."""
        return sum(t.numel() * t.element_size() for t in self.feat_map if torch.is_tensor(t))

    def nbytes_by_device(self) -> dict[str, int]:
        """Resident bytes grouped by device, for admission accounting."""
        totals: dict[str, int] = {}
        for tensor in self.feat_map:
            if torch.is_tensor(tensor):
                key = str(tensor.device)
                totals[key] = totals.get(key, 0) + tensor.numel() * tensor.element_size()
        return totals

    @property
    def started(self) -> bool:
        return self.frames_decoded > 0

    def release(self) -> None:
        """Drop the cache and return the session to its pre-stream state."""
        self.feat_map = [None] * len(self.feat_map)
        self.conv_idx = [0]
        self.frames_decoded = 0
        self.chunks_decoded = 0


@runtime_checkable
class SupportsStreamingDecode(Protocol):
    """A decoder that can emit a session's chunks incrementally.

    The lifecycle mirrors the AR session's: ``new_decode_state`` at session
    creation, ``decode_chunk`` per committed chunk, and ``release`` on reset or
    close, so decoder state and KV state are released together rather than
    through two independent lifecycles.
    """

    def new_decode_state(self, session_id: str) -> StreamingDecodeState:
        """Create empty decoder state for a new session."""
        ...

    def decode_chunk(
        self,
        latent: torch.Tensor,
        state: StreamingDecodeState,
        *,
        output_format: str = "video",
        produce_output: bool = True,
    ) -> torch.Tensor:
        """Decode one committed chunk, advancing ``state`` in place."""
        ...

    def declared_state_bytes(self, *, height: int, width: int, dtype: torch.dtype) -> int:
        """Upper bound on resident decoder bytes per session at this shape."""
        ...

    def release(self, state: StreamingDecodeState) -> None:
        """Drop ``state``'s cache and return it to its pre-stream state."""
        ...
