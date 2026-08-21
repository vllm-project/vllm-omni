# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Join a latent-producing realtime session to incremental decode.

An AR-Diffusion session returns latents. Turning those into delivered frames
is what the benchmark actually has to time, for two reasons:

* **Time to first frame, not first chunk.** A session's first latent chunk is
  not something a viewer can see. The gate this RFC proposes is TTFF, so the
  decode has to be inside the measured path.
* **The generate/decode split is the quantity overlapping them can recover.**
  With the two serialized, ``decode_share`` is the headroom; once they overlap
  it is what has to be shown shrinking. Without the split measured first,
  overlap work has no baseline to beat.

The wrapper also reports the delivered frame count from the decoder rather
than from a configured constant, so the harness measures the decoder's
temporal geometry instead of assuming it -- which matters because that
geometry is not declared anywhere the runtime can read.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


class LatentSession(Protocol):
    """A realtime session whose ticks produce latents."""

    async def next_chunk(self) -> Any: ...

    async def close(self) -> None: ...


class ChunkDecoder(Protocol):
    """The streaming-decode surface this wrapper needs."""

    def new_decode_state(self, session_id: str) -> Any: ...

    def decode_chunk(self, latent: Any, state: Any) -> Any: ...

    def release(self, state: Any) -> None: ...


@dataclass
class StageTiming:
    """Per-chunk split of one tick, filled in by :class:`DecodingSession`."""

    generate_s: float | None = None
    decode_s: float | None = None
    frames: int | None = None
    resident_decoder_bytes: int | None = None


@dataclass
class DecodingSession:
    """Wraps a latent session so each tick returns delivered frames.

    ``latent_of`` extracts the latent tensor from whatever the inner session
    returns, keeping this free of any engine's output type.
    """

    inner: LatentSession
    decoder: ChunkDecoder
    session_id: str
    latent_of: Callable[[Any], Any] = lambda output: output
    clock: Callable[[], float] = time.perf_counter
    last: StageTiming = field(default_factory=StageTiming)

    def __post_init__(self) -> None:
        self._state = self.decoder.new_decode_state(self.session_id)

    @property
    def decode_state(self) -> Any:
        return self._state

    async def next_chunk(self) -> Any:
        t0 = self.clock()
        output = await self.inner.next_chunk()
        t1 = self.clock()
        frames = self.decoder.decode_chunk(self.latent_of(output), self._state)
        t2 = self.clock()

        nbytes = getattr(self._state, "nbytes", None)
        self.last = StageTiming(
            generate_s=t1 - t0,
            decode_s=t2 - t1,
            frames=int(frames.shape[2]) if hasattr(frames, "shape") and len(frames.shape) >= 3 else None,
            resident_decoder_bytes=nbytes() if callable(nbytes) else None,
        )
        return frames

    async def close(self) -> None:
        # Decoder state and session state are released together: the cache has
        # no recompute source, so a session that ends must not leave it behind.
        try:
            self.decoder.release(self._state)
        finally:
            await self.inner.close()
