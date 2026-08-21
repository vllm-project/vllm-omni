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

import asyncio
import collections
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
    """Per-chunk split of one tick, filled in by the session wrappers."""

    generate_s: float | None = None
    decode_s: float | None = None
    frames: int | None = None
    resident_decoder_bytes: int | None = None
    overlap_s: float | None = None
    outstanding_generations: int | None = None


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


@dataclass
class OverlappedDecodingSession:
    """Decode chunk ``N`` while chunk ``N + 1`` is already being generated.

    Serialized, a tick costs ``generate + decode``. The generate call is a
    request to the worker rather than local compute, so the decode of the
    chunk just returned can run while the next one is in flight, and a tick
    costs ``max(generate, decode)`` instead.

    The session keeps at most ``lookahead`` generations outstanding. That bound
    is the backpressure: a consumer that stops taking chunks stops generation
    with it, rather than letting it run ahead into memory nothing is reading.
    ``outstanding_generations`` reports how many were in flight, so the bound
    is observable rather than asserted in a comment.

    Note what this is not. Because the caller drives ``next_chunk``, nothing
    ever *waits* on the bound from inside this class -- backpressure here is a
    ceiling on lookahead, not a measured stall. Timing a stall would need
    generation to run as its own producer task, which is a larger change and
    is not attempted here.

    ``overlap_s`` is how much generate time was hidden behind decode, which is
    what "VAE/DiT overlap efficiency" has to be computed from.
    """

    inner: LatentSession
    decoder: ChunkDecoder
    session_id: str
    latent_of: Callable[[Any], Any] = lambda output: output
    clock: Callable[[], float] = time.perf_counter
    lookahead: int = 1
    last: StageTiming = field(default_factory=StageTiming)

    def __post_init__(self) -> None:
        if self.lookahead < 1:
            raise ValueError("lookahead must be at least 1.")
        self._state = self.decoder.new_decode_state(self.session_id)
        self._pending: collections.deque[tuple[asyncio.Task, float]] = collections.deque()
        self._closed = False

    @property
    def decode_state(self) -> Any:
        return self._state

    def _fill_lookahead(self) -> None:
        """Keep the prefetch topped up to the bound, and never past it."""
        while not self._closed and len(self._pending) < self.lookahead:
            self._pending.append((asyncio.ensure_future(self.inner.next_chunk()), self.clock()))

    async def next_chunk(self) -> Any:
        t0 = self.clock()
        # A prefetch issued during the previous tick has been running while the
        # caller was away, so only the part still outstanding is charged here.
        self._fill_lookahead()
        task, started = self._pending.popleft()
        output = await task
        t1 = self.clock()
        generate_s = t1 - started
        overlap_s = max(0.0, generate_s - (t1 - t0))

        # Issue the next generation before decoding, so the two overlap.
        self._fill_lookahead()
        outstanding = len(self._pending)
        # ensure_future only *schedules* the task. Decode is synchronous and
        # blocks the event loop for its whole duration, so without handing
        # control back first the prefetch never starts and the overlap is
        # nominal: the request would not leave until decode had already
        # finished. One yield is enough -- the task only has to get as far as
        # issuing its request before it awaits.
        await asyncio.sleep(0)

        frames = self.decoder.decode_chunk(self.latent_of(output), self._state)
        t2 = self.clock()

        nbytes = getattr(self._state, "nbytes", None)
        self.last = StageTiming(
            generate_s=generate_s,
            decode_s=t2 - t1,
            frames=int(frames.shape[2]) if hasattr(frames, "shape") and len(frames.shape) >= 3 else None,
            resident_decoder_bytes=nbytes() if callable(nbytes) else None,
            overlap_s=overlap_s,
            outstanding_generations=outstanding,
        )
        return frames

    async def close(self) -> None:
        self._closed = True
        while self._pending:
            task, _ = self._pending.popleft()
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - shutdown must not mask close
                pass
        try:
            self.decoder.release(self._state)
        finally:
            await self.inner.close()
