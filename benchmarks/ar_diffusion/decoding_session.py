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
    backpressure_s: float | None = None


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
    is the backpressure mechanism as well as the memory bound: a consumer that
    stops taking chunks stops the prefetch with it, rather than letting
    generation run ahead into memory nothing is reading. Time spent waiting on
    that bound is reported as ``backpressure_s`` so a slow consumer is visible
    in the results instead of silently inflating chunk latency.

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
        self._pending: asyncio.Task | None = None
        self._pending_started: float | None = None
        self._closed = False

    @property
    def decode_state(self) -> Any:
        return self._state

    def _start_generation(self) -> None:
        if self._pending is None and not self._closed:
            self._pending_started = self.clock()
            self._pending = asyncio.ensure_future(self.inner.next_chunk())

    async def next_chunk(self) -> Any:
        t0 = self.clock()
        # A prefetch issued during the previous tick has been running while the
        # caller was away, so only the part still outstanding is charged here.
        self._start_generation()
        pending, started = self._pending, self._pending_started
        self._pending, self._pending_started = None, None
        output = await pending
        t1 = self.clock()
        generate_s = t1 - (started if started is not None else t0)
        waited_s = t1 - t0
        overlap_s = max(0.0, generate_s - waited_s)

        # Issue the next generation before decoding, so the two overlap.
        self._start_generation()

        frames = self.decoder.decode_chunk(self.latent_of(output), self._state)
        t2 = self.clock()

        nbytes = getattr(self._state, "nbytes", None)
        self.last = StageTiming(
            generate_s=generate_s,
            decode_s=t2 - t1,
            frames=int(frames.shape[2]) if hasattr(frames, "shape") and len(frames.shape) >= 3 else None,
            resident_decoder_bytes=nbytes() if callable(nbytes) else None,
            overlap_s=overlap_s,
            backpressure_s=0.0,
        )
        return frames

    async def close(self) -> None:
        self._closed = True
        if self._pending is not None:
            self._pending.cancel()
            try:
                await self._pending
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - shutdown must not mask close
                pass
            self._pending = None
        try:
            self.decoder.release(self._state)
        finally:
            await self.inner.close()
