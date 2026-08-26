# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Release generated media on the timeline a viewer plays it at.

A streaming diffusion session produces video in chunks and, today, hands each
chunk on the moment it is finished. For an offline render that is right. For an
*interactive* session it is not, because generation and playback then run on
different clocks and drift apart.

Say a chunk is 12 frames at 16 fps -- 0.75 s of video -- and generation runs at
twice real time::

    wall 0.0s -> chunk 1 produced, covering video 0.00-0.75s
    wall 0.4s -> chunk 2 produced, covering video 0.75-1.50s
    wall 4.0s -> generation has reached video second 10

The viewer is watching second 4 while the generator is at second 10. A control
input that arrives now has no good home: applied at second 10 it appears to do
nothing for six seconds, and applied at second 4 it invalidates six seconds of
finished work. The gap itself is the bug, so the fix is to not open it.

This class computes how long to hold a chunk so that it leaves at the moment its
first frame is due to play. Generation faster than real time waits; generation
slower than real time is never delayed, so the pacer costs nothing on a session
that is already struggling to keep up.

It deliberately does not implement consumer backlog or admission signalling.
Those are a separate policy with their own protocol surface, and a release
schedule is useful without them.
"""

from __future__ import annotations

import time
from collections.abc import Callable

__all__ = ["MediaTimePacer"]


class MediaTimePacer:
    """Media-time release schedule for one streaming session.

    Args:
        fps: Frame rate the produced media is meant to play at. Must be positive.
        lead_seconds: How far *ahead* of the play-out deadline a chunk may be
            released. Zero schedules every chunk just-in-time: it leaves exactly
            when its first frame is due, so the consumer's buffer runs down to
            nothing between arrivals and any jitter on the link shows up as an
            underrun. A client with its own jitter buffer is fine with that;
            otherwise set this to a chunk or two of media duration and the
            server keeps that much slack in hand.
        max_lag_seconds: How far behind schedule the session may fall before the
            schedule is re-based on the present. Without this, a session that
            stalls builds up a debt against a fixed origin and then releases
            everything it owes back to back once it recovers -- a burst, which
            is the behaviour the pacer exists to prevent. ``None`` disables
            re-basing and keeps the original origin forever.
        clock: Monotonic time source, injectable for tests.

    The pacer is not thread-safe and is not shared between sessions: it holds one
    session's origin.
    """

    def __init__(
        self,
        fps: float,
        *,
        lead_seconds: float = 0.0,
        max_lag_seconds: float | None = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not fps > 0:
            raise ValueError(f"fps must be positive, got {fps!r}")
        if lead_seconds < 0:
            raise ValueError(f"lead_seconds must not be negative, got {lead_seconds!r}")
        if max_lag_seconds is not None and max_lag_seconds < 0:
            raise ValueError(f"max_lag_seconds must not be negative, got {max_lag_seconds!r}")
        self._fps = float(fps)
        self._lead_seconds = float(lead_seconds)
        self._max_lag_seconds = max_lag_seconds
        self._clock = clock
        self._origin: float | None = None
        self._released_frames = 0
        self._rebases = 0

    @property
    def released_frames(self) -> int:
        """Frames handed out so far, including the chunk of the last call."""
        return self._released_frames

    @property
    def released_media_seconds(self) -> float:
        """Media duration handed out so far."""
        return self._released_frames / self._fps

    @property
    def rebases(self) -> int:
        """How often the schedule has been re-based after falling behind."""
        return self._rebases

    def delay_before_release(self, num_frames: int) -> float:
        """Seconds to wait before releasing a chunk of ``num_frames`` frames.

        Call once per chunk, immediately before sending it, and sleep for the
        returned duration. The frames are accounted at call time whether or not
        the caller honours the delay, so a caller that chooses to skip the wait
        still gets a correct schedule for the chunks after it.
        """
        if num_frames < 0:
            raise ValueError(f"num_frames must not be negative, got {num_frames!r}")
        now = self._clock()

        # The viewer's clock starts when the viewer gets something to watch, not
        # when generation was requested: the first chunk is never held back.
        if self._origin is None:
            self._origin = now
            self._released_frames = num_frames
            return 0.0

        due_at = self._origin + self.released_media_seconds - self._lead_seconds
        self._released_frames += num_frames

        delay = due_at - now
        if delay >= 0.0:
            return delay
        if self._max_lag_seconds is not None and -delay > self._max_lag_seconds:
            # Behind by more than the allowance. Move the origin so "now" is
            # exactly on schedule; the debt is written off rather than repaid in
            # a burst.
            self._origin = now - self.released_media_seconds + num_frames / self._fps + self._lead_seconds
            self._rebases += 1
        return 0.0
