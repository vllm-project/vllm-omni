# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Low-Latency HLS packager for a live fragmented-MP4 fragment stream.

Why this exists
---------------
vLLM-Omni can generate video faster than it plays: a 10.1s clip rendered in 8.7s.
That is a *streaming* claim, but the shipped consumers cannot demonstrate it as
one. ``streaming_video_client.py`` writes the bytes to a file and remuxes to a
progressive MP4 at the end, and ``gradio_demo.py`` appends fragments through MSE
in a single attached browser. Neither produces something a CDN can fan out or a
normal player can open, and neither measures when a frame first becomes visible.

This packager turns each generated fragment into an LL-HLS *part* the moment it
arrives, and republishes the playlist. Bytes pass through untouched: no
re-encoding and no transcode is involved. Each part's duration and whether it
starts on a keyframe are read from the fragment's own boxes.

The number that matters is not render time. It is prompt-to-glass: render, mux,
write, publish, fetch, decode, first frame. This records the terms it can see.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path

from fmp4 import fragment_timing, track_timescale

# Reference chunk: 9 frames at 16 fps. Used as the declared PART-TARGET unless the
# caller says otherwise.
REFERENCE_PART_SECONDS = 9 / 16

_EPSILON = 1e-6


@dataclass
class PartRecord:
    """One published LL-HLS part, with the timings we can observe locally."""

    index: int
    uri: str
    duration: float
    independent: bool
    bytes: int
    received_at: float
    published_at: float
    segment: int

    @property
    def publish_latency(self) -> float:
        """Seconds from having the bytes to the playlist advertising them."""
        return self.published_at - self.received_at


@dataclass
class LLHLSPackager:
    """Write a live LL-HLS playlist as fragments arrive.

    Each fragment is one part. A parent segment closes when a keyframe-led part
    arrives, so every segment starts on a keyframe and a late joiner can begin
    decoding at any segment boundary. A long GOP would otherwise make one segment
    as long as the GOP, so a segment also closes before it would pass
    ``max_segment_seconds``. Those forced splits can start mid-GOP, which is the
    cost of a bounded target duration.

    The playlist header is fixed for its whole life, as RFC 8216 section 6.2.1
    requires for EXT-X-TARGETDURATION. ``part_target`` is the declared
    EXT-X-PART-INF PART-TARGET, and any part longer than it is an error.
    """

    out_dir: Path
    part_target: float = REFERENCE_PART_SECONDS
    max_segment_seconds: float = 6.0
    playlist_name: str = "stream.m3u8"

    _timescale: int = 0
    _parts: list[PartRecord] = field(default_factory=list)
    _pending: list[bytes] = field(default_factory=list)  # bytes of the open segment
    _pending_duration: float = 0.0
    _segment_durations: list[float] = field(default_factory=list)  # closed segments
    _t0: float | None = None
    started_at: float | None = None

    def __post_init__(self) -> None:
        if self.part_target <= 0 or self.max_segment_seconds <= 0:
            raise ValueError("part_target and max_segment_seconds must be positive")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # -- lifecycle -----------------------------------------------------------

    def start(self, init_segment: bytes) -> None:
        """Write the init segment and mark the clock start.

        Called once, on the first fragment. The init segment becomes EXT-X-MAP,
        which every player fetches before any media. Its timescale converts each
        fragment's sample durations to seconds.
        """
        self._timescale = track_timescale(init_segment)
        self._t0 = time.perf_counter()
        self.started_at = self._t0
        (self.out_dir / "init.mp4").write_bytes(init_segment)

    def add_fragment(self, data: bytes, received_at: float | None = None) -> PartRecord:
        """Publish one fragment as the next LL-HLS part.

        The fragment is written verbatim. No transcode, no remux: the server already
        produced a valid fMP4 fragment, and re-encoding here would both add latency
        and make any quality measurement meaningless.
        """
        if self._t0 is None:
            raise RuntimeError("start() must be called with the init segment first")
        received_at = received_at if received_at is not None else time.perf_counter()
        timing = fragment_timing(data, self._timescale)
        if timing.duration > self.part_target + _EPSILON:
            raise ValueError(
                f"part {len(self._parts)} is {timing.duration:.4f}s, longer than the declared "
                f"PART-TARGET of {self.part_target:.4f}s; pass a larger part_target"
            )
        if timing.duration > self.max_segment_seconds + _EPSILON:
            raise ValueError(
                f"part {len(self._parts)} is {timing.duration:.4f}s, longer than "
                f"max_segment_seconds of {self.max_segment_seconds:.4f}s"
            )

        # A keyframe starts a new parent segment. So does a part that would push the
        # open segment past the declared target duration. The closed segment is
        # written to disk before the playlist that lists it is updated.
        if self._pending and (
            timing.independent or self._pending_duration + timing.duration > self.max_segment_seconds + _EPSILON
        ):
            self._close_segment()

        index = len(self._parts)
        uri = f"part{index:05d}.m4s"
        (self.out_dir / uri).write_bytes(data)
        self._pending.append(data)
        self._pending_duration += timing.duration

        record = PartRecord(
            index=index,
            uri=uri,
            duration=timing.duration,
            independent=timing.independent,
            bytes=len(data),
            received_at=received_at,
            published_at=0.0,
            segment=len(self._segment_durations),
        )
        self._parts.append(record)
        self._write_playlist()
        record.published_at = time.perf_counter()
        return record

    def finish(self) -> None:
        """Close the open segment, then append the end marker."""
        if self._pending:
            self._close_segment()
        self._write_playlist(ended=True)

    def _close_segment(self) -> None:
        """Write the open segment's file and make it an advertised, closed segment."""
        seg = f"segment{len(self._segment_durations):05d}.m4s"
        (self.out_dir / seg).write_bytes(b"".join(self._pending))
        self._segment_durations.append(self._pending_duration)
        self._pending.clear()
        self._pending_duration = 0.0

    # -- playlist ------------------------------------------------------------

    def _write_playlist(self, ended: bool = False) -> None:
        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:9",
            # Fixed for the playlist's life (RFC 8216 section 6.2.1). Every segment is
            # at most max_segment_seconds, so this is always >= each EXTINF when rounded.
            f"#EXT-X-TARGETDURATION:{math.ceil(self.max_segment_seconds - _EPSILON)}",
            f"#EXT-X-PART-INF:PART-TARGET={self.part_target:.5f}",
            # Hold-back is three part targets. The spec's minimum is two. There is no
            # CAN-BLOCK-RELOAD: a static file server cannot hold a playlist reload, so
            # the playlist does not claim one.
            f"#EXT-X-SERVER-CONTROL:PART-HOLD-BACK={self.part_target * 3:.5f}",
            "#EXT-X-MEDIA-SEQUENCE:0",
            '#EXT-X-MAP:URI="init.mp4"',
        ]

        # Each segment's parts are listed, then its EXTINF and URI. The URI appears
        # only once the segment is closed and its file is on disk.
        durations = self._segment_durations
        for seg, parts in groupby(self._parts, key=lambda p: p.segment):
            for part in parts:
                attrs = f'DURATION={part.duration:.5f},URI="{part.uri}"'
                if part.independent:
                    attrs += ",INDEPENDENT=YES"
                lines.append(f"#EXT-X-PART:{attrs}")
            if seg < len(durations):
                lines.append(f"#EXTINF:{durations[seg]:.5f},")
                lines.append(f"segment{seg:05d}.m4s")

        if ended:
            lines.append("#EXT-X-ENDLIST")
        else:
            lines.append(f'#EXT-X-PRELOAD-HINT:TYPE=PART,URI="part{len(self._parts):05d}.m4s"')

        (self.out_dir / self.playlist_name).write_text("\n".join(lines) + "\n")

    # -- reporting -----------------------------------------------------------

    def report(self) -> dict:
        """The latency budget, as far as this process can observe it."""
        if not self._parts or self._t0 is None:
            return {}
        first = self._parts[0]
        media_seconds = sum(p.duration for p in self._parts)
        wall = self._parts[-1].published_at - self._t0
        pub = [p.publish_latency for p in self._parts]
        return {
            "parts": len(self._parts),
            "segments": len(self._segment_durations) + (1 if self._pending else 0),
            "media_seconds": round(media_seconds, 3),
            "wall_seconds": round(wall, 3),
            "realtime_factor": round(media_seconds / wall, 3) if wall > 0 else None,
            "time_to_first_part_s": round(first.published_at - self._t0, 4),
            "publish_latency_ms": {
                "mean": round(sum(pub) / len(pub) * 1000, 3),
                "max": round(max(pub) * 1000, 3),
            },
            "bytes": sum(p.bytes for p in self._parts),
        }

    def write_report(self, path: Path) -> dict:
        data = self.report()
        path.write_text(json.dumps(data, indent=2) + "\n")
        return data
