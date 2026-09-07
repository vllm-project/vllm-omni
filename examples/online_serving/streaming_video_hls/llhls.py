# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
arrives, and republishes the playlist. Every generated chunk is 9 frames at 16fps
in the reference config, which is 0.5625s: already part-sized, so no re-encoding
and no transcode is involved. Bytes pass through untouched.

The number that matters is not render time. It is prompt-to-glass: render, mux,
write, publish, fetch, decode, first frame. This records the terms it can see.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PartRecord:
    """One published LL-HLS part, with the timings we can observe locally."""

    index: int
    uri: str
    duration: float
    bytes: int
    received_at: float
    published_at: float

    @property
    def publish_latency(self) -> float:
        """Seconds from having the bytes to the playlist advertising them."""
        return self.published_at - self.received_at


@dataclass
class LLHLSPackager:
    """Write a live LL-HLS playlist as fragments arrive.

    ``parts_per_segment`` groups parts into segments. With 0.5625s parts, 4 parts
    gives 2.25s segments, which keeps the playlist small while staying inside the
    usual 3x target-duration guidance for part hold-back.
    """

    out_dir: Path
    part_duration: float
    parts_per_segment: int = 4
    playlist_name: str = "stream.m3u8"

    _parts: list[PartRecord] = field(default_factory=list)
    _pending: list[bytes] = field(default_factory=list)
    _segment_index: int = 0
    _t0: float | None = None
    started_at: float | None = None

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # -- lifecycle -----------------------------------------------------------

    def start(self, init_segment: bytes) -> None:
        """Write the init segment and mark the clock start.

        Called once, on the first fragment. The init segment becomes EXT-X-MAP,
        which every player fetches before any media.
        """
        self._t0 = time.perf_counter()
        self.started_at = self._t0
        (self.out_dir / "init.mp4").write_bytes(init_segment)

    def add_fragment(self, data: bytes, received_at: float | None = None) -> PartRecord:
        """Publish one fragment as the next LL-HLS part.

        The fragment is written verbatim. No transcode, no remux: the server
        already produced a valid fMP4 fragment, and re-encoding here would both
        add latency and make any quality measurement meaningless.
        """
        if self._t0 is None:
            raise RuntimeError("start() must be called with the init segment first")
        received_at = received_at if received_at is not None else time.perf_counter()

        index = len(self._parts)
        uri = f"part{index:05d}.m4s"
        (self.out_dir / uri).write_bytes(data)
        self._pending.append(data)

        # A part belongs to a parent segment. When the segment completes, that
        # full segment file has to exist on disk: LL-HLS players fetch parts for
        # the live edge, but ordinary players and anyone joining late fetch the
        # segment. Writing only parts leaves those requests 404ing.
        if len(self._pending) == self.parts_per_segment:
            seg = f"segment{self._segment_index:05d}.m4s"
            (self.out_dir / seg).write_bytes(b"".join(self._pending))
            self._segment_index += 1
            self._pending.clear()

        record = PartRecord(
            index=index,
            uri=uri,
            duration=self.part_duration,
            bytes=len(data),
            received_at=received_at,
            published_at=0.0,
        )
        self._parts.append(record)
        self._write_playlist()
        record.published_at = time.perf_counter()
        return record

    def finish(self) -> None:
        """Flush any partial trailing segment, then append the end marker."""
        if self._pending:
            seg = f"segment{self._segment_index:05d}.m4s"
            (self.out_dir / seg).write_bytes(b"".join(self._pending))
            self._segment_index += 1
            self._pending.clear()
        self._write_playlist(ended=True)

    # -- playlist ------------------------------------------------------------

    def _write_playlist(self, ended: bool = False) -> None:
        seg_dur = self.part_duration * self.parts_per_segment
        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:9",
            f"#EXT-X-TARGETDURATION:{max(1, round(seg_dur))}",
            f"#EXT-X-PART-INF:PART-TARGET={self.part_duration:.5f}",
            # Part hold-back must be at least three part durations per the spec.
            f"#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES,PART-HOLD-BACK={self.part_duration * 3:.5f}",
            "#EXT-X-MEDIA-SEQUENCE:0",
            '#EXT-X-MAP:URI="init.mp4"',
        ]

        for i, part in enumerate(self._parts):
            lines.append(f'#EXT-X-PART:DURATION={part.duration:.5f},URI="{part.uri}",INDEPENDENT=YES')
            # Close a segment every parts_per_segment parts. The parts are the
            # media; EXTINF here just gives non-LL players a segment boundary.
            if (i + 1) % self.parts_per_segment == 0:
                lines.append(f"#EXTINF:{seg_dur:.5f},")
                lines.append(f"segment{i // self.parts_per_segment:05d}.m4s")

        if ended:
            leftover = len(self._parts) % self.parts_per_segment
            if leftover:
                lines.append(f"#EXTINF:{self.part_duration * leftover:.5f},")
                lines.append(f"segment{len(self._parts) // self.parts_per_segment:05d}.m4s")
            lines.append("#EXT-X-ENDLIST")
        else:
            nxt = f"part{len(self._parts):05d}.m4s"
            lines.append(f'#EXT-X-PRELOAD-HINT:TYPE=PART,URI="{nxt}"')

        (self.out_dir / self.playlist_name).write_text("\n".join(lines) + "\n")

    # -- reporting -----------------------------------------------------------

    def report(self) -> dict:
        """The latency budget, as far as this process can observe it."""
        if not self._parts or self._t0 is None:
            return {}
        first = self._parts[0]
        media_seconds = len(self._parts) * self.part_duration
        wall = self._parts[-1].published_at - self._t0
        pub = [p.publish_latency for p in self._parts]
        return {
            "parts": len(self._parts),
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
