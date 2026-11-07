# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Incremental fragmented-MP4 splitter.

vLLM-Omni's ``WS /v1/realtime/video`` sends fragmented MP4 over a WebSocket: one
binary frame per generated chunk, each preceded by a ``video.chunk_metadata`` JSON
frame. Transport framing and MP4 box boundaries are not the same thing, so a
consumer that wants to publish segments cannot assume one WebSocket frame equals
one media fragment.

This splitter takes bytes as they arrive, in any framing, and emits:

  * the init segment once (``ftyp`` through the end of ``moov``), which HLS needs
    as ``EXT-X-MAP``, and
  * one buffer per media fragment (``moof`` + its ``mdat``), which is the unit an
    LL-HLS part or segment is made of.

Stdlib only, on purpose: this is meant to be droppable into
``examples/online_serving/`` without adding a dependency.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field


@dataclass
class Box:
    """One top-level MP4 box: its four-character type and byte range."""

    type: str
    start: int
    size: int

    @property
    def end(self) -> int:
        return self.start + self.size


def iter_boxes(buf: bytes, offset: int = 0):
    """Yield complete top-level boxes in ``buf`` starting at ``offset``.

    Stops cleanly at the first truncated box, which is the normal case while
    streaming: the tail is an incomplete box we have not finished receiving.
    """
    i = offset
    n = len(buf)
    while i + 8 <= n:
        size = struct.unpack_from(">I", buf, i)[0]
        typ = buf[i + 4 : i + 8].decode("latin-1")
        if size == 1:
            # 64-bit extended size lives in the 8 bytes after the type.
            if i + 16 > n:
                return
            size = struct.unpack_from(">Q", buf, i + 8)[0]
        elif size == 0:
            # "to end of file" - only legal for the last box, so treat the rest
            # as one box and stop.
            yield Box(typ, i, n - i)
            return
        if size < 8 or i + size > n:
            return
        yield Box(typ, i, size)
        i += size


@dataclass
class FragmentedMP4Splitter:
    """Feed bytes in, get an init segment and whole media fragments out."""

    _buf: bytearray = field(default_factory=bytearray)
    _cursor: int = 0
    init_segment: bytes | None = None

    def feed(self, data: bytes) -> list[bytes]:
        """Append ``data`` and return every media fragment completed by it.

        The init segment is captured on the way past and exposed as
        ``init_segment`` rather than returned, since HLS treats it differently
        from media: it is referenced once by ``EXT-X-MAP``, not listed as a part.
        """
        self._buf.extend(data)
        fragments: list[bytes] = []
        pending_moof: Box | None = None

        for box in iter_boxes(bytes(self._buf), self._cursor):
            if box.type == "moov":
                # Init segment is everything from the start through moov, which
                # keeps ftyp and any leading boxes with it.
                self.init_segment = bytes(self._buf[: box.end])
                self._cursor = box.end
                continue
            if box.type == "moof":
                pending_moof = box
                continue
            if box.type == "mdat" and pending_moof is not None:
                fragments.append(bytes(self._buf[pending_moof.start : box.end]))
                self._cursor = box.end
                pending_moof = None
                continue
            # mfra, sidx, free and friends: skip, but keep the cursor moving so
            # the buffer can be trimmed.
            #
            # Before the init segment is captured, the cursor must NOT advance
            # past leading boxes. ftyp often completes in an earlier feed() than
            # moov, and trimming it away would leave the init segment starting at
            # moov. Only shows up when transport framing is smaller than a box,
            # which is exactly what a WebSocket does under load.
            if pending_moof is None and self.init_segment is not None:
                self._cursor = box.end

        # Drop what we have already emitted. Keeps memory flat across a long
        # generation rather than growing with total output size.
        if self._cursor:
            del self._buf[: self._cursor]
            self._cursor = 0
        return fragments

    @property
    def buffered(self) -> int:
        """Bytes held pending a complete box. Useful as a stall signal."""
        return len(self._buf)


# Sample flag bits from ISO/IEC 14496-12. A set sample_is_non_sync_sample bit means
# the sample cannot be decoded without earlier samples.
_NON_SYNC = 0x00010000

_TFHD_BASE_DATA_OFFSET = 0x000001
_TFHD_SAMPLE_DESCRIPTION = 0x000002
_TFHD_DEFAULT_DURATION = 0x000008
_TFHD_DEFAULT_SIZE = 0x000010
_TFHD_DEFAULT_FLAGS = 0x000020

_TRUN_DATA_OFFSET = 0x000001
_TRUN_FIRST_SAMPLE_FLAGS = 0x000004
_TRUN_SAMPLE_DURATION = 0x000100
_TRUN_SAMPLE_SIZE = 0x000200
_TRUN_SAMPLE_FLAGS = 0x000400
_TRUN_SAMPLE_COMPOSITION = 0x000800


def _u32(buf: bytes, pos: int) -> int:
    return struct.unpack_from(">I", buf, pos)[0]


def _children(buf: bytes, start: int, end: int):
    """Yield (type, payload_start, box_end) for each child box in buf[start:end]."""
    i = start
    while i + 8 <= end:
        size = _u32(buf, i)
        typ = buf[i + 4 : i + 8].decode("latin-1")
        header = 8
        if size == 1:
            size = struct.unpack_from(">Q", buf, i + 8)[0]
            header = 16
        elif size == 0:
            size = end - i
        if size < header or i + size > end:
            return
        yield typ, i + header, i + size
        i += size


def _find(buf: bytes, start: int, end: int, *path: str) -> tuple[int, int] | None:
    """Payload range of the first box at ``path``, nested under buf[start:end]."""
    for typ, p0, p1 in _children(buf, start, end):
        if typ == path[0]:
            if len(path) == 1:
                return p0, p1
            return _find(buf, p0, p1, *path[1:])
    return None


def track_timescale(init: bytes) -> int:
    """Ticks per second of the track in an init segment, from its mdhd box."""
    found = _find(init, 0, len(init), "moov", "trak", "mdia", "mdhd")
    if found is None:
        raise ValueError("init segment has no mdhd box")
    p0, _ = found
    # mdhd payload: version and flags, then creation and modification times (8 bytes
    # each in version 0, 16 each in version 1), then the timescale.
    version = init[p0]
    return _u32(init, p0 + 4 + (16 if version == 1 else 8))


@dataclass(frozen=True)
class FragmentTiming:
    """What a media fragment declares about its own timing."""

    duration: float  # seconds of media in the fragment
    independent: bool  # first sample is a sync sample: a player can start decoding here


def _tfhd_defaults(buf: bytes, p0: int) -> tuple[int | None, int | None]:
    """The tfhd default sample duration and default sample flags, if present."""
    flags = _u32(buf, p0) & 0xFFFFFF
    pos = p0 + 8  # version and flags, then track_ID
    if flags & _TFHD_BASE_DATA_OFFSET:
        pos += 8
    if flags & _TFHD_SAMPLE_DESCRIPTION:
        pos += 4
    default_duration = None
    if flags & _TFHD_DEFAULT_DURATION:
        default_duration = _u32(buf, pos)
        pos += 4
    if flags & _TFHD_DEFAULT_SIZE:
        pos += 4
    default_flags = _u32(buf, pos) if flags & _TFHD_DEFAULT_FLAGS else None
    return default_duration, default_flags


def _trun(buf: bytes, p0: int, end: int, default_duration: int | None, default_flags: int | None) -> tuple[int, bool]:
    """Total ticks in one trun box, and whether its first sample is a sync sample.

    Every read stays inside the box, [p0, end), so a trun cannot pull in the bytes
    of the boxes after it.
    """
    if end - p0 < 8:
        raise ValueError("trun box is shorter than its header")
    flags = _u32(buf, p0) & 0xFFFFFF
    count = _u32(buf, p0 + 4)
    pos = p0 + 8
    if flags & _TRUN_DATA_OFFSET:
        pos += 4
    if pos + (4 if flags & _TRUN_FIRST_SAMPLE_FLAGS else 0) > end:
        raise ValueError("trun header runs past its box")
    first_flags = None
    if flags & _TRUN_FIRST_SAMPLE_FLAGS:
        first_flags = _u32(buf, pos)
        pos += 4

    if count == 0:
        return 0, False
    # Each sample record is 4 bytes per per-sample field the flags name. Refuse a count
    # the box cannot hold before walking it, so a forged count cannot stall us.
    fields = sum(
        1
        for bit in (_TRUN_SAMPLE_DURATION, _TRUN_SAMPLE_SIZE, _TRUN_SAMPLE_FLAGS, _TRUN_SAMPLE_COMPOSITION)
        if flags & bit
    )
    if pos + count * 4 * fields > end:
        raise ValueError(f"trun declares {count} samples but its box ends before them")
    if not fields:
        # Every sample takes the defaults, so the total is arithmetic, not a walk.
        if default_duration is None:
            raise ValueError("sample has no duration: no trun duration and no tfhd default")
        sync = first_flags if first_flags is not None else default_flags
        return count * default_duration, sync is not None and not (sync & _NON_SYNC)

    ticks = 0
    first_sync = False
    for n in range(count):
        duration = default_duration
        if flags & _TRUN_SAMPLE_DURATION:
            duration = _u32(buf, pos)
            pos += 4
        if flags & _TRUN_SAMPLE_SIZE:
            pos += 4
        sample_flags = default_flags
        if flags & _TRUN_SAMPLE_FLAGS:
            sample_flags = _u32(buf, pos)
            pos += 4
        if flags & _TRUN_SAMPLE_COMPOSITION:
            pos += 4
        if duration is None:
            raise ValueError("sample has no duration: no trun duration and no tfhd default")
        ticks += duration
        if n == 0:
            if first_flags is not None:
                sample_flags = first_flags
            # Unknown flags count as not independent. Claiming a keyframe we cannot
            # see would send players to a frame they cannot decode.
            first_sync = sample_flags is not None and not (sample_flags & _NON_SYNC)
    return ticks, first_sync


def fragment_timing(fragment: bytes, timescale: int) -> FragmentTiming:
    """Read a moof+mdat fragment's duration and whether it starts on a keyframe.

    Durations come from the sample durations in each trun, falling back to the
    tfhd default. Independence comes from the first sample's sync flag.
    """
    moof = _find(fragment, 0, len(fragment), "moof")
    if moof is None:
        raise ValueError("fragment has no moof box")
    traf = _find(fragment, moof[0], moof[1], "traf")
    if traf is None:
        raise ValueError("moof has no traf box")

    ticks = 0
    first_sync: bool | None = None
    default_duration: int | None = None
    default_flags: int | None = None
    for typ, p0, end in _children(fragment, traf[0], traf[1]):
        if typ == "tfhd":
            default_duration, default_flags = _tfhd_defaults(fragment, p0)
        elif typ == "trun":
            run_ticks, run_sync = _trun(fragment, p0, end, default_duration, default_flags)
            ticks += run_ticks
            if first_sync is None:
                first_sync = run_sync
    if first_sync is None:
        raise ValueError("traf has no trun")
    return FragmentTiming(duration=ticks / timescale, independent=first_sync)
