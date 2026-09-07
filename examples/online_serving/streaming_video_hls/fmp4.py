# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
