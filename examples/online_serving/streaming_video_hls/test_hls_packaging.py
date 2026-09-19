# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packaging tests that need no GPU, no server, and no ffmpeg.

The fMP4 structures are built in memory, so this exercises the two things that
actually break in transport: box boundaries that do not align with WebSocket
frames, and a playlist that advertises files nobody wrote.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fmp4 import FragmentedMP4Splitter  # noqa: E402
from llhls import LLHLSPackager  # noqa: E402


def box(box_type: str, payload: bytes = b"") -> bytes:
    """One MP4 box: 32-bit size, four-character type, payload."""
    return struct.pack(">I", 8 + len(payload)) + box_type.encode("latin-1") + payload


def build_stream(fragments: int = 5, payload_size: int = 512) -> tuple[bytes, bytes, list[bytes]]:
    """Return (whole stream, expected init segment, expected fragments)."""
    init = box("ftyp", b"isom" * 2) + box("moov", b"\x00" * 64)
    frags = [
        box("moof", struct.pack(">I", i) + b"\x00" * 24) + box("mdat", bytes([i % 251]) * payload_size)
        for i in range(fragments)
    ]
    # mfra trails a real file and is not media; the splitter must ignore it.
    return init + b"".join(frags) + box("mfra", b"\x00" * 16), init, frags


@pytest.mark.parametrize("slice_size", [None, 4096, 337, 17, 1])
def test_splitter_is_framing_independent(slice_size: int | None) -> None:
    """Transport framing must not change what comes out."""
    stream, expected_init, expected_frags = build_stream()
    splitter = FragmentedMP4Splitter()
    got: list[bytes] = []

    step = slice_size or len(stream)
    for i in range(0, len(stream), step):
        got += splitter.feed(stream[i : i + step])

    assert splitter.init_segment == expected_init
    assert got == expected_frags
    assert splitter.buffered == 0, "trailing mfra should not be left buffered"


def test_every_fragment_starts_with_moof() -> None:
    stream, _, _ = build_stream(fragments=3)
    splitter = FragmentedMP4Splitter()
    for frag in splitter.feed(stream):
        assert frag[4:8] == b"moof"


def test_splitter_ignores_trailing_non_media_boxes() -> None:
    """mfra, sidx and friends must not be emitted as fragments."""
    stream, _, expected = build_stream(fragments=2)
    splitter = FragmentedMP4Splitter()
    assert splitter.feed(stream) == expected


def test_playlist_references_only_files_that_exist(tmp_path: Path) -> None:
    """The failure this catches: parts published without their parent segment.

    LL-HLS players fetch parts at the live edge, but late joiners and ordinary
    players fetch the segment. A playlist that names segments nobody wrote 404s
    for exactly those viewers.
    """
    stream, _, _ = build_stream(fragments=7)
    splitter = FragmentedMP4Splitter()
    packager = LLHLSPackager(out_dir=tmp_path, part_duration=9 / 16, parts_per_segment=4)

    started = False
    for frag in splitter.feed(stream):
        if not started:
            packager.start(splitter.init_segment)
            started = True
        packager.add_fragment(frag)
    packager.finish()

    playlist = (tmp_path / "stream.m3u8").read_text()
    referenced = set()
    for line in playlist.splitlines():
        if line.startswith("#EXT-X-PART:") or line.startswith("#EXT-X-MAP:"):
            uri = line.split('URI="', 1)[1].split('"', 1)[0]
            referenced.add(uri)
        elif line and not line.startswith("#"):
            referenced.add(line.strip())

    missing = sorted(u for u in referenced if not (tmp_path / u).exists())
    assert not missing, f"playlist references files that were never written: {missing}"


def test_report_counts_every_part(tmp_path: Path) -> None:
    stream, _, expected = build_stream(fragments=6)
    splitter = FragmentedMP4Splitter()
    packager = LLHLSPackager(out_dir=tmp_path, part_duration=9 / 16)

    started = False
    for frag in splitter.feed(stream):
        if not started:
            packager.start(splitter.init_segment)
            started = True
        packager.add_fragment(frag)
    packager.finish()

    report = packager.report()
    assert report["parts"] == len(expected)
    assert report["bytes"] == sum(len(f) for f in expected)
    assert report["media_seconds"] == pytest.approx(len(expected) * 9 / 16, abs=1e-3)


def test_fragment_before_init_is_rejected(tmp_path: Path) -> None:
    packager = LLHLSPackager(out_dir=tmp_path, part_duration=9 / 16)
    with pytest.raises(RuntimeError):
        packager.add_fragment(b"\x00" * 32)
