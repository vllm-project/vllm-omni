# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packaging tests that need no GPU, no server, and no ffmpeg.

The fMP4 structures are built in memory with the boxes that carry timing and
keyframe information, so this exercises what actually breaks in transport: box
boundaries that do not align with WebSocket frames, durations and keyframes that
must come from the fragment rather than a guess, and a playlist that advertises
files nobody wrote.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import types
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import hls_client  # noqa: E402
from fmp4 import FragmentedMP4Splitter, fragment_timing, track_timescale  # noqa: E402
from llhls import LLHLSPackager  # noqa: E402

TIMESCALE = 16000  # ticks per second
SAMPLE_TICKS = 1000  # one frame at 16 fps
SAMPLES_PER_FRAGMENT = 9  # the reference chunk: 9 frames, 0.5625 s
PART_SECONDS = SAMPLES_PER_FRAGMENT * SAMPLE_TICKS / TIMESCALE


def box(box_type: str, payload: bytes = b"") -> bytes:
    """One MP4 box: 32-bit size, four-character type, payload."""
    return struct.pack(">I", 8 + len(payload)) + box_type.encode("latin-1") + payload


def mdhd(timescale: int, version: int = 0) -> bytes:
    if version == 0:
        body = struct.pack(">IIIII", 0, 0, 0, timescale, 0)
    else:
        body = struct.pack(">IQQIQ", 1 << 24, 0, 0, timescale, 0)
    return box("mdhd", body + b"\x00" * 4)


def build_init(timescale: int = TIMESCALE, version: int = 0) -> bytes:
    """ftyp and moov, with one track whose mdhd carries the timescale."""
    trak = box("trak", box("mdia", mdhd(timescale, version)))
    return box("ftyp", b"isom" * 2) + box("moov", trak)


def build_fragment(index: int, keyframe: bool = True, payload_size: int = 512) -> bytes:
    """moof+mdat shaped like ffmpeg's frag_keyframe output.

    One trun with a duration and sync flag per sample. Only the first sample of a
    keyframe fragment is a sync sample.
    """
    trun_flags = 0x000001 | 0x000100 | 0x000400  # data offset, sample duration, sample flags
    rows = b"".join(
        struct.pack(">II", SAMPLE_TICKS, 0x02000000 if keyframe and n == 0 else 0x01010000)
        for n in range(SAMPLES_PER_FRAGMENT)
    )
    trun = box("trun", struct.pack(">IIi", trun_flags, SAMPLES_PER_FRAGMENT, 0) + rows)
    tfhd = box("tfhd", struct.pack(">II", 0x020000, 1))  # default-base-is-moof, track 1
    moof = box("moof", box("mfhd", struct.pack(">II", 0, index + 1)) + box("traf", tfhd + trun))
    return moof + box("mdat", bytes([index % 251]) * payload_size)


def build_stream(
    keyframes: Sequence[bool] = (True, False, False, True, False, False, True),
) -> tuple[bytes, bytes, list[bytes]]:
    """Return (whole stream, expected init segment, expected fragments)."""
    init = build_init()
    frags = [build_fragment(i, keyframe=k) for i, k in enumerate(keyframes)]
    # mfra trails a real file and is not media; the splitter must ignore it.
    return init + b"".join(frags) + box("mfra", b"\x00" * 16), init, frags


def package(out_dir: Path, stream: bytes, after_each: Callable[[Path], None] | None = None) -> LLHLSPackager:
    """Feed a stream through the splitter and packager, as the client does."""
    splitter = FragmentedMP4Splitter()
    packager = LLHLSPackager(out_dir=out_dir)
    started = False
    for frag in splitter.feed(stream):
        if not started:
            packager.start(splitter.init_segment)
            started = True
        packager.add_fragment(frag)
        if after_each:
            after_each(out_dir)
    packager.finish()
    return packager


def referenced_files(playlist: str) -> set[str]:
    """Every media file the playlist names. PRELOAD-HINT is excluded: it names a
    part that is not written yet, by design."""
    refs = set()
    for line in playlist.splitlines():
        if line.startswith(("#EXT-X-PART:", "#EXT-X-MAP:")):
            refs.add(line.split('URI="', 1)[1].split('"', 1)[0])
        elif line and not line.startswith("#"):
            refs.add(line.strip())
    return refs


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
    stream, _, _ = build_stream(keyframes=(True, False, True))
    splitter = FragmentedMP4Splitter()
    for frag in splitter.feed(stream):
        assert frag[4:8] == b"moof"


def test_splitter_ignores_trailing_non_media_boxes() -> None:
    """mfra, sidx and friends must not be emitted as fragments."""
    stream, _, expected = build_stream()
    splitter = FragmentedMP4Splitter()
    assert splitter.feed(stream) == expected


@pytest.mark.parametrize("version", [0, 1])
def test_track_timescale_reads_mdhd(version: int) -> None:
    assert track_timescale(build_init(version=version)) == TIMESCALE


def test_fragment_timing_sums_samples_and_reads_keyframe() -> None:
    key = fragment_timing(build_fragment(0, keyframe=True), TIMESCALE)
    delta = fragment_timing(build_fragment(1, keyframe=False), TIMESCALE)
    assert key.duration == pytest.approx(PART_SECONDS)
    assert key.independent is True
    assert delta.duration == pytest.approx(PART_SECONDS)
    assert delta.independent is False


def test_fragment_timing_uses_tfhd_default_duration() -> None:
    """With no per-sample durations, the tfhd default applies to every sample."""
    tfhd = box("tfhd", struct.pack(">III", 0x020008, 1, 2000))  # default-sample-duration
    trun = box("trun", struct.pack(">IIi", 0x000001, 3, 0))  # three samples, no per-sample fields
    moof = box("moof", box("traf", tfhd + trun))
    timing = fragment_timing(moof + box("mdat", b"\x00"), TIMESCALE)
    assert timing.duration == pytest.approx(3 * 2000 / TIMESCALE)
    # No sample flags anywhere, so the keyframe claim stays unproven.
    assert timing.independent is False


def test_fragment_timing_rejects_fragment_without_durations() -> None:
    tfhd = box("tfhd", struct.pack(">II", 0x020000, 1))
    trun = box("trun", struct.pack(">IIi", 0x000001, 3, 0))
    moof = box("moof", box("traf", tfhd + trun))
    with pytest.raises(ValueError, match="no duration"):
        fragment_timing(moof + box("mdat", b"\x00"), TIMESCALE)


def test_segments_close_on_keyframes(tmp_path: Path) -> None:
    keyframes = (True, False, False, True, False, False, False, True)
    stream, _, frags = build_stream(keyframes=keyframes)
    package(tmp_path, stream)

    playlist = (tmp_path / "stream.m3u8").read_text()
    extinfs = [
        float(line.removeprefix("#EXTINF:").rstrip(","))
        for line in playlist.splitlines()
        if line.startswith("#EXTINF:")
    ]
    # Segments: parts 0-2, parts 3-6, part 7. Each EXTINF is the sum of its parts.
    assert extinfs == pytest.approx([3 * PART_SECONDS, 4 * PART_SECONDS, 1 * PART_SECONDS])
    assert playlist.count("INDEPENDENT=YES") == sum(keyframes)
    assert (tmp_path / "segment00001.m4s").read_bytes() == b"".join(frags[3:7])


def test_long_gop_is_split_at_the_segment_cap(tmp_path: Path) -> None:
    """One keyframe, then a GOP longer than max_segment_seconds (6 s by default).

    Ten parts make 5.625 s. An eleventh would make 6.1875 s, past the cap, so the
    segment closes after ten. EXTINF must never exceed TARGETDURATION.
    """
    stream, _, _ = build_stream(keyframes=(True,) + (False,) * 12)
    package(tmp_path, stream)

    playlist = (tmp_path / "stream.m3u8").read_text()
    assert "#EXT-X-TARGETDURATION:6" in playlist
    extinfs = [
        float(line.removeprefix("#EXTINF:").rstrip(","))
        for line in playlist.splitlines()
        if line.startswith("#EXTINF:")
    ]
    assert extinfs == pytest.approx([10 * PART_SECONDS, 3 * PART_SECONDS])
    assert playlist.count("INDEPENDENT=YES") == 1


def test_header_tags_stay_fixed_while_the_stream_grows(tmp_path: Path) -> None:
    """RFC 8216 section 6.2.1: EXT-X-TARGETDURATION must not change in a live playlist.
    The LL-HLS part target is held to the same rule here."""
    stream, _, _ = build_stream()
    headers: set[tuple[str, ...]] = set()

    def record(out: Path) -> None:
        lines = (out / "stream.m3u8").read_text().splitlines()
        headers.add(
            tuple(
                line
                for line in lines
                if line.startswith(("#EXT-X-TARGETDURATION", "#EXT-X-PART-INF", "#EXT-X-SERVER-CONTROL"))
            )
        )

    package(tmp_path, stream, after_each=record)
    assert len(headers) == 1, f"header tags changed between playlist versions: {headers}"


def test_part_longer_than_declared_part_target_is_rejected(tmp_path: Path) -> None:
    packager = LLHLSPackager(out_dir=tmp_path, part_target=0.5)
    packager.start(build_init())
    with pytest.raises(ValueError, match="longer than the declared PART-TARGET"):
        packager.add_fragment(build_fragment(0))  # 0.5625 s


def test_playlist_never_lists_a_file_before_it_exists(tmp_path: Path) -> None:
    """Checked after every fragment, so a live player never sees a 404."""
    stream, _, _ = build_stream(keyframes=(True, False, True, False, False, True))

    def check(out: Path) -> None:
        playlist = (out / "stream.m3u8").read_text()
        missing = sorted(u for u in referenced_files(playlist) if not (out / u).exists())
        assert not missing, f"playlist names files that were not written: {missing}"

    package(tmp_path, stream, after_each=check)
    assert (tmp_path / "stream.m3u8").read_text().rstrip().endswith("#EXT-X-ENDLIST")


def test_playlist_makes_no_blocking_reload_claim(tmp_path: Path) -> None:
    """A static file server cannot hold a reload, so CAN-BLOCK-RELOAD must not appear."""
    stream, _, _ = build_stream()
    package(tmp_path, stream)
    playlist = (tmp_path / "stream.m3u8").read_text()
    assert "CAN-BLOCK-RELOAD" not in playlist
    assert "PART-HOLD-BACK=" in playlist


def test_forged_trun_count_fails_fast() -> None:
    """A count the fragment cannot hold is refused before it is walked."""
    trun = box("trun", struct.pack(">IIi", 0x000001 | 0x000100, 0xFFFFFFFF, 0))  # no rows
    tfhd = box("tfhd", struct.pack(">II", 0x020000, 1))
    moof = box("moof", box("mfhd", struct.pack(">II", 0, 1)) + box("traf", tfhd + trun))
    with pytest.raises(ValueError, match="ends before them"):
        fragment_timing(moof + box("mdat", b""), TIMESCALE)


def test_trun_cannot_read_rows_from_the_next_box() -> None:
    """A count that only fits by reading the mdat after the trun is refused."""
    trun = box("trun", struct.pack(">IIi", 0x000001 | 0x000100, 3, 0) + struct.pack(">I", SAMPLE_TICKS))
    tfhd = box("tfhd", struct.pack(">II", 0x020000, 1))
    moof = box("moof", box("mfhd", struct.pack(">II", 0, 1)) + box("traf", tfhd + trun))
    mdat = box("mdat", struct.pack(">III", SAMPLE_TICKS, SAMPLE_TICKS, SAMPLE_TICKS))
    with pytest.raises(ValueError, match="ends before them"):
        fragment_timing(moof + mdat, TIMESCALE)


def test_trun_without_per_sample_fields_uses_defaults() -> None:
    """With no per-sample records, the total is count times the tfhd default."""
    trun = box("trun", struct.pack(">IIi", 0x000001, 3, 0))  # data offset only
    tfhd = box("tfhd", struct.pack(">III", 0x020000 | 0x000008, 1, SAMPLE_TICKS))
    moof = box("moof", box("mfhd", struct.pack(">II", 0, 1)) + box("traf", tfhd + trun))
    timing = fragment_timing(moof + box("mdat", b""), TIMESCALE)
    assert timing.duration == pytest.approx(3 * SAMPLE_TICKS / TIMESCALE)
    assert timing.independent is False


def test_report_counts_every_part(tmp_path: Path) -> None:
    stream, _, frags = build_stream()
    packager = package(tmp_path, stream)

    report = packager.report()
    assert report["parts"] == len(frags)
    assert report["bytes"] == sum(len(f) for f in frags)
    assert report["media_seconds"] == pytest.approx(len(frags) * PART_SECONDS, abs=1e-3)


def test_fragment_before_init_is_rejected(tmp_path: Path) -> None:
    packager = LLHLSPackager(out_dir=tmp_path)
    with pytest.raises(RuntimeError):
        packager.add_fragment(b"\x00" * 32)


class _FakeSocket:
    """Stands in for websockets.connect: replays scripted messages, then ends."""

    def __init__(self, messages: list[bytes | str]) -> None:
        self._messages = messages

    async def __aenter__(self) -> _FakeSocket:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def send(self, _message: str) -> None:
        return None

    def __aiter__(self):
        return self._replay()

    async def _replay(self):
        for message in self._messages:
            yield message


def test_server_error_still_closes_playlist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A server error must end the playlist, not leave viewers waiting on it."""
    messages: list[bytes | str] = [
        build_init() + build_fragment(0) + build_fragment(1, keyframe=False),
        json.dumps({"type": "error", "message": "boom"}),
    ]
    fake = types.ModuleType("websockets")
    fake.connect = lambda url, **kwargs: _FakeSocket(messages)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "websockets", fake)

    out = tmp_path / "out"
    args = argparse.Namespace(
        host="127.0.0.1",
        port=8000,
        model="m",
        prompt="p",
        width=640,
        height=384,
        fps=16,
        num_frames=99,
        seed=None,
        guidance_scale=None,
        helios_preset=False,
        extra_params=None,
        timeout=30.0,
        out=str(out),
    )
    assert asyncio.run(hls_client.run(args)) == 1

    playlist = (out / "stream.m3u8").read_text()
    assert playlist.rstrip().endswith("#EXT-X-ENDLIST")
    assert not [u for u in referenced_files(playlist) if not (out / u).exists()]
