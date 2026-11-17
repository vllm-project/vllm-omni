# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import base64
import io
import os
import stat
import subprocess
from pathlib import Path

import av
import numpy as np
import pytest

from tests.helpers import media

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def synthetic_audio(mocker):
    """Keep video tests independent of the host's TTS installation."""
    sample_rate = 48000
    samples = np.arange(sample_rate, dtype=np.float32) / sample_rate
    waveform = (0.1 * np.sin(2 * np.pi * 440 * samples)).reshape(-1, 1)
    return mocker.patch.object(media, "generate_synthetic_audio", return_value={"np_array": waveform})


def test_audio_generation_failure_is_not_cached(mocker, tmp_path):
    failure = RuntimeError("TTS dependency unavailable")
    mocker.patch.object(media, "generate_synthetic_audio", side_effect=failure)

    with pytest.raises(RuntimeError, match="TTS dependency unavailable") as caught:
        media.generate_synthetic_video(64, 64, 6, embed_audio=True, cache_dir=tmp_path)

    assert caught.value is failure
    assert not list(tmp_path.glob("*.mp4"))


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("FFmpeg unavailable"),
        subprocess.CalledProcessError(1, ["ffmpeg"]),
        subprocess.TimeoutExpired(["ffmpeg"], 300),
    ],
    ids=["missing-ffmpeg", "ffmpeg-error", "ffmpeg-timeout"],
)
def test_audio_mux_failure_is_propagated(mocker, synthetic_audio, failure):
    mocker.patch("imageio_ffmpeg.get_ffmpeg_exe", return_value="ffmpeg")
    mocker.patch.object(media.subprocess, "run", side_effect=failure)

    with pytest.raises(type(failure)) as caught:
        media._mux_mp4_bytes_with_synthetic_audio(b"video", num_frames=6)

    assert caught.value is failure


def test_audio_in_video_repairs_video_only_cache(synthetic_audio, tmp_path):
    video_only = media.generate_synthetic_video(64, 64, 6, cache_dir=tmp_path)
    # Reproduce a cache entry written by the old silent audio-mux fallback.
    cache_path = tmp_path / "synth_video_w64_h64_nf6_ea1.mp4"
    original_bytes = Path(video_only["file_path"]).read_bytes()
    cache_path.write_bytes(original_bytes)

    result = media.generate_synthetic_video(64, 64, 6, embed_audio=True, cache_dir=tmp_path)

    assert result["file_path"] == str(cache_path)
    raw = base64.b64decode(result["base64"])
    assert raw == cache_path.read_bytes()
    assert raw != original_bytes
    assert result["np_array"].shape == (6, 64, 64, 3)
    with av.open(io.BytesIO(raw)) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 1
        audio_frames = list(container.decode(audio=0))
        assert sum(frame.samples for frame in audio_frames) > 0
        assert any(np.any(frame.to_ndarray()) for frame in audio_frames)


@pytest.mark.parametrize("embed_audio", [False, True], ids=["video-only", "with-audio"])
def test_valid_video_cache_is_reused(mocker, synthetic_audio, tmp_path, embed_audio):
    result = media.generate_synthetic_video(64, 64, 6, embed_audio=embed_audio, cache_dir=tmp_path)
    encode = mocker.patch("imageio.get_writer", side_effect=AssertionError("valid cache was regenerated"))

    cached = media.generate_synthetic_video(64, 64, 6, embed_audio=embed_audio, cache_dir=tmp_path)

    encode.assert_not_called()
    assert cached["base64"] == result["base64"]
    assert cached["file_path"] == result["file_path"]
    assert np.array_equal(cached["np_array"], media._np_array_from_mp4_bytes(base64.b64decode(result["base64"])))


@pytest.mark.parametrize(
    "corruption, embed_audio",
    [
        pytest.param("empty", False, id="empty-video-only"),
        pytest.param("empty", True, id="empty-with-audio"),
        pytest.param("invalid-header", False, id="invalid-header-video-only"),
        pytest.param("invalid-header", True, id="invalid-header-with-audio"),
        pytest.param("truncated-audio", True, id="truncated-audio"),
    ],
)
def test_corrupt_video_cache_is_regenerated(synthetic_audio, tmp_path, corruption, embed_audio):
    result = media.generate_synthetic_video(64, 64, 6, embed_audio=embed_audio, cache_dir=tmp_path)
    cache_path = Path(result["file_path"])
    raw = cache_path.read_bytes()
    if corruption == "empty":
        corrupt = b""
    elif corruption == "invalid-header":
        corrupt = b"not an MP4"
    else:
        with av.open(io.BytesIO(raw)) as container:
            packets = [
                (packet.stream.type, packet.pos, packet.size)
                for packet in container.demux()
                if packet.pos is not None and packet.size
            ]
        video_end = max(pos + size for kind, pos, size in packets if kind == "video")
        audio_tail = next(pos for kind, pos, _ in packets if kind == "audio" and pos >= video_end)
        corrupt = raw[:audio_tail]
        # Keep every video packet and truncate at an audio packet boundary:
        # neither video completeness nor a decode exception can reject this.
        assert len(media._np_array_from_mp4_bytes(corrupt)) == 6
        with av.open(io.BytesIO(corrupt)) as container:
            stream = container.streams.audio[0]
            samples = sum(frame.samples for frame in container.decode(audio=0))
            assert 0 < samples < stream.duration * stream.time_base * stream.codec_context.sample_rate
    if not embed_audio:
        with pytest.raises(RuntimeError):
            media._np_array_from_mp4_bytes(corrupt)
    cache_path.write_bytes(corrupt)

    repaired = media.generate_synthetic_video(64, 64, 6, embed_audio=embed_audio, cache_dir=tmp_path)

    repaired_bytes = base64.b64decode(repaired["base64"])
    assert repaired_bytes == cache_path.read_bytes()
    assert repaired_bytes != corrupt
    assert repaired["np_array"].shape == (6, 64, 64, 3)
    with av.open(io.BytesIO(repaired_bytes)) as container:
        if embed_audio:
            assert sum(frame.samples for frame in container.decode(audio=0)) > 0
        else:
            assert not container.streams.audio
            synthetic_audio.assert_not_called()


@pytest.mark.parametrize("embed_audio", [False, True], ids=["video-only", "with-audio"])
def test_incomplete_video_cache_is_regenerated(synthetic_audio, tmp_path, embed_audio):
    result = media.generate_synthetic_video(64, 64, 6, embed_audio=embed_audio, cache_dir=tmp_path)
    cache_path = Path(result["file_path"])
    if embed_audio:
        # Keep a complete audio stream while violating the six-frame cache key.
        shorter = media.generate_synthetic_video(64, 64, 5, embed_audio=True, cache_dir=tmp_path)
        corrupt = base64.b64decode(shorter["base64"])
    else:
        raw = cache_path.read_bytes()
        with av.open(io.BytesIO(raw)) as container:
            last_video = max(
                packet.pos for packet in container.demux(video=0) if packet.pos is not None and packet.size
            )
        corrupt = raw[:last_video]
    # Both modes must reach frame-count validation rather than a decode failure.
    assert 0 < len(media._np_array_from_mp4_bytes(corrupt)) < 6
    if embed_audio:
        with av.open(io.BytesIO(corrupt)) as container:
            stream = container.streams.audio[0]
            samples = sum(frame.samples for frame in container.decode(audio=0))
            assert samples > 0
            assert samples >= stream.duration * stream.time_base * stream.codec_context.sample_rate
    cache_path.write_bytes(corrupt)

    repaired = media.generate_synthetic_video(64, 64, 6, embed_audio=embed_audio, cache_dir=tmp_path)

    repaired_bytes = base64.b64decode(repaired["base64"])
    assert repaired["np_array"].shape == (6, 64, 64, 3)
    assert repaired_bytes == cache_path.read_bytes()
    assert repaired_bytes != corrupt
    assert len(media._np_array_from_mp4_bytes(repaired_bytes)) == 6
    with av.open(io.BytesIO(repaired_bytes)) as container:
        if embed_audio:
            assert sum(frame.samples for frame in container.decode(audio=0)) > 0
        else:
            assert not container.streams.audio
            synthetic_audio.assert_not_called()


def test_video_cache_is_published_atomically(monkeypatch, synthetic_audio, tmp_path):
    original = media.generate_synthetic_video(64, 64, 6, embed_audio=True, cache_dir=tmp_path)
    cache_path = Path(original["file_path"])
    old_bytes = cache_path.read_bytes()
    replace = media.os.replace
    published = []

    def observe_replace(source, destination):
        source = Path(source)
        assert Path(destination) == cache_path
        assert source.parent == cache_path.parent
        # Until publication, readers must still see the old complete MP4.
        assert cache_path.read_bytes() == old_bytes
        completed = source.read_bytes()
        with av.open(io.BytesIO(completed)) as container:
            assert len(list(container.decode(video=0))) == 6
        with av.open(io.BytesIO(completed)) as container:
            assert sum(frame.samples for frame in container.decode(audio=0)) > 0
        published.append(completed)
        replace(source, destination)

    monkeypatch.setattr(media.os, "replace", observe_replace)
    result = media.generate_synthetic_video(64, 64, 6, embed_audio=True, force_regenerate=True, cache_dir=tmp_path)

    assert published == [cache_path.read_bytes()]
    assert base64.b64decode(result["base64"]) == published[0]
    assert list(tmp_path.iterdir()) == [cache_path]


def test_failed_cache_publication_preserves_old_file(mocker, synthetic_audio, tmp_path):
    original = media.generate_synthetic_video(64, 64, 6, embed_audio=True, cache_dir=tmp_path)
    cache_path = Path(original["file_path"])
    old_bytes = cache_path.read_bytes()
    mocker.patch.object(media.os, "replace", side_effect=OSError("publication failed"))

    with pytest.raises(OSError, match="publication failed"):
        media.generate_synthetic_video(64, 64, 6, embed_audio=True, force_regenerate=True, cache_dir=tmp_path)

    assert cache_path.read_bytes() == old_bytes
    assert list(tmp_path.iterdir()) == [cache_path]


@pytest.mark.parametrize("creation_umask", [0o022, 0o002, 0o077])
def test_video_cache_respects_creation_permissions(synthetic_audio, tmp_path, creation_umask):
    previous_umask = os.umask(creation_umask)
    try:
        result = media.generate_synthetic_video(64, 64, 6, embed_audio=True, cache_dir=tmp_path)
    finally:
        os.umask(previous_umask)

    cache_path = Path(result["file_path"])
    assert stat.S_IMODE(cache_path.stat().st_mode) == 0o666 & ~creation_umask
