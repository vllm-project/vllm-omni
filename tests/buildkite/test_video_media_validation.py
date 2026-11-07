# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate complete decoded media independently from reported video metrics."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "video_validation", ROOT / "tests/dfx/perf/scripts/validate_sglang_videos.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write_video(path: Path, frames: int = 4):
    av = pytest.importorskip("av")
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=16)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        for _ in range(frames):
            frame = av.VideoFrame(64, 64, "yuv420p")
            for plane in frame.planes:
                plane.update(bytes(plane.buffer_size))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def media_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            [
                {
                    "test_name": "video-test",
                    "benchmark_params": [
                        {
                            "num_prompts": 1,
                            "num_warmups": 1,
                            "extra_body": {"width": 64, "height": 64, "num_frames": 4, "fps": 16},
                        }
                    ],
                }
            ]
        )
    )
    return path


def test_complete_videos_include_warmup_and_record_provenance(tmp_path, media_config, monkeypatch):
    write_video(tmp_path / "warmup.mp4")
    write_video(tmp_path / "measured.mp4")
    monkeypatch.setenv("BUILDKITE_JOB_ID", "job-id")
    result = module.validate_videos(tmp_path, media_config)
    assert result["status"] == "passed"
    assert result["job_id"] == "job-id"
    assert len(result["files"]) == 2
    assert all(f["frames"] == 4 and f["fps"] == 16 and len(f["sha256"]) == 64 for f in result["files"])


def test_missing_output_cannot_pass_on_reported_success(tmp_path, media_config):
    write_video(tmp_path / "only-warmup.mp4")
    with pytest.raises(ValueError, match="Expected 2 videos"):
        module.validate_videos(tmp_path, media_config)


def test_short_video_is_rejected(tmp_path, media_config):
    write_video(tmp_path / "warmup.mp4")
    write_video(tmp_path / "measured.mp4", frames=3)
    with pytest.raises(ValueError, match="Unexpected media"):
        module.validate_videos(tmp_path, media_config)


@pytest.fixture
def audio_config(media_config):
    (config,) = json.loads(media_config.read_text())
    config["media_expectations"] = {
        "width": 64,
        "height": 64,
        "frames": 4,
        "fps": 16,
        "require_audio": True,
        "audio_sample_rate": 32000,
        "audio_channels": 2,
        "audio_video_tolerance_s": 0.25,
    }
    # Native MiniMax accepts target timing, and rejects num_frames/fps in the body.
    config["benchmark_params"][0]["extra_body"] = {
        "task": "t2va",
        "target": {"short_edge": 64, "aspect_ratio": "1:1", "duration_seconds": 0.25},
    }
    media_config.write_text(json.dumps([config]))
    return media_config


def write_audio_video(
    path: Path,
    *,
    silent: bool = False,
    video_frames: int = 4,
    audio_samples: int = 8000,
    sample_rate: int = 32000,
    channels: int = 2,
):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    with av.open(str(path), "w") as container:
        video = container.add_stream("mpeg4", rate=16)
        video.width = video.height = 64
        video.pix_fmt = "yuv420p"
        audio = container.add_stream("aac", rate=sample_rate)
        layout = "stereo" if channels == 2 else "mono"
        audio.layout = layout
        for _ in range(video_frames):
            frame = av.VideoFrame(64, 64, "yuv420p")
            for plane in frame.planes:
                plane.update(bytes(plane.buffer_size))
            for packet in video.encode(frame):
                container.mux(packet)
        # Multiple audio frames make checking only the first decoded frame insufficient.
        for offset in range(0, audio_samples, 1024):
            count = min(1024, audio_samples - offset)
            signal = (
                np.zeros(count, dtype=np.float32)
                if silent
                else (0.25 * np.sin(2 * np.pi * 440 * np.arange(offset, offset + count) / sample_rate)).astype(
                    np.float32
                )
            )
            frame = av.AudioFrame.from_ndarray(np.stack([signal] * channels), format="fltp", layout=layout)
            frame.sample_rate = sample_rate
            frame.pts = offset
            for packet in audio.encode(frame):
                container.mux(packet)
        for stream in (video, audio):
            for packet in stream.encode():
                container.mux(packet)


def test_native_target_uses_media_expectations_and_decodes_all_audio(tmp_path, audio_config):
    av = pytest.importorskip("av")
    for name in ("warmup.mp4", "measured.mp4"):
        write_audio_video(tmp_path / name)
    result = module.validate_videos(tmp_path, audio_config)
    for item in result["files"]:
        with av.open(str(tmp_path / item["path"])) as container:
            frames = list(container.decode(container.streams.audio[0]))
        assert len(frames) > 1
        assert item["audio_samples"] == sum(frame.samples for frame in frames)
        assert item["audio_samples"] > frames[0].samples
        assert item["audio_sample_rate"] == 32000
        assert item["audio_channels"] == 2
        assert item["audio_duration_s"] == item["audio_samples"] / 32000
        assert item["frames"] == 4


@pytest.mark.parametrize("silent", [False, True], ids=["missing-audio", "silent-audio"])
def test_required_audio_rejects_missing_or_silent_track(tmp_path, audio_config, silent):
    write_audio_video(tmp_path / "warmup.mp4")
    if silent:
        write_audio_video(tmp_path / "measured.mp4", silent=True)
    else:
        write_video(tmp_path / "measured.mp4")
    with pytest.raises(ValueError, match="silent audio" if silent else "Missing audio"):
        module.validate_videos(tmp_path, audio_config)


@pytest.mark.parametrize(
    "overrides",
    [
        {"audio_samples": 1024},
        {"sample_rate": 16000, "audio_samples": 16000},
        {"channels": 1},
    ],
    ids=["short-audio", "wrong-sample-rate", "mono-audio"],
)
def test_audio_output_contract_is_enforced(tmp_path, audio_config, overrides):
    (config,) = json.loads(audio_config.read_text())
    config["media_expectations"]["frames"] = 16
    config["benchmark_params"][0]["extra_body"]["target"]["duration_seconds"] = 1
    audio_config.write_text(json.dumps([config]))
    write_audio_video(tmp_path / "warmup.mp4", video_frames=16, audio_samples=32000)
    options = {"video_frames": 16, "audio_samples": 32000, **overrides}
    write_audio_video(tmp_path / "measured.mp4", **options)
    with pytest.raises(ValueError, match="Unexpected audio rate, channels or duration"):
        module.validate_videos(tmp_path, audio_config)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_last_audio_frame_is_rejected(tmp_path, audio_config, monkeypatch, invalid):
    av = pytest.importorskip("av")
    for name in ("warmup.mp4", "measured.mp4"):
        write_audio_video(tmp_path / name)
    original_open = av.open
    visited = []

    class CorruptLastAudioFrame:
        def __init__(self, path):
            self.container = original_open(path)
            self.streams = self.container.streams

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.container.close()

        def decode(self, stream):
            if stream.type != "audio":
                yield from self.container.decode(stream)
                return
            frames = list(self.container.decode(stream))
            assert len(frames) > 1
            for frame in frames[:-1]:
                visited.append("valid")
                yield frame
            frame = frames[-1]
            values = frame.to_ndarray()
            values.flat[-1] = invalid
            visited.append("invalid")
            yield SimpleNamespace(sample_rate=frame.sample_rate, samples=frame.samples, to_ndarray=lambda: values)

    # Inject after decoding: lossy AAC encoding can discard NaN/Inf test samples.
    monkeypatch.setattr(av, "open", CorruptLastAudioFrame)
    with pytest.raises(ValueError, match="Invalid audio"):
        module.validate_videos(tmp_path, audio_config)
    assert visited[-1] == "invalid"
    assert "valid" in visited
