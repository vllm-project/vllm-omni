# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import subprocess

import pytest
import torch

from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.models.minimax_h3 import reference_video

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _video_meta(
    *,
    width: int,
    height: int,
    frame_count: int,
    fps: float = 24.0,
    has_audio: bool = False,
) -> dict[str, object]:
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": frame_count / fps if fps > 0 else 0.0,
        "format_names": ("matroska",),
        "video_codec": "vp9",
        "audio_codecs": ("opus",) if has_audio else (),
        "file_size": 1024,
    }


def test_prepare_edit_video_pads_short_ffmpeg_readable_source(monkeypatch, tmp_path):
    source = tmp_path / "short-source.webm"
    workdir = tmp_path / "prepared"
    prepared_path = workdir / "prepared.mp4"
    probes = {
        str(source): _video_meta(
            width=160,
            height=90,
            frame_count=12,
            has_audio=True,
        ),
        str(prepared_path): _video_meta(
            width=448,
            height=256,
            frame_count=90,
        ),
    }
    transcode_calls: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(reference_video, "_probe_video", lambda path: probes[str(path)])

    def fake_transcode(source_path, **kwargs):
        transcode_calls.append((source_path, kwargs))
        return str(prepared_path)

    monkeypatch.setattr(reference_video, "_transcode_reference_video", fake_transcode)

    descriptor = reference_video.prepare_edit_video(
        source,
        target_width=448,
        target_height=256,
        target_frame_count=90,
        workdir=str(workdir),
    )

    assert workdir.is_dir()
    assert descriptor == {
        "original_path": str(source),
        "prepared_path": str(prepared_path),
        "input_has_audio": True,
        "width": 448,
        "height": 256,
        "frame_count": 90,
    }
    assert transcode_calls == [
        (
            str(source),
            {
                "target_width": 448,
                "target_height": 256,
                "target_frame_count": 90,
                "workdir": str(workdir),
                "pad_last_frame": True,
            },
        )
    ]


def test_edit_transcode_clones_last_frame_and_emits_lossless_rgb(monkeypatch, tmp_path):
    commands: list[list[str]] = []
    monkeypatch.setattr(
        reference_video.subprocess,
        "run",
        lambda command, **_kwargs: commands.append(command),
    )

    output = reference_video._transcode_reference_video(
        "input.avi",
        target_width=448,
        target_height=256,
        target_frame_count=73,
        workdir=str(tmp_path),
        pad_last_frame=True,
    )

    assert output == str(tmp_path / "prepared.mp4")
    command = commands[0]
    assert command[command.index("-vf") + 1] == (
        "fps=24,scale=448:256:flags=lanczos,setsar=1,tpad=stop_mode=clone:stop=-1"
    )
    assert command[command.index("-frames:v") + 1] == "73"
    assert command[command.index("-c:v") + 1] == "libx264rgb"
    assert command[command.index("-crf") + 1] == "0"
    assert command[command.index("-pix_fmt") + 1] == "rgb24"
    assert "-an" in command


@pytest.mark.parametrize(
    ("value", "kwargs", "message"),
    [
        (["source.mp4"], {}, "single file path"),
        ("", {}, "non-empty file path"),
        ("source.mp4", {"target_width": 0}, "target_width must be a positive integer"),
        ("source.mp4", {"target_frame_count": 0}, "target_frame_count must be a positive integer"),
    ],
)
def test_prepare_edit_video_rejects_invalid_arguments(value, kwargs, message, tmp_path):
    arguments = {
        "target_width": 448,
        "target_height": 256,
        "target_frame_count": 73,
        "workdir": str(tmp_path),
        **kwargs,
    }

    with pytest.raises(OmniClientError, match=message):
        reference_video.prepare_edit_video(value, **arguments)


def test_prepare_edit_video_rejects_empty_video_stream(monkeypatch, tmp_path):
    monkeypatch.setattr(
        reference_video,
        "_probe_video",
        lambda _path: _video_meta(width=160, height=90, frame_count=0),
    )

    with pytest.raises(OmniClientError, match="video has no frames"):
        reference_video.prepare_edit_video(
            "empty.mp4",
            target_width=448,
            target_height=256,
            target_frame_count=73,
            workdir=str(tmp_path),
        )


def test_prepare_edit_video_maps_probe_and_transcode_failures_to_client_errors(
    monkeypatch,
    tmp_path,
):
    def fail_probe(_path):
        raise subprocess.CalledProcessError(1, ["ffprobe"])

    monkeypatch.setattr(reference_video, "_probe_video", fail_probe)
    with pytest.raises(OmniClientError, match="cannot inspect MiniMax H3 edit video"):
        reference_video.prepare_edit_video(
            "unreadable.bin",
            target_width=448,
            target_height=256,
            target_frame_count=73,
            workdir=str(tmp_path),
        )

    monkeypatch.setattr(
        reference_video,
        "_probe_video",
        lambda _path: _video_meta(width=160, height=90, frame_count=12),
    )

    def fail_transcode(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, ["ffmpeg"])

    monkeypatch.setattr(reference_video, "_transcode_reference_video", fail_transcode)
    with pytest.raises(OmniClientError, match="cannot prepare MiniMax H3 edit video"):
        reference_video.prepare_edit_video(
            "broken.mp4",
            target_width=448,
            target_height=256,
            target_frame_count=73,
            workdir=str(tmp_path),
        )


def test_prepare_edit_video_verifies_exact_output_shape(monkeypatch, tmp_path):
    prepared_path = tmp_path / "prepared.mp4"
    probes = iter(
        [
            _video_meta(width=160, height=90, frame_count=200),
            _video_meta(width=448, height=256, frame_count=72),
        ]
    )
    monkeypatch.setattr(reference_video, "_probe_video", lambda _path: next(probes))
    monkeypatch.setattr(
        reference_video,
        "_transcode_reference_video",
        lambda *_args, **_kwargs: str(prepared_path),
    )

    with pytest.raises(
        OmniClientError,
        match="does not match the requested 24 FPS 448x256x73 shape",
    ):
        reference_video.prepare_edit_video(
            "long.mov",
            target_width=448,
            target_height=256,
            target_frame_count=73,
            workdir=str(tmp_path),
        )


def test_prepare_edit_video_is_publicly_exported():
    assert "prepare_edit_video" in reference_video.__all__
    assert reference_video.prepare_edit_video is not None


def test_duration_capped_audio_load_uses_bounded_ffmpeg_decode(monkeypatch):
    commands: list[tuple[list[str], dict[str, object]]] = []
    expected = (torch.ones(2, 17), 32000)

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))

    def fake_soundfile_load(path):
        assert path.endswith("/audio.wav")
        return expected

    monkeypatch.setattr(reference_video.subprocess, "run", fake_run)
    monkeypatch.setattr(reference_video, "_soundfile_to_waveform", fake_soundfile_load)

    actual = reference_video.load_audio_file(
        "large-source.mp3",
        duration_seconds=2.75,
    )

    assert actual is expected
    assert len(commands) == 1
    command, kwargs = commands[0]
    assert kwargs == {"check": True}
    assert command[command.index("-i") + 1] == "large-source.mp3"
    assert command[command.index("-map") + 1] == "0:a:0"
    assert command[command.index("-t") + 1] == "2.750000"
    assert command[command.index("-ac") + 1] == "2"
    assert command[command.index("-ar") + 1] == "32000"
    assert command[command.index("-c:a") + 1] == "pcm_f32le"
    assert "-nostdin" in command


@pytest.mark.parametrize("duration", [True, 0, -1, float("inf"), float("nan"), "bad"])
def test_duration_capped_audio_load_rejects_invalid_duration(duration):
    with pytest.raises(OmniClientError, match="duration must be positive"):
        reference_video.load_audio_file("source.wav", duration_seconds=duration)
