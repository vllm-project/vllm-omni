# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import shutil
import subprocess
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path

import numpy as np
import pytest

from vllm_omni.model_executor.models.minimax_h3 import reference_video

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def reference_clip(tmp_path):
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("FFmpeg and ffprobe are required")
    encoders = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], text=True)
    if "libx264rgb" not in encoders:
        pytest.skip("FFmpeg with libx264rgb is required")
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=330x254:rate=30:duration=3.5",
            "-vf",
            "pad=330:256",
            "-c:v",
            "libx264",
            "-threads:v",
            "1",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
    )
    return clip


@pytest.mark.parametrize(("start", "frame_limit", "expected_frames"), [(0.0, 0, 84), (0.5, 49, 49)])
def test_ref2va_direct_rgb_matches_lossless_roundtrip(
    tmp_path, monkeypatch, reference_clip, start, frame_limit, expected_frames
):
    monkeypatch.setitem(sys.modules, "decord", None)
    prepared_common = {"target_frame_count": frame_limit, "start_time_seconds": start}
    file_result = reference_video.prepare_reference_videos(
        reference_clip, workdir=str(tmp_path / "file"), **prepared_common
    )[0]
    direct = reference_video.prepare_reference_videos(
        reference_clip, workdir=str(tmp_path / "direct"), decode_frames=True, **prepared_common
    )[0]
    file_frames = reference_video.load_video_frames(file_result["prepared_path"])
    direct_frames = direct.pop("decoded_frames")

    assert direct_frames.shape == (expected_frames, 256, 320, 3)
    assert direct_frames.dtype == np.uint8
    np.testing.assert_array_equal(direct_frames, file_frames)
    assert Path(file_result.pop("prepared_path")).is_file()
    assert direct == file_result
    assert not list((tmp_path / "direct").rglob("*.mp4"))

    expected = reference_video.sample_reference_video_frames(str(reference_clip), decoded_frames=file_frames)
    actual = reference_video.sample_reference_video_frames(str(reference_clip), decoded_frames=direct_frames)
    np.testing.assert_array_equal(actual["frames"], expected["frames"])
    assert actual["block_timestamps"] == expected["block_timestamps"]


@pytest.mark.parametrize("has_decord", [False, True], ids=["multiple-clips", "decord"])
def test_ref2va_preserves_file_backed_reader(tmp_path, monkeypatch, reference_clip, has_decord):
    monkeypatch.setattr(
        reference_video, "find_spec", lambda name: ModuleSpec(name, loader=None) if has_decord else None
    )
    clips = [reference_clip] if has_decord else [reference_clip, reference_clip]
    prepared = reference_video.prepare_reference_videos(
        clips, target_frame_count=48, workdir=str(tmp_path / "file"), decode_frames=True
    )
    assert len(prepared) == len(clips)
    assert all(Path(item["prepared_path"]).is_file() and "decoded_frames" not in item for item in prepared)
