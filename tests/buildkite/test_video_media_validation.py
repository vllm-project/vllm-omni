# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate complete decoded media independently from reported video metrics."""

import importlib.util
import json
from pathlib import Path

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
