# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.example]

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_SCRIPT = REPO_ROOT / "examples" / "offline_inference" / "mage_vl" / "end2end.py"


def _fake_mage_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "Mage"
    mage_vl = repo / "mage_vl"
    mage_vl.mkdir(parents=True)
    (mage_vl / "inference_base.py").write_text("print('base')\n", encoding="utf-8")
    (mage_vl / "inference_streaming.py").write_text("print('streaming')\n", encoding="utf-8")
    return repo


def test_mage_vl_offline_example_builds_codec_command(tmp_path: Path) -> None:
    mage_repo = _fake_mage_repo(tmp_path)
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"fake")
    output_dir = tmp_path / "outputs"

    subprocess.run(
        [
            sys.executable,
            str(EXAMPLE_SCRIPT),
            "--mage-repo",
            str(mage_repo),
            "--task",
            "video",
            "--video-path",
            str(video_path),
            "--video-backend",
            "codec",
            "--codec-engine",
            "traditional",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        check=True,
    )

    with open(output_dir / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["task"] == "video"
    assert summary["dry_run"] is True
    assert "--video-backend" in summary["command"]
    assert "codec" in summary["command"]
    assert "--codec-engine" in summary["command"]
    assert "traditional" in summary["command"]


def test_mage_vl_offline_example_builds_streaming_command(tmp_path: Path) -> None:
    mage_repo = _fake_mage_repo(tmp_path)
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"fake")
    output_dir = tmp_path / "outputs"

    subprocess.run(
        [
            sys.executable,
            str(EXAMPLE_SCRIPT),
            "--mage-repo",
            str(mage_repo),
            "--task",
            "streaming",
            "--video-path",
            str(video_path),
            "--max-segments",
            "1",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        check=True,
    )

    with open(output_dir / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    command = summary["command"]
    assert summary["task"] == "streaming"
    assert str(mage_repo / "mage_vl" / "inference_streaming.py") in command
    assert "--checkpoint" in command
    assert "--max_segments" in command
