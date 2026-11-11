# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import ast
import json
import math
import os
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.cpu, pytest.mark.core_model, pytest.mark.diffusion]
ROOT = Path(__file__).resolve().parents[4]


class OmniClientError(ValueError):
    pass


def load_module() -> ModuleType:
    path = ROOT / "vllm_omni/model_executor/models/minimax_h3/reference_video.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    module = ModuleType("reference_video_under_test")
    module.__dict__.update(
        Any=Any,
        OmniClientError=OmniClientError,
        Path=Path,
        json=json,
        math=math,
        np=np,
        os=os,
        subprocess=subprocess,
        torch=torch,
    )
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


reference_video = load_module()


def _descriptor_item(tmp_path):
    return {
        "original_path": "source.mp4",
        "prepared_path": "prepared.mp4",
        "input_has_audio": False,
        "width": 2,
        "height": 2,
        "start_time_seconds": 0.0,
        "duration_seconds": 2.0,
        "audio_duration_seconds": 2.0,
        "rgb_path": str(tmp_path / "prepared.rgb"),
        "frame_count": 2,
    }


def test_reference_rgb_reuse_gate_is_strict(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_H3_REF_VIDEO_DECODE_REUSE", raising=False)
    assert not reference_video.reference_video_transcode_rgb_reuse_enabled()
    monkeypatch.setenv("VLLM_OMNI_H3_REF_VIDEO_DECODE_REUSE", "1")
    assert reference_video.reference_video_transcode_rgb_reuse_enabled()
    monkeypatch.setenv("VLLM_OMNI_H3_REF_VIDEO_DECODE_REUSE", "true")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        reference_video.reference_video_transcode_rgb_reuse_enabled()


def test_prepared_reference_descriptor_round_trips_rgb_sidecar(tmp_path):
    item = _descriptor_item(tmp_path)
    encoded = reference_video.serialize_prepared_reference_videos([item], str(tmp_path))
    assert reference_video.deserialize_prepared_reference_videos(encoded) == (str(tmp_path), [item])

    payload = json.loads(encoded)
    del payload["videos"][0]["frame_count"]
    with pytest.raises(ValueError, match="invalid MiniMax H3"):
        reference_video.deserialize_prepared_reference_videos(json.dumps(payload))


def test_prepared_rgb_sidecar_validates_exact_size(tmp_path):
    item = _descriptor_item(tmp_path)
    expected = np.arange(24, dtype=np.uint8).reshape(2, 2, 2, 3)
    expected.tofile(item["rgb_path"])
    actual = reference_video.load_prepared_rgb_frames(item)
    np.testing.assert_array_equal(actual, expected)

    with open(item["rgb_path"], "ab") as output:
        output.write(b"x")
    with pytest.raises(OmniClientError, match="expected 24"):
        reference_video.load_prepared_rgb_frames(item)


def test_transcode_can_emit_video_and_raw_rgb(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(reference_video.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    rgb_path = str(tmp_path / "prepared.rgb")
    reference_video._transcode_reference_video(
        "source.mp4",
        target_width=32,
        target_height=32,
        target_frame_count=4,
        workdir=str(tmp_path),
        rgb_output_path=rgb_path,
    )
    command, kwargs = calls[0]
    assert "-filter_complex" in command
    assert "split=2[prepared][rgb]" in command[command.index("-filter_complex") + 1]
    assert command[-1] == rgb_path
    assert kwargs["stdin"] is reference_video.subprocess.DEVNULL
