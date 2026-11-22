# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Prepared sidecars and video output for the offline Bimanual runner."""

import json

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual import offline_bimanual

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_cli_reads_prepared_sidecar(tmp_path):
    sidecar = {"action_space": "normalized_unified_v1", "action": [[0.0] * 59] * 8, "domain_names": ["agibotworld"]}
    (tmp_path / "actions.json").write_text(json.dumps(sidecar))
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps({"action_path": "actions.json", "vision_path": "initial.png"}) + "\n")
    record = offline_bimanual._load_record(path, 0)
    assert record["action_space"] == "normalized_unified_v1" and record["image"] == "initial.png"
    assert record["action"] == sidecar["action"]
    assert record["domain_names"] == sidecar["domain_names"]


def test_runner_bfloat16_video_conversion() -> None:
    frames = offline_bimanual._video_frames(torch.full((1, 3, 5, 32, 32), -1, dtype=torch.bfloat16))
    assert len(frames) == 5 and frames[0].shape == (32, 32, 3)
    assert np.count_nonzero(frames) == 0


def test_unwrap_video_envelopes() -> None:
    video = torch.zeros(3, 5, 8, 8)
    assert offline_bimanual._unwrap_video([{"video": video}]) is video
    assert offline_bimanual._unwrap_video({"payload": {"video": video}, "metadata": {}}) is video
    with pytest.raises(ValueError, match="empty output list"):
        offline_bimanual._unwrap_video([])
    with pytest.raises(ValueError, match="Unsupported generation output mapping"):
        offline_bimanual._unwrap_video({"metadata": {}})
