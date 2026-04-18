# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Parity test for DreamZero video preprocessing order.

The upstream eager path in
`groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py:952-966`
casts the input video to `bfloat16` *before* applying `normalize_video`
(`x * 2 - 1`). That cast order matters on CUDA and must be preserved for
end-to-end parity.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import (
    DreamZeroPipeline,
)
from vllm_omni.diffusion.models.dreamzero.transform.roboarena import (
    RoboArenaTransform,
)

DREAMZERO_REPO = Path(os.environ.get("DREAMZERO_REPO", "~/code/dreamzero")).expanduser()
PROMPT = "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan"
SESSION_ID = "video-preprocess-source-parity"

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required"),
    pytest.mark.skipif(not DREAMZERO_REPO.exists(), reason="DreamZero source repo is required at ~/code/dreamzero"),
]


def _load_real_video() -> torch.Tensor:
    if str(DREAMZERO_REPO) not in sys.path:
        sys.path.insert(0, str(DREAMZERO_REPO))

    import test_client_AR as dreamzero_client

    camera_frames = dreamzero_client.load_camera_frames()
    obs = dreamzero_client._make_obs_from_video(camera_frames, [0], PROMPT, SESSION_ID)
    stitched = RoboArenaTransform().transform_input(obs)["images"]
    return torch.from_numpy(stitched).unsqueeze(0).to(device="cuda:0")


def test_preprocess_video_matches_source_bf16_cast_order() -> None:
    videos = _load_real_video()  # uint8 [B, T, H, W, C]

    actual = DreamZeroPipeline._preprocess_video(None, videos).float()

    expected = videos.permute(0, 4, 1, 2, 3)
    expected = expected.float() / 255.0
    expected = expected.to(dtype=torch.bfloat16)
    batch_size, channels, num_frames, height, width = expected.shape
    expected = expected.permute(0, 2, 1, 3, 4)
    expected = expected.reshape(batch_size * num_frames, channels, height, width)
    expected = expected * 2.0 - 1.0
    expected = expected.reshape(batch_size, num_frames, channels, height, width)
    expected = expected.permute(0, 2, 1, 3, 4).to(dtype=torch.bfloat16).float()

    diff = (actual - expected).abs()
    assert diff.max().item() == 0.0
    assert diff.mean().item() == 0.0
