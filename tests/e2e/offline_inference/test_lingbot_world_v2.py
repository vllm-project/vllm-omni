# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end smoke test for LingBot-World v2 generation."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunnerHandler
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = os.environ.get(
    "VLLM_OMNI_LINGBOT_WORLD_V2_CHECKPOINT_PATH",
    "robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
)
_IMAGE_PATH = os.environ.get("VLLM_OMNI_LINGBOT_WORLD_V2_IMAGE_PATH")
_ACTION_DIR_VALUE = os.environ.get("VLLM_OMNI_LINGBOT_WORLD_V2_ACTION_DIR")
_ACTION_DIR = Path(_ACTION_DIR_VALUE).expanduser().resolve() if _ACTION_DIR_VALUE else None
_HAS_INPUT_ASSETS = _IMAGE_PATH is not None and _ACTION_DIR is not None

_OMNI_RUNNER_PARAM = (
    MODEL,
    None,
    {
        "flow_shift": 5.0,
        "parallel_config": DiffusionParallelConfig(tensor_parallel_size=1),
        "model_config": {
            "lingbot_action_root": str(_ACTION_DIR.parent) if _ACTION_DIR is not None else "",
        },
    },
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.diffusion,
    pytest.mark.skipif(
        not _HAS_INPUT_ASSETS,
        reason=("VLLM_OMNI_LINGBOT_WORLD_V2_IMAGE_PATH and VLLM_OMNI_LINGBOT_WORLD_V2_ACTION_DIR are required"),
    ),
    pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True),
]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_lingbot_world_v2_tp1_one_block(omni_runner_handler: OmniRunnerHandler) -> None:
    """Load the real checkpoint and generate one causal block on one GPU."""

    assert _IMAGE_PATH is not None
    assert _ACTION_DIR is not None
    response = omni_runner_handler.send_diffusion_request(
        {
            "model": omni_runner_handler.runner.model_name,
            "prompt": "The camera moves slowly forward through the scene.",
            "multi_modal_data": {"image": str(Path(_IMAGE_PATH).expanduser().resolve())},
            "sampling_params": OmniDiffusionSamplingParams(
                height=480,
                width=832,
                num_frames=9,
                num_inference_steps=4,
                max_sequence_length=512,
                seed=42,
                fps=16,
                extra_args={
                    "action_path": _ACTION_DIR.name,
                    "flow_shift": 5.0,
                },
            ),
        }
    )

    assert response.success
    assert response.images is not None
    assert len(response.images) == 1
    video = np.asarray(response.images[0])
    assert video.shape == (1, 9, 480, 832, 3)
    assert np.isfinite(video).all()
    assert float(np.ptp(video)) > 1e-6
    assert float(np.mean(np.abs(np.diff(video, axis=1)))) > 1e-6
