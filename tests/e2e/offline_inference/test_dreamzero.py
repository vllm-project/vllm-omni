# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end test for DreamZero on the standard robot_policy task path.

Loads the real checkpoint through the shared ``robot_policy`` flow — registry
hooks (``build_robot_observations`` / ``process_robot_actions`` /
``finalize_robot_run``), the AR rollout loop, and the worker-side video
export — which is the path RFC #4539 requires every migrated model to cover.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path

MODEL_NAME = os.environ.get("VLLM_OMNI_DREAMZERO_MODEL", "GEAR-Dreams/DreamZero-DROID")
# Real DROID camera assets packaged for this example. When unset the test
# synthesizes zero-filled MP4s instead: the rollout still exercises the full
# engine path end-to-end, only the predicted content is meaningless.
ASSETS_DIR = os.environ.get("VLLM_OMNI_DREAMZERO_ASSETS_DIR")
DEPLOY_CONFIG_PATH = get_deploy_config_path("dreamzero.yaml")

_NUM_CHUNKS = 2
_HEIGHT = 180
_WIDTH = 320
# Chunk k needs frame 23 + 24*k, so 72 frames schedule the initial frame plus
# two 4-frame chunks without touching the repeat-chunk padding path.
_NUM_FRAMES = 72
_CAMERA_FILES = {
    "observation/exterior_image_0_left": "exterior_image_1_left.mp4",
    "observation/exterior_image_1_left": "exterior_image_2_left.mp4",
    "observation/wrist_image_left": "wrist_image_left.mp4",
}
_TASK = "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.diffusion,
]


def _synthesized_assets(tmp_dir: Path) -> Path:
    assets = tmp_dir / "dreamzero_assets"
    assets.mkdir(parents=True, exist_ok=True)
    for file_name in _CAMERA_FILES.values():
        writer = cv2.VideoWriter(
            str(assets / file_name),
            cv2.VideoWriter_fourcc(*"mp4v"),
            5.0,
            (_WIDTH, _HEIGHT),
        )
        for frame_index in range(_NUM_FRAMES):
            frame = np.full((_HEIGHT, _WIDTH, 3), (frame_index * 3) % 255, dtype=np.uint8)
            writer.write(frame)
        writer.release()
    return assets


@pytest.fixture(scope="module")
def omni(tmp_path_factory: pytest.TempPathFactory):
    from vllm_omni.model_extras import get_worker_extension_class

    with OmniRunner(
        MODEL_NAME,
        deploy_config=str(DEPLOY_CONFIG_PATH),
        worker_extension_cls=get_worker_extension_class("DreamZeroPipeline"),
    ) as runner:
        yield runner.omni


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_dreamzero_robot_policy_standard_path_runs_ar_rollout(omni, tmp_path: Path) -> None:
    """Run the shared robot_policy flow end-to-end and export the rollout video."""
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_extras import (
        build_robot_observations,
        finalize_robot_run,
        get_model_class_name,
        process_robot_actions,
    )

    model_class_name = get_model_class_name(omni)
    assert model_class_name == "DreamZeroPipeline"

    data_dir = ASSETS_DIR or str(_synthesized_assets(tmp_path))
    observations, metadata = build_robot_observations(
        model_class_name,
        model_dir=MODEL_NAME,
        task=_TASK,
        data_dir=data_dir,
        num_chunks=_NUM_CHUNKS,
    )
    assert metadata == {}
    assert len(observations) == _NUM_CHUNKS + 1
    assert observations[0]["reset"] is True
    assert all(not obs["reset"] for obs in observations[1:])

    generator = torch.Generator(device="cpu").manual_seed(42)
    results = []
    for extra_args in observations:
        sampling = OmniDiffusionSamplingParams(
            extra_args=extra_args,
            generator=generator,
        )
        outputs = omni.generate(extra_args["prompt"], sampling_params_list=[sampling])
        assert outputs, "each AR step must return one output"
        results.append(process_robot_actions(model_class_name, outputs[0], **metadata))

    assert len(results) == _NUM_CHUNKS + 1
    actions = [result["actions"] for result in results]
    for step, step_actions in enumerate(actions):
        assert step_actions.ndim == 2, f"step {step} actions must be [horizon, dim]"
        assert np.isfinite(step_actions).all(), f"step {step} actions must be finite"
    assert all(step.shape == actions[0].shape for step in actions)

    output_path = tmp_path / "robot_policy_output.npz"
    np.savez(output_path, actions=np.stack(actions, axis=0), num_steps=len(results))

    finalize_robot_run(model_class_name, omni, results, output_path)
    video_path = output_path.with_suffix(".mp4")
    assert video_path.exists() and video_path.stat().st_size > 0, "video export must produce a non-empty mp4"
    capture = cv2.VideoCapture(str(video_path))
    try:
        decoded_frames = 0
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            decoded_frames += 1
            assert frame is not None and frame.size > 0
    finally:
        capture.release()
    assert decoded_frames > 0, "exported mp4 must contain at least one decodable frame"
