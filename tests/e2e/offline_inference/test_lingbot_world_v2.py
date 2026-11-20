# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end smoke test for LingBot-World v2 generation.

Every case sends a one-block request and checks the shape and finiteness of
the RGB frames. CUDA Merge/Nightly and Ascend A2 Ready also check motion.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from vllm.assets.image import ImageAsset

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunnerHandler
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = os.environ.get(
    "VLLM_OMNI_LINGBOT_WORLD_V2_CHECKPOINT_PATH",
    "robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
)
_IMAGE_PATH_OVERRIDE = os.environ.get("VLLM_OMNI_LINGBOT_WORLD_V2_IMAGE_PATH")
_ACTION_DIR_OVERRIDE = os.environ.get("VLLM_OMNI_LINGBOT_WORLD_V2_ACTION_DIR")
# Default first frame: a shared vLLM asset, so no private fixture path is needed.
_IMAGE_ASSET = "2560px-Gfp-wisconsin-madison-the-nature-boardwalk"
# Read by the LingBot-World pipeline when model_config.lingbot_action_root is unset.
_ACTION_ROOT_ENV = "VLLM_OMNI_LINGBOT_ACTION_ROOT"

_HEIGHT = 480
_WIDTH = 832
_NUM_FRAMES = 9

_CUDA_RUNNER = (
    MODEL,
    None,
    {
        "flow_shift": 5.0,
        "parallel_config": DiffusionParallelConfig(tensor_parallel_size=1),
    },
)

_NPU_RUNNER = (
    MODEL,
    str(Path(__file__).resolve().parents[1] / "accuracy/lingbot_world/configs/npu_usp4_offload.yaml"),
    {"stage_init_timeout": 2400, "init_timeout": 2400},
)

pytestmark = [pytest.mark.diffusion]


def _write_forward_trajectory(action_dir: Path) -> None:
    """Write a valid camera trajectory: 9 frames moving forward along +z."""
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], _NUM_FRAMES, axis=0)
    poses[:, 2, 3] = np.linspace(0.0, 0.8, _NUM_FRAMES, dtype=np.float32)
    # fx, fy, cx, cy in the pipeline's 832x480 reference coordinates.
    intrinsics = np.tile(np.array([500.0, 500.0, _WIDTH / 2, _HEIGHT / 2], dtype=np.float32), (_NUM_FRAMES, 1))
    action_dir.mkdir(parents=True)
    np.save(action_dir / "poses.npy", poses)
    np.save(action_dir / "intrinsics.npy", intrinsics)


@pytest.fixture(scope="module", autouse=True)
def lingbot_action_path(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Set the action root before ``omni_runner`` starts; return the action_path."""
    if _ACTION_DIR_OVERRIDE:
        action_dir = Path(_ACTION_DIR_OVERRIDE).expanduser().resolve()
    else:
        action_dir = tmp_path_factory.mktemp("lingbot_world_actions") / "forward"
        _write_forward_trajectory(action_dir)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv(_ACTION_ROOT_ENV, str(action_dir.parent))
        yield action_dir.name


def _first_frame() -> str:
    if _IMAGE_PATH_OVERRIDE:
        return str(Path(_IMAGE_PATH_OVERRIDE).expanduser().resolve())
    return str(ImageAsset(_IMAGE_ASSET).get_path("jpg"))


def _generate_one_block(handler: OmniRunnerHandler, action_path: str, *, stepwise: bool = False) -> np.ndarray:
    """Generate and return the nine RGB frames from one causal block."""
    response = handler.send_diffusion_request(
        {
            "model": handler.runner.model_name,
            "prompt": "The camera moves slowly forward through the scene.",
            "multi_modal_data": {"image": _first_frame()},
            "sampling_params": OmniDiffusionSamplingParams(
                height=_HEIGHT,
                width=_WIDTH,
                num_frames=_NUM_FRAMES,
                max_sequence_length=512,
                seed=42,
                fps=16,
                extra_args={
                    "action_path": action_path,
                    "flow_shift": 5.0,
                },
            ),
        }
    )

    assert response.success
    assert response.images is not None
    assert len(response.images) == 1
    video = np.asarray(response.images[0])
    # Whole-clip inference includes a batch axis. Step execution returns one
    # decoded THWC chunk directly in the video payload.
    if not stepwise:
        assert video.shape == (1, _NUM_FRAMES, _HEIGHT, _WIDTH, 3)
        video = video[0]
    assert video.shape == (_NUM_FRAMES, _HEIGHT, _WIDTH, 3)
    assert np.isfinite(video).all()
    return video


def _assert_video_varies(video: np.ndarray) -> None:
    assert np.ptp(video) > 1e-6
    # Stepwise output uses uint8, so subtract in floating point.
    assert np.mean(np.abs(np.diff(video.astype(np.float32, copy=False), axis=0))) > 1e-6


@pytest.mark.core_model
@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=1)
@pytest.mark.parametrize("omni_runner", [_CUDA_RUNNER], indirect=True)
def test_lingbot_world_v2_tp1_one_block_ready(lingbot_action_path: str, omni_runner_handler: OmniRunnerHandler) -> None:
    """Load the real checkpoint and generate one causal block on one GPU."""
    _generate_one_block(omni_runner_handler, lingbot_action_path)


@pytest.mark.advanced_model
@pytest.mark.full_model
@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=1)
@pytest.mark.parametrize("omni_runner", [_CUDA_RUNNER], indirect=True)
def test_lingbot_world_v2_tp1_one_block(lingbot_action_path: str, omni_runner_handler: OmniRunnerHandler) -> None:
    """Generate one causal block and check that the video varies."""
    video = _generate_one_block(omni_runner_handler, lingbot_action_path)
    _assert_video_varies(video)


@pytest.mark.core_model
@hardware_test(res={"npu": "A2"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", [_NPU_RUNNER], indirect=True)
def test_lingbot_world_v2_usp4_one_block(lingbot_action_path: str, omni_runner_handler: OmniRunnerHandler) -> None:
    """Decode one block with Ulysses SP-4 and text-encoder offload on four A2 devices."""
    video = _generate_one_block(omni_runner_handler, lingbot_action_path, stepwise=True)
    _assert_video_varies(video)
