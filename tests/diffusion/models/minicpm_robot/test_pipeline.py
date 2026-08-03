# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MiniCPM-RobotManip diffusion pipeline."""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.diffusion.models.minicpm_robot import pipeline_minicpm_robot
from vllm_omni.diffusion.models.minicpm_robot.pipeline_minicpm_robot import (
    MiniCPMRobotManipPipeline,
)
from vllm_omni.diffusion.models.minicpm_robot.policy import (
    _format_prompt,
    normalize_robot_obs,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeMiniCPMRobotPolicy:
    instances: list["FakeMiniCPMRobotPolicy"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.reset_calls = 0
        self.seen_obs = None
        self.seen_seed = None
        self.state_dim = 80
        self.model = SimpleNamespace(config=SimpleNamespace(action_dim=80, action_horizon=30, state_dim=80))
        FakeMiniCPMRobotPolicy.instances.append(self)

    def get_action(self, obs, *, seed=None):
        self.seen_obs = obs
        self.seen_seed = seed
        return {"default": np.zeros((1, 30, 80), dtype=np.float32)}

    def reset(self):
        self.reset_calls += 1
        return {"reset": True}


@pytest.fixture(autouse=True)
def fake_policy(monkeypatch):
    FakeMiniCPMRobotPolicy.instances.clear()
    monkeypatch.setattr(
        pipeline_minicpm_robot,
        "MiniCPMRobotPolicy",
        FakeMiniCPMRobotPolicy,
    )


def _rgb(h=64, w=64, value=0):
    return np.full((h, w, 3), value, dtype=np.uint8)


def _robot_obs(**overrides):
    obs = {
        "images": [_rgb(value=0), _rgb(value=1)],
        "state": np.zeros(80, dtype=np.float32),
        "language": "pick up the black bowl",
    }
    obs.update(overrides)
    return obs


def _pipeline(**model_config_overrides):
    model_config = {
        "embodiment_id": 0,
    }
    model_config.update(model_config_overrides)
    od_config = SimpleNamespace(
        model="/path/to/MiniCPM-RobotManip",
        model_config=model_config,
    )
    return MiniCPMRobotManipPipeline(od_config=od_config)


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def test_normalize_robot_obs_preserves_sizes_and_accepts_dict_images():
    obs = normalize_robot_obs(
        {
            "images": {
                "cam_high": _rgb(120, 160, 7),
                "cam_low": _rgb(90, 100, 9),
            },
            "state": np.arange(80, dtype=np.float32),
            "prompt": "do the task",
        }
    )

    assert len(obs["images"]) == 2
    assert obs["images"][0].shape == (120, 160, 3)
    assert obs["images"][1].shape == (90, 100, 3)
    assert all(img.dtype == np.uint8 for img in obs["images"])
    assert obs["state"].shape == (80,)
    assert obs["language"] == "do the task"


def test_normalize_robot_obs_rejects_bad_state_dim():
    with pytest.raises(ValueError, match="state"):
        normalize_robot_obs(
            {
                "images": [_rgb()],
                "state": np.zeros(7, dtype=np.float32),
                "language": "task",
            }
        )


def test_normalize_robot_obs_rejects_missing_images():
    with pytest.raises(ValueError, match="images"):
        normalize_robot_obs(
            {
                "state": np.zeros(80, dtype=np.float32),
                "language": "task",
            }
        )


def test_format_prompt_handles_braces_in_instruction():
    text = _format_prompt(
        "Task: {instruction}",
        "use {gripper} carefully",
    )
    assert text == "Task: use {gripper} carefully"


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def test_pipeline_initializes_policy():
    pipeline = _pipeline()

    policy = FakeMiniCPMRobotPolicy.instances[0]
    assert policy.kwargs["model_path"] == "/path/to/MiniCPM-RobotManip"
    assert policy.kwargs["embodiment_id"] == 0
    assert policy.kwargs["device"] in ("cuda", "cpu")
    assert pipeline.weights_sources == ()
    assert pipeline.load_weights(iter(())) == set()


def test_load_weights_rejects_non_empty_weights():
    pipeline = _pipeline()
    with pytest.raises(RuntimeError, match="load_weights received"):
        pipeline.load_weights(iter([("a.b", np.array(1.0))]))


def test_weights_sources_is_empty():
    pipeline = _pipeline()
    assert pipeline.weights_sources == ()


# ---------------------------------------------------------------------------
# Forward path
# ---------------------------------------------------------------------------


def test_forward_returns_default_actions_in_output():
    pipeline = _pipeline()
    state = np.random.randn(80).astype(np.float32)
    req = OmniDiffusionRequest(
        prompt="pick up the black bowl",
        request_id="req-1",
        sampling_params=OmniDiffusionSamplingParams(
            seed=123,
            extra_args={
                "robot_obs": _robot_obs(state=state),
            },
        ),
    )

    output = pipeline.forward(req)

    assert output.error is None
    actions = output.output["actions"]
    assert set(actions) == {"default"}
    assert actions["default"].dtype == np.float32
    assert actions["default"].shape == (1, 30, 80)
    policy = FakeMiniCPMRobotPolicy.instances[0]
    assert policy.seen_obs is not None
    assert policy.seen_obs["state"].shape == (80,)
    np.testing.assert_allclose(policy.seen_obs["state"], state)
    assert policy.seen_obs["language"] == "pick up the black bowl"
    assert all(img.shape == (64, 64, 3) for img in policy.seen_obs["images"])
    assert policy.seen_seed == 123


def test_forward_reset_calls_policy_reset():
    pipeline = _pipeline()
    req = OmniDiffusionRequest(
        prompt="task",
        request_id="req-2",
        sampling_params=OmniDiffusionSamplingParams(
            extra_args={
                "robot_obs": _robot_obs(language="task"),
                "reset": True,
            }
        ),
    )

    output = pipeline.forward(req)
    assert output.error is None
    assert FakeMiniCPMRobotPolicy.instances[0].reset_calls == 1


def test_forward_missing_robot_obs_returns_error():
    pipeline = _pipeline()
    req = OmniDiffusionRequest(
        prompt="task",
        request_id="req-3",
        sampling_params=OmniDiffusionSamplingParams(),
    )

    output = pipeline.forward(req)
    assert output.error is not None
    assert "robot_obs" in output.error


def test_forward_non_dict_robot_obs_returns_error():
    pipeline = _pipeline()
    req = OmniDiffusionRequest(
        prompt="task",
        request_id="req-4",
        sampling_params=OmniDiffusionSamplingParams(
            extra_args={"robot_obs": "invalid"},
        ),
    )

    output = pipeline.forward(req)
    assert output.error is not None
    assert "dict" in output.error


def test_forward_invalid_obs_returns_error():
    pipeline = _pipeline()
    req = OmniDiffusionRequest(
        prompt="task",
        request_id="req-5",
        sampling_params=OmniDiffusionSamplingParams(
            extra_args={
                "robot_obs": {
                    "images": [],
                    "state": np.zeros(80, dtype=np.float32),
                    "language": "task",
                }
            }
        ),
    )

    output = pipeline.forward(req)
    assert output.error is not None
    assert "images" in output.error
    assert FakeMiniCPMRobotPolicy.instances[0].seen_obs is None


def test_dummy_warmup_returns_zero_actions():
    pipeline = _pipeline()
    req = OmniDiffusionRequest(
        prompt="dummy run",
        request_id="dummy_req_id",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
    )

    output = pipeline.forward(req)

    assert output.error is None
    actions = output.output["actions"]
    assert set(actions) == {"default"}
    assert actions["default"].shape == (1, 30, 80)
    assert actions["default"].dtype == np.float32
    assert not actions["default"].any()
    assert FakeMiniCPMRobotPolicy.instances[0].seen_obs is None


def test_reset_delegates_to_policy():
    pipeline = _pipeline()

    assert pipeline.reset() == {"reset": True}
    assert FakeMiniCPMRobotPolicy.instances[0].reset_calls == 1


# ---------------------------------------------------------------------------
# Pipeline registration
# ---------------------------------------------------------------------------


def test_pipeline_is_registered_in_diffusion_registry():
    from vllm_omni.diffusion.registry import DiffusionModelRegistry

    registered = DiffusionModelRegistry.get_supported_archs()
    assert "MiniCPMRobotManipPipeline" in registered


def test_pipeline_is_registered_in_omni_pipelines():
    from vllm_omni.config.pipeline_registry import OMNI_PIPELINES

    assert "MiniCPMRobotManip" in OMNI_PIPELINES


# ---------------------------------------------------------------------------
# Policy server config validation
# ---------------------------------------------------------------------------


def test_validate_policy_server_config_accepts_matching_dims():
    pipeline = _pipeline()
    pipeline._validate_policy_server_config(
        {
            "action_horizon": 30,
            "action_dim": 80,
            "state_dim": 80,
        }
    )


def test_validate_policy_server_config_raises_on_action_horizon_mismatch():
    pipeline = _pipeline()
    with pytest.raises(ValueError, match="policy_server_config.action_horizon"):
        pipeline._validate_policy_server_config(
            {
                "action_horizon": 99,
                "action_dim": 80,
                "state_dim": 80,
            }
        )


def test_validate_policy_server_config_raises_on_action_dim_mismatch():
    pipeline = _pipeline()
    with pytest.raises(ValueError, match="policy_server_config.action_dim"):
        pipeline._validate_policy_server_config(
            {
                "action_horizon": 30,
                "action_dim": 99,
                "state_dim": 80,
            }
        )


def test_validate_policy_server_config_raises_on_state_dim_mismatch():
    pipeline = _pipeline()
    with pytest.raises(ValueError, match="policy_server_config.state_dim"):
        pipeline._validate_policy_server_config(
            {
                "action_horizon": 30,
                "action_dim": 80,
                "state_dim": 99,
            }
        )
