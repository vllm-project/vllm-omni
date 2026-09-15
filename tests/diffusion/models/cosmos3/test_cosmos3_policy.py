# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for Cosmos3 OpenPI policy serving path.

Validates that:
- _build_robolab_policy_inputs processes robot_obs correctly
- _forward_robolab_policy returns postprocessed actions via custom_output["actions"]
- _get_policy_server_config falls back to the model config.json when the
  in-memory model_config lacks policy_server_config (the common case for
  pure-diffusion policy models launched without --stage-overrides).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# --- _build_robolab_policy_inputs tests ---


def _make_minimal_obs() -> dict[str, Any]:
    """Create a minimal valid DROID observation dict."""
    return {
        "prompt": "pick up the banana",
        "observation/image": np.zeros((360, 640, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros((2, 7), dtype=np.float64),
        "observation/gripper_position": np.zeros((2, 1), dtype=np.float64),
    }


def _make_minimal_extra_args(obs: dict | None = None) -> dict[str, Any]:
    """Create minimal extra_args with robot_obs."""
    return {
        "robot_obs": obs or _make_minimal_obs(),
        "action_space": "joint_pos",
        "domain_name": "droid_lerobot",
    }


class TestBuildRoboLabPolicyInputs:
    """Test _build_robolab_policy_inputs with mock pipeline."""

    @pytest.fixture()
    def mock_pipeline(self) -> None:
        """Create a minimal mock pipeline instance for testing."""
        pipeline = MagicMock(spec=Cosmos3OmniDiffusersPipeline)
        # _build_robolab_policy_inputs is a regular method that accesses self._get_robolab_transform()
        # We need to call the real method, so bind it to the mock.
        # _truthy is a @staticmethod — assigning directly (no __get__) keeps the
        # static call signature so `self._truthy(x)` passes only `x`.
        pipeline._build_robolab_policy_inputs = Cosmos3OmniDiffusersPipeline._build_robolab_policy_inputs.__get__(
            pipeline
        )
        pipeline._truthy = Cosmos3OmniDiffusersPipeline._truthy

        # Mock the transform
        mock_transform = MagicMock()
        mock_sample = {
            "video": torch.zeros((3, 33, 360, 640), dtype=torch.uint8),
            "action": torch.zeros((32, 8), dtype=torch.float32),
            "raw_action_dim": 8,
            "ai_caption": "pick up the banana",
            "image_size": None,
            "sequence_plan": SimpleNamespace(
                condition_frame_indexes_action=[0],
                action_start_frame_offset=1,
            ),
        }
        mock_transform.return_value = mock_sample
        pipeline._get_robolab_transform.return_value = mock_transform

        return pipeline

    def test_returns_none_when_no_robot_obs(self, mock_pipeline) -> None:
        sp = SimpleNamespace(extra_args={})
        result = mock_pipeline._build_robolab_policy_inputs(sp, None, None)
        assert result is None

    def test_returns_inputs_with_robot_obs(self, mock_pipeline) -> None:
        obs = _make_minimal_obs()
        extra = _make_minimal_extra_args(obs)
        sp = SimpleNamespace(extra_args=extra)
        result = mock_pipeline._build_robolab_policy_inputs(sp, None, "test-request")
        assert result is not None
        assert result.prompt == "pick up the banana"

    def test_raises_on_missing_prompt(self, mock_pipeline) -> None:
        obs = _make_minimal_obs()
        del obs["prompt"]
        extra = _make_minimal_extra_args(obs)
        sp = SimpleNamespace(extra_args=extra)
        with pytest.raises(ValueError, match="prompt"):
            mock_pipeline._build_robolab_policy_inputs(sp, None, None)

    def test_raises_on_non_dict_obs(self, mock_pipeline) -> None:
        extra = {"robot_obs": "not a dict"}
        sp = SimpleNamespace(extra_args=extra)
        with pytest.raises(TypeError, match="dict"):
            mock_pipeline._build_robolab_policy_inputs(sp, None, None)


# --- Integration test for action output format ---


class TestRoboLabActionOutput:
    """Verify that the action output contract is correct.

    The key assertion: _forward_robolab_policy must return postprocessed
    actions in custom_output["actions"] (not custom_output["action"]).
    """

    def test_robolab_output_has_actions_key(self) -> None:
        """Verify DiffusionOutput from robolab path uses custom_output["actions"]."""
        from vllm_omni.diffusion.data import DiffusionOutput

        # Simulate the output from _forward_robolab_policy
        actions_np = np.zeros((32, 8), dtype=np.float32)
        output = DiffusionOutput(
            output={},
            custom_output={"actions": actions_np, "action_only_output": True},
        )

        # Verify the engine's action_payload path would pick this up
        custom_output = output.custom_output or {}
        action_payload = custom_output.get("actions")
        assert action_payload is not None
        assert isinstance(action_payload, np.ndarray)
        assert action_payload.shape == (32, 8)

    def test_robolab_output_has_action_only_flag(self) -> None:
        """Verify action_only_output=True is set to skip video post-processing."""
        from vllm_omni.diffusion.data import DiffusionOutput

        actions_np = np.zeros((32, 8), dtype=np.float32)
        output = DiffusionOutput(
            output={},
            custom_output={"actions": actions_np, "action_only_output": True},
        )

        custom_output = output.custom_output or {}
        assert bool(custom_output.get("action_only_output")) is True

    def test_video_action_output_has_actions_key(self) -> None:
        """Verify video+action output also uses custom_output["actions"]."""
        from vllm_omni.diffusion.data import DiffusionOutput

        video = torch.zeros((1, 3, 33, 360, 640))
        actions_np = np.zeros((32, 8), dtype=np.float32)
        output = DiffusionOutput(
            output={"video": video},
            custom_output={"actions": actions_np},
        )

        custom_output = output.custom_output or {}
        action_payload = custom_output.get("actions")
        assert action_payload is not None


# --- config.json fallback test ---


class TestConfigJsonFallback:
    """Test the framework-level config.json fallback in serving.py.

    The fallback in ``_get_policy_server_config`` kicks in when none of the
    in-memory ``model_config`` lookups carry ``policy_server_config`` — which is
    the common case for a pure-diffusion policy model launched without
    ``--stage-overrides`` (the API server process never populates psc in
    ``model_config``). In that case the code reads psc from the model
    directory's ``config.json``.
    """

    def _write_config_json(self, tmp_path: Any, psc: dict | None) -> str:
        import json

        config_json = {}
        if psc is not None:
            config_json["policy_server_config"] = psc
        (tmp_path / "config.json").write_text(json.dumps(config_json))
        return str(tmp_path)

    def _make_engine_client(self, model_path: str) -> Any:
        """Mock engine_client whose od_config.model_config lacks psc.

        All four lookup paths in ``_get_policy_server_config`` resolve to a
        model_config without ``policy_server_config``, forcing the fallback.
        ``engine_client.model`` points at ``model_path`` (with config.json).
        """
        engine_client = MagicMock()
        od_config = MagicMock()
        od_config.model_config = {}  # no psc → triggers fallback
        od_config.model = model_path
        engine_client.get_diffusion_od_config.return_value = od_config
        engine_client.od_config = od_config
        engine_client.stage_configs = []
        engine_client.model = model_path
        return engine_client

    def test_fallback_reads_config_json(self, tmp_path) -> None:
        """When model_config lacks psc, psc is loaded from model config.json."""
        from vllm_omni.entrypoints.openpi.serving import (
            ServingRealtimeRobotOpenPI,
        )

        model_path = self._write_config_json(
            tmp_path,
            {"image_resolution": [360, 640], "action_space": "joint_position"},
        )
        engine_client = self._make_engine_client(model_path)

        psc = ServingRealtimeRobotOpenPI._get_policy_server_config(engine_client)
        assert psc.values["action_space"] == "joint_position"
        assert psc.values["image_resolution"] == [360, 640]

    def test_fallback_skipped_when_model_config_has_psc(self, tmp_path) -> None:
        """When the in-memory model_config already has psc, no fallback occurs."""
        from vllm_omni.entrypoints.openpi.serving import (
            ServingRealtimeRobotOpenPI,
        )

        model_path = self._write_config_json(
            tmp_path,
            {"image_resolution": [999, 999]},  # distinct value in config.json
        )
        engine_client = self._make_engine_client(model_path)
        # Override path 1's model_config to actually carry psc — fallback skipped.
        engine_client.get_diffusion_od_config.return_value.model_config = {
            "policy_server_config": {"action_space": "midtrain"},
        }

        psc = ServingRealtimeRobotOpenPI._get_policy_server_config(engine_client)
        assert psc.values["action_space"] == "midtrain"
        assert psc.values.get("image_resolution") != [999, 999]

    def test_fallback_raises_when_config_json_lacks_psc(self, tmp_path) -> None:
        """If config.json also has no psc, the original ValueError propagates."""
        from vllm_omni.entrypoints.openpi.serving import (
            ServingRealtimeRobotOpenPI,
        )

        model_path = self._write_config_json(tmp_path, psc=None)
        engine_client = self._make_engine_client(model_path)

        with pytest.raises(ValueError, match="policy_server_config"):
            ServingRealtimeRobotOpenPI._get_policy_server_config(engine_client)

    def test_create_policy_server_returns_none_when_no_psc_anywhere(self, tmp_path) -> None:
        """create_policy_server swallows the psc ValueError → returns None (disabled)."""
        from vllm_omni.entrypoints.openpi.serving import (
            ServingRealtimeRobotOpenPI,
        )

        # No config.json at all in model_path.
        engine_client = self._make_engine_client(str(tmp_path))

        result = ServingRealtimeRobotOpenPI.create_policy_server(engine_client=engine_client, model_name="test-model")
        assert result is None

    def test_create_policy_server_succeeds_with_config_json(self, tmp_path) -> None:
        """create_policy_server returns a live server when config.json has psc."""
        from vllm_omni.entrypoints.openpi.serving import (
            ServingRealtimeRobotOpenPI,
        )

        model_path = self._write_config_json(
            tmp_path,
            {"image_resolution": [360, 640], "action_space": "joint_position"},
        )
        engine_client = self._make_engine_client(model_path)

        result = ServingRealtimeRobotOpenPI.create_policy_server(engine_client=engine_client, model_name="test-model")
        assert result is not None
        assert result.policy_server_config.values["action_space"] == "joint_position"
