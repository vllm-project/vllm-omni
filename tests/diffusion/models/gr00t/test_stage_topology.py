# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for F8+F9: stage topology + deploy YAML + policy_server_config.

Verifies:
- GR00T_PIPELINE is declared with the expected stage topology.
- The pipeline registry resolves `model_type="gr00t"`.
- vllm_omni/deploy/gr00t.yaml parses and exposes the policy_server_config
  required fields (action_horizon, action_keys, supported_embodiments).
- PolicyServerConfig.from_model_config returns those fields to OpenPI
  clients.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[4]
DEPLOY_YAML = REPO_ROOT / "vllm_omni" / "deploy" / "gr00t.yaml"


# ---------------------------------------------------------------------------
# Stage topology
# ---------------------------------------------------------------------------


def test_pipeline_config_registered():
    from vllm_omni.config.stage_config import (
        PipelineConfig,
        StageExecutionType,
        StagePipelineConfig,
    )
    from vllm_omni.model_executor.models.gr00t.pipeline import GR00T_PIPELINE

    assert isinstance(GR00T_PIPELINE, PipelineConfig)
    assert GR00T_PIPELINE.model_type == "gr00t"
    assert GR00T_PIPELINE.model_arch == "Gr00tN1d7Pipeline"
    assert len(GR00T_PIPELINE.stages) == 1

    stage = GR00T_PIPELINE.stages[0]
    assert isinstance(stage, StagePipelineConfig)
    assert stage.stage_id == 0
    assert stage.model_stage == "diffusion"
    assert stage.execution_type is StageExecutionType.DIFFUSION
    assert stage.final_output is True
    assert stage.final_output_type == "actions"
    assert stage.model_arch == "Gr00tN1d7Pipeline"


def test_pipeline_registry_resolves_gr00t():
    """The central registry table must map ``model_type='gr00t'`` to the
    GR00T_PIPELINE module path."""
    from vllm_omni.config.pipeline_registry import _OMNI_PIPELINES

    assert "gr00t" in _OMNI_PIPELINES
    module_path, var_name = _OMNI_PIPELINES["gr00t"]
    assert module_path == "vllm_omni.model_executor.models.gr00t.pipeline"
    assert var_name == "GR00T_PIPELINE"


# ---------------------------------------------------------------------------
# Deploy YAML
# ---------------------------------------------------------------------------


def _load_deploy_yaml() -> dict:
    assert DEPLOY_YAML.exists(), f"Missing deploy YAML at {DEPLOY_YAML}"
    with open(DEPLOY_YAML) as f:
        return yaml.safe_load(f)


def test_deploy_yaml_top_level():
    payload = _load_deploy_yaml()
    assert payload["pipeline"] == "gr00t"
    assert payload["dtype"] == "bfloat16"
    assert len(payload["stages"]) == 1
    stage = payload["stages"][0]
    assert stage["model_class_name"] == "Gr00tN1d7Pipeline"


def test_deploy_yaml_embodiment_table_matches_isaac_upstream():
    """The embodiment_name_to_id table must agree with the Isaac upstream
    constant we ported in F6."""
    from vllm_omni.diffusion.models.gr00t.transform import (
        EMBODIMENT_TAG_TO_PROJECTOR_INDEX,
    )

    payload = _load_deploy_yaml()
    yaml_table = payload["stages"][0]["model_config"]["embodiment_name_to_id"]
    for tag, expected_id in yaml_table.items():
        assert tag in EMBODIMENT_TAG_TO_PROJECTOR_INDEX
        assert EMBODIMENT_TAG_TO_PROJECTOR_INDEX[tag] == expected_id


def test_deploy_yaml_policy_server_config_required_fields():
    payload = _load_deploy_yaml()
    psc = payload["stages"][0]["model_config"]["policy_server_config"]
    for required in (
        "image_resolution",
        "n_external_cameras",
        "needs_wrist_camera",
        "needs_stereo_camera",
        "needs_session_id",
        "action_horizon",
        "action_keys",
        "supported_embodiments",
    ):
        assert required in psc, f"deploy yaml missing policy_server_config.{required}"
    assert psc["action_horizon"] == 40
    assert "joint_position" in psc["action_keys"]
    assert "gripper_position" in psc["action_keys"]
    assert len(psc["supported_embodiments"]) >= 1


# ---------------------------------------------------------------------------
# PolicyServerConfig handshake
# ---------------------------------------------------------------------------


def test_policy_server_config_advertises_action_schema():
    """``PolicyServerConfig.from_model_config`` must surface action_horizon,
    action_keys, and supported_embodiments to OpenPI clients.

    Importing ``vllm_omni.entrypoints.openai`` may chain into
    ``vllm_omni.engine.output_processor`` which currently requires a vllm
    build that exports ``split_routed_experts`` — skip if that build is not
    available in the harness env (a pre-existing condition that affects
    other openpi serving tests identically).
    """
    try:
        from vllm_omni.entrypoints.openai.realtime.robot.openpi_serving import (
            PolicyServerConfig,
        )
    except ImportError as exc:  # pragma: no cover — env-only fallback
        pytest.skip(f"openpi_serving unavailable in this env: {exc}")

    payload = _load_deploy_yaml()
    model_config = payload["stages"][0]["model_config"]
    psc = PolicyServerConfig.from_model_config(model_config)
    advertised = psc.to_dict()
    assert advertised["action_horizon"] == 40
    assert "joint_position" in advertised["action_keys"]
    assert "supported_embodiments" in advertised
    # Round-trip through to_dict / from_model_config preserves the fields.
    psc2 = PolicyServerConfig(advertised)
    assert psc2.to_dict() == advertised
