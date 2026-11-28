# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Registry and deployment discovery for the Motubrain 2.0 adapter."""

import json

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.diffusion.models.motubrain2_pipeline_config import (
    MOTUBRAIN2_PIPELINE,
)
from vllm_omni.diffusion.registry import _DIFFUSION_MODELS, _NO_CACHE_ACCELERATION, DiffusionModelRegistry

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_pipeline_topology_is_registered():
    assert OMNI_PIPELINES["motubrain2"] is MOTUBRAIN2_PIPELINE
    assert "Motubrain2Pipeline" in _NO_CACHE_ACCELERATION
    assert _DIFFUSION_MODELS["Motubrain2Pipeline"] == (
        "motubrain2",
        "pipeline_motubrain2",
        "Motubrain2Pipeline",
    )
    stage = MOTUBRAIN2_PIPELINE.stages[0]
    assert stage.stage_id == 0
    assert stage.final_output
    assert stage.final_output_type == "action"


def test_registry_loads_motubrain_pipeline():
    from vllm_omni.diffusion.models.motubrain2.pipeline_motubrain2 import Motubrain2Pipeline

    assert DiffusionModelRegistry._try_load_model_cls("Motubrain2Pipeline") is Motubrain2Pipeline


def test_serving_directory_resolves_from_model_index(tmp_path):
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "Motubrain2Pipeline"}))
    assert StageConfigFactory.try_infer_model_type(str(tmp_path), trust_remote_code=False) == "motubrain2"
    assert MOTUBRAIN2_PIPELINE.default_deploy_config_name == "motubrain2.yaml"


@pytest.mark.parametrize(
    ("deploy", "width", "action_space", "devices", "encoder_device"),
    [
        ("motubrain2.yaml", 16, "cartesian_pose_xyzw_gripper", "0", None),
        ("motubrain2-astribot.yaml", 25, "cartesian_pose_xyzw_joint_position_gripper", "0", None),
        ("motubrain2-astribot-2gpu.yaml", 25, "cartesian_pose_xyzw_joint_position_gripper", "0,1", "cuda:1"),
    ],
)
def test_motubrain_deploy_exposes_policy_contract(deploy, width, action_space, devices, encoder_device):
    stage = VllmOmniConfig.from_pipeline_config(
        MOTUBRAIN2_PIPELINE, deploy_config_path=get_deploy_config_path(deploy)
    ).stage_configs[0]
    model_config = stage.diffusion_config.model_config
    handshake = model_config["motubrain2_metadata"]
    assert handshake["action_horizon"] == 24
    assert handshake["action_dim"] == width
    assert handshake["action_space"] == action_space
    assert handshake["needs_session_id"] is False
    assert not {"steps", "sampler", "video_shift", "action_shift"}.intersection(model_config)
    assert {"wan", "text_encoder", "da3"} <= model_config.keys()
    assert stage.runtime_config.devices == devices
    assert stage.parallel_config.world_size == 1
    assert stage.model_config.enforce_eager is False
    assert stage.model_config.default_sampling_params["seed"] == 0
    assert model_config.get("encoder_device") == encoder_device
