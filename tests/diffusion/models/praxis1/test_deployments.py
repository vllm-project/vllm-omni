# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The shipped Praxis-1 deployments resolve to the pipeline and advertise what it serves."""

from pathlib import Path

import pytest
import yaml

from vllm_omni.diffusion.models.praxis1 import processing as single_arm
from vllm_omni.model_extras.praxis1 import PraxisSessionOptions

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DEPLOY_DIR = Path(__file__).resolve().parents[4] / "vllm_omni/deploy"
# Checkpoint geometry the worker enforces on the handshake (see Praxis1Pipeline.__init__).
WORKER_CONTRACT = {
    "praxis1.yaml": {
        "image_resolution": list(single_arm.fit_resize_size(416)),
        "tile_size": [224, 416],
        "action_dim": 13,
        "control_mode": "droid_impedance",
        "model_id": "praxis-1-droid",
    },
    "praxis1_yam.yaml": {
        "image_resolution": [416, 416],
        "tile_size": [256, 416],
        "action_dim": 26,
        "control_mode": "none",
        "model_id": "praxis-1-yam",
    },
}


def _stage(name):
    deployment = yaml.safe_load((DEPLOY_DIR / name).read_text())
    assert deployment["dtype"] == "bfloat16" and deployment["async_chunk"] is False
    assert len(deployment["stages"]) == 1
    stage = deployment["stages"][0]
    assert stage["model_class_name"] == "Praxis1Pipeline"
    assert stage["max_num_seqs"] == 1 and stage["enforce_eager"] is True
    return stage


@pytest.mark.parametrize("name", sorted(WORKER_CONTRACT))
def test_deployment_resolves_to_generic_action_stage(monkeypatch, tmp_path, name):
    from vllm_omni.config import config_factory
    from vllm_omni.config.resolver import resolve_omni_config
    from vllm_omni.diffusion.utils import hf_utils

    def read(filename, *_args, **_kwargs):
        return {"model_type": "praxis", "torch_dtype": "bfloat16"} if filename == "config.json" else None

    monkeypatch.setattr("vllm.transformers_utils.config.get_hf_file_to_dict", read)
    monkeypatch.setattr(hf_utils, "get_hf_file_to_dict", read)
    monkeypatch.setattr(config_factory, "get_hf_file_to_dict", read)
    monkeypatch.setattr(config_factory.StageConfigFactory, "get_hf_config", lambda **_: None)
    hf_utils.is_diffusion_model.cache_clear()
    try:
        resolved = resolve_omni_config(
            str(tmp_path),
            trust_remote_code=False,
            deploy_config_path=str(DEPLOY_DIR / name),
            cli_overrides={},
            stage_overrides=None,
            strategy_config_path=None,
        )
    finally:
        hf_utils.is_diffusion_model.cache_clear()
    assert resolved.pipeline_config.model_type == "generic_diffusion"
    assert resolved.pipeline_config.stages[0].final_output_type == "actions"
    assert len(resolved.stage_configs) == 1
    diffusion_config = resolved.stage_configs[0].diffusion_config
    assert diffusion_config.model_class_name == "Praxis1Pipeline"
    assert diffusion_config.model_config["policy_server_config"] == _stage(name)["model_config"]["policy_server_config"]


@pytest.mark.parametrize("name", sorted(WORKER_CONTRACT))
def test_deployment_handshake_matches_worker(name):
    from vllm_omni.entrypoints.openpi.serving import PolicyServerConfig

    config = _stage(name)["model_config"]
    contract = WORKER_CONTRACT[name]
    # Session options in the deployment are fields the worker accepts and validates.
    options = {key: config[key] for key in PraxisSessionOptions.__dataclass_fields__ if key in config}
    assert PraxisSessionOptions(**options).exec_horizon == 24
    handshake = config["policy_server_config"]
    for flag in ("needs_session_id", "supports_close_session", "video_antialias"):
        assert handshake[flag] is True
    assert handshake["image_resolution"] == contract["image_resolution"]
    assert handshake["tile_size"] == contract["tile_size"]
    assert handshake["fps"] == single_arm.FPS
    assert handshake["action_dim"] == contract["action_dim"]
    assert handshake["action_horizon"] == config["exec_horizon"] == 24
    assert handshake["first_action_horizon"] == 21
    assert handshake["control_mode"] == config["control_mode"] == contract["control_mode"]
    assert handshake["policy"] == "praxis" and handshake["model_id"] == contract["model_id"]
    assert PolicyServerConfig.from_model_config(config).to_dict() == handshake
