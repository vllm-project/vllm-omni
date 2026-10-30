# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Platform overrides preserve nested deploy and user-overlay settings."""

from copy import deepcopy

import pytest
import yaml

from vllm_omni.config.stage_config import (
    _DEPLOY_DIR,
    DeployConfig,
    StageDeployConfig,
    _apply_platform_overrides,
    load_deploy_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_platform_nested_overrides_preserve_siblings_and_replace_values():
    original = {
        "hf_overrides": {"runtime": {"keep": 1, "replace": 2, "items": [1, 2], "reset": {"old": 1}}},
        "other": "preserved",
    }
    deploy = DeployConfig(
        stages=[StageDeployConfig(stage_id=0, engine_extras=deepcopy(original))],
        platforms={
            "cuda": {
                "stages": [
                    {
                        "stage_id": 0,
                        "engine_extras": {
                            "hf_overrides": {"runtime": {"replace": 3, "items": [4], "reset": None}},
                        },
                    }
                ],
            },
        },
    )
    resolved = _apply_platform_overrides(deploy, platform="cuda")
    assert resolved.stages[0].engine_extras == {
        "hf_overrides": {"runtime": {"keep": 1, "replace": 3, "items": [4], "reset": None}},
        "other": "preserved",
    }
    assert original["hf_overrides"]["runtime"]["items"] == [1, 2]


@pytest.mark.parametrize("platform", ["cuda", "npu", "rocm", "xpu", "musa", "cpu"])
def test_voxcpm2_platform_keeps_user_runtime_overlay(tmp_path, platform):
    path = tmp_path / "custom.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "base_config": str(_DEPLOY_DIR / "voxcpm2.yaml"),
                "stages": [
                    {
                        "stage_id": 0,
                        "engine_extras": {
                            "hf_overrides": {
                                "voxcpm2_runtime_config": {
                                    "cfg_value": 2.5,
                                    "startup_lora_path": "/tmp/custom-adapter",
                                }
                            }
                        },
                    }
                ],
            }
        )
    )
    deploy = _apply_platform_overrides(load_deploy_config(path), platform=platform)
    runtime = deploy.stages[0].engine_extras["hf_overrides"]["voxcpm2_runtime_config"]
    assert deploy.model_runner == ("v2" if platform in {"cuda", "npu"} else "v1")
    assert runtime["cfg_value"] == 2.5
    assert runtime["startup_lora_path"] == "/tmp/custom-adapter"
    assert runtime.get("enable_incremental_vae", False) is (platform == "cuda")
    assert runtime["vae_decode_every"] == (1 if platform == "cuda" else 3)
    assert runtime["audio_emit_every"] == (3 if platform == "cuda" else 1)
    assert runtime["enable_batched_vae_decode"] is (platform != "cuda")
    assert runtime["enable_loc_dit_fused_qkv"] is True
