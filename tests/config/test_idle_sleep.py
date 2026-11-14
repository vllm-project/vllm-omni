# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math
from pathlib import Path

import pytest

from vllm_omni.config.stage_config import IdleSleepRuntimeConfig, load_deploy_config, resolve_deploy_yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("timeout", [0, -1, math.inf, -math.inf, math.nan, True, "300"])
def test_idle_sleep_timeout_rejects_invalid_values(timeout):
    with pytest.raises(ValueError, match="idle_sleep.timeout_s"):
        IdleSleepRuntimeConfig(timeout_s=timeout)


def test_idle_sleep_timeout_normalizes_numeric_value():
    config = IdleSleepRuntimeConfig(timeout_s=300)

    assert config.timeout_s == 300.0
    assert isinstance(config.timeout_s, float)


def test_idle_sleep_defaults_to_disabled():
    assert IdleSleepRuntimeConfig().timeout_s is None


def test_idle_sleep_loads_and_inherits_from_deploy_yaml(tmp_path):
    (tmp_path / "base.yaml").write_text("idle_sleep:\n  timeout_s: 120\nstages: []\n")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("base_config: base.yaml\n")

    assert resolve_deploy_yaml(overlay)["idle_sleep"] == {"timeout_s": 120}
    assert load_deploy_config(overlay).idle_sleep.timeout_s == 120.0


def test_idle_sleep_requires_mapping(tmp_path):
    deploy = tmp_path / "deploy.yaml"
    deploy.write_text("idle_sleep: 300\nstages: []\n")

    with pytest.raises(ValueError, match="idle_sleep must be a mapping"):
        load_deploy_config(deploy)


def test_formal_moss_tts_nano_idle_sleep_profile_inherits_expected_defaults():
    repo_root = Path(__file__).resolve().parents[2]
    formal = repo_root / "vllm_omni" / "deploy" / "moss_tts_nano_idle_sleep.yaml"
    default = repo_root / "vllm_omni" / "deploy" / "moss_tts_nano.yaml"

    formal_config = load_deploy_config(formal)
    default_config = load_deploy_config(default)

    assert formal_config.idle_sleep.timeout_s == 300.0
    assert len(formal_config.stages) == 1
    assert formal_config.stages[0].engine_extras["enable_sleep_mode"] is True
    assert formal_config.stages[0].gpu_memory_utilization == 0.3

    assert default_config.idle_sleep.timeout_s is None
    assert "enable_sleep_mode" not in default_config.stages[0].engine_extras
